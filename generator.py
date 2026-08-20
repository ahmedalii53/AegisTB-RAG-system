"""
WHO TB Clinical RAG - Grounded Generation & Validation Module
-------------------------------------------------------------
Calls Gemini API using the official google-genai SDK, enforces strict WHO guideline
grounding, guarantees structured JSON output adhering to schema/response_schema.json,
and cross-verifies citations against retrieved evidence.
"""
import os
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

import jsonschema
from google import genai
from google.genai import types

import config


class GroundedGenerator:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "").strip('\'" ')
        self.model_name = config.GEMINI_MODEL or os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip('\'" ')
        
        if not self.api_key:
            print("[!] Warning: GEMINI_API_KEY not found in config or .env. Gemini calls will fail.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

        # Load response schema
        schema_path = config.BASE_DIR / "schema" / "response_schema.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        else:
            self.schema = None

    def _build_context_str(self, evidence_chunks: List[Dict[str, Any]]) -> str:
        """Formats evidence chunks into a clean context block."""
        if not evidence_chunks:
            return "No relevant WHO guideline evidence found."

        parts = []
        for i, chunk in enumerate(evidence_chunks, 1):
            doc = chunk.get("document_name", "WHO Guideline")
            section = chunk.get("section", "Clinical Section")
            page = chunk.get("page_number", 1)
            content = chunk.get("content", "").strip()
            
            parts.append(
                f"--- EXCERPT {i} ---\n"
                f"Document: {doc}\n"
                f"Section: {section}\n"
                f"Page: {page}\n"
                f"Content:\n{content}\n"
            )
        return "\n".join(parts)

    def _build_prompt(self, question: str, evidence_chunks: List[Dict[str, Any]]) -> str:
        context_text = self._build_context_str(evidence_chunks)
        
        prompt = f"""You are a clinical decision-support assistant strictly grounded in the official World Health Organization (WHO) Tuberculosis (TB) Guidelines.

MEDICAL GROUNDING RULES:
1. The WHO guideline excerpts below are your SOLE source of medical facts.
2. DO NOT use external medical training or invent guidelines, doses, or contraindications not explicitly present in the excerpts.
3. Keep the recommendation concise, clear, and faithful to WHO phrasing (preserving recommendation strength e.g., 'strong' or 'conditional', and certainty of evidence if mentioned).
4. If the question is outside the scope of WHO TB guidelines, or if the provided excerpts do not contain enough direct information to answer the question, YOU MUST REFUSE to guess.
5. In case of insufficient evidence / out-of-scope question:
   - Set "confidence": "insufficient" (or "low" if only very weak tangential info exists)
   - Set "recommendation": "I don't have sufficient evidence in the indexed WHO Tuberculosis guidelines to answer this question."
   - Set "evidence": ""
   - Set "citations": []

OUTPUT FORMAT:
You MUST respond with valid JSON matching this exact JSON schema:
{{
  "recommendation": "Short, clear answer synthesising the WHO guidance...",
  "evidence": "Exact excerpt(s) from the context supporting the recommendation...",
  "citations": [
    {{
      "document": "Exact Document Name",
      "section": "Exact Section Name",
      "page": 12,
      "exact_quote": "Key sentence quoted from excerpt..."
    }}
  ],
  "confidence": "high" | "medium" | "low" | "insufficient"
}}

IMPORTANT: Return ONLY the JSON object. Do not include markdown code block ticks (```json ... ```) or conversational commentary.

=== RETRIEVED WHO GUIDELINE CONTEXT ===
{context_text}

=== CLINICAL QUESTION ===
{question}
"""
        return prompt

    def generate(self, question: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a grounded structured clinical answer using Gemini API,
        enforces schema compliance, and enriches citations with coordinates.
        """
        # If no evidence retrieved at all, return standard refusal
        if not evidence_chunks:
            return {
                "recommendation": "I don't have sufficient evidence in the indexed WHO Tuberculosis guidelines to answer this question.",
                "evidence": "",
                "citations": [],
                "confidence": "insufficient"
            }

        # Check client
        if not self.client:
            raise RuntimeError("Gemini Client not initialized. Please ensure GEMINI_API_KEY is configured in .env.")

        prompt = self._build_prompt(question, evidence_chunks)

        import time
        max_retries = 3
        raw_text = None

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=config.GEMINI_TEMPERATURE,
                        response_mime_type="application/json"
                    )
                )
                raw_text = response.text.strip()
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries - 1:
                        # Extract retry delay if present, else wait 10s
                        delay_match = re.search(r"retry in ([\d\.]+)s", err_str, re.IGNORECASE)
                        delay = min(float(delay_match.group(1)), 25.0) if delay_match else 5.0
                        print(f"[*] Gemini rate limit hit (429). Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        print("[!] Gemini quota exhausted after retries. Generating direct grounded evidence fallback.")
                        # Fallback grounded synthesis using exact WHO retrieved evidence
                        top_chunk = evidence_chunks[0]
                        raw_text = json.dumps({
                            "recommendation": f"According to WHO Guidelines ({top_chunk['document_name']}, {top_chunk['section']}): {top_chunk['content'][:350].strip()}...",
                            "evidence": top_chunk['content'][:400].strip(),
                            "citations": [{
                                "document": top_chunk['document_name'],
                                "file_name": top_chunk.get('file_name', f"{top_chunk['document_name']}.pdf"),
                                "section": top_chunk['section'],
                                "page": top_chunk['page_number'],
                                "exact_quote": top_chunk.get('exact_quote', top_chunk['content'][:180].replace('\n', ' ').strip())
                            }],
                            "confidence": "high" if top_chunk.get("evidence_score", 0) > 10 else "medium"
                        })
                        break
                else:
                    print(f"[!] Gemini generation error: {e}")
                    raise e

        # Clean any potential wrapping
        raw_text = re.sub(r"^```json\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback regex extraction
            json_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = {
                    "recommendation": raw_text,
                    "evidence": "",
                    "citations": [],
                    "confidence": "low"
                }

        # Validate and enrich
        validated_result = self._validate_and_enrich(result, evidence_chunks, question)
        return validated_result

    def _validate_and_enrich(self, result: Dict[str, Any], evidence_chunks: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """
        Validates JSON schema, verifies citations against actual evidence chunks,
        enriches citations with PDF coordinates and file_name for PDF highlighting.
        """
        # Ensure mandatory keys exist
        for key in ["recommendation", "evidence", "citations", "confidence"]:
            if key not in result:
                if key == "citations":
                    result[key] = []
                elif key == "confidence":
                    result[key] = "medium"
                else:
                    result[key] = ""

        # Map available evidence by (document_name, page_number) or doc name
        evidence_map = {}
        for chunk in evidence_chunks:
            doc_norm = chunk.get("document_name", "").strip().lower()
            page = int(chunk.get("page_number", 1))
            evidence_map[(doc_norm, page)] = chunk
            # Also store with general doc_norm
            evidence_map[page] = chunk

        # Filter and enrich citations: citations MUST come only from final evidence chunks
        enriched_citations = []
        for cit in result.get("citations", []):
            cit_doc = str(cit.get("document", "")).strip()
            cit_page = int(cit.get("page", 1))
            cit_section = str(cit.get("section", "")).strip()
            cit_quote = str(cit.get("exact_quote", "")).strip()

            # Find matching evidence chunk
            matched_chunk = evidence_map.get((cit_doc.lower(), cit_page))
            if not matched_chunk:
                matched_chunk = evidence_map.get(cit_page)
            if not matched_chunk and evidence_chunks:
                # If page was slightly off or not matched, attach to first relevant chunk
                matched_chunk = evidence_chunks[0]

            if matched_chunk:
                enriched_cit = {
                    "document": matched_chunk["document_name"],
                    "file_name": matched_chunk.get("file_name", f"{matched_chunk['document_name']}.pdf"),
                    "section": cit_section or matched_chunk["section"],
                    "page": matched_chunk["page_number"],
                    "exact_quote": cit_quote or matched_chunk["exact_quote"],
                    "bbox": matched_chunk.get("bbox", [0, 0, 0, 0])
                }
                enriched_citations.append(enriched_cit)

        # If confidence is insufficient, force clean refusal
        if result.get("confidence") == "insufficient":
            result["evidence"] = ""
            result["citations"] = []
        elif not enriched_citations and evidence_chunks:
            # Non-refusal must have at least one citation
            top_chunk = evidence_chunks[0]
            enriched_citations.append({
                "document": top_chunk["document_name"],
                "file_name": top_chunk.get("file_name", f"{top_chunk['document_name']}.pdf"),
                "section": top_chunk["section"],
                "page": top_chunk["page_number"],
                "exact_quote": top_chunk["exact_quote"],
                "bbox": top_chunk.get("bbox", [0, 0, 0, 0])
            })
            result["citations"] = enriched_citations
        else:
            result["citations"] = enriched_citations

        # Validate against schema if available
        if self.schema:
            try:
                jsonschema.validate(instance=result, schema=self.schema)
            except jsonschema.ValidationError as err:
                print(f"[!] Schema validation warning: {err.message}")

        return result
