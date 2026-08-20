"""
WHO TB Clinical RAG - End-to-End Query Pipeline
-----------------------------------------------
Executes the full pipeline:
Retrieval → Hybrid Dense + BM25 → Reranking → Evidence Selection → Context → Gemini → Validation → Final Answer

Usage:
    python query.py "What is the recommended initial diagnostic test for pulmonary TB?"
"""
import sys
import json
import argparse
from typing import Dict, Any

import config
from retrieval import HybridRetriever
from evidence_selector import select_evidence
from generator import GroundedGenerator


# Singleton instances for fast reuse
_retriever = None
_generator = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def get_generator():
    global _generator
    if _generator is None:
        _generator = GroundedGenerator()
    return _generator


def run_rag_pipeline(question: str) -> Dict[str, Any]:
    """
    Executes the full WHO TB Clinical RAG pipeline:
    1. Hybrid Retrieval (Dense + BM25 + RRF)
    2. Evidence Selection & Noise Filtering
    3. Grounded Gemini Generation
    4. Validation & Citation Enrichment
    """
    question = question.strip()
    if not question:
        return {
            "recommendation": "Please provide a valid clinical question.",
            "evidence": "",
            "citations": [],
            "confidence": "insufficient",
            "retrieved_chunks_count": 0,
            "evidence_count": 0
        }

    retriever = get_retriever()
    generator = get_generator()

    # Step 1: Hybrid Retrieval
    retrieved_candidates = retriever.hybrid_search(question, top_k=config.RERANK_TOP_K)

    # Step 2: Evidence Selection & Noise Filtering
    selected_evidence = select_evidence(retrieved_candidates, question, max_evidence=config.FINAL_EVIDENCE_TOP_K)

    # Step 3 & 4: Grounded Generation & Validation via Gemini
    result = generator.generate(question, selected_evidence)

    # Attach diagnostic / audit metadata
    result["retrieved_chunks_count"] = len(retrieved_candidates)
    result["evidence_count"] = len(selected_evidence)
    result["selected_evidence_preview"] = [
        {
            "chunk_id": e["chunk_id"],
            "document": e["document_name"],
            "page": e["page_number"],
            "section": e["section"],
            "content_preview": e["content"][:200]
        }
        for e in selected_evidence
    ]

    return result


def print_formatted_response(result: Dict[str, Any], question: str):
    """Prints a beautiful formatted summary in the terminal."""
    print("=" * 70)
    print(f"CLINICAL QUESTION: {question}")
    print("=" * 70)
    print(f"\n[RECOMMENDATION] (Confidence: {result.get('confidence', '').upper()}):")
    print(f"{result.get('recommendation', '')}\n")

    evidence = result.get("evidence", "")
    if evidence:
        print("[WHO EVIDENCE EXCERPT]:")
        print(f'"{evidence}"\n')

    citations = result.get("citations", [])
    if citations:
        print(f"[CITATIONS ({len(citations)})]:")
        for i, c in enumerate(citations, 1):
            print(f"  [{i}] {c.get('document')} | {c.get('section')} | Page {c.get('page')}")
            if c.get("exact_quote"):
                print(f"      Quote: \"{c.get('exact_quote')[:120]}...\"")
    else:
        print("[CITATIONS]: None (Refusal / Insufficient Evidence)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Query the WHO TB Clinical RAG pipeline.")
    parser.add_argument("question", nargs="*", help="Clinical question to ask")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not args.question:
        # Default sample clinical query
        query_text = "What is the recommended initial diagnostic test for pulmonary TB in adults?"
    else:
        query_text = " ".join(args.question)

    result = run_rag_pipeline(query_text)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_formatted_response(result, query_text)


if __name__ == "__main__":
    main()
