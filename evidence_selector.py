"""
WHO TB Clinical RAG - Evidence Selection Module
------------------------------------------------
Filters out noise (references, acknowledgements, table of contents, boilerplate),
prioritizes direct WHO recommendations, target clinical populations, and treatment/diagnostic
sections, and selects the most relevant evidence chunks without over-strict lexical drop.
"""
import re
from typing import List, Dict, Any, Tuple


# Regex patterns for noise filtering
NOISE_SECTION_PATTERNS = [
    r"^acknowledgements?$",
    r"^abbreviations(\s+and\s+acronyms)?$",
    r"^table\s+of\s+contents$",
    r"^contents$",
    r"^contributors$",
    r"^declarations?\s+of\s+interest",
    r"^funding$",
    r"^executive\s+summary\s*-\s*target\s+audience$",
]

NOISE_CONTENT_INDICATORS = [
    r"guideline development group",
    r"external review group",
    r"conflict of interest",
    r"isbn \d{3}-\d+",
    r"all rights reserved",
    r"world health organization \d{4}",
]

# Clinical recommendation signals
RECOMMENDATION_SIGNALS = [
    r"\brecommendation\b",
    r"\brecommends?\b",
    r"\bshould\s+be\s+(used|offered|performed|administered|given|initiated|started)\b",
    r"\bmay\s+be\s+(used|offered|performed)\b",
    r"\bis\s+recommended\b",
    r"\bstrong\s+recommendation\b",
    r"\bconditional\s+recommendation\b",
    r"\bcertainty\s+of\s+evidence\b",
    r"\bhigh-certainty\b|\bmoderate-certainty\b|\blow-certainty\b",
]

# Population keywords
POPULATION_KEYWORDS = {
    "hiv": [r"\bhiv\b", r"\bplhiv\b", r"\bpeople\s+living\s+with\s+hiv\b", r"\bcd4\b", r"\bart\b"],
    "children": [r"\bchild(ren)?\b", r"\bpediatric\b", r"\badolescent(s)?\b", r"\bage\s+<\s+\d+\b"],
    "adult": [r"\badult(s)?\b", r"\bnon-pregnant\b"],
    "pregnant": [r"\bpregnan(t|cy)\b", r"\bmaternal\b"],
    "drug_resistant": [r"\bmdr\b", r"\bxdr\b", r"\brr-tb\b", r"\brifampicin-resistant\b", r"\bfluoroquinolone\b"],
    "drug_susceptible": [r"\bds-tb\b", r"\bdrug-susceptible\b", r"\bfirst-line\b", r"\b2hrze\b", r"\b4hr\b", r"\b2hpmz\b"],
}


def is_noise_chunk(chunk: Dict[str, Any], query: str) -> bool:
    """Detects if a chunk is administrative/bibliographic noise rather than clinical guidance."""
    section = chunk.get("section", "").lower().strip()
    content = chunk.get("content", "").lower()
    
    # 1. Section title matching noise
    for pat in NOISE_SECTION_PATTERNS:
        if re.search(pat, section):
            return True

    # 2. Bibliographic references section (unless query is specifically asking for references)
    if "reference" in section and not any(w in query.lower() for w in ["reference", "author", "study", "trial"]):
        # Check if content looks like reference list (e.g. many years or [1], [2])
        ref_matches = len(re.findall(r"\b(19\d\d|20\d\d)\b", content))
        if ref_matches >= 4:
            return True

    # 3. Pure acknowledgements/funding boilerplate check
    noise_count = sum(1 for pat in NOISE_CONTENT_INDICATORS if re.search(pat, content))
    if noise_count >= 2 and not any(rec in content for rec in ["recommend", "treatment", "diagnosis", "regimen"]):
        return True

    return False


def calculate_evidence_score(chunk: Dict[str, Any], query: str) -> float:
    """
    Computes a clinical evidence score combining:
    - Base hybrid retrieval score
    - Direct recommendation bonus
    - Specific population match bonus
    - Treatment / diagnostic section relevance
    """
    base_score = chunk.get("hybrid_score", 0.0) * 100.0  # scale RRF score
    content_lower = chunk.get("content", "").lower()
    section_lower = chunk.get("section", "").lower()
    query_lower = query.lower()

    score = base_score

    # 1. Boost explicit WHO recommendations
    rec_bonus = 0.0
    for pat in RECOMMENDATION_SIGNALS:
        if re.search(pat, content_lower):
            rec_bonus += 4.0
        if re.search(pat, section_lower):
            rec_bonus += 6.0
    score += min(rec_bonus, 20.0)

    # 2. Population matching alignment
    for pop_key, pop_patterns in POPULATION_KEYWORDS.items():
        query_has_pop = any(re.search(p, query_lower) for p in pop_patterns)
        chunk_has_pop = any(re.search(p, content_lower) for p in pop_patterns)
        
        if query_has_pop and chunk_has_pop:
            score += 15.0  # strong boost for intended population match
        elif query_has_pop and not chunk_has_pop:
            # penalize chunks that miss the specific requested sub-population
            score -= 5.0

    # 3. Section keyword alignment
    query_tokens = [w for w in re.findall(r"\b\w{3,}\b", query_lower) if w not in ["what", "which", "when", "how", "for", "the", "and", "with"]]
    for token in query_tokens:
        if token in section_lower:
            score += 5.0
        if token in content_lower:
            score += 2.0

    # 4. Penalty for definitions when asking for treatment/diagnosis actions
    if "definition" in section_lower and any(w in query_lower for w in ["treat", "dose", "diagnos", "regimen", "test", "recommend"]):
        score -= 8.0

    return max(0.0, score)


def is_query_out_of_scope(query: str, top_chunks: List[Dict[str, Any]]) -> bool:
    """Detects if query is unrelated to WHO Tuberculosis domain."""
    query_lower = query.lower()
    
    # Explicit out-of-scope triggers
    out_of_scope_terms = [
        "breast cancer", "mammograph", "prostate", "hypertension", "blood pressure",
        "weight loss diet", "weather", "metformin", "surgical protocol", "diabetes dose"
    ]
    for term in out_of_scope_terms:
        if term in query_lower and "tb" not in query_lower and "tuberculosis" not in query_lower:
            return True

    # Check if none of the top chunks share core clinical keywords with query
    query_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", query_lower))
    stop_words = {"what", "which", "when", "where", "should", "could", "would", "about", "guideline", "recommend", "recommended", "treatment", "adults", "patient", "patients"}
    clinical_query_words = query_words - stop_words

    if not clinical_query_words:
        return False

    # Check matches across top 3 chunks
    matched_words = set()
    for c in top_chunks[:3]:
        c_text = (c.get("content", "") + " " + c.get("section", "")).lower()
        for w in clinical_query_words:
            if w in c_text:
                matched_words.add(w)

    # If less than 20% of query keywords appear in top chunks and no TB terms
    overlap_ratio = len(matched_words) / len(clinical_query_words) if clinical_query_words else 1.0
    if overlap_ratio < 0.25 and not any(tb_term in query_lower for tb_term in ["tb", "tuberculosis", "mycobacteri", "rifampicin", "isoniazid", "lam"]):
        return True

    return False


def select_evidence(retrieved_chunks: List[Dict[str, Any]], query: str, max_evidence: int = 4) -> List[Dict[str, Any]]:
    """
    Selects the top distinct, highly relevant clinical evidence chunks.
    Filters noise, scores evidence, and ensures citations will originate
    solely from these final chosen chunks.
    """
    # Check out-of-scope query
    if is_query_out_of_scope(query, retrieved_chunks):
        return []

    # 1. Filter obvious noise
    candidate_chunks = [c for c in retrieved_chunks if not is_noise_chunk(c, query)]
    if not candidate_chunks:
        candidate_chunks = retrieved_chunks  # fallback if all were filtered

    # 2. Score candidates
    scored = []
    for c in candidate_chunks:
        ev_score = calculate_evidence_score(c, query)
        c_copy = dict(c)
        c_copy["evidence_score"] = ev_score
        scored.append(c_copy)

    # 3. Sort by evidence score descending
    scored.sort(key=lambda x: x["evidence_score"], reverse=True)

    # 4. Diversity / Deduplication (prevent picking 3 nearly identical chunks from same page/paragraph)
    selected: List[Dict[str, Any]] = []
    seen_pages = set()

    for item in scored:
        page_key = (item["document_name"], item["page_number"])
        # Allow at most 2 chunks from the exact same page if highly relevant
        page_count = sum(1 for (doc, p) in seen_pages if (doc, p) == page_key)
        
        if page_count < 2:
            selected.append(item)
            seen_pages.add(page_key)

        if len(selected) >= max_evidence:
            break

    # If we have too few due to page restriction, fill from remaining
    if len(selected) < max_evidence and len(scored) > len(selected):
        for item in scored:
            if item not in selected:
                selected.append(item)
            if len(selected) >= max_evidence:
                break

    return selected
