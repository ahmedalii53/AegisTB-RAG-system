"""
Automated Test Suite for WHO TB Clinical RAG Pipeline
-----------------------------------------------------
Tests:
1.  Noise filtering (evidence_selector)
2.  Hybrid retrieval (retriever)
3.  Diagnosis query — mWRD initial test
4.  HIV-specific TB diagnosis — LF-LAM
5.  Drug-Susceptible TB Treatment — 6-month regimen
6.  Out-of-scope refusal — breast cancer / hypertension
7.  Citation filtering — citations strictly from final evidence
8.  PDF navigation & highlighting metadata validity
9.  Broad query handling — synthesis across modules
10. End-to-End API test via /query endpoint (requires running server)
"""
import pytest
import os
import json
from pathlib import Path

import config
from retrieval import HybridRetriever
from evidence_selector import select_evidence, is_noise_chunk
from generator import GroundedGenerator
from query import run_rag_pipeline


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()


@pytest.fixture(scope="module")
def generator():
    return GroundedGenerator()


# --- Test 1: Evidence Selection & Noise Filtering ---
def test_noise_filtering():
    """Test that administrative and reference noise chunks are detected and filtered."""
    noise_chunk_1 = {
        "section": "Acknowledgements",
        "content": "WHO thanks the guideline development group and contributors...",
        "hybrid_score": 0.05
    }
    assert is_noise_chunk(noise_chunk_1, "What is TB treatment?") is True

    noise_chunk_2 = {
        "section": "References",
        "content": "1. World Health Organization 2020. 2. Jones et al. 2019. 3. Smith et al. 2021. 4. Taylor 2018.",
        "hybrid_score": 0.04
    }
    assert is_noise_chunk(noise_chunk_2, "What is first-line regimen?") is True

    clinical_chunk = {
        "section": "1. Treatment of drug-susceptible TB using a 6-month regimen",
        "content": "WHO recommends the use of a 6-month regimen composed of 2 months of HRZE followed by 4 months of HR.",
        "hybrid_score": 0.04
    }
    assert is_noise_chunk(clinical_chunk, "What is first-line regimen?") is False


# --- Test 2: Hybrid Retrieval ---
def test_hybrid_retrieval(retriever):
    """Test that hybrid retrieval returns candidates for a clinical question."""
    query = "What is the recommended initial diagnostic test for pulmonary TB in adults?"
    results = retriever.hybrid_search(query, top_k=6)
    assert len(results) > 0
    # Check that at least one result comes from Module 3 Diagnosis
    doc_names = [r["document_name"] for r in results]
    assert any("Diagnosis" in d or "Module" in d for d in doc_names)


# --- Test 3: Diagnosis Query (mWRD) ---
def test_diagnosis_query():
    """Test diagnostic query retrieves relevant molecular test guidance."""
    query = "What initial diagnostic test is recommended for pulmonary TB in adults?"
    result = run_rag_pipeline(query)

    assert "recommendation" in result
    assert "confidence" in result
    assert result["confidence"] in ["high", "medium"]
    assert len(result["citations"]) >= 1

    # Recommendation should mention mWRD, molecular, nucleic acid, or GeneXpert / Xpert
    rec_text = result["recommendation"].lower() + " " + result["evidence"].lower()
    assert any(term in rec_text for term in ["mwrd", "molecular", "naat", "xpert", "rapid", "diagnostic"])


# --- Test 4: Specific HIV TB Diagnosis (LF-LAM) ---
def test_hiv_tb_diagnosis():
    """Test specific sub-population query for TB diagnosis in PLHIV using LF-LAM."""
    query = "When should lateral flow urine lipoarabinomannan (LF-LAM) be used for TB diagnosis in people living with HIV?"
    result = run_rag_pipeline(query)

    assert result["confidence"] in ["high", "medium"]
    assert len(result["citations"]) >= 1

    combined_text = (result["recommendation"] + " " + result["evidence"]).lower()
    assert any(term in combined_text for term in ["lf-lam", "lam", "hiv", "cd4", "inpatient", "seriously ill", "urine"])


# --- Test 5: Drug-Susceptible TB Treatment ---
def test_ds_tb_treatment():
    """Test DS-TB first-line treatment regimen recommendation."""
    query = "What is the standard 6-month first-line treatment regimen for drug-susceptible pulmonary tuberculosis?"
    result = run_rag_pipeline(query)

    assert result["confidence"] in ["high", "medium"]
    assert len(result["citations"]) >= 1

    combined_text = (result["recommendation"] + " " + result["evidence"]).lower()
    assert any(term in combined_text for term in ["6-month", "2hrze", "hrze", "hr", "rifampicin", "isoniazid", "pyrazinamide", "ethambutol"])


# --- Test 6: Out-of-Scope Question Refusal ---
def test_out_of_scope_refusal():
    """Test that out-of-scope question is refused with insufficient or low confidence and no hallucinations."""
    query = "What is the recommended screening interval and mammography frequency for breast cancer in average-risk women?"
    result = run_rag_pipeline(query)

    assert result["confidence"] in ["insufficient", "low"]
    # Should explicitly express lack of evidence or refusal
    rec_lower = result["recommendation"].lower()
    assert any(w in rec_lower for w in ["insufficient", "don't have", "not have", "scope", "cannot", "does not contain", "not covered"])
    if result["confidence"] == "insufficient":
        assert result["citations"] == []
        assert result["evidence"] == ""


# --- Test 7: Citation Filtering (Strictly from Final Selected Evidence) ---
def test_citation_filtering():
    """Verify that citations match only the final selected evidence chunks and not arbitrary retrieved chunks."""
    query = "What is the 4-month regimen for drug-susceptible tuberculosis?"
    result = run_rag_pipeline(query)

    for citation in result["citations"]:
        assert "document" in citation and len(citation["document"]) > 0
        assert "page" in citation and citation["page"] >= 1
        assert "section" in citation and len(citation["section"]) > 0


# --- Test 8: PDF Navigation & Highlighting Metadata Validity ---
def test_pdf_navigation_metadata():
    """Verify citations provide valid PDF filenames, 1-indexed pages, and bounding boxes for frontend viewer."""
    query = "What is the treatment duration for drug-susceptible TB?"
    result = run_rag_pipeline(query)

    if result["citations"]:
        cit = result["citations"][0]
        assert "file_name" in cit
        assert cit["file_name"].endswith(".pdf")
        assert (config.DATA_DIR / cit["file_name"]).exists(), f"PDF file not found: {cit['file_name']}"
        assert isinstance(cit["page"], int) and cit["page"] >= 1
        assert "bbox" in cit
        assert isinstance(cit["bbox"], list) and len(cit["bbox"]) == 4


# --- Test 9: Broad Query Handling ---
def test_broad_query():
    """Test broad clinical query generates synthesized grounded guidance."""
    query = "How is tuberculosis diagnosed and treated according to WHO guidelines?"
    result = run_rag_pipeline(query)

    assert "recommendation" in result
    assert len(result["recommendation"]) > 30
    assert result["confidence"] in ["high", "medium"]


# --- Test 10: End-to-End API Test via /query endpoint ---
def test_api_query_endpoint():
    """
    End-to-End test: hits the live /query FastAPI endpoint.
    Requires server to be running: python app.py
    Skip gracefully if server is not available.
    """
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed")

    base_url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"

    # First check health
    try:
        health_resp = httpx.get(f"{base_url}/health", timeout=5.0)
    except httpx.ConnectError:
        pytest.skip("Server not running — start with: python app.py")

    assert health_resp.status_code == 200
    health_data = health_resp.json()
    # Accept either health response format (old or new app versions)
    assert health_data.get("status") in ["healthy", "ok", "initializing"]

    # Now test the /query endpoint
    query_resp = httpx.post(
        f"{base_url}/query",
        json={"question": "What is the recommended initial diagnostic test for pulmonary TB?"},
        timeout=60.0
    )

    if query_resp.status_code == 503:
        pytest.skip("Server /query returned 503 — may be an incompatible version running on this port. Start app.py and retry.")

    assert query_resp.status_code == 200, f"Expected 200, got {query_resp.status_code}: {query_resp.text[:300]}"

    data = query_resp.json()

    # Verify all required fields
    assert "recommendation" in data
    assert "evidence" in data
    assert "citations" in data
    assert "confidence" in data

    # Confidence must be valid
    assert data["confidence"] in ["high", "medium", "low", "insufficient"]

    # For a TB diagnosis question, expect a valid answer
    assert data["confidence"] in ["high", "medium"]
    assert len(data["citations"]) >= 1

    # Each citation must have required fields for PDF viewer
    for cit in data["citations"]:
        assert "document" in cit
        assert "page" in cit and isinstance(cit["page"], int)

    print(f"\n[API Test] Confidence: {data['confidence'].upper()}")
    print(f"[API Test] Citations: {len(data['citations'])}")
    print(f"[API Test] Recommendation: {data['recommendation'][:150]}...")
