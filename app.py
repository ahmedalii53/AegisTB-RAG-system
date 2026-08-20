"""
WHO TB Clinical RAG - FastAPI Backend Server
--------------------------------------------
Provides REST endpoints:
- GET  /health          : Health check and index metadata
- POST /query           : Executes RAG pipeline and returns structured response
- GET  /pdf/{filename}  : Serves PDF documents for in-browser viewing
- GET  /                : Serves clinical UI

Usage:
    python app.py
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from query import run_rag_pipeline, get_retriever

app = FastAPI(
    title="AegisTB — WHO Tuberculosis Clinical Decision Support API",
    description="Evidence-grounded clinical decision support system strictly based on WHO TB guidelines.",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = config.BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    question: str = Field(..., description="Clinical question to ask against WHO TB guidelines", min_length=2)


class CitationItem(BaseModel):
    document: str
    file_name: Optional[str] = None
    section: str
    page: int
    exact_quote: Optional[str] = None
    bbox: Optional[List[float]] = None


class QueryResponse(BaseModel):
    recommendation: str
    evidence: str
    citations: List[CitationItem]
    confidence: str
    retrieved_chunks_count: Optional[int] = 0
    evidence_count: Optional[int] = 0
    selected_evidence_preview: Optional[List[Dict[str, Any]]] = None


@app.get("/health")
def health_check():
    """Health check endpoint confirming API status and index readiness."""
    pdf_files = [f.name for f in config.DATA_DIR.glob("*.pdf")]
    
    chroma_ready = False
    total_chunks = 0
    try:
        retriever = get_retriever()
        total_chunks = retriever.collection.count()
        chroma_ready = True
    except Exception as e:
        chroma_ready = False

    return {
        "status": "healthy" if chroma_ready else "initializing",
        "service": "WHO TB Clinical RAG",
        "gemini_model": config.GEMINI_MODEL,
        "chroma_collection": config.COLLECTION_NAME,
        "total_indexed_chunks": total_chunks,
        "indexed_pdfs": pdf_files
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Executes the clinical RAG pipeline:
    Retrieval → Hybrid Dense + BM25 → Reranking → Evidence Selection → Context → Gemini → Validation
    """
    try:
        result = run_rag_pipeline(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/pdf/{filename}")
def get_pdf(filename: str):
    """Serves the PDF guideline file for the in-browser viewer."""
    safe_filename = Path(filename).name
    pdf_path = config.DATA_DIR / safe_filename
    
    # Check if exact file exists or with .pdf extension
    if not pdf_path.exists():
        pdf_path = config.DATA_DIR / f"{safe_filename}.pdf"

    if not pdf_path.exists():
        # Try matching stem
        matches = list(config.DATA_DIR.glob(f"*{safe_filename}*"))
        if matches:
            pdf_path = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found.")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
        headers={"Content-Disposition": f"inline; filename=\"{pdf_path.name}\""}
    )


# Mount static frontend
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serves the main frontend clinical chat application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "WHO TB Clinical RAG API is running. UI loading at /static/index.html"})


if __name__ == "__main__":
    print(f"[*] Starting WHO TB Clinical RAG Server on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run("app:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=False)
