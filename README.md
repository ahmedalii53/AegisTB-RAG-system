# AegisTB - AI-Powered WHO TB Clinical Decision Support System

A production-ready Retrieval-Augmented Generation (RAG) system that answers tuberculosis clinical
questions exclusively from official WHO TB Guidelines, with exact citations, confidence scoring,
and an interactive PDF document viewer.

---

## Table of Contents

1. Project Overview
2. System Architecture
3. Key Features
4. Tech Stack
5. Project Structure
6. Prerequisites
7. Installation and Setup
8. Running the Application
9. Testing and Evaluation
10. Running on a Different Machine
11. API Reference

---

## Project Overview

AegisTB is a clinical decision-support tool for the medical domain. It retrieves evidence exclusively
from WHO Tuberculosis guidelines and generates grounded, structured answers -- never hallucinating or
pulling information from outside the indexed knowledge base.

Knowledge Base Sources:
- Module 3: Diagnosis.pdf   -- 184 pages
- Module 4: Treatment and Care.pdf -- 458 pages
- Total indexed: 1,339 semantic chunks

---

## System Architecture

The pipeline processes every user query through these sequential stages:

    User Query
          |
          v
    FastAPI  <-- Serves frontend + REST API (/health /query /pdf)
          |
          v
    Retrieval Engine  (retrieval.py)
      Dense Semantic Search (ChromaDB + sentence-transformers)
      + BM25 Sparse Keyword Search
      = RRF Fusion (Reciprocal Rank Fusion)
          |
          v
    Evidence Selector  (evidence_selector.py)
      Filter noise, detect off-topic queries, rank by relevance
          |
          v
    Generator -- Gemini 2.5 Flash  (generator.py)
      Grounded strictly on WHO retrieved text
      Enforces JSON Schema, fallback to raw WHO text
          |
          v
    Structured JSON Response
      { recommendation, evidence, citations, confidence }
          |
          v
    AegisTB Frontend  (static/)
      Chat History Drawer, Split-Screen Layout,
      PDF.js Viewer, Sentence-Level Text Highlight

---

## Key Features

| Feature | Details |
|---|---|
| Hybrid Retrieval | Dense semantic (ChromaDB) fused with BM25 sparse via Reciprocal Rank Fusion |
| Zero Hallucination | Gemini grounded strictly on retrieved WHO chunks; no outside information |
| Out-of-Scope Refusal | Off-topic queries return Confidence: Insufficient -- never fabricated |
| Exact Citations | Cites only the specific WHO pages and sections used in generation |
| Interactive PDF Viewer | Click any citation to jump to exact page with sentence highlighting |
| Chat History | Sessions in browser localStorage with rename and delete support |
| Dark / Light Mode | Full theme toggle with AegisTB brand color palette |
| Split-Screen UI | Centered hero transitions to chat + PDF viewer on first query |
| Benchmark Evaluated | 100% pass rate on a 7-case TB Clinical Benchmark suite |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.10+, FastAPI, Uvicorn |
| LLM | Google Gemini 2.5 Flash (via google-generativeai) |
| Vector Store | ChromaDB (persistent local storage) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (fully local, no API cost) |
| Sparse Retrieval | BM25 via rank_bm25 |
| PDF Parsing | PyMuPDF (fitz) |
| Frontend | HTML5, Vanilla CSS, Vanilla JavaScript |
| PDF Viewer | PDF.js (CDN) |
| Testing | Pytest |
| Evaluation | Custom CSV-based benchmark runner |

---

## Project Structure

    AegisTB/
    |
    |-- app.py                   FastAPI server: /health /query /pdf endpoints
    |-- config.py                Central config: model, chunk size, weights, paths
    |-- ingest.py                PDF ingestion: extract text + bboxes -> ChromaDB + BM25
    |-- retrieval.py             Hybrid search: Dense + BM25 + RRF fusion
    |-- evidence_selector.py     Chunk scoring, noise filtering, off-topic detection
    |-- generator.py             Gemini-grounded generation + JSON schema validation
    |-- query.py                 Full pipeline runner and CLI interface
    |-- requirements.txt         All Python dependencies
    |-- .env.example             Template for environment variables
    |
    |-- data/                    WHO Guideline source PDFs (knowledge base)
    |   |-- Module 3 Diagnosis.pdf
    |   `-- Module 4 Treatment and care.pdf
    |
    |-- chroma_db/               Persistent ChromaDB vector index (built by ingest.py)
    |-- bm25_index.pkl           BM25 serialized index (built by ingest.py)
    |
    |-- schema/
    |   `-- response_schema.json  Enforced JSON Schema for all LLM responses
    |
    |-- static/                  Frontend assets served by FastAPI
    |   |-- index.html           Main UI: hero, split-screen layout, modals, drawers
    |   |-- app.js               Chat logic, session history, PDF viewer, highlighting
    |   |-- style.css            AegisTB design system, dark/light themes
    |   `-- logo.png             AegisTB brand logo (shield + lungs)
    |
    |-- eval/                    Evaluation suite
    |   |-- TB_Clinical_Benchmark.csv
    |   `-- run_eval.py          Benchmark runner and report generator
    |
    `-- tests/
        `-- test_pipeline.py     10-test Pytest suite (unit + integration)

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | https://www.python.org/downloads/ |
| pip | Included with Python |
| Google Gemini API Key | https://aistudio.google.com/app/apikey |
| Disk Space | ~2 GB (embedding model + ChromaDB + PDF files) |

---

## Installation and Setup

### Step 1 - Get the project

Copy the full project folder to your machine and open a terminal in the project root directory.

### Step 2 - Create a virtual environment (recommended)

    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate

### Step 3 - Install all dependencies

    pip install -r requirements.txt

Note: First run downloads the embedding model all-MiniLM-L6-v2 (~90 MB) automatically.

### Step 4 - Configure your API key

    # Windows
    copy .env.example .env

    # macOS / Linux
    cp .env.example .env

Open the .env file and fill in your Gemini API key:

    GEMINI_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY_HERE
    GEMINI_MODEL=gemini-2.5-flash
    GEMINI_TEMPERATURE=0.0

### Step 5 - Build the knowledge base index

SKIP this step if you transferred chroma_db/ and bm25_index.pkl from another machine.
Those files are portable and ready to use.

    python ingest.py

This reads all PDFs from data/, extracts text with page coordinates,
and builds both the ChromaDB vector store and the BM25 sparse index.
Expected result: 1,339 chunks indexed.

---

## Running the Application

### Start the server

    python app.py

Expected output:
    [*] Starting AegisTB Clinical RAG Server on http://127.0.0.1:8000
    INFO:     Uvicorn running on http://127.0.0.1:8000

### Open the web interface

Navigate to http://127.0.0.1:8000 in your browser.

- A centered hero search bar appears on initial load.
- After the first query: transitions to split-screen (chat left + PDF viewer right).
- Clicking any citation card jumps to the exact PDF page with highlighted text.

### Use the CLI directly (no browser needed)

    python query.py <question in quotes>
    python query.py <question in quotes> --json

Examples:
    python query.py "What is the recommended diagnostic test for pulmonary TB?"
    python query.py "What is the standard treatment for drug-susceptible TB?" --json

---

## Testing and Evaluation

### Run unit and integration tests

    python -m pytest tests/test_pipeline.py -v

Expected: 9 passed, 1 skipped (API endpoint test skipped without live server)

### Run the clinical benchmark

    python eval/run_eval.py

Expected output:
    === AegisTB Clinical Benchmark Report ===
    Category          Pass   Fail
    Diagnosis           2      0
    HIV Co-infection    1      0
    DS-TB Regimen       1      0
    Refusal Cases       2      0
    Broad Query         1      0
    ------------------------------------------
    TOTAL               7/7    Pass Rate: 100.0%

---

## Running on a Different Machine

### Option A - Transfer with pre-built index (fastest, recommended)

Copy the entire project folder including:
- chroma_db/       (pre-built vector index)
- bm25_index.pkl   (pre-built BM25 index)

Then on the new machine:

    pip install -r requirements.txt
    # Edit .env with your Gemini API key
    python app.py

No re-indexing needed. Index files work across different operating systems.

### Option B - Fresh install without index files

Use this if chroma_db/ and bm25_index.pkl were not included in the transfer.

    pip install -r requirements.txt
    # Edit .env with your Gemini API key
    python ingest.py     # Rebuild index from PDFs (takes 2-5 minutes)
    python app.py

---

## API Reference

### GET /health

Returns system status and knowledge base statistics.

    Response:
    {
      "status": "healthy",
      "service": "AegisTB Clinical RAG",
      "gemini_model": "gemini-2.5-flash",
      "chroma_collection": "tb_guidelines",
      "total_indexed_chunks": 1339,
      "indexed_pdfs": ["Module 3 Diagnosis.pdf", "Module 4 Treatment and care.pdf"]
    }

### POST /query

Submit a clinical question in natural language.

    Request body:
    { "question": "What is the recommended treatment for drug-susceptible TB?" }

    Response:
    {
      "recommendation": "WHO recommends a standard 6-month treatment regimen...",
      "evidence": "The standard treatment for DS-TB consists of...",
      "citations": [
        {
          "document": "WHO TB Module 4 - Treatment and Care 2022",
          "section": "4.1 Standard Treatment Regimens",
          "page": 42,
          "file_name": "Module 4 Treatment and care.pdf",
          "exact_quote": "The recommended treatment regimen for new TB patients..."
        }
      ],
      "confidence": "High"
    }

### GET /pdf/{filename}

Streams the original WHO guideline PDF file for the in-browser viewer.

    GET /pdf/Module%203%20Diagnosis.pdf
    GET /pdf/Module%204%20Treatment%20and%20care.pdf

---

> AegisTB -- Built for AI Hackathon | Powered by Google Gemini 2.5 Flash | Grounded exclusively on WHO TB Evidence