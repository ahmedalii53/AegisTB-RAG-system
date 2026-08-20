"""
Central configuration for the WHO TB Clinical RAG System.
All modules read settings from this file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
BM25_INDEX_PATH = BASE_DIR / "bm25_index.pkl"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "who_tb_guidelines")

# --- Chunking ---
# Values in tokens (approx 4 chars per token)
CHUNK_SIZE = 450
CHUNK_OVERLAP = 60

# --- Embeddings ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
LOCAL_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
FAST_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- Retrieval ---
DENSE_TOP_K = 15
SPARSE_TOP_K = 15
RERANK_TOP_K = 8
FINAL_EVIDENCE_TOP_K = 4
MIN_RELEVANCE_SCORE = 0.35

# --- Gemini Generation ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip('\'" ')
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip('\'" ')
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))

# --- Server ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
