"""
WHO TB Clinical RAG - Ingestion Pipeline
----------------------------------------
Extracts structured text from WHO TB guideline PDFs, detects section hierarchy,
records bounding boxes and page offsets for precise PDF highlighting, generates
dense embeddings (ChromaDB), and builds a BM25 sparse index for hybrid search.

Usage:
    python ingest.py
"""
import os
import sys
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any

import pymupdf  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import config


class SentenceTransformerEmbeddingFunction:
    """Chroma-compatible embedding function using local SentenceTransformer."""
    def __init__(self, model_name: str = config.FAST_EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()


def get_embedding_model():
    """Returns local sentence transformer model."""
    return SentenceTransformer(config.FAST_EMBED_MODEL)


def extract_sections_from_toc(doc) -> List[Dict[str, Any]]:
    """Extracts table of contents hierarchy from PyMuPDF document."""
    toc = doc.get_toc()
    sections = []
    for level, title, page in toc:
        clean_title = re.sub(r"\s+", " ", title).strip()
        sections.append({
            "level": level,
            "title": clean_title,
            "page": page  # 1-indexed page
        })
    return sections


def find_section_for_page(sections: List[Dict[str, Any]], page_num: int) -> str:
    """Finds the most specific section title applicable to a given 1-indexed page."""
    applicable = [s for s in sections if s["page"] <= page_num]
    if not applicable:
        return "General Guideline"
    # Sort by page descending, then level descending (most specific deepest level on that page or earlier)
    applicable.sort(key=lambda s: (s["page"], s["level"]), reverse=True)
    return applicable[0]["title"]


def load_tb_pdfs(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Parses TB PDF files in data_dir, extracting pages with TOC sections,
    text blocks, and bounding boxes.
    """
    all_pages = []
    pdf_files = sorted(data_dir.glob("*.pdf"))

    # Prefer TB module documents if present
    tb_files = [f for f in pdf_files if "Module" in f.name or "TB" in f.name or "tuberculosis" in f.name.lower()]
    target_files = tb_files if tb_files else pdf_files

    if not target_files:
        print(f"[!] No PDF files found in {data_dir}/")
        sys.exit(1)

    print(f"[*] Found {len(target_files)} PDF guideline(s) to process:")
    for f in target_files:
        print(f"    - {f.name}")

    for pdf_path in target_files:
        doc = pymupdf.open(str(pdf_path))
        doc_name = pdf_path.stem
        file_name = pdf_path.name
        sections = extract_sections_from_toc(doc)
        total_pages = len(doc)
        print(f"[*] Extracting '{file_name}' ({total_pages} pages)...")

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed
            page = doc[page_idx]
            
            # Extract text blocks with bbox coordinates
            blocks = page.get_text("blocks")
            # Filter text blocks (type == 0 is text)
            text_blocks = [b for b in blocks if len(b) >= 5 and b[6] == 0 and b[4].strip()]
            
            page_text = page.get_text("text").strip()
            if not page_text or len(page_text) < 30:
                continue  # Skip blank / cover-only decorative pages

            # Determine section
            section = find_section_for_page(sections, page_num)

            # Check if page has prominent in-page heading like "Recommendation 1", "Chapter X"
            rec_match = re.search(r"(Recommendation\s+\d+[\.\d]*[^\n]*)", page_text, re.IGNORECASE)
            if rec_match:
                rec_title = rec_match.group(1).strip()
                if len(rec_title) < 120:
                    section = f"{section} - {rec_title}"

            # Calculate primary bounding boxes for text on this page
            bboxes = []
            for b in text_blocks:
                # b: (x0, y0, x1, y1, text, block_no, block_type)
                bboxes.append({
                    "bbox": [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)],
                    "text_preview": b[4].strip()[:100]
                })

            all_pages.append({
                "text": page_text,
                "document_name": doc_name,
                "file_name": file_name,
                "page_number": page_num,
                "section": section,
                "bboxes": bboxes,
                "page_rect": [0, 0, round(page.rect.width, 2), round(page.rect.height, 2)]
            })

    print(f"[*] Total parsed content pages: {len(all_pages)}")
    return all_pages


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunks pages using recursive splitting while preserving section, document,
    page number, and coordinates metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_counter = 0

    for page in pages:
        page_text = page["text"]
        split_texts = splitter.split_text(page_text)

        for text_part in split_texts:
            clean_part = text_part.strip()
            if len(clean_part) < 40:
                continue

            chunk_id = f"{page['document_name']}-p{page['page_number']}-c{chunk_counter}"
            
            # Find best matching bbox for this chunk
            primary_bbox = [0, 0, 0, 0]
            for b in page.get("bboxes", []):
                preview = b["text_preview"].split()
                if preview and " ".join(preview[:3]).lower() in clean_part.lower():
                    primary_bbox = b["bbox"]
                    break

            chunks.append({
                "chunk_id": chunk_id,
                "content": clean_part,
                "document_name": page["document_name"],
                "file_name": page["file_name"],
                "page_number": page["page_number"],
                "section": page["section"],
                "bbox": primary_bbox,
                "exact_quote": clean_part[:180].replace("\n", " ").strip()
            })
            chunk_counter += 1

    print(f"[*] Created {len(chunks)} semantic chunks with citation metadata.")
    return chunks


def build_indices(chunks: List[Dict[str, Any]]):
    """
    Builds and persists:
    1. ChromaDB vector database with dense embeddings
    2. BM25 index with tokenized chunks and metadata
    """
    # 1. ChromaDB setup
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Reset or get collection
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("[*] Generating dense embeddings for ChromaDB...")
    emb_model = get_embedding_model()

    batch_size = 64
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        ids = [c["chunk_id"] for c in batch]
        docs = [c["content"] for c in batch]
        metadatas = [{
            "document_name": c["document_name"],
            "file_name": c["file_name"],
            "page_number": c["page_number"],
            "section": c["section"],
            "chunk_id": c["chunk_id"],
            "bbox": str(c["bbox"]),
            "exact_quote": c["exact_quote"]
        } for c in batch]
        
        embeddings = emb_model.encode(docs, normalize_embeddings=True, show_progress_bar=False).tolist()
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas
        )
        if (i + batch_size) % 256 == 0 or (i + batch_size) >= total:
            print(f"    Indexed {min(i + batch_size, total)}/{total} chunks into ChromaDB")

    # 2. BM25 index setup
    print("[*] Building BM25 sparse index...")
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9\-\_]+\b", text.lower())

    tokenized_corpus = [tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25": bm25,
        "chunks": chunks
    }
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"[*] Saved BM25 index to {config.BM25_INDEX_PATH}")
    print("[OK] Ingestion & indexing completed successfully!")


def main():
    print("=== WHO TB Clinical RAG: Ingestion & Indexing ===")
    pages = load_tb_pdfs(config.DATA_DIR)
    chunks = chunk_pages(pages)
    build_indices(chunks)


if __name__ == "__main__":
    main()
