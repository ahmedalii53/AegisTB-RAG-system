"""
WHO TB Clinical RAG - Hybrid Retrieval Module
---------------------------------------------
Combines Dense Vector Search (ChromaDB) with Sparse Keyword Search (BM25)
using Reciprocal Rank Fusion (RRF) and clinical term boosting.
"""
import pickle
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

import config


class HybridRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.client.get_collection(name=config.COLLECTION_NAME)
        self.embedding_model = SentenceTransformer(config.FAST_EMBED_MODEL)
        
        # Load BM25 index
        if not Path(config.BM25_INDEX_PATH).exists():
            raise FileNotFoundError(f"BM25 index not found at {config.BM25_INDEX_PATH}. Run ingest.py first.")
            
        with open(config.BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.chunks = bm25_data["chunks"]
            self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9\-\_]+\b", text.lower())

    def dense_search(self, query: str, top_k: int = config.DENSE_TOP_K) -> List[Dict[str, Any]]:
        """Dense semantic search using cosine similarity in ChromaDB."""
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        dense_hits = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), 1):
                # Chroma cosine distance is in [0, 2]; cosine similarity = 1 - (dist / 2) or 1 - dist
                similarity = max(0.0, 1.0 - (dist if dist <= 1.0 else dist / 2.0))
                dense_hits.append({
                    "chunk_id": cid,
                    "content": doc,
                    "document_name": meta.get("document_name", ""),
                    "file_name": meta.get("file_name", f"{meta.get('document_name')}.pdf"),
                    "page_number": int(meta.get("page_number", 1)),
                    "section": meta.get("section", ""),
                    "bbox": eval(meta.get("bbox", "[0,0,0,0]")),
                    "exact_quote": meta.get("exact_quote", doc[:180]),
                    "dense_rank": rank,
                    "dense_score": float(similarity)
                })
        return dense_hits

    def sparse_search(self, query: str, top_k: int = config.SPARSE_TOP_K) -> List[Dict[str, Any]]:
        """Sparse keyword search using BM25Okapi."""
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        sparse_hits = []
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        for rank, idx in enumerate(top_indices, 1):
            raw_score = float(scores[idx])
            if raw_score <= 0.0:
                continue
            chunk = self.chunks[idx]
            norm_score = min(1.0, raw_score / (max_score + 1e-6))
            sparse_hits.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "document_name": chunk["document_name"],
                "file_name": chunk["file_name"],
                "page_number": int(chunk["page_number"]),
                "section": chunk["section"],
                "bbox": chunk.get("bbox", [0, 0, 0, 0]),
                "exact_quote": chunk.get("exact_quote", chunk["content"][:180]),
                "sparse_rank": rank,
                "sparse_score": norm_score
            })
        return sparse_hits

    def hybrid_search(self, query: str, top_k: int = config.RERANK_TOP_K) -> List[Dict[str, Any]]:
        """
        Executes dense and sparse searches and fuses them using Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (60 + dense_rank) + 1 / (60 + sparse_rank)
        """
        dense_hits = self.dense_search(query, top_k=config.DENSE_TOP_K)
        sparse_hits = self.sparse_search(query, top_k=config.SPARSE_TOP_K)

        k_rrf = 60
        merged_scores: Dict[str, float] = {}
        merged_chunks: Dict[str, Dict[str, Any]] = {}

        # Process dense hits
        for item in dense_hits:
            cid = item["chunk_id"]
            rrf_score = 1.0 / (k_rrf + item["dense_rank"])
            merged_scores[cid] = merged_scores.get(cid, 0.0) + rrf_score
            merged_chunks[cid] = item

        # Process sparse hits
        for item in sparse_hits:
            cid = item["chunk_id"]
            rrf_score = 1.0 / (k_rrf + item["sparse_rank"])
            merged_scores[cid] = merged_scores.get(cid, 0.0) + rrf_score
            if cid not in merged_chunks:
                merged_chunks[cid] = item
            else:
                # Merge sparse info
                merged_chunks[cid]["sparse_rank"] = item["sparse_rank"]
                merged_chunks[cid]["sparse_score"] = item["sparse_score"]

        # Sort by merged RRF score
        sorted_cids = sorted(merged_scores.keys(), key=lambda cid: merged_scores[cid], reverse=True)

        results = []
        for cid in sorted_cids[:top_k]:
            chunk_data = merged_chunks[cid]
            chunk_data["hybrid_score"] = merged_scores[cid]
            results.append(chunk_data)

        return results
