"""
PaperTrail — Hybrid Search
============================
Industry-standard retrieval pipeline:

  1. Vector search      → ChromaDB cosine similarity (Gemini embeddings)
  2. BM25 keyword search → rank_bm25 (exact term matching)
  3. RRF fusion         → Reciprocal Rank Fusion merges both result lists
  4. Cohere Rerank 3.5  → Neural cross-encoder scores each chunk vs query
  5. Confidence filter  → Drop chunks below MIN_RELEVANCE_SCORE

This catches:
  - Semantic matches (vector)     e.g. "engine failure" ↔ "motor malfunction"
  - Exact matches (BM25)          e.g. product codes, names, acronyms
  - Relevance quality (reranker)  cross-attention between query and chunk text
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import chromadb
import cohere
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_DIR, BM25_DIR, COHERE_API_KEY,
    COHERE_RERANK_MODEL, RERANK_TOP_N,
    VECTOR_TOP_K, BM25_TOP_K, RRF_K, HYBRID_FINAL_K,
    MIN_RELEVANCE_SCORE,
)
from embeddings import embed_documents, embed_query
from parser import Chunk

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    text          : str
    metadata      : dict
    relevance_score: float          # Cohere relevance score 0–1
    vector_rank   : Optional[int]   # rank in vector results (None if not in top-K)
    bm25_rank     : Optional[int]   # rank in BM25 results


# ── ChromaDB ───────────────────────────────────────────────────────────────────

_chroma: Optional[chromadb.PersistentClient] = None

def _get_chroma() -> chromadb.PersistentClient:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma

def _col_name(filename: str) -> str:
   
    # 1. Get the filename without extension
    stem = Path(filename).stem
    
    # 2. Replace ANY sequence of non-alphanumeric chars with a SINGLE underscore
    # This prevents the "Lab_1__Practical" issue
    safe = re.sub(r'[^a-zA-Z0-9]+', '_', stem)
    
    # 3. Trim leading/trailing underscores to satisfy Chroma's start/end rules
    safe = safe.strip('_')
    
    # 4. Ensure it starts with a letter if required (optional, but safer)
    if not safe or not safe[0].isalnum():
        safe = "doc_" + safe
        
    # 5. Length check (3-63 chars)
    if len(safe) < 3:
        safe = f"col_{safe}"
        
    return safe[:63]


# ── BM25 store ─────────────────────────────────────────────────────────────────

class BM25Store:
    """Persists a BM25 index per document collection."""

    def __init__(self, col_name: str):
        self.path      = BM25_DIR / f"{col_name}.bm25.pkl"
        self.col_name  = col_name
        self._index    : Optional[BM25Okapi]  = None
        self._corpus   : List[str]            = []
        self._meta     : List[dict]           = []

    def build(self, chunks: List[Chunk]) -> None:
        self._corpus = [c.text for c in chunks]
        self._meta   = [c.to_dict() for c in chunks]
        tokenized    = [self._tokenize(t) for t in self._corpus]
        self._index  = BM25Okapi(tokenized)
        self._save()
        logger.info("BM25 index built for '%s' (%d docs)", self.col_name, len(chunks))

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self._index  = data["index"]
            self._corpus = data["corpus"]
            self._meta   = data["meta"]
            logger.info("BM25 index loaded for '%s'", self.col_name)
            return True
        except Exception as exc:
            logger.warning("BM25 load failed for '%s': %s", self.col_name, exc)
            return False

    def search(self, query: str, top_k: int) -> List[Tuple[int, float, str, dict]]:
        """Returns list of (rank, score, text, metadata)."""
        if self._index is None:
            return []
        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return [
            (rank, score, self._corpus[idx], self._meta[idx])
            for rank, (idx, score) in enumerate(ranked)
        ]

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _save(self) -> None:
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({
                "index" : self._index,
                "corpus": self._corpus,
                "meta"  : self._meta,
            }, f)


# ── Indexing ───────────────────────────────────────────────────────────────────

def index_document(
    chunks     : List[Chunk],
    filename   : str,
    on_progress=None,
) -> str:
    """
    Index a document into both ChromaDB (vector) and BM25 (keyword).
    Returns the collection name.
    """
    if not chunks:
        logger.warning(f"No chunks to index for {filename}")
        return
    
    col_name = _col_name(filename)
    client   = _get_chroma()

    # Clear existing
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    # BM25
    bm25 = BM25Store(col_name)
    bm25.build(chunks)

    # Vector — embed in batches
    batch_size = 50
    total      = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vecs  = embed_documents(texts)

        # Flatten if needed
        if vecs and isinstance(vecs[0], list) and isinstance(vecs[0][0], list):
            vecs = [v[0] for v in vecs]

        collection.add(
            ids        = [f"{col_name}_{c.chunk_index}" for c in batch],
            embeddings = vecs,
            documents  = texts,
            metadatas  = [c.to_dict() for c in batch],
        )

        if on_progress:
            done = min(i + batch_size, total)
            on_progress(done, total, f"Indexed {done}/{total} chunks")

    logger.info("Indexed %d chunks for '%s'", total, col_name)
    return col_name


# ── RRF Fusion ─────────────────────────────────────────────────────────────────

def _rrf_fuse(
    vector_hits: List[Tuple[str, dict]],   # (text, meta) ordered by vector rank
    bm25_hits  : List[Tuple[str, dict]],   # (text, meta) ordered by bm25 rank
    k          : int = RRF_K,
) -> List[Tuple[str, dict, float]]:
    """
    Reciprocal Rank Fusion:
      score(d) = Σ 1 / (k + rank(d))

    Returns merged list sorted by RRF score, deduplicated by text.
    """
    scores: Dict[str, float] = {}
    meta_map: Dict[str, dict] = {}

    for rank, (text, meta) in enumerate(vector_hits):
        key = text[:200]   # dedup key
        scores[key]   = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        meta_map[key] = meta

    for rank, (text, meta) in enumerate(bm25_hits):
        key = text[:200]
        scores[key]   = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        meta_map[key] = meta

    merged = sorted(scores.items(), key=lambda x: -x[1])
    return [(key, meta_map[key], rrf_score) for key, rrf_score in merged]


# ── Main search pipeline ───────────────────────────────────────────────────────

_cohere_client: Optional[cohere.Client] = None

def _get_cohere() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(api_key=COHERE_API_KEY)
    return _cohere_client


def hybrid_search(
    query          : str,
    collection_names: List[str],
    top_k          : int = VECTOR_TOP_K,
    bm25_k         : int = BM25_TOP_K,
    rerank_top_n   : int = RERANK_TOP_N,
) -> List[SearchResult]:
    """
    Full hybrid retrieval pipeline:
    vector search + BM25 → RRF → Cohere rerank → confidence filter
    """
    client = _get_chroma()
    query_vec = embed_query(query)

    vector_hits_raw : List[Tuple[str, dict, int]] = []   # text, meta, rank
    bm25_hits_raw   : List[Tuple[str, dict, int]] = []

    for col_name in collection_names:
        # ── Vector search ──────────────────────────────────────────────
        try:
            col   = client.get_collection(col_name)
            count = col.count()
            if count == 0:
                continue
            n = min(top_k, count)
            res = col.query(
                query_embeddings=[query_vec],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            for text, meta in zip(res["documents"][0], res["metadatas"][0]):
                vector_hits_raw.append((text, meta))
        except Exception as exc:
            logger.warning("Vector search failed for '%s': %s", col_name, exc)

        # ── BM25 search ────────────────────────────────────────────────
        bm25 = BM25Store(col_name)
        if bm25.load():
            for rank, score, text, meta in bm25.search(query, bm25_k):
                bm25_hits_raw.append((text, meta))

    if not vector_hits_raw and not bm25_hits_raw:
        logger.warning("No candidates found for query.")
        return []

    # ── RRF fusion ─────────────────────────────────────────────────────
    fused = _rrf_fuse(vector_hits_raw, bm25_hits_raw)
    fused = fused[:HYBRID_FINAL_K]

    if not fused:
        return []

    # ── Cohere Rerank ──────────────────────────────────────────────────
    try:
        co       = _get_cohere()
        docs     = [text for text, _, _ in fused]
        metas    = [meta for _, meta, _ in fused]

        rerank_response = co.rerank(
            model     = COHERE_RERANK_MODEL,
            query     = query,
            documents = docs,
            top_n     = rerank_top_n,
            return_documents=True,
        )

        results: List[SearchResult] = []
        for hit in rerank_response.results:
            score = hit.relevance_score
            if score < MIN_RELEVANCE_SCORE:
                continue
            idx  = hit.index
            text = docs[idx]
            meta = metas[idx]

            # Recover original ranks for transparency
            v_rank = next(
                (r for r, (t, _) in enumerate(vector_hits_raw) if t == text), None
            )
            b_rank = next(
                (r for r, (t, _) in enumerate(bm25_hits_raw) if t == text), None
            )

            results.append(SearchResult(
                text           = text,
                metadata       = meta,
                relevance_score= score,
                vector_rank    = v_rank,
                bm25_rank      = b_rank,
            ))

        logger.info(
            "Hybrid search: %d vector + %d BM25 → %d fused → %d reranked (≥%.2f)",
            len(vector_hits_raw), len(bm25_hits_raw),
            len(fused), len(results), MIN_RELEVANCE_SCORE,
        )
        return results

    except Exception as exc:
        logger.error("Cohere rerank failed: %s", exc)
        # Fallback: return fused results without reranking
        return [
            SearchResult(
                text            = text,
                metadata        = meta,
                relevance_score = rrf_score,
                vector_rank     = None,
                bm25_rank       = None,
            )
            for text, meta, rrf_score in fused[:rerank_top_n]
        ]


# ── Document management ────────────────────────────────────────────────────────

def list_collections() -> List[str]:
    return [c.name for c in _get_chroma().list_collections()]


def delete_collection(col_name: str) -> None:
    try:
        _get_chroma().delete_collection(col_name)
    except Exception:
        pass
    BM25Store(col_name).delete()
    logger.info("Deleted collection '%s'", col_name)
