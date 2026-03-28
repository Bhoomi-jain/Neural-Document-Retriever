"""
PaperTrail — Gemini Embeddings
================================
Uses Google's gemini-embedding-exp-03-07 model (3072-dim).
Handles batching, rate limits, and task-type differentiation
(RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for queries).
"""

from __future__ import annotations

import logging
import time
from typing import List

import google.generativeai as genai

from config import (
    GOOGLE_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    EMBEDDING_TASK_TYPE,
    QUERY_TASK_TYPE,
)

logger = logging.getLogger(__name__)

# Initialise once
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini embedding API allows max 100 texts per batch
_BATCH_SIZE = 10
_RETRY_WAIT = 10.0  # seconds between retries on rate limit


def _embed_batch(texts: List[str], task_type: str) -> List[List[float]]:
    """Embed a single batch with retry logic."""
    for attempt in range(3):
        try:
            result = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=texts,
                task_type=task_type,
            )
            return result["embedding"] if len(texts) == 1 else [e for e in result["embedding"]]
        except Exception as exc:
            logger.warning("Embedding attempt %d failed: %s", attempt + 1, exc)
            time.sleep(_RETRY_WAIT * (attempt + 1))
    raise RuntimeError(f"Embedding failed after 3 attempts for batch of {len(texts)}")


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of document texts for indexing.
    Uses RETRIEVAL_DOCUMENT task type for better retrieval performance.
    """
    if not texts:
        return []

    all_vecs = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        vecs  = _embed_batch(batch, EMBEDDING_TASK_TYPE)
        # Single text returns a flat list, not list of lists
        if batch and isinstance(vecs[0], float):
            vecs = [vecs]
        all_vecs.extend(vecs)
        logger.debug("Embedded batch %d/%d", i + _BATCH_SIZE, len(texts))

    return all_vecs


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.
    Uses RETRIEVAL_QUERY task type — different from document embedding,
    which is the correct industry practice for asymmetric retrieval.
    """
    result = genai.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        content=query,
        task_type=QUERY_TASK_TYPE,
    )
    vec = result["embedding"]
    # Flatten if nested
    if vec and isinstance(vec[0], list):
        vec = vec[0]
    return vec
