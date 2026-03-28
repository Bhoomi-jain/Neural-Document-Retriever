"""
PaperTrail Configuration
=========================
Industry-grade RAG configuration.

Stack:
  - Google Gemini Embeddings  (gemini-embedding-exp-03-07, 3072-dim)
  - Cohere Rerank 3.5         (neural cross-encoder reranker)
  - ChromaDB                  (vector store, persistent)
  - BM25                      (keyword search, via rank_bm25)
  - RRF                       (Reciprocal Rank Fusion for hybrid search)
  - Anthropic Claude          (generation + query rewriting)
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
UPLOAD_DIR   = BASE_DIR / "uploads"
CACHE_DIR    = BASE_DIR / "cache"
CHROMA_DIR   = BASE_DIR / "chroma_db"
BM25_DIR     = BASE_DIR / "bm25_index"
ASSETS_DIR   = BASE_DIR / "assets"

for _d in (UPLOAD_DIR, CACHE_DIR, CHROMA_DIR, BM25_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────
#ANTHROPIC_API_KEY : str = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY    : str = os.getenv("GOOGLE_API_KEY", "")
COHERE_API_KEY    : str = os.getenv("COHERE_API_KEY", "")

# ── Embedding Model ────────────────────────────────────────────────────────────
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
EMBEDDING_DIM          = 3072        # gemini-embedding-exp-03-07 output dim
EMBEDDING_TASK_TYPE    = "RETRIEVAL_DOCUMENT"   # for indexing
QUERY_TASK_TYPE        = "RETRIEVAL_QUERY"      # for query embedding

# ── LLM ───────────────────────────────────────────────────────────────────────
#ANTHROPIC_MODEL  = "claude-opus-4-5"
#MAX_TOKENS       = 2048

GEMINI_LLM_MODEL = "gemini-2.5-flash" 
MAX_TOKENS       = 2048

# ── Reranker ───────────────────────────────────────────────────────────────────
COHERE_RERANK_MODEL = "rerank-v3.5"
RERANK_TOP_N        = 5             # how many chunks survive reranking

# ── Semantic Chunking ──────────────────────────────────────────────────────────
CHUNK_MAX_CHARS     = 800           # hard ceiling per chunk
CHUNK_MIN_CHARS     = 80            # discard tiny fragments
CHUNK_OVERLAP_CHARS = 100           # overlap between consecutive chunks

# ── Hybrid Search ──────────────────────────────────────────────────────────────
VECTOR_TOP_K        = 20            # candidates from vector search
BM25_TOP_K          = 20            # candidates from BM25
RRF_K               = 60            # RRF constant (standard = 60)
HYBRID_FINAL_K      = 20            # merged candidates sent to reranker

# ── Confidence threshold ───────────────────────────────────────────────────────
MIN_RELEVANCE_SCORE = 0.2           # Cohere scores below this → "not found"

# ── Query rewriting ────────────────────────────────────────────────────────────
ENABLE_QUERY_REWRITE = True         # rewrite query before searching

# ── Conversation memory ────────────────────────────────────────────────────────
MAX_HISTORY_TURNS   = 6

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_MIN_CHARS       = 40
TESSERACT_LANG      = "eng+fra"
