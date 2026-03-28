"""
PaperTrail — Main Application
==============================
Industry-grade multi-document RAG with:
  - Gemini embeddings (asymmetric: document vs query task types)
  - Hybrid search: vector (ChromaDB) + keyword (BM25) fused via RRF
  - Cohere Rerank 3.5 neural reranker
  - Semantic chunking (paragraph/sentence aware)
  - Query rewriting via Gemini
  - Confidence threshold filtering
  - Retrieval quality evaluation metrics
  - Streaming responses with conversation memory
  - Chat export as Markdown
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# REMOVED ANTHROPIC_API_KEY from config imports
from config import (
    UPLOAD_DIR, GOOGLE_API_KEY, COHERE_API_KEY,
    RERANK_TOP_N, VECTOR_TOP_K, BM25_TOP_K,
)
from parser import SemanticParser
from cache_utils import load as cache_load, save as cache_save
from retrieval import (
    index_document, hybrid_search, list_collections,
    delete_collection, _col_name, SearchResult,
)
from llm import PaperTrailLLM, extract_citations, evaluate_retrieval

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperTrail · Industry RAG",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
# (Keeping your original CSS for the UI)
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg:       #07080A;
    --surface:  #0F1117;
    --surface2: #171A23;
    --border:   #1F2333;
    --border2:  #2A3050;
    --text:     #E2E8F0;
    --muted:    #4A5568;
    --muted2:   #718096;
    --accent:   #6366F1;
    --accent2:  #818CF8;
    --green:    #10B981;
    --amber:    #F59E0B;
    --red:      #EF4444;
    --blue:     #3B82F6;
  }

  .stApp { background: var(--bg); color: var(--text); font-family: 'Space Grotesk', sans-serif; }
  .pt-logo { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.6rem; color: var(--accent2); letter-spacing: -0.5px; }
  .pt-sub  { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--muted2); letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; }
  .key-ok   { background: rgba(16,185,129,0.12); color: var(--green); border: 1px solid rgba(16,185,129,0.3); border-radius: 4px; padding: 2px 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; }
  .key-bad  { background: rgba(239,68,68,0.12);  color: var(--red);   border: 1px solid rgba(239,68,68,0.3);  border-radius: 4px; padding: 2px 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; }
  .doc-card { background: var(--surface2); border: 1px solid var(--border2); border-left: 3px solid var(--accent); border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
  .doc-card.inactive { border-left-color: var(--border2); opacity: 0.6; }
  .msg-user { background: linear-gradient(135deg, #13162B, #1A1F3A); border: 1px solid #2D3561; border-radius: 12px 12px 2px 12px; padding: 14px 18px; margin: 10px 0 10px 80px; }
  .msg-assistant { background: var(--surface2); border: 1px solid var(--border2); border-radius: 2px 12px 12px 12px; padding: 14px 18px; margin: 10px 80px 10px 0; }
  .msg-role { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
  .msg-role.user { color: #6366F1; }
  .msg-role.assistant { color: var(--green); }
  .eval-excellent { color: var(--green); font-family: monospace; font-size: 0.75rem; }
  .eval-good { color: var(--blue); font-family: monospace; font-size: 0.75rem; }
  .eval-fair { color: var(--amber); font-family: monospace; font-size: 0.75rem; }
  .eval-poor { color: var(--red); font-family: monospace; font-size: 0.75rem; }
  .src-chunk { background: var(--surface); border-left: 2px solid var(--accent); border-radius: 0 4px 4px 0; padding: 8px 12px; margin: 4px 0; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }
  .src-header { color: var(--accent2); font-weight: 500; margin-bottom: 4px; }
  .src-text { color: var(--muted2); line-height: 1.6; }
  .badge { font-size: 0.62rem; padding: 1px 6px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; }
  .badge-vec { background: rgba(99,102,241,0.15); color: #818CF8; border: 1px solid rgba(99,102,241,0.3); }
  .badge-bm25 { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
  .badge-both { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
  .badge-type { background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
  .cite { background: rgba(99,102,241,0.15); color: var(--accent2); border: 1px solid rgba(99,102,241,0.3); border-radius: 4px; padding: 2px 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; margin: 2px; display: inline-block; }
  .stats-row { display: flex; gap: 16px; flex-wrap: wrap; border-top: 1px solid var(--border); padding-top: 8px; margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--muted2); }
  .stat-val { color: var(--accent2); }
  .pipe-step { display: flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--muted2); margin: 2px 0; }
  .pipe-dot-ok { width: 7px; height: 7px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
  .stButton > button { font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important; border-radius: 5px !important; }
  [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state & Helpers ────────────────────────────────────────────────────
def _init():
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("active_cols", [])
    st.session_state.setdefault("loaded_docs", {})
    st.session_state.setdefault("llm", None)
    st.session_state.setdefault("show_sources", True)
    st.session_state.setdefault("show_pipeline", True)
    st.session_state.setdefault("vector_k", VECTOR_TOP_K)
    st.session_state.setdefault("bm25_k", BM25_TOP_K)
    st.session_state.setdefault("rerank_n", RERANK_TOP_N)

_init()

def _get_llm() -> Optional[PaperTrailLLM]:
    if st.session_state.llm is None:
        try:
            st.session_state.llm = PaperTrailLLM()
        except Exception:
            return None
    return st.session_state.llm

def _elapsed(s: float) -> str:
    return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"

def _export() -> str:
    lines = ["# PaperTrail Chat Export\n"]
    for m in st.session_state.chat_history:
        role = "**You**" if m["role"] == "user" else "**PaperTrail**"
        lines.append(f"\n{role}\n\n{m['content']}\n")
    return "\n".join(lines)

def _key_badge(name: str, key: str) -> str:
    return f'<span class="key-ok">✓ {name}</span>' if key else f'<span class="key-bad">✗ {name} MISSING</span>'

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="pt-logo">PaperTrail</div><div class="pt-sub">Industry RAG · v3</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**API Keys**")
    # REMOVED ANTHROPIC BADGE
    st.markdown(
        _key_badge("Google", GOOGLE_API_KEY) + "&nbsp;&nbsp;" +
        _key_badge("Cohere", COHERE_API_KEY),
        unsafe_allow_html=True,
    )

    missing_keys = []
    if not GOOGLE_API_KEY: missing_keys.append(("GOOGLE_API_KEY", "Google Gemini"))
    if not COHERE_API_KEY: missing_keys.append(("COHERE_API_KEY", "Cohere"))

    for env_var, label in missing_keys:
        val = st.text_input(f"{label} API key", type="password", key=f"input_{env_var}")
        if val:
            os.environ[env_var] = val
            st.rerun()

    st.markdown("---")
    st.markdown("**Upload Documents**")
    uploaded = st.file_uploader("Drop PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded:
        parser = SemanticParser()
        for uf in uploaded:
            col = _col_name(uf.name)
            if col in st.session_state.loaded_docs: continue
            pdf_path = UPLOAD_DIR / uf.name
            pdf_path.write_bytes(uf.getbuffer())
            with st.status(f"Processing {uf.name}…", expanded=True) as status:
                t0 = time.time()
                chunks = cache_load(pdf_path)
                if chunks is None:
                    chunks = parser.parse(pdf_path, lambda c, t, m: None)
                    cache_save(pdf_path, chunks)
                index_document(chunks, uf.name, lambda c, t, m: None)
                st.session_state.loaded_docs[col] = {"filename": uf.name, "chunks": len(chunks), "pages": max((c.page for c in chunks), default=0), "elapsed": time.time()-t0}
                if col not in st.session_state.active_cols: st.session_state.active_cols.append(col)
                status.update(label=f"✅ {uf.name}", state="complete")

# ── Main area ──────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([5, 2])
with hcol1:
    st.markdown('<div class="pt-logo" style="font-size:2rem">PaperTrail</div>', unsafe_allow_html=True)
with hcol2:
    if st.session_state.show_pipeline and st.session_state.active_cols:
        st.markdown("""<div style="text-align:right"><div class="pipe-step" style="justify-content:flex-end"><span>Gemini AI Pipeline</span><span class="pipe-dot-ok"></span></div></div>""", unsafe_allow_html=True)

st.markdown("---")

# Welcome Screen (Update label from Claude to Gemini)
if not st.session_state.loaded_docs:
    st.markdown('<div style="text-align:center;padding:80px 20px"><div style="font-size:2.8rem;font-weight:700;color:#6366F1;">Ask your documents.</div><div style="color:#4A5568;">Powered by Gemini 2.0 Flash</div></div>', unsafe_allow_html=True)
    st.stop()

# Chat display logic
for msg in st.session_state.chat_history:
    # (Same as your original display logic...)
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user"><div class="msg-role user">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="msg-assistant"><div class="msg-role assistant">PaperTrail</div>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

# Input area
st.markdown("---")
query = st.text_area("Question", key="query_input", height=88, label_visibility="collapsed", placeholder="Ask anything about your documents…")
send = st.button("Send →", type="primary")

if send and query:
    llm = _get_llm()
    if not llm:
        st.error("Google API key missing in .env")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query.strip()})
        st.rerun()

# Execution Logic
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    user_text = st.session_state.chat_history[-1]["content"]
    llm = _get_llm()
    if llm:
        rewritten = llm.rewrite_query(user_text)
        results = hybrid_search(rewritten, st.session_state.active_cols, st.session_state.vector_k, st.session_state.bm25_k, st.session_state.rerank_n)
        
        full_resp = ""
        placeholder = st.empty()
        for fragment in llm.stream_answer(user_text, results):
            full_resp += fragment
            placeholder.markdown(full_resp + "▌")
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_resp})
        st.rerun()