"""
PaperTrail — LLM Interface
============================
Features:
  - Query rewriting      : Claude rewrites vague/conversational queries into
                           precise search queries before retrieval
  - Streaming generation : token-by-token streaming via Anthropic SDK
  - Conversation memory  : sliding window of last N turns
  - Citation extraction  : parses [Source: file, p.N] from responses
  - Retrieval evaluation : scores quality of retrieved chunks per query
"""

from __future__ import annotations

import logging
import re
from typing import Generator, List, Optional
import google.generativeai as genai
import anthropic

from config import (
    GOOGLE_API_KEY, GEMINI_LLM_MODEL, MAX_TOKENS,
    MAX_HISTORY_TURNS, ENABLE_QUERY_REWRITE,
)
from retrieval import SearchResult

logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)
# ── Prompts ────────────────────────────────────────────────────────────────────

# --- Prompts ---
SYSTEM_PROMPT = """You are PaperTrail, an expert assistant for analysing technical documents.
Core rules:
1. Answer ONLY using information from the provided context chunks.
2. After every factual claim, add a citation: [Source: filename, p.N]
3. If the context lacks sufficient information, clearly state: "The documents do not contain enough information to answer this."
4. Detect the language of the user's question and reply in the same language.
5. Use markdown formatting: headers, bullet points, and tables where appropriate.
"""

QUERY_REWRITE_PROMPT = """Given a conversation history and the user's latest message, rewrite it into an optimal search query. Output ONLY the rewritten query.
History: {history}
User's message: {query}"""


# ── Data types ─────────────────────────────────────────────────────────────────

class Turn:
    def __init__(self, role: str, content: str):
        self.role    = role
        self.content = content


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_retrieval(results: List[SearchResult]) -> dict:
    if not results: return {"status": "no_results", "mean_score": 0.0}
    scores = [r.relevance_score for r in results]
    mean = sum(scores) / len(scores)
    status = "excellent" if mean >= 0.7 else "good" if mean >= 0.45 else "fair" if mean >= 0.25 else "poor"
    return {"status": status, "mean_score": mean, "from_both": sum(1 for r in results if r.vector_rank is not None and r.bm25_rank is not None)}



# ── Main LLM class ─────────────────────────────────────────────────────────────

class PaperTrailLLM:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_LLM_MODEL,
            system_instruction=SYSTEM_PROMPT
        )
        self._chat = self.model.start_chat(history=[])
    # ── Query rewriting ────────────────────────────────────────────────────────

    def rewrite_query(self, query: str) -> str:
        if not ENABLE_QUERY_REWRITE or not self._chat.history:
            return query
        
        history_str = "\n".join([f"{m.role}: {m.parts[0].text[:200]}" for m in self._chat.history[-4:]])
        prompt = QUERY_REWRITE_PROMPT.format(history=history_str, query=query)
        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip() or query
        except Exception as e:
            logger.warning(f"Rewrite failed: {e}")
            return query
    # ── Streaming answer ───────────────────────────────────────────────────────

    def stream_answer(self, query: str, results: List[SearchResult]) -> Generator[str, None, None]:
        context = self._build_context(results)
        user_msg = f"{context}\n\n---\nQuestion: {query}"
        
        try:
            # Gemini streaming
            response = self._chat.send_message(user_msg, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            yield f"\n\n⚠️ Gemini Error: {exc}"

    # ── Utilities ──────────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        self._chat = self.model.start_chat(history=[])

    def _build_context(self, results: List[SearchResult]) -> str:
        if not results: return "No relevant context found."
        lines = ["### Retrieved Context\n"]
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source_file", "unknown")
            page = r.metadata.get("page", "?")
            lines.append(f"[Chunk {i} | {source}, p.{page}]\n{r.text}\n")
        return "\n".join(lines)

    def _build_messages(self, user_message: str) -> list:
        msgs = []
        for t in self._history[-(MAX_HISTORY_TURNS * 2):]:
            msgs.append({"role": t.role, "content": t.content})
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def _trim_history(self) -> None:
        cap = MAX_HISTORY_TURNS * 2
        if len(self._history) > cap:
            self._history = self._history[-cap:]


# ── Citation extraction ────────────────────────────────────────────────────────

def extract_citations(text: str) -> List[dict]:
    pattern = r'\[Source:\s*([^,\]]+),\s*p\.(\d+)\]'
    matches = re.findall(pattern, text)
    seen, out = set(), []
    for file, page in matches:
        if (file, page) not in seen:
            seen.add((file, page))
            out.append({"file": file.strip(), "page": int(page)})
    return out
