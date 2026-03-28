"""
PaperTrail — Semantic PDF Parser
==================================
Semantic-aware chunking that respects document structure:
  - Detects headings by font size / boldness
  - Groups text into logical sections before chunking
  - Chunks split at paragraph boundaries, not character counts
  - Tables extracted as structured text with headers
  - OCR fallback for scanned / image-only pages
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import fitz  # PyMuPDF ≥ 1.23

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from config import (
    CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS,
    OCR_MIN_CHARS, TESSERACT_LANG,
)

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text        : str
    source_file : str
    page        : int
    chunk_index : int
    element_type: str          # text | heading | table | ocr
    section     : str = ""     # nearest heading above
    bbox        : List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text"        : self.text,
            "source_file" : self.source_file,
            "page"        : self.page,
            "chunk_index" : self.chunk_index,
            "element_type": self.element_type,
            "section"     : self.section,
        }


# ── Heading detection ──────────────────────────────────────────────────────────

def _median_font_size(blocks: list) -> float:
    sizes = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                s = span.get("size", 0)
                if s > 0:
                    sizes.append(s)
    if not sizes:
        return 11.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _is_heading(span: dict, median_size: float) -> bool:
    size  = span.get("size", 0)
    flags = span.get("flags", 0)
    bold  = bool(flags & (1 << 4))
    return (size >= median_size * 1.25) or (bold and size >= median_size * 1.05)


# ── Semantic chunking ──────────────────────────────────────────────────────────

def _split_semantic(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Split text preferring paragraph breaks, then sentence breaks,
    then word breaks. Never splits mid-word.
    """
    if len(text) <= max_chars:
        return [text] if len(text) >= CHUNK_MIN_CHARS else []

    chunks = []
    start  = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            # Try paragraph break first
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + max_chars // 3:
                end = para_break + 2
            else:
                # Try sentence break
                sent_break = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind(".\n", start, end),
                )
                if sent_break > start + max_chars // 4:
                    end = sent_break + 1
                else:
                    # Word break
                    word_break = text.rfind(" ", start, end)
                    if word_break > start:
                        end = word_break

        chunk = text[start:end].strip()
        if len(chunk) >= CHUNK_MIN_CHARS:
            chunks.append(chunk)

        # Advance with overlap
        start = max(start + 1, end - overlap)

    return chunks


# ── Table formatting ───────────────────────────────────────────────────────────

def _table_to_text(table) -> str:
    rows = table.extract()
    if not rows:
        return ""
    # Use first row as header if it looks like one
    lines = []
    for i, row in enumerate(rows):
        cells = [str(c).strip().replace("\n", " ") if c else "" for c in row]
        lines.append(" | ".join(cells))
        if i == 0:
            lines.append("-" * max(len(l) for l in lines))
    return "\n".join(lines)


# ── OCR ────────────────────────────────────────────────────────────────────────

def _ocr_bytes(img_bytes: bytes) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(img, lang=TESSERACT_LANG).strip()
    except Exception as exc:
        logger.debug("OCR error: %s", exc)
        return ""


# ── Main parser ────────────────────────────────────────────────────────────────

class SemanticParser:
    """
    Parse a PDF into semantically chunked Chunk objects.

    Improvements over naive character-based chunking:
      1. Font-size analysis to detect headings per page
      2. Section accumulation — text under the same heading is grouped
         before being split, so chunks share context
      3. Split preference: paragraph > sentence > word boundary
      4. Native table extraction (PyMuPDF ≥ 1.23)
      5. Per-image OCR + full-page OCR fallback
    """

    def parse(
        self,
        pdf_path  : str | Path,
        on_progress: Optional[ProgressFn] = None,
    ) -> List[Chunk]:

        pdf_path = Path(pdf_path)
        source   = pdf_path.name
        all_chunks: List[Chunk] = []
        chunk_idx = 0

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error("Cannot open %s: %s", pdf_path, exc)
            return []

        total = len(doc)
        logger.info("Parsing '%s' (%d pages)", source, total)

        for page_num in range(total):
            page    = doc.load_page(page_num)
            page_no = page_num + 1

            if on_progress:
                on_progress(page_num + 1, total, f"Parsing page {page_no}/{total}")

            raw        = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
            blocks     = raw.get("blocks", [])
            med_size   = _median_font_size(blocks)

            current_section = ""
            section_buffer  = ""     # accumulate text under current heading
            page_char_count = 0

            def _flush_buffer(buf: str, etype: str) -> int:
                nonlocal chunk_idx
                added = 0
                for frag in _split_semantic(buf, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
                    all_chunks.append(Chunk(
                        text=frag, source_file=source, page=page_no,
                        chunk_index=chunk_idx, element_type=etype,
                        section=current_section,
                    ))
                    chunk_idx += 1
                    added     += 1
                return added

            # ── Text blocks ────────────────────────────────────────────────
            for block in blocks:
                if block.get("type") != 0:
                    continue

                block_text = ""
                block_is_heading = False

                for line in block.get("lines", []):
                    line_txt = ""
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if not t:
                            continue
                        if _is_heading(span, med_size):
                            block_is_heading = True
                        line_txt += t + " "
                    stripped = line_txt.strip()
                    if stripped:
                        block_text += stripped + "\n"

                block_text = block_text.strip()
                if not block_text:
                    continue

                page_char_count += len(block_text)

                if block_is_heading and len(block_text) < 150:
                    # Flush buffer under previous heading
                    if section_buffer.strip():
                        _flush_buffer(section_buffer.strip(), "text")
                        section_buffer = ""
                    current_section = block_text.strip()
                    # Headings become their own tiny chunk for searchability
                    all_chunks.append(Chunk(
                        text=block_text, source_file=source, page=page_no,
                        chunk_index=chunk_idx, element_type="heading",
                        section=current_section,
                    ))
                    chunk_idx += 1
                else:
                    section_buffer += block_text + "\n\n"

            # Flush remaining buffer
            if section_buffer.strip():
                _flush_buffer(section_buffer.strip(), "text")

            # ── Tables ────────────────────────────────────────────────────
            try:
                for tbl in page.find_tables().tables:
                    tbl_text = _table_to_text(tbl)
                    if not tbl_text.strip():
                        continue
                    full = f"[TABLE — page {page_no}]\n{tbl_text}"
                    for frag in _split_semantic(full, CHUNK_MAX_CHARS * 2, 0):
                        all_chunks.append(Chunk(
                            text=frag, source_file=source, page=page_no,
                            chunk_index=chunk_idx, element_type="table",
                            section=current_section,
                        ))
                        chunk_idx += 1
            except Exception:
                pass   # PyMuPDF < 1.23

            # ── Per-image OCR ─────────────────────────────────────────────
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    base   = doc.extract_image(xref)
                    ocr_tx = _ocr_bytes(base["image"])
                    if ocr_tx:
                        for frag in _split_semantic(ocr_tx, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
                            all_chunks.append(Chunk(
                                text=frag, source_file=source, page=page_no,
                                chunk_index=chunk_idx, element_type="ocr",
                                section=current_section,
                            ))
                            chunk_idx += 1
                except Exception:
                    pass

            # ── Full-page OCR fallback ────────────────────────────────────
            if page_char_count < OCR_MIN_CHARS and OCR_AVAILABLE:
                logger.info("Page %d: fallback OCR (only %d chars)", page_no, page_char_count)
                pix    = page.get_pixmap(dpi=200)
                ocr_tx = _ocr_bytes(pix.tobytes("png"))
                if ocr_tx:
                    for frag in _split_semantic(ocr_tx, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
                        all_chunks.append(Chunk(
                            text=frag, source_file=source, page=page_no,
                            chunk_index=chunk_idx, element_type="ocr",
                            section=current_section,
                        ))
                        chunk_idx += 1

        doc.close()

        if on_progress:
            on_progress(total, total, "Parsing complete")

        logger.info("Parsed %d semantic chunks from '%s'", len(all_chunks), source)
        return all_chunks
