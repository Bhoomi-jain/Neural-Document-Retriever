"""
PaperTrail — Cache Utilities
=============================
Hash-based cache so re-uploading the same file skips parsing.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Optional

from config import CACHE_DIR
from parser import Chunk

logger = logging.getLogger(__name__)


def _hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()[:16]


def _cache_path(pdf_path: Path) -> Path:
    return CACHE_DIR / f"{pdf_path.stem}_{_hash(pdf_path)}.pkl"


def load(pdf_path: Path) -> Optional[List[Chunk]]:
    cp = _cache_path(pdf_path)
    if not cp.exists():
        return None
    try:
        with open(cp, "rb") as f:
            data = pickle.load(f)
        logger.info("Cache hit: %s (%d chunks)", pdf_path.name, len(data))
        return data
    except Exception as exc:
        logger.warning("Corrupt cache for %s: %s", pdf_path.name, exc)
        cp.unlink(missing_ok=True)
        return None


def save(pdf_path: Path, chunks: List[Chunk]) -> None:
    cp = _cache_path(pdf_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump(chunks, f)
    logger.info("Cached %d chunks for %s", len(chunks), pdf_path.name)
