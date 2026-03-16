"""
Utility helpers used across the Enterprise RAG system.
"""

from __future__ import annotations

import hashlib
import re
import time
from functools import wraps
from typing import Any, Callable, List, Dict

from loguru import logger


# Text cleaning

def clean_text(text: str) -> str:
    """Remove excessive whitespace, control characters, and normalise line breaks."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def generate_doc_id(content: str, source: str) -> str:
    """Create a deterministic document ID from content + source."""
    raw = f"{source}::{content[:200]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# Timing decorator

def timed(func: Callable) -> Callable:
    """Decorator that logs the execution time of a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


# Metadata helpers

def build_metadata(
    source: str,
    source_type: str,
    page: int | None = None,
    url: str | None = None,
    row_index: int | None = None,
    extra: dict | None = None,
) -> Dict[str, Any]:
    """Build a standardised metadata dict attached to each chunk."""
    meta: Dict[str, Any] = {
        "source": source,
        "source_type": source_type,
    }
    if page is not None:
        meta["page"] = page
    if url is not None:
        meta["url"] = url
    if row_index is not None:
        meta["row_index"] = row_index
    if extra:
        meta.update(extra)
    return meta


def format_sources(documents: List[Dict[str, Any]]) -> str:
    """Format a list of document metadata dicts into a readable source citation block."""
    lines: list[str] = []
    seen: set[str] = set()
    for i, doc in enumerate(documents, 1):
        meta = doc.get("metadata", doc)
        source = meta.get("source", "Unknown")
        page = meta.get("page")
        url = meta.get("url")

        if page is not None:
            citation = f"{source} — page {page}"
        elif url is not None:
            citation = url
        else:
            citation = source

        if citation not in seen:
            seen.add(citation)
            lines.append(f"[{len(lines) + 1}] {citation}")

    return "\n".join(lines)