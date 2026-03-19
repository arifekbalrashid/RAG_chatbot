"""
Lightweight Re-Ranking Module.

Returns the top-k documents by their existing order (from fusion).
No cross-encoder model loaded — saves ~90 MB RAM.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from loguru import logger

from config import settings
from utils.helpers import timed


@timed
def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int | None = None,
) -> List[Document]:
    """
    Select top-k documents from the fused results.

    Since the fusion step (RRF) already ranks by relevance,
    we simply take the top-k without a cross-encoder.

    Args:
        query: The user query (unused, kept for interface compatibility).
        documents: Candidate documents (already ranked by RRF fusion).
        top_k: Number of top documents to return.

    Returns:
        Top-k documents from the fused list.
    """
    top_k = top_k or settings.top_k_rerank

    if not documents:
        return []

    top_docs = documents[:top_k]

    logger.info(
        f"Selected top {len(top_docs)} from {len(documents)} docs (lightweight mode)"
    )
    return top_docs