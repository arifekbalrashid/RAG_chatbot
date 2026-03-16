"""
Cross-Encoder Re-Ranking Module.

Re-ranks a list of candidate documents based on query-document relevance
scores computed by a cross-encoder model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import CrossEncoder

from config import settings
from utils.helpers import timed


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder model (cached)."""
    logger.info(f"Loading re-ranker model: {settings.reranker_model}")
    model = CrossEncoder(settings.reranker_model, max_length=512)
    logger.info("Re-ranker model loaded")
    return model


@timed
def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int | None = None,
) -> List[Document]:
    """
    Re-rank documents using a cross-encoder model.

    Args:
        query: The user query.
        documents: Candidate documents to re-rank.
        top_k: Number of top documents to return after re-ranking.

    Returns:
        Re-ranked list of Document objects (highest relevance first).
    """
    top_k = top_k or settings.top_k_rerank

    if not documents:
        return []

    model = _get_reranker()

    # Build query–document pairs
    pairs: List[Tuple[str, str]] = [
        (query, doc.page_content) for doc in documents
    ]

    # Score all pairs
    scores = model.predict(pairs)

    # Pair scores with docs and sort
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    top_docs = [doc for _, doc in scored_docs[:top_k]]

    logger.info(
        f"Re-ranked {len(documents)} docs → top {len(top_docs)} "
        f"(scores {scored_docs[0][0]:.3f} … {scored_docs[-1][0]:.3f})"
    )
    return top_docs