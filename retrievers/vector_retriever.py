"""
Vector Retriever.

Performs plain FAISS similarity search and returns ranked Document objects.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from loguru import logger

from config import settings
from utils.helpers import timed


class VectorRetriever:
    """Retrieve documents using FAISS vector similarity search."""

    def __init__(self, faiss_store):
        """
        Args:
            faiss_store: An instance of FAISSStore.
        """
        self._store = faiss_store

    @timed
    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        """
        Retrieve top-k documents by semantic similarity.

        Returns:
            List of Document objects ordered by relevance.
        """
        top_k = top_k or settings.top_k_retrieval
        if not self._store.is_ready:
            logger.warning("Vector store is not ready — returning empty results")
            return []

        results = self._store.similarity_search(query, k=top_k)
        logger.info(f"VectorRetriever returned {len(results)} documents")
        return results