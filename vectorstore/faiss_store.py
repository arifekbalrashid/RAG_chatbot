"""
FAISS Vector Store Module (HF Spaces Edition).

In-memory only — no persistence to disk. Each session starts fresh.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from config import settings
from embeddings.embedding_model import get_embedding_model
from utils.helpers import timed


class FAISSStore:
    """In-memory FAISS vector store."""

    def __init__(self):
        self._embeddings = get_embedding_model()
        self._store: Optional[FAISS] = None

    # Build / extend

    @timed
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the FAISS index.
        If an index already exists, merge into it; otherwise create a new one.
        """
        if not documents:
            logger.warning("No documents to add")
            return

        new_store = FAISS.from_documents(documents, self._embeddings)

        if self._store is not None:
            self._store.merge_from(new_store)
        else:
            self._store = new_store

        logger.info(f"Added {len(documents)} documents to FAISS index")

    # Query

    @timed
    def similarity_search(self, query: str, k: int | None = None) -> List[Document]:
        """Return the top-k most similar documents."""
        if self._store is None:
            logger.warning("FAISS index is empty — no results")
            return []
        k = k or settings.top_k_retrieval
        return self._store.similarity_search(query, k=k)

    @timed
    def similarity_search_with_score(
        self, query: str, k: int | None = None
    ) -> List[tuple[Document, float]]:
        """Return top-k documents with distance scores."""
        if self._store is None:
            return []
        k = k or settings.top_k_retrieval
        return self._store.similarity_search_with_score(query, k=k)

    # Reset

    def clear(self) -> None:
        """Clear the in-memory index."""
        self._store = None
        logger.info("FAISS index cleared")

    # Utility

    @property
    def is_ready(self) -> bool:
        """Return True if the index has been loaded or populated."""
        return self._store is not None

    @property
    def store(self) -> Optional[FAISS]:
        return self._store
