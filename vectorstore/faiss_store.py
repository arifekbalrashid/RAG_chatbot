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

    BATCH_SIZE = 10  # HF free Inference API can't handle large batches

    @timed
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the FAISS index in small batches.
        Batching avoids overwhelming the HF Inference API rate limits.
        """
        if not documents:
            logger.warning("No documents to add")
            return

        import time

        total = len(documents)
        added = 0

        for i in range(0, total, self.BATCH_SIZE):
            batch = documents[i : i + self.BATCH_SIZE]
            batch_num = i // self.BATCH_SIZE + 1
            total_batches = (total + self.BATCH_SIZE - 1) // self.BATCH_SIZE

            for attempt in range(3):  # up to 3 retries per batch
                try:
                    logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} docs)...")
                    new_store = FAISS.from_documents(batch, self._embeddings)

                    if self._store is not None:
                        self._store.merge_from(new_store)
                    else:
                        self._store = new_store

                    added += len(batch)
                    break  # success, move to next batch
                except Exception as exc:
                    logger.warning(f"Batch {batch_num} attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)  # backoff: 1s, 2s
                    else:
                        raise RuntimeError(
                            f"Failed to embed batch {batch_num} after 3 attempts: {exc}"
                        ) from exc

            # Small delay between batches to respect API rate limits
            if i + self.BATCH_SIZE < total:
                time.sleep(0.5)

        logger.info(f"Added {added}/{total} documents to FAISS index")

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
