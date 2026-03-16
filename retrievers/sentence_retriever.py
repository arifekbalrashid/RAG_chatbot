"""
Sentence Window Retriever.

For each top-k hit from the vector store, this retriever expands the
context window by including *neighbouring* chunks from the same source
and page. This helps the LLM see the information around the best-match
sentence.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document
from loguru import logger

from config import settings
from utils.helpers import timed


class SentenceWindowRetriever:
    """
    Retriever that augments vector-search results with surrounding context.

    It keeps an internal chunk registry built during ingestion so that it
    can look up neighbours by (source, page/row) key.
    """

    def __init__(self, faiss_store, window_size: int = 1):
        """
        Args:
            faiss_store: FAISSStore instance.
            window_size: Number of neighbouring chunks on each side to include.
        """
        self._store = faiss_store
        self._window_size = window_size

        # Registry: key → list of chunks in order
        # key = (source, page_or_row)
        self._registry: Dict[Tuple[str, int | str], List[Document]] = {}

    # Registry management

    def register_documents(self, documents: List[Document]) -> None:
        """Index documents by (source, page) so we can look up neighbours."""
        for doc in documents:
            meta = doc.metadata
            key = (meta.get("source", "unknown"), meta.get("page") or meta.get("row_index") or 0)
            self._registry.setdefault(key, []).append(doc)
        logger.debug(f"SentenceWindowRetriever registry now has {len(self._registry)} keys")

    def clear_registry(self) -> None:
        """Clear the internal chunk registry."""
        self._registry.clear()
        logger.debug("SentenceWindowRetriever registry cleared")

    # Retrieval

    @timed
    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        """
        Retrieve documents with expanded sentence windows.

        Steps:
          1. Get top-k from FAISS.
          2. For each result, find its position in the registry.
          3. Include surrounding chunks within the window.
          4. Deduplicate and return.
        """
        top_k = top_k or settings.top_k_retrieval
        if not self._store.is_ready:
            logger.warning("Vector store not ready")
            return []

        initial = self._store.similarity_search(query, k=top_k)

        expanded: List[Document] = []
        seen_contents: set[str] = set()

        for doc in initial:
            meta = doc.metadata
            key = (meta.get("source", "unknown"), meta.get("page") or meta.get("row_index") or 0)
            siblings = self._registry.get(key, [])

            if not siblings:
                # No registry info — just keep the original doc
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    expanded.append(doc)
                continue

            # Find index of current doc in the sibling list
            idx = None
            for i, s in enumerate(siblings):
                if s.page_content == doc.page_content:
                    idx = i
                    break

            if idx is None:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    expanded.append(doc)
                continue

            # Collect window
            start = max(0, idx - self._window_size)
            end = min(len(siblings), idx + self._window_size + 1)

            for sibling in siblings[start:end]:
                if sibling.page_content not in seen_contents:
                    seen_contents.add(sibling.page_content)
                    expanded.append(sibling)

        logger.info(
            f"SentenceWindowRetriever: {len(initial)} hits → {len(expanded)} expanded docs"
        )
        return expanded
