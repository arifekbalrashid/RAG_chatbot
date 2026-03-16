"""
Reciprocal Rank Fusion (RRF) Module.

Combines ranked result lists from multiple retrievers into a single
fused ranking using the RRF algorithm.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document
from loguru import logger

from utils.helpers import timed

# Standard RRF constant
RRF_K = 60


@timed
def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:
    """
    Fuse multiple ranked document lists using Reciprocal Rank Fusion.

    For each document, RRF score = Σ  1 / (k + rank_i)  across all lists
    where rank_i is the 1-based position in list i.

    Args:
        result_lists: A list of ranked Document lists (one per retriever).
        k: The RRF constant (default 60).

    Returns:
        A single list of Documents sorted by fused score (descending).
    """
    # Use page_content fingerprint as the document key
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = doc.page_content[:200]  # fingerprint
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    # Sort by descending RRF score
    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    fused = [doc_map[k_] for k_ in sorted_keys]

    logger.info(
        f"RRF fused {sum(len(r) for r in result_lists)} docs "
        f"from {len(result_lists)} retrievers → {len(fused)} unique docs"
    )
    return fused
