"""
Embedding Model Module.

Uses HuggingFace Endpoint Embeddings for zero-memory embeddings.
No local PyTorch or model downloads required.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFace Endpoint embedding instance.

    Uses the HF Inference API via the new router endpoint — no local model,
    no PyTorch, no memory overhead. Requires HF_TOKEN in environment.
    """
    logger.info(f"Initialising HF Endpoint embeddings: {settings.embedding_model}")
    model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
    )
    logger.info("Embedding model ready (API-based, zero local memory)")
    return model
