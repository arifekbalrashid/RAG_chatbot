"""
Embedding Model Module.

Provides a singleton HuggingFace embedding model for the entire application.
Uses BAAI/bge-small-en by default (configurable via env vars).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFace embedding model instance.

    The model is downloaded on first use and then reused across
    all ingestion and retrieval calls.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
    logger.info("Embedding model loaded successfully")
    return model
