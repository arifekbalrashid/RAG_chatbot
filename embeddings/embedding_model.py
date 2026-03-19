"""
Embedding Model Module.

Uses HuggingFace Inference API for zero-memory embeddings.
No local PyTorch or model downloads required.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

warnings.filterwarnings("ignore", message=".*HuggingFaceInferenceAPIEmbeddings.*")

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from loguru import logger

from config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceInferenceAPIEmbeddings:
    """
    Return a cached HuggingFace Inference API embedding instance.

    Uses the free HF Inference API — no local model, no PyTorch,
    no memory overhead. Requires HF_TOKEN in environment.
    """
    logger.info(f"Initialising HF Inference API embeddings: {settings.embedding_model}")
    model = HuggingFaceInferenceAPIEmbeddings(
        api_key=settings.hf_token,
        model_name=settings.embedding_model,
    )
    logger.info("Embedding model ready (API-based, zero local memory)")
    return model
