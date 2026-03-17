"""
RAG System — Centralised Configuration.
"""

from __future__ import annotations
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)


class Settings(BaseSettings):
    """Application-wide settings backed by environment variables."""

    # LLM API Keys
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    hf_token: str = Field(default="", env="HF_TOKEN")

    # Model names
    embedding_model: str = Field(default="BAAI/bge-small-en", env="EMBEDDING_MODEL")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL")
    groq_model: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL")

    # Chunking
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")

    # Retrieval
    top_k_retrieval: int = Field(default=10, env="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, env="TOP_K_RERANK")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton instance
settings = Settings()