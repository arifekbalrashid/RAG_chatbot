"""
Groq LLM Client — Primary LLM provider.

Groq provides blazing-fast inference via an OpenAI-compatible API.
Supported models: llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from config import settings
from utils.helpers import timed

RAG_PROMPT_TEMPLATE = """You are an enterprise knowledge assistant.
Answer the user's question using ONLY the provided context.
Be concise — respond in 2-4 sentences maximum. Include source citations where applicable.

If the context does not contain enough information to answer the question, say so explicitly.

## Context:
{context}

## Question:
{question}

## Answer (keep it concise):"""


class GroqClient:
    """Wrapper around Groq's OpenAI-compatible API for fast inference."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self):
        self._model_name = settings.groq_model
        self._api_key = settings.groq_api_key
        self._client = None

    def _ensure_client(self):
        """Lazy-init the OpenAI-compatible client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.BASE_URL,
            )
            logger.info(f"Groq client initialised: {self._model_name}")

    @timed
    def generate(
        self,
        question: str,
        context: str,
        prompt_template: str | None = None,
    ) -> str:
        """
        Generate an answer using Groq.

        Args:
            question: The user query.
            context: Retrieved context.
            prompt_template: Optional custom template.

        Returns:
            Generated answer string.

        Raises:
            RuntimeError: If the API call fails.
        """
        self._ensure_client()

        template = prompt_template or RAG_PROMPT_TEMPLATE
        prompt = template.format(context=context, question=question)

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "You are a concise enterprise knowledge assistant. Keep answers brief and to the point."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Groq generated {len(answer)} chars")
            return answer
        except Exception as exc:
            logger.error(f"Groq generation failed: {exc}")
            raise RuntimeError(f"Groq API error: {exc}") from exc

    @property
    def is_available(self) -> bool:
        """Check if the API key is configured."""
        return bool(self._api_key)