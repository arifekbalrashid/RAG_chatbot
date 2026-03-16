"""
PDF Document Ingestion Module.

Extracts text from PDF files using pdfplumber (with pypdf fallback),
cleans the output, splits into chunks, and attaches page-level metadata.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from config import settings
from utils.helpers import clean_text, build_metadata, timed


@timed
def load_pdf(file_path: str | Path | None = None, file_bytes: bytes | None = None, filename: str = "upload.pdf") -> List[Document]:
    """
    Ingest a PDF and return a list of LangChain *Document* objects.

    Args:
        file_path: Path to a PDF on disk (mutually exclusive with *file_bytes*).
        file_bytes: Raw PDF bytes (e.g. from an upload widget).
        filename: Human-readable source name for metadata.

    Returns:
        List of chunked Document objects with metadata.
    """
    if file_path is None and file_bytes is None:
        raise ValueError("Provide either file_path or file_bytes.")

    raw_pages: list[dict] = []

    try:
        source = file_path if file_bytes is None else io.BytesIO(file_bytes)
        with pdfplumber.open(source) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    raw_pages.append({"text": clean_text(text), "page": page_num})
    except Exception as exc:
        logger.warning(f"pdfplumber failed ({exc}); falling back to pypdf")
        raw_pages = _fallback_pypdf(file_path, file_bytes)

    if not raw_pages:
        logger.warning(f"No text extracted from {filename}")
        return []

    # ── Chunk each page ───────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: List[Document] = []
    for page_info in raw_pages:
        chunks = splitter.split_text(page_info["text"])
        for chunk in chunks:
            meta = build_metadata(
                source=filename,
                source_type="pdf",
                page=page_info["page"],
            )
            documents.append(Document(page_content=chunk, metadata=meta))

    logger.info(f"PDF '{filename}' → {len(documents)} chunks from {len(raw_pages)} pages")
    return documents


# ── Fallback extractor ────────────────────────────────────────────────────────

def _fallback_pypdf(file_path: str | Path | None, file_bytes: bytes | None) -> list[dict]:
    """Use pypdf as a backup extractor."""
    from pypdf import PdfReader

    if file_bytes is not None:
        reader = PdfReader(io.BytesIO(file_bytes))
    else:
        reader = PdfReader(str(file_path))

    pages: list[dict] = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": clean_text(text), "page": i})
    return pages
