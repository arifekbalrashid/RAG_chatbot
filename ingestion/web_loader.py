"""
Web Page Ingestion Module.

Fetches a URL, strips boilerplate with BeautifulSoup, cleans the text,
and splits it into chunked LangChain Documents.
"""

from __future__ import annotations

from typing import List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from config import settings
from utils.helpers import clean_text, build_metadata, timed

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_REMOVE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript"}


@timed
def load_webpage(url: str, timeout: int = 15) -> List[Document]:
    """
    Fetch a web page and return chunked Document objects.

    Args:
        url: The URL to scrape.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of chunked Document objects with metadata.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        response = requests.get(url, headers=_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Failed to fetch {url}: {exc}")
        return []

    soup = BeautifulSoup(response.text, "lxml")

    # Remove unwanted elements
    for tag in soup.find_all(_REMOVE_TAGS):
        tag.decompose()

    # Extract text from main content areas first, fall back to body
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main is None:
        logger.warning(f"No parseable body in {url}")
        return []

    text = clean_text(main.get_text(separator="\n"))
    if not text:
        logger.warning(f"No text content extracted from {url}")
        return []

    # Get page title
    title = soup.title.string.strip() if soup.title and soup.title.string else urlparse(url).netloc

    # ── Chunk ─────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)
    documents: List[Document] = []
    for chunk in chunks:
        meta = build_metadata(
            source=title,
            source_type="web",
            url=url,
        )
        documents.append(Document(page_content=chunk, metadata=meta))

    logger.info(f"Web '{url}' → {len(documents)} chunks")
    return documents
