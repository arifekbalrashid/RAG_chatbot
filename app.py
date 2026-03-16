"""
Enterprise RAG System — Streamlit App

Self-contained Streamlit application for document Q&A using RAG.
Users can upload PDFs or add website URLs and ask questions about the content.
Answers are generated using retrieval, reranking, and an LLM.

Usage:
  - Upload a PDF or add a website URL
  - Ask questions about the uploaded content
  - Get AI-powered answers with source citations
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from loguru import logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from ingestion.pdf_loader import load_pdf
from ingestion.web_loader import load_webpage
from langgraph_pipeline.rag_graph import RAGPipeline
from vectorstore.faiss_store import FAISSStore


# Page Config

st.set_page_config(
    page_title="Enterprise RAG — AI Document Q&A",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look

st.markdown("""
<style>
    /* ── Import Google Font ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root variables ─────────────────────────────────── */
    :root {
        --primary: #6C5CE7;
        --primary-light: #A29BFE;
        --accent: #00CEC9;
        --accent-warm: #FD79A8;
        --bg-dark: #0F0F1A;
        --bg-card: #1A1A2E;
        --bg-card-hover: #232342;
        --text-primary: #E8E8F0;
        --text-secondary: #9D9DB5;
        --success: #00B894;
        --warning: #FDCB6E;
        --error: #FF6B6B;
        --border: rgba(108, 92, 231, 0.2);
    }

    /* ── Global ─────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ── Header ─────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        background: linear-gradient(135deg, #6C5CE7, #A29BFE, #00CEC9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 300;
    }

    /* ── Cards ──────────────────────────────────────────── */
    .stCard, .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .result-card:hover {
        border-color: var(--primary-light);
        box-shadow: 0 8px 32px rgba(108, 92, 231, 0.15);
    }

    /* ── Answer box ─────────────────────────────────────── */
    .answer-box {
        background: linear-gradient(135deg, rgba(108, 92, 231, 0.08), rgba(0, 206, 201, 0.08));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        line-height: 1.7;
        font-size: 1rem;
    }

    /* ── Source citations ───────────────────────────────── */
    .source-box {
        background: rgba(0, 206, 201, 0.06);
        border: 1px solid rgba(0, 206, 201, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        font-family: 'Inter', monospace;
        font-size: 0.88rem;
    }

    .source-box .label {
        color: var(--accent);
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    /* ── Sidebar ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--primary-light);
        margin-bottom: 0.4rem;
        letter-spacing: 0.03em;
    }

    .sidebar-divider {
        height: 1px;
        background: var(--border);
        margin: 1.2rem 0;
    }

    /* ── Status badges ──────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    .badge-success {
        background: rgba(0, 184, 148, 0.15);
        color: var(--success);
        border: 1px solid rgba(0, 184, 148, 0.3);
    }

    .badge-warning {
        background: rgba(253, 203, 110, 0.15);
        color: var(--warning);
        border: 1px solid rgba(253, 203, 110, 0.3);
    }

    .badge-info {
        background: rgba(108, 92, 231, 0.15);
        color: var(--primary-light);
        border: 1px solid rgba(108, 92, 231, 0.3);
    }

    .badge-accent {
        background: rgba(0, 206, 201, 0.15);
        color: var(--accent);
        border: 1px solid rgba(0, 206, 201, 0.3);
    }

    /* ── Metrics ────────────────────────────────────────── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .metric-item {
        background: rgba(108, 92, 231, 0.06);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--accent);
    }

    .metric-label {
        font-size: 0.72rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.2rem;
    }


    /* ── Button styling ─────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-light));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.35);
    }

    /* ── Spinner ─────────────────────────────────────────── */
    .stSpinner > div {
        border-color: var(--primary-light) transparent transparent transparent;
    }

    /* ── How-it-works section ──────────────────────────── */
    .how-it-works {
        background: linear-gradient(135deg, rgba(108, 92, 231, 0.05), rgba(0, 206, 201, 0.05));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
    }

    .step-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .step-item {
        background: rgba(108, 92, 231, 0.06);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .step-item:hover {
        border-color: var(--primary-light);
        transform: translateY(-2px);
    }

    .step-num {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }

    .step-title {
        font-weight: 600;
        color: var(--primary-light);
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .step-desc {
        font-size: 0.78rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# Initialise session state

def _init_rag_engine():
    """Create and cache the RAG engine in session state."""
    if "faiss_store" not in st.session_state:
        st.session_state.faiss_store = FAISSStore()
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(st.session_state.faiss_store)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "sources_ingested" not in st.session_state:
        st.session_state.sources_ingested = []
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0


_init_rag_engine()

# Convenience references
faiss_store: FAISSStore = st.session_state.faiss_store
pipeline: RAGPipeline = st.session_state.rag_pipeline


# Helper functions

def ingest_pdf(uploaded_file) -> dict:
    """Process a PDF file and add to the knowledge base."""
    try:
        contents = uploaded_file.read()
        documents = load_pdf(file_bytes=contents, filename=uploaded_file.name)

        if not documents:
            return {"error": "No text could be extracted from the PDF."}

        faiss_store.add_documents(documents)
        pipeline.register_documents(documents)

        st.session_state.total_chunks += len(documents)
        st.session_state.sources_ingested.append({
            "name": uploaded_file.name,
            "type": "pdf",
            "chunks": len(documents),
        })

        return {
            "status": "success",
            "chunks_added": len(documents),
            "message": f"PDF '{uploaded_file.name}' ingested successfully",
        }
    except Exception as exc:
        logger.error(f"PDF ingestion failed: {exc}")
        return {"error": str(exc)}


def ingest_website(url: str) -> dict:
    """Scrape a website and add to the knowledge base."""
    # Basic URL validation
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if parsed.scheme not in ("http", "https"):
        return {"error": "Only HTTP/HTTPS URLs are allowed."}
    hostname = (parsed.hostname or "").lower()
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
        return {"error": "Internal URLs are not allowed."}

    try:
        documents = load_webpage(url)

        if not documents:
            return {"error": f"No content could be extracted from {url}"}

        faiss_store.add_documents(documents)
        pipeline.register_documents(documents)

        display_name = url[:50] + "..." if len(url) > 50 else url
        st.session_state.total_chunks += len(documents)
        st.session_state.sources_ingested.append({
            "name": display_name,
            "type": "web",
            "chunks": len(documents),
        })

        return {
            "status": "success",
            "chunks_added": len(documents),
            "message": f"Website ingested successfully",
        }
    except Exception as exc:
        logger.error(f"Website ingestion failed: {exc}")
        return {"error": str(exc)}


def query_rag(question: str) -> dict:
    """Query the RAG pipeline directly."""
    if not faiss_store.is_ready:
        return {"error": "No documents indexed yet. Please upload a document first."}

    try:
        result = pipeline.run(question)
        answer = result.get("answer") or "No answer was generated. Please try again."
        return {
            "answer": answer,
            "sources": result.get("sources", ""),
            "llm_provider": result.get("llm_provider", "unknown"),
            "timings": result.get("timings", {}),
            "reranked_docs": result.get("reranked_docs", []),
            "error": result.get("error"),
        }
    except Exception as exc:
        logger.error(f"Query failed: {exc}")
        return {"error": str(exc)}


def clear_knowledge_base():
    """Clear all ingested data and start fresh."""
    faiss_store.clear()
    pipeline.clear()
    st.session_state.chat_history = []
    st.session_state.sources_ingested = []
    st.session_state.total_chunks = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown('<p class="sidebar-title">📂 Knowledge Sources</p>', unsafe_allow_html=True)

    # ── Status ────────────────────────────────────────────
    groq_ok = bool(settings.groq_api_key)
    idx_ready = faiss_store.is_ready
    groq_badge = "badge-success" if groq_ok else "badge-warning"
    groq_label = "Groq ✓" if groq_ok else "Groq ✗"
    idx_badge = "badge-success" if idx_ready else "badge-warning"
    idx_label = f"Index: {st.session_state.total_chunks} chunks" if idx_ready else "Index: Empty"
    st.markdown(
        f'<span class="badge {groq_badge}">{groq_label}</span> '
        f'<span class="badge {idx_badge}">{idx_label}</span>',
        unsafe_allow_html=True,
    )

    if not groq_ok:
        st.warning("Set GROQ_API_KEY as an environment variable.")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── PDF Upload ────────────────────────────────────────
    st.markdown("**Upload PDF**")
    pdf_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key="pdf_uploader",
        label_visibility="collapsed",
    )
    if pdf_file and st.button("Process PDF", key="btn_pdf"):
        with st.spinner("Ingesting PDF — parsing, chunking, embedding..."):
            result = ingest_pdf(pdf_file)
        if "error" not in result:
            st.success(f"{result['message']} ({result['chunks_added']} chunks)")
        else:
            st.error(f"{result['error']}")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Website URL ───────────────────────────────────────
    st.markdown("**Add Website**")
    web_url = st.text_input(
        "Enter URL",
        placeholder="https://example.com/article",
        key="web_url_input",
        label_visibility="collapsed",
    )
    if web_url and st.button("Scrape Website", key="btn_web"):
        with st.spinner("Scraping website — fetching, parsing, embedding..."):
            result = ingest_website(web_url)
        if "error" not in result:
            st.success(f"{result['message']} ({result['chunks_added']} chunks)")
        else:
            st.error(f"{result['error']}")

    # ── Ingested sources list ─────────────────────────────
    if st.session_state.sources_ingested:
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Ingested Sources**")
        for src in st.session_state.sources_ingested:
            icon = "📄" if src["type"] == "pdf" else "🌐"
            name = src["name"]
            if len(name) > 40:
                name = name[:37] + "..."
            st.markdown(f"- {icon} {name} ({src['chunks']} chunks)")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ── Clear button ──────────────────────────────────
        if st.button("🗑️ Clear Knowledge Base", key="btn_clear"):
            clear_knowledge_base()
            st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN AREA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<div class="main-header">
    <h1>Enterprise RAG — AI Document Q&A</h1>
    <p>Upload documents · Ask questions · Get AI-powered answers with citations</p>
</div>
""", unsafe_allow_html=True)

# ── Question Input ────────────────────────────────────────────────────────────

question = st.text_input(
    "Ask a question about your documents",
    placeholder="What would you like to know from the uploaded documents?",
    key="question_input",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    ask_clicked = st.button("Ask", key="btn_ask", use_container_width=True)

# ── Query execution ───────────────────────────────────────────────────────────

if ask_clicked and question:
    if not faiss_store.is_ready:
        st.warning("⚠️ Please upload a PDF or add a website URL first.")
    elif not settings.groq_api_key:
        st.error("GROQ_API_KEY is not set. Go to Space Settings → Secrets to add it.")
    else:
        with st.spinner("Running RAG pipeline — retrieving, re-ranking, generating..."):
            start = time.perf_counter()
            result = query_rag(question)
            total_time = time.perf_counter() - start

        if result and not result.get("error"):
            # Store in session chat history
            st.session_state.chat_history.append({
                "question": question,
                "result": result,
                "total_time": total_time,
            })
        elif result:
            st.error(f"{result.get('error', 'Query failed')}")

# ── Display chat history ─────────────────────────────────────────────────────

for entry in reversed(st.session_state.chat_history):
    q = entry["question"]
    res = entry["result"]
    t = entry["total_time"]

    # Question
    st.markdown(f"#### {q}")

    # Provider & time badges
    provider = res.get("llm_provider", "unknown")
    st.markdown(
        f'<span class="badge badge-info">{provider}</span> '
        f'<span class="badge badge-accent">⏱ {t:.1f}s</span>',
        unsafe_allow_html=True,
    )

    # Answer
    answer = res.get("answer", "No answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # Sources
    sources = res.get("sources", "")
    if sources:
        st.markdown(
            f'<div class="source-box">'
            f'<div class="label">📎 Sources</div>'
            f'<pre style="margin:0;white-space:pre-wrap;color:var(--text-primary);">{sources}</pre>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Timing metrics
    timings = res.get("timings", {})
    if timings:
        cols = st.columns(len(timings) + 1)
        cols[0].metric("Total", f"{t:.2f}s")
        for i, (k, v) in enumerate(timings.items(), 1):
            label = k.replace("_", " ").title()
            cols[i % len(cols)].metric(label, f"{v:.2f}s")

    


# ── Empty state ───────────────────────────────────────────────────────────────

if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align:center; padding:3rem 2rem; color:var(--text-secondary);">
        <h3 style="color:var(--primary-light); font-weight:500;">Welcome! Upload a document to get started</h3>
        <p style="max-width:550px; margin:1rem auto; line-height:1.7;">
            Upload a PDF or add a website URL in the sidebar. Then ask questions about the content
            — the AI will find the most relevant passages and generate accurate answers with source citations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # How it works section
    st.markdown("""
    <div class="how-it-works">
        <h4 style="text-align:center; color:var(--primary-light); margin-bottom:0.5rem;">How It Works</h4>
        <div class="step-grid">
            <div class="step-item">
                <div class="step-num"></div>
                <div class="step-title">Upload Document</div>
                <div class="step-desc">Upload a PDF or paste a URL. The system extracts, cleans, and chunks the text.</div>
            </div>
            <div class="step-item">
                <div class="step-num"></div>
                <div class="step-title">Multi-Retrieval</div>
                <div class="step-desc">3 retrievers (Vector, Sentence Window, Graph) search for relevant passages.</div>
            </div>
            <div class="step-item">
                <div class="step-num"></div>
                <div class="step-title">Fusion + Re-Rank</div>
                <div class="step-desc">Results are fused with RRF and re-ranked with a cross-encoder model.</div>
            </div>
            <div class="step-item">
                <div class="step-num"></div>
                <div class="step-title">AI Answer</div>
                <div class="step-desc">Groq LLM generates a concise answer using only the retrieved context.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
