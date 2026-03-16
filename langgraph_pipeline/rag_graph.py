"""
LangGraph RAG Pipeline.

Orchestrates the full RAG workflow using LangGraph:
  Query Processing → Parallel Retrieval → Fusion → Re-Ranking → LLM Generation → Response Formatting

"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from loguru import logger

from config import settings
from fusion.rrf_fusion import reciprocal_rank_fusion
from llm.groq_client import GroqClient
from reranker.reranker import rerank_documents
from retrievers.graph_retriever import GraphRetriever
from retrievers.sentence_retriever import SentenceWindowRetriever
from retrievers.vector_retriever import VectorRetriever
from utils.helpers import format_sources, timed
from vectorstore.faiss_store import FAISSStore


# State Schema

class RAGState(TypedDict, total=False):
    """Typed state dictionary flowing through the LangGraph nodes."""
    query: str
    processed_query: str
    vector_results: List[Document]
    sentence_results: List[Document]
    graph_results: List[Document]
    fused_results: List[Document]
    reranked_results: List[Document]
    context: str
    answer: str
    sources: str
    llm_provider: str
    error: Optional[str]
    timings: Dict[str, float]


# Pipeline Builder

class RAGPipeline:
    """
    Builds and manages the LangGraph RAG pipeline.

    Usage:
        pipeline = RAGPipeline(faiss_store)
        pipeline.register_documents(docs)
        result = pipeline.run("What is machine learning?")
    """

    def __init__(self, faiss_store: FAISSStore):
        self._faiss_store = faiss_store

        # Retrievers
        self._vector_retriever = VectorRetriever(faiss_store)
        self._sentence_retriever = SentenceWindowRetriever(faiss_store, window_size=1)
        self._graph_retriever = GraphRetriever()

        # LLM client
        self._groq = GroqClient()

        # Build the graph
        self._graph = self._build_graph()

    def register_documents(self, documents: List[Document]) -> None:
        """Register documents with retrievers that need them (sentence window, graph)."""
        self._sentence_retriever.register_documents(documents)
        self._graph_retriever.build_graph(documents)

    def clear(self) -> None:
        """Clear all retriever state for a fresh session."""
        self._sentence_retriever.clear_registry()
        self._graph_retriever.clear_graph()

    # Graph Construction

    def _build_graph(self) -> Any:
        """Build the LangGraph state graph."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("query_processing", self._query_processing_node)
        workflow.add_node("parallel_retrieval", self._parallel_retrieval_node)
        workflow.add_node("fusion", self._fusion_node)
        workflow.add_node("reranking", self._reranking_node)
        workflow.add_node("llm_generation", self._llm_generation_node)
        workflow.add_node("response_formatting", self._response_formatting_node)

        # Define edges
        workflow.set_entry_point("query_processing")
        workflow.add_edge("query_processing", "parallel_retrieval")
        workflow.add_edge("parallel_retrieval", "fusion")
        workflow.add_edge("fusion", "reranking")
        workflow.add_edge("reranking", "llm_generation")
        workflow.add_edge("llm_generation", "response_formatting")
        workflow.add_edge("response_formatting", END)

        return workflow.compile()

    # Node Implementations

    def _query_processing_node(self, state: RAGState) -> dict:
        """Process and optionally expand the query."""
        start = time.perf_counter()
        query = state["query"].strip()

        # Basic query cleaning
        processed = query
        logger.info(f"Query processed: '{processed}'")

        elapsed = time.perf_counter() - start
        timings = state.get("timings", {})
        timings["query_processing"] = elapsed
        return {"processed_query": processed, "timings": timings}

    def _parallel_retrieval_node(self, state: RAGState) -> dict:
        """Run all three retrievers."""
        start = time.perf_counter()
        query = state["processed_query"]

        # Vector retrieval
        vector_results = self._vector_retriever.retrieve(query)

        # Sentence window retrieval
        sentence_results = self._sentence_retriever.retrieve(query)

        # Graph retrieval
        graph_results = self._graph_retriever.retrieve(query)

        elapsed = time.perf_counter() - start
        timings = state.get("timings", {})
        timings["retrieval"] = elapsed

        logger.info(
            f"Retrieval complete: vector={len(vector_results)}, "
            f"sentence={len(sentence_results)}, graph={len(graph_results)}"
        )

        return {
            "vector_results": vector_results,
            "sentence_results": sentence_results,
            "graph_results": graph_results,
            "timings": timings,
        }

    def _fusion_node(self, state: RAGState) -> dict:
        """Fuse results from all retrievers using RRF."""
        start = time.perf_counter()

        result_lists = [
            state.get("vector_results", []),
            state.get("sentence_results", []),
            state.get("graph_results", []),
        ]
        # Filter out empty lists
        result_lists = [r for r in result_lists if r]

        if not result_lists:
            return {"fused_results": [], "error": "No documents retrieved"}

        fused = reciprocal_rank_fusion(result_lists)

        elapsed = time.perf_counter() - start
        timings = state.get("timings", {})
        timings["fusion"] = elapsed

        return {"fused_results": fused, "timings": timings}

    def _reranking_node(self, state: RAGState) -> dict:
        """Re-rank fused documents using cross-encoder."""
        start = time.perf_counter()
        query = state["processed_query"]
        fused = state.get("fused_results", [])

        if not fused:
            return {"reranked_results": []}

        reranked = rerank_documents(query, fused, top_k=settings.top_k_rerank)

        elapsed = time.perf_counter() - start
        timings = state.get("timings", {})
        timings["reranking"] = elapsed

        return {"reranked_results": reranked, "timings": timings}

    def _llm_generation_node(self, state: RAGState) -> dict:
        """Generate an answer using Groq LLM."""
        start = time.perf_counter()
        query = state["processed_query"]
        reranked = state.get("reranked_results", [])

        if not reranked:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "context": "",
                "llm_provider": "none",
            }

        # Build context string
        context_parts: list[str] = []
        for i, doc in enumerate(reranked, 1):
            source_info = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page")
            url = doc.metadata.get("url")
            source_label = source_info
            if page:
                source_label += f" (page {page})"
            elif url:
                source_label = url
            context_parts.append(f"[Source {i}: {source_label}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        # Use Groq
        answer = ""
        provider = ""

        if self._groq.is_available:
            try:
                logger.info("Attempting Groq generation...")
                answer = self._groq.generate(query, context)
                if answer:
                    provider = "Groq"
                    logger.info(f"Groq returned answer of {len(answer)} chars")
                else:
                    logger.warning(f"Groq returned empty/None answer: {answer!r}")
                    answer = ""
            except Exception as exc:
                logger.error(f"Groq failed: {exc}")
                answer = "LLM generation failed. Please check your GROQ_API_KEY and try again."
                provider = "error"

        if not answer:
            answer = "No LLM provider is configured. Please set GROQ_API_KEY in the Space secrets."
            provider = "none"

        elapsed = time.perf_counter() - start
        timings = state.get("timings", {})
        timings["llm_generation"] = elapsed

        return {
            "answer": answer,
            "context": context,
            "llm_provider": provider,
            "timings": timings,
        }

    def _response_formatting_node(self, state: RAGState) -> dict:
        """Format the final response with source citations."""
        reranked = state.get("reranked_results", [])
        sources = format_sources([{"metadata": d.metadata} for d in reranked])

        return {"sources": sources}

    # Public API

    @timed
    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a given query.

        Args:
            query: The user's natural language question.

        Returns:
            Dict with keys: answer, sources, llm_provider, timings, error
        """
        initial_state: RAGState = {
            "query": query,
            "timings": {},
        }

        try:
            result = self._graph.invoke(initial_state)
            answer = result.get("answer") or ""
            sources = result.get("sources") or ""
            llm_provider = result.get("llm_provider") or ""
            timings = result.get("timings") or {}
            logger.info(f"Pipeline result — answer length: {len(answer)}, provider: {llm_provider}")
            return {
                "answer": answer,
                "sources": sources,
                "llm_provider": llm_provider,
                "timings": timings,
                "reranked_docs": result.get("reranked_results") or [],
                "error": result.get("error"),
            }
        except Exception as exc:
            logger.error(f"Pipeline execution failed: {exc}")
            return {
                "answer": f"An error occurred: {exc}",
                "sources": "",
                "llm_provider": "error",
                "timings": {},
                "reranked_docs": [],
                "error": str(exc),
            }
