"""
Graph-Based Knowledge Retriever.

Builds a lightweight knowledge graph from document entities and relationships
using NetworkX, then retrieves relevant sub-graphs for a query.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import networkx as nx
from langchain_core.documents import Document
from loguru import logger

from config import settings
from utils.helpers import timed


class GraphRetriever:
    """
    Builds and queries a knowledge graph derived from ingested documents.

    Entity extraction uses a lightweight regex/NLP approach (no external
    API calls) to keep the system self-contained.
    """

    def __init__(self):
        self._graph = nx.Graph()
        # Map: entity → list of source Documents that mention it
        self._entity_docs: Dict[str, List[Document]] = {}

    # Graph construction

    @timed
    def build_graph(self, documents: List[Document]) -> None:
        """
        Extract entities from documents and build a knowledge graph.

        Each entity becomes a node; co-occurrence within the same chunk
        creates an edge.
        """
        for doc in documents:
            entities = self._extract_entities(doc.page_content)
            for entity in entities:
                self._graph.add_node(entity)
                self._entity_docs.setdefault(entity, []).append(doc)

            # Create edges between co-occurring entities
            entity_list = list(entities)
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    if self._graph.has_edge(entity_list[i], entity_list[j]):
                        self._graph[entity_list[i]][entity_list[j]]["weight"] += 1
                    else:
                        self._graph.add_edge(entity_list[i], entity_list[j], weight=1)

        logger.info(
            f"Knowledge graph built: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

    def clear_graph(self) -> None:
        """Clear the knowledge graph and entity docs."""
        self._graph.clear()
        self._entity_docs.clear()
        logger.debug("GraphRetriever graph cleared")

    # Retrieval

    @timed
    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        """
        Retrieve documents related to the query via the knowledge graph.

        Steps:
          1. Extract entities from the query.
          2. Find matching nodes in the graph.
          3. Traverse 1-hop neighbours.
          4. Collect all associated documents.
        """
        top_k = top_k or settings.top_k_retrieval
        if self._graph.number_of_nodes() == 0:
            logger.warning("Knowledge graph is empty")
            return []

        query_entities = self._extract_entities(query)

        # Find matching graph nodes
        matched_nodes: Set[str] = set()
        query_lower = query.lower()
        for entity in query_entities:
            entity_lower = entity.lower()
            for node in self._graph.nodes():
                if entity_lower in node.lower() or node.lower() in entity_lower:
                    matched_nodes.add(node)

        # Also match individual query words against nodes
        query_words = set(query_lower.split())
        for node in self._graph.nodes():
            node_lower = node.lower()
            if any(w in node_lower for w in query_words if len(w) > 3):
                matched_nodes.add(node)

        if not matched_nodes:
            logger.info("No graph nodes matched the query")
            return []

        # 1-hop expansion
        expanded: Set[str] = set(matched_nodes)
        for node in matched_nodes:
            neighbours = list(self._graph.neighbors(node))
            # Sort by edge weight, take top neighbours
            neighbours.sort(
                key=lambda n: self._graph[node][n].get("weight", 0), reverse=True
            )
            expanded.update(neighbours[:5])

        # Collect documents
        seen: set[str] = set()
        result_docs: List[Document] = []
        for entity in expanded:
            for doc in self._entity_docs.get(entity, []):
                doc_id = doc.page_content[:100]
                if doc_id not in seen:
                    seen.add(doc_id)
                    result_docs.append(doc)
                    if len(result_docs) >= top_k:
                        break
            if len(result_docs) >= top_k:
                break

        logger.info(
            f"GraphRetriever: matched {len(matched_nodes)} nodes, "
            f"expanded to {len(expanded)}, returning {len(result_docs)} docs"
        )
        return result_docs

    # Entity extraction

    @staticmethod
    def _extract_entities(text: str) -> Set[str]:
        """
        Lightweight entity extraction using regex heuristics.

        Captures:
          - Capitalised multi-word phrases (proper nouns / named entities)
          - Quoted terms
          - Technical terms (words with special characters like dots, hyphens)
        """
        entities: Set[str] = set()

        # Capitalised phrases (2-4 words)
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text):
            entity = match.group(1).strip()
            if len(entity) > 2:
                entities.add(entity)

        # Quoted terms
        for match in re.finditer(r'"([^"]{2,50})"', text):
            entities.add(match.group(1).strip())

        # Technical terms with dots / hyphens (e.g., machine-learning, PyTorch.js)
        for match in re.finditer(r"\b([A-Za-z]+[-\.][A-Za-z]+(?:[-\.][A-Za-z]+)*)\b", text):
            term = match.group(1)
            if len(term) > 3:
                entities.add(term)

        # ALL-CAPS acronyms ≥ 2 chars
        for match in re.finditer(r"\b([A-Z]{2,6})\b", text):
            entities.add(match.group(1))

        return entities

    # Introspection

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
