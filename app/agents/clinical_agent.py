"""Assembles the clinical agent's LangGraph workflow.

Graph shape:

    route -> live_search -> retrieve -> rerank -> sufficiency --(insufficient)--> retrieve
                                                        |
                                                   (sufficient)
                                                        v
                                                    generate -> record_history -> END

`MemorySaver` gives the graph per-conversation memory: invoking with the
same `thread_id` (our `conversation_id`) resumes accumulated state,
which is how follow-up questions see prior turns via `history`. Swapping
to a persistent checkpointer (e.g. a future Redis- or Postgres-backed
one) is a one-line change here, isolated from every node.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    make_generate_node,
    make_live_search_node,
    make_record_history_node,
    make_rerank_node,
    make_retrieve_node,
    make_route_node,
    make_sufficiency_node,
    sufficiency_router,
)
from app.agents.state import AgentState
from app.core.config import Settings
from app.llm.groq_service import GroqService
from app.loaders.base_loader import BaseLoader
from app.retrievers.reranker import CrossEncoderReranker
from app.services.citation_service import CitationService
from app.services.retrieval_service import RetrievalService
from app.vectorstore.vector_store import VectorStore


def build_agent_graph(
    *,
    groq_service: GroqService,
    retrieval_service: RetrievalService,
    citation_service: CitationService,
    vector_store: VectorStore,
    pubmed_loader: BaseLoader,
    reranker: CrossEncoderReranker,
    settings: Settings,
) -> Any:
    """Wire all nodes together into a compiled, checkpointed LangGraph app."""
    graph = StateGraph(AgentState)

    graph.add_node("route", make_route_node(groq_service))
    graph.add_node("live_search", make_live_search_node(vector_store, pubmed_loader))
    graph.add_node("retrieve", make_retrieve_node(retrieval_service, k=settings.agent_retrieve_k))
    graph.add_node("rerank", make_rerank_node(reranker, top_k=settings.agent_rerank_top_k))
    graph.add_node("sufficiency", make_sufficiency_node(max_iterations=settings.agent_max_iterations))
    graph.add_node("generate", make_generate_node(groq_service, citation_service))
    graph.add_node("record_history", make_record_history_node())

    graph.set_entry_point("route")
    graph.add_edge("route", "live_search")
    graph.add_edge("live_search", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "sufficiency")
    graph.add_conditional_edges("sufficiency", sufficiency_router, {"generate": "generate", "retrieve": "retrieve"})
    graph.add_edge("generate", "record_history")
    graph.add_edge("record_history", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
