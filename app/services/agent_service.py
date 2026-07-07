"""Agentic orchestration service: LangGraph-driven, memory-aware clinical assistant.

This is the agentic counterpart to `ReportService`. Where `ReportService`
always does exactly retrieve-then-generate, `AgentService` runs a graph
that can choose its retrieval strategy, decide to search PubMed live,
retry with a broadened query if evidence is thin, and remember prior
turns in the same conversation.
"""
from __future__ import annotations

import time
from typing import Any

from app.core.logging import get_logger
from app.models.response_models import AgentChatResponse

logger = get_logger(__name__)


class AgentService:
    """Thin wrapper invoking the compiled clinical agent graph per conversation."""

    def __init__(self, agent_graph: Any) -> None:
        self._agent_graph = agent_graph

    def chat(self, conversation_id: str, query: str) -> AgentChatResponse:
        """Run one turn of the agentic workflow for the given conversation."""
        start = time.perf_counter()
        config = {"configurable": {"thread_id": conversation_id}}

        result = self._agent_graph.invoke(
            {
                "conversation_id": conversation_id,
                "query": query,
                "reformulated_query": query,
                "iterations": 0,
                "sufficient": False,
            },
            config=config,
        )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(f"agent_service conversation_id={conversation_id} latency_ms={latency_ms:.1f}")

        return AgentChatResponse(
            conversation_id=conversation_id,
            query=query,
            report=result["report"],
            retriever_used=result.get("retriever_type", "multi_query"),
            documents_retrieved=len(result.get("documents", [])),
            used_live_pubmed_search=bool(result.get("use_live_pubmed_search", False)),
            iterations=result.get("iterations", 1),
            latency_ms=latency_ms,
        )
