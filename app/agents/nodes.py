"""LangGraph node factories for the clinical agent workflow.

Each `make_*_node` function is a small factory that closes over its
dependencies and returns a plain `state -> partial_state_update`
callable — the shape LangGraph expects. This keeps every node testable
in isolation with fake dependencies, consistent with the rest of this
codebase's dependency-injection style.
"""
from __future__ import annotations

from collections.abc import Callable

from app.agents.router import route_query
from app.agents.state import AgentState
from app.core.constants import RetrieverType
from app.core.logging import get_logger
from app.llm.groq_service import GroqService
from app.llm.response_formatter import parse_clinical_report
from app.loaders.base_loader import BaseLoader
from app.prompts.clinical_prompt import build_clinical_report_messages
from app.retrievers.reranker import CrossEncoderReranker
from app.services.citation_service import CitationService
from app.services.retrieval_service import RetrievalService
from app.vectorstore.vector_store import VectorStore

logger = get_logger(__name__)

Node = Callable[[AgentState], dict]


def make_route_node(groq_service: GroqService) -> Node:
    """Decide retriever strategy, live-search need, and query reformulation."""

    def route_node(state: AgentState) -> dict:
        decision = route_query(state["query"], state.get("history", []), groq_service)
        logger.info(
            f"agent_route retriever={decision.retriever_type.value} "
            f"live_search={decision.use_live_pubmed_search}"
        )
        return {
            "retriever_type": decision.retriever_type.value,
            "use_live_pubmed_search": decision.use_live_pubmed_search,
            "search_terms": decision.search_terms,
            "reformulated_query": decision.reformulated_query,
        }

    return route_node


def make_live_search_node(vector_store: VectorStore, pubmed_loader: BaseLoader) -> Node:
    """If the router flagged it, search PubMed live and fold results into the index."""

    def live_search_node(state: AgentState) -> dict:
        if not state.get("use_live_pubmed_search"):
            # LangGraph requires every node to write at least one state key;
            # write back the unchanged flag as a harmless no-op.
            return {"use_live_pubmed_search": False}
        try:
            documents = pubmed_loader.safe_load(state.get("search_terms") or [state["query"]])
            if documents:
                vector_store.add_documents(documents)
                logger.info(f"agent_live_search documents_added={len(documents)}")
            return {"use_live_pubmed_search": True}
        except Exception as exc:  # noqa: BLE001
            logger.error(f"agent_live_search failed error={exc}")
            return {"use_live_pubmed_search": True}

    return live_search_node


def make_retrieve_node(retrieval_service: RetrievalService, k: int) -> Node:
    """Retrieve candidate documents using the router's chosen strategy."""

    def retrieve_node(state: AgentState) -> dict:
        retriever_type = RetrieverType(state.get("retriever_type", RetrieverType.MULTI_QUERY.value))
        query = state.get("reformulated_query") or state["query"]
        documents, _ = retrieval_service.retrieve_documents(query, retriever_type, k=k)
        return {"documents": documents}

    return retrieve_node


def make_rerank_node(reranker: CrossEncoderReranker, top_k: int) -> Node:
    """Rerank the candidate set with a cross-encoder for higher precision."""

    def rerank_node(state: AgentState) -> dict:
        documents = state.get("documents", [])
        reranked = reranker.rerank(state["query"], documents, top_k=top_k)
        return {"documents": reranked}

    return rerank_node


def make_sufficiency_node(max_iterations: int) -> Node:
    """Decide whether the current evidence is enough, or whether to retry with a broader query."""

    def sufficiency_node(state: AgentState) -> dict:
        iterations = state.get("iterations", 0) + 1
        has_enough_evidence = len(state.get("documents", [])) >= 2
        sufficient = has_enough_evidence or iterations >= max_iterations

        update: dict = {"iterations": iterations, "sufficient": sufficient}
        if not sufficient:
            update["reformulated_query"] = f"{state['query']} overview background general information"
        return update

    return sufficiency_node


def sufficiency_router(state: AgentState) -> str:
    """Conditional-edge function: loop back to retrieval, or move on to generation."""
    return "generate" if state.get("sufficient") else "retrieve"


def make_generate_node(groq_service: GroqService, citation_service: CitationService) -> Node:
    """Generate the final structured, cited clinical report from the retrieved evidence."""

    def generate_node(state: AgentState) -> dict:
        documents = state.get("documents", [])
        messages = build_clinical_report_messages(state["query"], documents)
        raw_output = groq_service.generate(messages, json_mode=True)
        report = parse_clinical_report(raw_output)

        references = citation_service.build_references(documents)
        if references:
            report = report.model_copy(update={"references": references})

        return {"report": report}

    return generate_node


def make_record_history_node() -> Node:
    """Append this turn to the conversation's persisted history for future turns."""

    def record_history_node(state: AgentState) -> dict:
        report = state["report"]
        return {"history": [{"query": state["query"], "answer_summary": report.summary[:300]}]}

    return record_history_node
