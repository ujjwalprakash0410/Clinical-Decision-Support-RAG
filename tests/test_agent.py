"""Tests for the agentic LangGraph workflow (router, reranker, nodes, full graph)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.agents.clinical_agent import build_agent_graph
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
from app.agents.router import RoutingDecision, route_query
from app.core.constants import CLINICAL_DISCLAIMER, RetrieverType
from app.core.config import get_settings
from app.retrievers.reranker import CrossEncoderReranker


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


def test_route_query_parses_valid_llm_json():
    fake_groq = MagicMock()
    fake_groq.generate.return_value = json.dumps(
        {
            "retriever_type": "hybrid",
            "use_live_pubmed_search": True,
            "search_terms": ["amoxicillin", "pneumonia"],
            "reformulated_query": "What dose of amoxicillin treats chest indrawing pneumonia?",
        }
    )
    decision = route_query("what about amoxicillin dosing", [], fake_groq)
    assert decision.retriever_type == RetrieverType.HYBRID
    assert decision.use_live_pubmed_search is True
    assert decision.search_terms == ["amoxicillin", "pneumonia"]


def test_route_query_falls_back_safely_on_malformed_json():
    fake_groq = MagicMock()
    fake_groq.generate.return_value = "not valid json"
    decision = route_query("some question", [], fake_groq)
    assert decision.retriever_type == RetrieverType.MULTI_QUERY
    assert decision.use_live_pubmed_search is False
    assert decision.reformulated_query == "some question"


def test_route_query_rejects_unknown_retriever_type():
    fake_groq = MagicMock()
    fake_groq.generate.return_value = json.dumps({"retriever_type": "not_a_real_type"})
    decision = route_query("q", [], fake_groq)
    assert decision.retriever_type == RetrieverType.MULTI_QUERY


# --------------------------------------------------------------------------
# Reranker
# --------------------------------------------------------------------------


def test_cross_encoder_reranker_orders_by_score(sample_documents):
    def fake_score_fn(pairs):
        # Score the second document higher regardless of input order.
        return [0.1, 0.9][: len(pairs)]

    reranker = CrossEncoderReranker(fake_score_fn)
    reranked = reranker.rerank("query", sample_documents, top_k=2)
    assert reranked[0] == sample_documents[1]


def test_cross_encoder_reranker_handles_empty_documents():
    reranker = CrossEncoderReranker(lambda pairs: [])
    assert reranker.rerank("query", [], top_k=5) == []


def test_cross_encoder_reranker_respects_top_k(sample_documents):
    reranker = CrossEncoderReranker(lambda pairs: [0.5] * len(pairs))
    reranked = reranker.rerank("query", sample_documents, top_k=1)
    assert len(reranked) == 1


# --------------------------------------------------------------------------
# Individual nodes
# --------------------------------------------------------------------------


def test_route_node_returns_expected_keys():
    fake_groq = MagicMock()
    fake_groq.generate.return_value = json.dumps(
        {"retriever_type": "similarity", "use_live_pubmed_search": False, "search_terms": [], "reformulated_query": "q"}
    )
    node = make_route_node(fake_groq)
    result = node({"query": "q", "history": []})
    assert result["retriever_type"] == "similarity"
    assert result["use_live_pubmed_search"] is False


def test_live_search_node_skips_when_not_flagged():
    fake_vector_store = MagicMock()
    fake_loader = MagicMock()
    node = make_live_search_node(fake_vector_store, fake_loader)
    result = node({"use_live_pubmed_search": False})
    assert result == {"use_live_pubmed_search": False}
    fake_loader.safe_load.assert_not_called()


def test_live_search_node_adds_documents_when_flagged(sample_documents):
    fake_vector_store = MagicMock()
    fake_loader = MagicMock()
    fake_loader.safe_load.return_value = sample_documents
    node = make_live_search_node(fake_vector_store, fake_loader)
    node({"use_live_pubmed_search": True, "search_terms": ["afib"], "query": "afib"})
    fake_vector_store.add_documents.assert_called_once_with(sample_documents)


def test_retrieve_node_calls_retrieval_service_with_reformulated_query(sample_documents):
    fake_retrieval_service = MagicMock()
    fake_retrieval_service.retrieve_documents.return_value = (sample_documents, 12.3)
    node = make_retrieve_node(fake_retrieval_service, k=5)
    result = node({"query": "orig", "reformulated_query": "better query", "retriever_type": "similarity"})
    args, kwargs = fake_retrieval_service.retrieve_documents.call_args
    assert args[0] == "better query"
    assert result["documents"] == sample_documents


def test_sufficiency_node_marks_sufficient_with_enough_documents(sample_documents):
    node = make_sufficiency_node(max_iterations=2)
    result = node({"documents": sample_documents, "iterations": 0, "query": "q"})
    assert result["sufficient"] is True
    assert result["iterations"] == 1


def test_sufficiency_node_forces_stop_after_max_iterations():
    node = make_sufficiency_node(max_iterations=2)
    result = node({"documents": [], "iterations": 1, "query": "q"})
    assert result["sufficient"] is True


def test_sufficiency_node_requests_retry_with_thin_evidence():
    node = make_sufficiency_node(max_iterations=3)
    result = node({"documents": [], "iterations": 0, "query": "q"})
    assert result["sufficient"] is False
    assert "reformulated_query" in result


def test_sufficiency_router_dispatches_correctly():
    assert sufficiency_router({"sufficient": True}) == "generate"
    assert sufficiency_router({"sufficient": False}) == "retrieve"


def test_generate_node_produces_report_with_disclaimer(sample_documents):
    fake_groq = MagicMock()
    fake_groq.generate.return_value = json.dumps(
        {
            "summary": "Test summary",
            "possible_conditions": [],
            "suggested_diagnostic_tests": [],
            "red_flag_symptoms": [],
            "evidence_summary": "Evidence",
            "clinical_guidelines": [],
            "references": [],
            "confidence": "moderate",
            "limitations": "Limited",
            "disclaimer": CLINICAL_DISCLAIMER,
        }
    )
    fake_citation_service = MagicMock()
    fake_citation_service.build_references.return_value = []
    node = make_generate_node(fake_groq, fake_citation_service)
    result = node({"query": "q", "documents": sample_documents})
    assert result["report"].disclaimer == CLINICAL_DISCLAIMER


def test_record_history_node_appends_turn():
    fake_report = MagicMock()
    fake_report.summary = "a" * 400
    node = make_record_history_node()
    result = node({"query": "q", "report": fake_report})
    assert result["history"][0]["query"] == "q"
    assert len(result["history"][0]["answer_summary"]) == 300


# --------------------------------------------------------------------------
# Full graph, with conversation memory across turns
# --------------------------------------------------------------------------


@pytest.fixture
def compiled_agent(sample_documents):
    fake_groq = MagicMock()

    def generate_side_effect(messages, json_mode=True):
        system_content = messages[0]["content"]
        if "routing agent" in system_content:
            return json.dumps(
                {
                    "retriever_type": "similarity",
                    "use_live_pubmed_search": False,
                    "search_terms": [],
                    "reformulated_query": "resolved standalone question",
                }
            )
        return json.dumps(
            {
                "summary": "Generated summary",
                "possible_conditions": [],
                "suggested_diagnostic_tests": [],
                "red_flag_symptoms": [],
                "evidence_summary": "Evidence",
                "clinical_guidelines": [],
                "references": [],
                "confidence": "moderate",
                "limitations": "Limited",
                "disclaimer": CLINICAL_DISCLAIMER,
            }
        )

    fake_groq.generate.side_effect = generate_side_effect

    fake_retrieval_service = MagicMock()
    fake_retrieval_service.retrieve_documents.return_value = (sample_documents, 5.0)

    fake_citation_service = MagicMock()
    fake_citation_service.build_references.return_value = []

    fake_vector_store = MagicMock()
    fake_pubmed_loader = MagicMock()
    fake_reranker = CrossEncoderReranker(lambda pairs: [1.0] * len(pairs))

    settings = get_settings()

    return build_agent_graph(
        groq_service=fake_groq,
        retrieval_service=fake_retrieval_service,
        citation_service=fake_citation_service,
        vector_store=fake_vector_store,
        pubmed_loader=fake_pubmed_loader,
        reranker=fake_reranker,
        settings=settings,
    )


def test_agent_graph_produces_a_report(compiled_agent):
    config = {"configurable": {"thread_id": "conv-1"}}
    result = compiled_agent.invoke(
        {"query": "What treats chest indrawing pneumonia?", "iterations": 0, "sufficient": False},
        config=config,
    )
    assert result["report"].disclaimer == CLINICAL_DISCLAIMER
    assert result["sufficient"] is True


def test_agent_graph_persists_history_across_turns(compiled_agent):
    config = {"configurable": {"thread_id": "conv-memory-test"}}
    first = compiled_agent.invoke(
        {"query": "First question", "iterations": 0, "sufficient": False}, config=config
    )
    assert len(first["history"]) == 1

    second = compiled_agent.invoke(
        {"query": "Follow-up question", "iterations": 0, "sufficient": False}, config=config
    )
    # History accumulates across turns for the same thread_id (conversation).
    assert len(second["history"]) == 2
    assert second["history"][0]["query"] == "First question"
    assert second["history"][1]["query"] == "Follow-up question"


def test_agent_graph_separate_conversations_have_independent_history(compiled_agent):
    compiled_agent.invoke(
        {"query": "Q in conv A", "iterations": 0, "sufficient": False},
        config={"configurable": {"thread_id": "conv-A"}},
    )
    result_b = compiled_agent.invoke(
        {"query": "Q in conv B", "iterations": 0, "sufficient": False},
        config={"configurable": {"thread_id": "conv-B"}},
    )
    assert len(result_b["history"]) == 1
