"""API-level tests. External dependencies (LLM, embeddings, vector store)
are replaced via FastAPI dependency overrides so tests run offline and fast.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.app_factory import create_app
from app.api.dependencies.di import (
    get_agent_service,
    get_indexing_service,
    get_report_service,
    get_vector_store,
)
from app.core.constants import CLINICAL_DISCLAIMER
from app.models.response_models import AgentChatResponse, AnalyzeResponse, ClinicalReport


@pytest.fixture
def client() -> TestClient:
    app = create_app()

    fake_report_service = MagicMock()
    fake_report_service.generate_report.return_value = AnalyzeResponse(
        query="test query",
        report=ClinicalReport(
            summary="Test summary",
            evidence_summary="Test evidence",
            confidence="moderate",
            limitations="Limited evidence available",
            disclaimer=CLINICAL_DISCLAIMER,
        ),
        retriever_used="multi_query",
        documents_retrieved=2,
        latency_ms=123.4,
    )

    fake_indexing_service = MagicMock()
    fake_indexing_service.index_sources.return_value = (5, ["who", "cdc"])

    fake_vector_store = MagicMock()
    fake_vector_store.document_count = 10

    fake_agent_service = MagicMock()
    fake_agent_service.chat.return_value = AgentChatResponse(
        conversation_id="conv-1",
        query="test query",
        report=ClinicalReport(
            summary="Agent summary",
            evidence_summary="Agent evidence",
            confidence="moderate",
            limitations="Limited evidence",
            disclaimer=CLINICAL_DISCLAIMER,
        ),
        retriever_used="multi_query",
        documents_retrieved=3,
        used_live_pubmed_search=False,
        iterations=1,
        latency_ms=456.7,
    )

    app.dependency_overrides[get_report_service] = lambda: fake_report_service
    app.dependency_overrides[get_indexing_service] = lambda: fake_indexing_service
    app.dependency_overrides[get_vector_store] = lambda: fake_vector_store
    app.dependency_overrides[get_agent_service] = lambda: fake_agent_service

    return TestClient(app)


def test_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sources_endpoint(client: TestClient):
    response = client.get("/api/v1/sources")
    assert response.status_code == 200
    assert "pubmed" in response.json()["available_sources"]


def test_metrics_endpoint(client: TestClient):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert response.json()["index_document_count"] == 10


def test_analyze_endpoint_returns_report(client: TestClient):
    response = client.post("/api/v1/analyze", json={"query": "What is the treatment for afib?"})
    assert response.status_code == 200
    body = response.json()
    assert body["report"]["disclaimer"] == CLINICAL_DISCLAIMER
    assert body["documents_retrieved"] == 2


def test_analyze_endpoint_rejects_empty_query(client: TestClient):
    response = client.post("/api/v1/analyze", json={"query": "  "})
    assert response.status_code == 422


def test_index_endpoint_returns_indexed_status(client: TestClient):
    response = client.post("/api/v1/index", json={"sources": ["who", "cdc"]})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "indexed"
    assert body["documents_indexed"] == 5


def test_agent_chat_endpoint_returns_report(client: TestClient):
    response = client.post(
        "/api/v1/agent/chat",
        json={"conversation_id": "conv-1", "query": "What treats fast breathing pneumonia?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv-1"
    assert body["report"]["disclaimer"] == CLINICAL_DISCLAIMER
    assert body["iterations"] == 1


def test_agent_chat_endpoint_rejects_missing_conversation_id(client: TestClient):
    response = client.post("/api/v1/agent/chat", json={"query": "valid question here"})
    assert response.status_code == 422


def test_agent_chat_endpoint_rejects_empty_query(client: TestClient):
    response = client.post("/api/v1/agent/chat", json={"conversation_id": "c1", "query": "  "})
    assert response.status_code == 422
