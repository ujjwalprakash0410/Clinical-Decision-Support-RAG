"""Retrieval orchestration service used by the API layer."""
from __future__ import annotations

import time

from langchain_core.documents import Document

from app.core.constants import RetrieverType
from app.core.logging import get_logger
from app.retrievers.retriever_factory import RetrieverRegistry, retrieve

logger = get_logger(__name__)


class RetrievalService:
    """Facade the API/report services use instead of touching retrievers directly."""

    def __init__(self, registry: RetrieverRegistry) -> None:
        self._registry = registry

    def retrieve_documents(
        self,
        query: str,
        retriever_type: RetrieverType,
        k: int = 6,
        document_type: str | None = None,
    ) -> tuple[list[Document], float]:
        """Retrieve documents and report retrieval latency in milliseconds."""
        start = time.perf_counter()
        documents = retrieve(self._registry, retriever_type, query, k=k, document_type=document_type)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"retrieval_service retriever={retriever_type.value} "
            f"documents={len(documents)} latency_ms={latency_ms:.1f}"
        )
        return documents, latency_ms
