"""Top-level orchestration service that produces a `ClinicalReport`.

This is the only service the `/analyze` route depends on. It composes
`RetrievalService`, prompt assembly, `GroqService`, response parsing,
and `CitationService` — no business logic lives in the API route.
"""
from __future__ import annotations

import time

from app.core.constants import RetrieverType
from app.core.logging import get_logger
from app.llm.groq_service import GroqService
from app.llm.response_formatter import parse_clinical_report
from app.models.response_models import AnalyzeResponse, ClinicalReport
from app.prompts.clinical_prompt import build_clinical_report_messages
from app.services.citation_service import CitationService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class ReportService:
    """Generates evidence-backed structured clinical reports."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        groq_service: GroqService,
        citation_service: CitationService,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._groq_service = groq_service
        self._citation_service = citation_service

    def generate_report(
        self,
        query: str,
        retriever_type: RetrieverType,
        k: int = 6,
        document_type: str | None = None,
    ) -> AnalyzeResponse:
        """Run the full retrieve -> prompt -> generate -> parse pipeline."""
        start = time.perf_counter()

        documents, _ = self._retrieval_service.retrieve_documents(
            query, retriever_type, k=k, document_type=document_type
        )
        messages = build_clinical_report_messages(query, documents)
        raw_output = self._groq_service.generate(messages, json_mode=True)
        report = parse_clinical_report(raw_output)
        report = self._attach_references(report, documents)

        total_latency_ms = (time.perf_counter() - start) * 1000
        logger.info(f"report_service query_len={len(query)} latency_ms={total_latency_ms:.1f}")

        return AnalyzeResponse(
            query=query,
            report=report,
            retriever_used=retriever_type.value,
            documents_retrieved=len(documents),
            latency_ms=round(total_latency_ms, 2),
        )

    def _attach_references(self, report: ClinicalReport, documents) -> ClinicalReport:
        """Prefer our own citation-built references over the LLM's, for accuracy."""
        references = self._citation_service.build_references(documents)
        if references:
            report = report.model_copy(update={"references": references})
        return report
