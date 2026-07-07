"""Response payload models returned by the public API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Reference(BaseModel):
    """A single citation backing the clinical report."""

    label: str
    title: str
    source: str
    url: str | None = None
    publication_year: int | None = None

    @field_validator("publication_year", mode="before")
    @classmethod
    def blank_string_to_none(cls, value):
        """The LLM sometimes returns "" instead of null for unknown years."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if stripped.isdigit():
                return int(stripped)
            return None
        return value


class ClinicalReport(BaseModel):
    """Structured, evidence-backed clinical decision support report.

    This is the core output contract of the system. Every field is
    required so downstream UIs can render a consistent layout, and the
    disclaimer field cannot be omitted by the LLM formatting step.
    """

    summary: str
    possible_conditions: list[str] = Field(default_factory=list)
    suggested_diagnostic_tests: list[str] = Field(default_factory=list)
    red_flag_symptoms: list[str] = Field(default_factory=list)
    evidence_summary: str
    clinical_guidelines: list[str] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    confidence: str = Field(description="One of: low, moderate, high")
    limitations: str
    disclaimer: str


class AnalyzeResponse(BaseModel):
    """Full response envelope for POST /analyze."""

    query: str
    report: ClinicalReport
    retriever_used: str
    documents_retrieved: int
    latency_ms: float


class IndexResponse(BaseModel):
    """Response for POST /index."""

    status: str
    documents_indexed: int
    sources_used: list[str]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    app_name: str
    version: str


class SourcesResponse(BaseModel):
    """Response for GET /sources."""

    available_sources: list[str]
    document_types: list[str]


class MetricsResponse(BaseModel):
    """Response for GET /metrics — lightweight runtime metrics snapshot."""

    total_requests: int
    average_latency_ms: float
    index_document_count: int


class AgentChatResponse(BaseModel):
    """Response for POST /agent/chat."""

    conversation_id: str
    query: str
    report: ClinicalReport
    retriever_used: str
    documents_retrieved: int
    used_live_pubmed_search: bool
    iterations: int
    latency_ms: float
