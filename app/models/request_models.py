"""Request payload models for the public API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.core.constants import MAX_QUERY_LENGTH, RetrieverType


class AnalyzeRequest(BaseModel):
    """Payload for POST /analyze — a clinical question to be researched."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=MAX_QUERY_LENGTH,
        description="Clinical question, e.g. 'Evidence for anticoagulation in AFib with CHA2DS2-VASc 2'",
    )
    retriever_type: RetrieverType = RetrieverType.MULTI_QUERY
    top_k: int = Field(default=6, ge=1, le=20)
    document_type_filter: str | None = Field(
        default=None, description="Optional metadata filter, e.g. 'who_guideline'"
    )
    stream: bool = False

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty or whitespace-only")
        return cleaned


class IndexRequest(BaseModel):
    """Payload for POST /index — trigger (re)ingestion of source documents."""

    sources: list[str] = Field(
        default_factory=lambda: ["pubmed", "who", "cdc", "guideline"],
        description="Which loaders to run",
    )
    rebuild: bool = Field(
        default=False, description="If true, drop and recreate the index instead of updating it"
    )
    query_terms: list[str] = Field(
        default_factory=list,
        description="Search terms passed to online loaders (e.g. PubMed) when applicable",
    )


class AgentChatRequest(BaseModel):
    """Payload for POST /agent/chat — a conversational, agentic clinical query."""

    conversation_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Stable identifier for this conversation; reuse it across turns for memory",
    )
    query: str = Field(..., min_length=3, max_length=MAX_QUERY_LENGTH)

    @field_validator("query")
    @classmethod
    def strip_and_validate_agent_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty or whitespace-only")
        return cleaned
