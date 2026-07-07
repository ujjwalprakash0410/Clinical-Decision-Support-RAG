"""Domain models describing ingested clinical documents and their metadata."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import DocumentType


class DocumentMetadata(BaseModel):
    """Metadata attached to every ingested chunk.

    This is intentionally rich because citation quality is a hard
    requirement for a clinical-facing system: every claim must be
    traceable back to a title, source, year, and URL.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    source: str = Field(description="Publisher / registry, e.g. 'PubMed', 'WHO', 'CDC'")
    publication_year: int | None = None
    page: int | None = None
    url: str | None = None
    document_type: DocumentType = DocumentType.OTHER
    authors: list[str] = Field(default_factory=list)
    ingested_at: date = Field(default_factory=date.today)


class ClinicalDocument(BaseModel):
    """A single chunk of source text plus its metadata.

    This is the internal representation used before conversion to a
    LangChain `Document`; keeping our own model avoids coupling the
    rest of the codebase directly to LangChain's schema.
    """

    content: str
    metadata: DocumentMetadata
    chunk_id: str | None = None

    def to_citation_label(self) -> str:
        """Build a short human-readable citation label, e.g. '(WHO, 2023)'."""
        year = self.metadata.publication_year or "n.d."
        return f"({self.metadata.source}, {year})"
