"""Builds citation/reference lists from retrieved documents."""
from __future__ import annotations

from langchain_core.documents import Document

from app.models.response_models import Reference


class CitationService:
    """Converts retrieved LangChain documents into `Reference` objects."""

    def build_references(self, documents: list[Document]) -> list[Reference]:
        """Build a de-duplicated, numbered reference list.

        Args:
            documents: Retrieved documents, in relevance order.

        Returns:
            One `Reference` per unique (title, source) pair, labeled
            "[1]", "[2]", ... matching the numbering used in the prompt
            context so the LLM's citations line up.
        """
        references: list[Reference] = []
        seen: set[tuple[str, str]] = set()

        for index, doc in enumerate(documents, start=1):
            title = doc.metadata.get("title", "Untitled")
            source = doc.metadata.get("source", "Unknown")
            key = (title, source)
            if key in seen:
                continue
            seen.add(key)
            references.append(
                Reference(
                    label=f"[{index}]",
                    title=title,
                    source=source,
                    url=doc.metadata.get("url"),
                    publication_year=doc.metadata.get("publication_year"),
                )
            )
        return references
