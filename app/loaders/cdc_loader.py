"""Loader for Centers for Disease Control and Prevention (CDC) guidance.

Mirrors `WHOLoader`'s approach of reading a local corpus, since CDC
guidance pages are HTML without a stable structured API. A future
enhancement could add a scraping loader implementing the same
`BaseLoader` interface without touching downstream code.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.core.constants import DocumentType
from app.loaders.base_loader import BaseLoader


class CDCLoader(BaseLoader):
    """Loads CDC guideline text files from a local directory."""

    source_name = "cdc"

    def __init__(self, guidelines_dir: str = "data/guidelines/cdc") -> None:
        self.guidelines_dir = Path(guidelines_dir)

    def load(self, query_terms: list[str] | None = None) -> list[Document]:
        if not self.guidelines_dir.exists():
            return []

        documents: list[Document] = []
        for file_path in sorted(self.guidelines_dir.glob("*.txt")):
            documents.append(self._file_to_document(file_path))
        return documents

    def _file_to_document(self, file_path: Path) -> Document:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        title, _, year = file_path.stem.rpartition("_")
        publication_year = int(year) if year.isdigit() else None

        return Document(
            page_content=text,
            metadata={
                "title": title.replace("_", " ").title() or file_path.stem,
                "source": "CDC",
                "publication_year": publication_year,
                "page": None,
                "url": None,
                "document_type": DocumentType.CDC_GUIDELINE.value,
            },
        )
