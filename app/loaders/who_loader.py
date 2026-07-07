"""Loader for World Health Organization guideline documents.

WHO does not expose a simple public search API for guideline full
text, so this loader reads locally-stored WHO guideline files (PDF or
text, dropped into `data/guidelines/who/` by an operator or a future
scraping job) and normalizes them into LangChain `Document` objects.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.core.constants import DocumentType
from app.loaders.base_loader import BaseLoader


class WHOLoader(BaseLoader):
    """Loads WHO guideline text files from a local directory."""

    source_name = "who"

    def __init__(self, guidelines_dir: str = "data/guidelines/who") -> None:
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
                "source": "WHO",
                "publication_year": publication_year,
                "page": None,
                "url": None,
                "document_type": DocumentType.WHO_GUIDELINE.value,
            },
        )
