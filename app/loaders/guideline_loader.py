"""Generic loader for arbitrary local clinical guideline documents.

Supports plain-text and PDF files placed under `data/guidelines/`.
This is the catch-all loader for society guidelines (e.g. ACC/AHA,
NICE) that don't warrant a dedicated loader class.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from app.core.constants import DocumentType
from app.core.exceptions import DocumentLoadingError
from app.loaders.base_loader import BaseLoader
from app.loaders.cdc_loader import CDCLoader
from app.loaders.pubmed_loader import PubMedLoader
from app.loaders.who_loader import WHOLoader


class GuidelineLoader(BaseLoader):
    """Loads generic clinical guideline files (.txt and .pdf)."""

    source_name = "guideline"

    def __init__(self, guidelines_dir: str = "data/guidelines/general") -> None:
        self.guidelines_dir = Path(guidelines_dir)

    def load(self, query_terms: list[str] | None = None) -> list[Document]:
        if not self.guidelines_dir.exists():
            return []

        documents: list[Document] = []
        for file_path in sorted(self.guidelines_dir.iterdir()):
            if file_path.suffix.lower() == ".txt":
                documents.append(self._text_to_document(file_path))
            elif file_path.suffix.lower() == ".pdf":
                documents.extend(self._pdf_to_documents(file_path))
        return documents

    def _text_to_document(self, file_path: Path) -> Document:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return self._build_document(text, file_path, page=None)

    def _pdf_to_documents(self, file_path: Path) -> list[Document]:
        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:  # noqa: BLE001
            raise DocumentLoadingError(f"Failed to parse PDF {file_path}: {exc}") from exc

        return [
            self._build_document(page.extract_text() or "", file_path, page=index + 1)
            for index, page in enumerate(reader.pages)
        ]

    def _build_document(self, text: str, file_path: Path, page: int | None) -> Document:
        return Document(
            page_content=text,
            metadata={
                "title": file_path.stem.replace("_", " ").title(),
                "source": "Clinical Guideline",
                "publication_year": None,
                "page": page,
                "url": None,
                "document_type": DocumentType.CLINICAL_GUIDELINE.value,
            },
        )


LOADER_REGISTRY: dict[str, type[BaseLoader]] = {
    "pubmed": PubMedLoader,
    "who": WHOLoader,
    "cdc": CDCLoader,
    "guideline": GuidelineLoader,
}


def build_loaders(source_names: list[str]) -> list[BaseLoader]:
    """Instantiate loaders for the requested source names.

    Args:
        source_names: Keys into `LOADER_REGISTRY`, e.g. ["pubmed", "who"].

    Returns:
        Instantiated loader objects, skipping unknown names.
    """
    loaders: list[BaseLoader] = []
    for name in source_names:
        loader_cls = LOADER_REGISTRY.get(name.lower())
        if loader_cls is not None:
            loaders.append(loader_cls())
    return loaders
