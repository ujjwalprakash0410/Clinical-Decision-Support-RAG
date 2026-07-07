"""Abstract interface that every source-document loader must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document

from app.core.exceptions import DocumentLoadingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseLoader(ABC):
    """Common contract for all document loaders.

    Concrete loaders (PubMed, WHO, CDC, local guideline files) fetch
    raw content from their respective source and convert it into
    LangChain `Document` objects with normalized metadata, so the rest
    of the pipeline (chunking, embedding, indexing) never needs to know
    where a document came from.
    """

    source_name: str = "unknown"

    @abstractmethod
    def load(self, query_terms: list[str] | None = None) -> list[Document]:
        """Fetch and return raw documents (pre-chunking).

        Args:
            query_terms: Optional search terms for loaders that hit a
                live API (e.g. PubMed). Ignored by static/local loaders.

        Returns:
            A list of LangChain `Document` objects with populated
            `page_content` and `metadata`.

        Raises:
            DocumentLoadingError: If the source cannot be reached or
                parsed.
        """
        raise NotImplementedError

    def safe_load(self, query_terms: list[str] | None = None) -> list[Document]:
        """Wrap `load` with consistent logging and error translation."""
        try:
            documents = self.load(query_terms)
            logger.info(f"loader={self.source_name} documents_loaded={len(documents)}")
            return documents
        except DocumentLoadingError:
            raise
        except Exception as exc:  # noqa: BLE001 - translate any failure
            logger.error(f"loader={self.source_name} failed error={exc}")
            raise DocumentLoadingError(
                f"Failed to load documents from {self.source_name}: {exc}"
            ) from exc
