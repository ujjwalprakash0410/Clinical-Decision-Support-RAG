"""Orchestrates document ingestion: loaders -> chunking -> vector store."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.core.exceptions import DocumentLoadingError
from app.core.logging import get_logger
from app.loaders.base_loader import BaseLoader
from app.loaders.guideline_loader import build_loaders
from app.retrievers.hybrid_retriever import HybridRetriever
from app.retrievers.parent_retriever import ParentDocumentRetrieverService
from app.vectorstore.vector_store import VectorStore

logger = get_logger(__name__)


class IndexingService:
    """Coordinates the full ingestion pipeline.

    Design note: chunking uses `RecursiveCharacterTextSplitter` today,
    but is isolated behind `_split_documents` so it can be swapped for
    a semantic chunker later without touching loader or vector-store
    code (per the "support future replacement" requirement).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        settings: Settings,
        parent_retriever: ParentDocumentRetrieverService | None = None,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._settings = settings
        self._parent_retriever = parent_retriever
        self._hybrid_retriever = hybrid_retriever
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def index_sources(
        self, source_names: list[str], query_terms: list[str], rebuild: bool = False
    ) -> tuple[int, list[str]]:
        """Load, chunk, and index documents from the requested sources.

        Returns:
            (number of chunks indexed, list of sources that returned data)
        """
        loaders = build_loaders(source_names)
        raw_documents, successful_sources = self._load_all(loaders, query_terms)

        if not raw_documents:
            logger.info("indexing_service no documents loaded from any source")
            return 0, successful_sources

        chunks = self._split_documents(raw_documents)

        if rebuild:
            self._vector_store.rebuild(chunks)
        else:
            self._vector_store.add_documents(chunks)

        if self._parent_retriever is not None:
            self._parent_retriever.seed(raw_documents)
        if self._hybrid_retriever is not None:
            self._hybrid_retriever.build(chunks)

        logger.info(f"indexing_service indexed chunks={len(chunks)} sources={successful_sources}")
        return len(chunks), successful_sources

    def _load_all(
        self, loaders: list[BaseLoader], query_terms: list[str]
    ) -> tuple[list[Document], list[str]]:
        documents: list[Document] = []
        successful: list[str] = []
        for loader in loaders:
            try:
                loaded = loader.safe_load(query_terms)
                if loaded:
                    documents.extend(loaded)
                    successful.append(loader.source_name)
            except DocumentLoadingError as exc:
                logger.error(f"indexing_service loader={loader.source_name} error={exc}")
                continue
        return documents, successful

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk documents. Isolated for future semantic-chunking swap."""
        return self._splitter.split_documents(documents)
