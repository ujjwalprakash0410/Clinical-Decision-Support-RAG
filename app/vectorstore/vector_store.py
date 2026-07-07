"""High-level vector store facade used by the rest of the application.

Wraps `FAISSManager` so that services and retrievers depend on a
narrow interface (`similarity_search`, `as_retriever`, `document_count`)
rather than FAISS/LangChain internals directly. This is also the seam
where a future swap to a different vector DB (pgvector, Qdrant, etc.)
would happen without touching callers.
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.vectorstore.faiss_manager import FAISSManager

logger = get_logger(__name__)


class VectorStore:
    """Facade over a `FAISSManager`-managed index."""

    def __init__(self, manager: FAISSManager, embeddings: Embeddings) -> None:
        self._manager = manager
        self._embeddings = embeddings
        self._store: FAISS | None = None

    def initialize(self) -> None:
        """Load the index from disk if present; otherwise leave it empty."""
        if self._manager.exists():
            self._store = self._manager.load()
        else:
            logger.info("vector_store no persisted index found; awaiting first /index call")

    def create(self, documents: list[Document]) -> None:
        self._store = self._manager.create(documents)

    def add_documents(self, documents: list[Document]) -> None:
        if self._store is None:
            self.create(documents)
            return
        self._store = self._manager.update(self._store, documents)

    def rebuild(self, documents: list[Document]) -> None:
        self._manager.wipe_all()
        self._store = None
        self.create(documents)

    def similarity_search(self, query: str, k: int = 6) -> list[Document]:
        self._ensure_ready()
        assert self._store is not None
        return self._store.similarity_search(query, k=k)

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        self._ensure_ready()
        assert self._store is not None
        return self._store.as_retriever(**kwargs)

    @property
    def document_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    def _ensure_ready(self) -> None:
        if self._store is None:
            raise VectorStoreError(
                "Vector store has not been indexed yet. Call POST /index first."
            )
