"""Parent-document retrieval: search small chunks, return larger parent context.

Small chunks embed more precisely, but LLMs generate better answers
with more surrounding context. `ParentDocumentRetriever` indexes small
child chunks for search while returning the larger parent document (or
a bigger parent chunk) they belong to.
"""
from __future__ import annotations

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.exceptions import RetrieverError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ParentDocumentRetrieverService:
    """Wraps LangChain's `ParentDocumentRetriever`.

    Note: this retriever owns a *separate* in-memory child vector
    index (required by LangChain's implementation) rather than reusing
    the main persisted FAISS index. It is seeded from the same source
    documents at indexing time. This keeps the primary FAISS index
    simple while still offering parent/child retrieval as a strategy.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
    ) -> None:
        self._docstore = InMemoryStore()
        self._child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)
        self._parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)
        self._vectorstore = FAISS.from_texts(["__init__placeholder__"], embeddings)
        self._retriever = ParentDocumentRetriever(
            vectorstore=self._vectorstore,
            docstore=self._docstore,
            child_splitter=self._child_splitter,
            parent_splitter=self._parent_splitter,
        )
        self._seeded = False

    def seed(self, documents: list[Document]) -> None:
        """Populate the parent/child stores from the full source document set."""
        if not documents:
            return
        try:
            self._retriever.add_documents(documents)
            self._seeded = True
            logger.info(f"parent_retriever seeded documents={len(documents)}")
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"Failed to seed parent document retriever: {exc}") from exc

    def retrieve(self, query: str, k: int = 6) -> list[Document]:
        if not self._seeded:
            raise RetrieverError(
                "ParentDocumentRetriever has not been seeded yet; call seed() during indexing"
            )
        try:
            results = self._retriever.invoke(query)
            return results[:k]
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"Parent document retrieval failed: {exc}") from exc
