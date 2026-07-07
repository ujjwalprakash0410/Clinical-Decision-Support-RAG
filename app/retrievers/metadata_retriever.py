"""Similarity retriever with optional metadata filtering."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.exceptions import RetrieverError
from app.vectorstore.vector_store import VectorStore


class MetadataFilteredRetriever:
    """Retrieves by vector similarity, optionally scoped to a document type.

    FAISS's native metadata filtering support is limited, so filtering
    is applied as a post-retrieval step here: we over-fetch, then keep
    only documents matching the requested `document_type`.
    """

    def __init__(self, vector_store: VectorStore, over_fetch_factor: int = 3) -> None:
        self._vector_store = vector_store
        self._over_fetch_factor = over_fetch_factor

    def retrieve(
        self, query: str, k: int = 6, document_type: str | None = None
    ) -> list[Document]:
        try:
            fetch_k = k * self._over_fetch_factor if document_type else k
            candidates = self._vector_store.similarity_search(query, k=fetch_k)
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"Metadata retriever failed: {exc}") from exc

        if not document_type:
            return candidates[:k]

        filtered = [
            doc for doc in candidates if doc.metadata.get("document_type") == document_type
        ]
        return filtered[:k]

    def as_langchain_retriever(self, k: int = 6) -> VectorStoreRetriever:
        """Expose a plain LangChain retriever for composition with other retrievers."""
        return self._vector_store.as_retriever(search_kwargs={"k": k})
