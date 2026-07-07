"""Contextual retriever: reorders retrieved chunks to fight "lost in the middle".

LLMs pay more attention to content near the start/end of their
context window. `LongContextReorder` places the most relevant chunks
at the edges of the retrieved list, which measurably improves answer
quality for longer retrieved contexts. This is distinct from
`compression.py`, which shortens *content*; this module only reorders.
"""
from __future__ import annotations

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

from app.core.exceptions import RetrieverError
from app.vectorstore.vector_store import VectorStore


class ContextualRetriever:
    """Similarity search followed by long-context-aware reordering."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store
        self._reorderer = LongContextReorder()

    def retrieve(self, query: str, k: int = 6) -> list[Document]:
        try:
            documents = self._vector_store.similarity_search(query, k=k)
            return list(self._reorderer.transform_documents(documents))
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"Contextual retrieval failed: {exc}") from exc
