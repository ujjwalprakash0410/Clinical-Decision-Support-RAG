"""Factory that builds the requested retriever strategy on demand.

This is the single seam through which `RetrievalService` accesses any
retriever. Adding a new strategy means: implement it in its own
module, register it in `_BUILDERS`, add an enum value to
`RetrieverType` — nothing else in the codebase changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from app.core.constants import RetrieverType
from app.core.exceptions import RetrieverError
from app.retrievers.compression import ContextualCompressionRetrieverService
from app.retrievers.contextual_retriever import ContextualRetriever
from app.retrievers.hybrid_retriever import HybridRetriever
from app.retrievers.metadata_retriever import MetadataFilteredRetriever
from app.retrievers.multiquery_retriever import MultiQueryRetrieverService
from app.retrievers.parent_retriever import ParentDocumentRetrieverService
from app.vectorstore.vector_store import VectorStore


@dataclass
class RetrieverRegistry:
    """Holds lazily-built retriever instances shared across requests."""

    vector_store: VectorStore
    llm: BaseLanguageModel
    _metadata: MetadataFilteredRetriever = field(init=False, default=None)
    _multiquery: MultiQueryRetrieverService = field(init=False, default=None)
    _contextual: ContextualRetriever = field(init=False, default=None)
    _compression: ContextualCompressionRetrieverService = field(init=False, default=None)
    parent: ParentDocumentRetrieverService | None = None
    hybrid: HybridRetriever | None = None

    def metadata(self) -> MetadataFilteredRetriever:
        if self._metadata is None:
            self._metadata = MetadataFilteredRetriever(self.vector_store)
        return self._metadata

    def multiquery(self) -> MultiQueryRetrieverService:
        if self._multiquery is None:
            self._multiquery = MultiQueryRetrieverService(self.vector_store, self.llm)
        return self._multiquery

    def contextual(self) -> ContextualRetriever:
        if self._contextual is None:
            self._contextual = ContextualRetriever(self.vector_store)
        return self._contextual

    def compression(self) -> ContextualCompressionRetrieverService:
        if self._compression is None:
            base = self.vector_store.as_retriever(search_kwargs={"k": 6})
            self._compression = ContextualCompressionRetrieverService(base, self.llm)
        return self._compression


RetrieveFn = Callable[[RetrieverRegistry, str, int, str | None], list[Document]]


def _retrieve_similarity(
    registry: RetrieverRegistry, query: str, k: int, document_type: str | None
) -> list[Document]:
    return registry.metadata().retrieve(query, k=k, document_type=None)


def _retrieve_metadata_filtered(
    registry: RetrieverRegistry, query: str, k: int, document_type: str | None
) -> list[Document]:
    return registry.metadata().retrieve(query, k=k, document_type=document_type)


def _retrieve_multiquery(
    registry: RetrieverRegistry, query: str, k: int, document_type: str | None
) -> list[Document]:
    return registry.multiquery().retrieve(query, k=k)


def _retrieve_contextual(
    registry: RetrieverRegistry, query: str, k: int, document_type: str | None
) -> list[Document]:
    return registry.contextual().retrieve(query, k=k)


def _retrieve_compression(
    registry: RetrieverRegistry, query: str, k: int, document_type: str | None
) -> list[Document]:
    return registry.compression().retrieve(query, k=k)


def _retrieve_parent(
    registry: RetrieverRegistry, query: str, k: int, document_type: str | None
) -> list[Document]:
    if registry.parent is None:
        raise RetrieverError("Parent document retriever was not initialized during indexing")
    return registry.parent.retrieve(query, k=k)


def _retrieve_hybrid(
    registry: RetrieverRegistry, query: str, k: int, document_type: str | None
) -> list[Document]:
    if registry.hybrid is None:
        raise RetrieverError("Hybrid retriever was not initialized during indexing")
    return registry.hybrid.retrieve(query, k=k)


_BUILDERS: dict[RetrieverType, RetrieveFn] = {
    RetrieverType.SIMILARITY: _retrieve_similarity,
    RetrieverType.METADATA_FILTERED: _retrieve_metadata_filtered,
    RetrieverType.MULTI_QUERY: _retrieve_multiquery,
    RetrieverType.CONTEXTUAL_COMPRESSION: _retrieve_compression,
    RetrieverType.PARENT_DOCUMENT: _retrieve_parent,
    RetrieverType.HYBRID: _retrieve_hybrid,
}


def retrieve(
    registry: RetrieverRegistry,
    retriever_type: RetrieverType,
    query: str,
    k: int = 6,
    document_type: str | None = None,
) -> list[Document]:
    """Dispatch a retrieval call to the requested strategy.

    Raises:
        RetrieverError: If the retriever type is unknown or the
            underlying strategy fails.
    """
    builder = _BUILDERS.get(retriever_type)
    if builder is None:
        raise RetrieverError(f"Unknown retriever type: {retriever_type}")
    return builder(registry, query, k, document_type)
