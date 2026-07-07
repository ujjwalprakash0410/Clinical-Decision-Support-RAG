"""Hybrid retrieval: combines dense (FAISS) and sparse (BM25) search.

Dense embeddings capture semantic similarity; BM25 captures exact
keyword/acronym matches (crucial in medicine — e.g. "MI", "CHF",
drug names). `EnsembleRetriever` merges both rankings via reciprocal
rank fusion.
"""
from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.core.exceptions import RetrieverError
from app.core.logging import get_logger
from app.vectorstore.vector_store import VectorStore

logger = get_logger(__name__)


class HybridRetriever:
    """Ensemble of FAISS dense retrieval and BM25 sparse retrieval.

    BM25 requires the full corpus in memory at construction time, so
    `build` must be called once with the same document set used to
    build the FAISS index (typically during indexing).
    """

    def __init__(self, vector_store: VectorStore, dense_weight: float = 0.5) -> None:
        self._vector_store = vector_store
        self._dense_weight = dense_weight
        self._ensemble: EnsembleRetriever | None = None

    def build(self, documents: list[Document], k: int = 6) -> None:
        if not documents:
            return
        try:
            bm25 = BM25Retriever.from_documents(documents)
            bm25.k = k
            dense = self._vector_store.as_retriever(search_kwargs={"k": k})
            self._ensemble = EnsembleRetriever(
                retrievers=[dense, bm25],
                weights=[self._dense_weight, 1 - self._dense_weight],
            )
            logger.info(f"hybrid_retriever built documents={len(documents)}")
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"Failed to build hybrid retriever: {exc}") from exc

    def retrieve(self, query: str, k: int = 6) -> list[Document]:
        if self._ensemble is None:
            raise RetrieverError("HybridRetriever.build() must be called before retrieve()")
        try:
            return self._ensemble.invoke(query)[:k]
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"Hybrid retrieval failed: {exc}") from exc
