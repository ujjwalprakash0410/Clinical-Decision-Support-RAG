"""Cross-encoder reranking: reorders retrieved documents by direct query-document relevance.

Unlike embedding similarity (which compares a query vector to a document
vector independently), a cross-encoder scores the (query, document) pair
jointly, which is typically far more precise at the cost of being too slow
to run over an entire corpus. The standard pattern — used here — is to let
a cheaper retriever fetch a candidate set, then rerank only that small set
with the cross-encoder.
"""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document

from app.core.exceptions import RetrieverError
from app.core.logging import get_logger

logger = get_logger(__name__)

ScoreFn = Callable[[list[tuple[str, str]]], list[float]]


class CrossEncoderReranker:
    """Reorders documents using an injected cross-encoder scoring function.

    The scorer is injected as a plain callable rather than constructed
    internally, following this project's dependency-injection convention:
    production wiring supplies a real HuggingFace cross-encoder (see
    `default_cross_encoder_score_fn`), while tests inject a deterministic
    fake with no model download or GPU required.
    """

    def __init__(self, score_fn: ScoreFn) -> None:
        self._score_fn = score_fn

    def rerank(self, query: str, documents: list[Document], top_k: int = 6) -> list[Document]:
        """Score each document against the query and return the top-k, best-first."""
        if not documents:
            return []
        try:
            pairs = [(query, doc.page_content[:512]) for doc in documents]
            scores = self._score_fn(pairs)
            ranked = sorted(zip(documents, scores, strict=True), key=lambda pair: pair[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as exc:  # noqa: BLE001
            logger.error(f"cross_encoder_reranker failed error={exc}")
            raise RetrieverError(f"Cross-encoder reranking failed: {exc}") from exc


def default_cross_encoder_score_fn(model_name: str) -> ScoreFn:
    """Build a real sentence-transformers CrossEncoder-backed scoring function.

    Imported lazily so this module can be imported (and tested with a fake
    scorer) in environments without the model cached or without network
    access to download it on first use.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    def score_fn(pairs: list[tuple[str, str]]) -> list[float]:
        return [float(score) for score in model.predict(pairs)]

    return score_fn
