"""Embedding service: a thin, swappable wrapper around an embedding model."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Encodes text into vectors using an injected `Embeddings` backend.

    The service never constructs a concrete embedding model itself —
    that responsibility belongs to `embedding_factory.py`. This keeps
    `EmbeddingService` swappable and easily mockable in tests.
    """

    def __init__(self, embeddings: Embeddings, model_name: str) -> None:
        self._embeddings = embeddings
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents for indexing."""
        if not texts:
            return []
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"embedding_model={self.model_name} embed_documents failed error={exc}")
            raise EmbeddingError(f"Failed to embed {len(texts)} documents") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed an empty query")
        try:
            return self._embeddings.embed_query(text)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"embedding_model={self.model_name} embed_query failed error={exc}")
            raise EmbeddingError("Failed to embed query") from exc

    @property
    def langchain_embeddings(self) -> Embeddings:
        """Expose the underlying LangChain `Embeddings` object.

        Needed because FAISS and several retrievers expect a raw
        `Embeddings` instance rather than our service wrapper.
        """
        return self._embeddings
