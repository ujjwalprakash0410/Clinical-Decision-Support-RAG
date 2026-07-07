"""Factory for constructing `EmbeddingService` instances.

Isolating construction here means swapping `BAAI/bge-small-en-v1.5`
for another HuggingFace model, or an entirely different provider
(OpenAI, Cohere, etc.), only requires changes in this one file.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings, get_settings
from app.embeddings.embedding_service import EmbeddingService


def create_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    """Build an `EmbeddingService` from application settings.

    Args:
        settings: Optional settings override, primarily for tests.

    Returns:
        A ready-to-use `EmbeddingService`.
    """
    settings = settings or get_settings()

    backend = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": settings.embedding_normalize},
    )

    return EmbeddingService(embeddings=backend, model_name=settings.embedding_model_name)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return a cached, process-wide `EmbeddingService` singleton.

    Loading `BAAI/bge-small-en-v1.5` is expensive; we only want to do
    it once per process. FastAPI dependencies should call this rather
    than `create_embedding_service` directly.
    """
    return create_embedding_service()
