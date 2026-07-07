"""Tests for `EmbeddingService`."""
from __future__ import annotations

import pytest

from app.core.exceptions import EmbeddingError
from app.embeddings.embedding_service import EmbeddingService


def test_embed_documents_returns_vector_per_text(fake_embeddings):
    service = EmbeddingService(fake_embeddings, model_name="fake-model")
    vectors = service.embed_documents(["hello world", "clinical evidence"])
    assert len(vectors) == 2
    assert all(isinstance(v, list) and len(v) == 16 for v in vectors)


def test_embed_documents_empty_input_returns_empty_list(fake_embeddings):
    service = EmbeddingService(fake_embeddings, model_name="fake-model")
    assert service.embed_documents([]) == []


def test_embed_query_returns_vector(fake_embeddings):
    service = EmbeddingService(fake_embeddings, model_name="fake-model")
    vector = service.embed_query("what is the treatment for afib")
    assert isinstance(vector, list)
    assert len(vector) == 16


def test_embed_query_raises_on_empty_string(fake_embeddings):
    service = EmbeddingService(fake_embeddings, model_name="fake-model")
    with pytest.raises(EmbeddingError):
        service.embed_query("   ")


def test_langchain_embeddings_property_exposes_backend(fake_embeddings):
    service = EmbeddingService(fake_embeddings, model_name="fake-model")
    assert service.langchain_embeddings is fake_embeddings
