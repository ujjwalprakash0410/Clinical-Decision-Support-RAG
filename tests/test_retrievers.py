"""Tests for retriever strategies."""
from __future__ import annotations

import pytest

from app.core.exceptions import VectorStoreError
from app.retrievers.metadata_retriever import MetadataFilteredRetriever
from app.vectorstore.faiss_manager import FAISSManager
from app.vectorstore.vector_store import VectorStore


@pytest.fixture
def populated_vector_store(tmp_path, fake_embeddings, sample_documents) -> VectorStore:
    manager = FAISSManager(
        index_dir=str(tmp_path / "vector_db"), index_name="test_index", embeddings=fake_embeddings
    )
    store = VectorStore(manager, fake_embeddings)
    store.create(sample_documents)
    return store


def test_similarity_search_returns_documents(populated_vector_store):
    results = populated_vector_store.similarity_search("anticoagulation", k=2)
    assert len(results) == 2


def test_metadata_filtered_retriever_filters_by_document_type(populated_vector_store):
    retriever = MetadataFilteredRetriever(populated_vector_store)
    results = retriever.retrieve("guidance", k=5, document_type="cdc_guideline")
    assert all(doc.metadata["document_type"] == "cdc_guideline" for doc in results)


def test_metadata_filtered_retriever_no_filter_returns_up_to_k(populated_vector_store):
    retriever = MetadataFilteredRetriever(populated_vector_store)
    results = retriever.retrieve("guidance", k=1)
    assert len(results) == 1


def test_vector_store_raises_before_indexing(tmp_path, fake_embeddings):
    manager = FAISSManager(
        index_dir=str(tmp_path / "empty_db"), index_name="empty_index", embeddings=fake_embeddings
    )
    store = VectorStore(manager, fake_embeddings)
    store.initialize()
    with pytest.raises(VectorStoreError):
        store.similarity_search("anything")


def test_vector_store_document_count_after_create(populated_vector_store):
    assert populated_vector_store.document_count == 2
