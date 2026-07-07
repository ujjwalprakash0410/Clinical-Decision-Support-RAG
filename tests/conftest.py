"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic, fast fake embeddings so tests don't load real models."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        # Simple deterministic hash-based vector; enough for FAISS to index.
        seed = sum(ord(c) for c in text) or 1
        return [((seed * (i + 1)) % 97) / 97 for i in range(16)]


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Direct oral anticoagulants are preferred in atrial fibrillation.",
            metadata={
                "title": "AFib Guidelines",
                "source": "WHO",
                "publication_year": 2023,
                "document_type": "who_guideline",
                "url": None,
            },
        ),
        Document(
            page_content="Post-exposure prophylaxis should start within hours of exposure.",
            metadata={
                "title": "PEP Guidance",
                "source": "CDC",
                "publication_year": 2022,
                "document_type": "cdc_guideline",
                "url": None,
            },
        ),
    ]
