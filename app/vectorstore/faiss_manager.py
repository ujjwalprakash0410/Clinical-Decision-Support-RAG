"""Low-level FAISS index lifecycle management (create/load/update/delete).

`FAISSManager` deals purely in LangChain `Document` + `Embeddings`
objects and file paths. Higher-level retrieval concerns live in
`vector_store.py` and `services/indexing_service.py`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)


class FAISSManager:
    """Owns the on-disk FAISS index lifecycle."""

    def __init__(self, index_dir: str, index_name: str, embeddings: Embeddings) -> None:
        self.index_dir = Path(index_dir)
        self.index_name = index_name
        self.embeddings = embeddings
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _index_file(self) -> Path:
        return self.index_dir / f"{self.index_name}.faiss"

    def exists(self) -> bool:
        """Whether a persisted index already exists on disk."""
        return self._index_file.exists()

    def create(self, documents: list[Document]) -> FAISS:
        """Create a brand-new index from scratch and persist it."""
        if not documents:
            raise VectorStoreError("Cannot create an index from zero documents")
        try:
            store = FAISS.from_documents(documents, self.embeddings)
            self._persist(store)
            logger.info(f"faiss create documents={len(documents)} index={self.index_name}")
            return store
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to create FAISS index: {exc}") from exc

    def load(self) -> FAISS:
        """Load a persisted index from disk."""
        if not self.exists():
            raise VectorStoreError(f"No index found at {self.index_dir}/{self.index_name}")
        try:
            store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"faiss load index={self.index_name}")
            return store
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to load FAISS index: {exc}") from exc

    def update(self, store: FAISS, documents: list[Document]) -> FAISS:
        """Add new documents to an existing index and persist it."""
        if not documents:
            return store
        try:
            store.add_documents(documents)
            self._persist(store)
            logger.info(f"faiss update documents_added={len(documents)} index={self.index_name}")
            return store
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to update FAISS index: {exc}") from exc

    def delete(self) -> None:
        """Delete the persisted index directory contents for this index."""
        try:
            for suffix in (".faiss", ".pkl"):
                file_path = self.index_dir / f"{self.index_name}{suffix}"
                if file_path.exists():
                    file_path.unlink()
            logger.info(f"faiss delete index={self.index_name}")
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to delete FAISS index: {exc}") from exc

    def wipe_all(self) -> None:
        """Danger: remove the entire index directory (used by rebuild)."""
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _persist(self, store: FAISS) -> None:
        store.save_local(str(self.index_dir), index_name=self.index_name)
