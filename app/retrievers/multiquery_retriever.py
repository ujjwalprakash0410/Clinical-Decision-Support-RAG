"""Multi-query retrieval: expands a single query into several variants.

Uses LangChain's `MultiQueryRetriever`, which asks the LLM to generate
paraphrased versions of the user's question, retrieves for each, and
de-duplicates the union of results. This improves recall for clinical
questions that can be phrased many different ways.
"""
from __future__ import annotations

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from app.core.exceptions import RetrieverError
from app.core.logging import get_logger
from app.vectorstore.vector_store import VectorStore

logger = get_logger(__name__)


class MultiQueryRetrieverService:
    """Wraps LangChain's `MultiQueryRetriever` behind our retriever interface."""

    def __init__(self, vector_store: VectorStore, llm: BaseLanguageModel) -> None:
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 6})
        self._retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    def retrieve(self, query: str, k: int = 6) -> list[Document]:
        try:
            results = self._retriever.invoke(query)
            return results[:k]
        except Exception as exc:  # noqa: BLE001
            logger.error(f"multi_query_retriever failed error={exc}")
            raise RetrieverError(f"Multi-query retrieval failed: {exc}") from exc
