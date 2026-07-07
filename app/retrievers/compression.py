"""Contextual compression: trims retrieved chunks down to the relevant span.

Wraps a base retriever with an LLM-driven extractor so the final
context passed to the report-generation prompt contains only the
sentences relevant to the query, reducing noise and token usage.
"""
from __future__ import annotations

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from app.core.exceptions import RetrieverError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ContextualCompressionRetrieverService:
    """Wraps LangChain's `ContextualCompressionRetriever` with an LLM extractor."""

    def __init__(self, base_retriever: BaseRetriever, llm: BaseLanguageModel) -> None:
        compressor = LLMChainExtractor.from_llm(llm)
        self._retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

    def retrieve(self, query: str, k: int = 6) -> list[Document]:
        try:
            results = self._retriever.invoke(query)
            return results[:k]
        except Exception as exc:  # noqa: BLE001
            logger.error(f"compression_retriever failed error={exc}")
            raise RetrieverError(f"Contextual compression retrieval failed: {exc}") from exc
