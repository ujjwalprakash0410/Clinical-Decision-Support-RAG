"""Dependency-injection wiring for FastAPI routes.

All object construction happens here (or in the cached factories it
calls into), so routes and services never instantiate their own
collaborators — enabling easy test overrides via
`app.dependency_overrides`.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from langchain_groq import ChatGroq

from app.agents.clinical_agent import build_agent_graph
from app.core.config import Settings, get_settings
from app.embeddings.embedding_factory import get_embedding_service
from app.embeddings.embedding_service import EmbeddingService
from app.llm.groq_service import GroqService
from app.loaders.pubmed_loader import PubMedLoader
from app.retrievers.hybrid_retriever import HybridRetriever
from app.retrievers.parent_retriever import ParentDocumentRetrieverService
from app.retrievers.reranker import CrossEncoderReranker, default_cross_encoder_score_fn
from app.retrievers.retriever_factory import RetrieverRegistry
from app.services.agent_service import AgentService
from app.services.citation_service import CitationService
from app.services.indexing_service import IndexingService
from app.services.report_service import ReportService
from app.services.retrieval_service import RetrievalService
from app.vectorstore.faiss_manager import FAISSManager
from app.vectorstore.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    embedding_service = get_embedding_service()
    manager = FAISSManager(
        index_dir=settings.vector_db_path,
        index_name=settings.faiss_index_name,
        embeddings=embedding_service.langchain_embeddings,
    )
    store = VectorStore(manager, embedding_service.langchain_embeddings)
    store.initialize()
    return store


@lru_cache
def get_chat_llm() -> ChatGroq:
    """LangChain-compatible chat model, used only by retrievers that need it
    (MultiQueryRetriever, ContextualCompressionRetriever query expansion)."""
    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.llm_temperature,
    )


@lru_cache
def get_parent_retriever_service() -> ParentDocumentRetrieverService:
    embedding_service = get_embedding_service()
    return ParentDocumentRetrieverService(embedding_service.langchain_embeddings)


@lru_cache
def get_hybrid_retriever_service() -> HybridRetriever:
    return HybridRetriever(get_vector_store())


@lru_cache
def get_retriever_registry() -> RetrieverRegistry:
    return RetrieverRegistry(
        vector_store=get_vector_store(),
        llm=get_chat_llm(),
        parent=get_parent_retriever_service(),
        hybrid=get_hybrid_retriever_service(),
    )


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(get_retriever_registry())


@lru_cache
def get_groq_service() -> GroqService:
    return GroqService()


@lru_cache
def get_citation_service() -> CitationService:
    return CitationService()


@lru_cache
def get_report_service() -> ReportService:
    return ReportService(
        retrieval_service=get_retrieval_service(),
        groq_service=get_groq_service(),
        citation_service=get_citation_service(),
    )


@lru_cache
def get_indexing_service() -> IndexingService:
    settings = get_settings()
    return IndexingService(
        vector_store=get_vector_store(),
        settings=settings,
        parent_retriever=get_parent_retriever_service(),
        hybrid_retriever=get_hybrid_retriever_service(),
    )


@lru_cache
def get_pubmed_loader() -> PubMedLoader:
    return PubMedLoader()


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    settings = get_settings()
    return CrossEncoderReranker(default_cross_encoder_score_fn(settings.reranker_model_name))


@lru_cache
def get_agent_graph():
    settings = get_settings()
    return build_agent_graph(
        groq_service=get_groq_service(),
        retrieval_service=get_retrieval_service(),
        citation_service=get_citation_service(),
        vector_store=get_vector_store(),
        pubmed_loader=get_pubmed_loader(),
        reranker=get_reranker(),
        settings=settings,
    )


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService(get_agent_graph())


def settings_dependency() -> Settings:
    return get_settings()


SettingsDep = Depends(settings_dependency)
EmbeddingServiceDep = Depends(get_embedding_service)
VectorStoreDep = Depends(get_vector_store)
ReportServiceDep = Depends(get_report_service)
IndexingServiceDep = Depends(get_indexing_service)
AgentServiceDep = Depends(get_agent_service)
