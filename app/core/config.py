"""
Centralized application configuration.

All tunable values (models, chunking parameters, API keys, retrieval
parameters, etc.) live here and are sourced from environment variables.
No module outside this file should read `os.environ` directly.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- General ---
    app_name: str = "Clinical Decision Support RAG"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # --- Groq LLM ---
    groq_api_key: str = Field(default="", description="API key for Groq LLM provider")
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_max_tokens: int = 1500
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3
    llm_retry_backoff_seconds: float = 1.5

    # --- Embeddings ---
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_normalize: bool = True

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Vector store ---
    vector_db_path: str = "data/vector_db"
    faiss_index_name: str = "clinical_index"

    # --- Retrieval ---
    retrieval_top_k: int = 6
    multiquery_enabled: bool = True
    compression_enabled: bool = True
    parent_child_enabled: bool = True
    metadata_filter_enabled: bool = True

    # --- Data ---
    guidelines_dir: str = "data/guidelines"

    # --- Evaluation ---
    ragas_dataset_dir: str = "app/evaluation/datasets"

    # --- Agent (LangGraph) ---
    agent_max_iterations: int = 2
    agent_rerank_top_k: int = 6
    agent_retrieve_k: int = 8
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- CORS ---
    allowed_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton `Settings` instance.

    Using `lru_cache` ensures environment variables are parsed once and
    the same immutable settings object is shared (and easily mocked in
    tests via dependency overrides).
    """
    return Settings()
