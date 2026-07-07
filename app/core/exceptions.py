"""
Application-specific exception hierarchy.

Using typed exceptions (instead of raising generic `Exception` or
`HTTPException` deep inside services) keeps business logic decoupled
from the web framework. The API layer's exception handlers translate
these into HTTP responses.
"""


class ClinicalRAGError(Exception):
    """Base class for all application-raised errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentLoadingError(ClinicalRAGError):
    """Raised when a source document/loader fails to fetch or parse data."""


class EmbeddingError(ClinicalRAGError):
    """Raised when the embedding model fails to encode text."""


class VectorStoreError(ClinicalRAGError):
    """Raised for FAISS index create/load/update/delete failures."""


class RetrieverError(ClinicalRAGError):
    """Raised when a retriever strategy fails to execute."""


class LLMGenerationError(ClinicalRAGError):
    """Raised when the Groq LLM call fails after retries or times out."""


class PromptFormattingError(ClinicalRAGError):
    """Raised when a prompt template cannot be rendered."""


class InvalidRequestError(ClinicalRAGError):
    """Raised for semantically invalid but schema-valid requests."""


class EvaluationError(ClinicalRAGError):
    """Raised when the evaluation pipeline fails to compute metrics."""
