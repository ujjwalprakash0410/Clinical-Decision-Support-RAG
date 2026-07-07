"""Registers global exception handlers translating domain errors to HTTP responses."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    ClinicalRAGError,
    DocumentLoadingError,
    EmbeddingError,
    InvalidRequestError,
    LLMGenerationError,
    RetrieverError,
    VectorStoreError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

_STATUS_MAP: dict[type[ClinicalRAGError], int] = {
    InvalidRequestError: 400,
    DocumentLoadingError: 502,
    EmbeddingError: 500,
    VectorStoreError: 503,
    RetrieverError: 502,
    LLMGenerationError: 502,
}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so domain exceptions never leak as raw 500s."""

    @app.exception_handler(ClinicalRAGError)
    async def handle_clinical_rag_error(request: Request, exc: ClinicalRAGError) -> JSONResponse:
        status_code = _STATUS_MAP.get(type(exc), 500)
        logger.error(f"path={request.url.path} error_type={type(exc).__name__} message={exc.message}")
        return JSONResponse(
            status_code=status_code,
            content={"error": type(exc).__name__, "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"path={request.url.path} unhandled_error={exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "InternalServerError", "message": "An unexpected error occurred."},
        )
