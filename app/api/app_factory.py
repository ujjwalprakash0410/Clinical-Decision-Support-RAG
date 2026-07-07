"""Builds and configures the FastAPI application instance."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.error_handler import register_exception_handlers
from app.api.middleware.logging_middleware import RequestLoggingMiddleware
from app.api.routes import agent, analyze, health, index, metrics, sources
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    """Application factory: assembles routers, middleware, and handlers."""
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Evidence-based Clinical Decision Support System. "
            "This system does not diagnose patients and does not replace "
            "professional medical judgment."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(sources.router, prefix=settings.api_prefix)
    app.include_router(metrics.router, prefix=settings.api_prefix)
    app.include_router(index.router, prefix=settings.api_prefix)
    app.include_router(analyze.router, prefix=settings.api_prefix)
    app.include_router(agent.router, prefix=settings.api_prefix)

    return app
