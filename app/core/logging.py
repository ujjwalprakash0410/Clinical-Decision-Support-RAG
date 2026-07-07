"""
Structured logging configuration for the application.

Uses Python's standard `logging` module with a JSON-ish formatter so
logs are easy to ship to an aggregator (CloudWatch, ELK, etc.) later.
"""
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured, single-line key=value output."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return " ".join(f"{k}={v!r}" for k, v in payload.items())


def configure_logging() -> None:
    """Configure the root logger once for the whole application."""
    settings = get_settings()
    root_logger = logging.getLogger()

    if root_logger.handlers:
        # Already configured (e.g. re-imported in tests).
        root_logger.setLevel(settings.log_level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root_logger.setLevel(settings.log_level)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
