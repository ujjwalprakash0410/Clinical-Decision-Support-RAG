"""Groq LLM client wrapper: chat completion, streaming, retries, timeouts."""
from __future__ import annotations

import time
from collections.abc import Iterator

from groq import Groq
from groq import APIError, APITimeoutError

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqService:
    """Wraps the Groq chat completions API with production-grade resilience."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = Groq(api_key=self._settings.groq_api_key)

    def generate(self, messages: list[dict[str, str]], json_mode: bool = True) -> str:
        """Run a single (non-streaming) chat completion with retry logic.

        Args:
            messages: Chat messages in `{"role", "content"}` form.
            json_mode: If True, requests a JSON-only response from Groq.

        Returns:
            The generated text content.

        Raises:
            LLMGenerationError: If all retry attempts fail.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._settings.llm_max_retries + 1):
            try:
                return self._call_once(messages, json_mode)
            except (APIError, APITimeoutError) as exc:
                last_error = exc
                logger.error(f"groq_call attempt={attempt} failed error={exc}")
                if attempt < self._settings.llm_max_retries:
                    time.sleep(self._settings.llm_retry_backoff_seconds * attempt)

        raise LLMGenerationError(
            f"Groq generation failed after {self._settings.llm_max_retries} attempts: {last_error}"
        )

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream a chat completion token-by-token.

        Yields:
            Incremental text chunks as they arrive from Groq.

        Raises:
            LLMGenerationError: If the stream fails to start or breaks.
        """
        try:
            stream = self._client.chat.completions.create(
                model=self._settings.groq_model,
                messages=messages,
                temperature=self._settings.llm_temperature,
                top_p=self._settings.llm_top_p,
                max_tokens=self._settings.llm_max_tokens,
                timeout=self._settings.llm_timeout_seconds,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (APIError, APITimeoutError) as exc:
            logger.error(f"groq_stream failed error={exc}")
            raise LLMGenerationError(f"Groq streaming failed: {exc}") from exc

    def _call_once(self, messages: list[dict[str, str]], json_mode: bool) -> str:
        response_format = {"type": "json_object"} if json_mode else None
        completion = self._client.chat.completions.create(
            model=self._settings.groq_model,
            messages=messages,
            temperature=self._settings.llm_temperature,
            top_p=self._settings.llm_top_p,
            max_tokens=self._settings.llm_max_tokens,
            timeout=self._settings.llm_timeout_seconds,
            response_format=response_format,
        )
        content = completion.choices[0].message.content
        if not content:
            raise LLMGenerationError("Groq returned an empty completion")
        return content
