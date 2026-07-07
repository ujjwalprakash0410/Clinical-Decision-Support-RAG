"""Parses raw LLM text output into validated `ClinicalReport` objects."""
from __future__ import annotations

import json

from pydantic import ValidationError

from app.core.constants import CLINICAL_DISCLAIMER
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger
from app.models.response_models import ClinicalReport

logger = get_logger(__name__)


def parse_clinical_report(raw_text: str) -> ClinicalReport:
    """Parse and validate the LLM's JSON response into a `ClinicalReport`.

    Enforces the mandatory disclaimer regardless of what the model
    returned, since this is a hard safety requirement, not a
    style preference the LLM should control.

    Raises:
        LLMGenerationError: If the response is not valid JSON or fails
            schema validation.
    """
    try:
        payload = json.loads(_strip_code_fences(raw_text))
    except json.JSONDecodeError as exc:
        logger.error(f"response_formatter invalid_json error={exc}")
        raise LLMGenerationError("LLM did not return valid JSON") from exc

    payload["disclaimer"] = CLINICAL_DISCLAIMER

    try:
        return ClinicalReport.model_validate(payload)
    except ValidationError as exc:
        logger.error(f"response_formatter schema_validation_failed error={exc}")
        raise LLMGenerationError(f"LLM response failed schema validation: {exc}") from exc


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences some models add despite instructions."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.removeprefix("json").strip()
    return cleaned
