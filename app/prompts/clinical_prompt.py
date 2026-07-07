"""Assembles the final prompt sent to the LLM for report generation."""
from __future__ import annotations

from langchain_core.documents import Document

from app.core.constants import CLINICAL_DISCLAIMER
from app.core.exceptions import PromptFormattingError
from app.prompts.prompt_templates import (
    CLINICAL_REPORT_SYSTEM_PROMPT,
    CLINICAL_REPORT_USER_TEMPLATE,
)


def format_evidence_context(documents: list[Document]) -> str:
    """Render retrieved documents as a numbered evidence block for the prompt."""
    if not documents:
        return "No relevant evidence was retrieved."

    lines = []
    for index, doc in enumerate(documents, start=1):
        title = doc.metadata.get("title", "Untitled")
        source = doc.metadata.get("source", "Unknown source")
        year = doc.metadata.get("publication_year", "n.d.")
        snippet = doc.page_content.strip().replace("\n", " ")[:800]
        lines.append(f"[{index}] {title} ({source}, {year}): {snippet}")
    return "\n\n".join(lines)


def build_clinical_report_messages(
    query: str, documents: list[Document]
) -> list[dict[str, str]]:
    """Build the chat message list for the clinical report generation call.

    Returns:
        A list of `{"role": ..., "content": ...}` dicts compatible with
        the Groq chat completions API.

    Raises:
        PromptFormattingError: If the query is empty.
    """
    if not query or not query.strip():
        raise PromptFormattingError("Cannot build a prompt from an empty query")

    context = format_evidence_context(documents)
    user_content = CLINICAL_REPORT_USER_TEMPLATE.format(
        query=query.strip(), context=context, disclaimer=CLINICAL_DISCLAIMER
    )
    return [
        {"role": "system", "content": CLINICAL_REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
