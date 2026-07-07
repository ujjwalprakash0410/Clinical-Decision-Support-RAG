"""Tests for prompt assembly logic."""
from __future__ import annotations

import pytest

from app.core.exceptions import PromptFormattingError
from app.prompts.clinical_prompt import build_clinical_report_messages, format_evidence_context


def test_format_evidence_context_with_no_documents():
    assert format_evidence_context([]) == "No relevant evidence was retrieved."


def test_format_evidence_context_numbers_documents(sample_documents):
    context = format_evidence_context(sample_documents)
    assert "[1]" in context
    assert "[2]" in context
    assert "WHO" in context
    assert "CDC" in context


def test_build_clinical_report_messages_structure(sample_documents):
    messages = build_clinical_report_messages("What is the treatment for afib?", sample_documents)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "afib" in messages[1]["content"]


def test_build_clinical_report_messages_rejects_empty_query(sample_documents):
    with pytest.raises(PromptFormattingError):
        build_clinical_report_messages("   ", sample_documents)


def test_build_clinical_report_messages_includes_disclaimer_instruction(sample_documents):
    messages = build_clinical_report_messages("query", sample_documents)
    assert "disclaimer" in messages[1]["content"].lower()
