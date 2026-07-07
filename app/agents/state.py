"""Shared state schema for the clinical agent's LangGraph workflow.

`history` uses an `operator.add` reducer so that each turn's summary is
*appended* to the conversation's persisted history rather than
overwriting it — this is what gives the agent multi-turn memory when
combined with LangGraph's checkpointer (see `clinical_agent.py`).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document

from app.models.response_models import ClinicalReport


class AgentState(TypedDict, total=False):
    """Mutable state threaded through every node in the agent graph."""

    conversation_id: str
    query: str
    reformulated_query: str
    retriever_type: str
    use_live_pubmed_search: bool
    search_terms: list[str]
    documents: list[Document]
    iterations: int
    sufficient: bool
    report: ClinicalReport
    history: Annotated[list[dict[str, Any]], operator.add]
