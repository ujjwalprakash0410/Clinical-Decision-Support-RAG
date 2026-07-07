"""LLM-driven query routing for the clinical agent.

This is the "agentic" decision point: rather than always using a fixed
retriever, the agent asks the LLM to look at the question (and recent
conversation history) and decide how to handle it — which retrieval
strategy fits, whether a live PubMed search is warranted, and how to
resolve a follow-up question into a standalone one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.core.constants import RetrieverType
from app.core.logging import get_logger
from app.llm.groq_service import GroqService
from app.prompts.prompt_templates import AGENT_ROUTER_SYSTEM_PROMPT, AGENT_ROUTER_USER_TEMPLATE

logger = get_logger(__name__)

_VALID_RETRIEVER_TYPES = {member.value for member in RetrieverType}


@dataclass
class RoutingDecision:
    """The agent's decision for how to handle a single query."""

    retriever_type: RetrieverType = RetrieverType.MULTI_QUERY
    use_live_pubmed_search: bool = False
    search_terms: list[str] = field(default_factory=list)
    reformulated_query: str = ""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none — this is the first question in the conversation)"
    lines = [f"- Q: {turn['query']}\n  A (summary): {turn['answer_summary']}" for turn in history[-3:]]
    return "\n".join(lines)


def route_query(query: str, history: list[dict], groq_service: GroqService) -> RoutingDecision:
    """Ask the LLM to route this query. Falls back to safe defaults on any failure.

    A routing failure should never break the user-facing request, so
    parsing errors are logged and a conservative default decision
    (multi-query retrieval, no live search, query used as-is) is returned
    instead of propagating an exception.
    """
    messages = [
        {"role": "system", "content": AGENT_ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": AGENT_ROUTER_USER_TEMPLATE.format(history=_format_history(history), query=query),
        },
    ]

    try:
        raw_output = groq_service.generate(messages, json_mode=True)
        payload = json.loads(raw_output)
        return _parse_decision(payload, query)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"agent_router failed, using default routing error={exc}")
        return RoutingDecision(reformulated_query=query)


def _parse_decision(payload: dict, original_query: str) -> RoutingDecision:
    retriever_value = payload.get("retriever_type", RetrieverType.MULTI_QUERY.value)
    if retriever_value not in _VALID_RETRIEVER_TYPES:
        retriever_value = RetrieverType.MULTI_QUERY.value

    return RoutingDecision(
        retriever_type=RetrieverType(retriever_value),
        use_live_pubmed_search=bool(payload.get("use_live_pubmed_search", False)),
        search_terms=list(payload.get("search_terms", [])) or [original_query],
        reformulated_query=str(payload.get("reformulated_query") or original_query),
    )
