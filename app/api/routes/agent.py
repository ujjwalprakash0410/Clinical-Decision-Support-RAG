"""POST /agent/chat — agentic, memory-aware clinical assistant endpoint."""
from fastapi import APIRouter

from app.api.dependencies.di import AgentServiceDep
from app.models.request_models import AgentChatRequest
from app.models.response_models import AgentChatResponse
from app.utils.metrics_tracker import get_metrics_tracker

router = APIRouter(tags=["agent"])


@router.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest, agent_service=AgentServiceDep) -> AgentChatResponse:
    """Multi-step, memory-aware clinical evidence assistant.

    Unlike `/analyze` (a single retrieve-then-generate call), this endpoint
    routes each query through an agentic LangGraph workflow that decides
    which retrieval strategy fits the question, whether to search PubMed
    live in addition to the indexed corpus, reranks results with a
    cross-encoder, and retries with a broadened query if the initial
    evidence is too thin — all while remembering prior turns under the
    same `conversation_id`.
    """
    response = agent_service.chat(conversation_id=request.conversation_id, query=request.query)
    get_metrics_tracker().record(response.latency_ms)
    return response
