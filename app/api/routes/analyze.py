"""POST /analyze — the core clinical decision support endpoint."""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.dependencies.di import ReportServiceDep, get_groq_service
from app.models.request_models import AnalyzeRequest
from app.models.response_models import AnalyzeResponse
from app.prompts.clinical_prompt import build_clinical_report_messages
from app.utils.metrics_tracker import get_metrics_tracker

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, report_service=ReportServiceDep) -> AnalyzeResponse:
    """Retrieve evidence and generate a structured, cited clinical report.

    No business logic lives here — this route only validates input
    (via Pydantic), delegates to `ReportService`, and records metrics.
    """
    response = report_service.generate_report(
        query=request.query,
        retriever_type=request.retriever_type,
        k=request.top_k,
        document_type=request.document_type_filter,
    )
    get_metrics_tracker().record(response.latency_ms)
    return response


@router.post("/analyze/stream")
def analyze_stream(request: AnalyzeRequest, report_service=ReportServiceDep):
    """Stream the raw LLM output for a clinical query token-by-token.

    Intended for interactive UIs; the non-streaming `/analyze` endpoint
    remains the source of truth for the fully-parsed, schema-validated
    `ClinicalReport`.
    """
    documents, _ = report_service._retrieval_service.retrieve_documents(
        request.query, request.retriever_type, k=request.top_k
    )
    messages = build_clinical_report_messages(request.query, documents)
    groq_service = get_groq_service()

    def token_generator():
        yield from groq_service.stream(messages)

    return StreamingResponse(token_generator(), media_type="text/event-stream")
