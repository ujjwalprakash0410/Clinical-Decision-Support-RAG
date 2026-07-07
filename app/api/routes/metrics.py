"""GET /metrics — lightweight in-process runtime metrics.

For a real production deployment this would be replaced with a
Prometheus exporter; the `MetricsTracker` interface is kept narrow so
that swap is a one-file change.
"""
from fastapi import APIRouter

from app.api.dependencies.di import VectorStoreDep
from app.models.response_models import MetricsResponse
from app.utils.metrics_tracker import get_metrics_tracker

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(vector_store=VectorStoreDep) -> MetricsResponse:
    """Return request-count / latency / index-size metrics."""
    tracker = get_metrics_tracker()
    return MetricsResponse(
        total_requests=tracker.total_requests,
        average_latency_ms=tracker.average_latency_ms,
        index_document_count=vector_store.document_count,
    )
