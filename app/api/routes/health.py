"""GET /health — liveness/readiness probe."""
from fastapi import APIRouter

from app.core.config import get_settings
from app.models.response_models import HealthResponse

router = APIRouter(tags=["health"])

_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return basic service liveness information."""
    settings = get_settings()
    return HealthResponse(status="ok", app_name=settings.app_name, version=_VERSION)
