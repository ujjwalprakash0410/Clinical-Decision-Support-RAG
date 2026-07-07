"""GET /sources — lists supported evidence sources and document types."""
from fastapi import APIRouter

from app.core.constants import DocumentType
from app.loaders.guideline_loader import LOADER_REGISTRY
from app.models.response_models import SourcesResponse

router = APIRouter(tags=["sources"])


@router.get("/sources", response_model=SourcesResponse)
def list_sources() -> SourcesResponse:
    """Return the loader sources and document types the system supports."""
    return SourcesResponse(
        available_sources=list(LOADER_REGISTRY.keys()),
        document_types=[doc_type.value for doc_type in DocumentType],
    )
