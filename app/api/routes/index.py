"""POST /index — trigger document ingestion into the vector store."""
from fastapi import APIRouter

from app.api.dependencies.di import IndexingServiceDep
from app.models.request_models import IndexRequest
from app.models.response_models import IndexResponse

router = APIRouter(tags=["index"])


@router.post("/index", response_model=IndexResponse)
def build_index(request: IndexRequest, indexing_service=IndexingServiceDep) -> IndexResponse:
    """Ingest documents from the requested sources and (re)build the index."""
    chunk_count, sources_used = indexing_service.index_sources(
        source_names=request.sources,
        query_terms=request.query_terms,
        rebuild=request.rebuild,
    )
    status = "indexed" if chunk_count > 0 else "no_documents_found"
    return IndexResponse(status=status, documents_indexed=chunk_count, sources_used=sources_used)
