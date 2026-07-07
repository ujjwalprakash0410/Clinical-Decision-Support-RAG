"""Static constants shared across the application.

Values that are genuinely fixed (not environment-dependent) belong
here. Anything that might reasonably change between deployments
belongs in `config.py` instead.
"""
from enum import Enum

CLINICAL_DISCLAIMER: str = (
    "This information is intended to support clinical decision making "
    "and should not replace professional medical judgment."
)

SAFETY_REFUSAL_MESSAGE: str = (
    "This system provides evidence-based clinical decision support only. "
    "It cannot and will not provide a patient diagnosis. Please consult a "
    "qualified healthcare professional for diagnostic decisions."
)


class DocumentType(str, Enum):
    """Supported source document classifications."""

    PUBMED = "pubmed"
    WHO_GUIDELINE = "who_guideline"
    CDC_GUIDELINE = "cdc_guideline"
    CLINICAL_GUIDELINE = "clinical_guideline"
    OTHER = "other"


class RetrieverType(str, Enum):
    """Identifiers for the retriever strategies the factory can build."""

    SIMILARITY = "similarity"
    MULTI_QUERY = "multi_query"
    PARENT_DOCUMENT = "parent_document"
    CONTEXTUAL_COMPRESSION = "contextual_compression"
    METADATA_FILTERED = "metadata_filtered"
    HYBRID = "hybrid"


class IndexOperation(str, Enum):
    """Supported operations on the vector index."""

    CREATE = "create"
    LOAD = "load"
    UPDATE = "update"
    DELETE = "delete"


MAX_QUERY_LENGTH: int = 2000
DEFAULT_CITATION_STYLE: str = "numbered"
