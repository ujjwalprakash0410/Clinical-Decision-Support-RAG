"""End-to-end benchmark runner: executes the live RAG pipeline against
a labeled query dataset and prepares results for `ragas_runner.py`.

Benchmark dataset format (JSON list), see
`app/evaluation/datasets/sample_dataset.json`:

    [
      {
        "question": "...",
        "ground_truth": "...",
        "relevant_ids": ["doc_1", "doc_3"]
      }
    ]

`relevant_ids` must correspond to values placed in each indexed
document's `metadata["chunk_id"]` field for retrieval metrics to be
meaningful.
"""
from __future__ import annotations

from app.core.constants import RetrieverType
from app.core.logging import get_logger
from app.evaluation.latency import time_call
from app.services.report_service import ReportService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


def run_benchmark(
    dataset: list[dict],
    retrieval_service: RetrievalService,
    report_service: ReportService,
    retriever_type: RetrieverType = RetrieverType.MULTI_QUERY,
    k: int = 5,
) -> list[dict]:
    """Run each benchmark question through retrieval + generation.

    Returns:
        The input dataset enriched with `retrieved_ids`, `contexts`,
        `answer`, and `retrieval_latency_ms` — ready for
        `ragas_runner.run_full_evaluation`.
    """
    enriched: list[dict] = []
    for item in dataset:
        documents, latency_ms = retrieval_service.retrieve_documents(
            item["question"], retriever_type, k=k
        )
        response, _ = time_call(
            report_service.generate_report, item["question"], retriever_type, k
        )

        enriched.append(
            {
                **item,
                "retrieved_ids": [doc.metadata.get("chunk_id", "") for doc in documents],
                "contexts": [doc.page_content for doc in documents],
                "answer": response.report.summary,
                "retrieval_latency_ms": round(latency_ms, 2),
            }
        )
        logger.info(f"benchmark question={item['question'][:60]!r} latency_ms={latency_ms:.1f}")

    return enriched
