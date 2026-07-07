#!/usr/bin/env python
"""CLI: run the full retrieval + generation evaluation benchmark.

Usage:
    python scripts/evaluate.py --dataset app/evaluation/datasets/sample_dataset.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies.di import get_chat_llm, get_report_service, get_retrieval_service  # noqa: E402
from app.core.constants import RetrieverType  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.embeddings.embedding_factory import get_embedding_service  # noqa: E402
from app.evaluation.benchmark import run_benchmark  # noqa: E402
from app.evaluation.ragas_runner import run_full_evaluation  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the clinical RAG pipeline.")
    parser.add_argument(
        "--dataset",
        default="app/evaluation/datasets/sample_dataset.json",
        help="Path to a benchmark dataset JSON file",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k documents to retrieve")
    parser.add_argument(
        "--retriever",
        default="multi_query",
        choices=[r.value for r in RetrieverType],
        help="Retriever strategy to benchmark",
    )
    parser.add_argument(
        "--output", default=None, help="Optional path to write the full results JSON"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    retrieval_service = get_retrieval_service()
    report_service = get_report_service()

    enriched = run_benchmark(
        dataset,
        retrieval_service=retrieval_service,
        report_service=report_service,
        retriever_type=RetrieverType(args.retriever),
        k=args.k,
    )
    results = run_full_evaluation(
        enriched,
        llm=get_chat_llm(),
        embeddings=get_embedding_service().langchain_embeddings,
        retrieval_k=args.k,
    )

    print(json.dumps(results, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps({"scores": results, "raw": enriched}, indent=2))
        print(f"Full results written to {args.output}")


if __name__ == "__main__":
    main()
