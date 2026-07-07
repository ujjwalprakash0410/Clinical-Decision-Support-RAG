"""Loads a benchmark dataset and runs full retrieval + generation evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from app.core.exceptions import EvaluationError
from app.core.logging import get_logger
from app.evaluation.generation_metrics import detect_hallucination_rate, evaluate_generation
from app.evaluation.retrieval_metrics import aggregate_retrieval_metrics

logger = get_logger(__name__)


def load_benchmark_dataset(path: str) -> list[dict]:
    """Load a benchmark dataset in the format defined by `benchmark.py`."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise EvaluationError(f"Benchmark dataset not found at {path}")
    try:
        return json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Benchmark dataset at {path} is not valid JSON") from exc


def run_full_evaluation(
    dataset: list[dict],
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    retrieval_k: int = 5,
) -> dict:
    """Run both retrieval and generation evaluation over a dataset.

    Expects each dataset item to have already been run through the
    live system (see `benchmark.py`) and contain:
      question, answer, contexts (list[str]), ground_truth,
      retrieved_ids (list[str]), relevant_ids (list[str])

    Args:
        llm: Judge LLM for Ragas metrics (e.g. the app's ChatGroq instance).
        embeddings: Embeddings model for Ragas's similarity-based metrics.
    """
    retrieval_pairs = [
        (item["retrieved_ids"], set(item["relevant_ids"])) for item in dataset
    ]
    retrieval_scores = aggregate_retrieval_metrics(retrieval_pairs, k=retrieval_k)

    generation_samples = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
            "ground_truth": item["ground_truth"],
        }
        for item in dataset
    ]
    generation_scores = evaluate_generation(generation_samples, llm=llm, embeddings=embeddings)
    hallucination_rate = detect_hallucination_rate(generation_samples, generation_scores)

    logger.info(f"evaluation retrieval={retrieval_scores} generation={generation_scores}")

    return {
        "retrieval": retrieval_scores,
        "generation": generation_scores,
        "hallucination_rate": hallucination_rate,
    }
