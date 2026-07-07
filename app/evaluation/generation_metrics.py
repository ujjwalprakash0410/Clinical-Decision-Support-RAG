"""Generation-quality evaluation using Ragas.

Wraps Ragas so the rest of the evaluation module (and callers) depend
on a simple function signature rather than Ragas's dataset/API details
directly — isolating us from breaking changes in that library.

Ragas's metrics need a "judge" LLM and an embeddings model. Left
unconfigured, Ragas defaults to OpenAI, which this project has no key
for — so the judge LLM/embeddings must always be passed in explicitly
(we use the same Groq chat model and BAAI/bge-small-en-v1.5 embeddings
already used elsewhere in the app).
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from app.core.exceptions import EvaluationError
from app.core.logging import get_logger

logger = get_logger(__name__)

_RAGAS_METRIC_NAMES = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


def evaluate_generation(
    samples: list[dict], llm: BaseLanguageModel, embeddings: Embeddings
) -> dict[str, float]:
    """Compute faithfulness, context precision/recall, and answer relevancy.

    Args:
        samples: list of dicts each with keys `question`, `answer`,
            `contexts` (list[str]), and `ground_truth`.
        llm: A LangChain chat model used as Ragas's judge (e.g. ChatGroq).
        embeddings: A LangChain embeddings model used for the
            similarity-based metrics (e.g. the app's BAAI/bge-small-en-v1.5).

    Returns:
        Dict mapping metric name -> average score across all samples.

    Raises:
        EvaluationError: If Ragas is unavailable or evaluation fails.
    """
    if not samples:
        return {name: 0.0 for name in _RAGAS_METRIC_NAMES}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise EvaluationError(
            "Ragas/datasets are not installed. Run `pip install ragas datasets`."
        ) from exc

    try:
        dataset = Dataset.from_list(samples)
        result = evaluate(
            dataset,
            metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
            llm=LangchainLLMWrapper(llm),
            embeddings=LangchainEmbeddingsWrapper(embeddings),
        )
        scores = result.to_pandas()[_RAGAS_METRIC_NAMES].mean().to_dict()
        return {name: round(float(value), 4) for name, value in scores.items()}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"ragas_evaluation_failed error={exc}")
        raise EvaluationError(f"Ragas evaluation failed: {exc}") from exc


def detect_hallucination_rate(samples: list[dict], scores: dict[str, float]) -> float:
    """Approximate hallucination rate as `1 - faithfulness`.

    This is a simplification: faithfulness (from Ragas) measures how
    well the answer is supported by the retrieved context, so its
    complement is a reasonable proxy for unsupported ("hallucinated")
    claims until a dedicated NLI-based checker is added.
    """
    faithfulness_score = scores.get("faithfulness", 0.0)
    return round(1.0 - faithfulness_score, 4)
