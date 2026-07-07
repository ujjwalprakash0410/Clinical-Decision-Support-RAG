"""Classic IR metrics for evaluating retriever quality against a labeled dataset.

Each function takes `retrieved_ids` (ranked, best-first) and
`relevant_ids` (the ground-truth relevant set for that query) and
returns a metric in [0, 1].
"""
from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant documents captured within the top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant document (0 if none found)."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k, with binary relevance."""
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(top_k, start=1) if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def aggregate_retrieval_metrics(
    per_query_results: list[tuple[list[str], set[str]]], k: int = 5
) -> dict[str, float]:
    """Average retrieval metrics across a full evaluation dataset."""
    if not per_query_results:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    precisions, recalls, rr_values, ndcgs = [], [], [], []
    for retrieved_ids, relevant_ids in per_query_results:
        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        rr_values.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k))

    count = len(per_query_results)
    return {
        "precision_at_k": round(sum(precisions) / count, 4),
        "recall_at_k": round(sum(recalls) / count, 4),
        "mrr": round(sum(rr_values) / count, 4),
        "ndcg_at_k": round(sum(ndcgs) / count, 4),
    }
