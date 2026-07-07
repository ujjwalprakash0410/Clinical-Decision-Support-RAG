"""Tests for retrieval evaluation metrics."""
from __future__ import annotations

from app.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_all_relevant():
    assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_precision_at_k_none_relevant():
    assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


def test_recall_at_k_partial_match():
    assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5


def test_mean_reciprocal_rank_first_hit():
    assert mean_reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_mean_reciprocal_rank_second_hit():
    assert mean_reciprocal_rank(["x", "a"], {"a"}) == 0.5


def test_mean_reciprocal_rank_no_hit():
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k_perfect_ranking():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_aggregate_retrieval_metrics_empty_dataset():
    result = aggregate_retrieval_metrics([], k=5)
    assert result["precision_at_k"] == 0.0


def test_aggregate_retrieval_metrics_averages_across_queries():
    per_query = [
        (["a", "b"], {"a"}),
        (["x", "y"], {"x", "y"}),
    ]
    result = aggregate_retrieval_metrics(per_query, k=2)
    assert 0.0 < result["precision_at_k"] <= 1.0
