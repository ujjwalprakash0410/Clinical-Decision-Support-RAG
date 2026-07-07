"""Latency measurement helpers for benchmarking retrieval and generation."""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any


def time_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Call `fn` and return `(result, elapsed_ms)`."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Compute p50/p95/mean/max summary statistics for a list of latencies."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}

    sorted_latencies = sorted(latencies_ms)
    return {
        "p50_ms": round(statistics.median(sorted_latencies), 2),
        "p95_ms": round(_percentile(sorted_latencies, 0.95), 2),
        "mean_ms": round(statistics.mean(sorted_latencies), 2),
        "max_ms": round(max(sorted_latencies), 2),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(percentile * (len(sorted_values) - 1))))
    return sorted_values[index]
