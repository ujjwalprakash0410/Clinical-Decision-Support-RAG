"""Thread-safe in-process counter for request count / latency metrics."""
from __future__ import annotations

import threading
from functools import lru_cache


class MetricsTracker:
    """Accumulates simple runtime metrics for the /metrics endpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_latency_ms = 0.0

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._total_requests += 1
            self._total_latency_ms += latency_ms

    @property
    def total_requests(self) -> int:
        return self._total_requests

    @property
    def average_latency_ms(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return round(self._total_latency_ms / self._total_requests, 2)


@lru_cache
def get_metrics_tracker() -> MetricsTracker:
    return MetricsTracker()
