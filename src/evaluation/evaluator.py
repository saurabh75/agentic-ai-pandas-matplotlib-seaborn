"""
Stage 6 — Evaluation & Feedback

Tracks per-query metrics matching the architecture diagram:
  Answer Correctness | Retrieval Precision | Retrieval Recall
  Latency | Cost (token estimate) | User Feedback

Metrics are stored in-process (session-scoped) and surfaced in the UI.
For production use, swap the in-memory store for a database or OTEL backend.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, List, Optional

from src.models.agent import EvaluationMetrics
from src.logger import get_logger

logger = get_logger(__name__)

# Maximum number of query records to keep in memory
_MAX_HISTORY = 200


# =============================================================================
# Metric Store
# =============================================================================


class MetricStore:
    """
    In-memory ring buffer of EvaluationMetrics, one entry per query.

    Thread-safety: Streamlit runs in a single thread per session, so
    a simple deque is sufficient here.
    """

    def __init__(self, maxlen: int = _MAX_HISTORY) -> None:
        self._records: Deque[EvaluationMetrics] = deque(maxlen=maxlen)
        self._feedback: Dict[int, int] = {}  # record_index → thumbs up/down

    def record(self, metrics: EvaluationMetrics) -> int:
        self._records.append(metrics)
        idx = len(self._records) - 1
        logger.debug(
            f"Eval recorded | latency={metrics.latency_ms:.0f}ms "
            f"| confidence={metrics.confidence:.2f} "
            f"| groundedness={metrics.groundedness_score:.2f} "
            f"| iterations={metrics.iterations_used}"
        )
        return idx

    def add_feedback(self, record_index: int, thumbs_up: bool) -> None:
        self._feedback[record_index] = 1 if thumbs_up else -1

    def all(self) -> List[EvaluationMetrics]:
        return list(self._records)

    def summary(self) -> Dict[str, float]:
        """Aggregate statistics over all recorded queries."""
        records = self.all()
        if not records:
            return {}

        n = len(records)
        return {
            "total_queries": n,
            "avg_latency_ms": sum(r.latency_ms for r in records) / n,
            "avg_confidence": sum(r.confidence for r in records) / n,
            "avg_groundedness": sum(r.groundedness_score for r in records) / n,
            "avg_iterations": sum(r.iterations_used for r in records) / n,
            "avg_retrieval_k": sum(r.retrieval_k for r in records) / n,
            "p95_latency_ms": _percentile([r.latency_ms for r in records], 95),
            "thumbs_up_rate": self._thumbs_up_rate(),
        }

    def _thumbs_up_rate(self) -> float:
        if not self._feedback:
            return 0.0
        positives = sum(1 for v in self._feedback.values() if v > 0)
        return positives / len(self._feedback)

    def clear(self) -> None:
        self._records.clear()
        self._feedback.clear()


def _percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


# =============================================================================
# Latency Timer (context manager)
# =============================================================================


class LatencyTimer:
    """Context manager that captures wall-clock latency in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "LatencyTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# =============================================================================
# Retrieval Precision / Recall Estimator (heuristic)
# =============================================================================


class RetrievalEvaluator:
    """
    Estimates retrieval quality without human-labelled relevance judgements.

    Uses a proxy signal: fraction of chunks whose reranker_score exceeds a
    threshold as "estimated precision", and normalises by total candidates
    for "estimated recall".
    """

    @staticmethod
    def estimate(
        candidates,  # List[RetrievalCandidate]
        reranker_threshold: float = 0.0,
    ) -> Dict[str, float]:
        if not candidates:
            return {"estimated_precision": 0.0, "estimated_recall": 0.0}

        relevant = [
            c for c in candidates
            if (c.reranker_score is not None and c.reranker_score >= reranker_threshold)
            or (c.reranker_score is None and c.hybrid_score > 0.01)
        ]
        precision = len(relevant) / len(candidates)
        # Recall is inherently unknown without ground truth; use a heuristic
        recall = min(1.0, len(candidates) / 20)  # normalised against expected fetch_k

        return {
            "estimated_precision": round(precision, 3),
            "estimated_recall": round(recall, 3),
        }


# =============================================================================
# Singleton metric store (shared across Streamlit sessions via module state)
# =============================================================================

_metric_store = MetricStore()


def get_metric_store() -> MetricStore:
    return _metric_store
