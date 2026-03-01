"""Pareto frontier utilities for accuracy/latency trade-off analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ParetoPoint:
    """A single operating point from a pruning configuration."""

    label: str
    accuracy: float
    latency_ms: float
    sparsity: float


@dataclass(frozen=True)
class ParetoResult:
    """Frontier and dominated point partition for post-hoc analysis."""

    frontier: List[ParetoPoint]
    dominated: List[ParetoPoint]


def is_dominated(candidate: ParetoPoint, others: Iterable[ParetoPoint]) -> bool:
    """Return True when another point is >= accuracy and <= latency (strict at least one)."""
    for point in others:
        if point is candidate:
            continue
        no_worse = point.accuracy >= candidate.accuracy and point.latency_ms <= candidate.latency_ms
        strictly_better = point.accuracy > candidate.accuracy or point.latency_ms < candidate.latency_ms
        if no_worse and strictly_better:
            return True
    return False


def compute_pareto_frontier(points: Iterable[ParetoPoint]) -> ParetoResult:
    """Compute non-dominated points for maximizing accuracy and minimizing latency."""
    pool = list(points)
    frontier = [point for point in pool if not is_dominated(point, pool)]
    frontier = sorted(frontier, key=lambda p: p.latency_ms)
    dominated = [point for point in pool if point not in frontier]
    return ParetoResult(frontier=frontier, dominated=dominated)
