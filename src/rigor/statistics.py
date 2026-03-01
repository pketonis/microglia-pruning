"""Statistical testing helpers for benchmark and ablation analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class BootstrapResult:
    """Result for a bootstrap significance analysis."""

    metric_name: str
    baseline_mean: float
    treatment_mean: float
    effect_size: float
    ci_low: float
    ci_high: float
    p_value: float


def _to_numpy(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("At least one value is required.")
    return arr


def bootstrap_ci(
    values: Iterable[float],
    num_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute a non-parametric bootstrap confidence interval for the mean."""
    samples = _to_numpy(values)
    rng = np.random.default_rng(seed)

    boot_means = np.empty(num_bootstrap, dtype=float)
    n = samples.shape[0]
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = samples[idx].mean()

    alpha = 1.0 - ci
    low_q, high_q = alpha / 2.0, 1.0 - alpha / 2.0
    return {
        "mean": float(samples.mean()),
        "ci_low": float(np.quantile(boot_means, low_q)),
        "ci_high": float(np.quantile(boot_means, high_q)),
    }


def paired_bootstrap_test(
    baseline: Iterable[float],
    treatment: Iterable[float],
    metric_name: str = "accuracy",
    num_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap test on treatment-baseline difference.

    Inputs should be paired per-example outcomes (e.g. correctness arrays of 0/1).
    """
    base = _to_numpy(baseline)
    trt = _to_numpy(treatment)
    if base.shape != trt.shape:
        raise ValueError("Baseline and treatment arrays must have the same shape.")

    diff = trt - base
    rng = np.random.default_rng(seed)
    n = diff.shape[0]

    boot_effect = np.empty(num_bootstrap, dtype=float)
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_effect[i] = diff[idx].mean()

    alpha = 1.0 - ci
    low_q, high_q = alpha / 2.0, 1.0 - alpha / 2.0
    ci_low = float(np.quantile(boot_effect, low_q))
    ci_high = float(np.quantile(boot_effect, high_q))

    # Two-sided p-value via bootstrap sign test approximation
    effect = float(diff.mean())
    p_left = np.mean(boot_effect <= 0.0)
    p_right = np.mean(boot_effect >= 0.0)
    p_val = float(2.0 * min(p_left, p_right))
    p_val = min(p_val, 1.0)

    return BootstrapResult(
        metric_name=metric_name,
        baseline_mean=float(base.mean()),
        treatment_mean=float(trt.mean()),
        effect_size=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_val,
    )


def summarize_significance(results: List[BootstrapResult]) -> List[Dict[str, float]]:
    """Convert bootstrap result objects to serializable dictionaries."""
    return [
        {
            "metric": r.metric_name,
            "baseline_mean": r.baseline_mean,
            "treatment_mean": r.treatment_mean,
            "effect_size": r.effect_size,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "p_value": r.p_value,
        }
        for r in results
    ]
