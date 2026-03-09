"""Statistical testing helpers for benchmark and ablation analyses."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional

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
    d = cohen_d(base, trt)
    p_left = np.mean(boot_effect <= 0.0)
    p_right = np.mean(boot_effect >= 0.0)
    p_val = float(2.0 * min(p_left, p_right))
    p_val = min(p_val, 1.0)

    return BootstrapResult(
        metric_name=metric_name,
        baseline_mean=float(base.mean()),
        treatment_mean=float(trt.mean()),
        effect_size=d,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_val,
    )



def cohen_d(baseline: Iterable[float], treatment: Iterable[float]) -> float:
    """Compute Cohen's d effect size for two samples."""
    base = _to_numpy(baseline)
    trt = _to_numpy(treatment)
    if base.shape != trt.shape:
        raise ValueError("Baseline and treatment arrays must have the same shape.")

    delta = trt - base
    std = float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(np.mean(delta) / std)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bca_bootstrap_ci(
    values: Iterable[float],
    statistic_fn=np.mean,
    num_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute BCa bootstrap CI for a scalar statistic."""
    samples = _to_numpy(values)
    n = samples.shape[0]
    rng = np.random.default_rng(seed)

    theta_hat = float(statistic_fn(samples))

    boot = np.empty(num_bootstrap, dtype=float)
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(statistic_fn(samples[idx]))

    prop_less = float(np.mean(boot < theta_hat))
    prop_less = min(max(prop_less, 1e-10), 1 - 1e-10)
    z0 = NormalDist().inv_cdf(prop_less)

    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = float(statistic_fn(np.delete(samples, i)))
    jack_mean = float(np.mean(jack))
    num = float(np.sum((jack_mean - jack) ** 3))
    den = float(np.sum((jack_mean - jack) ** 2))
    a = num / (6.0 * (den ** 1.5)) if den > 0 else 0.0

    alpha = 1.0 - ci
    z_low = NormalDist().inv_cdf(alpha / 2.0)
    z_high = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    adj_low = _norm_cdf(z0 + (z0 + z_low) / max(1e-12, (1 - a * (z0 + z_low))))
    adj_high = _norm_cdf(z0 + (z0 + z_high) / max(1e-12, (1 - a * (z0 + z_high))))

    ci_low = float(np.quantile(boot, min(max(adj_low, 0.0), 1.0)))
    ci_high = float(np.quantile(boot, min(max(adj_high, 0.0), 1.0)))

    return {"mean": theta_hat, "ci_low": ci_low, "ci_high": ci_high}


def holm_bonferroni(p_values: Iterable[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni rejection decisions in original order."""
    pv = list(float(p) for p in p_values)
    order = sorted(range(len(pv)), key=lambda i: pv[i])
    reject = [False] * len(pv)
    m = len(pv)
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pv[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


def power_analysis_min_detectable_effect(
    n: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Approximate minimum detectable standardized effect size (two-sided z-test)."""
    if n <= 1:
        raise ValueError("n must be greater than 1")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    return float((z_alpha + z_power) / sqrt(n))


def permutation_test_paired(
    baseline: Iterable[float],
    treatment: Iterable[float],
    n_permutations: int = 5000,
    seed: int = 42,
) -> float:
    """Paired permutation test using random sign flips."""
    base = _to_numpy(baseline)
    trt = _to_numpy(treatment)
    if base.shape != trt.shape:
        raise ValueError("Baseline and treatment arrays must have the same shape.")

    diff = trt - base
    observed = abs(float(np.mean(diff)))
    rng = np.random.default_rng(seed)

    samples = 0
    extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=diff.shape[0])
        stat = abs(float(np.mean(diff * signs)))
        samples += 1
        if stat >= observed:
            extreme += 1
    return float((extreme + 1) / (samples + 1))

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
