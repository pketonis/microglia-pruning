import numpy as np

from src.rigor.statistics import (
    bca_bootstrap_ci,
    bootstrap_ci,
    cohen_d,
    holm_bonferroni,
    paired_bootstrap_test,
    permutation_test_paired,
    power_analysis_min_detectable_effect,
)


def test_bootstrap_ci_contains_mean_for_binary_data():
    values = [0, 1, 1, 0, 1, 1, 1, 0]
    result = bootstrap_ci(values, num_bootstrap=500, seed=123)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_bca_bootstrap_ci_basic():
    values = [0, 1, 1, 0, 1, 1, 1, 0]
    result = bca_bootstrap_ci(values, num_bootstrap=500, seed=123)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_paired_bootstrap_detects_positive_effect():
    baseline = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    treatment = np.array([1, 1, 1, 1, 1, 0, 1, 1])
    result = paired_bootstrap_test(baseline, treatment, num_bootstrap=1000, seed=7)
    assert result.effect_size > 0
    assert result.ci_high >= result.ci_low


def test_cohen_d_zero_for_identical_samples():
    x = np.array([1, 2, 3, 4])
    assert cohen_d(x, x) == 0.0


def test_holm_bonferroni_monotonic_rejections():
    reject = holm_bonferroni([0.001, 0.01, 0.2], alpha=0.05)
    assert reject == [True, True, False]


def test_power_analysis_positive():
    mde = power_analysis_min_detectable_effect(1319)
    assert mde > 0


def test_permutation_test_small_sample_runs():
    p = permutation_test_paired([0, 1, 0, 1], [1, 1, 1, 1], n_permutations=1000, seed=1)
    assert 0 <= p <= 1
