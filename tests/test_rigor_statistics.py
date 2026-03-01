import numpy as np

from src.rigor.statistics import bootstrap_ci, paired_bootstrap_test


def test_bootstrap_ci_contains_mean_for_binary_data():
    values = [0, 1, 1, 0, 1, 1, 1, 0]
    result = bootstrap_ci(values, num_bootstrap=500, seed=123)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_paired_bootstrap_detects_positive_effect():
    baseline = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    treatment = np.array([1, 1, 1, 1, 1, 0, 1, 1])
    result = paired_bootstrap_test(baseline, treatment, num_bootstrap=1000, seed=7)
    assert result.effect_size > 0
    assert result.ci_high >= result.ci_low
