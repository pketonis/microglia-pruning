"""Utilities for reproducible evaluation and rigorous experimentation.

Example:
    >>> from src.rigor import bootstrap_ci
    >>> stats = bootstrap_ci([0, 1, 1, 0], num_bootstrap=50, ci=0.95)
    >>> stats["ci_low"] <= stats["ci_high"]
    True
"""

from .statistics import (
    bca_bootstrap_ci,
    bootstrap_ci,
    cohen_d,
    holm_bonferroni,
    paired_bootstrap_test,
    permutation_test_paired,
    power_analysis_min_detectable_effect,
)
from .tracking import ExperimentTracker

__all__ = [
    "bootstrap_ci",
    "bca_bootstrap_ci",
    "paired_bootstrap_test",
    "cohen_d",
    "holm_bonferroni",
    "permutation_test_paired",
    "power_analysis_min_detectable_effect",
    "ExperimentTracker",
]
