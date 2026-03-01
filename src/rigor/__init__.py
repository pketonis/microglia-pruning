"""Utilities for reproducible evaluation and rigorous experimentation.

Example:
    >>> from src.rigor import bootstrap_ci
    >>> stats = bootstrap_ci([0, 1, 1, 0], num_bootstrap=50, ci=0.95)
    >>> stats["ci_low"] <= stats["ci_high"]
    True
"""

from .statistics import bootstrap_ci, paired_bootstrap_test
from .tracking import ExperimentTracker

__all__ = ["bootstrap_ci", "paired_bootstrap_test", "ExperimentTracker"]
