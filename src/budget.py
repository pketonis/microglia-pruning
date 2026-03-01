"""Dynamic pruning budget controller for complexity-aware inference.

Example:
    >>> from src.budget import DynamicPruningBudget
    >>> budget = DynamicPruningBudget()
    >>> 0.2 <= budget.compute_keep_ratio("What is 2+2?") <= 0.95
    True
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict


@dataclass
class DynamicPruningBudget:
    """Computes a keep-ratio budget conditioned on input complexity."""

    min_keep_ratio: float = 0.35
    max_keep_ratio: float = 0.95
    length_weight: float = 0.45
    numeric_weight: float = 0.35
    symbol_weight: float = 0.20

    def _features(self, prompt: str) -> Dict[str, float]:
        tokens = max(len(prompt.split()), 1)
        numbers = len(re.findall(r"\d", prompt))
        symbols = len(re.findall(r"[\+\-\*\/=\(\)\[\]\{\}]", prompt))

        length_score = min(tokens / 256.0, 1.0)
        numeric_score = min(numbers / max(tokens, 1), 1.0)
        symbol_score = min(symbols / max(tokens, 1), 1.0)

        return {
            "tokens": float(tokens),
            "length_score": length_score,
            "numeric_score": numeric_score,
            "symbol_score": symbol_score,
        }

    def compute_keep_ratio(self, prompt: str) -> float:
        """Return keep-ratio in [min_keep_ratio, max_keep_ratio]."""
        feats = self._features(prompt)
        complexity = (
            self.length_weight * feats["length_score"]
            + self.numeric_weight * feats["numeric_score"]
            + self.symbol_weight * feats["symbol_score"]
        )

        # Smooth non-linear mapping favors conservative budgets on complex inputs.
        complexity = 1.0 / (1.0 + math.exp(-6.0 * (complexity - 0.35)))
        keep_ratio = self.min_keep_ratio + (self.max_keep_ratio - self.min_keep_ratio) * complexity
        return float(max(self.min_keep_ratio, min(self.max_keep_ratio, keep_ratio)))

    def summarize(self, prompt: str) -> Dict[str, float]:
        feats = self._features(prompt)
        feats["keep_ratio"] = self.compute_keep_ratio(prompt)
        return feats
