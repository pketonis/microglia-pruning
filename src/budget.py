"""
Microglia-Inspired Dynamic Pruning for Reasoning Models

Authors: Tommaso R. Marena (The Catholic University of America)
         Panos Ketonis (Yale University)

Copyright (c) 2026
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, List, Optional


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
        """Return keep-ratio in [min_keep_ratio, max_keep_ratio] for one input."""
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

    def adjust(
        self,
        prompts: Iterable[str],
        static_override: Optional[float] = None,
    ) -> List[float]:
        """Compute per-sample budgets for a batch.

        Args:
            prompts: Batch of prompts.
            static_override: Optional fixed keep ratio; when set, overrides
                complexity-aware scores for all samples.

        Returns:
            List of keep ratios with one value per input prompt.

        Raises:
            ValueError: If static_override is outside [0, 1].
        """
        prompts = list(prompts)
        if static_override is not None:
            if not 0.0 <= float(static_override) <= 1.0:
                raise ValueError(
                    "static_override must be in [0, 1]. Fix by: pass a ratio between 0 and 1."
                )
            keep = float(max(self.min_keep_ratio, min(self.max_keep_ratio, float(static_override))))
            return [keep for _ in prompts]

        return [self.compute_keep_ratio(prompt) for prompt in prompts]
