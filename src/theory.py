"""Theoretical analysis helpers linking dynamic pruning to lottery-ticket style behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class LotteryTicketAnalysis:
    """Summary statistics for sparse subnetwork stability."""

    mean_overlap: float
    overlap_std: float
    early_late_overlap: float
    winning_ticket_score: float


def _binarize_masks(masks: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (masks >= threshold).astype(np.float32)


def analyze_lottery_ticket_behavior(mask_trajectory: np.ndarray) -> LotteryTicketAnalysis:
    """Quantify whether early discovered subnetworks persist across training.

    Args:
        mask_trajectory: numpy array with shape [epochs, layers, heads].

    Returns:
        LotteryTicketAnalysis with overlap-based stability metrics.
    """
    if mask_trajectory.ndim != 3:
        raise ValueError("mask_trajectory must have shape [epochs, layers, heads]")

    binary = _binarize_masks(mask_trajectory)
    flat = binary.reshape(binary.shape[0], -1)

    overlaps = []
    for i in range(1, flat.shape[0]):
        prev = flat[i - 1]
        curr = flat[i]
        union = np.maximum(prev, curr).sum()
        if union == 0:
            overlaps.append(1.0)
        else:
            overlaps.append(float(np.minimum(prev, curr).sum() / union))

    early = flat[0]
    late = flat[-1]
    union = np.maximum(early, late).sum()
    early_late_overlap = 1.0 if union == 0 else float(np.minimum(early, late).sum() / union)

    mean_overlap = float(np.mean(overlaps)) if overlaps else 1.0
    overlap_std = float(np.std(overlaps)) if overlaps else 0.0
    winning_ticket_score = float(0.6 * mean_overlap + 0.4 * early_late_overlap)

    return LotteryTicketAnalysis(
        mean_overlap=mean_overlap,
        overlap_std=overlap_std,
        early_late_overlap=early_late_overlap,
        winning_ticket_score=winning_ticket_score,
    )


def summarize_theoretical_claims(analysis: LotteryTicketAnalysis) -> Dict[str, str]:
    """Return publication-ready textual conclusions from overlap metrics."""
    if analysis.winning_ticket_score >= 0.7:
        stance = "strong evidence"
    elif analysis.winning_ticket_score >= 0.5:
        stance = "moderate evidence"
    else:
        stance = "weak evidence"

    return {
        "stance": stance,
        "claim": (
            "Dynamic microglia agents repeatedly recover sparse subnetworks with "
            f"mean Jaccard overlap {analysis.mean_overlap:.3f}, providing {stance} "
            "for lottery-ticket-style trainability of pruned attention heads."
        ),
    }
