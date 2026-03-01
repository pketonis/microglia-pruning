"""Run theoretical analysis tying dynamic pruning to lottery ticket behavior."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.theory import analyze_lottery_ticket_behavior, summarize_theoretical_claims


def main() -> None:
    parser = argparse.ArgumentParser(description="Lottery ticket style analysis for pruning masks")
    parser.add_argument("--mask_file", required=True, help=".npy file with shape [epochs, layers, heads]")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mask_trajectory = np.load(args.mask_file)

    analysis = analyze_lottery_ticket_behavior(mask_trajectory)
    claims = summarize_theoretical_claims(analysis)

    report = {
        "analysis": {
            "mean_overlap": analysis.mean_overlap,
            "overlap_std": analysis.overlap_std,
            "early_late_overlap": analysis.early_late_overlap,
            "winning_ticket_score": analysis.winning_ticket_score,
        },
        "claims": claims,
    }

    out = os.path.join(args.output_dir, "theoretical_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
