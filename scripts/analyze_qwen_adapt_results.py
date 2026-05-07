#!/usr/bin/env python3
"""Analyze Qwen ADAPT experiment output directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("unpruned", "static", "adapt")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_accuracy(rows: list[dict[str, str]], group_key: str) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        grouped[row[group_key]][row["condition"]] = {
            "accuracy": float(row["accuracy"]),
            "correct": int(row["correct"]),
            "total": int(row["total"]),
            "latency_ms": float(row["mean_latency_ms"]),
            "keep_ratio": float(row["mean_keep_ratio"]),
        }
    return grouped


def paired_correct_by_group(rows: list[dict[str, str]], group_key: str) -> dict[str, dict[str, dict[int, int]]]:
    grouped: dict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[row[group_key]][row["condition"]][int(row["index"])] = int(row["correct"])
    return grouped


def paired_delta_ci(
    grouped_predictions: dict[str, dict[int, int]],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> tuple[float, float, int]:
    static = grouped_predictions["static"]
    adapt = grouped_predictions["adapt"]
    paired_indices = sorted(set(static) & set(adapt))
    if not paired_indices:
        return math.nan, math.nan, 0

    deltas = np.array([adapt[index] - static[index] for index in paired_indices], dtype=float)
    if deltas.size == 1:
        value = float(deltas.mean())
        return value, value, 1

    samples = rng.choice(deltas, size=(bootstrap_samples, deltas.size), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high), int(deltas.size)


def summarize_task_rows(
    result_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    group_key: str,
    group_output_key: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    accuracies = grouped_accuracy(result_rows, group_key)
    predictions = paired_correct_by_group(prediction_rows, group_key)
    summary_rows = []

    for group_name, values in sorted(accuracies.items()):
        if not all(condition in values for condition in CONDITIONS):
            continue
        ci_low, ci_high, paired_n = paired_delta_ci(predictions[group_name], rng, bootstrap_samples)
        static_acc = values["static"]["accuracy"]
        adapt_acc = values["adapt"]["accuracy"]
        unpruned_acc = values["unpruned"]["accuracy"]
        summary_rows.append({
            group_output_key: group_name,
            "unpruned_accuracy": unpruned_acc,
            "static_accuracy": static_acc,
            "adapt_accuracy": adapt_acc,
            "adapt_minus_static": adapt_acc - static_acc,
            "adapt_minus_static_ci_low": ci_low,
            "adapt_minus_static_ci_high": ci_high,
            "adapt_minus_unpruned": adapt_acc - unpruned_acc,
            "static_minus_unpruned": static_acc - unpruned_acc,
            "paired_examples": paired_n,
            "static_keep_ratio": values["static"]["keep_ratio"],
            "adapt_keep_ratio": values["adapt"]["keep_ratio"],
            "unpruned_latency_ms": values["unpruned"]["latency_ms"],
            "static_latency_ms": values["static"]["latency_ms"],
            "adapt_latency_ms": values["adapt"]["latency_ms"],
        })

    return summary_rows


def summarize_overall(task_rows: list[dict[str, object]], rng: np.random.Generator, bootstrap_samples: int) -> dict[str, object]:
    deltas = np.array([float(row["adapt_minus_static"]) for row in task_rows], dtype=float)
    samples = rng.choice(deltas, size=(bootstrap_samples, deltas.size), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "groups": int(deltas.size),
        "mean_adapt_minus_static": float(deltas.mean()),
        "mean_adapt_minus_static_ci_low": float(low),
        "mean_adapt_minus_static_ci_high": float(high),
        "adapt_wins": int((deltas > 0).sum()),
        "ties": int((deltas == 0).sum()),
        "static_wins": int((deltas < 0).sum()),
        "max_gain": float(deltas.max()),
        "max_loss": float(deltas.min()),
    }


def save_bbh_gap_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["adapt_minus_static"]))
    labels = [str(row["task"]) for row in ordered]
    deltas = np.array([float(row["adapt_minus_static"]) for row in ordered])
    ci_low = np.array([float(row["adapt_minus_static_ci_low"]) for row in ordered])
    ci_high = np.array([float(row["adapt_minus_static_ci_high"]) for row in ordered])
    xerr = np.vstack([deltas - ci_low, ci_high - deltas])

    fig_height = max(7.0, 0.34 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = ["#b94a48" if value < 0 else "#4f7da7" for value in deltas]
    ax.barh(labels, deltas, xerr=xerr, color=colors, alpha=0.9, ecolor="#333333", capsize=2)
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_xlabel("ADAPT - Static Accuracy")
    ax.set_ylabel("BBH Task")
    ax.set_title("BBH Accuracy Gap with Paired Bootstrap 95% CI")
    ax.grid(axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_math_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: int(row["level"]))
    levels = [int(row["level"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(levels, [float(row["unpruned_accuracy"]) for row in ordered], marker="o", label="unpruned")
    ax.plot(levels, [float(row["static_accuracy"]) for row in ordered], marker="o", label="static")
    ax.plot(levels, [float(row["adapt_accuracy"]) for row in ordered], marker="o", label="adapt")
    ax.set_xlabel("MATH Difficulty Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("MATH Accuracy by Difficulty")
    ax.set_xticks(levels)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def cosine_similarity_matrix(mask_npz_path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(mask_npz_path)
    labels = []
    vectors = []
    for key in sorted(data.files):
        if not key.endswith("_adapt"):
            continue
        label = key.removesuffix("_adapt")
        labels.append(label)
        vectors.append(data[key].mean(axis=0).reshape(-1))

    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-12, None)
    return labels, normalized @ normalized.T


def save_mask_similarity_plot(labels: list[str], similarity: np.ndarray, output_path: Path) -> None:
    fig_size = max(8.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(similarity, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("ADAPT Mask Cosine Similarity Across BBH Tasks")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir or Path("analysis") / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    bbh_rows = summarize_task_rows(
        read_csv(run_dir / "bbh_results.csv"),
        read_csv(run_dir / "bbh_predictions.csv"),
        "task",
        "task",
        rng,
        args.bootstrap_samples,
    )
    math_rows = summarize_task_rows(
        read_csv(run_dir / "math_results.csv"),
        read_csv(run_dir / "math_predictions.csv"),
        "level",
        "level",
        rng,
        args.bootstrap_samples,
    )

    bbh_rows.sort(key=lambda row: float(row["adapt_minus_static"]), reverse=True)
    math_rows.sort(key=lambda row: int(row["level"]))

    summary = {
        "run_dir": str(run_dir),
        "bbh": summarize_overall(bbh_rows, rng, args.bootstrap_samples),
        "math": summarize_overall(math_rows, rng, args.bootstrap_samples),
    }

    common_fields = [
        "unpruned_accuracy",
        "static_accuracy",
        "adapt_accuracy",
        "adapt_minus_static",
        "adapt_minus_static_ci_low",
        "adapt_minus_static_ci_high",
        "adapt_minus_unpruned",
        "static_minus_unpruned",
        "paired_examples",
        "static_keep_ratio",
        "adapt_keep_ratio",
        "unpruned_latency_ms",
        "static_latency_ms",
        "adapt_latency_ms",
    ]
    write_csv(output_dir / "bbh_task_summary.csv", bbh_rows, ["task", *common_fields])
    write_csv(output_dir / "math_level_summary.csv", math_rows, ["level", *common_fields])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save_bbh_gap_plot(bbh_rows, output_dir / "figure_bbh_gap_ci.png")
    save_math_plot(math_rows, output_dir / "figure_math_accuracy_clean.png")
    labels, similarity = cosine_similarity_matrix(run_dir / "bbh_mask_logs.npz")
    save_mask_similarity_plot(labels, similarity, output_dir / "figure_bbh_mask_similarity_clean.png")
    np.savetxt(output_dir / "bbh_mask_similarity.csv", similarity, delimiter=",", header=",".join(labels), comments="")

    print(json.dumps(summary, indent=2))
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
