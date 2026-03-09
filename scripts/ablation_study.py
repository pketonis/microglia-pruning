"""Automated ablation framework for microglia pruning experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Dict, List

from src.rigor.statistics import bootstrap_ci
from src.rigor.tracking import ExperimentTracker
from src.system import MicrogliaPruningSystem
from src.utils import set_seed


def run_single_ablation(
    base_model: str,
    model_path: str,
    dataset: str,
    seed: int,
    max_examples: int,
    hidden_dim: int,
    temperature: float,
    hard_prune: bool,
) -> Dict[str, float]:
    set_seed(seed)
    system = MicrogliaPruningSystem(
        model=base_model,
        hidden_dim=hidden_dim,
        temperature=temperature,
        seed=seed,
    )
    system.load(model_path)
    system.set_hard_prune(hard_prune)
    metrics = system.evaluate(dataset_name=dataset, split="test", max_samples=max_examples, use_pruning=True)
    metrics.update(
        {
            "seed": seed,
            "hidden_dim": hidden_dim,
            "temperature": temperature,
            "hard_prune": hard_prune,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated ablation studies.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 0.7, 1.0, 1.3])
    parser.add_argument("--alpha_max", nargs="+", type=float, default=[0.2, 0.3, 0.4])
    parser.add_argument("--hard_prune", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--resume", action="store_true", help="Skip finished configs from existing output")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="microglia-pruning")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tracker = ExperimentTracker(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name="ablation-study",
        config=vars(args),
    )

    out_path = os.path.join(args.output_dir, "ablation_results.json")
    structured_path = os.path.join(args.output_dir, "ablation_results_structured.json")

    results: List[Dict[str, float]] = []
    structured_results: List[Dict[str, object]] = []
    completed = set()
    if args.resume and os.path.exists(structured_path):
        with open(structured_path, "r", encoding="utf-8") as f:
            structured_results = json.load(f)
        for item in structured_results:
            cfg = item["config"]
            completed.add((cfg["hidden_dim"], cfg["temperature"], cfg["alpha_max"], cfg["hard_prune"]))

    for hidden_dim, temperature, alpha_max, hard_prune in itertools.product(
        args.hidden_dims, args.temperatures, args.alpha_max, args.hard_prune
    ):
        key = (hidden_dim, temperature, alpha_max, int(bool(hard_prune)))
        if key in completed:
            continue
        combo_acc: List[float] = []
        seed_rows = []
        for seed in args.seeds:
            metrics = run_single_ablation(
                base_model=args.base_model,
                model_path=args.model_path,
                dataset=args.dataset,
                seed=seed,
                max_examples=args.max_examples,
                hidden_dim=hidden_dim,
                temperature=temperature,
                hard_prune=bool(hard_prune),
            )
            metrics["alpha_max"] = alpha_max
            results.append(metrics)
            seed_rows.append(metrics)
            combo_acc.append(metrics["accuracy"])
            tracker.log({
                "ablation/accuracy": metrics["accuracy"],
                "ablation/sparsity": metrics.get("sparsity", 0.0),
                "ablation/hidden_dim": hidden_dim,
                "ablation/temperature": temperature,
                "ablation/hard_prune": int(bool(hard_prune)),
                "ablation/alpha_max": alpha_max,
                "ablation/seed": seed,
            })

        ci = bootstrap_ci(combo_acc)
        aggregate = {"mean": ci["mean"], "std": float(__import__("numpy").std(combo_acc, ddof=1)) if len(combo_acc) > 1 else 0.0, "ci": ci}
        structured_results.append({
            "config": {
                "hidden_dim": hidden_dim,
                "temperature": temperature,
                "alpha_max": alpha_max,
                "hard_prune": int(bool(hard_prune)),
            },
            "seeds": seed_rows,
            "aggregate": aggregate,
        })
        tracker.log(
            {
                "ablation/mean_accuracy": ci["mean"],
                "ablation/ci_low": ci["ci_low"],
                "ablation/ci_high": ci["ci_high"],
                "ablation/hidden_dim": hidden_dim,
                "ablation/temperature": temperature,
                "ablation/hard_prune": int(bool(hard_prune)),
                "ablation/alpha_max": alpha_max,
            }
        )

    tracker.finish()

    out_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(structured_path, "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2)

    print(f"Saved {len(results)} ablation runs to {out_path}")
    print(f"Saved {len(structured_results)} grouped configs to {structured_path}")


if __name__ == "__main__":
    main()
