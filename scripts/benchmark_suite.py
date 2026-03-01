"""Comprehensive benchmark suite for GSM8K, MATH, and BIG-Bench style tasks."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from datasets import load_dataset

from src.rigor.statistics import bootstrap_ci, paired_bootstrap_test
from src.rigor.tracking import ExperimentTracker
from src.system import MicrogliaPruningSystem
from src.utils import set_seed


DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "gsm8k": {
        "path": "gsm8k",
        "name": "main",
        "split": "test",
        "question_col": "question",
        "answer_col": "answer",
    },
    "math": {
        "path": "hendrycks/competition_math",
        "name": "default",
        "split": "test",
        "question_col": "problem",
        "answer_col": "solution",
    },
    "bigbench": {
        "path": "lukaemon/bbh",
        "name": "boolean_expressions",
        "split": "test",
        "question_col": "input",
        "answer_col": "target",
    },
}


@dataclass
class EvalRun:
    dataset: str
    seed: int
    mode: str
    accuracy: float
    total: int
    correct: int
    sparsity: float


def _extract_final_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    marker = re.findall(r"####\s*([^\n]+)", text)
    if marker:
        return marker[-1].strip()

    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        return nums[-1]

    return text.strip().lower()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evaluate_dataset(
    system: MicrogliaPruningSystem,
    dataset_key: str,
    max_examples: int,
    use_pruning: bool,
) -> Tuple[List[int], EvalRun]:
    cfg = DATASET_CONFIGS[dataset_key]
    if cfg["path"] == "lukaemon/bbh":
        ds = load_dataset(cfg["path"])[cfg["split"]]
    else:
        ds = load_dataset(cfg["path"], cfg["name"])[cfg["split"]]
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))

    outcomes: List[int] = []
    for row in ds:
        prompt = f"Question: {row[cfg['question_col']]}\nAnswer:"
        pred = system.generate(prompt, max_new_tokens=128, use_pruning=use_pruning)
        pred_ans = _normalize(_extract_final_answer(pred))
        gt_ans = _normalize(_extract_final_answer(str(row[cfg["answer_col"]])))
        outcomes.append(int(pred_ans == gt_ans))

    accuracy = sum(outcomes) / max(len(outcomes), 1)
    run = EvalRun(
        dataset=dataset_key,
        seed=0,
        mode="pruned" if use_pruning else "baseline",
        accuracy=accuracy,
        total=len(outcomes),
        correct=int(sum(outcomes)),
        sparsity=float(system.get_sparsity() if use_pruning else 0.0),
    )
    return outcomes, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rigorous benchmark suite.")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--base_model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "math", "bigbench"])
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="microglia-pruning")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tracker = ExperimentTracker(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name="benchmark-suite",
        config=vars(args),
    )

    system = MicrogliaPruningSystem(model=args.base_model)
    system.load(args.model_path)

    summary = {"runs": [], "statistics": []}

    for dataset_key in args.datasets:
        base_outcomes, base_run = evaluate_dataset(system, dataset_key, args.max_examples, use_pruning=False)
        pruned_outcomes, pruned_run = evaluate_dataset(system, dataset_key, args.max_examples, use_pruning=True)

        sig = paired_bootstrap_test(base_outcomes, pruned_outcomes, metric_name=f"{dataset_key}_accuracy")
        base_ci = bootstrap_ci(base_outcomes)
        pruned_ci = bootstrap_ci(pruned_outcomes)

        summary["runs"].append(asdict(base_run))
        summary["runs"].append(asdict(pruned_run))
        summary["statistics"].append(
            {
                "dataset": dataset_key,
                "baseline_ci": base_ci,
                "pruned_ci": pruned_ci,
                "effect": asdict(sig),
            }
        )

        tracker.log(
            {
                f"{dataset_key}/baseline_accuracy": base_run.accuracy,
                f"{dataset_key}/pruned_accuracy": pruned_run.accuracy,
                f"{dataset_key}/effect": sig.effect_size,
                f"{dataset_key}/p_value": sig.p_value,
            }
        )

    out_file = os.path.join(args.output_dir, "benchmark_suite_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tracker.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
