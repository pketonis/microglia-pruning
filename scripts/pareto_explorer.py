"""Explore Pareto frontier of pruning budgets for accuracy-latency tradeoffs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from typing import List

from datasets import load_dataset

from src.pareto import ParetoPoint, compute_pareto_frontier
from src.system import MicrogliaPruningSystem


def benchmark_budget(
    system: MicrogliaPruningSystem,
    questions: List[str],
    answers: List[str],
    budget: float,
    max_new_tokens: int,
) -> ParetoPoint:
    outcomes = []
    latencies = []
    for question, answer in zip(questions, answers):
        prompt = f"Question: {question}\nAnswer:"
        start = time.perf_counter()
        output = system.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            use_pruning=True,
            budget_keep_ratio=budget,
        )
        latencies.append((time.perf_counter() - start) * 1000.0)
        outcomes.append(int(system._extract_answer(output) == system._extract_answer(str(answer))))

    accuracy = sum(outcomes) / max(1, len(outcomes))
    latency_ms = sum(latencies) / max(1, len(latencies))
    sparsity = max(0.0, 1.0 - budget)
    return ParetoPoint(
        label=f"budget_{budget:.2f}",
        accuracy=float(accuracy),
        latency_ms=float(latency_ms),
        sparsity=float(sparsity),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto exploration for pruning budgets")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="phi3-mini")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    system = MicrogliaPruningSystem(model=args.base_model)
    system.load(args.model_path)

    ds = load_dataset(args.dataset, "main", split="test")
    ds = ds.select(range(min(args.max_examples, len(ds))))
    questions = [row["question"] for row in ds]
    answers = [row["answer"] for row in ds]

    points = [benchmark_budget(system, questions, answers, budget, args.max_new_tokens) for budget in args.budgets]
    pareto = compute_pareto_frontier(points)

    results_json = {
        "points": [asdict(p) for p in points],
        "frontier": [asdict(p) for p in pareto.frontier],
    }

    json_path = os.path.join(args.output_dir, "pareto_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    csv_path = os.path.join(args.output_dir, "pareto_frontier.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "accuracy", "latency_ms", "sparsity"])
        writer.writeheader()
        for point in pareto.frontier:
            writer.writerow(asdict(point))

    print(json.dumps(results_json, indent=2))


if __name__ == "__main__":
    main()
