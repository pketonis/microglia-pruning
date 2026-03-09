#!/usr/bin/env bash
set -euo pipefail

SEEDS=(42 43 44)
MODEL_PATH=${1:-checkpoints/pruning_system.pt}
OUT=${2:-results/repro}
mkdir -p "$OUT"

for s in "${SEEDS[@]}"; do
  python scripts/train.py --seed "$s" --output_dir "$OUT/train_seed_$s" --num_epochs 3
  python scripts/evaluate.py --model_path "$OUT/train_seed_$s/pruning_system.pt" --output_file "$OUT/eval_seed_$s.json"
  python scripts/benchmark.py --model_path "$OUT/train_seed_$s/pruning_system.pt" --output_dir "$OUT/bench_seed_$s"
done

python scripts/ablation_study.py --model_path "$MODEL_PATH" --output_dir "$OUT/ablation" --resume
