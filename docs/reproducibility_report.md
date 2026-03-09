# Reproducibility Report

## Determinism
- Use `src.utils.set_seed(seed, deterministic=True)` to set Python/NumPy/PyTorch/Transformers seeds.
- Sets `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Enables deterministic cuDNN and deterministic algorithms (`warn_only=True`).

## Software
- Install from pinned `requirements.txt`.
- Frozen snapshot: `requirements-frozen.txt`.

## Multi-seed Protocol
- Recommended seeds: `42 43 44`.
- Report mean ± std and bootstrap CI for accuracy/latency.

## Tracking
- `ExperimentTracker` supports W&B + local JSONL fallback (`wandb_offline/tracker_fallback.jsonl`).
- Use `WANDB_MODE=offline` for disconnected runs.

## Runtime Notes
- GPU timing uses CUDA events with warmup.
- Use dedicated GPU when possible to reduce interference.

## Hardware Metadata Checklist
- GPU model / count
- CUDA + driver versions
- Python + torch/transformers versions
- Git commit SHA
