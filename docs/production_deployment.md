# Production Deployment

## Serving
Run API serving with:

```bash
python scripts/serve_api.py --model gpt2 --backend hf
```

## Inference Backends
Use `InferenceEngine` with either `hf` or `vllm` backend.

## Monitoring
Enable W&B via training flag `--wandb`.
