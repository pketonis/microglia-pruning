"""Benchmarking script for efficiency measurements."""

import argparse
import os
import sys
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fvcore.nn import FlopCountAnalysis
from src.system import MicrogliaPruningSystem


def measure_latency(system, prompt, num_runs=50, warmup_runs=10):
    """Measure latency with warmup and robust summary stats."""
    latencies = []

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            effective_prompt = prompt[0] if isinstance(prompt, list) else prompt
            _ = system.generate(effective_prompt, max_new_tokens=10)

    if torch.cuda.is_available():
        for _ in range(num_runs):
            torch.cuda.empty_cache()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.no_grad():
                effective_prompt = prompt[0] if isinstance(prompt, list) else prompt
                _ = system.generate(effective_prompt, max_new_tokens=256)
            end_event.record()

            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))  # Already in ms
    else:
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                effective_prompt = prompt[0] if isinstance(prompt, list) else prompt
                _ = system.generate(effective_prompt, max_new_tokens=256)
            latencies.append((time.time() - start) * 1000)
    
    arr = torch.tensor(latencies, dtype=torch.float64)
    median = float(torch.median(arr).item())
    n = len(latencies)
    idx = torch.randint(0, n, (1000, n))
    boot = arr[idx].median(dim=1).values
    ci_low = float(torch.quantile(boot, 0.025).item())
    ci_high = float(torch.quantile(boot, 0.975).item())

    return {
        'mean_ms': float(arr.mean().item()),
        'median_ms': median,
        'min_ms': float(arr.min().item()),
        'max_ms': float(arr.max().item()),
        'std_ms': float(arr.std().item()) if len(latencies) > 1 else 0.0,
        'ci95_median_low_ms': ci_low,
        'ci95_median_high_ms': ci_high,
    }


def measure_memory():
    """Measure GPU memory usage."""
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        }
    return {'allocated_mb': 0, 'reserved_mb': 0}


def main():
    parser = argparse.ArgumentParser(description="Benchmark pruning system efficiency")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/phi-3-mini-4k-instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=50,
        help="Number of runs for latency measurement"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for benchmarking"
    )
    parser.add_argument("--warmup_runs", type=int, default=10, help="Warmup iterations before measurement")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1,4,8,16,32], help="Batch sizes to benchmark")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save results"
    )
    parser.add_argument(
        "--hard_prune",
        action="store_true",
        help="Use hard thresholding for pruning"
    )
    parser.add_argument(
        "--no_pruning",
        action="store_true",
        help="Disable pruning for benchmark (baseline measurement)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Microglia Pruning System Benchmarking")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Runs: {args.num_runs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # Load system
    print("\nLoading model...")
    system = MicrogliaPruningSystem(model=args.base_model)
    system.load(args.model_path)
    
    # Configure pruning for benchmark
    system._enable_pruning(not args.no_pruning)
    system.set_hard_prune(args.hard_prune)

    # Test prompt
    test_prompt = "What is 15% of 240?"
    
    # Measure latency
    print("\nMeasuring latency...")
    latency_by_batch = {}
    for batch_size in args.batch_sizes:
        prompts = [test_prompt] * batch_size
        latency_by_batch[str(batch_size)] = measure_latency(system, prompts, args.num_runs, args.warmup_runs)
    latency_metrics = latency_by_batch[str(args.batch_size)]
    
    # Measure memory
    print("Measuring memory usage...")
    memory_metrics = measure_memory()
    
    # Get sparsity
    sparsity = system.get_sparsity()
    
    # Measure FLOPs using fvcore
    print("Estimating FLOPs...")
    try:
        inputs = system.tokenizer(test_prompt, return_tensors="pt").to(system.device)
        # Use a wrapper to make it compatible with fvcore
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, input_ids, attention_mask):
                return self.model(input_ids=input_ids, attention_mask=attention_mask)

        wrapper = ModelWrapper(system.model)
        flops = FlopCountAnalysis(wrapper, (inputs['input_ids'], inputs['attention_mask']))
        total_flops = flops.total()
        print(f"Total FLOPs: {total_flops:.2e}")
    except Exception as e:
        print(f"FLOP calculation failed: {e}")
        total_flops = 0

    # Compile results
    results = {
        'latency': latency_metrics,
        'latency_by_batch': latency_by_batch,
        'memory': memory_metrics,
        'sparsity': sparsity,
        'flops': total_flops,
        'config': {
            'num_runs': args.num_runs,
            'batch_size': args.batch_size,
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"Median Latency: {latency_metrics['median_ms']:.2f} ms")
    print(f"Median 95% CI: [{latency_metrics['ci95_median_low_ms']:.2f}, {latency_metrics['ci95_median_high_ms']:.2f}] ms")
    print(f"Average Latency: {latency_metrics['mean_ms']:.2f} ms")
    print(f"Min Latency: {latency_metrics['min_ms']:.2f} ms")
    print(f"Max Latency: {latency_metrics['max_ms']:.2f} ms")
    print(f"GPU Memory: {memory_metrics['allocated_mb']:.1f} MB")
    print(f"Sparsity: {sparsity:.1%} heads pruned")
    print("="*60)
    
    # Save results
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    print(f"\nSaving results to {results_path}...")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
