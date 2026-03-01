"""Evaluation script for microglia pruning system."""

import argparse
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.system import MicrogliaPruningSystem


def main():
    parser = argparse.ArgumentParser(description="Evaluate microglia pruning system")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/phi-3-mini-4k-instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Evaluation dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for agents"
    )
    parser.add_argument(
        "--hard_prune",
        action="store_true",
        help="Use hard thresholding for pruning"
    )
    parser.add_argument(
        "--no_pruning",
        action="store_true",
        help="Disable pruning for evaluation (baseline)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max evaluation samples (default: 100; use with --fast for 50)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Max new tokens per generation (default: 128; use with --fast for 64)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast eval for debugging: max_samples=50, max_new_tokens=64"
    )
    
    args = parser.parse_args()
    
    if args.fast:
        args.max_samples = args.max_samples if args.max_samples is not None else 50
        args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 64
        print("Fast mode: max_samples=50, max_new_tokens=64")
    if args.max_samples is None:
        args.max_samples = 100
    if args.max_new_tokens is None:
        args.max_new_tokens = 128
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Microglia Pruning System Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset} ({args.split} split)")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # Initialize system
    print("\nLoading model...")
    system = MicrogliaPruningSystem(
        model=args.base_model,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.model_path}...")
    system.load(args.model_path)
    
    # Configure pruning
    system.set_hard_prune(args.hard_prune)

    # Evaluate
    print("\nRunning evaluation...")
    metrics = system.evaluate(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        use_pruning=not args.no_pruning
    )
    
    # Print results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")
    print(f"Sparsity: {system.get_sparsity():.1%} heads pruned")
    print("="*60)
    
    # Save results
    results_path = os.path.join(args.output_dir, f"{args.dataset}_results.json")
    print(f"\nSaving results to {results_path}...")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
