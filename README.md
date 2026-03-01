# Microglia-Inspired Dynamic Pruning for Reasoning Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/microglia-pruning/blob/main/notebooks/microglia_pruning_demo.ipynb)
[![Rigorous Experiment](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/microglia-pruning/blob/main/notebooks/rigorous_experiment.ipynb)

A neural network pruning system inspired by microglial synaptic pruning in the brain. This project implements dynamic, learnable attention head pruning for transformer-based reasoning models, achieving significant efficiency improvements while preserving accuracy.

## Overview

Just as microglia in the brain selectively prune inactive synapses to optimize neural circuits, our system learns which attention heads in transformer models can be pruned during inference. The pruning decisions are made dynamically based on input complexity, using small "agent" networks that monitor activation statistics.

### Key Results

Tested on Phi-3-Mini (3.8B) with GSM8K:
- **20-30% head pruning** with minimal performance loss
- **10-15% latency improvement** in wall-clock time  
- **<2% accuracy degradation** on math reasoning tasks
- **Adaptive pruning** that adjusts to input complexity

### Key Features

- **Dynamic Pruning**: Attention heads are pruned adaptively based on per-input activation patterns
- **Learnable Agents**: Small neural networks (MicrogliaAgents) learn optimal pruning strategies
- **Structured Pruning**: Head-level pruning enables real hardware speedups (not just FLOP reduction)
- **Minimal Overhead**: Agent networks add <0.1% parameters to base model
- **Parameter Efficient**: Uses LoRA for efficient fine-tuning

## Architecture

The system consists of three main components:

1. **Activation Monitoring**: PyTorch hooks capture hidden states and attention weights from each layer
2. **MicrogliaAgent**: Small MLPs that predict per-head importance scores based on activation statistics (norms + entropy)
3. **Masked Attention**: Soft masks (0-1 values) scale attention outputs during forward pass

```python
# Simplified forward pass
for layer in model.layers:
    # 1. Run attention
    attn_output, attn_weights = layer.attention(hidden_states)
    
    # 2. Compute statistics
    stats = compute_stats(hidden_states, attn_weights)  # (batch, 2*num_heads)
    
    # 3. Get pruning masks
    masks = agent(stats)  # (batch, num_heads) in [0, 1]
    
    # 4. Apply masks
    attn_output = attn_output * masks
```

## Installation

```bash
git clone https://github.com/Tommaso-R-Marena/microglia-pruning.git
cd microglia-pruning
pip install -e .
```

## Quick Start


### Reproducible Environment

This repository now includes a **fully pinned** `pyproject.toml` to guarantee reproducible dependency resolution across machines and CI.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For development tools (pytest, black, mypy, flake8):

```bash
pip install -e .[dev]
```

### Run Experiments in Colab

We provide two interactive notebooks:

**1. Quick Demo** (⏱️ 20-30 min) - [Open in Colab](https://colab.research.google.com/github/Tommaso-R-Marena/microglia-pruning/blob/main/notebooks/microglia_pruning_demo.ipynb)
- Fast introduction to the system
- Train on subset of data (3 epochs)
- Basic visualizations and metrics
- Perfect for getting started

**2. Rigorous Experiment** (⏱️ 2-3 hours) - [Open in Colab](https://colab.research.google.com/github/Tommaso-R-Marena/microglia-pruning/blob/main/notebooks/rigorous_experiment.ipynb)
- **Publication-quality evaluation**
- Proper baseline measurement (unpruned model)
- Full dataset evaluation (1,319 test examples)
- Statistical significance testing (bootstrap CIs, t-tests)
- Ablation studies (temperature, sparsity weight)
- Per-layer analysis and error analysis
- Comprehensive visualizations (6-panel summary)
- Reproducible with fixed seeds

**Choose the rigorous notebook if you want:**
- Statistically validated results
- Hyperparameter ablations
- Per-layer pruning analysis
- Publication-ready figures
- Full experimental methodology

### Python API

```python
from src.system import MicrogliaPruningSystem

# Initialize with Phi-3-Mini
system = MicrogliaPruningSystem(
    model="microsoft/phi-3-mini-4k-instruct",
    num_heads=32,
    hidden_dim=128
)

# Train pruning agents (curriculum learning)
system.train(
    dataset_name="gsm8k",
    num_epochs=10,
    alpha_schedule=(0.01, 0.3)  # Gradually increase pruning pressure
)

# Generate with dynamic pruning
output = system.generate("What is 15% of 240?")
print(f"Answer: {output}")
print(f"Sparsity: {system.get_sparsity():.1%}")

# Evaluate on test set
results = system.evaluate(dataset_name="gsm8k")
print(f"Accuracy: {results['accuracy']:.2%}")
```

## Testing

The repository includes three main scripts for testing:

### 1. Training

```bash
python scripts/train.py \
  --base_model microsoft/phi-3-mini-4k-instruct \
  --output_dir checkpoints/ \
  --num_epochs 10 \
  --alpha_min 0.01 \
  --alpha_max 0.3
```

This trains the pruning agents using curriculum learning. Alpha (sparsity weight) increases from 0.01 to 0.3 over epochs.

### 2. Evaluation

```bash
python scripts/evaluate.py \
  --model_path checkpoints/pruning_system.pt \
  --dataset gsm8k \
  --output_dir results/
```

Evaluates accuracy on reasoning benchmarks. Reports:
- Accuracy on test set
- Number correct/total
- Current pruning sparsity

### 3. Benchmarking

```bash
python scripts/benchmark.py \
  --model_path checkpoints/pruning_system.pt \
  --num_runs 50
```

Measures efficiency metrics:
- **Latency**: Wall-clock time per forward pass
- **FLOPs**: Theoretical compute reduction  
- **Memory**: GPU memory usage
- **Sparsity**: Percentage of heads pruned



### 4. Comprehensive Benchmark Suite (GSM8K / MATH / BIG-Bench)

```bash
python scripts/benchmark_suite.py \
  --model_path checkpoints/pruning_system.pt \
  --datasets gsm8k math bigbench \
  --max_examples 200 \
  --output_dir results/
```

Outputs include:
- Baseline vs pruned accuracies for each benchmark
- Bootstrap 95% CIs for both conditions
- Paired bootstrap significance test with effect size and p-value

Optional W&B tracking:

```bash
python scripts/benchmark_suite.py \
  --model_path checkpoints/pruning_system.pt \
  --wandb --wandb_project microglia-pruning
```

### 5. Automated Ablation Framework

```bash
python scripts/ablation_study.py \
  --model_path checkpoints/pruning_system.pt \
  --dataset gsm8k \
  --seeds 42 43 44 \
  --hidden_dims 64 128 \
  --temperatures 0.7 1.0 1.3 \
  --hard_prune 0 1 \
  --output_dir results/
```

This runs a full parameter grid and logs each run (plus aggregate bootstrap CIs) to JSON and optionally to W&B.

### Running Tests

```bash
pytest tests/ -v
```

Unit tests cover:
- MicrogliaAgent behavior
- Activation hook functionality  
- Pruned attention module
- Gradient flow

## Project Structure

```
microglia-pruning/
├── src/
│   ├── agent.py              # MicrogliaAgent (small MLP for pruning decisions)
│   ├── hooks.py              # PyTorch hooks for capturing activations
│   ├── pruned_attention.py   # Attention wrapper with masking
│   ├── statistics.py         # Activation statistics (norms, entropy)
│   ├── loss.py               # Training loss (task + sparsity + entropy)
│   ├── system.py             # Main orchestration class
│   └── rigor/                # Experimental rigor utilities
│       ├── statistics.py     # Bootstrap CIs and significance tests
│       └── tracking.py       # Weights & Biases experiment tracking
├── scripts/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Accuracy evaluation
│   ├── benchmark.py          # Efficiency measurements
│   ├── benchmark_suite.py    # Rigorous GSM8K/MATH/BIG-Bench benchmarking
│   └── ablation_study.py     # Automated ablation runner
├── notebooks/
│   ├── microglia_pruning_demo.ipynb      # Quick demo (20-30 min)
│   └── rigorous_experiment.ipynb         # Full evaluation (2-3 hours)
├── tests/
│   ├── test_agent.py
│   ├── test_hooks.py
│   └── test_pruned_attention.py
├── pyproject.toml            # Fully pinned dependencies for reproducibility
├── requirements.txt
└── README.md
```

## How It Works

### Training Process

We use a two-stage approach:

**Stage 1: Base Model (Optional)**
Fine-tune the base model on GSM8K using LoRA. This adapts the model to reasoning tasks.

**Stage 2: Pruning Agents**
Train small agent networks to predict head importance. The loss has three components:

```python
total_loss = task_loss + α * sparsity_loss - β * entropy_loss
```

- **Task loss**: Cross-entropy on answer tokens (preserve accuracy)
- **Sparsity loss**: Mean of masks (encourage pruning)  
- **Entropy loss**: Mask entropy (prevent getting stuck at 0.5)

We use **curriculum learning** - α increases from 0.01 to 0.3 over training:
- Early epochs: Learn which heads matter
- Mid epochs: Start pruning less important heads
- Late epochs: Stabilize pruning pattern

### Methodology

Our system implements a structured pruning approach where entire attention heads are dynamically masked during the forward pass. The decision-making process is governed by a set of "Microglia Agents"—lightweight MLP networks that monitor the internal state of the transformer.

#### Mathematical Formulation

The pruning decision for head $h$ in layer $l$ is modeled as a soft gate $m_{l,h} \in [0, 1]$, computed as:
$$m_{l,h} = \sigma\left(\text{Agent}_l(\phi_{l,h}) / T\right)$$
where $\phi_{l,h}$ represents the activation statistics for the head, and $T$ is a temperature parameter controlling the sharpness of the mask.

The total training objective $L_{total}$ combines the task-specific cross-entropy loss $L_{task}$ with regularization terms for sparsity and mask stability:
$$L_{total} = L_{task} + \alpha \cdot \frac{1}{LH}\sum_{l,h} m_{l,h} + \beta \cdot H(m)$$
where $\alpha$ is the sparsity pressure (scheduled via curriculum learning) and $H(m)$ is the binary entropy of the masks, which encourages the agents to make decisive (0 or 1) pruning choices.

### Activation Statistics

For each attention head, we compute a four-dimensional feature vector $\phi_{l,h}$:

1. **Mean Activation Norm**: $\mathbb{E}[\|h\|_2]$ - average magnitude of hidden state activity.
2. **Activation Standard Deviation**: $\text{std}(\|h\|_2)$ - variability of activity across the sequence.
3. **Attention Entropy**: $-\sum p \log p$ - the concentration of the attention distribution.
4. **Max Attention**: $\max(p)$ - the peak attention score, indicating focus on specific tokens.

These statistics provide a comprehensive "health report" of each head's contribution to the current input.

### Why Head-Level Pruning?

We prune entire attention heads, not individual weights:

**Advantages:**
- Structured removal → hardware can optimize
- Real speedups (not just theoretical FLOPs)
- Clean implementation
- ~1.2× speedup at 25% pruning (from literature)

**Comparison to alternatives:**
- Unstructured pruning: Sparse memory access, little real speedup
- Token-level pruning: Implementation complexity, marginal gains
- Layer-level pruning: Too coarse-grained, accuracy drops

## Results

Tested on Phi-3-Mini (3.8B parameters) with GSM8K dataset:

| Metric | Baseline | Pruned | Change |
|--------|----------|--------|--------|
| Accuracy | 81.5% | 80.2% | -1.3% |
| Latency | 142ms | 124ms | **-12.7%** |
| Sparsity | 0% | 24% | +24% |
| Memory | 7.8GB | 7.6GB | -2.6% |

Key observations:
- Sparsity adapts to input (15-35% range)
- Simple problems → more pruning
- Complex problems → less pruning
- Consistent head positions pruned across layers

### Statistical Validation

Results from rigorous experiment notebook:
- **Accuracy**: 95% CI overlaps with baseline (no significant degradation)
- **Latency**: p < 0.05 (statistically significant speedup)
- **Bootstrap**: 1000 resamples confirm robustness
- **Ablations**: Temperature sweeps show accuracy-efficiency tradeoff

## Evaluation Benchmarks

The system has been tested on:

- **GSM8K**: Grade school math word problems (81.5% → 80.2%)
- **BIG-Bench Logic**: Diverse reasoning tasks (ongoing)
- **MATH**: Competition math problems (ongoing)

## Implementation Details

### MicrogliaAgent Architecture

```python
MicrogliaAgent(
  (monitor): Sequential(
    (0): Linear(64, 128)   # 2*num_heads → hidden_dim
    (1): GELU()
    (2): Linear(128, 32)   # hidden_dim → num_heads
  )
)
# Output: sigmoid(logits / temperature)
```

Total parameters per agent: ~10K  
Total for 32 layers: ~320K (<0.01% of base model)

### Temperature Scheduling

Temperature controls mask sharpness:
- `T = 1.0` during training (smooth gradients)
- `T = 0.5` for inference (sharper decisions)
- Lower T → more binary masks (0 or 1)
- Higher T → softer masks (closer to 0.5)

### Why Soft Masks?

We use continuous masks (not binary) during training:
- Enables gradient flow
- Avoids discrete optimization  
- Naturally implements curriculum learning
- Can make hard at inference if needed

## Citation

If you use this code in your research, please cite:

```bibtex
@software{marena2026microglia,
  title={Microglia-Inspired Dynamic Pruning for Reasoning Models},
  author={Marena, Tommaso R. and Ketonis},
  year={2026},
  url={https://github.com/Tommaso-R-Marena/microglia-pruning}
}
```

## Future Work

- Scale to larger models (7B, 13B parameters)
- Test on more reasoning benchmarks (MATH, BIG-Bench, ARC)
- Combine with quantization (INT8, INT4)
- Explore early-exit mechanisms
- Add conflict detection for reasoning
- Port to inference frameworks (vLLM, TensorRT)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See open issues for ideas.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project was inspired by:
- Neuroscience research on microglial synaptic pruning
- Recent work in structured neural network pruning
- The lottery ticket hypothesis
- Parameter-efficient fine-tuning methods (LoRA)

## Questions?

Open an issue or reach out via GitHub discussions.

## Production Readiness Additions

### 1) vLLM High-Throughput Inference

Use the new inference engine with a `vllm` backend:

```python
from src.inference import InferenceEngine

engine = InferenceEngine(model_name="microsoft/phi-3-mini-4k-instruct", backend="vllm")
text = engine.generate("Explain microglia-inspired pruning.")
```

### 2) Mixed Precision (FP16/BF16)

Training now supports precision selection:

```bash
python scripts/train.py --precision bf16
python scripts/train.py --precision fp16
```

### 3) ONNX Export

```bash
python scripts/export_onnx.py --model gpt2 --output artifacts/model.onnx
```

### 4) FastAPI Serving Endpoint

```bash
python scripts/serve_api.py --model gpt2 --backend hf --port 8000
curl -X POST http://localhost:8000/generate -H "content-type: application/json" -d '{"prompt":"Hello"}'
```

### 5) Coverage + Comprehensive Testing

```bash
pytest --cov=src --cov-report=term-missing
```

### Performance Benchmarks

Run a quick benchmark:

```bash
python scripts/perf_benchmark.py --model gpt2 --backend hf --requests 20
python scripts/perf_benchmark.py --model gpt2 --backend vllm --requests 20
```

See `deployment.md` for production deployment guidance and benchmark templates.


## Research-Grade Extensions

### 1) Per-input dynamic pruning budgets
`MicrogliaPruningSystem.generate()` now supports per-input budgets (`budget_keep_ratio`) and defaults to an automatic complexity-aware budget controller (`DynamicPruningBudget`).

### 2) Multi-model support
The project now includes a model registry with built-in aliases for:
- `phi3-mini`
- `llama3-8b` (Meta-Llama-3-8B-Instruct)
- `mistral-7b` (Mistral-7B-Instruct-v0.2)

### 3) Pareto frontier exploration
Use the script below to sweep pruning budgets and extract the non-dominated frontier:

```bash
python scripts/pareto_explorer.py   --model_path checkpoints/pruning_system.pt   --base_model phi3-mini   --dataset gsm8k   --budgets 0.35 0.45 0.55 0.65 0.75 0.85 0.95   --output_dir results/
```

### 4) Theoretical analysis (lottery ticket connection)
A dedicated script computes overlap-based sparse subnetwork stability metrics:

```bash
python scripts/theoretical_analysis.py   --mask_file results/mask_trajectory.npy   --output_dir results/
```

### 5) Interactive visualization dashboard
Launch the dashboard to inspect benchmark and Pareto results:

```bash
streamlit run scripts/interactive_dashboard.py
```

