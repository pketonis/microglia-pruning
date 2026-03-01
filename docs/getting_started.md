# Getting Started

## Installation

```bash
pip install -e .
```

## First Training Run

```bash
python scripts/train.py --base_model phi3-mini --dataset gsm8k --num_epochs 1 --use-budget
```

## First Inference Run

```python
from src.system import MicrogliaPruningSystem

system = MicrogliaPruningSystem(model="phi3-mini")
print(system.generate("What is 12*12?", use_pruning=True))
```
