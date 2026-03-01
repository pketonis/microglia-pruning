"""Model registry for reproducible multi-model experiments.

Example:
    >>> from src.model_registry import resolve_model_spec
    >>> resolve_model_spec("phi3").name
    'microsoft/phi-3-mini-4k-instruct'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelSpec:
    """Metadata required to initialize pruning agents across model families."""

    name: str
    num_heads: int
    num_layers: int
    description: str


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "phi3-mini": ModelSpec(
        name="microsoft/phi-3-mini-4k-instruct",
        num_heads=32,
        num_layers=32,
        description="Baseline model used in the original project.",
    ),
    "llama3-8b": ModelSpec(
        name="meta-llama/Meta-Llama-3-8B-Instruct",
        num_heads=32,
        num_layers=32,
        description="Llama-3 instruction-tuned model for cross-family validation.",
    ),
    "mistral-7b": ModelSpec(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        num_heads=32,
        num_layers=32,
        description="Mistral dense transformer for robust pruning transfer studies.",
    ),
}

MODEL_ALIASES: Dict[str, str] = {
    "phi3": "phi3-mini",
    "phi-3": "phi3-mini",
    "llama3": "llama3-8b",
    "llama-3": "llama3-8b",
    "mistral": "mistral-7b",
}


def resolve_model_spec(model_name_or_key: str) -> ModelSpec:
    """Resolve either a model key or a raw HF identifier into a ModelSpec."""
    key = model_name_or_key.strip().lower()
    key = MODEL_ALIASES.get(key, key)
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]

    return ModelSpec(
        name=model_name_or_key,
        num_heads=32,
        num_layers=32,
        description="Custom model (defaults used; heads/layers auto-corrected at runtime).",
    )
