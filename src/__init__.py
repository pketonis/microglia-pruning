"""Microglia-Inspired Dynamic Pruning for Reasoning Models."""

from .agent import MicrogliaAgent
from .hooks import create_activation_hook, register_hooks
from .pruned_attention import PrunedAttention
from .statistics import compute_layer_stats
from .loss import compute_pruning_loss
from .system import MicrogliaPruningSystem
from .budget import DynamicPruningBudget
from .pareto import ParetoPoint, ParetoResult, compute_pareto_frontier
from .theory import LotteryTicketAnalysis, analyze_lottery_ticket_behavior
from .inference import InferenceEngine, GenerationConfig
from .export import export_to_onnx

__version__ = "0.1.0"

__all__ = [
    "MicrogliaAgent",
    "create_activation_hook",
    "register_hooks",
    "PrunedAttention",
    "compute_layer_stats",
    "compute_pruning_loss",
    "MicrogliaPruningSystem",
    "DynamicPruningBudget",
    "ParetoPoint",
    "ParetoResult",
    "compute_pareto_frontier",
    "LotteryTicketAnalysis",
    "analyze_lottery_ticket_behavior",
    "InferenceEngine",
    "GenerationConfig",
    "export_to_onnx",
]
