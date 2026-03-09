"""
Microglia-Inspired Dynamic Pruning for Reasoning Models

Authors: Tommaso R. Marena (The Catholic University of America)
         Panos Ketonis (Yale University)

Copyright (c) 2026
"""
from typing import Any, Dict, Optional

import math
import torch
import torch.nn.functional as F


def compute_pruning_loss(
    task_loss: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 0.01,
    distillation_weight: float = 0.0,
    student_logits: Optional[torch.Tensor] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    kd_temperature: float = 1.0,
    layer_sparsity_targets: Optional[torch.Tensor] = None,
    layer_target_weight: float = 1.0,
) -> Dict[str, Any]:
    """Computes the combined loss for training pruning agents.

    L_total = L_task + α*L_sparsity + β*L_entropy + λ_kd*L_kd + λ_layer*L_layer_target
    """
    sparsity_loss = masks.mean()

    eps = 1e-10
    mask_entropy = -(
        masks * (masks + eps).log() + (1 - masks) * (1 - masks + eps).log()
    ).mean()

    if layer_sparsity_targets is not None:
        layer_targets = layer_sparsity_targets.to(device=masks.device, dtype=masks.dtype)
        if layer_targets.dim() == 1:
            layer_targets = layer_targets.unsqueeze(1)
        observed_keep_ratio = masks.mean(dim=1, keepdim=True)
        layer_target_loss = F.mse_loss(observed_keep_ratio, 1.0 - layer_targets)
    else:
        layer_target_loss = torch.zeros((), device=masks.device, dtype=masks.dtype)

    if student_logits is not None and teacher_logits is not None and distillation_weight > 0:
        temp = max(kd_temperature, 1e-4)
        student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp ** 2)
    else:
        kd_loss = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)

    total_loss = (
        task_loss
        + alpha * sparsity_loss
        + beta * mask_entropy
        + distillation_weight * kd_loss
        + layer_target_weight * layer_target_loss
    )

    return {
        "total_loss": total_loss,
        "task_loss": task_loss.item(),
        "sparsity_loss": sparsity_loss.item(),
        "entropy_loss": mask_entropy.item(),
        "distillation_loss": kd_loss.item(),
        "layer_sparsity_target_loss": layer_target_loss.item(),
    }


def get_alpha_schedule(epoch: int, max_epochs: int,
                      alpha_min: float = 0.01,
                      alpha_max: float = 0.3,
                      schedule_type: str = "linear") -> float:
    """Calculates the sparsity weight (alpha) for curriculum learning.

    The pruning pressure (alpha) increases over training to allow the
    agents to first identify head importance before enforcing sparsity.

    Args:
        epoch (int): Current epoch (0-indexed).
        max_epochs (int): Total number of epochs.
        alpha_min (float): Starting value of alpha.
        alpha_max (float): Final value of alpha.
        schedule_type (str): One of {"linear", "cosine", "exponential"}.

    Returns:
        float: The sparsity weight for the current epoch.

    Raises:
        ValueError: If alpha values or schedule settings are unsafe.
    """
    if alpha_min < 0 or alpha_max < 0:
        raise ValueError("alpha values must be non-negative. Fix by: use alpha >= 0.")
    if alpha_min > 1.0 or alpha_max > 1.0:
        raise ValueError("alpha values must be <= 1.0. Fix by: keep alpha in [0, 1].")
    if alpha_max < alpha_min:
        raise ValueError("alpha_max must be >= alpha_min. Fix by: swap or adjust schedule bounds.")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be >= 1. Fix by: set a positive epoch count.")

    if max_epochs == 1:
        return float(alpha_max)

    progress = min(max(epoch / max(max_epochs - 1, 1), 0.0), 1.0)

    if schedule_type == "linear":
        alpha = alpha_min + (alpha_max - alpha_min) * progress
    elif schedule_type == "cosine":
        alpha = alpha_min + (alpha_max - alpha_min) * 0.5 * (1.0 - math.cos(math.pi * progress))
    elif schedule_type == "exponential":
        growth = 5.0
        alpha = alpha_min + (alpha_max - alpha_min) * ((math.exp(growth * progress) - 1.0) / (math.exp(growth) - 1.0))
    else:
        raise ValueError("schedule_type must be one of {'linear', 'cosine', 'exponential'}. Fix by: choose a supported schedule.")

    return float(alpha)


def compute_efficiency_metrics(masks: torch.Tensor) -> Dict[str, float]:
    """Computes pruning efficiency metrics."""
    with torch.no_grad():
        binary_masks = (masks > 0.5).float()

        active_heads = binary_masks.sum(dim=1).mean().item()
        total_heads = masks.shape[1]
        sparsity = 1.0 - (active_heads / total_heads)
        mean_mask = masks.mean().item()

    return {
        "sparsity": sparsity,
        "mean_mask": mean_mask,
        "active_heads": active_heads,
    }
