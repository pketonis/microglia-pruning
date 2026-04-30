"""Tests for loss module."""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.loss import compute_pruning_loss, get_alpha_schedule


def test_compute_pruning_loss_with_distillation_and_layer_targets() -> None:
    task_loss = torch.tensor(1.2)
    masks = torch.tensor(
        [[0.9, 0.7, 0.2, 0.1], [0.8, 0.5, 0.3, 0.2], [0.7, 0.6, 0.4, 0.2]],
        dtype=torch.float32,
    )
    student_logits = torch.randn(3, 10)
    teacher_logits = torch.randn(3, 10)
    layer_targets = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32)

    out = compute_pruning_loss(
        task_loss=task_loss,
        masks=masks,
        alpha=0.1,
        beta=0.01,
        distillation_weight=0.2,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        kd_temperature=2.0,
        layer_sparsity_targets=layer_targets,
        layer_target_weight=0.3,
    )

    assert "distillation_loss" in out
    assert "layer_sparsity_target_loss" in out
    assert out["total_loss"].item() > 0


def test_compute_pruning_loss_without_optional_terms() -> None:
    task_loss = torch.tensor(0.8)
    masks = torch.rand(2, 4)

    out = compute_pruning_loss(task_loss=task_loss, masks=masks)

    assert out["distillation_loss"] == 0.0
    assert out["layer_sparsity_target_loss"] == 0.0


def test_get_alpha_schedule() -> None:
    # Linear
    assert get_alpha_schedule(0, 10, 0.1, 0.5, "linear") == 0.1
    assert get_alpha_schedule(9, 10, 0.1, 0.5, "linear") == 0.5
    # Cosine
    assert get_alpha_schedule(0, 10, 0.1, 0.5, "cosine") == 0.1
    assert get_alpha_schedule(9, 10, 0.1, 0.5, "cosine") == 0.5
    # Exponential
    assert abs(get_alpha_schedule(0, 10, 0.1, 0.5, "exponential") - 0.1) < 1e-6
    assert abs(get_alpha_schedule(9, 10, 0.1, 0.5, "exponential") - 0.5) < 1e-6


def test_alpha_schedule_invalid() -> None:
    with pytest.raises(ValueError, match="alpha values must be non-negative"):
        get_alpha_schedule(0, 10, alpha_min=-1.0)
    with pytest.raises(ValueError, match="alpha values must be <= 1.0"):
        get_alpha_schedule(0, 10, alpha_min=1.5)
    with pytest.raises(ValueError, match="alpha_max must be >= alpha_min"):
        get_alpha_schedule(0, 10, alpha_min=0.5, alpha_max=0.1)
    with pytest.raises(ValueError, match="max_epochs must be >= 1"):
        get_alpha_schedule(0, 0)
    with pytest.raises(ValueError, match="schedule_type must be one of"):
        get_alpha_schedule(0, 10, schedule_type="invalid")


def test_compute_efficiency_metrics() -> None:
    from src.loss import compute_efficiency_metrics
    masks = torch.tensor([[0.9, 0.1], [0.6, 0.4]])
    out = compute_efficiency_metrics(masks)
    assert out["active_heads"] == 1.0  # (1+1)/2
    assert out["sparsity"] == 0.5
