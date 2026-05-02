"""
Microglia-Inspired Dynamic Pruning for Reasoning Models

Authors: Tommaso R. Marena (The Catholic University of America)
         Panos Ketonis (Yale University)

Copyright (c) 2026
"""
from typing import Optional

import torch


NUM_STATS_PER_HEAD: int = 6


def _compute_gradient_magnitude(
    attn_weights: torch.Tensor,
    task_loss: Optional[torch.Tensor] = None,
    attn_grads: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-head gradient magnitudes with autograd fallback."""
    if attn_grads is None and task_loss is not None and attn_weights.requires_grad:
        attn_grads = torch.autograd.grad(
            outputs=task_loss,
            inputs=attn_weights,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

    if attn_grads is None:
        return torch.zeros(
            attn_weights.shape[0],
            attn_weights.shape[1],
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )

    return attn_grads.abs().mean(dim=(-2, -1))


def _compute_cross_head_correlation(hidden_states_heads: torch.Tensor) -> torch.Tensor:
    """Compute mean absolute correlation of each head to all other heads."""
    batch_size, seq_len, num_heads, head_dim = hidden_states_heads.shape
    flattened = hidden_states_heads.permute(0, 2, 1, 3).reshape(batch_size, num_heads, seq_len * head_dim)
    centered = flattened - flattened.mean(dim=-1, keepdim=True)
    norm = centered.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = centered / norm

    corr_matrix = torch.matmul(normalized, normalized.transpose(-1, -2))
    eye = torch.eye(num_heads, device=hidden_states_heads.device, dtype=hidden_states_heads.dtype).unsqueeze(0)
    corr_without_diag = corr_matrix.abs() * (1.0 - eye)
    return corr_without_diag.sum(dim=-1) / max(num_heads - 1, 1)


def compute_layer_stats(
    hidden_states: torch.Tensor,
    attn_weights: torch.Tensor,
    task_loss: Optional[torch.Tensor] = None,
    attn_grads: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-head statistics from layer activations.
    
    We compute six key statistics for each attention head:
    1. Activation Norm (Mean): Magnitude of hidden state activity.
    2. Activation Std: Variability of hidden state activity.
    3. Attention Entropy: Spread of attention distribution.
    4. Max Attention: Peak attention score (indicates focusing on specific tokens).
    5. Gradient Magnitude: Sensitivity of task loss to each head.
    6. Cross-head Correlation: Redundancy with other heads.
    
    Args:
        hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
        attn_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
        
    Returns:
        stats: Tensor of shape (batch, 6*num_heads) containing concatenated
               activation statistics and attention metrics for each head.
    """
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    batch_size = hidden_states.shape[0]
    num_heads = attn_weights.shape[1]
    
    # Reshape hidden states to separate heads
    # Assuming hidden_dim = num_heads * head_dim
    hidden_dim = hidden_states.shape[-1]
    head_dim = hidden_dim // num_heads
    
    # Reshape: (batch, seq_len, hidden_dim) -> (batch, seq_len, num_heads, head_dim)
    hidden_states_heads = hidden_states.view(batch_size, -1, num_heads, head_dim)
    
    # Statistic 1: Per-head activation norms (Mean)
    # Shape: (batch, num_heads)
    norms = hidden_states_heads.norm(dim=-1)
    act_norms_mean = norms.mean(dim=1)
    
    # Statistic 2: Per-head activation norms (Std)
    act_norms_std = norms.std(dim=1)

    # Statistic 3: Attention entropy per head
    # Use torch.special.entr for better performance and stability
    # Shape: (batch, num_heads, seq_len, seq_len) -> (batch, num_heads)
    entropy = torch.special.entr(attn_weights).sum(dim=-1).mean(dim=-1)
    
    # Statistic 4: Max attention per head
    # Use torch.amax to avoid unnecessary index computation
    max_attn = torch.amax(attn_weights, dim=-1).mean(dim=-1)

    gradient_magnitude = _compute_gradient_magnitude(attn_weights, task_loss=task_loss, attn_grads=attn_grads)
    cross_head_corr = _compute_cross_head_correlation(hidden_states_heads)

    # Concatenate statistics
    # Shape: (batch, 6*num_heads)
    stats = torch.cat(
        [act_norms_mean, act_norms_std, entropy, max_attn, gradient_magnitude, cross_head_corr],
        dim=-1,
    )
    
    return stats


def compute_head_importance(
    hidden_states: torch.Tensor,
    attn_weights: torch.Tensor,
    task_loss: torch.Tensor,
) -> torch.Tensor:
    """Compute head importance scores based on gradient magnitudes.
    
    This is used during training to identify which heads contribute most
    to task performance. Higher importance = more critical for accuracy.
    
    Args:
        hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
        attn_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
        task_loss: Scalar loss tensor (must have gradients)
        
    Returns:
        importance: Tensor of shape (batch, num_heads) with importance scores
    """
    # Compute gradients of loss w.r.t. attention weights
    if attn_weights.requires_grad:
        importance = _compute_gradient_magnitude(attn_weights, task_loss=task_loss)
    else:
        # Fallback: use activation norms as proxy
        batch_size = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[-1]
        num_heads = attn_weights.shape[1]
        head_dim = hidden_dim // num_heads
        
        hidden_states_heads = hidden_states.view(batch_size, -1, num_heads, head_dim)
        importance = hidden_states_heads.norm(dim=-1).mean(dim=1)
    
    return importance
