"""
Microglia-Inspired Dynamic Pruning for Reasoning Models

Authors: Tommaso R. Marena (The Catholic University of America)
         Panos Ketonis (Yale University)

Copyright (c) 2026
"""
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn

from .statistics import NUM_STATS_PER_HEAD


class MicrogliaAgent(nn.Module):
    """Small MLP that learns which attention heads to prune.
    
    Takes activation statistics as input and outputs soft masks (0-1 values)
    for each attention head. The agent learns to identify unimportant heads
    that can be pruned without hurting task performance.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        temperature: float = 1.0,
        num_layers: int = 1,
        layer_idx: int = 0,
    ):
        """Initializes the MicrogliaAgent.

        Args:
            hidden_dim (int): Hidden dimension of the MLP.
            num_heads (int): Number of attention heads in the layer.
            temperature (float): Temperature for sigmoid activation (lower = more binary masks).
            num_layers (int): Total number of layers for positional encoding.
            layer_idx (int): Layer index for this agent (0-indexed).
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0. Fix by: pass a positive value such as 1.0.")
        if temperature < 0.1:
            warnings.warn(
                "temperature < 0.1 may saturate sigmoid gates and harm gradient flow. "
                "Fix by: using temperature >= 0.1 for stable training.",
                UserWarning,
            )
        self.num_heads: int = num_heads
        self.temperature: float = temperature
        
        # Input: 6 statistics per head + layer-aware positional encoding
        input_dim: int = NUM_STATS_PER_HEAD * num_heads

        self.num_layers: int = max(1, num_layers)
        self.layer_idx: int = int(layer_idx)

        self.fc1: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.fc3: nn.Linear = nn.Linear(hidden_dim, num_heads)

        self.act: nn.GELU = nn.GELU()
        self.norm1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.norm2: nn.LayerNorm = nn.LayerNorm(hidden_dim)

        self.positional_projection: nn.Linear = nn.Linear(2, hidden_dim)
        
    def _layer_positional_encoding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create sinusoidal encoding from normalized layer position."""
        position = torch.tensor(self.layer_idx / max(self.num_layers - 1, 1), device=device, dtype=torch.float32)
        encoding = torch.stack([torch.sin(math.pi * position), torch.cos(math.pi * position)], dim=0)
        return encoding.unsqueeze(0).expand(batch_size, -1)

    def forward(self, activation_stats: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Predicts pruning masks from activation statistics.
        
        Args:
            activation_stats (torch.Tensor): Tensor of shape (batch, 6*num_heads) containing
                activation and attention statistics for each head.
            layer_idx (Optional[int]): Optional layer index override used for positional encoding.
                            
        Returns:
            torch.Tensor: Masks of shape (batch, num_heads) with values in [0, 1].
                Values close to 1 mean "keep this head", close to 0 mean "prune it".
        """
        if layer_idx is not None:
            self.layer_idx = int(layer_idx)

        x = self.fc1(activation_stats)

        pos_encoding = self._layer_positional_encoding(activation_stats.shape[0], activation_stats.device)
        x = x + self.positional_projection(pos_encoding).to(x.dtype)

        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x + residual

        residual2 = x
        x = self.norm2(x)
        x = self.act(x)
        logits = self.fc3(x + residual2)
        
        # Soft gating: sigmoid with temperature scaling
        masks = torch.sigmoid(logits / self.temperature)
        
        return masks
    
    def set_temperature(self, temperature: float) -> None:
        """Updates the temperature parameter.
        
        Lower temperatures produce more binary masks (closer to 0 or 1).
        Higher temperatures produce softer masks (closer to 0.5).

        Args:
            temperature (float): The new temperature value.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0. Fix by: pass a positive value such as 1.0.")
        if temperature < 0.1:
            warnings.warn(
                "temperature < 0.1 may saturate sigmoid gates and harm gradient flow. "
                "Fix by: using temperature >= 0.1 for stable training.",
                UserWarning,
            )
        self.temperature = temperature
