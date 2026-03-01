"""Pruned attention wrapper with dynamic masking."""

import torch
import torch.nn as nn
from .statistics import compute_layer_stats
from .agent import MicrogliaAgent


from typing import Tuple, Optional

class PrunedAttention(nn.Module):
    """Wraps an attention layer with learned dynamic pruning mechanisms.
    
    During the forward pass, this module:
    1. Executes the standard attention mechanism.
    2. Computes per-head activation statistics.
    3. Utilizes a MicrogliaAgent to predict head-level pruning masks.
    4. Applies these masks to the attention output.
    """
    
    def __init__(self, original_attn: nn.Module, agent: MicrogliaAgent, hard_prune: bool = False):
        """Initializes the PrunedAttention module.

        Args:
            original_attn (nn.Module): The original attention layer to wrap.
            agent (MicrogliaAgent): The agent network that predicts pruning masks.
            hard_prune (bool): Whether to apply a hard binary threshold (0.5) to masks.
        """
        super().__init__()
        self.attn: nn.Module = original_attn
        self.agent: MicrogliaAgent = agent
        self.hard_prune: bool = hard_prune
        self.last_masks: Optional[torch.Tensor] = None
        self.last_stats: Optional[torch.Tensor] = None
        self.enable_pruning: bool = False
        self.budget_keep_ratio: Optional[float] = None

    def set_budget_keep_ratio(self, keep_ratio: Optional[float]) -> None:
        """Set per-input keep ratio budget for this layer."""
        self.budget_keep_ratio = keep_ratio

    def _apply_budget(self, masks: torch.Tensor) -> torch.Tensor:
        """Project soft masks onto a fixed top-k support when budget is set."""
        if self.budget_keep_ratio is None:
            return masks

        keep_ratio = float(max(0.0, min(1.0, self.budget_keep_ratio)))
        num_heads = masks.shape[1]
        keep_k = max(1, int(round(keep_ratio * num_heads)))
        if keep_k >= num_heads:
            return masks

        topk_indices = masks.topk(k=keep_k, dim=1).indices
        budget_mask = torch.zeros_like(masks)
        budget_mask.scatter_(1, topk_indices, 1.0)
        return masks * budget_mask
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs) -> Tuple:
        """Forward pass with optional dynamic head pruning.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (torch.Tensor, optional): Attention mask.
            position_ids (torch.Tensor, optional): Position IDs.
            past_key_value (Tuple, optional): Cached key/values.
            output_attentions (bool): Whether to return attention weights.
            use_cache (bool): Whether to use KV cache.

        Returns:
            Tuple: Attention output and optional metadata.
        """
        
        # Run original attention
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions or self.enable_pruning,  # Need weights for pruning
            use_cache=use_cache,
            **kwargs
        )
        
        # Only apply pruning if explicitly enabled
        if not self.enable_pruning:
            return attn_outputs
        
        # Unpack outputs (format varies by model)
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
        else:
            attn_output = attn_outputs
            attn_weights = None
        
        # If we have attention weights, compute stats and apply pruning
        if attn_weights is not None:
            try:
                # Store original dtype to cast back later
                original_dtype = hidden_states.dtype

                # Compute per-head statistics
                stats = compute_layer_stats(hidden_states, attn_weights)
                
                # Ensure stats are float32 for the agent
                stats = stats.to(torch.float32)

                # Get pruning masks from agent
                masks = self.agent(stats)  # (batch, num_heads)
                
                # Cast masks back to original dtype for application
                masks = masks.to(original_dtype)

                # Apply hard threshold in eval mode
                if self.hard_prune and not self.training:
                    masks = (masks > 0.5).float()

                masks = self._apply_budget(masks)
                
                # Store for monitoring
                self.last_masks = masks.detach()
                self.last_stats = stats.detach()
                
                # Apply masks to attention output
                batch_size, seq_len, hidden_dim = attn_output.shape
                num_heads = masks.shape[1]
                head_dim = hidden_dim // num_heads
                
                # Reshape to separate heads
                attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
                
                # Broadcast masks: (batch, num_heads) -> (batch, 1, num_heads, 1)
                masks_expanded = masks.unsqueeze(1).unsqueeze(-1)
                
                # Apply masking
                attn_output = attn_output * masks_expanded
                
                # Reshape back
                attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
                
                # Return in original format
                if isinstance(attn_outputs, tuple):
                    return (attn_output,) + attn_outputs[1:]
                else:
                    return attn_output
            except Exception as e:
                # If pruning fails, just return original output
                print(f"Pruning failed: {e}, using original output")
                return attn_outputs
        
        # Return original if no pruning applied
        return attn_outputs
