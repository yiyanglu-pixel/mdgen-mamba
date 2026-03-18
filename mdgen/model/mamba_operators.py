# Copyright (c) 2024 MDGEN Mamba Integration
# Mamba and Bi-Mamba operators for temporal sequence modeling
"""
Mamba-based operators for MDGEN temporal modeling.

These modules provide O(n) linear-complexity alternatives to the standard
O(n²) self-attention mechanism for processing long MD trajectories.
"""

import torch
import torch.nn as nn
from typing import Optional

# Try to import Mamba components - use Mamba v1 instead of Mamba2 for better compatibility
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None


class MambaOperator(nn.Module):
    """Mamba wrapper for temporal modeling in MDGEN.

    Replaces mha_t with linear-complexity SSM.
    Input: (B*L, T, C) - batch*residues, frames, channels
    Output: (B*L, T, C)

    Args:
        d_model: Model dimension (embed_dim)
        d_state: SSM state dimension (default 64, aligned with parsing.py)
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "Mamba is not installed. Please install with: "
                "pip install mamba-ssm causal-conv1d"
            )

        self.d_model = d_model
        # Use Mamba v1 instead of Mamba2 for better compatibility
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C)
            mask: Optional mask of shape (B, T). Padding positions (0) are zeroed out.

        Returns:
            Output tensor of shape (B, T, C)
        """
        # Zero out padding positions so SSM state is not polluted
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        # Mamba v1 expects (B, L, D) and returns same shape
        # Ensure contiguous for CUDA
        if not x.is_contiguous():
            x = x.contiguous()
        return self.mamba(x)


class BiMambaOperator(nn.Module):
    """Bidirectional Mamba for non-causal temporal modeling.

    Processes sequences in both forward and backward directions,
    then combines the outputs. Essential for molecular dynamics where
    causality is not enforced - past and future frames both matter.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (default 64, aligned with parsing.py)
        d_conv: Convolution kernel size
        expand: Expansion factor
        combine_mode: How to combine forward/backward outputs ('add', 'concat', 'gate')
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        combine_mode: str = 'concat',
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "Mamba is not installed. Please install with: "
                "pip install mamba-ssm causal-conv1d"
            )

        self.d_model = d_model
        self.combine_mode = combine_mode

        # Forward Mamba (v1)
        self.mamba_forward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Backward Mamba (v1)
        self.mamba_backward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        if combine_mode == 'concat':
            # Project concatenated outputs back to d_model
            self.combine_proj = nn.Linear(2 * d_model, d_model)
        elif combine_mode == 'gate':
            # Learnable gating between forward and backward
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C)
            mask: Optional mask of shape (B, T). Padding positions (0) are zeroed out.

        Returns:
            Output tensor of shape (B, T, C)
        """
        # Zero out padding positions so SSM state is not polluted
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        # Mamba v1 has fewer stride restrictions than Mamba2
        # Just ensure contiguous for CUDA
        if not x.is_contiguous():
            x = x.contiguous()

        # Forward pass
        out_forward = self.mamba_forward(x)

        # Backward pass: flip sequence, process, flip back
        x_backward = torch.flip(x, dims=[1]).contiguous()
        out_backward = self.mamba_backward(x_backward)
        out_backward = torch.flip(out_backward, dims=[1])

        # Combine forward and backward
        if self.combine_mode == 'add':
            # Simple averaging
            out = (out_forward + out_backward) / 2
        elif self.combine_mode == 'concat':
            # Concatenate and project
            out = torch.cat([out_forward, out_backward], dim=-1)
            out = self.combine_proj(out)
        elif self.combine_mode == 'gate':
            # Learnable gating
            combined = torch.cat([out_forward, out_backward], dim=-1)
            gate = self.gate(combined)
            out = gate * out_forward + (1 - gate) * out_backward
        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")

        return out


class MambaTemporalBlock(nn.Module):
    """Drop-in replacement for AttentionWithRoPE used in LatentMDGenLayer.

    Provides the same interface as AttentionWithRoPE but uses Mamba internally.
    Mamba has its own positional modeling via selective scan, so no external
    positional encoding (e.g. RoPE) is needed.

    Args:
        d_model: Same as embed_dim in AttentionWithRoPE
        bidirectional: Whether to use BiMamba (True) or unidirectional Mamba (False)
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        combine_mode: For BiMamba, how to combine directions
    """

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        combine_mode: str = 'concat',
        **kwargs  # Absorb unused args like num_heads, add_bias_kv, etc.
    ):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional

        if bidirectional:
            self.mamba = BiMambaOperator(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                combine_mode=combine_mode,
            )
        else:
            self.mamba = MambaOperator(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C)
            mask: Optional attention mask of shape (B, T)

        Returns:
            Output tensor of shape (B, T, C)
        """
        return self.mamba(x, mask)


# Keep old name as alias for backward compatibility
MambaWithRoPE = MambaTemporalBlock


def check_mamba_available() -> bool:
    """Check if Mamba is available for import."""
    return MAMBA_AVAILABLE
