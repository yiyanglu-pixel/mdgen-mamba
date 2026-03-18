# Copyright (c) 2024 MDGEN Mamba Integration
# Mamba and Bi-Mamba operators for temporal sequence modeling
"""
Mamba-based operators for MDGEN temporal modeling.

Supports both Mamba-1 (selective scan) and Mamba-2 (SSD, chunk-wise parallel).
These modules provide O(n) linear-complexity alternatives to the standard
O(n²) self-attention mechanism for processing long MD trajectories.
"""

import torch
import torch.nn as nn
from typing import Optional

# Import Mamba v1 and v2 with graceful fallback
MAMBA_AVAILABLE = False
MAMBA2_AVAILABLE = False
Mamba = None
Mamba2 = None

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    pass

try:
    from mamba_ssm import Mamba2
    MAMBA2_AVAILABLE = True
except ImportError:
    pass


def _get_mamba_cls(version: int):
    """Return the appropriate Mamba class for the requested version."""
    if version == 2:
        if MAMBA2_AVAILABLE:
            return Mamba2
        if MAMBA_AVAILABLE:
            import warnings
            warnings.warn(
                "Mamba2 not available, falling back to Mamba v1. "
                "Install mamba-ssm >= 2.0 for Mamba-2 support.",
                stacklevel=3,
            )
            return Mamba
    elif version == 1:
        if MAMBA_AVAILABLE:
            return Mamba
    raise ImportError(
        f"Mamba v{version} is not installed. Please install with: "
        "pip install mamba-ssm causal-conv1d"
    )


def _build_mamba(version: int, d_model: int, d_state: int, d_conv: int,
                 expand: int, headdim: int):
    """Instantiate a Mamba v1 or v2 block with the right kwargs."""
    cls = _get_mamba_cls(version)
    if cls is Mamba2:
        # Mamba-2 constraint: d_inner (= d_model * expand) must be divisible by headdim
        d_inner = d_model * expand
        if d_inner % headdim != 0:
            # Auto-adjust headdim down to the largest factor <= requested headdim
            for hd in range(headdim, 0, -1):
                if d_inner % hd == 0:
                    headdim = hd
                    break
        return cls(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
    else:
        # Mamba v1 — does not accept headdim
        return cls(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )


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
        mamba_version: 1 for Mamba-1, 2 for Mamba-2 (default)
        headdim: Head dimension for Mamba-2 grouped SSD (ignored by v1)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        mamba_version: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.mamba_version = mamba_version
        self.mamba = _build_mamba(mamba_version, d_model, d_state, d_conv,
                                 expand, headdim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C)
            mask: Optional mask of shape (B, T). Padding positions (0) are zeroed out.

        Returns:
            Output tensor of shape (B, T, C)
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

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
        mamba_version: 1 for Mamba-1, 2 for Mamba-2 (default)
        headdim: Head dimension for Mamba-2 (ignored by v1)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        combine_mode: str = 'concat',
        mamba_version: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.combine_mode = combine_mode
        self.mamba_version = mamba_version

        self.mamba_forward = _build_mamba(mamba_version, d_model, d_state,
                                         d_conv, expand, headdim)
        self.mamba_backward = _build_mamba(mamba_version, d_model, d_state,
                                          d_conv, expand, headdim)

        if combine_mode == 'concat':
            self.combine_proj = nn.Linear(2 * d_model, d_model)
        elif combine_mode == 'gate':
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
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

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
            out = (out_forward + out_backward) / 2
        elif self.combine_mode == 'concat':
            out = torch.cat([out_forward, out_backward], dim=-1)
            out = self.combine_proj(out)
        elif self.combine_mode == 'gate':
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
        mamba_version: 1 for Mamba-1, 2 for Mamba-2 (default)
        headdim: Head dimension for Mamba-2 (ignored by v1)
    """

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        combine_mode: str = 'concat',
        mamba_version: int = 2,
        headdim: int = 64,
        **kwargs  # Absorb unused args like num_heads, add_bias_kv, etc.
    ):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional

        mamba_kwargs = dict(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, mamba_version=mamba_version, headdim=headdim,
        )

        if bidirectional:
            self.mamba = BiMambaOperator(combine_mode=combine_mode, **mamba_kwargs)
        else:
            self.mamba = MambaOperator(**mamba_kwargs)

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


def check_mamba_available(version: int = 2) -> bool:
    """Check if Mamba is available for import."""
    if version == 2:
        return MAMBA2_AVAILABLE or MAMBA_AVAILABLE
    return MAMBA_AVAILABLE
