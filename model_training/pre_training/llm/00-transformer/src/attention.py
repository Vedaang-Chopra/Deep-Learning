### Implementing the Self Attention Block

import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from typing import NamedTuple, Optional


# ─────────────────────────────────────────────────────────────
#  Return type
# ─────────────────────────────────────────────────────────────
class AttentionOutput(NamedTuple):
    out: torch.Tensor          # (B, T, d_out)
    weights: torch.Tensor      # (B, [h,] T, T)  — for visualization



# ─────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────
def causal_mask(
    T: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Lower-triangular causal mask.

    Args:
        T: sequence length.
        device: target device.
        dtype: output dtype (bool is most memory-efficient for masked_fill).

    Returns:
        (T, T) tensor.  1/True = allowed,  0/False = blocked.
        Broadcasts to (B, T, T) and (B, h, T, T).
    """
    return torch.tril(torch.ones(T, T, device=device, dtype=dtype))




#  SingleHeadAttention  (standalone building block)
# ─────────────────────────────────────────────────────────────
class SingleHeadAttention(nn.Module):
    """One attention head.

    Projects input to Q, K, V, computes scaled dot-product attention,
    and returns the weighted value aggregation.

    Args:
        d_in:  input embedding dimension (what this head *receives*).
        d_out: Q/K/V/output dimension (what this head *produces per token*).
        dropout: dropout probability applied to attention weights.

    Standalone usage:
        SingleHeadAttention(d_model, d_model)   # full-width head

    Inside MultiHeadAttention:
        SingleHeadAttention(d_model, head_size)  # one slice
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError(f"d_in and d_out must be positive, got {d_in}, {d_out}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_in = d_in
        self.d_out = d_out
        self._scale = d_out ** -0.5          # precompute 1 / sqrt(d_out)

        self._w_query = nn.Linear(d_in, d_out, bias=False)
        self._w_key = nn.Linear(d_in, d_out, bias=False)
        self._w_value = nn.Linear(d_in, d_out, bias=False)
        self._dropout = nn.Dropout(dropout)
 
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> AttentionOutput:
        """Compute single-head attention.

        Args:
            x:    (B, T, d_in)
            mask: optional broadcastable tensor.
                  True/1 = keep,  False/0 = mask out.
                  Typical shapes: (T, T) for causal, (B, 1, 1, T) for padding.

        Returns:
            AttentionOutput with:
                out:     (B, T, d_out)
                weights: (B, T, T)
        """
        B, T, _ = x.shape

        q = self._w_query(x)                      # (B, T, d_out)
        k = self._w_key(x)                        # (B, T, d_out)
        v = self._w_value(x)                      # (B, T, d_out)

        # Scaled dot-product
        scores = (q @ k.transpose(-2, -1)) * self._scale   # (B, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = self._dropout(F.softmax(scores, dim=-1))  # (B, T, T)

        out = weights @ v                     # (B, T, d_out)
        return AttentionOutput(out=out, weights=weights)



# ─────────────────────────────────────────────────────────────
#  MultiHeadAttention  (composed from SingleHeadAttention)
# ─────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention, built from `n_heads` SingleHeadAttention blocks.

    Each head receives the full `d_model` input and produces a `head_size`-dim
    output.  Heads are concatenated and passed through a single output
    projection `W_O : (d_model → d_model)`.

    Args:
        d_model:  model (embedding) dimension.
        n_heads:  number of parallel attention heads.
        dropout:  dropout probability for attention weights.

    Raises:
        ValueError: if d_model is not divisible by n_heads.
    """
    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        
        ### THis is done to ensure proper matrix splitting
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        
        self.head_size = d_model // n_heads
        
        # ── The actual attention heads ──
        self.heads = nn.ModuleList([
            SingleHeadAttention(
                d_in=d_model,       # each head sees the full embedding
                d_out=self.head_size,  # but operates in a sub-space
                dropout=dropout,
            )
            for _ in range(n_heads)
        ])
        
        # ── Output projection (applied after concatenation) ──
        self._w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> AttentionOutput:
        """Compute multi-head attention.

        Args:
            x:    (B, T, d_model)
            mask: optional, broadcastable to (B, n_heads, T, T).
                e.g. causal_mask(T) of shape (T, T) works for all B and h.

        Returns:
            AttentionOutput with:
                out:     (B, T, d_model)
                weights: (B, n_heads, T, T)
        """
        B, T, _ = x.shape

        # Run every head in parallel (same input, different sub-space)
        head_outputs = [head(x, mask=mask) for head in self.heads]

        # Concatenate along the feature dim:
        #   each: (B, T, head_size)  →  stack: (B, T, n_heads * head_size)
        #                                =     (B, T, d_model)
        concat = torch.cat([h.out for h in head_outputs], dim=-1)  # (B, T, d_model)
        
        out = self._w_o(concat)                                       # (B, T, d_model)

        # Stack weights for visualisation: (n_heads, B, T, T) → (B, n_heads, T, T)
        weights = torch.stack([h.weights for h in head_outputs], dim=1)

        return AttentionOutput(out=out, weights=weights)
