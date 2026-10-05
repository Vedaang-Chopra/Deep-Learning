"""
transformer_blocks.py — Transformer Building Blocks from Scratch
==================================================================

Student implements:
  - SinusoidalPositionalEncoding: fixed sinusoidal PE
  - TransformerBlock: pre-norm transformer layer
  - TransformerEncoder: stack of TransformerBlocks

No nn.Transformer allowed. Uses MultiHeadAttention from attention.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ─────────────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Parameters
    ----------
    d_model : int — model dimension
    max_len : int — maximum sequence length
    dropout : float

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. Compute PE matrix: (1, max_len, d_model)          ║
    ║     pos = torch.arange(max_len).unsqueeze(1)          ║
    ║     div = torch.exp(arange(0, d_model, 2) *           ║
    ║                     -log(10000) / d_model)             ║
    ║     pe[:, :, 0::2] = sin(pos * div)                   ║
    ║     pe[:, :, 1::2] = cos(pos * div)                   ║
    ║  2. Register as buffer (not a parameter)              ║
    ║  3. Dropout layer                                      ║
    ║                                                       ║
    ║  forward(x):                                          ║
    ║    x: (batch, seq_len, d_model)                        ║
    ║    return dropout(x + pe[:, :seq_len, :])              ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        raise NotImplementedError("TODO: implement SinusoidalPositionalEncoding.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement SinusoidalPositionalEncoding.forward()")


# ─────────────────────────────────────────────────────
# Position-wise Feed-Forward Network
# ─────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    FFN(x) = Linear(GELU(Linear(x)))

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║    Linear(d_model, d_ff) → GELU → Dropout →           ║
    ║    Linear(d_ff, d_model) → Dropout                     ║
    ║                                                       ║
    ║  forward(x):                                          ║
    ║    Pass through the FFN                                ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        raise NotImplementedError("TODO: implement FeedForward.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement FeedForward.forward()")


# ─────────────────────────────────────────────────────
# Transformer Block (Pre-Norm)
# ─────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    A single Transformer block with Pre-LayerNorm.

    Pre-Norm Architecture:
        x → LayerNorm → MultiHeadAttention → + (residual) →
          → LayerNorm → FeedForward → + (residual) → output

    Parameters
    ----------
    d_model : int
    num_heads : int
    d_ff : int — feed-forward inner dimension (typically 4 * d_model)
    dropout : float
    is_decoder : bool — if True, uses causal masking

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. LayerNorm for attention sub-layer                  ║
    ║  2. MultiHeadAttention (from attention.py)             ║
    ║  3. LayerNorm for FFN sub-layer                        ║
    ║  4. FeedForward                                        ║
    ║  5. Dropout for residual                               ║
    ║                                                       ║
    ║  forward(x, mask=None):                               ║
    ║  1. Residual attention:                                ║
    ║     normed = LayerNorm(x)                              ║
    ║     attn_out, weights = MHA(normed, normed, normed,    ║
    ║                             mask=mask)                 ║
    ║     x = x + dropout(attn_out)                          ║
    ║  2. Residual FFN:                                      ║
    ║     normed = LayerNorm(x)                              ║
    ║     ff_out = FeedForward(normed)                       ║
    ║     x = x + dropout(ff_out)                            ║
    ║  3. Return x, weights                                  ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.is_decoder = is_decoder
        raise NotImplementedError("TODO: implement TransformerBlock.__init__()")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        """Returns (output, attention_weights)."""
        raise NotImplementedError("TODO: implement TransformerBlock.forward()")


# ─────────────────────────────────────────────────────
# Transformer Encoder (stack of blocks)
# ─────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Stack of TransformerBlocks for sequence classification.

    Architecture:
        Embedding → PositionalEncoding → N × TransformerBlock →
        [CLS] pooling or mean pooling → Linear(d_model, num_classes)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  Use nn.ModuleList for the transformer blocks.        ║
    ║  For classification, take the mean of all positions   ║
    ║  and project to num_classes.                           ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        num_classes: int = 10,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        raise NotImplementedError("TODO: implement TransformerEncoder.__init__()")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x : (batch, seq_len) — token indices

        Returns
        -------
        logits : (batch, num_classes)
        """
        raise NotImplementedError("TODO: implement TransformerEncoder.forward()")
