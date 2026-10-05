"""
attention.py — Attention Mechanisms from Scratch
==================================================

Student implements:
  - dot_product_attention(): raw scaled dot-product
  - masked_attention(): with causal and/or padding masks
  - MultiHeadAttention: full multi-head with Q/K/V projections

No nn.MultiheadAttention allowed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────
# Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────

def dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

    Parameters
    ----------
    query : (..., seq_q, d_k)
    key :   (..., seq_k, d_k)
    value : (..., seq_k, d_v)
    mask :  (..., seq_q, seq_k) — True/1 where attention is BLOCKED
            (positions to mask get -inf before softmax)
    dropout_p : float
    training : bool

    Returns
    -------
    output : (..., seq_q, d_v) — attention-weighted values
    weights : (..., seq_q, seq_k) — attention weights (after softmax)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. d_k = query.shape[-1]                              ║
    ║  2. scores = query @ key.transpose(-2, -1) / √d_k     ║
    ║  3. If mask is not None:                               ║
    ║     scores = scores.masked_fill(mask, float('-inf'))   ║
    ║  4. weights = softmax(scores, dim=-1)                  ║
    ║  5. If dropout_p > 0 and training:                     ║
    ║     weights = F.dropout(weights, p=dropout_p)          ║
    ║  6. output = weights @ value                           ║
    ║  7. return output, weights                             ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement dot_product_attention()")


# ─────────────────────────────────────────────────────
# Mask Utilities
# ─────────────────────────────────────────────────────

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (autoregressive) attention mask.

    Returns a boolean mask of shape (seq_len, seq_len) where
    True = position is BLOCKED (future tokens).

    Example for seq_len=4:
        [[False,  True,  True,  True],
         [False, False,  True,  True],
         [False, False, False,  True],
         [False, False, False, False]]

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  Hint: torch.triu(torch.ones(n, n), diagonal=1).bool()║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement create_causal_mask()")


def create_padding_mask(
    lengths: torch.Tensor, max_len: int
) -> torch.Tensor:
    """
    Create a padding mask from sequence lengths.

    Parameters
    ----------
    lengths : (batch,) — actual length of each sequence
    max_len : int — maximum sequence length

    Returns
    -------
    mask : (batch, max_len) — True where position is PADDING

    Example: lengths=[3, 5], max_len=6
        [[False, False, False,  True,  True,  True],
         [False, False, False, False, False,  True]]

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  Hint: torch.arange(max_len) >= lengths.unsqueeze(1)  ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement create_padding_mask()")


# ─────────────────────────────────────────────────────
# Multi-Head Attention
# ─────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention from scratch.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
    where head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)

    Parameters
    ----------
    d_model : int — model dimension
    num_heads : int — number of attention heads
    dropout : float

    Constraint: d_model must be divisible by num_heads.
    d_k = d_v = d_model // num_heads

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. W_q, W_k, W_v: Linear(d_model, d_model)          ║
    ║  2. W_o: Linear(d_model, d_model)                     ║
    ║                                                       ║
    ║  forward(query, key, value, mask=None):                ║
    ║  1. Project: Q = query @ W_q, K = key @ W_k, etc.     ║
    ║  2. Reshape for heads:                                 ║
    ║     (batch, seq, d_model) → (batch, heads, seq, d_k)  ║
    ║  3. Apply dot_product_attention to each head           ║
    ║  4. Concatenate heads back                             ║
    ║     (batch, heads, seq, d_k) → (batch, seq, d_model)  ║
    ║  5. Final projection: output = concat @ W_o            ║
    ║  6. Return output, attention_weights                   ║
    ║                                                       ║
    ║  Mask handling:                                        ║
    ║  If mask is (seq, seq), unsqueeze to (1, 1, seq, seq)  ║
    ║  to broadcast across batch and heads.                  ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        raise NotImplementedError("TODO: implement MultiHeadAttention.__init__()")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : (batch, seq_q, d_model)
        key :   (batch, seq_k, d_model)
        value : (batch, seq_k, d_model)
        mask :  broadcastable to (batch, heads, seq_q, seq_k)

        Returns
        -------
        output : (batch, seq_q, d_model)
        attn_weights : (batch, heads, seq_q, seq_k)
        """
        raise NotImplementedError("TODO: implement MultiHeadAttention.forward()")


# ─────────────────────────────────────────────────────
# Additive (Bahdanau) Attention — for Seq2Seq
# ─────────────────────────────────────────────────────

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention for Seq2Seq models.

    score(s_t, h_j) = v^T @ tanh(W_s @ s_t + W_h @ h_j)

    Parameters
    ----------
    decoder_dim : int — decoder hidden size
    encoder_dim : int — encoder hidden size
    attention_dim : int — internal attention dimension

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. W_s: Linear(decoder_dim, attention_dim, bias=False)║
    ║  2. W_h: Linear(encoder_dim, attention_dim, bias=False)║
    ║  3. v: Linear(attention_dim, 1, bias=False)            ║
    ║                                                       ║
    ║  forward(decoder_hidden, encoder_outputs):             ║
    ║  1. decoder_hidden: (batch, decoder_dim)               ║
    ║  2. encoder_outputs: (batch, src_len, encoder_dim)     ║
    ║  3. score = v(tanh(W_s(dec) + W_h(enc)))               ║
    ║     → (batch, src_len, 1) → squeeze → (batch, src_len)║
    ║  4. weights = softmax(score, dim=-1)                   ║
    ║  5. context = weights.unsqueeze(1) @ encoder_outputs   ║
    ║     → (batch, 1, encoder_dim) → squeeze                ║
    ║  6. return context, weights                            ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, decoder_dim: int, encoder_dim: int, attention_dim: int):
        super().__init__()
        raise NotImplementedError("TODO: implement AdditiveAttention.__init__()")

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (context, attention_weights)."""
        raise NotImplementedError("TODO: implement AdditiveAttention.forward()")


# ─────────────────────────────────────────────────────
# Visualization (provided)
# ─────────────────────────────────────────────────────

def plot_attention_weights(
    weights: torch.Tensor,
    x_labels: list = None,
    y_labels: list = None,
    title: str = "Attention Weights",
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot attention weights as a heatmap. Provided utility.

    Parameters
    ----------
    weights : (seq_q, seq_k) or (heads, seq_q, seq_k)
    """
    w = weights.detach().cpu().numpy()
    if w.ndim == 3:
        num_heads = w.shape[0]
        fig, axes = plt.subplots(1, num_heads, figsize=(figsize[0] * num_heads // 2, figsize[1]))
        if num_heads == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            im = ax.imshow(w[i], cmap="Blues", aspect="auto")
            ax.set_title(f"Head {i}")
            if x_labels:
                ax.set_xticks(range(len(x_labels)))
                ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            if y_labels:
                ax.set_yticks(range(len(y_labels)))
                ax.set_yticklabels(y_labels, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(w, cmap="Blues", aspect="auto")
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
        plt.colorbar(im, ax=ax)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
