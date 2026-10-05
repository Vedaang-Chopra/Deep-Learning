"""
rnn_cells.py — RNN & LSTM Cells from First Principles
=======================================================

Student implements:
  - ManualRNNCell: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
  - ManualLSTMCell: full gating mechanism (i, f, o, g gates)
  - RNNLanguageModel: character-level RNN using ManualRNNCell
  - LSTMLanguageModel: character-level LSTM using ManualLSTMCell

No nn.RNN or nn.LSTM allowed.
All gate equations must be written from raw linear transforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────
# Manual RNN Cell
# ─────────────────────────────────────────────────────

class ManualRNNCell(nn.Module):
    """
    A vanilla RNN cell implemented from first principles.

    Equation:
        h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

    Parameters
    ----------
    input_size : int — dimension of input x_t
    hidden_size : int — dimension of hidden state h_t

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. Create nn.Parameter for W_ih: (hidden, input)     ║
    ║  2. Create nn.Parameter for W_hh: (hidden, hidden)    ║
    ║  3. Create nn.Parameter for b_ih: (hidden,)           ║
    ║  4. Create nn.Parameter for b_hh: (hidden,)           ║
    ║  5. Initialize with uniform(-k, k) where k=1/√hidden  ║
    ║                                                       ║
    ║  forward(x_t, h_prev):                                ║
    ║  1. x_t: (batch, input_size)                          ║
    ║  2. h_prev: (batch, hidden_size) or None              ║
    ║  3. If h_prev is None, init to zeros                  ║
    ║  4. h_t = tanh(x_t @ W_ih.T + b_ih +                 ║
    ║               h_prev @ W_hh.T + b_hh)                 ║
    ║  5. return h_t                                         ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        raise NotImplementedError("TODO: implement ManualRNNCell.__init__()")

    def forward(
        self, x_t: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (batch, input_size) — input at time t
        h_prev : (batch, hidden_size) or None — previous hidden state

        Returns
        -------
        h_t : (batch, hidden_size) — new hidden state
        """
        raise NotImplementedError("TODO: implement ManualRNNCell.forward()")


# ─────────────────────────────────────────────────────
# Manual LSTM Cell
# ─────────────────────────────────────────────────────

class ManualLSTMCell(nn.Module):
    """
    An LSTM cell implemented from first principles.

    Equations:
        i_t = σ(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)   ← input gate
        f_t = σ(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)   ← forget gate
        g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg) ← candidate
        o_t = σ(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)   ← output gate

        c_t = f_t * c_{t-1} + i_t * g_t                        ← cell state
        h_t = o_t * tanh(c_t)                                   ← hidden state

    Parameters
    ----------
    input_size : int
    hidden_size : int

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  Efficient approach: concatenate all 4 gates into     ║
    ║  single weight matrices:                               ║
    ║    W_ih: (4*hidden, input)  — all input projections    ║
    ║    W_hh: (4*hidden, hidden) — all hidden projections   ║
    ║    b_ih: (4*hidden,)                                   ║
    ║    b_hh: (4*hidden,)                                   ║
    ║                                                       ║
    ║  Then chunk the result into 4 pieces:                  ║
    ║    gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh      ║
    ║    i, f, g, o = gates.chunk(4, dim=-1)                 ║
    ║    i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)        ║
    ║    g = tanh(g)                                         ║
    ║                                                       ║
    ║  forward(x_t, state):                                  ║
    ║    state = (h_prev, c_prev) or None                    ║
    ║    returns (h_t, c_t)                                   ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        raise NotImplementedError("TODO: implement ManualLSTMCell.__init__()")

    def forward(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_t : (batch, input_size)
        state : (h_prev, c_prev) each (batch, hidden_size), or None

        Returns
        -------
        (h_t, c_t) : both (batch, hidden_size)
        """
        raise NotImplementedError("TODO: implement ManualLSTMCell.forward()")


# ─────────────────────────────────────────────────────
# RNN Language Model
# ─────────────────────────────────────────────────────

class RNNLanguageModel(nn.Module):
    """
    Character-level language model using ManualRNNCell.

    Architecture:
        Embedding(vocab_size, embed_dim)
        ManualRNNCell(embed_dim, hidden_size)  ← unrolled across sequence
        Linear(hidden_size, vocab_size)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  forward(x, h=None):                                  ║
    ║    x: (batch, seq_len) — token indices                ║
    ║    1. Embed: (batch, seq_len, embed_dim)               ║
    ║    2. Loop over time steps:                            ║
    ║       h = rnn_cell(embed[:, t, :], h)                  ║
    ║       collect all h's                                  ║
    ║    3. Stack hidden states: (batch, seq_len, hidden)     ║
    ║    4. Project to vocab: (batch, seq_len, vocab_size)    ║
    ║    5. Return logits and final h                        ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
        super().__init__()
        raise NotImplementedError("TODO: implement RNNLanguageModel.__init__()")

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, h_final)."""
        raise NotImplementedError("TODO: implement RNNLanguageModel.forward()")


# ─────────────────────────────────────────────────────
# LSTM Language Model
# ─────────────────────────────────────────────────────

class LSTMLanguageModel(nn.Module):
    """
    Character-level language model using ManualLSTMCell.

    Same architecture as RNNLanguageModel but with LSTM.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement — same pattern as RNNLanguageModel   ║
    ║  but forward passes (h, c) tuple as state.            ║
    ║                                                       ║
    ║  forward(x, state=None):                              ║
    ║    state = (h, c) or None                              ║
    ║    returns (logits, (h_final, c_final))                ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
        super().__init__()
        raise NotImplementedError("TODO: implement LSTMLanguageModel.__init__()")

    def forward(
        self, x: torch.Tensor, state=None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns (logits, (h_final, c_final))."""
        raise NotImplementedError("TODO: implement LSTMLanguageModel.forward()")
