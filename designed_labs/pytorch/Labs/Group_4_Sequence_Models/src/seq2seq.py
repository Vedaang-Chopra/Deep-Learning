"""
seq2seq.py — Encoder-Decoder with Attention
=============================================

Student implements:
  - Encoder: bidirectional LSTM encoder (uses ManualLSTMCell or nn layers)
  - AttentionDecoder: decoder with Bahdanau attention
  - Seq2Seq: full model with teacher forcing

No high-level shortcuts. Attention mechanism from attention.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Sequence encoder using LSTM.

    Architecture:
        Embedding → LSTM (potentially multi-layer) → encoder outputs + final state

    Parameters
    ----------
    vocab_size : int
    embed_dim : int
    hidden_size : int
    num_layers : int (default 1)
    dropout : float

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  You may use nn.LSTM here (the manual cell was for    ║
    ║  learning — in Seq2Seq we use nn.LSTM for efficiency).║
    ║                                                       ║
    ║  forward(src):                                        ║
    ║    src: (batch, src_len) — source token indices        ║
    ║    1. Embed: (batch, src_len, embed_dim)               ║
    ║    2. LSTM: outputs (batch, src_len, hidden)           ║
    ║    3. Return (encoder_outputs, (h_n, c_n))             ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        raise NotImplementedError("TODO: implement Encoder.__init__()")

    def forward(self, src: torch.Tensor):
        """
        Returns
        -------
        outputs : (batch, src_len, hidden_size)
        (h_n, c_n) : final hidden/cell states
        """
        raise NotImplementedError("TODO: implement Encoder.forward()")


# ─────────────────────────────────────────────────────
# Attention Decoder
# ─────────────────────────────────────────────────────

class AttentionDecoder(nn.Module):
    """
    Decoder with Bahdanau (additive) attention.

    At each time step:
    1. Compute attention context from encoder outputs
    2. Concatenate context with embedded input
    3. Feed through LSTM cell
    4. Project to vocabulary

    Parameters
    ----------
    vocab_size : int
    embed_dim : int
    hidden_size : int
    encoder_hidden_size : int
    attention_dim : int
    dropout : float

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward_step.           ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. Embedding                                          ║
    ║  2. AdditiveAttention (from attention.py)              ║
    ║  3. LSTMCell:                                          ║
    ║     input_size = embed_dim + encoder_hidden_size       ║
    ║  4. Linear(hidden_size, vocab_size) — output proj      ║
    ║                                                       ║
    ║  forward_step(input_token, state, encoder_outputs):    ║
    ║    input_token: (batch,) — single token index          ║
    ║    state: (h, c)                                       ║
    ║    1. Embed input: (batch, embed_dim)                  ║
    ║    2. Compute attention: context, weights               ║
    ║    3. Concat [embed, context]: (batch, embed+enc_h)    ║
    ║    4. LSTM step: new (h, c)                            ║
    ║    5. Logits: Linear(h) → (batch, vocab_size)          ║
    ║    6. Return logits, (h, c), weights                   ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        encoder_hidden_size: int,
        attention_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        raise NotImplementedError("TODO: implement AttentionDecoder.__init__()")

    def forward_step(
        self,
        input_token: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
    ):
        """
        Single decoder step.

        Returns
        -------
        logits : (batch, vocab_size)
        state : (h, c) new states
        attn_weights : (batch, src_len)
        """
        raise NotImplementedError("TODO: implement AttentionDecoder.forward_step()")


# ─────────────────────────────────────────────────────
# Seq2Seq Model
# ─────────────────────────────────────────────────────

class Seq2Seq(nn.Module):
    """
    Full Seq2Seq model with teacher forcing.

    Parameters
    ----------
    encoder : Encoder
    decoder : AttentionDecoder
    sos_idx : int — start-of-sequence token index
    eos_idx : int — end-of-sequence token index

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement forward.                             ║
    ║                                                       ║
    ║  forward(src, tgt, teacher_forcing_ratio=0.5):         ║
    ║  1. Encode src → encoder_outputs, (h, c)               ║
    ║  2. Initialize decoder state from encoder final state  ║
    ║  3. First decoder input = SOS token                    ║
    ║  4. For each target time step:                         ║
    ║     a. decoder.forward_step(...)                       ║
    ║     b. Collect logits                                   ║
    ║     c. Teacher forcing: with prob teacher_forcing_ratio ║
    ║        use ground truth tgt[:, t] as next input        ║
    ║        otherwise use argmax of logits                   ║
    ║  5. Return all logits: (batch, tgt_len, vocab_size)    ║
    ║                                                       ║
    ║  Note on teacher forcing ratio:                        ║
    ║  - 1.0 = always use ground truth (fast convergence)    ║
    ║  - 0.0 = always use own predictions (exposure bias)    ║
    ║  - 0.5 = flip a coin each step (common default)        ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: AttentionDecoder,
        sos_idx: int = 0,
        eos_idx: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Returns logits: (batch, tgt_len, vocab_size)."""
        raise NotImplementedError("TODO: implement Seq2Seq.forward()")


# ─────────────────────────────────────────────────────
# Simple Reversal Dataset (provided)
# ─────────────────────────────────────────────────────

class ReversalDataset(torch.utils.data.Dataset):
    """
    A simple dataset for Seq2Seq: reverse a sequence of tokens.
    Input: [3, 7, 2, 5] → Target: [5, 2, 7, 3]

    Tokens 0 = SOS, 1 = EOS, 2 = PAD, 3+ = actual tokens.
    Provided utility.
    """

    def __init__(self, num_samples: int = 2000, max_len: int = 10, vocab_size: int = 20):
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(3, max_len)
            seq = [random.randint(3, vocab_size - 1) for _ in range(length)]
            src = seq + [1]  # append EOS
            tgt = [0] + list(reversed(seq)) + [1]  # SOS + reversed + EOS
            self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_seq2seq(batch, pad_idx: int = 2):
    """Pad and collate Seq2Seq batch. Provided utility."""
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)
    src_padded = torch.full((len(srcs), max_src), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(tgts), max_tgt), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, :len(s)] = torch.tensor(s)
        tgt_padded[i, :len(t)] = torch.tensor(t)
    return src_padded, tgt_padded
