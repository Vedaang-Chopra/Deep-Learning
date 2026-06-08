"""
gpt_decoder.py — Tiny GPT Decoder from Scratch
=================================================

Student implements:
  - TinyGPT: decoder-only transformer for language modeling
  - KVCache: key-value cache for efficient autoregressive inference
  - generate(): autoregressive token generation
  - compute_perplexity(): perplexity metric

No nn.Transformer or HuggingFace allowed.
Uses MultiHeadAttention and TransformerBlock from other modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple, List, Dict


# ─────────────────────────────────────────────────────
# KV Cache
# ─────────────────────────────────────────────────────

class KVCache:
    """
    Key-Value cache for efficient autoregressive decoding.

    During generation, previous K and V don't change.
    Instead of recomputing attention over the full sequence,
    we cache K and V and only compute the new token's attention.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__, update, and get.           ║
    ║                                                       ║
    ║  __init__(num_layers, num_heads, d_k):                ║
    ║    Initialize empty cache for each layer               ║
    ║    self.cache = {layer_idx: {'k': None, 'v': None}}   ║
    ║                                                       ║
    ║  update(layer_idx, new_k, new_v):                     ║
    ║    If cache is empty for this layer:                    ║
    ║      Store new_k, new_v                                ║
    ║    Else:                                               ║
    ║      Concatenate along sequence dim:                    ║
    ║      cached_k = cat([cached_k, new_k], dim=-2)         ║
    ║      cached_v = cat([cached_v, new_v], dim=-2)         ║
    ║    Return (full_k, full_v)                             ║
    ║                                                       ║
    ║  get(layer_idx):                                      ║
    ║    Return (cached_k, cached_v) or (None, None)         ║
    ║                                                       ║
    ║  reset():                                             ║
    ║    Clear all cached values                             ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, num_layers: int):
        raise NotImplementedError("TODO: implement KVCache.__init__()")

    def update(
        self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO: implement KVCache.update()")

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        raise NotImplementedError("TODO: implement KVCache.get()")

    def reset(self):
        raise NotImplementedError("TODO: implement KVCache.reset()")


# ─────────────────────────────────────────────────────
# Tiny GPT
# ─────────────────────────────────────────────────────

class TinyGPT(nn.Module):
    """
    A decoder-only transformer for language modeling (GPT-style).

    Architecture:
        Token Embedding + Positional Encoding →
        N × TransformerBlock (with causal mask) →
        LayerNorm → Linear(d_model, vocab_size)

    Parameters
    ----------
    vocab_size : int
    d_model : int (default 128)
    num_heads : int (default 4)
    num_layers : int (default 4)
    d_ff : int (default 512)
    max_len : int (default 512)
    dropout : float

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. Token embedding: nn.Embedding(vocab_size, d_model)║
    ║  2. Positional encoding (sinusoidal from              ║
    ║     transformer_blocks.py OR learned nn.Embedding)     ║
    ║  3. Stack of TransformerBlocks (is_decoder=True)       ║
    ║  4. Final LayerNorm                                    ║
    ║  5. Output projection: Linear(d_model, vocab_size)     ║
    ║                                                       ║
    ║  forward(x, kv_cache=None):                           ║
    ║    x: (batch, seq_len) — token indices                ║
    ║    1. Embed + positional encoding                      ║
    ║    2. Create causal mask                                ║
    ║    3. For each TransformerBlock:                        ║
    ║       - If kv_cache: only process new token(s)         ║
    ║       - Pass through block with causal mask            ║
    ║       - Update kv_cache if provided                    ║
    ║    4. LayerNorm → Linear → logits                      ║
    ║    5. Return logits: (batch, seq_len, vocab_size)       ║
    ║                                                       ║
    ║  Note on KV cache integration:                        ║
    ║  During cached inference, x contains ONLY the new      ║
    ║  token(s). The causal mask should cover the full        ║
    ║  sequence (cached + new). K and V are concatenated     ║
    ║  from cache. Q is only for the new token(s).           ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        raise NotImplementedError("TODO: implement TinyGPT.__init__()")

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Returns logits: (batch, seq_len, vocab_size)
        """
        raise NotImplementedError("TODO: implement TinyGPT.forward()")


# ─────────────────────────────────────────────────────
# Generation (Autoregressive Decoding)
# ─────────────────────────────────────────────────────

def generate(
    model: TinyGPT,
    prompt: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    use_kv_cache: bool = False,
) -> torch.Tensor:
    """
    Autoregressive text generation.

    Parameters
    ----------
    model : TinyGPT
    prompt : (1, prompt_len) — starting token indices
    max_new_tokens : int
    temperature : float — scales logits before softmax (lower = more deterministic)
    top_k : int or None — if set, sample only from top-k most likely tokens
    use_kv_cache : bool — whether to use KV caching for speed

    Returns
    -------
    generated : (1, prompt_len + max_new_tokens) — full sequence

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  WITHOUT KV cache:                                    ║
    ║  1. Start with prompt tokens                           ║
    ║  2. For each new token:                                ║
    ║     a. Forward FULL sequence through model              ║
    ║     b. Take logits at last position                     ║
    ║     c. Apply temperature                                ║
    ║     d. Optional top-k filtering                         ║
    ║     e. Sample from distribution (or argmax if temp≈0)   ║
    ║     f. Append to sequence                               ║
    ║                                                       ║
    ║  WITH KV cache:                                       ║
    ║  1. Initialize KVCache                                 ║
    ║  2. Forward prompt through model (populates cache)     ║
    ║  3. For each new token:                                ║
    ║     a. Forward ONLY the last token (cache has the rest)║
    ║     b. Same sampling logic                              ║
    ║     c. Append to sequence                               ║
    ║                                                       ║
    ║  Top-k filtering:                                     ║
    ║  1. Find the k-th largest logit value                  ║
    ║  2. Set all logits below this to -inf                  ║
    ║  3. Softmax + sample — only top-k tokens have nonzero  ║
    ║     probability                                        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement generate()")


# ─────────────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────────────

def compute_perplexity(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Compute perplexity of a language model on a dataset.

    Perplexity = exp(average cross-entropy loss per token)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  1. model.eval(), torch.no_grad()                      ║
    ║  2. For each batch:                                    ║
    ║     input = batch[:, :-1]                               ║
    ║     target = batch[:, 1:]                               ║
    ║     logits = model(input)                               ║
    ║     loss = F.cross_entropy(logits.reshape(-1, V),       ║
    ║                            target.reshape(-1),          ║
    ║                            reduction='sum')             ║
    ║     Accumulate total loss and total tokens              ║
    ║  3. avg_loss = total_loss / total_tokens                ║
    ║  4. perplexity = exp(avg_loss)                          ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compute_perplexity()")


# ─────────────────────────────────────────────────────
# Character-Level Dataset (provided)
# ─────────────────────────────────────────────────────

class CharDataset(torch.utils.data.Dataset):
    """
    Character-level language modeling dataset.
    Splits text into fixed-length chunks for next-char prediction.

    Provided utility.
    """

    def __init__(self, text: str, block_size: int = 64):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        # Create non-overlapping chunks
        n_chunks = len(data) // (block_size + 1)
        data = data[: n_chunks * (block_size + 1)]
        self.chunks = data.view(n_chunks, block_size + 1)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk  # model splits into input/target internally

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi.get(c, 0) for c in text], dtype=torch.long)

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return "".join(self.itos.get(t, "?") for t in tokens)


# ─────────────────────────────────────────────────────
# Tiny Shakespeare loader (provided)
# ─────────────────────────────────────────────────────

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_tiny_shakespeare(max_chars: int = 100_000) -> str:
    """
    Download and return Tiny Shakespeare text, truncated to max_chars.
    Provided utility.
    """
    import urllib.request
    import os

    cache_path = os.path.join("data", "tiny_shakespeare.txt")
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(cache_path):
        print(f"Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, cache_path)
        print(f"Saved to {cache_path}")

    with open(cache_path, "r") as f:
        text = f.read()

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        print(f"Truncated to {max_chars:,} characters")

    print(f"Text length: {len(text):,} chars, vocab: {len(set(text))} unique chars")
    return text
