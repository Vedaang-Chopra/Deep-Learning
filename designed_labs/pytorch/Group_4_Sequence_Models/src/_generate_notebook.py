#!/usr/bin/env python3
"""Generate the Group 4 Sequence Models Lab notebook."""
import json, os

def md(source):
    if isinstance(source, str): source = source.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in source[:-1]] + [source[-1]]}

def code(source):
    if isinstance(source, str): source = source.split("\n")
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in source[:-1]] + [source[-1]], "execution_count": None, "outputs": []}

cells = []

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
cells.append(md("""\
# 🔤 Notebook 4 — Sequence Modeling Lab
## From RNNs to Transformers and Autoregressive Decoding

**Group 4 — Sequence Models & Transformers**

---

### 🎯 Learning Objectives

1. Implement RNN and LSTM cells from raw equations (no `nn.RNN`/`nn.LSTM`)
2. Build Seq2Seq with Bahdanau attention and teacher forcing
3. Implement scaled dot-product and multi-head attention from scratch
4. Build Transformer blocks with causal and padding masks
5. Implement sinusoidal positional encoding
6. Build a tiny GPT decoder for character-level language modeling
7. Implement KV caching for efficient autoregressive inference
8. Compute perplexity and measure inference throughput

### 📂 File Structure

```
Group_4_Sequence_Models/
├── notebooks/
│   └── 04_sequence_models_lab.ipynb   ← you are here
└── src/
    ├── rnn_cells.py             ← ManualRNNCell, ManualLSTMCell, LM wrappers
    ├── attention.py             ← dot-product attention, masks, MultiHeadAttention
    ├── seq2seq.py               ← Encoder, AttentionDecoder, Seq2Seq
    ├── transformer_blocks.py    ← PositionalEncoding, TransformerBlock
    └── gpt_decoder.py           ← TinyGPT, KVCache, generate, perplexity
```

> ⚠️ **No shortcuts**: `nn.RNN`, `nn.LSTM`, `nn.Transformer`, `nn.MultiheadAttention`, and HuggingFace are all banned. You implement everything from weight matrices up."""))

# ═══════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════
cells.append(md("## 0 — Environment Setup"))

cells.append(code("""\
import sys, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.pardir, "src"))

from rnn_cells import ManualRNNCell, ManualLSTMCell, RNNLanguageModel, LSTMLanguageModel
from attention import (
    dot_product_attention, create_causal_mask, create_padding_mask,
    MultiHeadAttention, AdditiveAttention, plot_attention_weights,
)
from seq2seq import Encoder, AttentionDecoder, Seq2Seq, ReversalDataset, collate_seq2seq
from transformer_blocks import (
    SinusoidalPositionalEncoding, FeedForward, TransformerBlock, TransformerEncoder,
)
from gpt_decoder import (
    TinyGPT, KVCache, generate, compute_perplexity,
    CharDataset, load_tiny_shakespeare,
)

print(f"PyTorch version : {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device          : {DEVICE}")"""))

cells.append(code("""\
# ── Load Tiny Shakespeare for character-level experiments ──
text = load_tiny_shakespeare(max_chars=100_000)
print(f"\\nSample: {text[:200]}...")"""))

# ═══════════════════════════════════════════════════════
# SECTION 1 — RNN CELL FROM FIRST PRINCIPLES
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 1 — RNN Cell from First Principles

## 1.1 Conceptual Background

### The Recurrence Equation

An RNN processes a sequence one token at a time, maintaining a **hidden state** that summarizes everything seen so far:

```
h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
```

Where:
- `x_t` : input at time t (embed_dim)
- `h_{t-1}` : previous hidden state (hidden_size)
- `W_ih` : input-to-hidden weights (hidden_size × embed_dim)
- `W_hh` : hidden-to-hidden weights (hidden_size × hidden_size)
- `tanh` : squashes output to [-1, 1]

### Backpropagation Through Time (BPTT)

To train, we unroll the RNN and backpropagate through all time steps. The gradient of the loss w.r.t. hidden state at time t involves a **product of Jacobians**:

```
dL/dh_t = dL/dh_T · ∏_{k=t+1}^{T} dh_k/dh_{k-1}
```

Each Jacobian `dh_k/dh_{k-1}` involves the weight matrix `W_hh`. If its spectral radius is:
- **> 1**: gradients explode exponentially
- **< 1**: gradients vanish exponentially

This is why vanilla RNNs struggle with long sequences (T > 50-100).

### Gradient Clipping

A pragmatic fix for exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

This rescales the gradient vector if its norm exceeds `max_norm`, preserving direction."""))

cells.append(md("""\
## 1.2 Implementation Tasks

Open `src/rnn_cells.py` and implement:

1. **`ManualRNNCell`** — vanilla RNN cell from raw weight matrices
2. **`RNNLanguageModel`** — unroll the cell across a sequence for char-level LM"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify ManualRNNCell                    ║
# ╚═══════════════════════════════════════════════════════╝

rnn_cell = ManualRNNCell(input_size=32, hidden_size=64)
x = torch.randn(4, 32)  # batch=4, input_size=32

# Test without initial hidden
h = rnn_cell(x)
assert h.shape == (4, 64), f"Expected (4,64), got {h.shape}"
print(f"  ✅ ManualRNNCell: x{x.shape} → h{h.shape}")

# Test with initial hidden
h_prev = torch.randn(4, 64)
h = rnn_cell(x, h_prev)
assert h.shape == (4, 64)
print(f"  ✅ ManualRNNCell with h_prev: works correctly")

# Test gradients exist
h.sum().backward()
assert rnn_cell.W_ih.grad is not None, "No gradient for W_ih"
print(f"  ✅ Gradients flow through ManualRNNCell")"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify RNNLanguageModel                 ║
# ╚═══════════════════════════════════════════════════════╝

dataset = CharDataset(text, block_size=64)
rnn_lm = RNNLanguageModel(vocab_size=dataset.vocab_size, embed_dim=64, hidden_size=128)
x = torch.randint(0, dataset.vocab_size, (2, 50))  # batch=2, seq_len=50
logits, h_final = rnn_lm(x)
assert logits.shape == (2, 50, dataset.vocab_size), f"Expected (2,50,{dataset.vocab_size}), got {logits.shape}"
assert h_final.shape == (2, 128), f"Expected (2,128), got {h_final.shape}"
print(f"  ✅ RNNLanguageModel: input {x.shape} → logits {logits.shape}, h {h_final.shape}")
print(f"     Vocab size: {dataset.vocab_size}, Parameters: {sum(p.numel() for p in rnn_lm.parameters()):,}")"""))

cells.append(md("### 1.3 Experiment — Train Character-Level RNN"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Train RNN on Tiny Shakespeare             ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create CharDataset with block_size=64              ║
# ║  2. Create DataLoader                                  ║
# ║  3. Build RNNLanguageModel(vocab, embed=64, hidden=128)║
# ║  4. Train for 5-10 epochs                               ║
# ║  5. For each batch:                                     ║
# ║     input = batch[:, :-1], target = batch[:, 1:]        ║
# ║     logits, _ = model(input)                             ║
# ║     loss = F.cross_entropy(logits.reshape(-1, V),        ║
# ║                            target.reshape(-1))           ║
# ║  6. Log and plot: loss per step, grad norms             ║
# ║  7. Apply gradient clipping (max_norm=5.0)              ║
# ║  8. Generate text by sampling from the model            ║
# ╚═══════════════════════════════════════════════════════╝

dataset = CharDataset(text, block_size=64)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# TODO: Build, train, plot, generate
"""))

cells.append(md("""\
### 1.4 Reflection

1. **What happens to gradient norms as you increase sequence length?**
2. **Does gradient clipping fully solve the vanishing gradient problem?**
3. **Why can't the RNN capture patterns that span > 50 tokens?**"""))

# ═══════════════════════════════════════════════════════
# SECTION 2 — LSTM FROM SCRATCH
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 2 — LSTM from Scratch

## 2.1 Conceptual Background

### The Gating Mechanism

LSTM solves vanishing gradients by adding a **cell state** `c_t` with controlled gating:

```
i_t = σ(W_i·[x_t, h_{t-1}] + b_i)     ← Input gate:  what to ADD
f_t = σ(W_f·[x_t, h_{t-1}] + b_f)     ← Forget gate: what to KEEP
g_t = tanh(W_g·[x_t, h_{t-1}] + b_g)  ← Candidate:   new info
o_t = σ(W_o·[x_t, h_{t-1}] + b_o)     ← Output gate: what to EXPOSE

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t       ← Cell state update
h_t = o_t ⊙ tanh(c_t)                  ← Hidden state
```

### Why This Helps Gradients

The cell state `c_t` flows through time via:
```
c_t = f_t ⊙ c_{t-1} + ...
```

If `f_t ≈ 1`, the gradient flows **undiminished** through time (like a highway).
The forget gate learns which information to keep long-term.

### Practical Tip: Forget Gate Bias

Initialize the forget gate bias to **+1.0** (not 0.0):
```python
self.b_hh[hidden_size:2*hidden_size].fill_(1.0)
```
This makes `f_t ≈ σ(1) ≈ 0.73` initially → keeps most information → better gradient flow."""))

cells.append(md("""\
## 2.2 Implementation Task

Open `src/rnn_cells.py` and implement:

1. **`ManualLSTMCell`** — LSTM cell with all 4 gates from raw equations
2. **`LSTMLanguageModel`** — character-level LM using ManualLSTMCell"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify ManualLSTMCell                   ║
# ╚═══════════════════════════════════════════════════════╝

lstm_cell = ManualLSTMCell(input_size=32, hidden_size=64)
x = torch.randn(4, 32)

# Without initial state
h, c = lstm_cell(x)
assert h.shape == (4, 64), f"h: expected (4,64), got {h.shape}"
assert c.shape == (4, 64), f"c: expected (4,64), got {c.shape}"
print(f"  ✅ ManualLSTMCell: x{x.shape} → h{h.shape}, c{c.shape}")

# With initial state
h2, c2 = lstm_cell(x, (h, c))
assert h2.shape == h.shape
print(f"  ✅ ManualLSTMCell with state: works correctly")

# Verify gradients
(h2.sum() + c2.sum()).backward()
assert lstm_cell.W_ih.grad is not None
print(f"  ✅ Gradients flow through ManualLSTMCell")
print(f"     W_ih: {lstm_cell.W_ih.shape}, W_hh: {lstm_cell.W_hh.shape}")"""))

cells.append(md("### 2.3 Experiment — RNN vs LSTM Comparison"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Compare RNN vs LSTM                       ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Train RNNLanguageModel for 5 epochs                ║
# ║  2. Train LSTMLanguageModel for 5 epochs               ║
# ║     (same hyperparams: embed=64, hidden=128)           ║
# ║  3. Compare:                                           ║
# ║     - Final loss (lower = better)                       ║
# ║     - Perplexity                                        ║
# ║     - Gradient norms distribution                       ║
# ║     - Generated text quality                            ║
# ║  4. Plot loss curves overlay                            ║
# ║  5. Plot gradient norm histograms                       ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Train both, compare, plot
"""))

cells.append(md("""\
### 2.4 Reflection

1. **How does the LSTM gradient norm distribution differ from the RNN?**
2. **What role does the forget gate play in long sequences?**
3. **On which type of text pattern does LSTM clearly outperform RNN?**"""))

# ═══════════════════════════════════════════════════════
# SECTION 3 — SEQ2SEQ WITH ATTENTION
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 3 — Seq2Seq with Attention

## 3.1 Conceptual Background

### The Bottleneck Problem

In vanilla Seq2Seq, the encoder compresses the entire source sequence into a **single vector** (the final hidden state). This is the bottleneck — long sequences lose information.

```
Source: "The quick brown fox jumps over the lazy dog"
              ↓ encoder ↓
      h_final = [0.3, -0.1, ...]  ← ALL information is here
              ↓ decoder ↓
Target: "Le rapide renard brun saute par-dessus le chien paresseux"
```

### Attention: The Fix

Instead of using only `h_final`, let the decoder **look at all encoder hidden states** at each step:

```
Decoder step t:
  1. Compute score(h_dec_t, h_enc_j) for all source positions j
  2. Normalize scores → attention weights α_j
  3. Context vector = Σ α_j · h_enc_j
  4. Feed [context, input] to decoder RNN
```

### Bahdanau (Additive) Attention

```
score(s, h) = v^T · tanh(W_s · s + W_h · h)
```

### Teacher Forcing & Exposure Bias

| Mode | Decoder Input | Benefit | Risk |
|------|--------------|---------|------|
| Teacher forcing | Ground truth tgt_{t-1} | Fast convergence | Exposure bias |
| Free running | Own prediction argmax(logits_{t-1}) | Robust at inference | Slow training |
| Scheduled sampling | Mix (anneal ratio) | Best of both | Slightly complex |

**Exposure bias**: During training (teacher forcing), the decoder always sees correct previous tokens. At inference, it sees its own (possibly wrong) predictions → distribution mismatch → error accumulation."""))

cells.append(md("""\
## 3.2 Implementation Tasks

1. Open `src/attention.py` and implement **`AdditiveAttention`**
2. Open `src/seq2seq.py` and implement:
   - **`Encoder`**
   - **`AttentionDecoder`**
   - **`Seq2Seq.forward()`** with teacher forcing"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify Seq2Seq components               ║
# ╚═══════════════════════════════════════════════════════╝

VOCAB = 20
# Test Encoder
enc = Encoder(vocab_size=VOCAB, embed_dim=32, hidden_size=64)
src = torch.randint(3, VOCAB, (4, 8))  # batch=4, src_len=8
enc_out, (h_n, c_n) = enc(src)
assert enc_out.shape == (4, 8, 64), f"Encoder outputs: expected (4,8,64) got {enc_out.shape}"
print(f"  ✅ Encoder: src {src.shape} → outputs {enc_out.shape}")

# Test AdditiveAttention
attn = AdditiveAttention(decoder_dim=64, encoder_dim=64, attention_dim=32)
dec_h = torch.randn(4, 64)
context, weights = attn(dec_h, enc_out)
assert context.shape == (4, 64), f"Context: expected (4,64) got {context.shape}"
assert weights.shape == (4, 8), f"Weights: expected (4,8) got {weights.shape}"
assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5), "Weights should sum to 1"
print(f"  ✅ AdditiveAttention: context {context.shape}, weights {weights.shape}")

# Test AttentionDecoder
dec = AttentionDecoder(vocab_size=VOCAB, embed_dim=32, hidden_size=64, encoder_hidden_size=64)
token = torch.randint(0, VOCAB, (4,))  # single token
logits, (h, c), attn_w = dec.forward_step(token, (h_n.squeeze(0), c_n.squeeze(0)), enc_out)
assert logits.shape == (4, VOCAB), f"Logits: expected (4,{VOCAB}) got {logits.shape}"
print(f"  ✅ AttentionDecoder step: logits {logits.shape}")

# Test full Seq2Seq
model = Seq2Seq(enc, dec, sos_idx=0, eos_idx=1)
tgt = torch.randint(0, VOCAB, (4, 6))
output = model(src, tgt, teacher_forcing_ratio=1.0)
assert output.shape == (4, 6, VOCAB), f"Seq2Seq: expected (4,6,{VOCAB}) got {output.shape}"
print(f"  ✅ Seq2Seq: src {src.shape} + tgt {tgt.shape} → output {output.shape}")"""))

cells.append(md("### 3.3 Experiment — Train Seq2Seq on Reversal Task"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Seq2Seq on sequence reversal              ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create ReversalDataset(2000 samples, max_len=10)   ║
# ║  2. Build Seq2Seq model with attention                 ║
# ║  3. Train for 15-20 epochs with teacher_forcing=0.5    ║
# ║  4. Evaluate: input [3,7,2] → should predict [2,7,3]  ║
# ║  5. Visualize attention weights (should show diagonal  ║
# ║     reversal pattern)                                   ║
# ║  6. Compare teacher_forcing=1.0 vs 0.5 vs 0.0          ║
# ╚═══════════════════════════════════════════════════════╝

from functools import partial

rev_dataset = ReversalDataset(num_samples=2000, max_len=10, vocab_size=20)
rev_loader = torch.utils.data.DataLoader(
    rev_dataset, batch_size=64, shuffle=True,
    collate_fn=partial(collate_seq2seq, pad_idx=2),
)

# TODO: Build, train, evaluate, visualize attention
"""))

cells.append(md("""\
### 3.4 Reflection

1. **What pattern should attention weights show for sequence reversal?**
2. **How does teacher forcing ratio affect convergence speed vs final quality?**
3. **What is exposure bias, and when does it cause problems?**"""))

# ═══════════════════════════════════════════════════════
# SECTION 4 — TRANSFORMER MULTI-HEAD ATTENTION
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 4 — Transformer Multi-Head Attention

## 4.1 Conceptual Background

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Why scale by √d_k? Without scaling, when d_k is large, the dot products grow large → softmax saturates → gradients vanish.

### Multi-Head Attention

Instead of one attention function, run `h` parallel attention heads with different learned projections:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

Each head can attend to different aspects of the sequence (e.g., syntax, semantics, position).

### Why Transformers Beat RNNs

| RNN | Transformer |
|-----|-------------|
| Sequential: O(T) steps | Parallel: O(1) steps |
| Fixed-size hidden state | Direct access to all positions |
| Gradient degrades over time | Gradient flows directly via attention |
| Slow to train on long seqs | Fast but O(T²) memory |

### Causal Mask for Autoregressive Models

For language modeling, position t should only attend to positions ≤ t:

```
Mask (True = BLOCKED):
    [[F, T, T, T],
     [F, F, T, T],
     [F, F, F, T],
     [F, F, F, F]]
```

Positions with True in the mask get -∞ before softmax → zero attention weight."""))

cells.append(md("""\
## 4.2 Implementation Tasks

1. Open `src/attention.py` and implement:
   - **`dot_product_attention()`** — scaled dot-product
   - **`create_causal_mask()`** — upper triangular boolean mask
   - **`create_padding_mask()`** — mask for padded positions
   - **`MultiHeadAttention`** — full multi-head with Q/K/V projections

2. Open `src/transformer_blocks.py` and implement:
   - **`FeedForward`** — position-wise FFN
   - **`TransformerBlock`** — pre-norm block with residual connections"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify attention components             ║
# ╚═══════════════════════════════════════════════════════╝

# Test dot_product_attention
q = torch.randn(2, 4, 8, 16)  # (batch, heads, seq, d_k)
k = torch.randn(2, 4, 8, 16)
v = torch.randn(2, 4, 8, 16)
out, weights = dot_product_attention(q, k, v)
assert out.shape == (2, 4, 8, 16), f"Attn output: {out.shape}"
assert weights.shape == (2, 4, 8, 8), f"Attn weights: {weights.shape}"
assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 8), atol=1e-5)
print(f"  ✅ dot_product_attention: Q{q.shape} → output {out.shape}, weights {weights.shape}")

# Test with causal mask
mask = create_causal_mask(8)
assert mask.shape == (8, 8), f"Mask shape: {mask.shape}"
assert mask[0, 0] == False, "Position (0,0) should not be masked"
assert mask[0, 1] == True, "Position (0,1) should be masked (future)"
print(f"  ✅ Causal mask ({mask.shape}):\\n{mask.int()}")

out_masked, w_masked = dot_product_attention(q, k, v, mask=mask)
# Verify future positions have zero weight
assert w_masked[0, 0, 0, 1:].sum() < 1e-6, "Position 0 should not attend to future"
print(f"  ✅ Masked attention: future positions correctly zeroed")

# Test padding mask
lengths = torch.tensor([3, 5])
pad_mask = create_padding_mask(lengths, max_len=6)
assert pad_mask.shape == (2, 6)
assert pad_mask[0, 2] == False and pad_mask[0, 3] == True  # length 3
print(f"  ✅ Padding mask: {pad_mask.int()}")"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify MultiHeadAttention               ║
# ╚═══════════════════════════════════════════════════════╝

mha = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.0)
x = torch.randn(2, 10, 64)  # (batch, seq, d_model)

# Self-attention
out, weights = mha(x, x, x)
assert out.shape == (2, 10, 64), f"MHA output: {out.shape}"
assert weights.shape == (2, 4, 10, 10), f"MHA weights: {weights.shape}"
print(f"  ✅ MultiHeadAttention (self): input {x.shape} → output {out.shape}")

# Masked self-attention
causal = create_causal_mask(10, device=x.device)
out_m, weights_m = mha(x, x, x, mask=causal)
assert weights_m[0, 0, 0, 1:].sum() < 1e-6
print(f"  ✅ MultiHeadAttention (causal): future positions blocked")

# Test TransformerBlock
block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)
out_b, attn_w = block(x)
assert out_b.shape == x.shape
print(f"  ✅ TransformerBlock: {x.shape} → {out_b.shape}")"""))

cells.append(md("### 4.3 Experiment — Transformer vs LSTM on Tiny Shakespeare"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Compare Transformer vs LSTM for LM        ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Build a 2-layer TransformerEncoder for char LM      ║
# ║     (d_model=128, num_heads=4, d_ff=512)               ║
# ║  2. Compare against LSTMLanguageModel (hidden=128)     ║
# ║  3. Train both for 5 epochs on Tiny Shakespeare         ║
# ║  4. Compare:                                           ║
# ║     - Loss curves                                       ║
# ║     - Training time per epoch                           ║
# ║     - Gradient norms                                    ║
# ║  5. Visualize attention weights for a sample sequence   ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Build, train, compare
"""))

# ═══════════════════════════════════════════════════════
# SECTION 5 — POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 5 — Positional Encoding

## 5.1 Conceptual Background

### Why Position Matters

Transformers process all tokens in parallel — they have **no notion of order**.
Without positional information, "the cat sat on the mat" and "mat the on sat cat the" produce the same output.

### Sinusoidal Encoding (Vaswani et al.)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Properties:
- Each position gets a unique encoding
- Relative positions can be computed via linear combinations
- Extrapolates to unseen lengths (unlike learned embeddings)

### Learned vs Fixed

| Type | Pros | Cons |
|------|------|------|
| **Sinusoidal** | Extrapolates, no parameters | Fixed representation |
| **Learned** | Adapts to data | Cannot extrapolate past max_len |"""))

cells.append(md("## 5.2 Implementation Task\n\nOpen `src/transformer_blocks.py` and implement **`SinusoidalPositionalEncoding`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify Positional Encoding              ║
# ╚═══════════════════════════════════════════════════════╝

pe = SinusoidalPositionalEncoding(d_model=64, max_len=200, dropout=0.0)
x = torch.zeros(1, 100, 64)  # no input signal — pure PE
out = pe(x)
assert out.shape == (1, 100, 64), f"PE output: {out.shape}"

# Visualize PE
pe_matrix = out[0].detach().numpy()
plt.figure(figsize=(12, 4))
plt.imshow(pe_matrix.T, cmap="RdBu", aspect="auto")
plt.xlabel("Position")
plt.ylabel("Dimension")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.tight_layout()
plt.show()

# Verify orthogonality-like property
print(f"  ✅ PE shape: {out.shape}")
print(f"     Max value: {pe_matrix.max():.4f}, Min: {pe_matrix.min():.4f}")"""))

# ═══════════════════════════════════════════════════════
# SECTION 6 — TINY GPT DECODER
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 6 — Tiny GPT Decoder

## 6.1 Conceptual Background

### Autoregressive Language Modeling

GPT is a **decoder-only** transformer that predicts the next token given all previous tokens:

```
P(x_1, x_2, ..., x_T) = ∏ P(x_t | x_1, ..., x_{t-1})
```

### Training

- Input: tokens [x_1, x_2, ..., x_{T-1}]
- Target: tokens [x_2, x_3, ..., x_T]
- Loss: cross-entropy at each position
- **Causal mask** ensures position t only sees positions ≤ t

### Perplexity

The standard metric for language models:

```
PPL = exp(average cross-entropy loss per token)
```

- **PPL = 1**: perfect prediction (impossible in practice)
- **PPL = V** (vocab size): random guessing
- **Lower is better**

A character-level model on English typically achieves PPL ≈ 1.5-3.0.

### Temperature Sampling

```python
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)
```

- **temperature < 1**: sharper distribution → more deterministic
- **temperature = 1**: original distribution
- **temperature > 1**: flatter distribution → more random"""))

cells.append(md("""\
## 6.2 Implementation Tasks

Open `src/gpt_decoder.py` and implement:

1. **`TinyGPT`** — decoder-only transformer LM
2. **`compute_perplexity()`** — perplexity metric
3. **`generate()`** — autoregressive token generation"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify TinyGPT                         ║
# ╚═══════════════════════════════════════════════════════╝

dataset = CharDataset(text, block_size=64)
gpt = TinyGPT(
    vocab_size=dataset.vocab_size,
    d_model=128, num_heads=4, num_layers=4, d_ff=512, max_len=256,
)
x = torch.randint(0, dataset.vocab_size, (2, 50))
logits = gpt(x)
assert logits.shape == (2, 50, dataset.vocab_size), f"GPT: expected (2,50,{dataset.vocab_size}) got {logits.shape}"
print(f"  ✅ TinyGPT: input {x.shape} → logits {logits.shape}")
print(f"     Parameters: {sum(p.numel() for p in gpt.parameters()):,}")

# Verify causal masking
logits1 = gpt(x[:, :30])
logits2 = gpt(x[:, :50])
# Logits for position 29 should be identical in both cases
# (causal mask means future tokens don't affect earlier positions)
diff = (logits1[:, 29, :] - logits2[:, 29, :]).abs().max()
print(f"  ✅ Causal mask check: max diff at pos 29 = {diff.item():.6f}")
assert diff < 1e-4, "Causal masking broken — future tokens affect past predictions" """))

cells.append(md("### 6.3 Experiment — Train Tiny GPT on Shakespeare"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Train TinyGPT on Tiny Shakespeare         ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create CharDataset(block_size=128)                  ║
# ║  2. Build TinyGPT(d_model=128, heads=4, layers=4)      ║
# ║  3. Train for 10-15 epochs                              ║
# ║     input=batch[:, :-1], target=batch[:, 1:]            ║
# ║     loss = cross_entropy(logits.view(-1,V), target.view(-1))║
# ║  4. Plot loss curve + compute perplexity                ║
# ║  5. Generate text with different temperatures           ║
# ║     (0.5, 0.8, 1.0, 1.5)                               ║
# ║  6. Generate with top-k filtering (k=5, k=10, k=50)    ║
# ╚═══════════════════════════════════════════════════════╝

dataset = CharDataset(text, block_size=128)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Dataset: {len(dataset)} chunks, vocab: {dataset.vocab_size}")

# TODO: Build, train, plot, generate
"""))

# ═══════════════════════════════════════════════════════
# SECTION 7 — KV CACHE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 7 — KV Cache for Inference

## 7.1 Conceptual Background

### Why Naive Decoding Is Slow

Without caching, generating T tokens requires:
- Step 1: forward 1 token
- Step 2: forward 2 tokens (re-compute token 1)
- Step 3: forward 3 tokens (re-compute tokens 1-2)
- ...
- Step T: forward T tokens

Total work: O(T² · d²) — quadratic in sequence length.

### KV Caching

**Key insight**: In causal attention, the K and V projections for previous tokens **never change** (because they can't see future tokens).

```
Step 1: Q₁, K₁, V₁ → Attention → output₁
        Cache: K=[K₁], V=[V₁]

Step 2: Q₂, K₂, V₂
        Full K = [K₁, K₂] (cached + new)
        Full V = [V₁, V₂]
        Attention(Q₂, [K₁,K₂], [V₁,V₂]) → output₂
        Cache: K=[K₁,K₂], V=[V₁,V₂]

Step t: Only compute Q_t, K_t, V_t for the NEW token
        Concatenate with cache
        Attention(Q_t, cache_K, cache_V)
```

Total work: O(T · d²) — **linear** in sequence length!

### Memory Tradeoff

KV cache uses O(T · num_layers · d_model) memory per sequence.
For large models (GPT-3: 175B params), this can be gigabytes per sequence."""))

cells.append(md("""\
## 7.2 Implementation Task

Open `src/gpt_decoder.py` and implement:

1. **`KVCache`** — key-value cache with update/get/reset
2. **Integrate KV cache into `TinyGPT.forward()`**
3. **Implement `generate()` with `use_kv_cache=True`**"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify KV Cache                        ║
# ╚═══════════════════════════════════════════════════════╝

cache = KVCache(num_layers=4)

# Simulate step 1: cache a token
k1 = torch.randn(1, 4, 1, 32)  # (batch, heads, seq=1, d_k)
v1 = torch.randn(1, 4, 1, 32)
full_k, full_v = cache.update(0, k1, v1)
assert full_k.shape == (1, 4, 1, 32), f"After 1 token: {full_k.shape}"

# Simulate step 2: add another token
k2 = torch.randn(1, 4, 1, 32)
v2 = torch.randn(1, 4, 1, 32)
full_k, full_v = cache.update(0, k2, v2)
assert full_k.shape == (1, 4, 2, 32), f"After 2 tokens: expected (1,4,2,32) got {full_k.shape}"
print(f"  ✅ KVCache: correctly accumulates K/V along sequence dim")

# Verify reset
cache.reset()
cached_k, cached_v = cache.get(0)
assert cached_k is None, "Cache should be empty after reset"
print(f"  ✅ KVCache: reset works correctly")"""))

cells.append(md("### 7.3 Experiment — KV Cache Speedup"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Compare naive vs cached generation        ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Load your trained TinyGPT model                    ║
# ║  2. Generate 200 tokens WITHOUT KV cache — time it     ║
# ║  3. Generate 200 tokens WITH KV cache — time it        ║
# ║  4. Verify outputs are identical                        ║
# ║  5. Plot: tokens/sec vs sequence length for both       ║
# ║  6. Vary max_new_tokens: [50, 100, 200, 500]           ║
# ║     and plot speedup factor                             ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Benchmark and compare
# prompt = dataset.encode("ROMEO:\\n").unsqueeze(0).to(DEVICE)
#
# # Naive
# t0 = time.time()
# out_naive = generate(model, prompt, max_new_tokens=200, use_kv_cache=False)
# t_naive = time.time() - t0
#
# # Cached
# t0 = time.time()
# out_cached = generate(model, prompt, max_new_tokens=200, use_kv_cache=True)
# t_cached = time.time() - t0
#
# print(f"Naive:  {t_naive:.2f}s ({200/t_naive:.0f} tok/s)")
# print(f"Cached: {t_cached:.2f}s ({200/t_cached:.0f} tok/s)")
# print(f"Speedup: {t_naive/t_cached:.1f}x")
"""))

cells.append(md("""\
### 7.4 Reflection

1. **Why does KV caching reduce complexity from O(T²) to O(T)?**
2. **What's the memory cost of KV caching for a model with L layers, H heads, d_k per head?**
3. **Why can't we cache Q (queries) too?**"""))

# ═══════════════════════════════════════════════════════
# FINAL CHALLENGE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# 🧪 Final Challenge — Sequence Debug Toolkit

## Objective

Build a comprehensive **Sequence Debug Toolkit** for diagnosing and understanding sequence models.

## Required Components

| Tool | What It Does |
|------|-------------|
| `visualize_attention_multihead(model, text)` | Plot attention weights for each head at each layer |
| `gradient_flow_rnn(model, seq)` | Plot gradient norms through time steps (BPTT) |
| `perplexity_tracker(model, dataset)` | Track perplexity over training epochs, plot curve |
| `generation_comparison(model, prompt)` | Generate with varying temperature and top-k, display side-by-side |
| `kv_cache_profiler(model, lengths)` | Measure tokens/sec with and without cache across sequence lengths |

## Requirements

- All functions must work on your trained models
- Include visualizations (matplotlib)
- Demonstrate each tool on your TinyGPT model
- Analyze: what do different attention heads learn?

No solutions provided — only the requirements above."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  FINAL CHALLENGE: Implement the Sequence Debug Toolkit ║
# ║                                                       ║
# ║  Implement each function and demonstrate on your       ║
# ║  trained models from this notebook.                    ║
# ╚═══════════════════════════════════════════════════════╝

def visualize_attention_multihead(model, input_tokens, layer_idx=0):
    \"\"\"
    Extract and plot attention weights from a specific layer.
    TODO: Implement
    \"\"\"
    raise NotImplementedError

def gradient_flow_rnn(model, sequence, loss_fn):
    \"\"\"
    Plot gradient norms at each time step of an RNN.
    Shows how gradients vanish/explode over time.
    TODO: Implement
    \"\"\"
    raise NotImplementedError

def generation_comparison(model, dataset, prompt_text, temperatures, top_ks):
    \"\"\"
    Generate text with various settings and display side by side.
    TODO: Implement
    \"\"\"
    raise NotImplementedError

# TODO: Demonstrate each tool
"""))

# ═══════════════════════════════════════════════════════
# SUMMARY CHECKLIST
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# ✅ Summary Checklist

| # | Skill | Confident? |
|---|-------|-----------|
| 1 | I can implement an RNN cell from raw weight matrices | ☐ |
| 2 | I understand why RNN gradients vanish and how LSTM fixes it | ☐ |
| 3 | I can implement LSTM gates from first principles | ☐ |
| 4 | I can build Seq2Seq with attention and teacher forcing | ☐ |
| 5 | I can implement scaled dot-product and multi-head attention | ☐ |
| 6 | I understand causal and padding masks and can implement them | ☐ |
| 7 | I can implement sinusoidal positional encoding | ☐ |
| 8 | I can build a GPT-style decoder-only transformer | ☐ |
| 9 | I can implement KV caching for efficient inference | ☐ |
| 10 | I can compute perplexity and compare language models | ☐ |

### 🔧 Toolkit Summary

```python
# Your reusable sequence modeling toolkit:
from rnn_cells import ManualRNNCell, ManualLSTMCell
from attention import dot_product_attention, MultiHeadAttention, create_causal_mask
from transformer_blocks import SinusoidalPositionalEncoding, TransformerBlock
from gpt_decoder import TinyGPT, KVCache, generate, compute_perplexity
```

---

### 🔜 Next: Group 5 — Advanced Training & Optimization

Mixed precision, distributed training, custom CUDA kernels, profiling, hyperparameter search."""))

# ═══════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out_path = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "notebooks", "04_sequence_models_lab.ipynb"
))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook: {out_path}")
print(f"Cells: {len(cells)} (markdown: {md_count}, code: {code_count})")
