#!/usr/bin/env python3
"""Generate the Group 7 Research Extensions Lab notebook."""
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
# 🔬 Notebook 7 — Research Implementation Lab
## Custom Gradients, LoRA, Contrastive Learning, Gradient Surgery, and RL Losses

**Group 7 — Research Extensions in PyTorch**

---

### 🎯 Learning Objectives

1. Build custom `autograd.Function` with manual forward/backward + gradient checking
2. Implement LoRA (Low-Rank Adaptation) from scratch — no PEFT library
3. Implement InfoNCE contrastive loss with numerical stability (SimCLR-style)
4. Apply gradient surgery (PCGrad) for multi-task learning
5. Implement REINFORCE with baseline on a toy bandit task

### 📂 File Structure

```
Group_7_Research_Extensions/
├── notebooks/
│   └── 07_research_extensions_lab.ipynb   ← you are here
└── src/
    ├── custom_autograd_ops.py     ← SwishFn, GradientReversal, STE
    ├── lora_from_scratch.py       ← LoRALinear, inject/merge/unmerge
    ├── contrastive_losses.py      ← InfoNCE, SimCLRProjector
    ├── grad_surgery.py            ← PCGrad, gradient conflict analysis
    └── reinforce_toy.py           ← K-armed bandit, REINFORCE + baseline
```

> ⚠️ **Research-grade implementation**: No HuggingFace Trainer, no PEFT, no Lightning. You translate equations to code.

> 📌 **Philosophy**: Every section follows the pattern: **Equation → Implementation → Correctness Test → Experiment → Ablation**"""))

# ═══════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════
cells.append(md("## 0 — Environment Setup"))

cells.append(code("""\
import sys, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.pardir, "src"))

from custom_autograd_ops import (
    SwishFn, Swish, GradientReversalFn, GradientReversalLayer,
    StraightThroughEstimatorFn, gradient_check,
)
from lora_from_scratch import (
    LoRALinear, inject_lora, merge_lora, unmerge_lora,
    state_dict_adapters_only, count_parameters,
)
from contrastive_losses import (
    info_nce_loss, SimCLRProjector, gather_embeddings_across_ranks,
    get_simclr_augmentations, TwoViewDataset,
)
from grad_surgery import (
    compute_task_gradients, pcgrad_project, pcgrad_step,
    cosine_similarity_tasks, MultiTaskNet, plot_gradient_cosines,
)
from reinforce_toy import (
    KArmedBandit, PolicyNetwork, reinforce_update,
    train_bandit, plot_bandit_results,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}")
print(f"Device:  {DEVICE}")"""))

# ═══════════════════════════════════════════════════════
# SECTION 1 — CUSTOM AUTOGRAD
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 1 — Custom `autograd.Function`

## 1.1 When Do You Need Custom Autograd?

Most operations in PyTorch are auto-differentiable. You need `autograd.Function` when:

1. **Non-standard backward**: Gradient Reversal (negate gradients), STE (pass through discrete ops)
2. **Performance**: fused forward+backward with less memory
3. **Numerical stability**: custom implementation of stable log-sum-exp, etc.
4. **Research**: implementing new ops from papers

### The Contract

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ...):
        # ctx.save_for_backward(x)  ← save tensors for backward
        # return result

    @staticmethod
    def backward(ctx, grad_output):
        # x, = ctx.saved_tensors    ← retrieve saved tensors
        # return (grad_x, ...)      ← one gradient per forward arg
```

### Swish Activation

$$
\\text{Swish}(x) = x \\cdot \\sigma(x)
$$

$$
\\frac{d}{dx}\\text{Swish}(x) = \\sigma(x) + x \\cdot \\sigma(x) \\cdot (1 - \\sigma(x)) = \\text{Swish}(x) + \\sigma(x)(1 - \\text{Swish}(x))
$$

### Gradient Reversal

$$
\\text{Forward: } f(x) = x \\qquad \\text{Backward: } \\frac{\\partial L}{\\partial x} = -\\lambda \\cdot \\frac{\\partial L}{\\partial f}
$$

Used in domain adaptation to learn domain-invariant features."""))

cells.append(md("## 1.2 Implementation Tasks\n\nOpen `src/custom_autograd_ops.py` and implement:\n\n1. **`SwishFn`** — Swish forward and backward\n2. **`GradientReversalFn`** — identity forward, negated backward\n3. **`StraightThroughEstimatorFn`** — discrete forward, identity backward\n4. **`gradient_check()`** — finite-difference verification"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT 1: Swish Gradient Check                    ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create a small test tensor (requires_grad=True)    ║
# ║  2. Run torch.autograd.gradcheck on SwishFn.apply      ║
# ║  3. Compare your Swish output with F.silu (built-in)   ║
# ║  4. Report max absolute error                          ║
# ╚═══════════════════════════════════════════════════════╝

# 1. Test that SwishFn produces correct values
x = torch.randn(5, 3, device=DEVICE)
custom_out = SwishFn.apply(x)
builtin_out = F.silu(x)
# TODO: max_error = (custom_out - builtin_out).abs().max()
# print(f"Max output error: {max_error:.2e}")

# 2. Gradient check with finite differences
x_check = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
# TODO: passed = gradcheck(SwishFn.apply, (x_check,), eps=1e-6, atol=1e-4)
# print(f"Gradient check passed: {passed}")
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT 2: Gradient Reversal Layer                 ║
# ║                                                       ║
# ║  TODO: Verify that gradients are indeed negated.       ║
# ╚═══════════════════════════════════════════════════════╝

x = torch.randn(3, 4, requires_grad=True, device=DEVICE)

# Forward through identity (no reversal)
y_normal = x.sum()
y_normal.backward()
grad_normal = x.grad.clone()

# Forward through gradient reversal
x.grad = None
y_reversed = GradientReversalFn.apply(x, 1.0).sum()
y_reversed.backward()
grad_reversed = x.grad.clone()

# TODO: Verify grad_reversed ≈ -grad_normal
# diff = (grad_reversed + grad_normal).abs().max()  # should be ~0
# print(f"Reversal check: max|g_rev + g_normal| = {diff:.2e} (should be ~0)")
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT 3: Gradient Error vs Epsilon               ║
# ║                                                       ║
# ║  TODO: Plot how finite-difference error changes with   ║
# ║  different epsilon values. Shows the numerical tradeoff ║
# ║  between truncation error (large eps) and rounding      ║
# ║  error (small eps).                                     ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Use gradient_check() with multiple eps values
# result = gradient_check(SwishFn.apply, (x_check,),
#                         eps_values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
# 
# plt.figure(figsize=(8, 4))
# eps_vals = sorted(result['errors_by_eps'].keys())
# errors = [result['errors_by_eps'][e] for e in eps_vals]
# plt.loglog(eps_vals, errors, 'o-', linewidth=2)
# plt.xlabel("Epsilon")
# plt.ylabel("Max Gradient Error")
# plt.title("Finite Difference Error vs Epsilon")
# plt.grid(True, alpha=0.3)
# plt.show()
"""))

# ═══════════════════════════════════════════════════════
# SECTION 2 — LoRA
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 2 — LoRA from Scratch

## 2.1 Conceptual Background

### Low-Rank Adaptation (Hu et al., 2021)

Instead of fine-tuning all parameters W₀ ∈ ℝ^{d×k}:

$$
W = W_0 + \\Delta W = W_0 + BA
$$

Where:
- $B \\in \\mathbb{R}^{d \\times r}$, $A \\in \\mathbb{R}^{r \\times k}$
- $r \\ll \\min(d, k)$ — the rank
- $W_0$ is **frozen**; only $A, B$ are trained

### Why It Works

```
Full fine-tuning:   d × k parameters (e.g., 768 × 768 = 590K)
LoRA (r=8):         d × r + r × k = 768 × 8 + 8 × 768 = 12.3K
Reduction:          98% fewer trainable parameters!
```

### Key Design Choices

| Choice | Implementation Detail |
|--------|----------------------|
| **Initialization** | A ~ Kaiming, B = 0 → LoRA starts as identity (ΔW = 0) |
| **Scaling** | Output is scaled by α/r. Larger α = stronger adapter. |
| **Target modules** | Typically: Q, V projections in attention (not K) |
| **Merge for inference** | W_merged = W₀ + BA·(α/r) — no extra latency |"""))

cells.append(md("## 2.2 Implementation Tasks\n\nOpen `src/lora_from_scratch.py` and implement:\n\n1. **`LoRALinear`** — low-rank adapter module (init, forward, merge, unmerge)\n2. **`inject_lora()`** — dynamically replace Linear layers\n3. **`merge_lora()` / `unmerge_lora()`** — fold adapters into base weights\n4. **`state_dict_adapters_only()`** — save only LoRA params"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: LoRA Injection + Parameter Count          ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create a small model (e.g., from previous labs)    ║
# ║  2. Count parameters before LoRA                      ║
# ║  3. Inject LoRA into specific layers                   ║
# ║  4. Count trainable params after injection             ║
# ║  5. Show the reduction                                 ║
# ╚═══════════════════════════════════════════════════════╝

# Small model for testing
class TinyClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = TinyClassifier().to(DEVICE)
print("Before LoRA:")
print(count_parameters(model))

# TODO: Inject LoRA
# model = inject_lora(model, target_modules={'fc1', 'fc2'}, r=4, alpha=8)
# print("\\nAfter LoRA:")
# print(count_parameters(model))
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Merge/Unmerge Correctness                 ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Forward pass BEFORE merge → get output_before      ║
# ║  2. merge_lora(model)                                  ║
# ║  3. Forward pass AFTER merge → get output_after        ║
# ║  4. Verify outputs match within tolerance              ║
# ║  5. unmerge_lora(model), forward again → matches       ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Test merge/unmerge parity
# x = torch.randn(8, 784, device=DEVICE)
# 
# # Before merge (adapter path active)
# output_before = model(x)
# 
# # Merge adapters into base weights
# merge_lora(model)
# output_merged = model(x)
# 
# diff = (output_before - output_merged).abs().max()
# print(f"Merge parity: max diff = {diff:.2e} (should be < 1e-5)")
# assert diff < 1e-5, "Merge changed outputs!"
# 
# # Unmerge and verify
# unmerge_lora(model)
# output_unmerged = model(x)
# diff2 = (output_before - output_unmerged).abs().max()
# print(f"Unmerge parity: max diff = {diff2:.2e} (should be < 1e-5)")
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: LoRA vs Full Fine-Tuning                  ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Train full model on MNIST/CIFAR subset (all params)║
# ║  2. Train LoRA model (frozen base + adapters only)     ║
# ║  3. Compare: accuracy, trainable params, train time    ║
# ╚═══════════════════════════════════════════════════════╝

# Use a small subset for fast iteration
from torchvision.datasets import MNIST

transform_mnist = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
mnist_train = MNIST("./data", train=True, download=True, transform=transform_mnist)
mnist_test = MNIST("./data", train=False, download=True, transform=transform_mnist)

# Use subset for speed
subset_size = 5000
train_subset = torch.utils.data.Subset(mnist_train, range(subset_size))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=False)

# TODO: Implement comparison experiment
# 1. Full fine-tune: 3 epochs, track accuracy + time
# 2. LoRA fine-tune (r=4, alpha=8, target fc1+fc2): 3 epochs
# 3. Print comparison table
"""))

# ═══════════════════════════════════════════════════════
# SECTION 3 — CONTRASTIVE LEARNING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 3 — Contrastive Learning: InfoNCE from Scratch

## 3.1 Conceptual Background

### The InfoNCE / NT-Xent Loss (Chen et al., 2020)

For a batch of N images, create 2 augmented views each → 2N total embeddings.

For anchor $z_i$ with positive pair $z_j$:

$$
\\mathcal{L}_i = -\\log \\frac{\\exp(\\text{sim}(z_i, z_j) / \\tau)}{\\sum_{k \\neq i} \\exp(\\text{sim}(z_i, z_k) / \\tau)}
$$

Where $\\text{sim}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}$ (cosine similarity).

### Numerical Stability

The exponentials can overflow. The standard trick:

$$
\\log \\sum_k \\exp(x_k) = x_{\\max} + \\log \\sum_k \\exp(x_k - x_{\\max})
$$

In practice: **subtract the max logit per row** before softmax/cross-entropy.

### Temperature τ

| τ | Effect |
|---|--------|
| Low (0.1) | Sharper distribution → focuses on hard negatives, but harder to train |
| Medium (0.5) | Standard choice for SimCLR |
| High (1.0+) | Softer → easier training but less discriminative |"""))

cells.append(md("## 3.2 Implementation Tasks\n\nOpen `src/contrastive_losses.py` and implement:\n\n1. **`info_nce_loss()`** — numerically stable InfoNCE with correct positive indexing\n2. **`SimCLRProjector`** — projection MLP head"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: InfoNCE Correctness Check                 ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create test embeddings where positives are similar ║
# ║  2. Verify loss goes down when positives are aligned   ║
# ║  3. Verify accuracy metric makes sense                 ║
# ╚═══════════════════════════════════════════════════════╝

# Test 1: Perfect positive alignment → low loss
N, D = 32, 128
z1 = F.normalize(torch.randn(N, D, device=DEVICE), dim=1)
z2 = z1 + 0.01 * torch.randn_like(z1)  # Near-identical positives
z2 = F.normalize(z2, dim=1)

# TODO:
# loss, info = info_nce_loss(z1, z2, temperature=0.5)
# print(f"Near-identical: loss={loss:.4f}, acc={info['accuracy']:.2%}")
# assert info['accuracy'] > 0.9, "Accuracy should be high for near-identical pairs"

# Test 2: Random pairs → higher loss
z2_random = F.normalize(torch.randn(N, D, device=DEVICE), dim=1)
# loss_rand, info_rand = info_nce_loss(z1, z2_random, temperature=0.5)
# print(f"Random pairs:   loss={loss_rand:.4f}, acc={info_rand['accuracy']:.2%}")
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: SimCLR Training on CIFAR-10               ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Build encoder (ResNet-18 backbone) + projector     ║
# ║  2. Create TwoViewDataset with augmentations           ║
# ║  3. Train with InfoNCE for 10 epochs                   ║
# ║  4. Evaluate: linear probe on frozen embeddings        ║
# ╚═══════════════════════════════════════════════════════╝

# Setup
augmentations = get_simclr_augmentations(image_size=32)
cifar_raw = torchvision.datasets.CIFAR10("./data", train=True, download=True)
contrastive_dataset = TwoViewDataset(cifar_raw, augmentations)

# Use subset for speed
subset_idx = torch.randperm(len(contrastive_dataset))[:5000]
contrastive_subset = torch.utils.data.Subset(contrastive_dataset, subset_idx)
contrastive_loader = torch.utils.data.DataLoader(
    contrastive_subset, batch_size=256, shuffle=True, num_workers=2, drop_last=True
)

# TODO: Build encoder + projector
# encoder = torchvision.models.resnet18(num_classes=128)  # feature dim
# encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
# encoder.maxpool = nn.Identity()
# projector = SimCLRProjector(128, 256, 128).to(DEVICE)
# encoder = encoder.to(DEVICE)

# TODO: Training loop
# optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=3e-4)
# for epoch in range(10):
#     total_loss = 0
#     for view1, view2, _ in contrastive_loader:
#         view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
#         z1 = projector(encoder(view1))
#         z2 = projector(encoder(view2))
#         loss, info = info_nce_loss(z1, z2, temperature=0.5)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch}: loss={total_loss/len(contrastive_loader):.4f}")
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Temperature Ablation                      ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Train (or evaluate) with τ ∈ {0.1, 0.3, 0.5, 1.0} ║
# ║  2. Compare: loss value, contrastive accuracy          ║
# ║  3. Plot temperature vs metrics                        ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Temperature sweep
# temperatures = [0.1, 0.3, 0.5, 1.0]
# for tau in temperatures:
#     loss, info = info_nce_loss(z1_test, z2_test, temperature=tau)
#     print(f"τ={tau:.1f}: loss={loss:.4f}, acc={info['accuracy']:.2%}, "
#           f"pos_sim={info['avg_pos_sim']:.3f}, neg_sim={info['avg_neg_sim']:.3f}")
"""))

# ═══════════════════════════════════════════════════════
# SECTION 4 — GRADIENT SURGERY
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 4 — Gradient Surgery: PCGrad

## 4.1 Conceptual Background

### The Multi-Task Gradient Conflict Problem

When training a shared backbone on multiple tasks, gradients can **conflict**:

$$
\\cos(g_1, g_2) < 0 \\implies \\text{gradient directions oppose each other}
$$

Updating in the direction of $g_1$ **hurts** task 2, and vice versa.

### PCGrad (Yu et al., 2020)

**Idea**: When task gradients conflict, project one onto the normal plane of the other.

For gradients $g_i$ and $g_j$ with $g_i \\cdot g_j < 0$:

$$
g_i' = g_i - \\frac{g_i \\cdot g_j}{\\|g_j\\|^2} g_j
$$

This removes the component of $g_i$ that conflicts with $g_j$.

```
Before PCGrad:          After PCGrad:
  g1 ↗                  g1' →
  g2 ↙  (conflict!)     g2' →  (no conflict)
```

### When to Use

| Situation | Use PCGrad? |
|-----------|------------|
| Tasks clearly related | Not needed — gradients align |
| Tasks partially conflict | Yes — PCGrad helps |
| Tasks fundamentally incompatible | PCGrad helps but can't fix bad task design |"""))

cells.append(md("## 4.2 Implementation Tasks\n\nOpen `src/grad_surgery.py` and implement:\n\n1. **`compute_task_gradients()`** — extract per-task gradient vectors\n2. **`pcgrad_project()`** — project conflicting gradients\n3. **`pcgrad_step()`** — full PCGrad update step\n4. **`cosine_similarity_tasks()`** — measure gradient conflict"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Multi-Task Training with PCGrad           ║
# ║                                                       ║
# ║  Setup: CIFAR-10 with two tasks:                       ║
# ║    Task 1: "coarse" labels (vehicle vs animal)         ║
# ║    Task 2: fine-grained 10-class labels                ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create multi-task model (shared backbone + 2 heads)║
# ║  2. Train with vanilla joint loss (baseline)           ║
# ║  3. Train with PCGrad                                  ║
# ║  4. Compare task losses over time                      ║
# ║  5. Plot gradient cosine similarity                    ║
# ╚═══════════════════════════════════════════════════════╝

# CIFAR-10 coarse labels (provided mapping)
COARSE_MAP = {
    0: 0, 1: 0, 8: 0, 9: 0,  # vehicles: airplane, automobile, ship, truck
    2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,  # animals: bird, cat, deer, dog, frog, horse
}

# TODO: Multi-task setup
# model = MultiTaskNet(in_features=3*32*32, hidden=256, num_tasks=2, task_classes=[2, 10])
# model = model.to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# ...
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Gradient Conflict Visualization           ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Record cosine_similarity_tasks at each step        ║
# ║  2. Plot with plot_gradient_cosines()                  ║
# ║  3. Show that PCGrad reduces negative cosines          ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Record and plot
# cosine_history_vanilla = [...]
# cosine_history_pcgrad = [...]
# plot_gradient_cosines(cosine_history_vanilla, labels=["coarse", "fine"])
# plot_gradient_cosines(cosine_history_pcgrad, labels=["coarse", "fine"])
"""))

# ═══════════════════════════════════════════════════════
# SECTION 5 — REINFORCE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 5 — RL-Style Losses: REINFORCE on a Toy Bandit

## 5.1 Conceptual Background

### The Policy Gradient Theorem

For a stochastic policy $\\pi_\\theta(a)$, the objective is to maximize expected reward:

$$
J(\\theta) = \\mathbb{E}_{a \\sim \\pi_\\theta}[R(a)]
$$

The REINFORCE gradient estimator:

$$
\\nabla_\\theta J = \\mathbb{E}_{a \\sim \\pi_\\theta}[ R(a) \\cdot \\nabla_\\theta \\log \\pi_\\theta(a) ]
$$

**The log-probability trick**: sample an action, observe reward, update weights to
increase probability of high-reward actions.

### Variance Reduction via Baselines

The raw REINFORCE estimator has **high variance**. Subtract a baseline $b$:

$$
\\nabla_\\theta J = \\mathbb{E}[ (R(a) - b) \\cdot \\nabla_\\theta \\log \\pi_\\theta(a) ]
$$

This doesn't change the expected gradient (unbiased) but **reduces variance**.

Common baselines:
- **Moving average of rewards** — simple, effective
- **Learned value function** — more complex but more effective

### The K-Armed Bandit

Simplest RL setting: K actions, no states, just immediate rewards.

```
Action a=3 → Reward ~ N(μ₃, 1.0)
Goal: learn which arm has highest μ
```"""))

cells.append(md("## 5.2 Implementation Tasks\n\nOpen `src/reinforce_toy.py` and implement:\n\n1. **`KArmedBandit`** — environment with K arms\n2. **`PolicyNetwork`** — softmax policy (learnable logits)\n3. **`reinforce_update()`** — one REINFORCE step\n4. **`train_bandit()`** — full training loop with baseline"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: REINFORCE with and without Baseline       ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create a 10-armed bandit                           ║
# ║  2. Train policy WITHOUT baseline for 2000 steps       ║
# ║  3. Train policy WITH baseline for 2000 steps          ║
# ║  4. Compare: reward curves, optimal action %, variance ║
# ║  5. Plot with plot_bandit_results()                    ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Run experiment
# env = KArmedBandit(k=10, seed=42)
# print(f"True means: {env.true_means}")
# print(f"Optimal arm: {env.optimal_arm} (mean={env.true_means[env.optimal_arm]:.3f})")
# 
# # Without baseline
# policy_no_bl = PolicyNetwork(k=10)
# opt_no_bl = torch.optim.Adam(policy_no_bl.parameters(), lr=0.01)
# results_no_bl = train_bandit(env, policy_no_bl, opt_no_bl, num_steps=2000, use_baseline=False)
# 
# # With baseline
# policy_bl = PolicyNetwork(k=10)
# opt_bl = torch.optim.Adam(policy_bl.parameters(), lr=0.01)
# results_bl = train_bandit(env, policy_bl, opt_bl, num_steps=2000, use_baseline=True)
# 
# # Compare
# plot_bandit_results(results_no_bl, results_bl)
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Final Policy Analysis                     ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Show learned action probabilities                  ║
# ║  2. Compare with true arm means                        ║
# ║  3. Verify policy concentrates on optimal arm          ║
# ╚═══════════════════════════════════════════════════════╝

# TODO:
# probs = policy_bl.forward().detach().cpu().numpy()
# 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# 
# ax1.bar(range(10), env.true_means, alpha=0.7, label="True means")
# ax1.axhline(y=env.true_means[env.optimal_arm], color='r', linestyle='--', alpha=0.5)
# ax1.set_xlabel("Arm"); ax1.set_ylabel("Mean Reward"); ax1.set_title("True Arm Values")
# ax1.legend()
# 
# ax2.bar(range(10), probs, alpha=0.7, color='green', label="Learned π(a)")
# ax2.set_xlabel("Arm"); ax2.set_ylabel("Probability"); ax2.set_title("Learned Policy")
# ax2.legend()
# 
# plt.tight_layout()
# plt.show()
"""))

# ═══════════════════════════════════════════════════════
# FINAL CHALLENGE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# 🧪 Final Challenge — Paper Replication Mindset

## Required Report

Write a short report answering:

1. **Correctness tests used**: What tests did you run for each section? (gradcheck, merge parity, indexing asserts, etc.)
2. **Numerical stability fixes**: What stability issues arose and how did you fix them? (logit subtraction, epsilon in division, etc.)
3. **Ablations that mattered**: Which hyperparameters mattered most? (LoRA rank, temperature, baseline decay)
4. **Scaling considerations**: What would need to change for large models? (memory, distributed, mixed precision)"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  FINAL REPORT                                         ║
# ║                                                       ║
# ║  TODO: Fill in your findings.                          ║
# ╚═══════════════════════════════════════════════════════╝

FINAL_REPORT = \"\"\"
# Research Implementation Report

## 1. Correctness Tests
- Custom autograd: TODO — what gradcheck results did you get?
- LoRA merge/unmerge: TODO — max diff observed?
- InfoNCE: TODO — how did you verify positive indexing?
- REINFORCE: TODO — did the policy converge to optimal arm?

## 2. Numerical Stability Fixes
- InfoNCE: TODO — what happened without subtracting max logits?
- Gradient surgery: TODO — division by zero handling?
- REINFORCE: TODO — any NaN issues with log_prob?

## 3. Ablations That Mattered
- LoRA: TODO — rank r vs accuracy tradeoff?
- InfoNCE: TODO — which temperature worked best and why?
- REINFORCE: TODO — baseline decay impact on convergence speed?
- PCGrad: TODO — did projection actually help your task pair?

## 4. Scaling Considerations
- LoRA: TODO — what about applying to attention layers in a real Transformer?
- Contrastive: TODO — how would distributed negatives help?
- PCGrad: TODO — cost of per-task backward passes?
\"\"\"
print(FINAL_REPORT)"""))

# ═══════════════════════════════════════════════════════
# CHECKLIST
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# ✅ Summary Checklist

| # | Competency | Confident? |
|---|-----------|-----------|
| 1 | I can write `autograd.Function` with correct forward/backward | ☐ |
| 2 | I can verify gradients with finite differences / gradcheck | ☐ |
| 3 | I understand gradient reversal and straight-through estimation | ☐ |
| 4 | I can implement LoRA: inject, train, merge, unmerge, save adapters | ☐ |
| 5 | I can implement InfoNCE with numerical stability | ☐ |
| 6 | I understand temperature's effect on contrastive learning | ☐ |
| 7 | I can extract per-task gradients and apply PCGrad projection | ☐ |
| 8 | I can measure gradient conflict with cosine similarity | ☐ |
| 9 | I can implement REINFORCE with the log-probability trick | ☐ |
| 10 | I understand variance reduction via baselines | ☐ |

### Common Failure Modes

```
✗ autograd.Function backward returns wrong number of gradients
✗ LoRA B initialized non-zero → model starts with random perturbation
✗ InfoNCE includes self-similarity in negatives → loss is always low
✗ Not subtracting max logits → exp overflow → NaN
✗ PCGrad: modifying original gradient tensor instead of clone → corruption
✗ REINFORCE without baseline → reward signal too noisy to learn
✗ Using loss.item() in backward → detaches from graph
```"""))

# ═══════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells,
}

out_path = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "notebooks", "07_research_extensions_lab.ipynb"
))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook: {out_path}")
print(f"Cells: {len(cells)} (markdown: {md_count}, code: {code_count})")
