#!/usr/bin/env python3
"""Generate the Group 2 Training Dynamics & Stability Lab notebook."""
import json, os

def md(source):
    if isinstance(source, str):
        source = source.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in source[:-1]] + [source[-1]]}

def code(source):
    if isinstance(source, str):
        source = source.split("\n")
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in source[:-1]] + [source[-1]], "execution_count": None, "outputs": []}

cells = []

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
cells.append(md("""\
# 🔬 Notebook 2 — Training Dynamics Lab
## Initialization, Stability, and Reproducibility in PyTorch

**Group 2 — Training Dynamics & Stability**

---

### 🎯 Single Concept

How training **succeeds** or **quietly fails** — and how to systematically diagnose and fix it.

### Learning Objectives

By the end of this lab you will:

1. Choose and implement sensible initialization for MLPs and CNNs
2. Detect vanishing/exploding gradients and dead ReLUs
3. Configure optimizers and schedulers intentionally
4. Run controlled ablations with reproducible runs
5. Verify evaluation correctness (`train()` vs `eval()`)
6. Build a reusable **Stability & Reproducibility Toolkit** for later groups

### 📂 File Structure

```
Group_2_Training_Dynamics/
├── notebooks/
│   └── 02_training_dynamics_stability.ipynb   ← you are here
└── src/
    ├── init_and_activations.py    ← init schemes + activation factory (you implement)
    ├── optim_and_schedules.py     ← optimizer + scheduler builders    (you implement)
    ├── stability_tools.py         ← grad/activation diagnostics      (you implement)
    └── reproducibility.py         ← seed/determinism utilities       (you implement)
```

> ⚠️ **Rule**: Core logic lives in the `src/` modules as TODO stubs. This notebook orchestrates experiments and calls your implementations."""))

# ═══════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════
cells.append(md("## 0 — Environment Setup"))

cells.append(code("""\
import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from collections import OrderedDict

# Add src/ to path
sys.path.insert(0, os.path.join(os.pardir, "src"))

# Your modules (implement the TODOs!)
from init_and_activations import make_activation, init_weights, SmallCNN
from optim_and_schedules import build_optimizer, build_scheduler, WarmupThenCosineScheduler, plot_lr_schedule
from stability_tools import (
    grad_norms, activation_stats, register_activation_hooks, detect_anomalies,
    plot_grad_norms_layerwise, plot_dead_relu_pct, plot_training_curves, plot_multi_curves,
)
from reproducibility import seed_everything, seed_worker, make_dataloader, verify_reproducibility

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device          : {DEVICE}")"""))

# ── CIFAR-10 Data Loading ──
cells.append(md("### 0.1 — CIFAR-10 Dataset"))

cells.append(code("""\
import torchvision
import torchvision.transforms as T

# ── Transforms ──
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# ── Full datasets ──
train_dataset_full = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

# ── Subset option for speed ──
SUBSET_SIZE = 5000  # Set to None for full dataset
if SUBSET_SIZE is not None:
    train_dataset = torch.utils.data.Subset(train_dataset_full, range(SUBSET_SIZE))
    print(f"Using subset: {SUBSET_SIZE} training samples")
else:
    train_dataset = train_dataset_full
    print(f"Using full dataset: {len(train_dataset_full)} training samples")

print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset_full.classes}")"""))

# ── Minimal Trainer ──
cells.append(md("""\
### 0.2 — Minimal Training Scaffold (Provided)

This is a lightweight training loop that integrates with your stability tools.
You do **not** need to modify this — it calls your `src/` functions."""))

cells.append(code("""\
def train_one_epoch(model, loader, optimizer, scheduler, device, hooks_cache=None):
    \"\"\"Train for one epoch. Returns avg loss and per-step grad norms.\"\"\"
    model.train()
    total_loss = 0.0
    step_grad_norms = []
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # Capture grad norms before clipping (student tool)
        try:
            gn = grad_norms(model)
            step_grad_norms.append(gn.get("global", 0.0))
        except NotImplementedError:
            step_grad_norms.append(0.0)

        optimizer.step()
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader), step_grad_norms


@torch.no_grad()
def evaluate(model, loader, device):
    \"\"\"Evaluate accuracy. Model MUST be in eval() mode.\"\"\"
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def run_training(
    config: dict,
    train_ds=None,
    test_ds=None,
    verbose: bool = True,
) -> dict:
    \"\"\"
    Full training run from config dict. Returns metrics.

    Config keys:
      seed, deterministic, init_scheme, act_name, optimizer_name,
      lr, weight_decay, scheduler_name, epochs, batch_size,
      grad_clip_norm (optional), dropout_p (optional)
    \"\"\"
    # ── Reproducibility ──
    try:
        seed_everything(config.get("seed", 42), config.get("deterministic", True))
    except NotImplementedError:
        torch.manual_seed(config.get("seed", 42))

    ds_train = train_ds or train_dataset
    ds_test = test_ds or test_dataset

    try:
        train_loader = make_dataloader(
            ds_train, batch_size=config.get("batch_size", 64),
            shuffle=True, num_workers=0, seed=config.get("seed", 42),
            deterministic=config.get("deterministic", True),
        )
    except NotImplementedError:
        train_loader = torch.utils.data.DataLoader(
            ds_train, batch_size=config.get("batch_size", 64), shuffle=True, num_workers=0,
        )
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=256, shuffle=False, num_workers=0)

    # ── Model ──
    model = SmallCNN(
        act_name=config.get("act_name", "relu"),
        init_scheme=config.get("init_scheme", "kaiming_normal"),
        dropout_p=config.get("dropout_p", 0.3),
    ).to(DEVICE)

    # ── Optimizer ──
    try:
        optimizer = build_optimizer(
            model.parameters(), name=config.get("optimizer_name", "adamw"),
            lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 0.01),
        )
    except NotImplementedError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3))

    # ── Scheduler ──
    total_steps = config.get("epochs", 5) * len(train_loader)
    try:
        sched_kwargs = config.get("scheduler_kwargs", {})
        scheduler = build_scheduler(
            optimizer, name=config.get("scheduler_name", "cosine"),
            T_max=total_steps, **sched_kwargs,
        )
    except NotImplementedError:
        scheduler = None

    # ── Training ──
    train_losses, val_accs, epoch_grad_norms = [], [], []
    for epoch in range(config.get("epochs", 5)):
        loss, gn_list = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_acc = evaluate(model, test_loader, DEVICE)
        train_losses.append(loss)
        val_accs.append(val_acc)
        epoch_grad_norms.append(np.mean(gn_list) if gn_list else 0.0)
        if verbose:
            print(f"  Epoch {epoch+1}/{config['epochs']}  loss={loss:.4f}  val_acc={val_acc:.1f}%  avg_grad_norm={epoch_grad_norms[-1]:.4f}")

    return {
        "train_loss": train_losses,
        "val_acc": val_accs,
        "grad_norms": epoch_grad_norms,
        "final_val_acc": val_accs[-1],
        "model": model,
    }"""))

# ═══════════════════════════════════════════════════════
# SECTION 1 — PURPOSE & FAILURE MODES
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 1 — Purpose & Failure Modes

## What Is a "Quiet Failure"?

A quiet failure is when your model **trains without crashing** but:

| Symptom | What It Looks Like | Root Cause |
|---------|-------------------|------------|
| Loss plateau | Loss stops decreasing after epoch 2 | Bad init, dead ReLUs, LR too low |
| Accuracy stuck | Val accuracy stays at ~10% (random) | Symmetry breaking failure, data bug |
| Unstable metrics | Loss oscillates wildly | LR too high, no gradient clipping |
| NaN loss | `loss = nan` after a few steps | Numerical instability, exploding grads |
| Train ≫ Val | 99% train, 30% val accuracy | No regularization, overfitting |
| Non-reproducible | Different results each run | Missing seed, non-deterministic ops |
| eval() bug | Val metrics change unexpectedly | Forgot `model.eval()`, BatchNorm/Dropout |

## What We'll Build

A **Stability & Reproducibility Toolkit** consisting of:

1. `init_weights()` — configurable initialization
2. `make_activation()` — activation layer factory
3. `grad_norms()` — per-layer gradient monitoring
4. `activation_stats()` — dead neuron detection
5. `detect_anomalies()` — NaN/Inf checks
6. `seed_everything()` — full determinism
7. `make_dataloader()` — reproducible data loading
8. `overfit_one_batch()` — fastest sanity check

> 💡 **Key Insight**: "It runs" ≠ "It works." Systematic diagnostics catch what loss curves hide."""))

# ═══════════════════════════════════════════════════════
# SECTION 2 — INITIALIZATION
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 2 — Initialization: Why It Matters

## 2.1 Theory

### The Symmetry Problem

If all weights are initialized identically (e.g., zeros or same constant):
- All neurons compute the same output
- All receive the same gradient
- They can **never** learn different features
- The network has effectively **one neuron per layer**

### Variance Propagation

For a deep network, if we want the **variance of activations** to stay roughly constant across layers:

| Init Scheme | Best For | Formula |
|-------------|----------|---------|
| **Xavier Uniform/Normal** | Sigmoid, Tanh | `Var(w) = 2 / (fan_in + fan_out)` |
| **Kaiming (He) Uniform/Normal** | ReLU, LeakyReLU | `Var(w) = 2 / fan_in` |
| **Orthogonal** | RNNs, very deep nets | Orthogonal matrix, preserves norm |

### Why Kaiming for ReLU?

ReLU zeros out ~50% of activations, halving the variance. Kaiming compensates by **doubling** the initial variance compared to Xavier.

### Practical Rule

> **Default**: Use Kaiming Normal with ReLU. Use Xavier with Tanh/Sigmoid. Use orthogonal when training very deep or recurrent networks.

### Common Pitfall

PyTorch's default init (`nn.Linear`, `nn.Conv2d`) uses Kaiming Uniform — which is *usually* fine but not always optimal."""))

cells.append(md("""\
## 2.2 Implementation Tasks

Open `src/init_and_activations.py` and implement:

1. **`make_activation(name)`** — Activation factory supporting: relu, leaky_relu, gelu, silu
2. **`init_weights(module, scheme, nonlinearity)`** — Initialization applicator
3. **`SmallCNN.__init__()` and `.forward()`** — Build the CNN using your init + activation functions

Then return here to run the experiments."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify your implementations            ║
# ╚═══════════════════════════════════════════════════════╝

# Test make_activation
for name in ["relu", "leaky_relu", "gelu", "silu"]:
    act = make_activation(name)
    test_input = torch.randn(4)
    output = act(test_input)
    assert output.shape == test_input.shape, f"{name} shape mismatch"
    print(f"  ✅ make_activation('{name}') → {act.__class__.__name__}")

# Test init_weights
test_model = nn.Linear(10, 5)
for scheme in ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal"]:
    init_weights(test_model, scheme=scheme, nonlinearity="relu")
    assert not torch.all(test_model.weight == 0), f"{scheme} produced all zeros"
    print(f"  ✅ init_weights(scheme='{scheme}') applied successfully")

# Test SmallCNN
model = SmallCNN(act_name="relu", init_scheme="kaiming_normal")
x_test = torch.randn(2, 3, 32, 32)
out_test = model(x_test)
assert out_test.shape == (2, 10), f"Expected (2,10) but got {out_test.shape}"
print(f"  ✅ SmallCNN forward pass: input {x_test.shape} → output {out_test.shape}")
print(f"  ✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}")"""))

# ── Init ablation experiment ──
cells.append(md("""\
## 2.3 Experiment — Initialization Ablation

Train the same CNN with different initialization schemes.
Observe the impact on convergence, gradient norms, and final accuracy."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Compare initialization schemes            ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Define configs for at least 4 schemes:             ║
# ║     - zeros (BAD)                                      ║
# ║     - constant_small (BAD)                              ║
# ║     - xavier_normal                                     ║
# ║     - kaiming_normal                                    ║
# ║  2. Run run_training() for each                        ║
# ║  3. Collect results into a dict for plot_multi_curves  ║
# ║  4. Print a comparison table                           ║
# ║                                                       ║
# ║  Expected: zeros/constant should fail badly.            ║
# ║  Xavier vs Kaiming should show subtle differences.      ║
# ╚═══════════════════════════════════════════════════════╝

EPOCHS = 5  # Increase for clearer separation
BASE_CONFIG = dict(
    seed=42, deterministic=True, act_name="relu",
    optimizer_name="adamw", lr=1e-3, weight_decay=0.01,
    scheduler_name="cosine", epochs=EPOCHS, batch_size=64,
)

init_schemes = ["zeros", "constant_small", "xavier_normal", "kaiming_normal"]
results = {}

for scheme in init_schemes:
    print(f"\\n{'='*50}")
    print(f"  Init: {scheme}")
    print(f"{'='*50}")
    cfg = {**BASE_CONFIG, "init_scheme": scheme}
    # TODO: results[scheme] = run_training(cfg)

# TODO: Plot comparison
# plot_multi_curves(results)"""))

cells.append(md("""\
### 2.4 Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| All-zeros init | Loss never decreases, all outputs identical | Use Xavier or Kaiming |
| Too-small init | Very slow convergence, vanishing activations | Increase init scale |
| Xavier + ReLU | Slightly worse than Kaiming for deep nets | Switch to Kaiming for ReLU |
| Forgetting bias init | Usually minor, but can matter for BatchNorm | Init bias to zeros |"""))

# ═══════════════════════════════════════════════════════
# SECTION 3 — GRADIENT PATHOLOGIES & DEAD NEURONS
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 3 — Gradient Pathologies & Dead Neurons

## 3.1 Theory

### Vanishing Gradients
- Gradients shrink exponentially through many layers
- Sigmoid/Tanh saturate → derivatives near zero
- Early layers stop learning

### Exploding Gradients
- Gradients grow exponentially
- Manifests as NaN loss, parameter overflow
- Common in RNNs and very deep nets

### Dead ReLUs
- Once a ReLU neuron outputs 0 for all inputs, it stays dead
- Its gradient is exactly 0, so it never recovers
- Can happen due to large negative bias or large weight updates
- **Detection**: `% zero activations > 50%` per layer

### Fixes

| Problem | Solution |
|---------|----------|
| Vanishing | LeakyReLU/GELU, proper init, residual connections |
| Exploding | Gradient clipping, lower LR, proper init |
| Dead ReLU | LeakyReLU (small negative slope), better init |"""))

cells.append(md("""\
## 3.2 Implementation Tasks

Open `src/stability_tools.py` and implement:

1. **`grad_norms(model)`** — per-layer + global gradient norms
2. **`register_activation_hooks(model)`** — forward hooks for activation capture
3. **`activation_stats(cache)`** — mean/std/%zeros from hook cache
4. **`detect_anomalies(loss, model)`** — NaN/Inf detection

Then return here to run diagnostics."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify stability tools                  ║
# ╚═══════════════════════════════════════════════════════╝

test_model = SmallCNN(act_name="relu", init_scheme="kaiming_normal").to(DEVICE)
x = torch.randn(4, 3, 32, 32, device=DEVICE)
y = torch.randint(0, 10, (4,), device=DEVICE)

# Forward + backward
output = test_model(x)
loss = F.cross_entropy(output, y)
loss.backward()

# Test grad_norms
gn = grad_norms(test_model)
assert "global" in gn, "grad_norms must return a 'global' key"
assert gn["global"] > 0, "global grad norm should be > 0"
print(f"  ✅ grad_norms: global={gn['global']:.4f}, {len(gn)-1} layers")

# Test activation hooks
handles, cache = register_activation_hooks(test_model)
_ = test_model(x)  # forward to populate cache
stats = activation_stats(cache)
assert len(stats) > 0, "activation_stats returned empty"
print(f"  ✅ activation_stats: {len(stats)} layers captured")
for h in handles:
    h.remove()

# Test detect_anomalies
anomalies = detect_anomalies(loss, test_model)
assert "is_healthy" in anomalies, "detect_anomalies must return 'is_healthy'"
print(f"  ✅ detect_anomalies: healthy={anomalies['is_healthy']}")"""))

# ── Exploding gradients experiment ──
cells.append(md("## 3.3 Experiment — Creating and Fixing Exploding Gradients"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Intentional exploding gradients           ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create SmallCNN with a VERY high learning rate     ║
# ║     (e.g., lr=1.0 or lr=10.0 with SGD)                ║
# ║  2. Train for a few steps                              ║
# ║  3. Log grad_norms at each step                        ║
# ║  4. Show that gradients explode                        ║
# ║  5. Apply gradient clipping and show it helps:         ║
# ║     torch.nn.utils.clip_grad_norm_(params, max_norm=1) ║
# ╚═══════════════════════════════════════════════════════╝

# Without clipping
print("=== NO GRADIENT CLIPPING (lr=1.0) ===")
# TODO: Train a few steps, print grad norms

# With clipping
print("\\n=== WITH GRADIENT CLIPPING (max_norm=1.0) ===")
# TODO: Same setup but clip_grad_norm_ before optimizer.step()

# TODO: plot_grad_norms_layerwise for both cases"""))

# ── Dead ReLU experiment ──
cells.append(md("## 3.4 Experiment — Dead ReLU Detection"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Dead ReLUs                                ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create SmallCNN with bad init (constant_small)     ║
# ║     and activation='relu'                               ║
# ║  2. Register activation hooks                          ║
# ║  3. Forward a batch                                     ║
# ║  4. Compute activation_stats                            ║
# ║  5. Plot % zero activations per layer                  ║
# ║     using plot_dead_relu_pct                            ║
# ║  6. Repeat with activation='leaky_relu'                 ║
# ║  7. Compare: LeakyReLU should have far fewer zeros     ║
# ╚═══════════════════════════════════════════════════════╝

# ReLU with bad init
print("=== ReLU + constant_small init ===")
# TODO

# LeakyReLU with same bad init
print("\\n=== LeakyReLU + constant_small init ===")
# TODO"""))

cells.append(md("""\
### 3.5 Common Pitfalls

| Pitfall | How To Catch It |
|---------|----------------|
| Not checking grad norms during training | Plot grad norms per epoch |
| Dead neurons invisible in loss curve | Use activation hooks + % zeros plot |
| NaN appears late in training | Run `detect_anomalies` periodically |
| Gradient clipping too aggressive | Monitor clipped vs unclipped norm ratio |"""))

# ═══════════════════════════════════════════════════════
# SECTION 4 — OPTIMIZERS
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 4 — Optimizers: State & Weight Decay

## 4.1 Theory

### SGD + Momentum vs Adam vs AdamW

| Optimizer | Key Property | When To Use |
|-----------|-------------|-------------|
| **SGD+Momentum** | Simple, well-understood | Large-scale vision (ResNet etc.) |
| **Adam** | Adaptive LR per parameter | Fast convergence, noisy gradients |
| **AdamW** | Decoupled weight decay | Transformers, modern default |

### Weight Decay vs L2 Regularization

**They are NOT the same thing with Adam!**

- **L2 Regularization**: Add `λ‖w‖²` to the loss → gradient becomes `∇L + 2λw`
- **Weight Decay**: Directly scale weights: `w ← w - α·decay·w` *after* the update

With SGD, these are mathematically equivalent. With Adam, they are **not** because Adam normalizes the gradient by its running variance. AdamW correctly decouples decay from the adaptive gradient step.

> 💡 **Rule**: Use AdamW (not Adam + L2) unless you have a specific reason."""))

cells.append(md("""\
## 4.2 Implementation Tasks

Open `src/optim_and_schedules.py` and implement:

1. **`build_optimizer(params, name, lr, weight_decay)`** — Optimizer factory
2. Run the ablation below to compare optimizer behavior"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify build_optimizer                  ║
# ╚═══════════════════════════════════════════════════════╝

dummy_model = nn.Linear(10, 5)
for opt_name in ["sgd", "adam", "adamw"]:
    opt = build_optimizer(dummy_model.parameters(), name=opt_name, lr=1e-3, weight_decay=0.01)
    print(f"  ✅ build_optimizer('{opt_name}') → {opt.__class__.__name__}")"""))

cells.append(md("## 4.3 Experiment — Optimizer Comparison"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: SGD+Momentum vs AdamW                    ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Fix init_scheme='kaiming_normal', act='relu'       ║
# ║  2. Train with SGD (momentum=0.9, lr=0.01)            ║
# ║  3. Train with AdamW (lr=1e-3)                         ║
# ║  4. Compare train/val curves                           ║
# ║  5. Also compare weight_decay=0.0 vs 0.01 vs 0.1      ║
# ╚═══════════════════════════════════════════════════════╝

EPOCHS = 10

optimizer_configs = {
    "SGD_momentum": dict(optimizer_name="sgd", lr=0.01, weight_decay=0.0),
    "AdamW_wd0":    dict(optimizer_name="adamw", lr=1e-3, weight_decay=0.0),
    "AdamW_wd001":  dict(optimizer_name="adamw", lr=1e-3, weight_decay=0.01),
    "AdamW_wd01":   dict(optimizer_name="adamw", lr=1e-3, weight_decay=0.1),
}

results = {}
for label, opt_cfg in optimizer_configs.items():
    print(f"\\n{'='*50}\\n  {label}\\n{'='*50}")
    cfg = {
        **BASE_CONFIG, **opt_cfg,
        "epochs": EPOCHS, "scheduler_name": "cosine",
    }
    # TODO: results[label] = run_training(cfg)

# TODO: plot_multi_curves(results)
# TODO: Print final accuracy table"""))

# ═══════════════════════════════════════════════════════
# SECTION 5 — SCHEDULERS & WARMUP
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 5 — Schedulers & Warmup

## 5.1 Theory

### Why Schedules Matter

A fixed learning rate is suboptimal: you want **large steps early** (explore) and **small steps late** (fine-tune).

| Schedule | Shape | Best For |
|----------|-------|----------|
| **StepLR** | Stairs (decay every N epochs) | Simple baselines |
| **CosineAnnealing** | Smooth cosine curve | Most training |
| **OneCycleLR** | Ramp up then down | Fast convergence |
| **Warmup + Cosine** | Linear ramp → cosine | Transformers, large LR |

### Warmup Intuition

At the start of training:
- Adam's running statistics are initialized to 0 → first few updates are wild
- A high LR on top of that → instability
- **Warmup** starts with near-zero LR and linearly ramps up

> 💡 **Rule**: Always use warmup if LR > 1e-3 or if using Transformers."""))

cells.append(md("""\
## 5.2 Implementation Task

Open `src/optim_and_schedules.py` and implement:

1. **`build_scheduler(optimizer, name, **kwargs)`** — Scheduler factory
2. **`WarmupThenCosineScheduler`** — Custom warmup + cosine decay"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify schedulers                       ║
# ╚═══════════════════════════════════════════════════════╝

dummy_model = nn.Linear(10, 5)
dummy_opt = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)

# Test build_scheduler
for sched_name in ["step", "cosine"]:
    sched = build_scheduler(dummy_opt, name=sched_name, step_size=10, gamma=0.5, T_max=100)
    print(f"  ✅ build_scheduler('{sched_name}') → {sched.__class__.__name__}")

# Test WarmupThenCosineScheduler
dummy_opt2 = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)
warmup_sched = WarmupThenCosineScheduler(dummy_opt2, warmup_steps=20, T_max=100)
lrs = []
for _ in range(100):
    lrs.append(dummy_opt2.param_groups[0]["lr"])
    dummy_opt2.step()
    warmup_sched.step()
assert lrs[0] < lrs[19], "LR should increase during warmup"
assert lrs[20] > lrs[-1], "LR should decrease after warmup"
print(f"  ✅ WarmupThenCosineScheduler: warmup peak LR={max(lrs):.6f}")"""))

cells.append(md("## 5.3 Experiment — Scheduler Comparison"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Compare LR schedules                     ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Compare at least 3 schedules:                      ║
# ║     - constant (no scheduler)                          ║
# ║     - cosine                                           ║
# ║     - warmup_cosine                                    ║
# ║  2. Use same optimizer (AdamW), init, activation       ║
# ║  3. Plot LR curves AND train/val curves                ║
# ║  4. Show if warmup helps stability in early epochs     ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Visualize LR schedules first using plot_lr_schedule

# TODO: Train with each and compare
"""))

# ═══════════════════════════════════════════════════════
# SECTION 6 — REPRODUCIBILITY
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 6 — Reproducibility: Determinism Done Right

## 6.1 Theory

### Why `seed = 42` Is Not Enough

Setting `torch.manual_seed(42)` only seeds the CPU generator. For full reproducibility:

| Layer | What To Seed |
|-------|-------------|
| Python | `random.seed(seed)` |
| NumPy | `np.random.seed(seed)` |
| PyTorch CPU | `torch.manual_seed(seed)` |
| PyTorch GPU | `torch.cuda.manual_seed_all(seed)` |
| cuDNN | `cudnn.deterministic=True`, `benchmark=False` |
| DataLoader | `worker_init_fn`, explicit `Generator` |
| CUBLAS | `CUBLAS_WORKSPACE_CONFIG=":4096:8"` |

### Performance Cost

Deterministic mode disables cuDNN autotuning and forces deterministic CUDA kernels, which can be **10-30% slower**. This is acceptable for debugging and ablations, not for production training.

### `model.train()` vs `model.eval()` Is Critical

| Layer | `train()` mode | `eval()` mode |
|-------|---------------|--------------|
| **Dropout** | Active (random zeros) | Disabled (identity) |
| **BatchNorm** | Uses batch statistics | Uses running statistics |

**The Bug**: Forgetting `model.eval()` before evaluation → Dropout still active → accuracy looks worse than it is. Or forgetting `model.train()` after eval → BatchNorm uses stale stats during training."""))

cells.append(md("""\
## 6.2 Implementation Tasks

Open `src/reproducibility.py` and implement:

1. **`seed_everything(seed, deterministic)`** — Full reproducibility setup
2. **`seed_worker(worker_id)`** — DataLoader worker seeding
3. **`make_dataloader(dataset, ..., seed, deterministic)`** — Reproducible DataLoader"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify reproducibility tools            ║
# ╚═══════════════════════════════════════════════════════╝

# Test seed_everything
seed_everything(123, deterministic=True)
a = torch.randn(5)
seed_everything(123, deterministic=True)
b = torch.randn(5)
assert torch.equal(a, b), "seed_everything should produce identical tensors"
print("  ✅ seed_everything: identical tensors with same seed")

# Test make_dataloader
dl = make_dataloader(train_dataset, batch_size=32, seed=42, num_workers=0)
batch1 = next(iter(dl))[0]
dl = make_dataloader(train_dataset, batch_size=32, seed=42, num_workers=0)
batch2 = next(iter(dl))[0]
assert torch.equal(batch1, batch2), "make_dataloader should produce identical batches"
print("  ✅ make_dataloader: identical batch ordering with same seed")"""))

cells.append(md("## 6.3 Experiment — Reproducibility Verification"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Verify deterministic training             ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Run run_training 3 times with deterministic=True   ║
# ║     and same seed → results MUST be identical          ║
# ║  2. Run 3 times with deterministic=False               ║
# ║     → results should vary slightly                     ║
# ║  3. Use verify_reproducibility() helper                ║
# ╚═══════════════════════════════════════════════════════╝

cfg_deterministic = {
    **BASE_CONFIG, "epochs": 3, "deterministic": True, "seed": 42,
}

print("=== Deterministic Mode ===")
# TODO: verify_reproducibility(run_training, cfg_deterministic)

cfg_nondeterministic = {
    **BASE_CONFIG, "epochs": 3, "deterministic": False, "seed": 42,
}

print("\\n=== Non-Deterministic Mode ===")
# TODO: verify_reproducibility(run_training, cfg_nondeterministic)"""))

# ── train vs eval ──
cells.append(md("## 6.4 Experiment — `train()` vs `eval()` Correctness"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Show train() vs eval() matters            ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Train a model for a few epochs                     ║
# ║  2. Evaluate with model.eval() → record accuracy       ║
# ║  3. Evaluate with model.train() → record accuracy      ║
# ║  4. Show the difference (should be significant with    ║
# ║     Dropout and BatchNorm)                              ║
# ║  5. Run eval multiple times in train() mode to show    ║
# ║     the variance from active Dropout                    ║
# ╚═══════════════════════════════════════════════════════╝

# Train a model first
# TODO

# Evaluate correctly
# model.eval()
# acc_eval = evaluate(model, test_loader, DEVICE)

# Evaluate INCORRECTLY (forgot eval)
# model.train()
# acc_train_mode = evaluate(model, test_loader, DEVICE)

# Run multiple times in train mode to show variance
# accs_train_mode = [evaluate(model, test_loader, DEVICE) for _ in range(5)]

# print(f"eval() mode accuracy: {acc_eval:.2f}%")
# print(f"train() mode accuracy: {acc_train_mode:.2f}%")
# print(f"train() mode 5 runs: {accs_train_mode}")
# print(f"  → variance from Dropout: {np.std(accs_train_mode):.4f}")"""))

# ═══════════════════════════════════════════════════════
# SECTION 7 — OVERFIT ONE BATCH
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 7 — The "Overfit One Batch" Test

## 7.1 Why This Is the Fastest Sanity Check

Before training for hours, try to **perfectly overfit a single batch**.

If your model **cannot** reach ~100% accuracy on one batch:
- ❌ Bug in the data pipeline (labels scrambled, transforms wrong)
- ❌ Bug in the model (output shape mismatch, wrong activation)
- ❌ Bug in the loss function (using wrong loss for the task)
- ❌ Learning rate is way too low or too high

If it **can** overfit one batch, you know the fundamentals work. Scale up to the full dataset.

## 7.2 Implementation Task

Implement `overfit_one_batch()` below."""))

cells.append(code("""\
def overfit_one_batch(
    model: nn.Module,
    batch: tuple,
    optimizer=None,
    steps: int = 200,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    \"\"\"
    Attempt to overfit a single batch to near-zero loss.

    Parameters
    ----------
    model : nn.Module
    batch : (images, labels) tuple
    optimizer : Optimizer (if None, use AdamW with lr=1e-3)
    steps : int
    device : str or torch.device
    verbose : bool

    Returns
    -------
    dict with:
      'success': bool     ← True if loss < 0.01
      'final_loss': float
      'final_acc': float  ← % correct on the batch
      'losses': list      ← loss at each step

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Move model and batch to device                    ║
    ║  2. model.train()                                     ║
    ║  3. If optimizer is None, create AdamW(lr=1e-3)       ║
    ║  4. Loop for `steps`:                                  ║
    ║     a. zero_grad, forward, loss, backward, step       ║
    ║     b. Record loss                                     ║
    ║     c. Check for NaN → break early                    ║
    ║     d. Optionally print every 50 steps                ║
    ║  5. Compute final accuracy on the batch                ║
    ║  6. Return results dict                                ║
    ║                                                       ║
    ║  Hint: success threshold is loss < 0.01               ║
    ╚═══════════════════════════════════════════════════════╝
    \"\"\"
    raise NotImplementedError("TODO: implement overfit_one_batch()")"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  TEST: Overfit one batch                               ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Get one batch from the training DataLoader         ║
# ║  2. Create a fresh SmallCNN                            ║
# ║  3. Run overfit_one_batch(model, batch, ...)           ║
# ║  4. Assert success == True                              ║
# ║  5. Plot the loss curve                                ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Get one batch
# train_loader = ...
# batch = next(iter(train_loader))

# TODO: Run overfit test
# model = SmallCNN(act_name="relu", init_scheme="kaiming_normal").to(DEVICE)
# result = overfit_one_batch(model, batch, steps=300, device=DEVICE)

# TODO: Plot
# plt.plot(result["losses"])
# plt.xlabel("Step"); plt.ylabel("Loss")
# plt.title(f"Overfit One Batch — {'✅ PASS' if result['success'] else '❌ FAIL'}")
# plt.grid(alpha=0.3); plt.show()"""))

# ═══════════════════════════════════════════════════════
# SECTION 8 — FINAL CHALLENGE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# 🧪 Section 8 — Final Challenge: Stability Report Card

## Objective

Run a controlled ablation study over **6 configurations** and produce a "Stability Report Card."

## Configurations to Test

| # | Init Scheme | Activation | Optimizer | Scheduler | Weight Decay |
|---|-------------|-----------|-----------|-----------|-------------|
| 1 | kaiming_normal | relu | AdamW | cosine | 0.01 |
| 2 | xavier_normal | relu | AdamW | cosine | 0.01 |
| 3 | kaiming_normal | gelu | AdamW | warmup_cosine | 0.01 |
| 4 | kaiming_normal | relu | SGD+momentum | step | 0.0 |
| 5 | constant_small | relu | AdamW | cosine | 0.0 |
| 6 | kaiming_normal | leaky_relu | AdamW | cosine | 0.1 |

## For Each Configuration, Report

| Metric | How |
|--------|-----|
| Best val accuracy | `max(val_accs)` |
| Final val accuracy | `val_accs[-1]` |
| Time per epoch | `time.time()` around epoch |
| Anomalies detected | `detect_anomalies()` |
| Avg gradient norm | `grad_norms()` over epochs |
| Dead neuron % | `activation_stats()` after last epoch |

## Output

A single summary table + overlay plot of all train/val curves.
No solution provided — only the requirements above."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  FINAL CHALLENGE: Stability Report Card                ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Define the 6 configs as dicts                      ║
# ║  2. For each config:                                    ║
# ║     a. Run training (use run_training or manual loop)  ║
# ║     b. After training, register hooks and compute      ║
# ║        activation_stats on one batch                    ║
# ║     c. Run detect_anomalies on the last step           ║
# ║     d. Collect all metrics                              ║
# ║  3. Print a formatted report card table                ║
# ║  4. Plot all curves with plot_multi_curves             ║
# ║  5. Write your reflection below                        ║
# ╚═══════════════════════════════════════════════════════╝

ABLATION_EPOCHS = 10
configs = {
    "C1_kaiming_relu_adamw_cos": dict(
        init_scheme="kaiming_normal", act_name="relu",
        optimizer_name="adamw", scheduler_name="cosine",
        weight_decay=0.01, lr=1e-3, epochs=ABLATION_EPOCHS,
        seed=42, deterministic=True, batch_size=64,
    ),
    # TODO: Define C2 through C6 (see table above)
}

all_results = {}
for label, cfg in configs.items():
    print(f"\\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    # TODO: all_results[label] = run_training(cfg)

# TODO: Build report card table
# TODO: plot_multi_curves(all_results)"""))

cells.append(md("""\
## 8.2 Reflection (Write Your Answers)

1. **Which configuration worked best and why?**
   Consider: init + activation compatibility, optimizer choice, regularization.

2. **What failure mode was most instructive?**
   Which config failed most dramatically, and what did the diagnostics reveal?

3. **What tools will you carry forward?**
   Which functions from this lab will be most useful in subsequent groups?

4. **When would you NOT use deterministic mode?**
   What's the tradeoff?"""))

cells.append(code("# TODO: Write your reflection answers here as comments or markdown\n"))

# ═══════════════════════════════════════════════════════
# SUMMARY CHECKLIST
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# ✅ Summary Checklist

| # | Skill | Confident? |
|---|-------|-----------|
| 1 | I can choose initialization based on activation function | ☐ |
| 2 | I can detect dead ReLUs using activation hooks | ☐ |
| 3 | I can detect vanishing/exploding gradients using grad norms | ☐ |
| 4 | I understand weight decay vs L2 and why AdamW matters | ☐ |
| 5 | I can implement a warmup + cosine LR scheduler | ☐ |
| 6 | I can set up fully reproducible training | ☐ |
| 7 | I know when `model.eval()` vs `model.train()` matters | ☐ |
| 8 | I can use the overfit-one-batch test for sanity checking | ☐ |
| 9 | I built a reusable Stability & Reproducibility Toolkit | ☐ |

### 🔧 Toolkit Summary

```python
# Your reusable toolkit from this lab:
from init_and_activations import make_activation, init_weights, SmallCNN
from optim_and_schedules import build_optimizer, build_scheduler, WarmupThenCosineScheduler
from stability_tools import grad_norms, activation_stats, register_activation_hooks, detect_anomalies
from reproducibility import seed_everything, make_dataloader
```

---

### 🔜 Next: Group 3 — CNNs & Vision Systems

Convolution math, custom layers, segmentation (U-Net), object detection metrics (IoU, NMS, mAP), YOLOv1 core logic."""))

# ═══════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "notebooks", "02_training_dynamics_stability.ipynb"
)
out_path = os.path.normpath(out_path)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook generated: {out_path}")
print(f"Total cells: {len(cells)}")
md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Markdown: {md_count}, Code: {code_count}")
