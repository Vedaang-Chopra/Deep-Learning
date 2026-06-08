#!/usr/bin/env python3
"""Generate the Group 5 Efficient Training Lab notebook."""
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
# ⚡ Notebook 5 — Efficient Training Lab
## AMP, Accumulation, Checkpointing, Profiling, torch.compile

**Group 5 — Systems-Aware Training**

---

### 🎯 Learning Objectives

1. Measure training performance rigorously (warmup, CUDA sync, percentiles)
2. Integrate mixed precision (AMP) with GradScaler correctly
3. Implement gradient accumulation for effective batch scaling
4. Apply activation checkpointing to trade compute for memory
5. Debug GPU memory: allocated vs reserved, fragmentation, OOM
6. Profile training with torch.profiler and NVTX ranges
7. Use torch.compile and diagnose graph breaks
8. Produce a consolidated efficiency report card

### 📂 File Structure

```
Group_5_Efficient_Training/
├── notebooks/
│   └── 05_efficient_training_systems.ipynb   ← you are here
└── src/
    ├── perf_harness.py          ← timing + throughput measurement
    ├── memory_tools.py          ← VRAM snapshots + OOM testing
    ├── amp_and_accum.py         ← AMP train step + gradient accumulation
    ├── checkpointing.py         ← activation checkpointing
    ├── compile_tools.py         ← torch.compile + graph break debugging
    └── profiling_tools.py       ← torch.profiler + NVTX ranges
```

> ⚠️ **Measurement discipline**: Every optimization claim must be backed by numbers. No "it feels faster" — only wall-clock time, peak VRAM, and throughput."""))

# ═══════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════
cells.append(md("## 0 — Environment Setup"))

cells.append(code("""\
import sys, os, time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

sys.path.insert(0, os.path.join(os.pardir, "src"))

from perf_harness import measure_step_time, throughput, StepTimer, print_step_stats
from memory_tools import memory_snapshot, reset_memory_stats, try_run_with_batch_sizes, MemoryTimeline, plot_memory_comparison
from amp_and_accum import train_step_fp32, train_step_amp, train_step_amp_accum, GradAccumulator
from checkpointing import apply_checkpointing, CheckpointedSequential
from compile_tools import compile_model, detect_graph_breaks, benchmark_compile, check_for_sync_traps
from profiling_tools import annotated_train_step, profile_n_steps, summarize_profile, quick_profile

print(f"PyTorch version : {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device          : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU             : {torch.cuda.get_device_name()}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")"""))

cells.append(code("""\
# ── Model + Dataset: ResNet-18 on CIFAR-10 ──
# This is our benchmark workload throughout the notebook.

transform = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

def make_model():
    model = torchvision.models.resnet18(num_classes=10)
    # Adapt for 32x32 CIFAR (ResNet expects 224x224)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
print(f"Dataset: {len(train_set)} images, batch_size=128")
print(f"Model:   ResNet-18 adapted for CIFAR-10")"""))

# ═══════════════════════════════════════════════════════
# SECTION 1 — BASELINE MEASUREMENT
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 1 — Baseline Measurement Discipline

## 1.1 Why Measurement Matters

Systems optimization without measurement is guesswork. Common traps:

| Trap | What Happens | Fix |
|------|-------------|-----|
| **No warmup** | First iterations include JIT/CUDA init | Skip 10+ warmup steps |
| **No CUDA sync** | GPU ops are async → timer only measures launch | `torch.cuda.synchronize()` |
| **Using `.item()` in loop** | Forces CPU-GPU sync every iteration | Accumulate on GPU, sync once |
| **Measuring wall clock only** | Hides data loading bottlenecks | Profile each phase separately |
| **Single measurement** | High variance from system noise | Report p50/p95/mean |

### CUDA Synchronization

```python
# ❌ WRONG: measures only kernel launch (microseconds)
t0 = time.time()
output = model(input)
t1 = time.time()  # ← GPU is still computing!

# ✅ RIGHT: measures actual computation
torch.cuda.synchronize()
t0 = time.time()
output = model(input)
torch.cuda.synchronize()  # ← wait for GPU to finish
t1 = time.time()
```"""))

cells.append(md("""\
## 1.2 Implementation Tasks

Open `src/perf_harness.py` and implement:

1. **`measure_step_time()`** — warmup + sync + statistics
2. **`throughput()`** — images/sec or tokens/sec"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  BASELINE: Measure fp32 training step                  ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create model + optimizer                           ║
# ║  2. Get one batch from train_loader                    ║
# ║  3. Create a step function (no arguments)              ║
# ║  4. Use measure_step_time() to get timing stats        ║
# ║  5. Compute throughput (images/sec)                    ║
# ║  6. Get memory snapshot                                 ║
# ╚═══════════════════════════════════════════════════════╝

model = make_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch = next(iter(train_loader))
batch = (batch[0].to(DEVICE), batch[1].to(DEVICE))

def baseline_step():
    return train_step_fp32(model, batch, optimizer, loss_fn)

# TODO: Measure and report
# stats = measure_step_time(baseline_step, warmup=10, iters=50, device=DEVICE)
# print_step_stats(stats, label="FP32 Baseline")
# tp = throughput("images", batch_size=128, step_time_ms=stats['mean_ms'])
# print(f"Throughput: {tp:.0f} images/sec")
# snap = memory_snapshot(DEVICE)
# print(f"Memory: allocated={snap['allocated_mb']:.1f}MB, reserved={snap['reserved_mb']:.1f}MB")
"""))

# ═══════════════════════════════════════════════════════
# SECTION 2 — AMP
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 2 — Mixed Precision Training (AMP)

## 2.1 Conceptual Background

### What AMP Does

Mixed precision uses **fp16** for most operations and **fp32** for sensitive ones:

```
fp32: loss computation, softmax, layer norm, weight updates
fp16: matmuls, convolutions (the bulk of compute)
```

On modern GPUs (V100+), fp16 matmuls run **2-8x faster** via Tensor Cores.

### GradScaler: Why It's Needed

fp16 has limited dynamic range (max ~65504, min ~6e-8). Small gradients underflow to zero.

**GradScaler** multiplies the loss by a large factor before `.backward()`:
- Gradients stay in representable range
- Before optimizer step, scaler divides gradients back
- If any gradient is Inf/NaN, scaler reduces the scale factor and **skips the step**

```python
scaler = GradScaler()

# Forward in fp16
with autocast():
    loss = model(input)

# Backward with scaled loss
scaler.scale(loss).backward()

# Unscale → clip → step
scaler.unscale_(optimizer)          # gradients back to true scale
clip_grad_norm_(params, max_norm)   # clip AFTER unscale!
scaler.step(optimizer)              # skips if Inf/NaN
scaler.update()                     # adjust scale factor
```

### Common Pitfalls

| Mistake | Consequence |
|---------|------------|
| Clipping before `unscale_()` | Clips at wrong magnitude |
| Calling `optimizer.step()` instead of `scaler.step()` | No Inf/NaN checking |
| Using `loss.item()` inside autocast | Forces sync, breaks async |
| BatchNorm in fp16 | Can be unstable — PyTorch auto-handles this |"""))

cells.append(md("""\
## 2.2 Implementation Task

Open `src/amp_and_accum.py` and implement **`train_step_amp()`**."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: FP32 vs AMP                               ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create GradScaler                                  ║
# ║  2. Build AMP step function using train_step_amp        ║
# ║  3. Measure step times for both fp32 and AMP           ║
# ║  4. Compare throughput and memory                       ║
# ║  5. Plot comparison bar charts                          ║
# ╚═══════════════════════════════════════════════════════╝

from torch.cuda.amp import GradScaler

model_amp = make_model()
optimizer_amp = torch.optim.AdamW(model_amp.parameters(), lr=1e-3)
scaler = GradScaler()

def amp_step():
    return train_step_amp(model_amp, batch, optimizer_amp, scaler, loss_fn)

# TODO: Measure and compare with baseline
"""))

# ═══════════════════════════════════════════════════════
# SECTION 3 — GRADIENT ACCUMULATION
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 3 — Gradient Accumulation

## 3.1 Conceptual Background

### Effective Batch Size

When GPU memory limits your batch size, gradient accumulation lets you simulate larger batches:

```
effective_batch_size = micro_batch_size × accum_steps
```

### How It Works

```
Step 1: Forward(micro_batch_1) → loss/accum_steps → backward (DON'T step)
Step 2: Forward(micro_batch_2) → loss/accum_steps → backward (DON'T step)
Step 3: Forward(micro_batch_3) → loss/accum_steps → backward (DON'T step)
Step 4: Forward(micro_batch_4) → loss/accum_steps → backward → STEP optimizer
```

Gradients **accumulate** (add up) across micro-batches.

### Critical: Loss Scaling

**You MUST divide loss by `accum_steps`**, otherwise:
- Accumulated gradient = accum_steps × single-batch gradient
- Effective learning rate is accum_steps× too large
- Training diverges

### BatchNorm Caveat

BatchNorm statistics are computed **per micro-batch**, not per effective batch.
With small micro-batches, BN stats have high variance → unstable training.
**Fix**: Use GroupNorm or LayerNorm when using accumulation with small micro-batches."""))

cells.append(md("## 3.2 Implementation Task\n\nOpen `src/amp_and_accum.py` and implement **`train_step_amp_accum()`** and **`GradAccumulator`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Gradient Accumulation                     ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Keep effective batch size = 128:                   ║
# ║     Config A: batch=128, accum=1 (baseline)            ║
# ║     Config B: batch=32, accum=4                        ║
# ║     Config C: batch=16, accum=8                        ║
# ║  2. Measure for each:                                  ║
# ║     - Step time (per optimizer step)                   ║
# ║     - Throughput                                       ║
# ║     - Peak memory                                      ║
# ║  3. Plot comparison                                    ║
# ║                                                       ║
# ║  Question: Which config uses least memory?              ║
# ║  Question: Which has highest throughput?                 ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Implement and compare
"""))

# ═══════════════════════════════════════════════════════
# SECTION 4 — ACTIVATION CHECKPOINTING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 4 — Activation Checkpointing

## 4.1 Conceptual Background

### The Memory Problem

During forward pass, PyTorch stores **all intermediate activations** for backward:

```
Layer 1 → save act_1 → Layer 2 → save act_2 → ... → Layer N → save act_N
```

For a model with N layers and activation size A:
- Memory ≈ N × A (can be gigabytes for large models)

### Checkpointing: Trade Compute for Memory

Instead of saving all activations, **recompute them** during backward:

```
Forward:  Only save checkpointed layer inputs (e.g., every 2nd layer)
Backward: Recompute activations from saved checkpoints on-the-fly
```

Memory: **√N × A** (if checkpointing every √N layers)
Extra compute: **~33%** overhead (one extra forward per checkpointed segment)

### Selective Checkpointing

Not all layers benefit equally. Checkpoint large layers (attention, FFN) but not small ones (LayerNorm, activations).

```python
from torch.utils.checkpoint import checkpoint

# Wrap specific layers
x = checkpoint(self.attention_block, x, use_reentrant=False)
```"""))

cells.append(md("## 4.2 Implementation Task\n\nOpen `src/checkpointing.py` and implement **`apply_checkpointing()`** and **`CheckpointedSequential`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Checkpoint vs No Checkpoint               ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Train without checkpointing → measure peak VRAM    ║
# ║  2. Apply checkpointing → measure peak VRAM            ║
# ║  3. Compare step time (expect ~30% slower with ckpt)   ║
# ║  4. Plot: memory vs time trade-off                     ║
# ║  5. Try selective checkpointing (every other block)    ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Implement and compare
"""))

# ═══════════════════════════════════════════════════════
# SECTION 5 — MEMORY DEBUGGING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 5 — Memory Debugging: OOM & Fragmentation

## 5.1 Conceptual Background

### Allocated vs Reserved

```
┌────────────────────────── GPU VRAM ──────────────────────────┐
│                                                               │
│   ┌─── Reserved by PyTorch Caching Allocator ───┐             │
│   │                                              │             │
│   │   ┌── Actually Allocated by Tensors ──┐     │             │
│   │   │  model weights, activations,      │     │             │
│   │   │  gradients, optimizer states       │     │  Free GPU   │
│   │   └──────────────────────────────────┘     │  memory      │
│   │                                              │             │
│   │   "Free" pool: available for reuse           │             │
│   │   WITHOUT calling cudaMalloc                  │             │
│   └──────────────────────────────────────────────┘             │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

- **Allocated**: memory actually holding tensors
- **Reserved**: memory grabbed from CUDA (includes freed-but-cached blocks)
- **Fragmentation**: reserved >> allocated → blocks too small to reuse

### Why `torch.cuda.empty_cache()` Doesn't Fix OOM

`empty_cache()` returns cached blocks to CUDA. But:
- The blocks are fragmented (many small blocks, not one big one)
- CUDA can't coalesce them efficiently
- The real fix: reduce peak allocation, not clean up after

### Common Memory Hogs

| Consumer | Typical Size | Fix |
|----------|-------------|-----|
| Activations | 40-60% of total | Checkpointing |
| Optimizer states (Adam) | 2× model params | SGD, or lower precision |
| Gradients | 1× model params | Gradient accumulation (smaller micro-batch) |
| Model params | Baseline | Quantization, pruning |"""))

cells.append(md("## 5.2 Implementation Tasks\n\nOpen `src/memory_tools.py` and implement **`memory_snapshot()`**, **`reset_memory_stats()`**, **`try_run_with_batch_sizes()`**, and **`MemoryTimeline`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Find Max Batch Size per Configuration     ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Define a make_step_fn(batch_size) factory that     ║
# ║     creates a training step for a given batch size     ║
# ║  2. Test batch sizes: [64, 128, 256, 512, 1024, 2048] ║
# ║  3. Run for each config:                               ║
# ║     A: fp32 baseline                                   ║
# ║     B: AMP                                             ║
# ║     C: AMP + checkpointing                             ║
# ║     D: AMP + checkpointing + accum (micro=64)         ║
# ║  4. Build deliverable table:                           ║
# ║     Config | Max Batch | Peak VRAM | Notes             ║
# ╚═══════════════════════════════════════════════════════╝

batch_sizes_to_test = [64, 128, 256, 512, 1024, 2048]

# TODO: Define factory, test configs, produce table
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Memory Timeline During Training           ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create MemoryTimeline()                            ║
# ║  2. Train for 20 steps, recording memory at each step  ║
# ║  3. Plot the timeline (allocated + reserved over steps)║
# ║  4. Identify: when does peak happen? After which phase?║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Record and plot memory timeline
"""))

# ═══════════════════════════════════════════════════════
# SECTION 6 — PROFILING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 6 — Profiling with torch.profiler

## 6.1 Conceptual Background

### Why Profile?

Intuition about bottlenecks is usually wrong. Profiling reveals:
- Is the bottleneck data loading, forward, backward, or optimizer?
- Are there unnecessary CPU-GPU synchronizations?
- Is the GPU actually busy or waiting?

### torch.profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(10):
        train_step()
        prof.step()

# View results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### NVTX Ranges

Label your code phases and they show up in the profiler:

```python
with record_function("forward"):
    output = model(input)

with record_function("backward"):
    loss.backward()
```

### Common Findings

| Finding | Meaning | Fix |
|---------|---------|-----|
| CPU time >> CUDA time | Data loading bottleneck | More workers, pin_memory |
| Many small CUDA kernels | Kernel launch overhead | torch.compile, fused ops |
| Large gaps between kernels | CPU-GPU sync stalls | Remove .item(), .cpu() |
| Backward takes 3× forward | Normal (gradient computation) | Checkpointing if memory-bound |"""))

cells.append(md("## 6.2 Implementation Tasks\n\nOpen `src/profiling_tools.py` and implement **`annotated_train_step()`**, **`profile_n_steps()`**, and **`summarize_profile()`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Profile Baseline vs Optimized             ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create annotated_train_step with NVTX ranges      ║
# ║  2. Profile 20 steps of baseline (fp32)                ║
# ║  3. Print summarize_profile() output                   ║
# ║  4. Profile 20 steps of optimized (AMP + compile)      ║
# ║  5. Compare: which phase improved most?                ║
# ║  6. Identify top 2 bottlenecks and describe fixes      ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Profile and analyze
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Detect CUDA Sync Traps                    ║
# ║                                                       ║
# ║  Use check_for_sync_traps on your training loop code   ║
# ║  to find hidden synchronization points.                ║
# ╚═══════════════════════════════════════════════════════╝

# Example: scan a training loop for sync traps
bad_loop = '''
for batch in loader:
    output = model(batch)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")  # ← SYNC TRAP!
    if loss.item() < 0.1:          # ← SYNC TRAP!
        break
'''

traps = check_for_sync_traps(bad_loop)
print("CUDA Sync Traps Found:")
for line, pattern, text in traps:
    print(f"  Line {line}: '{pattern}' in: {text}")"""))

# ═══════════════════════════════════════════════════════
# SECTION 7 — torch.compile
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 7 — torch.compile Deep Dive

## 7.1 Conceptual Background

### What torch.compile Does

```
Python → Dynamo (captures graph) → IR → Inductor (backend) → Optimized kernels
```

1. **TorchDynamo**: intercepts Python bytecode, captures the computation graph
2. **TorchInductor**: compiles the graph to optimized CUDA/CPU kernels
3. Benefits: kernel fusion, reduced overhead, memory optimization

### Compile Modes

| Mode | First-Run Cost | Steady-State Speed | Best For |
|------|---------------|-------------------|----------|
| `"default"` | Moderate | Fast | General use |
| `"reduce-overhead"` | High (CUDA graphs) | Fastest | Inference, fixed shapes |
| `"max-autotune"` | Very high | Fastest (tuned) | Production deployment |

### Graph Breaks

A **graph break** occurs when Dynamo can't trace through your code:

```python
# ❌ Causes graph break:
if x.sum() > 0:     # Python-side control flow on tensor
    x = x * 2
print(x.shape)       # print() is not traceable
val = x.item()       # forces CPU sync

# ✅ No graph break:
x = torch.where(x.sum() > 0, x * 2, x)  # tensor-side control flow
```

Each graph break adds overhead: separate kernel launches, no cross-break fusion.

### Guards

Dynamo records **guards** — assumptions about inputs (shapes, dtypes, device).
If a guard is violated (e.g., different input shape), Dynamo recompiles.
Too many recompilations = worse than eager mode."""))

cells.append(md("## 7.2 Implementation Tasks\n\nOpen `src/compile_tools.py` and implement **`compile_model()`**, **`detect_graph_breaks()`**, and **`benchmark_compile()`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Eager vs Compiled                         ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Check for graph breaks in your model               ║
# ║  2. Compile model with mode='default'                  ║
# ║  3. Measure first-step time (compilation overhead)     ║
# ║  4. Measure steady-state step time                     ║
# ║  5. Compare: eager vs compiled vs compiled+AMP         ║
# ║  6. Try 'reduce-overhead' and 'max-autotune' modes    ║
# ╚═══════════════════════════════════════════════════════╝

model_for_compile = make_model()

# Step 1: Check for graph breaks
# result = detect_graph_breaks(model_for_compile, batch[0])
# print(f"Graph breaks: {result['num_breaks']}")
# if result['has_breaks']:
#     for reason in result['break_reasons']:
#         print(f"  Break: {reason}")

# Step 2: Compile and benchmark
# TODO: Compare modes
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Fix a Graph Break                         ║
# ║                                                       ║
# ║  Here's a model with intentional graph breaks.         ║
# ║  TODO:                                                 ║
# ║  1. Run detect_graph_breaks on BrokenModel             ║
# ║  2. Identify what causes the breaks                    ║
# ║  3. Create FixedModel that avoids the breaks           ║
# ║  4. Verify: no graph breaks in FixedModel              ║
# ║  5. Benchmark: BrokenModel vs FixedModel (compiled)    ║
# ╚═══════════════════════════════════════════════════════╝

class BrokenModel(nn.Module):
    \"\"\"Model with intentional graph breaks for debugging practice.\"\"\"
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # ❌ Graph break: Python-side control flow on tensor value
        if x.mean().item() > 0:
            x = x * 1.1

        x = F.relu(self.fc2(x))

        # ❌ Graph break: print inside forward
        print(f"Shape after fc2: {x.shape}")

        x = self.fc3(x)
        return x

# TODO: Fix the model and compare compiled performance
"""))

# ═══════════════════════════════════════════════════════
# SECTION 8 — FINAL CHALLENGE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# 🧪 Final Challenge — Efficiency Report Card

## Objective

Produce a comprehensive efficiency comparison across all techniques.

## Required Report Card

| Configuration | Peak VRAM (MB) | Step Time p50 (ms) | Step Time p95 (ms) | Throughput (img/s) | Stability | Notes |
|--------------|---------------|-------------------|-------------------|-------------------|-----------|-------|
| fp32 eager | | | | | | baseline |
| AMP eager | | | | | | |
| AMP + accum (4×) | | | | | | |
| AMP + checkpoint | | | | | | |
| AMP + ckpt + accum | | | | | | |
| compile(fp32) | | | | | | |
| compile(AMP) | | | | | | |
| **best overall** | | | | | | |

## Requirements

1. Fill in ALL cells with actual measurements
2. "Stability" column: any NaN losses? Divergence?
3. "Notes" column: graph breaks? compile overhead? anomalies?
4. Write a short conclusion (3-5 sentences):
   - Which technique helped most?
   - What trade-offs were introduced?
   - What would change on bigger hardware (A100 vs your GPU)?"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  FINAL CHALLENGE: Build the Report Card                ║
# ║                                                       ║
# ║  Run all configurations and collect measurements.      ║
# ║  Output the table programmatically.                    ║
# ╚═══════════════════════════════════════════════════════╝

import pandas as pd

# Template for results
configs = [
    "fp32_eager", "amp_eager", "amp_accum_4x",
    "amp_checkpoint", "amp_ckpt_accum",
    "compile_fp32", "compile_amp", "best_overall",
]
columns = ["peak_vram_mb", "step_p50_ms", "step_p95_ms", "throughput_img_s", "stability", "notes"]

# TODO: Fill in with actual measurements
# results = pd.DataFrame(index=configs, columns=columns)
# results.loc["fp32_eager"] = [peak, p50, p95, tp, "ok", "baseline"]
# ...
# display(results)
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  VISUALIZATION: Comparison Plots                       ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Bar chart: Peak VRAM per configuration             ║
# ║  2. Bar chart: Throughput per configuration             ║
# ║  3. Scatter plot: Memory vs Throughput trade-off        ║
# ║     (each config is a point)                           ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Create comparison visualizations
"""))

# ═══════════════════════════════════════════════════════
# SUMMARY CHECKLIST
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# ✅ Summary Checklist

| # | Skill | Confident? |
|---|-------|-----------|
| 1 | I can measure training step time correctly (warmup, CUDA sync) | ☐ |
| 2 | I understand AMP: autocast regions, GradScaler, fp32-sensitive ops | ☐ |
| 3 | I can implement gradient accumulation with correct loss scaling | ☐ |
| 4 | I understand activation checkpointing trade-offs and can apply it | ☐ |
| 5 | I can debug GPU memory issues (allocated vs reserved, fragmentation) | ☐ |
| 6 | I can use torch.profiler and NVTX ranges to find bottlenecks | ☐ |
| 7 | I can use torch.compile and diagnose graph breaks | ☐ |
| 8 | I produced a consolidated efficiency report card with measurements | ☐ |

### Common Pitfalls Checklist

```
✗ Timing GPU code without torch.cuda.synchronize()
✗ Forgetting to divide loss by accum_steps
✗ Gradient clipping BEFORE scaler.unscale_()
✗ Using .item() inside training loop (sync trap)
✗ Calling optimizer.step() instead of scaler.step()
✗ Assuming torch.cuda.empty_cache() fixes OOM
✗ Not warming up before benchmarking
✗ Using fullgraph=True without fixing graph breaks first
```

---

### 🔜 Next: Group 6 — Generative Models (GANs, VAEs, Diffusion)

Build DCGAN, VAE, and simple diffusion model from scratch."""))

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
    os.pardir, "notebooks", "05_efficient_training_systems.ipynb"
))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook: {out_path}")
print(f"Cells: {len(cells)} (markdown: {md_count}, code: {code_count})")
