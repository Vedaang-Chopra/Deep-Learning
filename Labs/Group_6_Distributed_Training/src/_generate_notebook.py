#!/usr/bin/env python3
"""Generate the Group 6 Distributed Training Lab notebook."""
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
# 🌐 Notebook 6 — Distributed Training Lab
## DDP, FSDP, Scaling, and Debugging Failures

**Group 6 — Distributed Training Systems**

---

### 🎯 Learning Objectives

1. Initialize and tear down distributed process groups correctly
2. Build a DDP training loop with DistributedSampler and epoch seeding
3. Aggregate metrics correctly across ranks (not naïve averaging)
4. Measure scaling efficiency: throughput vs GPUs
5. Wrap models with FSDP and compare memory scaling vs DDP
6. Save and load checkpoints correctly in distributed context
7. Debug distributed failures: hangs, timeouts, deadlocks, NCCL errors
8. Produce a consolidated distributed training report card

### 📂 File Structure

```
Group_6_Distributed_Training/
├── notebooks/
│   └── 06_distributed_training_lab.ipynb   ← you are here
└── src/
    ├── dist_setup.py            ← process group init/cleanup
    ├── ddp_train.py             ← DDP DataLoader, wrapping, training loop
    ├── fsdp_train.py            ← FSDP wrapping, checkpointing
    ├── dist_metrics.py          ← all-reduce, global accuracy, scaling
    └── dist_debug.py            ← distributed_assert, barrier, failure tools
```

> ⚠️ **No shortcuts**: No Lightning, no HuggingFace Trainer. You implement the distributed training loop yourself.

> 📌 **Single-GPU mode**: If only 1 GPU is available, the notebook runs in single-process mode and skips multi-GPU sections gracefully."""))

# ═══════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════
cells.append(md("## 0 — Environment Setup"))

cells.append(code("""\
import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.pardir, "src"))

from dist_setup import (
    setup_process_group, cleanup_process_group,
    get_local_rank, get_rank, get_world_size,
    is_main_process, is_distributed, print_launch_instructions,
)
from ddp_train import (
    build_dataloader_ddp, build_dataloader_single,
    wrap_model_ddp, train_one_epoch_ddp, evaluate_ddp,
)
from fsdp_train import (
    wrap_model_fsdp, train_one_epoch_fsdp,
    save_checkpoint_rank0, save_checkpoint_sharded_fsdp, load_checkpoint,
)
from dist_metrics import (
    all_reduce_mean, all_reduce_sum,
    compute_global_accuracy, measure_scaling_efficiency, plot_scaling,
)
from dist_debug import (
    distributed_assert, sync_barrier, set_nccl_debug_env,
    print_hang_checklist,
    simulate_rank_skip_collective, simulate_oom_one_rank,
)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
NUM_GPUS = torch.cuda.device_count()
print(f"GPUs available  : {NUM_GPUS}")
for i in range(NUM_GPUS):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Determine mode
if NUM_GPUS >= 2:
    print("\\n✅ Multi-GPU available — full distributed exercises enabled")
else:
    print("\\n⚠️ Single GPU — running in single-process mode. Multi-GPU exercises are conceptual.")"""))

cells.append(code("""\
# ── Model + Dataset ──

transform = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = T.Compose([
    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_set = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

def make_model():
    \"\"\"ResNet-18 adapted for CIFAR-10 (32x32 images).\"\"\"
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

loss_fn = nn.CrossEntropyLoss()
print(f"Dataset: {len(train_set)} train, {len(test_set)} test")
print(f"Model:   ResNet-18 adapted for CIFAR-10")"""))

cells.append(code("""\
# ── Launch instructions ──
print_launch_instructions()"""))

# ═══════════════════════════════════════════════════════
# SECTION 1 — DISTRIBUTED BASICS
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 1 — Distributed Training Basics

## 1.1 Key Concepts

### Process-Per-GPU Model

In PyTorch distributed training, each GPU runs its own **process**:

```
Process 0 (GPU 0)  ─┐
Process 1 (GPU 1)  ─┤── world_size = 4
Process 2 (GPU 2)  ─┤
Process 3 (GPU 3)  ─┘
```

### Terminology

| Term | Meaning |
|------|---------|
| **world_size** | Total number of processes (= total GPUs) |
| **rank** | Global process ID (0 to world_size-1) |
| **local_rank** | GPU index on the current machine (0 to gpus_per_node-1) |
| **NCCL** | NVIDIA's communication library (fast GPU↔GPU comms) |

### How DDP Works

1. Each process has a **full copy** of the model
2. Each process gets a **different subset** of the data (via DistributedSampler)
3. Forward pass: each process computes independently on its data shard
4. Backward pass: DDP hooks perform **all-reduce** to average gradients across processes
5. Optimizer step: identical gradients → identical weight updates → models stay in sync

```
                    ┌─────────────────────────────┐
                    │     All-Reduce (NCCL)        │
                    │  Averages gradients across   │
                    │  all ranks during backward   │
                    └──────────┬──────────────────┘
                               │
           ┌───────────┬───────┴───┬───────────┐
       GPU 0       GPU 1       GPU 2       GPU 3
     Shard A     Shard B     Shard C     Shard D
       ↓           ↓           ↓           ↓
    Forward     Forward     Forward     Forward
       ↓           ↓           ↓           ↓
    Backward    Backward    Backward    Backward
     + allreduce gradients ──────────────────→
       ↓           ↓           ↓           ↓
    opt.step    opt.step    opt.step    opt.step
     (identical updates → models stay in sync)
```

### Why DDP Is Fast

- Gradient all-reduce **overlaps** with backward computation
- NCCL uses Ring or Tree all-reduce algorithms
- Near-linear scaling up to ~16 GPUs on modern hardware

### When DDP Breaks

| Problem | Cause | Fix |
|---------|-------|-----|
| Hang | One rank skips a collective | `distributed_assert()` |
| Wrong metrics | Naïve averaging across ranks | Weighted all-reduce |
| Same data every epoch | Missing `sampler.set_epoch()` | Always call it |
| Slow training | Data loading bottleneck | More workers, pin_memory |
| OOM on one rank | Uneven batch sizes | `drop_last=True` |"""))

# ═══════════════════════════════════════════════════════
# SECTION 2 — PROCESS GROUP
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 2 — Initialize / Tear Down Process Group

## 2.1 Implementation Tasks

Open `src/dist_setup.py` and implement:

1. **`setup_process_group()`** — init NCCL backend with rank/world_size
2. **`cleanup_process_group()`** — destroy process group
3. **`get_local_rank()`** — read from environment variables"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Process Group Setup                     ║
# ║                                                       ║
# ║  In notebook mode (single-process), we verify the     ║
# ║  functions work correctly for WORLD_SIZE=1.            ║
# ║  Multi-GPU testing requires torchrun.                  ║
# ╚═══════════════════════════════════════════════════════╝

# Single-process mode check
if NUM_GPUS >= 1:
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

print(f"Running in single-process mode on {DEVICE}")
print(f"  get_rank()       = {get_rank()}")
print(f"  get_world_size() = {get_world_size()}")
print(f"  is_distributed() = {is_distributed()}")
print(f"  is_main_process()= {is_main_process()}")"""))

# ═══════════════════════════════════════════════════════
# SECTION 3 — DDP TRAINING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 3 — DDP Training Loop Correctness

## 3.1 Conceptual Background

### DistributedSampler

Each rank gets a **non-overlapping** subset of the dataset:

```
Dataset: [0, 1, 2, 3, 4, 5, 6, 7]  (8 samples)
World size = 2

Rank 0 sees: [0, 2, 4, 6]  (even indices)
Rank 1 sees: [1, 3, 5, 7]  (odd indices)
```

### Why `sampler.set_epoch(epoch)` Matters

The sampler uses `epoch` as part of the random seed to shuffle data:

```python
# Without set_epoch: same seed every epoch → same shard order!
sampler = DistributedSampler(dataset, seed=42)
# Epoch 1: rank 0 sees [0, 4, 2, 6]
# Epoch 2: rank 0 sees [0, 4, 2, 6]  ← SAME ORDER!

# With set_epoch: different shuffle each epoch
sampler.set_epoch(1)  # seed = 42 + 1
# Epoch 1: rank 0 sees [0, 4, 2, 6]
sampler.set_epoch(2)  # seed = 42 + 2
# Epoch 2: rank 0 sees [4, 0, 6, 2]  ← DIFFERENT ORDER ✓
```"""))

cells.append(md("""\
## 3.2 Implementation Tasks

Open `src/ddp_train.py` and implement:

1. **`build_dataloader_ddp()`** — DataLoader with DistributedSampler
2. **`wrap_model_ddp()`** — wrap model with DDP
3. **`train_one_epoch_ddp()`** — correct training loop with `set_epoch()`
4. **`evaluate_ddp()`** — compute local metrics for aggregation"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Single-GPU Training (Baseline)            ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create model + optimizer                           ║
# ║  2. Create DataLoader (non-distributed)                ║
# ║  3. Train for 3 epochs                                 ║
# ║  4. Record: loss per epoch, accuracy, step time        ║
# ║  5. This is your baseline for scaling comparison       ║
# ╚═══════════════════════════════════════════════════════╝

model = make_model().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
loader = build_dataloader_single(train_set, batch_size=128)
test_loader = build_dataloader_single(test_set, batch_size=256)

baseline_losses = []
for epoch in range(3):
    t0 = time.time()
    # TODO: Train one epoch (can reuse train_one_epoch_ddp with sampler=None)
    # result = train_one_epoch_ddp(model, loader, optimizer, loss_fn, DEVICE, epoch)
    # baseline_losses.append(result['avg_loss'])
    elapsed = time.time() - t0
    # print(f"Epoch {epoch}: loss={result['avg_loss']:.4f}, time={elapsed:.1f}s")

# TODO: Evaluate
# eval_result = evaluate_ddp(model, test_loader, loss_fn, DEVICE)
# print(f"Accuracy: {eval_result['local_correct']}/{eval_result['local_total']}")
"""))

cells.append(md("""\
### 3.3 Multi-GPU DDP Training

> **Note**: This section requires `torchrun` to run with multiple GPUs.
> Below is the training script you would run and the launch command.

```bash
# Save as ddp_experiment.py and run with:
torchrun --nproc_per_node=2 ddp_experiment.py
```"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  REFERENCE: DDP Training Script                        ║
# ║                                                       ║
# ║  This is the script you would save and run with        ║
# ║  torchrun for actual multi-GPU training.               ║
# ╚═══════════════════════════════════════════════════════╝

DDP_SCRIPT = '''
#!/usr/bin/env python3
\"\"\"DDP training script — run with: torchrun --nproc_per_node=2 this_script.py\"\"\"
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from dist_setup import setup_process_group, cleanup_process_group, get_local_rank, get_rank, get_world_size, is_main_process
from ddp_train import build_dataloader_ddp, wrap_model_ddp, train_one_epoch_ddp, evaluate_ddp
from dist_metrics import compute_global_accuracy

def main():
    # 1. Setup
    rank, world_size = setup_process_group()
    device = torch.device(f"cuda:{get_local_rank()}")

    # 2. Dataset
    transform = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10("./data", train=True, download=(rank==0), transform=transform)
    test_set = torchvision.datasets.CIFAR10("./data", train=False, download=False, transform=T.Compose([
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]))

    # 3. Model
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    ddp_model = wrap_model_ddp(model, device)

    # 4. DataLoader
    loader, sampler = build_dataloader_ddp(train_set, batch_size=128, rank=rank, world_size=world_size)
    test_loader, _ = build_dataloader_ddp(test_set, batch_size=256, rank=rank, world_size=world_size)

    # 5. Train
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        t0 = time.time()
        result = train_one_epoch_ddp(ddp_model, loader, optimizer, loss_fn, device, epoch, sampler)
        elapsed = time.time() - t0
        if is_main_process():
            print(f"Epoch {epoch}: loss={result['avg_loss']:.4f}, time={elapsed:.1f}s")

    # 6. Evaluate
    eval_result = evaluate_ddp(ddp_model, test_loader, loss_fn, device)
    global_acc = compute_global_accuracy(eval_result['local_correct'], eval_result['local_total'], device)
    if is_main_process():
        print(f"Final accuracy: {global_acc['global_accuracy']:.2%}")

    cleanup_process_group()

if __name__ == "__main__":
    main()
'''

print(DDP_SCRIPT)
print("\\n# To run: torchrun --nproc_per_node=2 ddp_experiment.py")"""))

cells.append(md("""\
### 3.4 Data Integrity Check

When using DDP, verify that:
1. Each rank sees a **different** subset of the data
2. No samples overlap between ranks within an epoch
3. All samples are covered across all ranks"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Data Partitioning Verification            ║
# ║                                                       ║
# ║  TODO (conceptual, run with torchrun):                 ║
# ║  1. On each rank, collect all sample indices seen      ║
# ║  2. Gather indices to rank 0                           ║
# ║  3. Verify:                                            ║
# ║     - No overlap between ranks                         ║
# ║     - Union covers all dataset indices                 ║
# ║     - Each rank sees len(dataset)/world_size samples   ║
# ╚═══════════════════════════════════════════════════════╝

# Demonstrate DistributedSampler behavior in single-process
from torch.utils.data import DistributedSampler

# Simulate 2-GPU partitioning
sampler_0 = DistributedSampler(train_set, num_replicas=2, rank=0, shuffle=True, seed=42)
sampler_1 = DistributedSampler(train_set, num_replicas=2, rank=1, shuffle=True, seed=42)

indices_0 = list(iter(sampler_0))
indices_1 = list(iter(sampler_1))

print(f"Rank 0: {len(indices_0)} samples, first 10: {indices_0[:10]}")
print(f"Rank 1: {len(indices_1)} samples, first 10: {indices_1[:10]}")

# Verify no overlap
overlap = set(indices_0) & set(indices_1)
print(f"\\nOverlap: {len(overlap)} indices (should be 0)")
assert len(overlap) == 0, "Data partitioning has overlaps!"
print("✅ No overlap between ranks")

# Verify coverage
total = set(indices_0) | set(indices_1)
print(f"Coverage: {len(total)} / {len(train_set)} indices")"""))

# ═══════════════════════════════════════════════════════
# SECTION 4 — DISTRIBUTED METRICS
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 4 — Distributed Metric Aggregation

## 4.1 Why Naïve Averaging Is Wrong

Consider 2 GPUs evaluating on different-sized shards:

```
GPU 0: 1000 samples, 800 correct → 80% accuracy
GPU 1:  500 samples, 450 correct → 90% accuracy

Naïve average:   (80% + 90%) / 2 = 85.0%
Correct average: (800 + 450) / (1000 + 500) = 83.3%
```

The naïve approach gives **equal weight** to each GPU, but GPU 1 had fewer samples.
The correct approach uses **all-reduce SUM** on both numerator and denominator.

### All-Reduce Patterns

```
all_reduce(SUM):  each rank ends up with the global sum
all_reduce(AVG):  each rank ends up with the global average
                  (= SUM / world_size)

For accuracy:
  global_correct = all_reduce_sum(local_correct)
  global_total   = all_reduce_sum(local_total)
  accuracy       = global_correct / global_total
```"""))

cells.append(md("## 4.2 Implementation Tasks\n\nOpen `src/dist_metrics.py` and implement **`all_reduce_mean()`**, **`all_reduce_sum()`**, **`compute_global_accuracy()`**, and **`measure_scaling_efficiency()`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Naïve vs Correct Metric Aggregation       ║
# ║                                                       ║
# ║  Simulate the problem in single-process mode.          ║
# ╚═══════════════════════════════════════════════════════╝

# Simulate different shard sizes
gpu0_correct, gpu0_total = 800, 1000
gpu1_correct, gpu1_total = 450,  500

# Naïve average (WRONG)
naive_acc = (gpu0_correct/gpu0_total + gpu1_correct/gpu1_total) / 2
print(f"Naïve average accuracy:   {naive_acc:.2%}")

# Correct weighted aggregate
correct_acc = (gpu0_correct + gpu1_correct) / (gpu0_total + gpu1_total)
print(f"Correct global accuracy:  {correct_acc:.2%}")
print(f"Difference:               {abs(naive_acc - correct_acc)*100:.1f} percentage points")
print(f"\\n⚠️ For loss aggregation, the same principle applies!")"""))

# ═══════════════════════════════════════════════════════
# SECTION 5 — SCALING EFFICIENCY
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 5 — Scaling Efficiency Measurement

## 5.1 Conceptual Background

### Throughput Scaling

| GPUs | Ideal Throughput | Typical Actual | Why Less? |
|------|-----------------|----------------|-----------|
| 1 | 1× (baseline) | 1× | — |
| 2 | 2× | ~1.85× | comm overhead |
| 4 | 4× | ~3.5× | more comm, bus contention |
| 8 | 8× | ~6.5× | PCIe/NVLink saturation |

### Scaling Efficiency

```
speedup(N) = throughput(N GPUs) / throughput(1 GPU)
efficiency(N) = speedup(N) / N × 100%
```

- **90%+ efficiency**: excellent (typically ≤4 GPUs)
- **80-90%**: good (4-8 GPUs)
- **< 70%**: something is wrong (data loading? small model? too much comm?)

### Common Bottlenecks

| Bottleneck | Symptom | Fix |
|-----------|---------|-----|
| Data loading | GPU utilization < 100% | More workers, pin_memory |
| Small model | Comm > compute | Don't distribute |
| Small batch | Under-utilizes Tensor Cores | Increase batch, use accum |
| Slow interconnect | Comm time dominates | NVLink, InfiniBand |"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Scaling Efficiency Measurement            ║
# ║                                                       ║
# ║  TODO (requires torchrun with multiple GPUs):          ║
# ║  1. Measure images/sec with 1 GPU                     ║
# ║  2. Measure images/sec with 2 GPUs (DDP)              ║
# ║  3. Measure images/sec with 4 GPUs (if available)     ║
# ║  4. Compute scaling efficiency for each                ║
# ║  5. Plot with plot_scaling()                           ║
# ╚═══════════════════════════════════════════════════════╝

# Example results (fill in with your measurements):
# scaling_results = {
#     1: 3200,   # images/sec with 1 GPU
#     2: 5900,   # images/sec with 2 GPUs
#     4: 11200,  # images/sec with 4 GPUs
# }
# plot_scaling(scaling_results)

# Compute efficiency:
# for n in [2, 4]:
#     if n in scaling_results:
#         eff = measure_scaling_efficiency(scaling_results[1], scaling_results[n], n)
#         print(f"{n} GPUs: speedup={eff['speedup']:.2f}x, efficiency={eff['efficiency']:.1f}%")
"""))

# ═══════════════════════════════════════════════════════
# SECTION 6 — FSDP
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 6 — FSDP: Fit Bigger Models

## 6.1 Conceptual Background

### DDP vs FSDP Memory

| Resource | DDP | FSDP (FULL_SHARD) |
|----------|-----|-------------------|
| Model params | Full copy per GPU | Sharded across GPUs |
| Gradients | Full per GPU → all-reduce | Sharded |
| Optimizer states | Full per GPU | Sharded |
| Activations | Full per GPU | Full per GPU (same) |
| **Memory per GPU** | ~16× params | ~16/N × params + activations |

For a 1B parameter model in fp32:
- **DDP**: 4 GB params + 4 GB grads + 8 GB Adam states = **16 GB per GPU**
- **FSDP (4 GPUs)**: (4 + 4 + 8) / 4 = **4 GB per GPU** + activations

### How FSDP Works

```
Forward pass:
  all-gather params for current layer → compute → discard params

Backward pass:
  all-gather params → recompute activations → compute gradients
  → reduce-scatter gradients (each rank keeps its shard)
```

### Sharding Strategies

| Strategy | What's Sharded | Comm Cost | Memory Savings |
|----------|---------------|-----------|----------------|
| `FULL_SHARD` | params + grads + opt states | Highest | Maximum |
| `SHARD_GRAD_OP` | grads + opt states only | Medium | Moderate |
| `NO_SHARD` | nothing (= DDP) | Lowest | None |"""))

cells.append(md("## 6.2 Implementation Tasks\n\nOpen `src/fsdp_train.py` and implement **`wrap_model_fsdp()`** and **`train_one_epoch_fsdp()`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: DDP vs FSDP Comparison                    ║
# ║                                                       ║
# ║  TODO (requires torchrun):                             ║
# ║  1. DDP: measure peak VRAM, step time, max batch size  ║
# ║  2. FSDP: same measurements                           ║
# ║  3. Produce comparison table                           ║
# ╚═══════════════════════════════════════════════════════╝

# Expected comparison table:
# | Metric          | DDP       | FSDP      |
# |-----------------|-----------|-----------|
# | Peak VRAM (MB)  |           |           |
# | Step time (ms)  |           |           |
# | Max batch size  |           |           |
# | Memory saved    |   —       |     %     |

print("Complete this experiment with torchrun on multi-GPU.")
print("Fill in the comparison table with your measurements.")"""))

# ═══════════════════════════════════════════════════════
# SECTION 7 — CHECKPOINTING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 7 — Checkpointing in Distributed Context

## 7.1 Conceptual Background

### Why Naïve torch.save on Every Rank Is Wrong

```python
# ❌ WRONG: Every rank saves to the same file → race condition!
torch.save(model.state_dict(), "checkpoint.pt")

# ❌ ALSO WRONG: Different filenames but wastes disk
torch.save(model.state_dict(), f"checkpoint_rank{rank}.pt")
# Each file is a FULL copy → N× disk usage

# ✅ CORRECT for DDP: Only rank 0 saves
if rank == 0:
    torch.save(model.module.state_dict(), "checkpoint.pt")
dist.barrier()  # Other ranks wait for rank 0 to finish
```

### FSDP Checkpointing

FSDP adds complexity because parameters are **sharded**:

| Method | What Happens | Pros | Cons |
|--------|-------------|------|------|
| `FULL_STATE_DICT` | Gather all params to rank 0 → save | Compatible with non-FSDP | Peak memory spike |
| `SHARDED_STATE_DICT` | Each rank saves its shard | Low memory | Must load with same world_size |
| `LOCAL_STATE_DICT` | Each rank saves raw local state | Fastest save | Can't load to different config |"""))

cells.append(md("## 7.2 Implementation Tasks\n\nOpen `src/fsdp_train.py` and implement **`save_checkpoint_rank0()`**, **`save_checkpoint_sharded_fsdp()`**, and **`load_checkpoint()`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Checkpoint Save & Resume                  ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Train for 3 epochs, save checkpoint               ║
# ║  2. Create new model + optimizer                      ║
# ║  3. Load checkpoint and resume for 2 more epochs      ║
# ║  4. Verify:                                           ║
# ║     - Loss continues from where it left off           ║
# ║     - No sudden spike in loss at resume point          ║
# ║  5. Plot full loss curve with resume point marked      ║
# ╚═══════════════════════════════════════════════════════╝

# Single-GPU checkpoint test
model = make_model().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# TODO: Train 3 epochs, save, load, resume 2 more
"""))

# ═══════════════════════════════════════════════════════
# SECTION 8 — DEBUGGING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 8 — Debugging Distributed Failures

## 8.1 The Failure Taxonomy

### Type 1: Hangs (No Error, No Progress)

**Cause**: One rank reaches a collective call that another rank never reaches.

```python
# ❌ BUG: rank 1 never calls all_reduce
if rank == 0:
    dist.all_reduce(tensor)  # Rank 0 blocks waiting for rank 1
    # ← HANG forever
```

**Debugging**:
1. Set `NCCL_TIMEOUT` to a small value (60s)
2. Add `sync_barrier("before_allreduce")` before each collective
3. Use `distributed_assert()` to verify all ranks are healthy

### Type 2: Silent OOM on One Rank

**Cause**: One rank runs out of memory but doesn't crash cleanly.

```
Rank 0: training normally...
Rank 1: CUDA OOM! (process crashes)
Rank 0: HANG (waiting for rank 1 in all-reduce)
```

**Fix**: Use `distributed_assert(not_oom, "OOM detected")` before collectives.

### Type 3: NCCL Errors

Common NCCL errors and their meanings:

| Error | Meaning | Fix |
|-------|---------|-----|
| `NCCL timeout` | One rank unresponsive | Check for OOM, deadlock |
| `unhandled system error` | Network/driver issue | Check NCCL version, GPU driver |
| `invalid rank` | Wrong world_size/rank config | Verify torchrun args |

### Type 4: Gradient Issues with `find_unused_parameters`

If your model has parameters that aren't used in every forward pass,
DDP will **hang** waiting for gradients that never arrive.

```python
# Model with conditional paths:
class ConditionalModel(nn.Module):
    def forward(self, x, use_branch_b=False):
        if use_branch_b:
            return self.branch_b(x)  # branch_a params unused!
        return self.branch_a(x)     # branch_b params unused!

# Fix:
ddp_model = DDP(model, find_unused_parameters=True)
```"""))

cells.append(md("## 8.2 Implementation Tasks\n\nOpen `src/dist_debug.py` and implement **`distributed_assert()`**, **`sync_barrier()`**, and **`set_nccl_debug_env()`**."))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXERCISE: Failure Injection — Understanding Hangs     ║
# ║                                                       ║
# ║  Study these failure scenarios. In single-GPU mode,    ║
# ║  we demonstrate them conceptually.                     ║
# ╚═══════════════════════════════════════════════════════╝

# --- Scenario 1: One rank skips a collective ---
print("=== Scenario 1: Collective Skip ===")
print("What happens when rank 1 skips an all_reduce:")
print()
simulate_rank_skip_collective(rank=0, skip_rank=1)
print()
print("Fix: Use distributed_assert() before collectives to ensure")
print("all ranks are at the same point in execution.")

print("\\n" + "="*60)

# --- Scenario 2: OOM on one rank ---
print("\\n=== Scenario 2: OOM on One Rank ===")
print("When one rank OOMs, it crashes. Other ranks hang waiting.")
print("Fix: Use try_run_with_batch_sizes() to find safe batch size first.")
print("     Use distributed_assert(not_oom) after allocation-heavy ops.")"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXERCISE: Hang Debugging Checklist                    ║
# ╚═══════════════════════════════════════════════════════╝

print_hang_checklist()"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXERCISE: Write Your Debugging Playbook               ║
# ║                                                       ║
# ║  Based on what you've learned, write a short playbook  ║
# ║  for debugging distributed training failures.          ║
# ║                                                       ║
# ║  TODO: Fill in below (as a Python multi-line string):   ║
# ╚═══════════════════════════════════════════════════════╝

MY_DEBUGGING_PLAYBOOK = \"\"\"
# Distributed Training Debugging Playbook

## Step 1: When Training Hangs
- TODO: What to check first
- TODO: Environment variables to set
- TODO: How to narrow down which collective is stuck

## Step 2: When One Rank Crashes
- TODO: How to detect which rank failed
- TODO: How to prevent cascading hangs

## Step 3: When Metrics Look Wrong
- TODO: How to verify metric aggregation
- TODO: Common mistakes

## Step 4: When Training Is Slow
- TODO: How to diagnose scaling bottlenecks
- TODO: What to measure

## Quick Reference
- TODO: Most useful commands and env vars
\"\"\"

print(MY_DEBUGGING_PLAYBOOK)"""))

# ═══════════════════════════════════════════════════════
# FINAL CHALLENGE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# 🧪 Final Challenge — Distributed Training Report Card

## Required Report Card

| Configuration | Images/sec | Time/Epoch (s) | Peak VRAM (MB) | Max Batch | Accuracy (5 epochs) | Notes |
|--------------|-----------|----------------|----------------|-----------|---------------------|-------|
| Single GPU | | | | | | baseline |
| DDP (2 GPU) | | | | | | |
| FSDP (2 GPU) | | | | | | |
| Best config | | | | | | AMP? compile? |

## Requirements

1. Fill in ALL cells with actual measurements (or estimates + reasoning if single-GPU)
2. Write a short conclusion:
   - Where did scaling break down and why?
   - What would you tune next (batch size, comm, dataloader)?
   - When would you prefer DDP vs FSDP?"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  FINAL CHALLENGE: Build the Report Card                ║
# ║                                                       ║
# ║  Collect all measurements and produce the table.       ║
# ║  If single-GPU only, provide best estimates with       ║
# ║  reasoning for what multi-GPU would look like.         ║
# ╚═══════════════════════════════════════════════════════╝

import pandas as pd

configs = ["single_gpu", "ddp_2gpu", "fsdp_2gpu", "best_config"]
columns = ["images_sec", "time_per_epoch_s", "peak_vram_mb", "max_batch", "accuracy_5ep", "notes"]

# TODO: Fill in with measurements
# report = pd.DataFrame(index=configs, columns=columns)
# report.loc["single_gpu"] = [3200, 15.6, 2400, 256, 0.85, "baseline"]
# display(report)
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  CONCLUSION                                            ║
# ║                                                       ║
# ║  TODO: Write your conclusion (3-5 sentences):          ║
# ║  - Where did scaling break down and why?               ║
# ║  - What would you tune next?                           ║
# ║  - When to prefer DDP vs FSDP?                         ║
# ╚═══════════════════════════════════════════════════════╝

CONCLUSION = \"\"\"
TODO: Write your conclusion here.
\"\"\"
print(CONCLUSION)"""))

# ═══════════════════════════════════════════════════════
# SUMMARY CHECKLIST
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# ✅ Summary Checklist

| # | Skill | Confident? |
|---|-------|-----------|
| 1 | I can initialize and tear down a distributed process group | ☐ |
| 2 | I understand rank, local_rank, world_size, and NCCL backend | ☐ |
| 3 | I can build a DDP training loop with DistributedSampler | ☐ |
| 4 | I know why sampler.set_epoch() is needed and what happens without it | ☐ |
| 5 | I can aggregate metrics correctly (not naïve averaging) | ☐ |
| 6 | I can measure scaling efficiency and identify bottlenecks | ☐ |
| 7 | I understand FSDP sharding and when to use it vs DDP | ☐ |
| 8 | I can save and load checkpoints correctly in distributed context | ☐ |
| 9 | I can debug distributed hangs, OOM, and NCCL errors | ☐ |
| 10 | I have a personal debugging playbook for distributed failures | ☐ |

### Common Pitfalls Checklist

```
✗ Forgetting sampler.set_epoch(epoch) → same data order every epoch
✗ Naïvely averaging metrics across ranks instead of weighted all-reduce
✗ Saving checkpoint on all ranks → race condition or N× disk
✗ Using shuffle=True with DistributedSampler → error
✗ Not using drop_last=True → uneven batches → hang
✗ Rank-conditional code around collective calls → deadlock
✗ Missing find_unused_parameters for models with conditional paths
✗ Not setting NCCL_TIMEOUT for debugging → 30-min hangs
```

---

### 🔜 Next: Group 7 — Generative Models (GANs, VAEs, Diffusion)

Build DCGAN, VAE, and a simple diffusion model from scratch."""))

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
    os.pardir, "notebooks", "06_distributed_training_lab.ipynb"
))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook: {out_path}")
print(f"Cells: {len(cells)} (markdown: {md_count}, code: {code_count})")
