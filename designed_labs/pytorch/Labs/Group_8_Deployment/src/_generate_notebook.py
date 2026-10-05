#!/usr/bin/env python3
"""Generate the Group 8 Deployment Lab notebook."""
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
# 🚀 Notebook 8 — Deployment Lab
## Export, Quantize, Benchmark, and Serve PyTorch Models

**Group 8 — Deployment & Production**

---

### 🎯 Learning Objectives

1. Build parity tests that prove correctness across runtimes
2. Export models via TorchScript (trace/script) and torch.export
3. Benchmark with discipline: warmup, CUDA sync, p50/p95/p99 latency
4. Apply PTQ and QAT quantization and measure trade-offs
5. Implement dynamic batching for inference throughput
6. Build a minimal REST inference server with FastAPI
7. Analyze latency vs throughput trade-offs across configurations

### 📂 File Structure

```
Group_8_Deployment/
├── notebooks/
│   └── 08_deployment_inference_serving.ipynb   ← you are here
└── src/
    ├── parity_tests.py          ← correctness testing across runtimes
    ├── export_tools.py          ← TorchScript, torch.export, ONNX
    ├── quantization_tools.py    ← PTQ calibration, QAT, evaluation
    ├── inference_bench.py       ← latency/throughput benchmarking
    ├── batching.py              ← dynamic batching with queueing
    └── serve_app.py             ← FastAPI inference server
```

> ⚠️ **Correctness first**: Before ANY benchmark, prove parity between eager and exported models.

> 📌 **No black boxes**: No HuggingFace serving, no Triton, no vLLM. You build the serving stack."""))

# ═══════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════
cells.append(md("## 0 — Environment Setup"))

cells.append(code("""\
import sys, os, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.pardir, "src"))

from parity_tests import (
    compare_outputs, parity_check_model,
    assert_close_or_report, print_parity_report,
)
from export_tools import (
    export_torchscript_trace, export_torchscript_script,
    export_torch_export, save_artifact, load_artifact,
    print_export_summary,
)
from quantization_tools import (
    prepare_ptq, calibrate, convert_ptq,
    prepare_qat, convert_qat, evaluate,
    model_size_mb, print_quantization_report,
)
from inference_bench import (
    benchmark_fn, benchmark_model, compute_throughput,
    benchmark_batch_sizes,
    plot_latency_histogram, plot_throughput_vs_batch,
)
from batching import Batcher, simulate_load, plot_batching_results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")
print(f"PyTorch: {torch.__version__}")
print(f"Device:  {DEVICE}")"""))

cells.append(code("""\
# ── Model + Dataset ──

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_train = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_set = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

# Calibration subset (small)
calib_indices = torch.randperm(len(train_set))[:1000]
calib_subset = torch.utils.data.Subset(train_set, calib_indices)
calib_loader = torch.utils.data.DataLoader(calib_subset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

def make_model():
    \"\"\"ResNet-18 adapted for CIFAR-10.\"\"\"
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# Create the reference model
model = make_model().to(DEVICE)
model.eval()
example_input = torch.randn(1, 3, 32, 32, device=DEVICE)
print(f"Model: ResNet-18 ({sum(p.numel() for p in model.parameters()):,} params)")
print(f"Model size: {model_size_mb(model):.2f} MB")"""))

# ═══════════════════════════════════════════════════════
# SECTION 1 — PRODUCTION-READY
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 1 — What "Production-Ready" Means

### The Deployment Checklist

| Step | Question | Tool |
|------|----------|------|
| 1. **Correctness** | Does the exported model produce the same outputs? | Parity tests |
| 2. **Export** | Can the model run without Python? | TorchScript / torch.export |
| 3. **Performance** | How fast is inference? Is it fast enough? | Benchmarking |
| 4. **Optimization** | Can we make it faster/smaller? | Quantization, compilation |
| 5. **Serving** | Can clients send requests and get predictions? | REST API |
| 6. **Scaling** | Can we handle high query rates? | Batching, concurrency |

### Latency vs Throughput

```
Latency:    Time for ONE request (ms)
Throughput: Requests processed per second (req/s)

These are in tension:
  - Small batches → low latency, low throughput
  - Large batches → high latency, high throughput
  
The art of deployment: find the sweet spot for your SLA.
```

### Common Deployment Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| No parity test | Silently wrong predictions | compare_outputs() |
| No warmup in benchmarks | First-run JIT overhead included | Warmup iterations |
| No CUDA sync | GPU times are too optimistic | torch.cuda.synchronize() |
| Saving on GPU, loading on CPU | Runtime error | map_location="cpu" |
| Same batch size for all configs | Unfair comparison | Test multiple batch sizes |"""))

# ═══════════════════════════════════════════════════════
# SECTION 2 — PARITY TESTING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 2 — Parity Testing Harness

## 2.1 Why Parity Tests Matter

"Accuracy is the same" is **insufficient**. You need per-output comparison:

```python
# Model A: [0.31, 0.69]  → class 1  ✓
# Model B: [0.49, 0.51]  → class 1  ✓ (same class!)
# But outputs are VERY different → parity FAILS

# This matters because:
# - Downstream systems may use probabilities, not just argmax
# - Small differences compound across layers in larger pipelines
# - Calibration/temperature scaling depends on output magnitudes
```

### Tolerance Values

| Scenario | Typical atol | Why |
|----------|-------------|-----|
| Eager → TorchScript | 1e-6 | Numerically identical |
| Eager → torch.export | 1e-6 | Same ops, different graph |
| FP32 → FP16 | 1e-3 | Reduced precision |
| FP32 → INT8 (PTQ) | 1e-1 | Significant precision loss |"""))

cells.append(md("## 2.2 Implementation Tasks\n\nOpen `src/parity_tests.py` and implement:\n\n1. **`compare_outputs()`** — element-wise comparison with tolerance\n2. **`parity_check_model()`** — compare two models on real data\n3. **`assert_close_or_report()`** — detailed diff report"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Basic Parity Check                        ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Run model twice on same input                      ║
# ║  2. Verify outputs are identical (eager vs eager)      ║
# ║  3. This establishes your baseline parity tolerance    ║
# ╚═══════════════════════════════════════════════════════╝

model.eval()
with torch.no_grad():
    y1 = model(example_input)
    y2 = model(example_input)

# TODO: Check parity
# result = compare_outputs(y1, y2, atol=1e-6, rtol=1e-5)
# print(f"Eager vs Eager: max_diff={result['max_abs_diff']:.2e}, passed={result['passed']}")
"""))

# ═══════════════════════════════════════════════════════
# SECTION 3 — EXPORT
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 3 — Export Workflows

## 3.1 TorchScript vs torch.export

| Feature | TorchScript (trace) | TorchScript (script) | torch.export |
|---------|--------------------|--------------------|-------------|
| Captures | Computation graph for given inputs | Python source → IR | Full graph with guards |
| Control flow | ❌ Data-dependent | ✅ Supported | ✅ With constraints |
| Dynamic shapes | ⚠️ Fixed by traced input | ✅ | ✅ With specifications |
| Python-free | ✅ | ✅ | ✅ |
| Future support | Legacy | Legacy | ✅ Recommended |

### When to Use What

```
Simple model, fixed shapes → TorchScript trace (fastest path)
Model with if/else         → TorchScript script
Future-proof, PyTorch 2.x  → torch.export
Interop with ONNX Runtime  → ONNX export
```"""))

cells.append(md("## 3.2 Implementation Tasks\n\nOpen `src/export_tools.py` and implement:\n\n1. **`export_torchscript_trace()`** — trace-based export\n2. **`export_torchscript_script()`** — script-based export\n3. **`export_torch_export()`** — torch.export workflow\n4. **`save_artifact()` / `load_artifact()`** — serialization"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Export + Parity Check                     ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Export via TorchScript trace                       ║
# ║  2. Export via torch.export (if available)             ║
# ║  3. Run parity checks for each                        ║
# ║  4. Save artifacts to disk                             ║
# ║  5. Print export summary                              ║
# ╚═══════════════════════════════════════════════════════╝

model_cpu = copy.deepcopy(model).cpu()
example_cpu = example_input.cpu()

# TODO: Export
# ts_traced = export_torchscript_trace(model_cpu, (example_cpu,))
# save_artifact(ts_traced, "resnet18_traced.pt")
#
# te_exported = export_torch_export(model_cpu, (example_cpu,))

# TODO: Parity checks
# parity_results = {}
# with torch.no_grad():
#     y_ref = model_cpu(example_cpu)
#     y_ts = ts_traced(example_cpu)
#     parity_results['TorchScript trace'] = compare_outputs(y_ref, y_ts)
# print_parity_report(parity_results)

# TODO: Export summary
# artifacts = {'torchscript_trace': 'resnet18_traced.pt'}
# print_export_summary(artifacts)
"""))

# ═══════════════════════════════════════════════════════
# SECTION 4 — BENCHMARKING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 4 — Benchmarking Discipline

## 4.1 Why Most Benchmarks Are Wrong

### Common Mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| No warmup | JIT compilation included in timing | 30+ warmup iters |
| No CUDA sync | GPU timings measure kernel launch only | `torch.cuda.synchronize()` |
| Using `time.time()` | Low resolution (ms) | `time.perf_counter()` (μs) |
| Only measuring mean | Hides tail latency spikes | Report p50/p95/p99 |
| Fixed batch size | Misses throughput sweet spot | Sweep batch sizes |

### Correct Timing Pattern

```python
# ✅ CORRECT GPU timing
torch.cuda.synchronize()           # wait for all prior ops
t0 = time.perf_counter()
model(input)
torch.cuda.synchronize()           # wait for forward pass to finish
elapsed = time.perf_counter() - t0

# ❌ WRONG — measures kernel launch, not compute
t0 = time.perf_counter()
model(input)                        # returns immediately (async!)
elapsed = time.perf_counter() - t0  # WAY too fast
```"""))

cells.append(md("## 4.2 Implementation Tasks\n\nOpen `src/inference_bench.py` and implement:\n\n1. **`benchmark_fn()`** — latency with warmup + CUDA sync + percentiles\n2. **`benchmark_model()`** — benchmark a model on fixed input\n3. **`compute_throughput()`** — items/sec calculation\n4. **`benchmark_batch_sizes()`** — throughput vs batch size sweep"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Benchmark Eager vs Exported Models        ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Benchmark eager FP32                               ║
# ║  2. Benchmark TorchScript traced                       ║
# ║  3. Benchmark torch.export (if available)              ║
# ║  4. Print comparison table                             ║
# ║  5. Plot latency histogram for each                    ║
# ╚═══════════════════════════════════════════════════════╝

batch_input = torch.randn(32, 3, 32, 32, device=DEVICE)

# TODO: Benchmark each runtime
# stats_eager = benchmark_model(model, batch_input, device=DEVICE)
# stats_ts = benchmark_model(ts_model, batch_input, device=DEVICE)
# 
# print(f"{'Runtime':<25} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10}")
# print("-" * 60)
# for name, stats in [("Eager FP32", stats_eager), ("TorchScript", stats_ts)]:
#     print(f"{name:<25} {stats['p50_ms']:>10.3f} {stats['p95_ms']:>10.3f} {stats['p99_ms']:>10.3f}")
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Throughput vs Batch Size                  ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Benchmark at batch_sizes = [1, 4, 16, 32, 64, 128]║
# ║  2. Compute throughput for each                        ║
# ║  3. Plot with plot_throughput_vs_batch()               ║
# ╚═══════════════════════════════════════════════════════╝

# TODO:
# batch_sizes = [1, 4, 16, 32, 64, 128]
# results = benchmark_batch_sizes(model, (3, 32, 32), batch_sizes, DEVICE)
# plot_throughput_vs_batch(results)
"""))

# ═══════════════════════════════════════════════════════
# SECTION 5 — PTQ
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 5 — Post-Training Quantization (PTQ)

## 5.1 How PTQ Works

```
1. Insert observers → record activation ranges during calibration
2. Run calibration data → observers collect min/max/histogram
3. Convert → replace float modules with int8 equivalents

Linear(float32) → QuantizedLinear(int8)
   Weight: float32 → int8 + scale + zero_point
   Activation: float32 → int8 (using observed ranges)
```

### Where Quantization Helps Most

| Hardware | Backend | Benefit |
|----------|---------|---------|
| x86 CPU | fbgemm | 2-4× faster int8 instructions (VNNI/AVX-512) |
| ARM/Mobile | qnnpack | Significant speedup on mobile SoCs |
| GPU | ❌ | PyTorch eager quantization does NOT accelerate GPU |

### Accuracy Impact

| Model Type | Typical PTQ Accuracy Drop |
|-----------|--------------------------|
| Large CNNs | < 1% |
| Small CNNs | 1-3% |
| Transformers | 1-2% (varies by layer) |
| Already-small models | 3-5%+ (harder) |"""))

cells.append(md("## 5.2 Implementation Tasks\n\nOpen `src/quantization_tools.py` and implement:\n\n1. **`prepare_ptq()`** — insert observers\n2. **`calibrate()`** — run calibration data\n3. **`convert_ptq()`** — convert to int8\n4. **`evaluate()`** — measure accuracy"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: PTQ Workflow                              ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Evaluate FP32 model accuracy (CPU)                 ║
# ║  2. Prepare model for PTQ (CPU only!)                  ║
# ║  3. Calibrate on calibration subset                    ║
# ║  4. Convert to int8                                    ║
# ║  5. Evaluate int8 accuracy                             ║
# ║  6. Benchmark CPU latency: fp32 vs int8                ║
# ║  7. Run parity check (with relaxed tolerance)          ║
# ╚═══════════════════════════════════════════════════════╝

model_fp32_cpu = copy.deepcopy(model).cpu().eval()

# TODO: FP32 baseline
# fp32_metrics = evaluate(model_fp32_cpu, test_loader, device=CPU)
# print(f"FP32 accuracy: {fp32_metrics['accuracy']:.2%}")

# TODO: PTQ
# model_ptq = copy.deepcopy(model_fp32_cpu)
# model_ptq = prepare_ptq(model_ptq, backend='fbgemm')
# calibrate(model_ptq, calib_loader, num_batches=50)
# model_int8 = convert_ptq(model_ptq)

# TODO: Evaluate
# int8_metrics = evaluate(model_int8, test_loader, device=CPU)
# print(f"INT8 accuracy: {int8_metrics['accuracy']:.2%}")
# print(f"Accuracy drop: {fp32_metrics['accuracy'] - int8_metrics['accuracy']:.2%}")
# print(f"Size: FP32={model_size_mb(model_fp32_cpu):.2f}MB, INT8={model_size_mb(model_int8):.2f}MB")
"""))

# ═══════════════════════════════════════════════════════
# SECTION 6 — QAT
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 6 — Quantization-Aware Training (QAT)

## 6.1 Why QAT Recovers Accuracy

PTQ applies quantization **after** training — the model never learned to handle the
precision loss. QAT inserts "fake quantize" modules **during** training:

```
Forward:  x → FakeQuantize(x) → Linear(FakeQuantize(W), fq_x)
          ↑ simulates int8 rounding during training
          ↑ model learns to be robust to quantization noise

Backward: STE (straight-through estimator) — gradients pass through FakeQuantize
```

### PTQ vs QAT

| Aspect | PTQ | QAT |
|--------|-----|-----|
| Training required | No | Yes (few epochs) |
| Accuracy | Good for large models | Better, especially small models |
| Complexity | Simple | More complex |
| Time | Minutes | Hours (but fewer epochs than full training) |"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: QAT Fine-tuning                          ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Prepare model for QAT                             ║
# ║  2. Fine-tune for 2-3 epochs with fake quantization   ║
# ║  3. Convert to int8                                    ║
# ║  4. Compare: PTQ accuracy vs QAT accuracy              ║
# ║  5. Benchmark both                                     ║
# ║  6. Print quantization report table                    ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: QAT
# model_qat = copy.deepcopy(model_fp32_cpu)
# model_qat.train()
# model_qat = prepare_qat(model_qat, backend='fbgemm')
# 
# # Short QAT fine-tune
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
# optimizer = torch.optim.SGD(model_qat.parameters(), lr=1e-4, momentum=0.9)
# loss_fn = nn.CrossEntropyLoss()
# 
# for epoch in range(2):
#     model_qat.train()
#     for i, (images, labels) in enumerate(train_loader):
#         if i >= 100: break  # short training
#         optimizer.zero_grad()
#         loss = loss_fn(model_qat(images), labels)
#         loss.backward()
#         optimizer.step()
# 
# model_qat_int8 = convert_qat(model_qat)
# qat_metrics = evaluate(model_qat_int8, test_loader, device=CPU)

# TODO: Report
# quant_report = {
#     'FP32': {'accuracy': fp32_metrics['accuracy'], 'size_mb': model_size_mb(model_fp32_cpu), 'latency_ms': ...},
#     'PTQ INT8': {'accuracy': int8_metrics['accuracy'], 'size_mb': model_size_mb(model_int8), 'latency_ms': ...},
#     'QAT INT8': {'accuracy': qat_metrics['accuracy'], 'size_mb': model_size_mb(model_qat_int8), 'latency_ms': ...},
# }
# print_quantization_report(quant_report)
"""))

# ═══════════════════════════════════════════════════════
# SECTION 7 — BATCHING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 7 — Dynamic Batching

## 7.1 The Batching Trade-off

```
Individual requests:  Low latency, low GPU utilization
                      GPU does tiny mat-muls → inefficient

Large batches:        High latency (waiting to fill batch), high throughput
                      GPU does large mat-muls → efficient

Dynamic batching:     Collect requests over a short window,
                      flush on max_batch_size OR max_wait_time
```

### Batching Policy Parameters

| Parameter | Effect of ↑ | Effect of ↓ |
|-----------|------------|------------|
| max_batch_size | ↑ throughput, ↑ latency | ↓ throughput, ↓ latency |
| max_wait_ms | ↑ batch fill rate, ↑ latency | ↓ latency, ↓ utilization |

### Architecture

```
Client 1 ──┐
Client 2 ──┤── Queue ──→ [Batcher] ──→ Model(batch) ──→ Split ──→ Futures
Client 3 ──┘                 ↑
                    Flush policy:
                    max_batch OR max_wait
```"""))

cells.append(md("## 7.2 Implementation Tasks\n\nOpen `src/batching.py` and implement:\n\n1. **`Batcher`** — the dynamic batcher class (submit, worker loop, shutdown)\n2. **`simulate_load()`** — Poisson arrival rate simulation"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Dynamic Batching Under Load               ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create a predict function                          ║
# ║  2. Create Batcher with different policies             ║
# ║  3. Simulate load and measure latency/throughput       ║
# ║  4. Compare policies                                   ║
# ╚═══════════════════════════════════════════════════════╝

# Predict function
model_cpu = copy.deepcopy(model).cpu().eval()

@torch.no_grad()
def predict_fn(batch):
    return model_cpu(batch)

# TODO: Test different policies
# policies = {
#     'batch=1, wait=10ms': (1, 10),
#     'batch=8, wait=20ms': (8, 20),
#     'batch=32, wait=50ms': (32, 50),
# }
# 
# all_results = {}
# for name, (max_bs, max_wait) in policies.items():
#     batcher = Batcher(predict_fn, max_batch_size=max_bs, max_wait_ms=max_wait)
#     load_result = simulate_load(batcher, input_shape=(3, 32, 32),
#                                 num_requests=100, arrival_rate=50)
#     latencies = load_result['latencies_ms']
#     all_results[name] = {
#         'p50_ms': np.percentile(latencies, 50),
#         'p95_ms': np.percentile(latencies, 95),
#         'throughput': len(latencies) / (sum(latencies) / 1000),
#     }
#     batcher.shutdown()
# 
# plot_batching_results(all_results)
"""))

# ═══════════════════════════════════════════════════════
# SECTION 8 — SERVING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 8 — Minimal REST Inference Server

## 8.1 Server Architecture

```
Client           Server (FastAPI)
  │                    │
  ├─── POST /predict ──→ Deserialize → Preprocess → Model → Response
  │                    │
  ├─── POST /predict_batch ──→ Stack batch → Model → Split → Responses
  │                    │
  └─── GET /health ────→ Status check
```

### Key Concerns

| Concern | Solution |
|---------|----------|
| Serialization | Base64-encode images in JSON |
| Warmup | Run dummy inference on startup |
| Concurrency | FastAPI async + thread pool |
| Monitoring | Log latency per request |
| Batching | Route through Batcher |"""))

cells.append(md("## 8.2 Implementation Tasks\n\nOpen `src/serve_app.py` and implement:\n\n1. **`/health`** — status endpoint\n2. **`/predict`** — single image prediction\n3. **`/predict_batch`** — batch endpoint"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  Server Launch + Load Test                             ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Fill in serve_app.py endpoints                     ║
# ║  2. Launch server (in a terminal):                     ║
# ║     cd src && uvicorn serve_app:app --port 8000        ║
# ║  3. Run load test below                                ║
# ╚═══════════════════════════════════════════════════════╝

from serve_app import LAUNCH_INSTRUCTIONS, generate_test_payload
print(LAUNCH_INSTRUCTIONS)

# Simple load test (run after server is up)
# import requests
# 
# # Health check
# r = requests.get("http://localhost:8000/health")
# print(f"Health: {r.json()}")
# 
# # Single prediction
# payload = generate_test_payload(1)
# r = requests.post("http://localhost:8000/predict", json=payload)
# print(f"Predict: {r.json()}")
# 
# # Batch prediction
# payload = generate_test_payload(8)
# r = requests.post("http://localhost:8000/predict_batch", json=payload)
# print(f"Batch: {r.json()['batch_size']} images, {r.json()['total_latency_ms']:.2f}ms")
"""))

# ═══════════════════════════════════════════════════════
# FINAL CHALLENGE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# 🧪 Final Challenge — Inference Report Card + Deployment Memo

## Required Report Card

| Configuration | p50 (ms) | p95 (ms) | Throughput (img/s) | Accuracy | Size (MB) | Notes |
|--------------|---------|---------|-------------------|----------|-----------|-------|
| Eager FP32 (CPU) | | | | | | baseline |
| Eager FP32 (GPU) | | | | | | |
| TorchScript (CPU) | | | | | | |
| PTQ INT8 (CPU) | | | | | | |
| QAT INT8 (CPU) | | | | | | |
| Best config | | | | | | |

## Deployment Decision Memo

Answer in 3-5 sentences each:

1. **Export format**: Which would you use for production and why?
2. **Quantization**: PTQ vs QAT — when is each worth it?
3. **Batch policy**: For a p95 SLA of 50ms, what batch config would you use?
4. **GPU vs CPU**: When does each make sense for inference?
5. **Correctness**: How did you ensure parity? What tolerances did you use?"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  FINAL: Build the Report Card                         ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Collect all measurements into one table
# import pandas as pd
# configs = ["eager_fp32_cpu", "eager_fp32_gpu", "torchscript_cpu",
#            "ptq_int8_cpu", "qat_int8_cpu"]
# columns = ["p50_ms", "p95_ms", "throughput", "accuracy", "size_mb", "notes"]
# report = pd.DataFrame(index=configs, columns=columns)
# display(report)
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  DEPLOYMENT DECISION MEMO                             ║
# ╚═══════════════════════════════════════════════════════╝

DEPLOYMENT_MEMO = \"\"\"
# Deployment Decision Memo

## 1. Export Format
TODO: Which format and why?

## 2. Quantization Trade-offs
TODO: PTQ vs QAT — when is each worth it?

## 3. Batch Policy for SLA
TODO: Given p95 SLA = 50ms, what config?

## 4. GPU vs CPU Inference
TODO: When does each make sense?

## 5. Correctness Assurance
TODO: How did you ensure parity?
\"\"\"
print(DEPLOYMENT_MEMO)"""))

# ═══════════════════════════════════════════════════════
# CHECKLIST
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# ✅ Summary Checklist

| # | Competency | Confident? |
|---|-----------|-----------|
| 1 | I can write parity tests that compare outputs element-wise | ☐ |
| 2 | I can export models via TorchScript trace and script | ☐ |
| 3 | I understand torch.export and when to use it | ☐ |
| 4 | I can benchmark with proper warmup, CUDA sync, and percentile reporting | ☐ |
| 5 | I can apply PTQ: prepare, calibrate, convert, evaluate | ☐ |
| 6 | I can run QAT and measure accuracy recovery over PTQ | ☐ |
| 7 | I can implement a dynamic batcher with configurable policy | ☐ |
| 8 | I can build a minimal REST inference server | ☐ |
| 9 | I can analyze the latency vs throughput trade-off | ☐ |
| 10 | I can write a deployment decision memo with evidence | ☐ |

### Common Deployment Mistakes

```
✗ No parity test → silently wrong predictions in production
✗ Benchmarking without warmup → includes JIT overhead
✗ Benchmarking GPU without sync → reports kernel launch time, not compute
✗ PTQ on GPU → no speedup (eager quantization is CPU-only)
✗ Saving model on GPU, loading on CPU → device mismatch error
✗ Dynamic batching with too large max_wait → SLA violations
✗ Not testing batch endpoint with variable sizes → crashes
```"""))

# ═══════════════════════════════════════════════════════
# BUILD
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
    os.pardir, "notebooks", "08_deployment_inference_serving.ipynb"
))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook: {out_path}")
print(f"Cells: {len(cells)} (markdown: {md_count}, code: {code_count})")
