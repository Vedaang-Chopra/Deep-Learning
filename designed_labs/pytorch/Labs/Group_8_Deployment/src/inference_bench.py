"""
inference_bench.py — Inference Benchmarking Discipline
========================================================

Student implements:
  - benchmark_fn(): precise latency measurement with warmup + CUDA sync
  - benchmark_model(): benchmark a model on real data
  - throughput(): compute items/sec from latency

Correct timing: warmup, CUDA sync, percentile reporting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from typing import Callable, Dict, Optional, Any
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────
# Core Benchmark Function
# ─────────────────────────────────────────────────────

def benchmark_fn(
    fn: Callable,
    warmup: int = 30,
    iters: int = 200,
    sync_cuda: bool = True,
) -> Dict[str, float]:
    """
    Benchmark a callable with proper methodology.

    Parameters
    ----------
    fn : callable — the function to benchmark (no args)
    warmup : int — warmup iterations (not timed)
    iters : int — timed iterations
    sync_cuda : bool — call torch.cuda.synchronize() for accurate GPU timing

    Returns
    -------
    dict with: 'mean_ms', 'std_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'min_ms', 'max_ms'

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Warmup: call fn() `warmup` times (not timed)       ║
    ║     If sync_cuda: torch.cuda.synchronize() after each  ║
    ║  2. Timed runs:                                        ║
    ║     For each iteration:                                ║
    ║       a. If sync_cuda: torch.cuda.synchronize()        ║
    ║       b. t0 = time.perf_counter()                      ║
    ║       c. fn()                                          ║
    ║       d. If sync_cuda: torch.cuda.synchronize()        ║
    ║       e. elapsed = time.perf_counter() - t0            ║
    ║       f. Record elapsed * 1000 (convert to ms)         ║
    ║  3. Compute statistics from the list of times:         ║
    ║     mean, std, p50, p95, p99, min, max                 ║
    ║     Use np.percentile for p50/p95/p99                  ║
    ║  4. Return stats dict                                  ║
    ║                                                       ║
    ║  PITFALL: Without CUDA sync, GPU timings are WRONG.    ║
    ║  GPU ops are async — time.perf_counter() only measures ║
    ║  the kernel launch time, not the actual compute time.  ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement benchmark_fn()")


# ─────────────────────────────────────────────────────
# Model Benchmark
# ─────────────────────────────────────────────────────

def benchmark_model(
    model: nn.Module,
    example_input: torch.Tensor,
    warmup: int = 30,
    iters: int = 200,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Benchmark a model on a fixed input tensor.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  1. model.eval(), move to device                       ║
    ║  2. example_input = example_input.to(device)           ║
    ║  3. def fn():                                          ║
    ║       with torch.no_grad():                            ║
    ║         model(example_input)                           ║
    ║  4. sync_cuda = (device.type == 'cuda')                ║
    ║  5. return benchmark_fn(fn, warmup, iters, sync_cuda)  ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement benchmark_model()")


# ─────────────────────────────────────────────────────
# Throughput
# ─────────────────────────────────────────────────────

def compute_throughput(
    batch_size: int,
    latency_ms: float,
) -> float:
    """
    Compute throughput in items/second.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  throughput = batch_size / (latency_ms / 1000)         ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compute_throughput()")


def benchmark_batch_sizes(
    model: nn.Module,
    input_shape: tuple,
    batch_sizes: list,
    device: torch.device,
    warmup: int = 20,
    iters: int = 100,
) -> Dict[int, Dict]:
    """
    Benchmark model at multiple batch sizes and compute throughput.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  For each batch_size:                                  ║
    ║  1. Create input: torch.randn(batch_size, *input_shape)║
    ║  2. Benchmark with benchmark_model()                   ║
    ║  3. Add throughput: compute_throughput(batch_size, p50) ║
    ║  4. Store results                                      ║
    ║                                                       ║
    ║  Return dict[batch_size] -> stats dict                 ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement benchmark_batch_sizes()")


# ─────────────────────────────────────────────────────
# Visualization (provided)
# ─────────────────────────────────────────────────────

def plot_latency_histogram(times_ms: list, title: str = "Latency Distribution"):
    """Plot latency histogram with percentile lines. Provided."""
    times = np.array(times_ms)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(times, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    for p, color, label in [(50, 'green', 'p50'), (95, 'orange', 'p95'), (99, 'red', 'p99')]:
        val = np.percentile(times, p)
        ax.axvline(val, color=color, linestyle='--', linewidth=2, label=f'{label}: {val:.2f}ms')
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_throughput_vs_batch(results: Dict[int, Dict]):
    """Plot throughput and latency vs batch size. Provided."""
    bs = sorted(results.keys())
    throughputs = [results[b]['throughput'] for b in bs]
    latencies = [results[b]['p50_ms'] for b in bs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(bs, throughputs, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (images/sec)")
    ax1.set_title("Throughput vs Batch Size")
    ax1.grid(alpha=0.3)

    ax2.plot(bs, latencies, 's-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("p50 Latency (ms)")
    ax2.set_title("Latency vs Batch Size")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
