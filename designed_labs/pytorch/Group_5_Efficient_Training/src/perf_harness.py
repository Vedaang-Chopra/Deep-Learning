"""
perf_harness.py — Performance Measurement Harness
====================================================

Student implements:
  - measure_step_time(): rigorous GPU-aware step timing
  - throughput(): compute images/sec or tokens/sec
  - StepTimer: context manager for step-level timing

Core measurement discipline: warmup, CUDA sync, statistics.
"""

import torch
import time
import numpy as np
from typing import Callable, Dict, Optional, Literal
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────
# Step Timer Context Manager (provided)
# ─────────────────────────────────────────────────────

class StepTimer:
    """
    Context manager for timing a single training step with proper CUDA sync.
    Provided utility.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


# ─────────────────────────────────────────────────────
# Measure Step Time
# ─────────────────────────────────────────────────────

def measure_step_time(
    step_fn: Callable[[], None],
    warmup: int = 10,
    iters: int = 50,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Measure step time with proper warmup and CUDA synchronization.

    Parameters
    ----------
    step_fn : callable
        A function that executes one training/forward step.
        Must take no arguments. Use functools.partial if needed.
    warmup : int
        Number of warmup iterations (not measured).
    iters : int
        Number of timed iterations.
    device : torch.device or None
        If CUDA, synchronize before/after each step.

    Returns
    -------
    dict with:
      'mean_ms' : float — mean step time in milliseconds
      'std_ms' : float — standard deviation
      'p50_ms' : float — median
      'p95_ms' : float — 95th percentile
      'min_ms' : float — minimum
      'max_ms' : float — maximum
      'times_ms' : list — all individual timings

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. If device is None, infer from torch.cuda          ║
    ║  2. Run `warmup` iterations WITHOUT recording          ║
    ║     (but DO call step_fn — this warms up CUDA, JIT)   ║
    ║  3. For each of `iters` iterations:                    ║
    ║     a. If CUDA: torch.cuda.synchronize()               ║
    ║     b. Record start time (time.perf_counter)           ║
    ║     c. Call step_fn()                                   ║
    ║     d. If CUDA: torch.cuda.synchronize()               ║
    ║     e. Record elapsed time in ms                       ║
    ║  4. Compute statistics using numpy                     ║
    ║  5. Return results dict                                ║
    ║                                                       ║
    ║  CRITICAL: The CUDA sync is essential. Without it,     ║
    ║  GPU ops are async → time.perf_counter only measures   ║
    ║  kernel launch time, NOT actual execution.             ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement measure_step_time()")


# ─────────────────────────────────────────────────────
# Throughput Calculator
# ─────────────────────────────────────────────────────

def throughput(
    metric: Literal["images", "tokens"],
    batch_size: int,
    step_time_ms: float,
    seq_len: Optional[int] = None,
) -> float:
    """
    Compute throughput in items/second.

    Parameters
    ----------
    metric : 'images' or 'tokens'
    batch_size : int
    step_time_ms : float — per-step time in milliseconds
    seq_len : int (required if metric='tokens')

    Returns
    -------
    float : items per second

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  images/sec = batch_size / (step_time_ms / 1000)       ║
    ║  tokens/sec = batch_size * seq_len / (step_time_ms/1000)║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement throughput()")


# ─────────────────────────────────────────────────────
# Reporting Helper (provided)
# ─────────────────────────────────────────────────────

def print_step_stats(stats: Dict[str, float], label: str = "") -> None:
    """Pretty-print step timing statistics. Provided utility."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Step time:")
    print(f"  mean   = {stats['mean_ms']:.2f} ms")
    print(f"  p50    = {stats['p50_ms']:.2f} ms")
    print(f"  p95    = {stats['p95_ms']:.2f} ms")
    print(f"  std    = {stats['std_ms']:.2f} ms")
    print(f"  min    = {stats['min_ms']:.2f} ms")
    print(f"  max    = {stats['max_ms']:.2f} ms")
