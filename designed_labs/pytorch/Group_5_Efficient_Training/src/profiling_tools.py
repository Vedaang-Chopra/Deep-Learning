"""
profiling_tools.py — PyTorch Profiling & NVTX Ranges
======================================================

Student implements:
  - profile_n_steps(): run profiler over training steps
  - annotated_train_step(): training step with NVTX ranges
  - summarize_profile(): extract key metrics from profiler output

Understanding where time is actually spent: data, forward, backward, optimizer.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Callable, Dict, Optional, List
import os


# ─────────────────────────────────────────────────────
# Annotated Training Step
# ─────────────────────────────────────────────────────

def annotated_train_step(
    model: nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler=None,
    device: torch.device = None,
) -> float:
    """
    A training step with NVTX ranges for profiling.

    Each phase is wrapped with torch.profiler.record_function
    so it appears as a labeled region in the profiler output.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Use record_function() to label each phase:           ║
    ║                                                       ║
    ║  with record_function("data_transfer"):                ║
    ║      images, labels = images.to(device), labels.to(..)║
    ║                                                       ║
    ║  optimizer.zero_grad()                                 ║
    ║                                                       ║
    ║  with record_function("forward"):                      ║
    ║      outputs = model(images)                           ║
    ║      loss = loss_fn(outputs, labels)                   ║
    ║                                                       ║
    ║  with record_function("backward"):                     ║
    ║      loss.backward()  (or scaler.scale(loss).backward)║
    ║                                                       ║
    ║  with record_function("optimizer"):                    ║
    ║      optimizer.step()  (or scaler.step + update)       ║
    ║                                                       ║
    ║  return loss.item()                                    ║
    ║                                                       ║
    ║  If scaler is provided, use AMP flow.                  ║
    ║  If scaler is None, use standard fp32 flow.            ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement annotated_train_step()")


# ─────────────────────────────────────────────────────
# Profile N Steps
# ─────────────────────────────────────────────────────

def profile_n_steps(
    step_fn: Callable[[], float],
    n_steps: int = 20,
    wait: int = 2,
    warmup_steps: int = 3,
    active: int = 5,
    repeat: int = 2,
    output_dir: str = "./profiler_output",
    with_stack: bool = True,
    record_shapes: bool = True,
) -> "torch.profiler.profile":
    """
    Profile training steps using torch.profiler with a schedule.

    Parameters
    ----------
    step_fn : callable() -> float
        Function that executes one training step.
    n_steps : int — total steps to run
    wait, warmup_steps, active, repeat : profiler schedule params
    output_dir : str — where to save Chrome trace
    with_stack : bool — capture Python stack traces
    record_shapes : bool — record tensor shapes

    Returns
    -------
    profiler object (can call .key_averages(), etc.)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Define activities:                                 ║
    ║     [ProfilerActivity.CPU]                             ║
    ║     + [ProfilerActivity.CUDA] if available             ║
    ║  2. Define schedule:                                   ║
    ║     torch.profiler.schedule(                           ║
    ║       wait=wait, warmup=warmup_steps,                  ║
    ║       active=active, repeat=repeat)                     ║
    ║  3. Define on_trace_ready:                             ║
    ║     torch.profiler.tensorboard_trace_handler(output_dir)║
    ║     OR save Chrome trace manually                      ║
    ║  4. Create profiler context:                           ║
    ║     with profile(activities, schedule, on_trace_ready,  ║
    ║                  record_shapes, with_stack) as prof:    ║
    ║       for step in range(n_steps):                      ║
    ║         step_fn()                                      ║
    ║         prof.step()                                    ║
    ║  5. Return prof                                        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement profile_n_steps()")


# ─────────────────────────────────────────────────────
# Profile Summary
# ─────────────────────────────────────────────────────

def summarize_profile(prof, top_n: int = 15) -> str:
    """
    Extract and format key metrics from a profiler run.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Use prof.key_averages() to get aggregated stats:      ║
    ║                                                       ║
    ║  table = prof.key_averages().table(                    ║
    ║      sort_by="cuda_time_total",  # or "cpu_time_total"║
    ║      row_limit=top_n,                                  ║
    ║  )                                                     ║
    ║                                                       ║
    ║  Also extract custom record_function regions:           ║
    ║  For each of ["data_transfer", "forward", "backward",  ║
    ║               "optimizer"]:                             ║
    ║    Find its entry in key_averages and report time       ║
    ║                                                       ║
    ║  Return the formatted string                           ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement summarize_profile()")


# ─────────────────────────────────────────────────────
# Quick Profile Helper (provided)
# ─────────────────────────────────────────────────────

def quick_profile(step_fn: Callable, steps: int = 10, label: str = ""):
    """
    Minimal profiling: just time each step and print stats.
    Provided utility for quick checks.
    """
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    times = []
    for i in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    import numpy as np
    t = np.array(times)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}{steps} steps: mean={t.mean():.1f}ms, "
          f"p50={np.percentile(t,50):.1f}ms, p95={np.percentile(t,95):.1f}ms")
