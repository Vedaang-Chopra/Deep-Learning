"""
tensor_utils.py — Minimal Utilities for Tensor Internals Lab
=============================================================

This module provides ONLY helper functions for:
  - Plotting / visualization
  - Benchmarking wrappers

All core tensor inspection and diagnostic logic must be
implemented by the student inside the notebook.
"""

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable


# ─────────────────────────────────────────────────────
# Benchmarking Helpers
# ─────────────────────────────────────────────────────

def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 5,
    repeats: int = 50,
    sync_cuda: bool = True,
    **kwargs,
) -> Dict[str, float]:
    """
    Benchmark a callable with warmup and repeated timing.

    Parameters
    ----------
    fn : Callable
        Function to benchmark.
    warmup : int
        Number of warmup iterations (not timed).
    repeats : int
        Number of timed iterations.
    sync_cuda : bool
        Whether to call torch.cuda.synchronize() before/after each call.

    Returns
    -------
    dict with keys: mean_ms, std_ms, min_ms, max_ms, times_ms
    """
    device_is_cuda = any(
        isinstance(a, torch.Tensor) and a.is_cuda for a in args
    )

    for _ in range(warmup):
        fn(*args, **kwargs)
        if sync_cuda and device_is_cuda:
            torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if sync_cuda and device_is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if sync_cuda and device_is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "times_ms": times.tolist(),
    }


# ─────────────────────────────────────────────────────
# Plotting Helpers
# ─────────────────────────────────────────────────────

def plot_benchmark_comparison(
    labels: List[str],
    results: List[Dict[str, float]],
    title: str = "Benchmark Comparison",
    ylabel: str = "Time (ms)",
    figsize: Tuple[int, int] = (8, 5),
) -> None:
    """
    Bar chart comparing benchmark results with error bars.

    Parameters
    ----------
    labels : list of str
        Bar labels.
    results : list of dict
        Each dict must contain 'mean_ms' and 'std_ms' keys
        (as returned by `benchmark_fn`).
    """
    means = [r["mean_ms"] for r in results]
    stds = [r["std_ms"] for r in results]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(labels)],
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_gradient_norms(
    layer_names: List[str],
    grad_norms: List[float],
    title: str = "Gradient Norms per Layer",
    figsize: Tuple[int, int] = (10, 5),
    log_scale: bool = True,
) -> None:
    """
    Bar chart of gradient norms per layer.

    Parameters
    ----------
    layer_names : list of str
    grad_norms : list of float
    log_scale : bool
        If True, use log scale on y-axis (useful for
        detecting vanishing/exploding gradients).
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(layer_names))
    colors = []
    for g in grad_norms:
        if g < 1e-6:
            colors.append("#3498db")   # vanishing → blue
        elif g > 1e3:
            colors.append("#e74c3c")   # exploding → red
        else:
            colors.append("#2ecc71")   # healthy → green

    ax.bar(x, grad_norms, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Gradient Norm", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_activation_distributions(
    layer_names: List[str],
    activation_stats: List[Dict[str, float]],
    title: str = "Activation Statistics per Layer",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Grouped bar chart showing mean and std of activations per layer.

    Parameters
    ----------
    layer_names : list of str
    activation_stats : list of dict
        Each dict should contain 'mean' and 'std' keys.
    """
    means = [s["mean"] for s in activation_stats]
    stds = [s["std"] for s in activation_stats]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(layer_names))
    width = 0.35

    ax.bar(x - width / 2, means, width, label="Mean", color="#4C72B0",
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, stds, width, label="Std", color="#DD8452",
           edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_curves(
    losses: List[float],
    grad_norms_per_step: Optional[List[float]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plot loss curve and optionally gradient norm curve side-by-side.
    """
    n_plots = 1 + (grad_norms_per_step is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Loss curve
    axes[0].plot(losses, color="#4C72B0", linewidth=1.5)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Gradient norm curve
    if grad_norms_per_step is not None:
        axes[1].plot(grad_norms_per_step, color="#DD8452", linewidth=1.5)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Total Grad Norm")
        axes[1].set_title("Gradient Norms", fontweight="bold")
        axes[1].set_yscale("log")
        axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
