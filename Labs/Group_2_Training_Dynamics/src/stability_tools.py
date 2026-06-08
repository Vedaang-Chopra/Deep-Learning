"""
stability_tools.py — Gradient & Activation Diagnostics
========================================================

Student implements:
  - grad_norms(): global + per-layer gradient norms
  - activation_stats(): mean/std/%zeros per layer from hook cache
  - register_activation_hooks(): forward hooks for activation logging
  - detect_anomalies(): NaN/Inf detection in loss, params, grads

These tools are reusable across all subsequent groups.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict


# ─────────────────────────────────────────────────────
# Gradient Norms
# ─────────────────────────────────────────────────────

def grad_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute and return gradient L2 norms for every named parameter,
    plus the global (total) gradient norm.

    Returns
    -------
    dict with:
      - 'global': float — total grad norm across all params
      - '<param_name>': float — per-parameter grad norm

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Iterate model.named_parameters()                  ║
    ║  2. Skip params where .grad is None                   ║
    ║  3. Compute torch.norm(p.grad) for each               ║
    ║  4. Compute global norm = sqrt(sum of squared norms)  ║
    ║  5. Return dict with 'global' + per-param entries     ║
    ║                                                       ║
    ║  Hint: global_norm = torch.norm(                      ║
    ║      torch.stack([torch.norm(p.grad)                  ║
    ║                   for p in model.parameters()         ║
    ║                   if p.grad is not None])             ║
    ║  )                                                    ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement grad_norms()")


# ─────────────────────────────────────────────────────
# Activation Hooks & Stats
# ─────────────────────────────────────────────────────

def register_activation_hooks(
    model: nn.Module,
    whitelist_layers: Tuple[type, ...] = (nn.Conv2d, nn.Linear),
) -> Tuple[List[torch.utils.hooks.RemovableHook], Dict[str, torch.Tensor]]:
    """
    Register forward hooks on specified layer types to capture activations.

    Parameters
    ----------
    model : nn.Module
    whitelist_layers : tuple of types
        Only hook into these layer types.

    Returns
    -------
    handles : list of RemovableHook
        Call handle.remove() to clean up.
    cache : OrderedDict mapping layer_name → activation tensor
        Populated after each forward pass. Tensors are DETACHED.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Create an OrderedDict for the cache               ║
    ║  2. Iterate model.named_modules()                     ║
    ║  3. If isinstance(module, whitelist_layers):           ║
    ║     a. Define a hook fn that stores output.detach()    ║
    ║        in cache[name]                                  ║
    ║     b. Use a closure or functools.partial to capture   ║
    ║        the layer name                                  ║
    ║     c. Register via module.register_forward_hook(fn)   ║
    ║     d. Append handle to handles list                   ║
    ║  4. Return (handles, cache)                            ║
    ║                                                       ║
    ║  Hint: The closure trick:                              ║
    ║    def make_hook(name):                                ║
    ║        def hook(mod, inp, out):                        ║
    ║            cache[name] = out.detach()                  ║
    ║        return hook                                     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement register_activation_hooks()")


def activation_stats(cache: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Compute activation statistics from a hook cache.

    Parameters
    ----------
    cache : dict mapping layer_name → activation tensor

    Returns
    -------
    dict mapping layer_name → {
        'mean': float,
        'std': float,
        'min': float,
        'max': float,
        'pct_zeros': float,   ← percentage of exactly-zero activations
        'has_nan': bool,
        'has_inf': bool,
    }

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. For each (name, tensor) in cache:                 ║
    ║     - Flatten tensor                                  ║
    ║     - Compute mean, std, min, max                     ║
    ║     - Compute pct_zeros = (t == 0).float().mean()     ║
    ║     - Check torch.isnan/isinf                         ║
    ║  2. Return the stats dict                             ║
    ║                                                       ║
    ║  Hint: pct_zeros is critical for dead ReLU detection  ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement activation_stats()")


# ─────────────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────────────

def detect_anomalies(
    loss: torch.Tensor,
    model: nn.Module,
) -> Dict[str, Any]:
    """
    Check for NaN/Inf in loss, parameters, and gradients.

    Returns
    -------
    dict with:
      'loss_nan': bool
      'loss_inf': bool
      'param_nans': list of str (param names with NaN)
      'param_infs': list of str (param names with Inf)
      'grad_nans': list of str (param names with NaN grads)
      'grad_infs': list of str (param names with Inf grads)
      'is_healthy': bool (True if no anomalies found)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Check loss with torch.isnan/isinf                 ║
    ║  2. Iterate model.named_parameters()                  ║
    ║     - Check param.data for NaN/Inf                    ║
    ║     - Check param.grad (if not None) for NaN/Inf     ║
    ║  3. Set is_healthy = True only if all checks pass     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement detect_anomalies()")


# ─────────────────────────────────────────────────────
# Plotting Helpers (provided)
# ─────────────────────────────────────────────────────

def plot_grad_norms_layerwise(
    norms: Dict[str, float],
    title: str = "Gradient Norms per Layer",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Plot gradient norms as a bar chart. Provided utility."""
    names = [k for k in norms if k != "global"]
    values = [norms[k] for k in names]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    colors = ["#e74c3c" if v > 100 else "#3498db" if v < 1e-6 else "#2ecc71" for v in values]
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"{title}  |  Global: {norms.get('global', 0):.4f}", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dead_relu_pct(
    stats: Dict[str, Dict[str, float]],
    title: str = "% Zero Activations per Layer",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Plot percentage of dead (zero) activations per layer. Provided utility."""
    names = list(stats.keys())
    pcts = [stats[n]["pct_zeros"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    colors = ["#e74c3c" if p > 50 else "#f39c12" if p > 20 else "#2ecc71" for p in pcts]
    ax.bar(x, pcts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("% Zero Activations")
    ax.set_title(title, fontweight="bold")
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% dead threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_accs: List[float],
    label: str = "",
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """Plot train loss and val accuracy side-by-side. Provided utility."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(train_losses, linewidth=1.5, color="#4C72B0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title(f"Train Loss {label}", fontweight="bold")
    ax1.grid(alpha=0.3)

    ax2.plot(val_accs, linewidth=1.5, color="#DD8452")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy (%)")
    ax2.set_title(f"Val Accuracy {label}", fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_multi_curves(
    results: Dict[str, Dict[str, List[float]]],
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Overlay multiple training runs for comparison.

    Parameters
    ----------
    results : dict
        {run_label: {"train_loss": [...], "val_acc": [...]}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.cm.Set2

    for i, (label, data) in enumerate(results.items()):
        color = cmap(i / max(len(results) - 1, 1))
        if "train_loss" in data:
            ax1.plot(data["train_loss"], label=label, linewidth=1.5, color=color)
        if "val_acc" in data:
            ax2.plot(data["val_acc"], label=label, linewidth=1.5, color=color)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss")
    ax1.set_title("Train Loss Comparison", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy (%)")
    ax2.set_title("Val Accuracy Comparison", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
