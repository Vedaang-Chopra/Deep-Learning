"""
dist_metrics.py — Distributed Metric Aggregation
===================================================

Student implements:
  - all_reduce_mean(): average a tensor across ranks
  - all_reduce_sum(): sum a tensor across ranks
  - compute_global_accuracy(): correct distributed accuracy
  - gather_tensor(): gather tensors from all ranks to rank 0

Why naïve averaging is WRONG in distributed training.
"""

import torch
import torch.distributed as dist
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────
# All-Reduce Operations
# ─────────────────────────────────────────────────────

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average a tensor across all ranks.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. If not distributed, return tensor as-is            ║
    ║  2. Clone tensor (don't modify in-place)               ║
    ║  3. dist.all_reduce(tensor_clone, op=dist.ReduceOp.SUM)║
    ║  4. tensor_clone /= dist.get_world_size()              ║
    ║  5. Return tensor_clone                                ║
    ║                                                       ║
    ║  NOTE: all_reduce is IN-PLACE on the tensor.           ║
    ║  Always clone first if you need the original.          ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement all_reduce_mean()")


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Sum a tensor across all ranks.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Same as all_reduce_mean but without the division.     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement all_reduce_sum()")


# ─────────────────────────────────────────────────────
# Global Accuracy
# ─────────────────────────────────────────────────────

def compute_global_accuracy(
    local_correct: int,
    local_total: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute globally correct accuracy using all-reduce.

    Parameters
    ----------
    local_correct : int — correct predictions on this rank
    local_total : int — total samples on this rank
    device : torch.device

    Returns
    -------
    dict with 'global_accuracy', 'global_correct', 'global_total'

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  WHY naïve averaging is wrong:                        ║
    ║  If rank 0 has 100 samples (80 correct) = 80%          ║
    ║  and rank 1 has 50 samples (45 correct) = 90%          ║
    ║  Naïve: (80% + 90%) / 2 = 85%                         ║
    ║  Correct: 125 / 150 = 83.3%                            ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Convert to tensors on device                       ║
    ║  2. All-reduce SUM both correct and total              ║
    ║  3. accuracy = global_correct / global_total            ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compute_global_accuracy()")


# ─────────────────────────────────────────────────────
# Scaling Efficiency
# ─────────────────────────────────────────────────────

def measure_scaling_efficiency(
    throughput_1gpu: float,
    throughput_ngpu: float,
    world_size: int,
) -> Dict[str, float]:
    """
    Compute scaling efficiency metrics.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  speedup = throughput_ngpu / throughput_1gpu            ║
    ║  efficiency = speedup / world_size * 100  (percentage) ║
    ║  ideal_throughput = throughput_1gpu * world_size        ║
    ║  overhead = ideal_throughput - throughput_ngpu          ║
    ║                                                       ║
    ║  Return dict with all metrics.                         ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement measure_scaling_efficiency()")


# ─────────────────────────────────────────────────────
# Plotting (provided)
# ─────────────────────────────────────────────────────

def plot_scaling(results: Dict[int, float], metric: str = "images/sec"):
    """
    Plot throughput vs number of GPUs with ideal scaling line.
    Provided utility.
    """
    gpus = sorted(results.keys())
    vals = [results[g] for g in gpus]
    base = vals[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Throughput
    ax1.plot(gpus, vals, "o-", label="Actual", linewidth=2, markersize=8)
    ax1.plot(gpus, [base * g for g in gpus], "--", label="Ideal (linear)", alpha=0.5)
    ax1.set_xlabel("GPUs")
    ax1.set_ylabel(metric)
    ax1.set_title("Throughput Scaling")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Efficiency
    effs = [(results[g] / (base * g)) * 100 for g in gpus]
    ax2.bar(range(len(gpus)), effs, tick_label=[str(g) for g in gpus], color="#4CAF50")
    ax2.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="100% efficiency")
    ax2.set_xlabel("GPUs")
    ax2.set_ylabel("Efficiency (%)")
    ax2.set_title("Scaling Efficiency")
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
