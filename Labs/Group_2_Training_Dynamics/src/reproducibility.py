"""
reproducibility.py — Determinism & Reproducibility Utilities
=============================================================

Student implements:
  - seed_everything(): set all random seeds + deterministic flags
  - seed_worker(): DataLoader worker init function
  - make_dataloader(): reproducible DataLoader factory

These tools ensure bit-exact reproducible training runs.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional


# ─────────────────────────────────────────────────────
# Seed Everything
# ─────────────────────────────────────────────────────

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    Must set:
      - random.seed(seed)
      - np.random.seed(seed)
      - torch.manual_seed(seed)
      - torch.cuda.manual_seed_all(seed)  (if CUDA available)

    If deterministic=True, also set:
      - torch.backends.cudnn.deterministic = True
      - torch.backends.cudnn.benchmark = False
      - torch.use_deterministic_algorithms(True, warn_only=True)
      - os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    Parameters
    ----------
    seed : int
    deterministic : bool
        If True, enable fully deterministic mode (may reduce performance).

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Hints:                                               ║
    ║  - torch.use_deterministic_algorithms may fail for    ║
    ║    some ops on CUDA. Use warn_only=True to avoid      ║
    ║    crashes during development.                         ║
    ║  - CUBLAS_WORKSPACE_CONFIG is needed for CUDA ≥10.2   ║
    ║  - cudnn.benchmark=True is faster but non-deterministic║
    ║  - Setting this BEFORE model/data creation matters    ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement seed_everything()")


# ─────────────────────────────────────────────────────
# DataLoader Worker Seeding
# ─────────────────────────────────────────────────────

def seed_worker(worker_id: int) -> None:
    """
    Worker init function for DataLoader reproducibility.

    Each DataLoader worker runs in a separate process with its own
    random state. Without explicit seeding, workers produce different
    random augmentations across runs.

    Must set:
      - Get the per-worker seed from torch.utils.data.get_worker_info()
        OR compute from torch.initial_seed() % 2**32
      - Set np.random.seed(worker_seed)
      - Set random.seed(worker_seed)

    Usage:
      DataLoader(..., worker_init_fn=seed_worker)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Hints:                                               ║
    ║  - worker_seed = torch.initial_seed() % 2**32         ║
    ║    This ensures each worker gets a unique but          ║
    ║    deterministic seed derived from the main seed.     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement seed_worker()")


# ─────────────────────────────────────────────────────
# Reproducible DataLoader Factory
# ─────────────────────────────────────────────────────

def make_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    seed: int = 42,
    deterministic: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader with proper reproducibility settings.

    Key requirements:
      - Use torch.Generator seeded with `seed` for shuffling
      - Set worker_init_fn=seed_worker
      - Set persistent_workers=True if num_workers > 0

    Parameters
    ----------
    dataset : Dataset
    batch_size : int
    shuffle : bool
    num_workers : int
    seed : int
    deterministic : bool
        If True, uses seeded generator + worker_init_fn.
    pin_memory : bool
    drop_last : bool

    Returns
    -------
    DataLoader

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Create a torch.Generator and seed it:             ║
    ║     g = torch.Generator()                             ║
    ║     g.manual_seed(seed)                               ║
    ║  2. Build DataLoader with:                            ║
    ║     generator=g (if deterministic)                    ║
    ║     worker_init_fn=seed_worker (if deterministic)     ║
    ║     persistent_workers=(num_workers > 0)               ║
    ║  3. Return the DataLoader                             ║
    ║                                                       ║
    ║  Hint: The generator controls shuffle order.          ║
    ║  Without it, shuffle order varies across runs.        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement make_dataloader()")


# ─────────────────────────────────────────────────────
# Verification Helper (provided)
# ─────────────────────────────────────────────────────

def verify_reproducibility(
    train_fn,
    config: dict,
    n_runs: int = 3,
    metric_key: str = "val_acc",
) -> bool:
    """
    Run the same training function n_runs times with the same config.
    Check if results are identical (for deterministic runs) or
    within tolerance (for non-deterministic).

    Parameters
    ----------
    train_fn : callable
        A function that takes a config dict and returns a dict of metrics.
    config : dict
        Training configuration (must include 'seed').
    n_runs : int
    metric_key : str
        Key in the returned metrics dict to compare.

    Returns
    -------
    bool : True if all runs produced identical metric values.
    """
    results = []
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...", end=" ")
        metrics = train_fn(config)
        val = metrics[metric_key]
        print(f"{metric_key}={val:.4f}")
        results.append(val)

    all_same = all(abs(r - results[0]) < 1e-6 for r in results)
    if all_same:
        print(f"  ✅ All {n_runs} runs produced identical {metric_key}={results[0]:.4f}")
    else:
        spread = max(results) - min(results)
        print(f"  ⚠️  Results varied: spread={spread:.6f}")
        print(f"     Values: {[f'{r:.4f}' for r in results]}")
    return all_same
