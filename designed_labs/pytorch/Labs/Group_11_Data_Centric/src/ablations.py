"""
ablations.py — Structured Ablation Harness
=============================================

Student implements:
  - generate_grid(): expand config space to list of configs
  - run_sweep(): run training for each config × seed
  - summarize_results(): aggregate mean/std/best

No one-off notebook tuning — controlled experiments only.
"""

import itertools
import copy
import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────
# Grid Generation
# ─────────────────────────────────────────────────────

def generate_grid(config_space: Dict[str, List]) -> List[Dict]:
    """
    Expand a config space dict into a flat list of configs.

    Parameters
    ----------
    config_space : dict mapping param_name -> list of values
        e.g. {'optimizer': ['sgd','adamw'], 'lr': [0.01, 0.001]}

    Returns
    -------
    list of dicts, one per combination
        e.g. [{'optimizer':'sgd','lr':0.01}, {'optimizer':'sgd','lr':0.001}, ...]

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. keys = list(config_space.keys())                   ║
    ║  2. values = list(config_space.values())               ║
    ║  3. combos = itertools.product(*values)                ║
    ║  4. Return [dict(zip(keys, c)) for c in combos]        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement generate_grid()")


# ─────────────────────────────────────────────────────
# Sweep Runner
# ─────────────────────────────────────────────────────

def run_sweep(
    configs: List[Dict],
    train_fn: Callable[[Dict, int], Dict],
    seeds: List[int] = None,
) -> List[Dict]:
    """
    Run training for each config × seed combination.

    Parameters
    ----------
    configs : list of config dicts
    train_fn : callable(config, seed) -> metrics_dict
               must return at least {'accuracy': float}
    seeds : list of random seeds to repeat each config

    Returns
    -------
    list of result dicts, each with:
      {'config': {...}, 'seed': int, 'metrics': {...}, 'elapsed_s': float}

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. If seeds is None: seeds = [42]                     ║
    ║  2. results = []                                       ║
    ║  3. For each config:                                   ║
    ║       For each seed:                                   ║
    ║         a. Print progress                              ║
    ║         b. t0 = time.time()                            ║
    ║         c. metrics = train_fn(config, seed)            ║
    ║         d. elapsed = time.time() - t0                  ║
    ║         e. Append result dict                          ║
    ║  4. Return results                                     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement run_sweep()")


# ─────────────────────────────────────────────────────
# Results Summary
# ─────────────────────────────────────────────────────

def summarize_results(
    results: List[Dict],
    metric_key: str = "accuracy",
) -> Dict:
    """
    Aggregate results: mean/std per config, find best.

    Returns
    -------
    dict with:
      'per_config': list of {config_str, mean, std, n_seeds, values}
      'best_config': str
      'best_mean': float

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Group results by config (convert config to frozenset)║
    ║  2. For each config group:                             ║
    ║     a. Collect metric values across seeds              ║
    ║     b. Compute mean and std                            ║
    ║  3. Sort by mean (descending)                          ║
    ║  4. Return summary                                     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement summarize_results()")


# ─────────────────────────────────────────────────────
# Visualization (provided)
# ─────────────────────────────────────────────────────

def plot_ablation_results(summary: Dict, metric_name: str = "Accuracy"):
    """Bar chart of ablation results with error bars. Provided."""
    configs = summary['per_config']
    names = [c['config_str'] for c in configs]
    means = [c['mean'] for c in configs]
    stds = [c['std'] for c in configs]

    # Truncate long names
    display_names = [n[:30] + '...' if len(n) > 30 else n for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.5), 5))
    bars = ax.bar(range(len(configs)), means, yerr=stds, capsize=5,
                  color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Ablation Results ({metric_name})")
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = np.argmax(means)
    bars[best_idx].set_color('#4CAF50')

    plt.tight_layout()
    plt.show()
