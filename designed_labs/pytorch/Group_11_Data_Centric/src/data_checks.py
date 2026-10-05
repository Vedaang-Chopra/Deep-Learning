"""
data_checks.py — Data Quality Checks
========================================

Student implements:
  - check_class_balance(): class distribution table
  - check_missing_or_corrupt(): identify bad samples
  - check_label_noise_suspects(): flag potential mislabels

Data checks are unit tests for your dataset.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────
# Class Balance
# ─────────────────────────────────────────────────────

def check_class_balance(
    labels: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Check class distribution and flag severe imbalance.

    Returns
    -------
    dict with:
      'counts': dict[class] -> count
      'fractions': dict[class] -> fraction
      'imbalance_ratio': max_count / min_count
      'is_balanced': bool (ratio < 3)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Count each class                                   ║
    ║  2. Compute fractions                                  ║
    ║  3. imbalance_ratio = max_count / min_count            ║
    ║  4. Map to class_names if provided                     ║
    ║  5. Return report                                      ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement check_class_balance()")


# ─────────────────────────────────────────────────────
# Missing / Corrupt Samples
# ─────────────────────────────────────────────────────

def check_missing_or_corrupt(
    samples,
    check_fn=None,
) -> Dict:
    """
    Check for missing or corrupt samples.

    Parameters
    ----------
    samples : iterable — dataset or list of (image, label) tuples
    check_fn : optional callable(sample) -> bool
               returns True if sample is OK, False if corrupt
               If None, uses a default check that catches exceptions

    Returns
    -------
    dict with:
      'total_checked', 'num_ok', 'num_corrupt',
      'corrupt_indices': list, 'passed': bool

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Iterate over samples                               ║
    ║  2. For each sample, try:                              ║
    ║     a. If check_fn: result = check_fn(sample)          ║
    ║     b. Else: try accessing sample[0] (data)            ║
    ║        check shape is valid, dtype is numeric, etc.    ║
    ║  3. Catch exceptions → mark as corrupt                 ║
    ║  4. Return report                                      ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement check_missing_or_corrupt()")


# ─────────────────────────────────────────────────────
# Label Noise Suspects
# ─────────────────────────────────────────────────────

def check_label_noise_suspects(
    model_probs: np.ndarray,
    labels: np.ndarray,
    top_k: int = 50,
) -> Dict:
    """
    Flag potential label noise: high-confidence wrong predictions.

    A sample is "suspect" if:
      - The model is highly confident (max_prob > threshold)
      - But the model's prediction disagrees with the label

    These are likely mislabeled samples.

    Parameters
    ----------
    model_probs : (N, C) — softmax probabilities
    labels : (N,) — integer labels
    top_k : number of top suspects to return

    Returns
    -------
    dict with:
      'suspects': list of dicts with {index, label, predicted, confidence, loss}
      'num_suspects': int
      'suspect_rate': float

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. predicted = argmax(model_probs, axis=1)            ║
    ║  2. max_prob = max(model_probs, axis=1)                ║
    ║  3. Find where predicted != labels AND max_prob > 0.8  ║
    ║  4. Compute per-sample cross-entropy loss              ║
    ║  5. Sort by loss (descending) → highest loss = suspect ║
    ║  6. Return top_k suspects                              ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement check_label_noise_suspects()")


# ─────────────────────────────────────────────────────
# Visualization (provided)
# ─────────────────────────────────────────────────────

def plot_class_balance(balance_report: Dict, title: str = "Class Distribution"):
    """Plot class distribution bar chart. Provided."""
    counts = balance_report['counts']
    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]
    names = [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(names, values, color='steelblue', edgecolor='white')
    mean_val = np.mean(values)
    ax.axhline(mean_val, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.0f}')
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}  (imbalance ratio: {balance_report['imbalance_ratio']:.1f}x)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_label_noise_gallery(suspects: List[Dict], dataset, class_names=None, n_show=10):
    """Show a grid of suspected mislabeled samples. Provided."""
    n = min(n_show, len(suspects))
    fig, axes = plt.subplots(2, n // 2 + n % 2, figsize=(3 * (n // 2 + 1), 6))
    axes = axes.flatten() if n > 1 else [axes]
    for i, s in enumerate(suspects[:n]):
        img, _ = dataset[s['index']]
        if hasattr(img, 'numpy'):
            img = img.numpy()
        if img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 0.5 + 0.5, 0, 1)  # unnormalize approx
        axes[i].imshow(img)
        lbl = class_names[s['label']] if class_names else s['label']
        pred = class_names[s['predicted']] if class_names else s['predicted']
        axes[i].set_title(f"L:{lbl}\nP:{pred} ({s['confidence']:.0%})", fontsize=8)
        axes[i].axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.suptitle("Label Noise Suspects", fontsize=12)
    plt.tight_layout()
    plt.show()
