"""
drift_monitoring.py — Distribution Drift Detection
=====================================================

Student implements:
  - compute_feature_stats(): mean, covariance, quantiles
  - psi(): Population Stability Index
  - js_divergence(): Jensen-Shannon divergence
  - drift_report(): comprehensive drift report

Detect when your data distribution shifts.
"""

import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt


def compute_feature_stats(embeddings: np.ndarray) -> Dict:
    """
    Compute summary statistics for a set of embeddings.

    ╔═════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                   ║
    ║  Return: mean (D,), std (D,),                      ║
    ║  quantiles [0.25, 0.5, 0.75] per feature,         ║
    ║  num_samples                                        ║
    ╚═════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO")


def psi(reference_hist: np.ndarray, current_hist: np.ndarray, eps: float = 1e-8) -> float:
    """
    Population Stability Index between two histograms.

    PSI = Σ (p_i - q_i) * ln(p_i / q_i)

    ╔═════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                   ║
    ║  1. Normalize both to probability distributions     ║
    ║  2. Add eps for numerical stability                 ║
    ║  3. PSI = sum((p - q) * log(p / q))                ║
    ║                                                     ║
    ║  PSI < 0.1: no drift                               ║
    ║  PSI 0.1–0.25: moderate drift                      ║
    ║  PSI > 0.25: significant drift                     ║
    ╚═════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO")


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """
    Jensen-Shannon divergence (symmetric, bounded).

    JS(P‖Q) = 0.5 * KL(P‖M) + 0.5 * KL(Q‖M),  M = (P+Q)/2

    ╔═════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                   ║
    ║  1. Normalize p and q                              ║
    ║  2. m = 0.5 * (p + q)                              ║
    ║  3. KL(a‖b) = sum(a * log(a / (b + eps) + eps))   ║
    ║  4. JS = 0.5 * KL(p‖m) + 0.5 * KL(q‖m)           ║
    ╚═════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO")


def drift_report(
    ref_embeddings: np.ndarray,
    cur_embeddings: np.ndarray,
    n_bins: int = 50,
    top_k_features: int = 5,
) -> Dict:
    """
    Comprehensive drift report comparing reference vs current.

    ╔═════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                   ║
    ║  1. Compute feature stats for both                 ║
    ║  2. For each feature dimension:                    ║
    ║     histogram both, compute PSI and JS divergence  ║
    ║  3. Overall: mean PSI, max PSI, mean JS            ║
    ║  4. Find top_k drifted features                    ║
    ║  5. Return report dict                             ║
    ╚═════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO")


# ── Visualization (provided) ────────────────────────

def plot_drift_comparison(ref_stats, cur_stats, feature_idx=0, title="Feature Drift"):
    """Plot reference vs current distribution for one feature. Provided."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ref_stats, bins=50, alpha=0.5, color='steelblue', label='Reference', density=True)
    ax.hist(cur_stats, bins=50, alpha=0.5, color='coral', label='Current', density=True)
    ax.set_xlabel(f"Feature {feature_idx}")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_drift_summary(report: Dict, title="Drift Summary"):
    """Bar chart of per-feature PSI. Provided."""
    if 'per_feature_psi' not in report:
        print("No per_feature_psi in report"); return
    psis = report['per_feature_psi']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(psis)), psis, color='steelblue', alpha=0.7)
    ax.axhline(0.1, color='orange', linestyle='--', label='Moderate (0.1)')
    ax.axhline(0.25, color='red', linestyle='--', label='Significant (0.25)')
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("PSI")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.show()
