"""
error_analysis.py — Error Analysis + Slice Metrics
=====================================================

Student implements:
  - confusion_matrix(): compute confusion matrix
  - per_class_metrics(): precision, recall, F1 per class
  - calibration_curve(): reliability diagram data
  - slice_by_feature(): group samples by a feature
  - slice_metrics(): compute metrics per slice

Accuracy hides failure patterns.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix. cm[i][j] = count of true=i, pred=j.

    ╔═════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                   ║
    ║  cm = np.zeros((num_classes, num_classes), int)     ║
    ║  For each (t, p): cm[t][p] += 1                    ║
    ╚═════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO")


def per_class_metrics(cm: np.ndarray, class_names: Optional[List[str]] = None) -> List[Dict]:
    """
    Precision, recall, F1 per class from confusion matrix.

    ╔═════════════════════════════════════════════════════╗
    ║  TODO: For each class i:                           ║
    ║  TP=cm[i][i], FP=col_sum-TP, FN=row_sum-TP        ║
    ║  precision, recall, f1, support                    ║
    ╚═════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO")


def calibration_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    """
    Compute (bin_accuracies, bin_confidences, bin_counts) for reliability diagram.

    ╔═════════════════════════════════════════════════════╗
    ║  TODO: Create n_bins equal-width bins [0,1).       ║
    ║  For each bin: mean accuracy, mean confidence, cnt ║
    ╚═════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO")


def slice_by_feature(samples: np.ndarray, feature_fn: Callable, n_bins: int = 5) -> Dict[str, List[int]]:
    """
    Group sample indices by feature into percentile bins.

    ╔═════════════════════════════════════════════════════╗
    ║  TODO: Compute feature, bin by percentile, return  ║
    ║  dict[bin_label] -> list of indices                ║
    ╚═════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO")


def slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, slices: Dict[str, List[int]]) -> Dict[str, Dict]:
    """
    Accuracy + count per slice.

    ╔═════════════════════════════════════════════════════╗
    ║  TODO: For each slice: accuracy, count             ║
    ╚═════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO")


# ── Visualization (provided) ────────────────────────

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    """Plot confusion matrix heatmap. Provided."""
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n*0.8), max(5, n*0.7)))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    labels = class_names or [str(i) for i in range(n)]
    ax.set(xticks=range(n), yticks=range(n), xticklabels=labels, yticklabels=labels,
           ylabel='True', xlabel='Predicted', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]}", ha='center', va='center',
                    color='white' if cm[i,j] > thresh else 'black', fontsize=7)
    plt.tight_layout(); plt.show()


def plot_calibration(bin_acc, bin_conf, bin_counts, title="Calibration Curve"):
    """Plot reliability diagram. Provided."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    mask = bin_counts > 0
    ax1.plot([0,1],[0,1],'k--',alpha=0.5,label='Perfect')
    ax1.bar(bin_conf[mask], bin_acc[mask], width=1.0/len(bin_acc), alpha=0.7, label='Model')
    ax1.set_xlabel("Confidence"); ax1.set_ylabel("Accuracy"); ax1.set_title(title)
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.bar(bin_conf[mask], bin_counts[mask], width=1.0/len(bin_acc), alpha=0.7, color='orange')
    ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count"); ax2.set_title("Histogram")
    ax2.grid(alpha=0.3); plt.tight_layout(); plt.show()


def plot_slice_metrics(report, title="Slice Metrics"):
    """Plot accuracy per slice. Provided."""
    names = list(report.keys())
    accs = [report[n]['accuracy'] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(names, accs, color='steelblue')
    ax.axhline(np.mean(accs), color='red', linestyle='--', alpha=0.5, label='Mean')
    ax.set_ylabel("Accuracy"); ax.set_title(title); ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout(); plt.show()
