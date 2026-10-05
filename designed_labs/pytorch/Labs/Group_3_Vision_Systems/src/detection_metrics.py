"""
detection_metrics.py — Object Detection Metrics from Scratch
=============================================================

Student implements:
  - compute_iou(): Intersection over Union for two bounding boxes
  - non_max_suppression(): NMS algorithm
  - mean_average_precision(): mAP computation

No torchvision.ops allowed.
All implementations must be from first principles.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────

def compute_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    format: str = "xyxy",
) -> torch.Tensor:
    """
    Compute Intersection over Union between two sets of boxes.

    Parameters
    ----------
    box1 : (N, 4) tensor
    box2 : (M, 4) tensor
    format : str
        'xyxy' → [x1, y1, x2, y2]
        'xywh' → [cx, cy, w, h]

    Returns
    -------
    iou : (N, M) tensor of pairwise IoU values

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps (for 'xyxy' format):                           ║
    ║  1. Compute intersection rectangle:                   ║
    ║     inter_x1 = max(box1_x1, box2_x1)                 ║
    ║     inter_y1 = max(box1_y1, box2_y1)                  ║
    ║     inter_x2 = min(box1_x2, box2_x2)                 ║
    ║     inter_y2 = min(box1_y2, box2_y2)                  ║
    ║  2. inter_area = clamp(w, min=0) * clamp(h, min=0)   ║
    ║  3. union = area1 + area2 - inter_area                ║
    ║  4. iou = inter_area / (union + 1e-8)                 ║
    ║                                                       ║
    ║  For 'xywh': convert to 'xyxy' first.                 ║
    ║                                                       ║
    ║  Hint: Use broadcasting. Reshape box1 to (N,1,4)      ║
    ║  and box2 to (1,M,4) for pairwise computation.        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compute_iou()")


# ─────────────────────────────────────────────────────
# Non-Max Suppression
# ─────────────────────────────────────────────────────

def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Apply Non-Maximum Suppression.

    Parameters
    ----------
    boxes : (N, 4) tensor in xyxy format
    scores : (N,) tensor of confidence scores
    iou_threshold : float
        Boxes with IoU > this with a higher-scoring box are suppressed.
    score_threshold : float
        Boxes with score < this are discarded before NMS.

    Returns
    -------
    keep : (K,) tensor of indices to keep

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement the NMS algorithm.                   ║
    ║                                                       ║
    ║  Algorithm:                                           ║
    ║  1. Filter boxes by score_threshold                   ║
    ║  2. Sort remaining by score (descending)              ║
    ║  3. Initialize keep = []                              ║
    ║  4. While boxes remain:                               ║
    ║     a. Take the highest-scoring box → add to keep     ║
    ║     b. Compute IoU of this box with all remaining     ║
    ║     c. Remove boxes with IoU > iou_threshold          ║
    ║  5. Return indices of kept boxes                      ║
    ║                                                       ║
    ║  Hint: Use your compute_iou() function.               ║
    ║  Watch out: compute_iou returns (N,M), you may need   ║
    ║  (1, remaining) → squeeze for comparison.             ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement non_max_suppression()")


# ─────────────────────────────────────────────────────
# Mean Average Precision
# ─────────────────────────────────────────────────────

def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """
    Compute Average Precision using 11-point interpolation.

    Parameters
    ----------
    recalls : (N,) sorted recall values
    precisions : (N,) corresponding precision values

    Returns
    -------
    float : AP value

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement 11-point interpolation AP.           ║
    ║                                                       ║
    ║  For each threshold t in [0.0, 0.1, ..., 1.0]:        ║
    ║    p(t) = max precision at recall >= t                ║
    ║  AP = (1/11) * sum(p(t) for t in thresholds)          ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compute_ap()")


def mean_average_precision(
    pred_boxes: List[Dict],
    gt_boxes: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 10,
) -> Dict[str, float]:
    """
    Compute mAP across all classes.

    Parameters
    ----------
    pred_boxes : list of dict
        Each dict: {'image_id': int, 'class': int, 'confidence': float,
                    'box': [x1, y1, x2, y2]}
    gt_boxes : list of dict
        Each dict: {'image_id': int, 'class': int, 'box': [x1, y1, x2, y2]}
    iou_threshold : float
    num_classes : int

    Returns
    -------
    dict with:
      'mAP': float
      'per_class_ap': dict mapping class_id → AP

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement mAP computation.                     ║
    ║                                                       ║
    ║  For each class c:                                     ║
    ║  1. Collect all predictions of class c, sort by conf   ║
    ║  2. Collect all ground truths of class c               ║
    ║  3. For each prediction (in conf order):               ║
    ║     a. Find best matching GT (highest IoU)             ║
    ║     b. If IoU > threshold and GT not yet matched → TP  ║
    ║     c. Else → FP                                       ║
    ║  4. Compute precision & recall at each threshold       ║
    ║  5. Compute AP using compute_ap()                      ║
    ║  mAP = mean of per-class APs                          ║
    ║                                                       ║
    ║  Hint: Track which GTs are "used" per image with a    ║
    ║  dict of sets.                                         ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement mean_average_precision()")


# ─────────────────────────────────────────────────────
# Visualization Helpers (provided)
# ─────────────────────────────────────────────────────

def visualize_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    title: str = "",
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """
    Visualize bounding boxes on an image. Provided utility.

    Parameters
    ----------
    image : (3, H, W) or (H, W, 3) tensor
    boxes : (N, 4) tensor in xyxy format
    scores : (N,) optional confidence scores
    labels : (N,) optional class indices
    """
    if image.dim() == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.cpu().numpy()
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)

    colors = plt.cm.Set2(np.linspace(0, 1, 10))
    for i, box in enumerate(boxes.cpu().numpy()):
        x1, y1, x2, y2 = box
        label_idx = int(labels[i]) if labels is not None else 0
        color = colors[label_idx % len(colors)]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
            boxstyle="round,pad=0.02",
        )
        ax.add_patch(rect)
        text = ""
        if class_names and labels is not None:
            text = class_names[label_idx]
        if scores is not None:
            text += f" {scores[i]:.2f}"
        if text:
            ax.text(x1, y1 - 4, text, fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_nms_comparison(
    image: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    keep_indices: torch.Tensor,
    figsize: Tuple[int, int] = (16, 7),
) -> None:
    """Visualize before/after NMS side by side. Provided utility."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if image.dim() == 3 and image.shape[0] in (1, 3):
        img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image.cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)

    for ax, title, idxs in [(ax1, f"Before NMS ({len(boxes)} boxes)", range(len(boxes))),
                             (ax2, f"After NMS ({len(keep_indices)} boxes)", keep_indices.cpu().numpy())]:
        ax.imshow(img_np)
        for i in idxs:
            x1, y1, x2, y2 = boxes[i].cpu().numpy()
            rect = mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"{scores[i]:.2f}", fontsize=8, color="white",
                    bbox=dict(boxstyle="round", facecolor="green", alpha=0.7))
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
