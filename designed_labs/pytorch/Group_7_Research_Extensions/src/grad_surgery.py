"""
grad_surgery.py — Gradient Surgery for Multi-Task Learning
=============================================================

Student implements:
  - pcgrad_step(): PCGrad (Yu et al., 2020) gradient projection
  - compute_task_gradients(): extract per-task gradients
  - cosine_similarity_tasks(): measure gradient conflict

When task gradients conflict, project them to reduce interference.
"""

import torch
import torch.nn as nn
import copy
import random
from typing import List, Tuple, Dict, Callable
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────
# Per-Task Gradient Extraction
# ─────────────────────────────────────────────────────

def compute_task_gradients(
    model: nn.Module,
    task_losses: List[torch.Tensor],
    shared_params: List[nn.Parameter] = None,
) -> List[torch.Tensor]:
    """
    Compute per-task gradient vectors as flat tensors.

    Parameters
    ----------
    model : nn.Module — the shared model
    task_losses : list of scalar tensors — one loss per task
    shared_params : list of parameters to extract gradients for.
                    If None, uses all model.parameters()

    Returns
    -------
    list of flat gradient tensors, one per task

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  For each task loss:                                   ║
    ║  1. model.zero_grad()                                  ║
    ║  2. loss.backward(retain_graph=True)                   ║
    ║     ↑ retain_graph=True so we can backward again       ║
    ║  3. Collect gradients from shared_params               ║
    ║  4. Flatten and concatenate into a single vector       ║
    ║  5. Append to list                                     ║
    ║  6. model.zero_grad() after (clean up)                 ║
    ║                                                       ║
    ║  Return list of flat gradient vectors.                 ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compute_task_gradients()")


# ─────────────────────────────────────────────────────
# PCGrad: Projecting Conflicting Gradients
# ─────────────────────────────────────────────────────

def pcgrad_project(
    task_gradients: List[torch.Tensor],
) -> torch.Tensor:
    """
    PCGrad: project conflicting task gradients.

    Algorithm (Yu et al., 2020):
    For each pair of task gradients (g_i, g_j):
      if g_i · g_j < 0:  (conflict!)
        g_i = g_i - (g_i · g_j / ||g_j||²) * g_j
        (project g_i onto the normal plane of g_j)

    Parameters
    ----------
    task_gradients : list of flat gradient tensors

    Returns
    -------
    combined gradient tensor (sum of projected gradients)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement PCGrad projection.                   ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Clone all task_gradients (don't modify originals)  ║
    ║  2. Shuffle task order randomly                        ║
    ║  3. For each task gradient g_i:                        ║
    ║     For each OTHER task gradient g_j:                  ║
    ║       dot = g_i · g_j                                  ║
    ║       if dot < 0:  # conflict                          ║
    ║         g_i = g_i - (dot / (g_j · g_j)) * g_j         ║
    ║  4. Return sum of all projected gradients               ║
    ║                                                       ║
    ║  NUMERICAL: avoid division by zero (||g_j|| == 0)      ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement pcgrad_project()")


def pcgrad_step(
    model: nn.Module,
    task_losses: List[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    shared_params: List[nn.Parameter] = None,
) -> Dict[str, float]:
    """
    Full PCGrad step: compute task grads, project, apply.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  1. task_grads = compute_task_gradients(...)            ║
    ║  2. combined = pcgrad_project(task_grads)               ║
    ║  3. Manually set .grad for each shared parameter       ║
    ║     using slices of the combined flat gradient          ║
    ║  4. optimizer.step()                                   ║
    ║  5. Return info dict with cosine similarities, norms   ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement pcgrad_step()")


# ─────────────────────────────────────────────────────
# Gradient Conflict Analysis
# ─────────────────────────────────────────────────────

def cosine_similarity_tasks(
    task_gradients: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between task gradients.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  Returns (num_tasks, num_tasks) matrix of cosines.     ║
    ║  Negative values indicate conflicting gradients.       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement cosine_similarity_tasks()")


# ─────────────────────────────────────────────────────
# Multi-Task Model (provided)
# ─────────────────────────────────────────────────────

class MultiTaskNet(nn.Module):
    """
    Simple multi-task model: shared backbone + task-specific heads.
    Provided for experiments.
    """

    def __init__(self, in_features: int, hidden: int, num_tasks: int, task_classes: List[int]):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, c) for c in task_classes
        ])

    def forward(self, x) -> List[torch.Tensor]:
        features = self.backbone(x)
        return [head(features) for head in self.heads]

    def shared_parameters(self):
        return list(self.backbone.parameters())


# ─────────────────────────────────────────────────────
# Visualization (provided)
# ─────────────────────────────────────────────────────

def plot_gradient_cosines(cosine_history: List[torch.Tensor], labels=None):
    """Plot gradient cosine similarity between tasks over time. Provided."""
    steps = len(cosine_history)
    num_tasks = cosine_history[0].shape[0]
    pairs = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            vals = [c[i, j].item() for c in cosine_history]
            label = f"Task {i} vs {j}" if labels is None else f"{labels[i]} vs {labels[j]}"
            pairs.append((label, vals))

    plt.figure(figsize=(10, 4))
    for label, vals in pairs:
        plt.plot(vals, label=label, linewidth=2)
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Conflict boundary")
    plt.xlabel("Step")
    plt.ylabel("Cosine Similarity")
    plt.title("Task Gradient Cosine Similarity Over Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
