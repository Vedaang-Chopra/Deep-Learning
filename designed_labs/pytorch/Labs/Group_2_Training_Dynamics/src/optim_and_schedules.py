"""
optim_and_schedules.py — Optimizer & Scheduler Builders
========================================================

Student implements:
  - build_optimizer(): configurable optimizer factory
  - build_scheduler(): configurable scheduler factory
  - WarmupThenCosineScheduler: custom warmup + cosine decay LR scheduler

Called from the notebook for optimizer/scheduler ablation experiments.
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, StepLR, CosineAnnealingLR, OneCycleLR
from typing import Any, Dict, Iterator


# ─────────────────────────────────────────────────────
# Optimizer Factory
# ─────────────────────────────────────────────────────

def build_optimizer(
    params: Iterator[nn.Parameter],
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs,
) -> Optimizer:
    """
    Build and return a PyTorch optimizer.

    Supported optimizers:
      - 'sgd'    → torch.optim.SGD    (expects momentum in kwargs, default 0.9)
      - 'adam'   → torch.optim.Adam
      - 'adamw'  → torch.optim.AdamW

    Parameters
    ----------
    params : iterator of nn.Parameter
        Model parameters (model.parameters()).
    name : str
        Optimizer name (case-insensitive).
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay coefficient.
    **kwargs : dict
        Extra kwargs forwarded to the optimizer constructor
        (e.g., momentum=0.9 for SGD, betas=(0.9,0.999) for Adam).

    Returns
    -------
    torch.optim.Optimizer

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Hints:                                               ║
    ║  - Normalize name to lowercase                        ║
    ║  - For SGD, default momentum to 0.9 if not in kwargs  ║
    ║  - Pass weight_decay to all optimizers                ║
    ║  - Raise ValueError for unsupported names             ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement build_optimizer()")


# ─────────────────────────────────────────────────────
# Scheduler Factory
# ─────────────────────────────────────────────────────

def build_scheduler(
    optimizer: Optimizer,
    name: str = "cosine",
    **kwargs,
) -> _LRScheduler:
    """
    Build and return a learning rate scheduler.

    Supported schedulers:
      - 'step'       → StepLR(optimizer, step_size=..., gamma=...)
      - 'cosine'     → CosineAnnealingLR(optimizer, T_max=...)
      - 'onecycle'   → OneCycleLR(optimizer, max_lr=..., total_steps=...)
      - 'warmup_cosine' → WarmupThenCosineScheduler (student-implemented below)

    Parameters
    ----------
    optimizer : Optimizer
    name : str
        Scheduler name (case-insensitive).
    **kwargs : dict
        Scheduler-specific parameters forwarded to the constructor.

    Returns
    -------
    _LRScheduler

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Hints:                                               ║
    ║  - For 'step': require step_size and gamma in kwargs  ║
    ║  - For 'cosine': require T_max in kwargs              ║
    ║  - For 'onecycle': require max_lr, total_steps        ║
    ║  - For 'warmup_cosine': require warmup_steps, T_max   ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement build_scheduler()")


# ─────────────────────────────────────────────────────
# Custom Warmup + Cosine Scheduler
# ─────────────────────────────────────────────────────

class WarmupThenCosineScheduler(_LRScheduler):
    """
    Linear warmup for `warmup_steps` steps, then cosine annealing
    from base_lr to `eta_min` over the remaining steps.

    Total schedule:
      step < warmup_steps:
          lr = base_lr * (step / warmup_steps)
      step >= warmup_steps:
          lr = eta_min + 0.5*(base_lr - eta_min)*(1 + cos(π * progress))
          where progress = (step - warmup_steps) / (T_max - warmup_steps)

    Parameters
    ----------
    optimizer : Optimizer
    warmup_steps : int
        Number of linear warmup steps.
    T_max : int
        Total number of steps (warmup + cosine).
    eta_min : float
        Minimum learning rate (default: 0).

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and get_lr                  ║
    ║                                                       ║
    ║  Hints:                                               ║
    ║  - Call super().__init__(optimizer, last_epoch=-1)     ║
    ║  - Override get_lr() to return list of LRs            ║
    ║  - self.last_epoch gives current step count           ║
    ║  - self.base_lrs gives initial LRs per param group    ║
    ║  - Use math.cos and math.pi for cosine                ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        # TODO: call super().__init__()
        raise NotImplementedError("TODO: implement WarmupThenCosineScheduler.__init__()")

    def get_lr(self):
        """
        Compute current LR for each param group.

        Returns
        -------
        list of float

        TODO:
        - If self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        - Else:
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.T_max - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor
                    for base_lr in self.base_lrs]
        """
        raise NotImplementedError("TODO: implement get_lr()")


# ─────────────────────────────────────────────────────
# LR Curve Plotter (provided)
# ─────────────────────────────────────────────────────

def plot_lr_schedule(scheduler, total_steps: int, title: str = "LR Schedule"):
    """
    Visualize a learning rate schedule over `total_steps`.
    Provided utility — no student implementation needed.
    """
    import matplotlib.pyplot as plt

    lrs = []
    dummy_loss = torch.tensor(0.0)
    for _ in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        # Dummy optimizer step (weights don't matter here)
        scheduler.optimizer.step()
        scheduler.step()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lrs, linewidth=1.5, color="#4C72B0")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return lrs
