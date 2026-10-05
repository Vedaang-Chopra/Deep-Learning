"""
init_and_activations.py — Initialization Schemes & Activation Factories
========================================================================

Student implements:
  - init_weights(): apply initialization schemes to model parameters
  - make_activation(): configurable activation layer factory

This module is called from the notebook to initialize models and build
activation layers for ablation experiments.
"""

import torch
import torch.nn as nn
from typing import Optional


# ─────────────────────────────────────────────────────
# Activation Factory
# ─────────────────────────────────────────────────────

def make_activation(name: str) -> nn.Module:
    """
    Return an activation module by name.

    Supported activations:
      - 'relu'       → nn.ReLU(inplace=True)
      - 'leaky_relu' → nn.LeakyReLU(negative_slope=0.01, inplace=True)
      - 'gelu'       → nn.GELU()
      - 'silu'       → nn.SiLU(inplace=True)

    Parameters
    ----------
    name : str
        Case-insensitive activation name.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError if name is unsupported.

    ╔═══════════════════════════════════════════════╗
    ║  TODO: Implement this function.               ║
    ║                                               ║
    ║  Hints:                                       ║
    ║  - Normalize `name` to lowercase              ║
    ║  - Use a dict mapping for clean dispatch      ║
    ║  - For ReLU/LeakyReLU, inplace=True saves     ║
    ║    memory but be aware of autograd caveats     ║
    ╚═══════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement make_activation()")


# ─────────────────────────────────────────────────────
# Weight Initialization
# ─────────────────────────────────────────────────────

def init_weights(
    module: nn.Module,
    scheme: str = "kaiming_normal",
    nonlinearity: str = "relu",
) -> None:
    """
    Apply a weight initialization scheme to a single module IN-PLACE.

    This function should be used with `model.apply(partial(init_weights, ...))`.

    Supported schemes:
      - 'xavier_uniform'   → nn.init.xavier_uniform_
      - 'xavier_normal'    → nn.init.xavier_normal_
      - 'kaiming_uniform'  → nn.init.kaiming_uniform_  (uses nonlinearity)
      - 'kaiming_normal'   → nn.init.kaiming_normal_   (uses nonlinearity)
      - 'orthogonal'       → nn.init.orthogonal_
      - 'constant_small'   → nn.init.constant_(w, 0.01)  [BAD — for demonstration]
      - 'zeros'            → nn.init.zeros_               [BAD — for demonstration]

    Parameters
    ----------
    module : nn.Module
        The module whose weights/biases to initialize.
        Only acts on nn.Linear and nn.Conv2d.
    scheme : str
        Initialization scheme name.
    nonlinearity : str
        Nonlinearity for Kaiming init (e.g., 'relu', 'leaky_relu').

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Check if `module` is nn.Linear or nn.Conv2d       ║
    ║     (skip other module types — return early)          ║
    ║  2. Based on `scheme`, apply the correct init to      ║
    ║     module.weight                                     ║
    ║  3. If module.bias is not None, initialize to zeros   ║
    ║  4. For Kaiming, pass `nonlinearity` and correct      ║
    ║     `mode` (fan_out for Conv2d is common in ResNets)  ║
    ║                                                       ║
    ║  Hints:                                               ║
    ║  - Use isinstance() to check module type              ║
    ║  - nn.init functions modify tensors IN-PLACE          ║
    ║  - For 'constant_small', use nn.init.constant_        ║
    ║  - For 'zeros', ALL weights = 0 → symmetry problem!  ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement init_weights()")


# ─────────────────────────────────────────────────────
# Model Factory (provided skeleton)
# ─────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    """
    A small CNN for CIFAR-10 experiments.

    Architecture:
      Conv2d(3, 32, 3, padding=1) → Act → Conv2d(32, 32, 3, padding=1) → Act → MaxPool(2)
      Conv2d(32, 64, 3, padding=1) → Act → Conv2d(64, 64, 3, padding=1) → Act → MaxPool(2)
      Conv2d(64, 128, 3, padding=1) → Act → AdaptiveAvgPool(4)
      Flatten → Linear(128*4*4, 256) → Act → Dropout → Linear(256, 10)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward                 ║
    ║                                                       ║
    ║  Requirements:                                        ║
    ║  - Use make_activation(act_name) for all activations  ║
    ║  - Use nn.BatchNorm2d after each Conv2d               ║
    ║  - Accept `dropout_p` parameter for the FC dropout    ║
    ║  - After building, apply init_weights via             ║
    ║    self.apply(...)                                     ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        num_classes: int = 10,
        act_name: str = "relu",
        init_scheme: str = "kaiming_normal",
        dropout_p: float = 0.3,
    ):
        super().__init__()
        # TODO: Build the CNN layers described above
        # TODO: Apply initialization via self.apply(...)
        raise NotImplementedError("TODO: implement SmallCNN.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward pass through conv blocks → flatten → FC
        raise NotImplementedError("TODO: implement SmallCNN.forward()")
