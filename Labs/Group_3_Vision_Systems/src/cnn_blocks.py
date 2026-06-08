"""
cnn_blocks.py — CNN Building Blocks
=====================================

Student implements:
  - ConvBlock: Conv2d + optional BatchNorm + activation
  - ResidualBlock: skip-connection block with shape alignment
  - build_plain_cnn(): stack of ConvBlocks (no residuals)
  - build_residual_cnn(): stack with ResidualBlocks

Called from notebook for architecture comparison experiments.
"""

import torch
import torch.nn as nn
from typing import Optional, List


# ─────────────────────────────────────────────────────
# ConvBlock
# ─────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    A single convolution block: Conv2d → [BatchNorm] → Activation.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int (default 3)
    stride : int (default 1)
    padding : int (default 1)
    use_batchnorm : bool (default True)
    activation : str (default 'relu')
        One of: 'relu', 'leaky_relu', 'gelu', 'none'

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. nn.Conv2d with given params                       ║
    ║  2. nn.BatchNorm2d(out_channels) if use_batchnorm     ║
    ║  3. Activation layer based on `activation` string     ║
    ║  4. Use nn.Sequential or store as attributes          ║
    ║                                                       ║
    ║  forward(x):                                          ║
    ║  1. x → Conv → [BN] → Act → return                   ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        raise NotImplementedError("TODO: implement ConvBlock.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement ConvBlock.forward()")


# ─────────────────────────────────────────────────────
# ResidualBlock
# ─────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    A residual block with two ConvBlocks and a skip connection.

    Architecture:
        input ─→ ConvBlock(in, out, stride) → ConvBlock(out, out, stride=1) → + → output
          │                                                                    ↑
          └─── [optional 1x1 conv for shape alignment] ────────────────────────┘

    The skip connection must handle:
      1. Channel mismatch: use 1x1 conv to project input channels → out channels
      2. Spatial mismatch: use stride in the 1x1 conv to match spatial dims

    Parameters
    ----------
    in_channels : int
    out_channels : int
    stride : int (default 1)
        Stride for the first conv (and the skip 1x1 if needed).
    use_batchnorm : bool (default True)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. First ConvBlock: (in_channels, out_channels,      ║
    ║     stride=stride)                                    ║
    ║  2. Second ConvBlock: (out_channels, out_channels,    ║
    ║     stride=1, activation='none')                      ║
    ║  3. If in_channels != out_channels OR stride != 1:    ║
    ║     create a 1x1 conv shortcut (+ BN)                 ║
    ║  4. Final activation after the addition               ║
    ║                                                       ║
    ║  forward(x):                                          ║
    ║  1. identity = shortcut(x) if shortcut else x         ║
    ║  2. out = block1(x) → block2(out)                     ║
    ║  3. out = out + identity                              ║
    ║  4. out = activation(out)                             ║
    ║  5. return out                                         ║
    ║                                                       ║
    ║  Hint: The second ConvBlock has activation='none'     ║
    ║  because we activate AFTER the addition.               ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("TODO: implement ResidualBlock.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement ResidualBlock.forward()")


# ─────────────────────────────────────────────────────
# Plain CNN (no residuals)
# ─────────────────────────────────────────────────────

class PlainCNN(nn.Module):
    """
    A plain CNN built from ConvBlocks (no skip connections).

    Architecture (CIFAR-10):
      ConvBlock(3, 64) → ConvBlock(64, 64) → MaxPool(2)
      ConvBlock(64, 128) → ConvBlock(128, 128) → MaxPool(2)
      ConvBlock(128, 256) → ConvBlock(256, 256) → MaxPool(2)
      AdaptiveAvgPool(1) → Flatten → Linear(256, num_classes)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement using ConvBlock.                     ║
    ║                                                       ║
    ║  Hint: Use nn.Sequential to group stage layers.       ║
    ║  Apply Kaiming init after building.                    ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, num_classes: int = 10, use_batchnorm: bool = True):
        super().__init__()
        raise NotImplementedError("TODO: implement PlainCNN.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement PlainCNN.forward()")


# ─────────────────────────────────────────────────────
# Residual CNN
# ─────────────────────────────────────────────────────

class ResidualCNN(nn.Module):
    """
    A CNN built from ResidualBlocks.

    Architecture (CIFAR-10):
      ConvBlock(3, 64, stride=1)              ← stem
      ResidualBlock(64, 64)                   ← stage 1
      ResidualBlock(64, 128, stride=2)        ← stage 2 (downsample)
      ResidualBlock(128, 256, stride=2)       ← stage 3 (downsample)
      AdaptiveAvgPool(1) → Flatten → Linear(256, num_classes)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement using ResidualBlock + ConvBlock.     ║
    ║                                                       ║
    ║  Hint: The stem is a regular ConvBlock.               ║
    ║  Each stage with stride=2 halves spatial dims.        ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, num_classes: int = 10, use_batchnorm: bool = True):
        super().__init__()
        raise NotImplementedError("TODO: implement ResidualCNN.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement ResidualCNN.forward()")


# ─────────────────────────────────────────────────────
# Receptive Field Calculator (provided)
# ─────────────────────────────────────────────────────

def compute_receptive_field(
    layers: List[dict],
) -> List[dict]:
    """
    Compute the receptive field at each layer.

    Parameters
    ----------
    layers : list of dict
        Each dict: {'name': str, 'kernel': int, 'stride': int, 'padding': int}

    Returns
    -------
    list of dict with added 'rf' (receptive field) and 'jump' keys.

    This is a provided utility — no student implementation needed.
    """
    rf = 1
    jump = 1
    results = []
    for layer in layers:
        k = layer["kernel"]
        s = layer["stride"]
        rf = rf + (k - 1) * jump
        jump = jump * s
        results.append({**layer, "rf": rf, "jump": jump})
    return results
