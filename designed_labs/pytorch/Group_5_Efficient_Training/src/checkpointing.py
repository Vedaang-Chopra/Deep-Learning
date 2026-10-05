"""
checkpointing.py — Activation Checkpointing
==============================================

Student implements:
  - apply_checkpointing(): apply torch.utils.checkpoint to model blocks
  - CheckpointedSequential: sequential container with checkpointing

Trade compute for memory: recompute activations during backward.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Callable, List, Optional


# ─────────────────────────────────────────────────────
# Apply Checkpointing
# ─────────────────────────────────────────────────────

def apply_checkpointing(
    model: nn.Module,
    policy: str = "all",
    block_class: type = None,
) -> nn.Module:
    """
    Apply activation checkpointing to a model's submodules.

    Parameters
    ----------
    model : nn.Module
    policy : str
        'all' — checkpoint every block matching block_class
        'every_other' — checkpoint every other block
        'none' — no checkpointing (baseline)
    block_class : type (optional)
        The class of blocks to checkpoint (e.g., TransformerBlock, ResidualBlock).
        If None, attempts to checkpoint all nn.Module children.

    Returns
    -------
    model with checkpointing applied (modified in-place)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Strategy:                                            ║
    ║  1. Find all submodules matching block_class           ║
    ║  2. Based on policy, decide which to checkpoint        ║
    ║  3. Wrap their forward methods with                    ║
    ║     torch.utils.checkpoint.checkpoint                  ║
    ║                                                       ║
    ║  Implementation approach:                              ║
    ║  For each selected module, replace its forward:        ║
    ║                                                       ║
    ║  original_forward = module.forward                     ║
    ║  def checkpointed_forward(*args, **kwargs):            ║
    ║      return checkpoint(original_forward, *args,        ║
    ║                        use_reentrant=False, **kwargs)  ║
    ║  module.forward = checkpointed_forward                 ║
    ║                                                       ║
    ║  Note: use_reentrant=False is the recommended mode.    ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement apply_checkpointing()")


# ─────────────────────────────────────────────────────
# Checkpointed Sequential
# ─────────────────────────────────────────────────────

class CheckpointedSequential(nn.Module):
    """
    A sequential container that applies activation checkpointing
    to selected layers during the forward pass.

    Parameters
    ----------
    layers : list of nn.Module
    checkpoint_every : int (default 1)
        Checkpoint every N-th layer. 1 = all, 2 = every other, etc.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  forward(x):                                          ║
    ║  For each layer i:                                     ║
    ║    if (i % checkpoint_every == 0):                      ║
    ║      x = checkpoint(layer, x, use_reentrant=False)     ║
    ║    else:                                               ║
    ║      x = layer(x)                                      ║
    ║  return x                                              ║
    ║                                                       ║
    ║  Note: checkpoint requires inputs with                 ║
    ║  requires_grad=True. If x doesn't, this will fail.    ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, layers: List[nn.Module], checkpoint_every: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_every = checkpoint_every

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement CheckpointedSequential.forward()")


# ─────────────────────────────────────────────────────
# Memory Savings Estimator (provided)
# ─────────────────────────────────────────────────────

def estimate_checkpoint_savings(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
) -> dict:
    """
    Estimate memory savings from activation checkpointing by
    comparing a forward pass with and without checkpoint.
    Provided utility.
    """
    import gc

    # Without checkpointing
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    x = torch.randn(*input_shape, device=device, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    peak_no_ckpt = torch.cuda.max_memory_allocated(device) / 1024**2

    del x, out, loss
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "peak_no_checkpoint_mb": peak_no_ckpt,
        "note": "Run again with checkpointing to compare",
    }
