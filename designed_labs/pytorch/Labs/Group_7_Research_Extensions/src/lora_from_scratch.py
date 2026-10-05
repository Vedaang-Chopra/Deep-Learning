"""
lora_from_scratch.py — LoRA (Low-Rank Adaptation) from First Principles
==========================================================================

Student implements:
  - LoRALinear: low-rank adapter wrapping a frozen Linear layer
  - inject_lora(): dynamically replace Linear layers with LoRA versions
  - merge_lora() / unmerge_lora(): fold adapters into base weights
  - state_dict_adapters_only(): save only LoRA parameters

No PEFT library. Manual implementation of Hu et al. 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Set, Dict


# ─────────────────────────────────────────────────────
# LoRA Linear Layer
# ─────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    A Linear layer with frozen base weights and trainable low-rank adapters.

    Original: y = x @ W0^T + b
    LoRA:     y = x @ W0^T + (x @ A^T @ B^T) * (alpha / r) + b

    Parameters
    ----------
    base_linear : nn.Linear — the original (frozen) linear layer
    r : int — rank of the low-rank decomposition
    alpha : float — scaling factor (default: r)
    dropout : float — dropout on adapter path

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. Store base_linear reference                        ║
    ║  2. Freeze base weights:                               ║
    ║     base_linear.weight.requires_grad_(False)           ║
    ║     if base_linear.bias: bias.requires_grad_(False)    ║
    ║  3. Create trainable A and B:                          ║
    ║     self.lora_A = nn.Parameter(                        ║
    ║       torch.zeros(r, base_linear.in_features))         ║
    ║     self.lora_B = nn.Parameter(                        ║
    ║       torch.zeros(base_linear.out_features, r))        ║
    ║  4. Initialize A with Kaiming uniform                  ║
    ║     (B is init to zero → LoRA starts as identity)      ║
    ║  5. Store scaling = alpha / r                          ║
    ║  6. Optional dropout                                   ║
    ║  7. self.merged = False                                ║
    ║                                                       ║
    ║  forward(x):                                          ║
    ║  1. base_output = self.base_linear(x)                  ║
    ║  2. if self.merged: return base_output                 ║
    ║  3. lora_output = (dropout(x) @ A^T) @ B^T * scaling   ║
    ║  4. return base_output + lora_output                   ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha if alpha is not None else float(r)
        self.merged = False
        raise NotImplementedError("TODO: implement LoRALinear.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement LoRALinear.forward()")

    def merge(self) -> None:
        """
        Fold LoRA weights into the base weight for efficient inference.

        ╔═══════════════════════════════════════════════════════╗
        ║  TODO: Implement.                                     ║
        ║                                                       ║
        ║  W_merged = W0 + B @ A * scaling                      ║
        ║  self.base_linear.weight.data += (B @ A) * scaling     ║
        ║  self.merged = True                                    ║
        ╚═══════════════════════════════════════════════════════╝
        """
        raise NotImplementedError("TODO: implement LoRALinear.merge()")

    def unmerge(self) -> None:
        """
        Unfold LoRA weights from the base weight.

        ╔═══════════════════════════════════════════════════════╗
        ║  TODO: Implement.                                     ║
        ║                                                       ║
        ║  self.base_linear.weight.data -= (B @ A) * scaling     ║
        ║  self.merged = False                                   ║
        ╚═══════════════════════════════════════════════════════╝
        """
        raise NotImplementedError("TODO: implement LoRALinear.unmerge()")


# ─────────────────────────────────────────────────────
# LoRA Injection
# ─────────────────────────────────────────────────────

def inject_lora(
    model: nn.Module,
    target_modules: Set[str],
    r: int = 8,
    alpha: float = None,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Replace target Linear layers with LoRALinear versions.

    Parameters
    ----------
    model : nn.Module
    target_modules : set of str — names of modules to replace
        e.g., {'query', 'value'} or {'fc1', 'fc2'}
    r : int — LoRA rank
    alpha : float — scaling factor
    dropout : float

    Returns
    -------
    model with LoRA injected (modified in-place)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Recursively traverse model's named_modules.           ║
    ║  For each (name, module):                              ║
    ║    if module name matches target_modules AND           ║
    ║       isinstance(module, nn.Linear):                   ║
    ║      Replace with LoRALinear(module, r, alpha, dropout)║
    ║                                                       ║
    ║  Use setattr on parent to replace the child module.    ║
    ║                                                       ║
    ║  Hint: Split name by '.' to get parent path.           ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement inject_lora()")


# ─────────────────────────────────────────────────────
# Merge / Unmerge All
# ─────────────────────────────────────────────────────

def merge_lora(model: nn.Module) -> None:
    """
    Merge all LoRA adapters into base weights for inference.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Iterate over model.modules(), call .merge()    ║
    ║  on each LoRALinear instance.                          ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement merge_lora()")


def unmerge_lora(model: nn.Module) -> None:
    """
    Unmerge all LoRA adapters from base weights.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Iterate and call .unmerge() on each LoRALinear. ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement unmerge_lora()")


# ─────────────────────────────────────────────────────
# Adapter-Only State Dict
# ─────────────────────────────────────────────────────

def state_dict_adapters_only(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters (A, B) from the model state dict.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  Filter model.state_dict() for keys containing        ║
    ║  'lora_A' or 'lora_B'.                                 ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement state_dict_adapters_only()")


# ─────────────────────────────────────────────────────
# Param Count Helper (provided)
# ─────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters. Provided."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": trainable / total * 100 if total > 0 else 0,
    }
