"""
grad_diagnostics.py — Scaffolding for Gradient & Autograd Diagnostics
======================================================================

This module provides:
  - A skeleton class `AutogradDebugger` with method signatures and TODOs.
  - Small logging utilities the student may use.

ALL core logic MUST be implemented by the student.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from collections import OrderedDict
import warnings


# ─────────────────────────────────────────────────────
# Logging Utility (provided)
# ─────────────────────────────────────────────────────

class DiagnosticLog:
    """
    Simple container that accumulates diagnostic records
    and can print a formatted summary.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add(self, tag: str, **kwargs) -> None:
        """Add a diagnostic record."""
        self.records.append({"tag": tag, **kwargs})

    def summarize(self, last_n: Optional[int] = None) -> None:
        """Print a formatted summary of recorded diagnostics."""
        entries = self.records if last_n is None else self.records[-last_n:]
        print(f"\n{'='*60}")
        print(f"  Diagnostic Log — {len(entries)} entries")
        print(f"{'='*60}")
        for i, rec in enumerate(entries):
            tag = rec.pop("tag", "unknown")
            print(f"\n  [{i}] {tag}")
            for k, v in rec.items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.6e}")
                else:
                    print(f"      {k}: {v}")
            rec["tag"] = tag  # restore
        print(f"\n{'='*60}\n")

    def clear(self) -> None:
        self.records.clear()


# ─────────────────────────────────────────────────────
# AutogradDebugger — Skeleton (student implements)
# ─────────────────────────────────────────────────────

class AutogradDebugger:
    """
    A reusable debugger that can be attached to any nn.Module to:
      - Log gradient norms per parameter
      - Log activation means/stds per layer
      - Detect NaN/Inf in gradients and activations
      - Warn about non-contiguous tensors where contiguity matters
      - Optionally clip or flag anomalous gradient magnitudes

    ╔══════════════════════════════════════════════════════╗
    ║  STUDENT: You must implement every method below.     ║
    ║  Follow the docstrings and TODO instructions.        ║
    ╚══════════════════════════════════════════════════════╝
    """

    def __init__(self, model: nn.Module, log: Optional[DiagnosticLog] = None):
        self.model = model
        self.log = log or DiagnosticLog()
        self._forward_hooks = []
        self._backward_hooks = []
        self._activation_stats: OrderedDict = OrderedDict()
        self._gradient_stats: OrderedDict = OrderedDict()

    # ── Attach / Detach ──────────────────────────────

    def attach(self) -> "AutogradDebugger":
        """
        Register forward and backward hooks on every layer of self.model.
        Store hook handles in self._forward_hooks and self._backward_hooks
        so they can be removed later.

        Returns self for chaining.

        TODO:
        - Iterate over self.model.named_modules()
        - For each module, register:
            1. A forward hook that calls self._forward_hook_fn
            2. A full backward hook that calls self._backward_hook_fn
        - Append each hook handle to the respective list.

        Hints:
        - Use module.register_forward_hook(...)
        - Use module.register_full_backward_hook(...)
        - The hook functions are defined below — implement them too.
        """
        raise NotImplementedError("TODO: implement attach()")

    def detach(self) -> None:
        """
        Remove all registered hooks.

        TODO:
        - Iterate over self._forward_hooks and self._backward_hooks
        - Call .remove() on each handle
        - Clear both lists
        """
        raise NotImplementedError("TODO: implement detach()")

    # ── Hook Functions ───────────────────────────────

    def _forward_hook_fn(
        self, module: nn.Module, input: Any, output: Any
    ) -> None:
        """
        Called after every forward pass of a module.

        TODO:
        - Compute mean, std, min, max of the output tensor
        - Check for NaN / Inf values
        - Check if output is contiguous
        - Store stats in self._activation_stats[module_name]
        - Log warnings to self.log if NaN/Inf detected

        Hints:
        - Get module name via next((n for n, m in self.model.named_modules() if m is module), "unknown")
        - Handle the case where output is a tuple (take first element)
        """
        raise NotImplementedError("TODO: implement _forward_hook_fn()")

    def _backward_hook_fn(
        self, module: nn.Module, grad_input: Any, grad_output: Any
    ) -> None:
        """
        Called after backward pass through a module.

        TODO:
        - Extract gradient tensors from grad_output (may be tuple)
        - Compute norm, mean, std, max of each gradient tensor
        - Check for NaN / Inf
        - Store stats in self._gradient_stats[module_name]
        - Log warnings if anomalies detected

        Hints:
        - grad_output is a tuple; some entries may be None
        - Use torch.norm(...) for gradient norms
        """
        raise NotImplementedError("TODO: implement _backward_hook_fn()")

    # ── Reporting ────────────────────────────────────

    def report_gradient_health(self) -> Dict[str, Dict[str, float]]:
        """
        Return a dict mapping layer_name -> {norm, mean, std, max, has_nan, has_inf}
        for each layer's gradients observed in the last backward pass.

        TODO:
        - Return self._gradient_stats (or a processed version)
        - Print a formatted table to stdout

        Hints:
        - Use f-strings with alignment for pretty printing
        - Flag layers with norm > 100 as "EXPLODING"
        - Flag layers with norm < 1e-7 as "VANISHING"
        """
        raise NotImplementedError("TODO: implement report_gradient_health()")

    def report_activation_health(self) -> Dict[str, Dict[str, float]]:
        """
        Return a dict mapping layer_name -> {mean, std, min, max, has_nan, is_contiguous}
        for each layer's activations observed in the last forward pass.

        TODO:
        - Return self._activation_stats (or a processed version)
        - Print a formatted table to stdout
        """
        raise NotImplementedError("TODO: implement report_activation_health()")

    def check_contiguity_warnings(self) -> List[str]:
        """
        Return a list of layer names where activations were non-contiguous.

        TODO:
        - Iterate self._activation_stats
        - Collect names where is_contiguous == False
        - Print warnings
        """
        raise NotImplementedError("TODO: implement check_contiguity_warnings()")

    def reset_stats(self) -> None:
        """Clear accumulated stats between steps."""
        self._activation_stats.clear()
        self._gradient_stats.clear()
