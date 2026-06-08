"""
parity_tests.py — Correctness / Parity Testing Across Runtimes
=================================================================

Student implements:
  - compare_outputs(): element-wise comparison with tolerance
  - parity_check_model(): compare two models on real data
  - assert_close_or_report(): detailed diff report

Correctness first — before ANY benchmark, prove parity.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np


# ─────────────────────────────────────────────────────
# Output Comparison
# ─────────────────────────────────────────────────────

def compare_outputs(
    y_ref: torch.Tensor,
    y_test: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Dict:
    """
    Compare two output tensors element-wise.

    Parameters
    ----------
    y_ref : reference output (ground truth)
    y_test : output to test
    atol : absolute tolerance
    rtol : relative tolerance

    Returns
    -------
    dict with:
      'max_abs_diff', 'mean_abs_diff', 'max_rel_diff',
      'num_mismatches', 'total_elements',
      'worst_index', 'passed'

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. abs_diff = (y_ref - y_test).abs()                  ║
    ║  2. max_abs_diff = abs_diff.max().item()                ║
    ║  3. mean_abs_diff = abs_diff.mean().item()              ║
    ║  4. rel_diff = abs_diff / (y_ref.abs() + 1e-8)         ║
    ║  5. max_rel_diff = rel_diff.max().item()                ║
    ║  6. mismatches = abs_diff > (atol + rtol * y_ref.abs()) ║
    ║  7. worst_index = unravel_index of abs_diff.argmax()    ║
    ║  8. passed = num_mismatches == 0                        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compare_outputs()")


def parity_check_model(
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    n_batches: int = 5,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Dict:
    """
    Compare two models by running them on real data batches.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Set both models to eval mode                       ║
    ║  2. For n_batches:                                     ║
    ║     a. Get batch from dataloader                       ║
    ║     b. Run both models (with torch.no_grad())          ║
    ║     c. compare_outputs() on the two results            ║
    ║     d. Accumulate stats                                ║
    ║  3. Return summary:                                    ║
    ║     overall max_diff, mean_diff, all_passed, per_batch ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement parity_check_model()")


def assert_close_or_report(
    y_ref: torch.Tensor,
    y_test: torch.Tensor,
    label: str = "parity check",
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """
    Assert outputs are close, printing detailed report if not.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  1. result = compare_outputs(y_ref, y_test, atol, rtol)║
    ║  2. Print formatted report:                            ║
    ║     [PASS/FAIL] label                                  ║
    ║     max abs diff: ...                                  ║
    ║     mean abs diff: ...                                 ║
    ║     worst index: ...                                   ║
    ║     mismatches: .../...                                 ║
    ║  3. Return result['passed']                            ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement assert_close_or_report()")


# ─────────────────────────────────────────────────────
# Report Table (provided)
# ─────────────────────────────────────────────────────

def print_parity_report(results: Dict[str, Dict]):
    """
    Print a formatted parity report comparing multiple runtimes.
    Provided utility.

    Parameters
    ----------
    results : dict mapping runtime_name -> compare_outputs result
    """
    print(f"\n{'─'*70}")
    print(f"{'Runtime':<25} {'Max Diff':>12} {'Mean Diff':>12} {'Status':>10}")
    print(f"{'─'*70}")
    for name, r in results.items():
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"{name:<25} {r['max_abs_diff']:>12.2e} {r['mean_abs_diff']:>12.2e} {status:>10}")
    print(f"{'─'*70}\n")
