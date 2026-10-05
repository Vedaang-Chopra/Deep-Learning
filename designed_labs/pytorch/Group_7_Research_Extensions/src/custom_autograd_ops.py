"""
custom_autograd_ops.py — Custom Autograd Functions
=====================================================

Student implements:
  - SwishFn: Swish activation as autograd.Function with manual backward
  - GradientReversalFn: gradient reversal layer for domain adaptation
  - StraightThroughEstimator: STE for discrete operations
  - gradient_check(): finite-difference gradient verification

The fundamental research skill: translating math → correct backward pass.
"""

import torch
import torch.nn as nn
from torch.autograd import Function, gradcheck
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────
# Swish / SiLU Activation (Custom Autograd)
# ─────────────────────────────────────────────────────

class SwishFn(Function):
    """
    Swish activation: f(x) = x · σ(x)

    Forward:  y = x · sigmoid(x)
    Backward: dy/dx = sigmoid(x) + x · sigmoid(x) · (1 - sigmoid(x))
                    = sigmoid(x) · (1 + x · (1 - sigmoid(x)))
                    = y + sigmoid(x) · (1 - y)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement forward and backward.                ║
    ║                                                       ║
    ║  forward(ctx, x):                                     ║
    ║    1. Compute sigmoid_x = torch.sigmoid(x)             ║
    ║    2. Compute y = x * sigmoid_x                        ║
    ║    3. ctx.save_for_backward(x, sigmoid_x)              ║
    ║    4. Return y                                         ║
    ║                                                       ║
    ║  backward(ctx, grad_output):                           ║
    ║    1. Retrieve x, sigmoid_x from ctx                   ║
    ║    2. y = x * sigmoid_x                                ║
    ║    3. grad_x = grad_output * (y + sigmoid_x * (1 - y)) ║
    ║    4. Return (grad_x,)                                 ║
    ║                                                       ║
    ║  IMPORTANT: backward must return a tuple with one      ║
    ║  gradient per forward input. Use None for non-tensor.  ║
    ╚═══════════════════════════════════════════════════════╝
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement SwishFn.forward()")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        raise NotImplementedError("TODO: implement SwishFn.backward()")


class Swish(nn.Module):
    """nn.Module wrapper for SwishFn."""
    def forward(self, x):
        return SwishFn.apply(x)


# ─────────────────────────────────────────────────────
# Gradient Reversal Layer
# ─────────────────────────────────────────────────────

class GradientReversalFn(Function):
    """
    Gradient Reversal Layer (Ganin et al., 2015)

    Forward:  identity — passes input through unchanged
    Backward: negates the gradient — reverses gradient direction

    Used in domain adaptation: the feature extractor learns domain-invariant
    features by adversarially training against a domain classifier.

    Forward:  y = x
    Backward: dy/dx = -λ · grad_output

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement forward and backward.                ║
    ║                                                       ║
    ║  forward(ctx, x, lambda_):                             ║
    ║    1. ctx.lambda_ = lambda_                            ║
    ║    2. Return x (identity)                              ║
    ║                                                       ║
    ║  backward(ctx, grad_output):                           ║
    ║    1. Return (-ctx.lambda_ * grad_output, None)        ║
    ║       ↑ Two returns: one per forward arg               ║
    ║       ↑ None for lambda_ (scalar, no gradient)         ║
    ╚═══════════════════════════════════════════════════════╝
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        raise NotImplementedError("TODO: implement GradientReversalFn.forward()")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("TODO: implement GradientReversalFn.backward()")


class GradientReversalLayer(nn.Module):
    """nn.Module wrapper for GradientReversalFn."""
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambda_)


# ─────────────────────────────────────────────────────
# Straight-Through Estimator
# ─────────────────────────────────────────────────────

class StraightThroughEstimatorFn(Function):
    """
    Straight-Through Estimator (STE) for discrete/quantized operations.

    Forward: apply a non-differentiable operation (e.g., rounding, sign)
    Backward: pass gradient through as if the op were identity

    Used in: binary neural networks, quantization-aware training, VQ-VAE.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement forward and backward.                ║
    ║                                                       ║
    ║  forward(ctx, x):                                     ║
    ║    1. Return x.sign()  (or torch.round(x))            ║
    ║       — the discrete operation                        ║
    ║                                                       ║
    ║  backward(ctx, grad_output):                           ║
    ║    1. Return (grad_output,)                            ║
    ║       — pass gradient through unchanged (STE)          ║
    ╚═══════════════════════════════════════════════════════╝
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement STE forward()")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("TODO: implement STE backward()")


# ─────────────────────────────────────────────────────
# Gradient Checking Utility
# ─────────────────────────────────────────────────────

def gradient_check(
    fn,
    inputs: Tuple[torch.Tensor, ...],
    eps_values: list = None,
) -> dict:
    """
    Verify custom autograd function gradients using finite differences.

    Parameters
    ----------
    fn : callable — the autograd Function.apply or an nn.Module
    inputs : tuple of tensors (must have requires_grad=True)
    eps_values : list of epsilon values to test

    Returns
    -------
    dict with:
      'max_error' : float — max absolute error across all elements
      'passed' : bool — max_error < 1e-5
      'errors_by_eps' : dict mapping eps -> max_error

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  For each input tensor x and each element x_i:        ║
    ║  1. Compute f(x_i + eps) and f(x_i - eps)             ║
    ║  2. Numerical gradient = (f(x+eps) - f(x-eps)) / 2eps ║
    ║  3. Analytical gradient = autograd backward             ║
    ║  4. Error = |numerical - analytical|                   ║
    ║                                                       ║
    ║  OR simply use torch.autograd.gradcheck:               ║
    ║    passed = gradcheck(fn, inputs, eps=eps, atol=1e-5)  ║
    ║                                                       ║
    ║  Also compute errors for multiple eps values to show   ║
    ║  the convergence of finite differences.                ║
    ╚═══════════════════════════════════════════════════════╝
    """
    if eps_values is None:
        eps_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    raise NotImplementedError("TODO: implement gradient_check()")
