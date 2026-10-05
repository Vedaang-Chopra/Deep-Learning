#!/usr/bin/env python3
"""Generate the Group 1 Tensor Internals & Autograd Lab notebook."""
import json, os

def md(source):
    """Create a markdown cell."""
    if isinstance(source, str):
        source = source.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in source[:-1]] + [source[-1]]}

def code(source):
    """Create a code cell."""
    if isinstance(source, str):
        source = source.split("\n")
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in source[:-1]] + [source[-1]], "execution_count": None, "outputs": []}

cells = []

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
cells.append(md("""\
# 🧠 Notebook 1 — Tensor Internals, Memory Layout & Autograd Graph Mechanics

**Group 1 — Tensor Internals & Autograd Mastery**

---

### 🎯 Learning Objectives

By the end of this lab you will:

1. Understand tensor **storage vs metadata** separation
2. Master **sizes, strides, and contiguity** — and know exactly when `.view()` fails
3. Understand **autograd DAG construction** — leaf vs non-leaf, `grad_fn` chains
4. Debug **gradient accumulation** issues and know when `retain_graph=True` is needed
5. Use **forward & backward hooks** to inspect activations and gradients in any model
6. Understand how **in-place operations break autograd**
7. Build a reusable **Tensor Autograd Debugger** tool

### ⚙️ Prerequisites

- PyTorch ≥ 2.0
- matplotlib, numpy
- Basic familiarity with neural network forward/backward passes

### 📂 File Structure

```
Group_1_Tensor_Autograd/
├── notebooks/
│   └── 01_tensor_autograd_lab.ipynb   ← you are here
└── src/
    ├── tensor_utils.py                ← plotting & benchmarking helpers (provided)
    └── grad_diagnostics.py            ← AutogradDebugger skeleton (you implement)
```

> ⚠️ **Rule**: You implement ALL core logic. The `src/` files only provide plotting wrappers and class skeletons."""))

# ═══════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════
cells.append(md("## 0 — Environment Setup"))

cells.append(code("""\
import sys, os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Add src/ to path
sys.path.insert(0, os.path.join(os.pardir, "src"))
from tensor_utils import benchmark_fn, plot_benchmark_comparison, plot_gradient_norms
from tensor_utils import plot_activation_distributions, plot_training_curves
from grad_diagnostics import DiagnosticLog, AutogradDebugger

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
print(f"Device          : {'cuda' if torch.cuda.is_available() else 'cpu'}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")"""))

# ═══════════════════════════════════════════════════════
# SECTION 1 — TENSOR STORAGE & STRIDES
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 1 — Tensor Storage & Strides

## 1.1 Conceptual Background

### Storage vs Tensor

A PyTorch `Tensor` is **not** the data itself. It is a **view** into contiguous memory called `Storage`.

```
Storage (1D contiguous block of memory)
┌────┬────┬────┬────┬────┬────┐
│ a₀ │ a₁ │ a₂ │ a₃ │ a₄ │ a₅ │
└────┴────┴────┴────┴────┴────┘
       ↑
       Tensor metadata:
         • shape   = (2, 3)
         • stride  = (3, 1)
         • offset  = 0
```

**Key insight**: Multiple tensors can share the *same* storage with different shapes, strides, and offsets. This is why `transpose()`, `permute()`, and slicing are O(1) — they just change metadata.

### Stride Mechanics

The **stride** of a tensor tells you how many elements in the underlying storage you must skip to move to the next element along each dimension.

For a `(2, 3)` tensor with row-major layout:
- Stride = `(3, 1)` → skip 3 elements to move to the next row, 1 to move to the next column.

After `transpose(0, 1)` → shape becomes `(3, 2)`, stride becomes `(1, 3)`.  
The storage is **unchanged** — only metadata changed.

### Contiguous vs Non-Contiguous

A tensor is **contiguous** if its elements are stored in row-major (C) order in memory — i.e., the stride is a *decreasing sequence consistent with the shape*.

After `permute()` or `transpose()`, the tensor is typically **non-contiguous** because the strides no longer match a row-major layout.

### When `.contiguous()` Is Required

- `.view()` **requires** contiguous memory (it cannot rearrange physical layout)
- `.reshape()` will silently copy if needed (prefer `.view()` for explicitness)
- Some CUDA kernels are optimized for contiguous memory
- `.contiguous()` allocates new memory and copies — it has a cost

### Why `.view()` Fails After `transpose()`

`view()` requires the tensor to be contiguous because it only reinterprets the shape metadata without touching memory. After `transpose()`, strides are rearranged and the tensor is no longer contiguous — so `view()` cannot guarantee a valid memory reinterpretation."""))

# ── Task 1.2: inspect_tensor ──
cells.append(md("""\
## 1.2 Implementation Task — `inspect_tensor()`

Implement a function that prints a comprehensive diagnostic of any tensor.

**Requirements**:
- Print: `shape`, `stride`, `storage_offset`, `is_contiguous`, `dtype`, `device`
- Print: number of elements, storage size, whether storage is shared with another tensor
- Return a dict of these values for programmatic use

**Hints**:
- `tensor.storage_offset()` gives the offset into the storage
- `tensor.storage()` gives the underlying storage object
- `tensor.data_ptr()` gives the memory address — useful for checking if two tensors share storage"""))

cells.append(code("""\
def inspect_tensor(t: torch.Tensor) -> dict:
    \"\"\"
    Print and return a comprehensive diagnostic of a tensor.

    Must print:
      - shape, stride, storage_offset
      - is_contiguous, dtype, device
      - numel, storage size
      - data_ptr (memory address)

    Returns:
      dict with all the above as key-value pairs.
    \"\"\"
    # ╔═══════════════════════════════════════════╗
    # ║  TODO: Implement this function            ║
    # ║                                           ║
    # ║  Hint: Use t.shape, t.stride(),           ║
    # ║        t.storage_offset(), t.is_contiguous ║
    # ║        t.dtype, t.device, t.numel(),      ║
    # ║        t.storage().size(), t.data_ptr()   ║
    # ║                                           ║
    # ║  Format output as a clean table.          ║
    # ╚═══════════════════════════════════════════╝
    raise NotImplementedError("Implement inspect_tensor")"""))

# ── Task 1.3: Create and inspect various tensors ──
cells.append(md("""\
## 1.3 Exploration — Creating Views & Checking Strides

Create the tensors described below and run `inspect_tensor()` on each.
Pay attention to how strides change and whether storage is shared.

**Tasks**:
1. Create a `(3, 4)` tensor `A`
2. Create `B = A.T` (transpose)
3. Create `C = A[1:]` (row slice)
4. Create `D = A.permute(1, 0)` — verify strides match `B`
5. Create `E = A.expand(2, 3, 4)` — note the stride of the new dimension
6. Create `F` using `torch.as_strided()` to extract the diagonal of `A`

For each, answer:
- Does it share storage with `A`? (Check `data_ptr`)
- Is it contiguous?
- Will `.view(-1)` work on it?"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO: Create tensors A through F         ║
# ║  Call inspect_tensor() on each             ║
# ║  Answer the questions in comments          ║
# ╚═══════════════════════════════════════════╝

A = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("=== A (original) ===")
# inspect_tensor(A)

# B = ...
# C = ...
# D = ...
# E = ...
# F = ... (use torch.as_strided — be careful!)

# ⚠️  as_strided WARNING:
# This is a low-level function. Wrong strides can read garbage memory.
# Only use it when you fully understand what you're doing."""))

# ── Task 1.4: View failure demo ──
cells.append(md("""\
## 1.4 Experiment — When `.view()` Fails

Demonstrate that `.view()` fails on a non-contiguous tensor and explain why."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO:                                     ║
# ║  1. Create a tensor and transpose it       ║
# ║  2. Try .view(-1) and catch the error      ║
# ║  3. Show that .contiguous().view(-1) works ║
# ║  4. Show that .reshape(-1) works too       ║
# ║  5. Explain the difference in comments     ║
# ╚═══════════════════════════════════════════╝
"""))

# ── Task 1.5: Benchmarking ──
cells.append(md("""\
## 1.5 Benchmark — Contiguous vs Non-Contiguous MatMul

Compare the performance of matrix multiplication on:
1. A contiguous tensor pair
2. A non-contiguous (transposed) tensor pair

Use the provided `benchmark_fn()` and `plot_benchmark_comparison()` helpers from `tensor_utils.py`.

**Expected observation**: Non-contiguous tensors may be slower because CUDA kernels prefer contiguous memory layouts."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO:                                     ║
# ║  1. Create a large square tensor (e.g.     ║
# ║     1024x1024)                              ║
# ║  2. Create a contiguous pair for matmul    ║
# ║  3. Create a non-contiguous pair           ║
# ║     (transpose one operand)                ║
# ║  4. Benchmark both using benchmark_fn()    ║
# ║  5. Plot with plot_benchmark_comparison()  ║
# ╚═══════════════════════════════════════════╝

# Example usage of benchmark_fn:
# result = benchmark_fn(torch.matmul, A, B, warmup=5, repeats=50)
# result is a dict: {mean_ms, std_ms, min_ms, max_ms, times_ms}
"""))

# ── Reflection 1 ──
cells.append(md("""\
## 1.6 Reflection Questions

Answer these in your own words (add markdown cells below):

1. **Why does `permute()` not allocate new memory?**
   What does it change instead? What are the performance implications?

2. **Why does `.view()` fail after `.transpose()`?**
   What specific condition does `.view()` check, and why can't it proceed?

3. **When is calling `.contiguous()` expensive?**
   Under what circumstances does it actually need to copy data?

4. **Two tensors have the same shape but different strides. Do they contain the same data?**
   Give a concrete example.

5. **What is the stride of a scalar tensor? What about a 1-element tensor?**"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO: Write your answers as comments     ║
# ║  or create new markdown cells below       ║
# ╚═══════════════════════════════════════════╝
"""))

# ═══════════════════════════════════════════════════════
# SECTION 2 — AUTOGRAD GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 2 — Autograd Graph Construction

## 2.1 Conceptual Background

### Dynamic Computational Graph (DAG)

PyTorch builds a **directed acyclic graph** (DAG) *on the fly* during the forward pass. Each operation creates a node (`grad_fn`) that records:

- What function was applied
- What inputs it received
- How to compute the local Jacobian (for backward)

```
        x (leaf, requires_grad=True)
        │
    ┌───▼───┐
    │ MulBackward │    ← created by: y = x * 2
    └───┬───┘
        │
    ┌───▼───┐
    │ AddBackward │    ← created by: z = y + 3
    └───┬───┘
        │
        z.backward()  → traverses DAG in reverse
```

### Leaf vs Non-Leaf Tensors

| Property | Leaf Tensor | Non-Leaf Tensor |
|----------|------------|-----------------|
| Created by | user, or `torch.tensor()` | result of an operation |
| `is_leaf` | `True` | `False` |
| `grad_fn` | `None` | points to creating op |
| `.grad` after backward | ✅ stored | ❌ not stored (unless `retain_grad()`) |

**Why?** Intermediate gradients are not stored by default to save memory. Only leaf parameter gradients matter for optimization.

### Gradient Accumulation

Gradients **accumulate** by default in `.grad`. If you call `backward()` twice without `zero_grad()`, gradients double. This is a feature (used in gradient accumulation for large batches) but a common source of bugs.

### In-Place Operations & Autograd

In-place operations (e.g., `x.add_(1)`) modify data in place. This is dangerous because:
- It invalidates saved tensors needed for backward
- PyTorch uses a **version counter** to detect this and raises `RuntimeError`
- Never do in-place ops on tensors that require grad *after* they participate in a computation"""))

# ── Task 2.2: visualize_graph ──
cells.append(md("""\
## 2.2 Implementation Task — `visualize_graph()`

Write a function that traverses the `grad_fn` chain of a tensor and prints the computational graph.

**Requirements**:
- Start from the tensor's `grad_fn`
- Recursively visit `next_functions`
- Print the DAG with indentation showing depth
- Handle leaf nodes (where `grad_fn` is `None`)"""))

cells.append(code("""\
def visualize_graph(tensor: torch.Tensor, indent: int = 0) -> None:
    \"\"\"
    Recursively traverse and print the autograd graph rooted at `tensor`.

    Example output:
        AddBackward0
          MulBackward0
            AccumulateGrad (leaf: x)
          AccumulateGrad (leaf: y)

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor whose grad_fn chain to traverse.
    indent : int
        Current indentation level (for recursion).

    TODO:
    - Check if tensor has grad_fn; if not, print 'Leaf tensor' and return
    - Print the grad_fn class name with proper indentation
    - Iterate over grad_fn.next_functions
    - Each entry is a tuple (fn, output_nr)
    - If fn is None, skip
    - If fn is AccumulateGrad, print as leaf
    - Otherwise, recurse

    Hint: grad_fn.next_functions returns child nodes in the DAG.
    \"\"\"
    # ╔═══════════════════════════════════════════╗
    # ║  TODO: Implement this function            ║
    # ╚═══════════════════════════════════════════╝
    raise NotImplementedError("Implement visualize_graph")"""))

# ── Task 2.3: Leaf vs non-leaf ──
cells.append(md("""\
## 2.3 Experiment — Leaf vs Non-Leaf Tensors

Demonstrate the difference between leaf and non-leaf tensors.

**Tasks**:
1. Create a leaf tensor `x` with `requires_grad=True`
2. Compute `y = x * 2 + 1`
3. Print `is_leaf`, `grad_fn`, `requires_grad` for both
4. Call `y.backward()` (with appropriate scalar if needed)
5. Show that `x.grad` is populated but `y.grad` is `None`
6. Use `retain_grad()` on `y` and repeat — show `y.grad` is now populated"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO: Demonstrate leaf vs non-leaf       ║
# ║                                           ║
# ║  Step 1: x = torch.tensor(...,            ║
# ║            requires_grad=True)             ║
# ║  Step 2: y = x * 2 + 1                    ║
# ║  Step 3: Check .is_leaf, .grad_fn,        ║
# ║          .requires_grad                    ║
# ║  Step 4: backward and check .grad         ║
# ║  Step 5: Redo with retain_grad()          ║
# ╚═══════════════════════════════════════════╝
"""))

# ── Task 2.4: Gradient accumulation bug ──
cells.append(md("""\
## 2.4 Experiment — Gradient Accumulation Bug

Demonstrate one of the most common PyTorch bugs: forgetting to zero gradients.

**Tasks**:
1. Create a simple computation: `loss = (x * w).sum()`
2. Call `loss.backward()` **three times** without zeroing gradients
3. Print `w.grad` after each call — observe it accumulates!
4. Show the correct pattern: `w.grad.zero_()` before each backward
5. Show the optimizer pattern: `optimizer.zero_grad()`"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO: Demonstrate gradient accumulation   ║
# ║                                           ║
# ║  Show the BUG:                             ║
# ║    Loop 3 times: compute loss, backward,  ║
# ║    print w.grad (watch it grow!)           ║
# ║                                           ║
# ║  Show the FIX:                             ║
# ║    Loop 3 times: zero_grad, compute loss, ║
# ║    backward, print w.grad (stays correct) ║
# ╚═══════════════════════════════════════════╝
"""))

# ── Task 2.5: In-place ops ──
cells.append(md("""\
## 2.5 Experiment — In-Place Operations Break Autograd

Show how in-place operations can corrupt the computational graph.

**Tasks**:
1. Create `x` with `requires_grad=True` and compute `y = x * 2`
2. Do an in-place op on `y`: `y.add_(1)`
3. Try `y.backward()` — observe the `RuntimeError`
4. Explain *why* this fails (version counter mechanism)
5. Show the safe alternative: `y = y + 1` (creates new tensor)"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO: Demonstrate in-place op failure     ║
# ║                                           ║
# ║  1. x = torch.tensor([1.0, 2.0, 3.0],    ║
# ║         requires_grad=True)                ║
# ║  2. y = x * 2                             ║
# ║  3. y.add_(1)   # in-place!               ║
# ║  4. Try y.sum().backward() in try/except  ║
# ║  5. Show the version counter:             ║
# ║     print(y._version)                     ║
# ╚═══════════════════════════════════════════╝
"""))

# ── Reflection 2 ──
cells.append(md("""\
## 2.6 Reflection Questions

1. **Why don't intermediate tensors store gradients by default?**
   What memory savings does this provide in a deep network?

2. **When should you use `retain_graph=True`?**
   Give a concrete use case. What happens if you don't use it but call backward twice?

3. **What is the relationship between `grad_fn` and `next_functions`?**
   Draw the graph for `z = (a * b) + (c * d)` where a, b, c, d are leaf tensors.

4. **Why does PyTorch use a dynamic graph instead of a static one (like TensorFlow 1.x)?**
   What are the tradeoffs?"""))

cells.append(code("# TODO: Write your answers\n"))

# ═══════════════════════════════════════════════════════
# SECTION 3 — GRADIENT DEBUGGING & STABILITY
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 3 — Gradient Debugging & Stability

## 3.1 Conceptual Background

### Vanishing Gradients
In deep networks, gradients shrink exponentially as they propagate backward through many layers. This happens when:
- Activation functions saturate (sigmoid, tanh in extreme regions)
- Weight initialization is too small
- Many multiplicative operations chain together

**Symptom**: Early layers learn very slowly or not at all.

### Exploding Gradients
The opposite: gradients grow exponentially. Causes:
- Weight initialization too large
- No gradient clipping
- Unstable architectures (deep vanilla RNNs)

**Symptom**: Loss becomes NaN, weights blow up.

### NaN Debugging Checklist
1. Check for division by zero
2. Check for `log(0)` or `log(negative)`
3. Check for `sqrt(0)` gradient (undefined)
4. Check learning rate (too high?)
5. Check data (NaN in inputs?)
6. Use `torch.autograd.set_detect_anomaly(True)` (slow but finds the op)

### Gradient Clipping Strategies
- **Clip by norm**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`
- **Clip by value**: `torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)`
- Norm clipping is generally preferred (preserves gradient direction)"""))

# ── Task 3.2: gradient_stats ──
cells.append(md("""\
## 3.2 Implementation Task — `gradient_stats()`

Write a function that computes and prints gradient statistics for every parameter in a model.

**Requirements**:
- For each named parameter with a gradient, compute: mean, std, max, min, norm, has_nan, has_inf
- Print a formatted table
- Return a dict of stats"""))

cells.append(code("""\
def gradient_stats(model: nn.Module) -> dict:
    \"\"\"
    Compute and print gradient statistics for all model parameters.

    For each parameter with .grad not None, compute:
      - mean, std, min, max of the gradient
      - L2 norm of the gradient
      - Whether gradient contains NaN or Inf

    Print a formatted table and return a dict:
      {param_name: {mean, std, min, max, norm, has_nan, has_inf}}

    TODO:
    - Iterate model.named_parameters()
    - Skip params with grad == None
    - Compute all stats using torch operations
    - Format and print a clean table
    - Flag anomalies (NaN, Inf, very large norms)
    \"\"\"
    # ╔═══════════════════════════════════════════╗
    # ║  TODO: Implement this function            ║
    # ╚═══════════════════════════════════════════╝
    raise NotImplementedError("Implement gradient_stats")"""))

# ── Task 3.3: Build an MLP and break it ──
cells.append(md("""\
## 3.3 Experiment — Intentionally Creating Gradient Pathologies

Build a simple MLP and intentionally create:
1. **Exploding gradients** — using large weight initialization
2. **Vanishing gradients** — using sigmoid activations with improper init

Then fix them using:
- Proper initialization (Xavier/Kaiming)
- Gradient clipping"""))

cells.append(code("""\
class TinyMLP(nn.Module):
    \"\"\"
    A simple multi-layer perceptron for gradient experiments.

    TODO:
    - Build a network with 8-10 linear layers (deeper = more dramatic effects)
    - Use configurable activation function
    - Use configurable initialization

    Architecture suggestion:
      Input(32) → Linear(32,64) → Act → Linear(64,64) → Act → ... → Linear(64,1)
    \"\"\"
    def __init__(self, n_layers=8, activation='relu', init_scale=1.0):
        super().__init__()
        # ╔═══════════════════════════════════════════╗
        # ║  TODO: Build the network                   ║
        # ║                                           ║
        # ║  Hints:                                    ║
        # ║  - Use nn.ModuleList for layers            ║
        # ║  - Initialize weights with:                ║
        # ║      nn.init.normal_(w, std=init_scale)    ║
        # ║  - Activation: 'relu', 'sigmoid', 'tanh'  ║
        # ╚═══════════════════════════════════════════╝
        raise NotImplementedError

    def forward(self, x):
        # ╔═══════════════════════════════════════════╗
        # ║  TODO: Forward pass through all layers     ║
        # ╚═══════════════════════════════════════════╝
        raise NotImplementedError"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 1: Exploding Gradients                        ║
# ║                                                           ║
# ║  TODO:                                                     ║
# ║  1. Create TinyMLP with init_scale=5.0, activation='relu' ║
# ║  2. Forward a random batch: x = torch.randn(16, 32)       ║
# ║  3. Compute loss = output.sum()                            ║
# ║  4. Backward                                               ║
# ║  5. Call gradient_stats(model)                             ║
# ║  6. Observe the gradient norms                             ║
# ║  7. Plot with plot_gradient_norms()                        ║
# ╚═══════════════════════════════════════════════════════════╝
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 2: Vanishing Gradients                        ║
# ║                                                           ║
# ║  TODO:                                                     ║
# ║  1. Create TinyMLP with init_scale=0.01,                  ║
# ║     activation='sigmoid'                                   ║
# ║  2. Same forward/backward procedure                        ║
# ║  3. Call gradient_stats(model)                             ║
# ║  4. Observe the tiny gradient norms in early layers        ║
# ║  5. Plot with plot_gradient_norms()                        ║
# ╚═══════════════════════════════════════════════════════════╝
"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 3: Fix with Proper Init + Clipping            ║
# ║                                                           ║
# ║  TODO:                                                     ║
# ║  1. Create TinyMLP, use nn.init.kaiming_normal_           ║
# ║  2. Forward/backward                                       ║
# ║  3. Apply gradient clipping:                               ║
# ║     torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  ║
# ║  4. Compare gradient stats before and after clipping       ║
# ║  5. Plot both for comparison                               ║
# ╚═══════════════════════════════════════════════════════════╝
"""))

# ── Visualization 3.4 ──
cells.append(md("""\
## 3.4 Visualization — Gradient Norm per Layer

Using the experiments above, create a side-by-side comparison plot showing
gradient norms for each scenario:
1. Exploding (large init)
2. Vanishing (small init + sigmoid)
3. Healthy (Kaiming init + ReLU + clipping)"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO: Create a 1x3 subplot figure       ║
# ║  showing gradient norms for each case     ║
# ║  Use plot_gradient_norms() or custom plot ║
# ╚═══════════════════════════════════════════╝
"""))

# ═══════════════════════════════════════════════════════
# SECTION 4 — HOOKS & DEEP INSPECTION
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 4 — Hooks & Deep Inspection

## 4.1 Conceptual Background

### Forward Hooks

A **forward hook** is called after a module's `forward()` method completes.

```python
def hook_fn(module, input, output):
    # input is a tuple of inputs to the module
    # output is the module's output
    pass

handle = module.register_forward_hook(hook_fn)
```

**Use cases**: Logging activations, feature extraction, debugging shapes.

### Backward Hooks

A **backward hook** (specifically `register_full_backward_hook`) is called after the backward pass through a module.

```python
def hook_fn(module, grad_input, grad_output):
    # grad_input: gradients w.r.t. module inputs
    # grad_output: gradients w.r.t. module outputs
    pass

handle = module.register_full_backward_hook(hook_fn)
```

**Use cases**: Gradient inspection, gradient modification, gradient clipping per layer.

### Important Details

- Hooks return **handles** — call `handle.remove()` to unregister
- Forward hooks can **modify** the output by returning a value
- Backward hooks can **modify** gradients by returning modified grad_input
- Always clean up hooks after use to avoid memory leaks
- Use `register_full_backward_hook` (not the deprecated `register_backward_hook`)"""))

# ── Task 4.2: Forward hook ──
cells.append(md("""\
## 4.2 Implementation Task — Forward Hook for Activation Logging

Register forward hooks on every layer of a model to log activation statistics.

**Requirements**:
- For each layer, record: mean, std, min, max of the output
- Store results in a dictionary keyed by layer name
- Print a summary table after a forward pass"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════════╗
# ║  TODO: Implement activation logging with forward hooks     ║
# ║                                                           ║
# ║  Steps:                                                    ║
# ║  1. Create a dict to store activation stats                ║
# ║  2. Define a hook function that computes mean/std/min/max  ║
# ║     of the output and stores in the dict                   ║
# ║  3. Register the hook on every layer of the model          ║
# ║  4. Run a forward pass                                     ║
# ║  5. Print the activation stats table                       ║
# ║  6. Plot with plot_activation_distributions()              ║
# ║  7. Remove all hooks                                       ║
# ║                                                           ║
# ║  Hints:                                                    ║
# ║  - Use model.named_modules() to iterate layers             ║
# ║  - Handle case where output is a tuple                     ║
# ║  - Use functools.partial or closure to capture layer name  ║
# ╚═══════════════════════════════════════════════════════════╝

activation_stats = {}
hooks = []

# TODO: Define hook, register on model, forward pass, print, plot, cleanup
"""))

# ── Task 4.3: Backward hook ──
cells.append(md("""\
## 4.3 Implementation Task — Backward Hook for Gradient Flow Analysis

Register backward hooks to trace how gradients flow through the network.

**Requirements**:
- For each layer, record: gradient norm, mean, std of grad_output
- Compare gradient magnitudes across layers to detect vanishing/exploding patterns"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════════╗
# ║  TODO: Implement gradient flow logging with backward hooks ║
# ║                                                           ║
# ║  Steps:                                                    ║
# ║  1. Create a dict to store gradient flow stats             ║
# ║  2. Define a backward hook function                        ║
# ║  3. Register on every layer                                ║
# ║  4. Forward + backward pass                                ║
# ║  5. Print gradient flow table                              ║
# ║  6. Plot gradient norms (should show flow from output      ║
# ║     layers back to input layers)                           ║
# ║  7. Cleanup hooks                                          ║
# ║                                                           ║
# ║  Hint: grad_output is a tuple. Elements may be None.       ║
# ╚═══════════════════════════════════════════════════════════╝

grad_flow_stats = {}
hooks = []

# TODO: Implement
"""))

# ── Task 4.4: Compare across layers ──
cells.append(md("""\
## 4.4 Experiment — Comparing Activation & Gradient Stats

Using the hooks from 4.2 and 4.3, run a complete forward+backward pass and produce a combined analysis:

1. Side-by-side plot: activation stats (left) vs gradient stats (right) per layer
2. Identify layers where activations saturate or gradients vanish
3. Discuss: what does it mean if a layer has high activation std but near-zero gradient?"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO: Combined analysis                  ║
# ║  Run forward + backward with both hooks   ║
# ║  Create side-by-side visualization         ║
# ╚═══════════════════════════════════════════╝
"""))

# ═══════════════════════════════════════════════════════
# SECTION 5 — PROFILING & SYNCHRONIZATION TRAPS
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 5 — Profiling & Synchronization Traps

## 5.1 Conceptual Background

### The `.item()` Synchronization Trap

On CUDA, operations are **asynchronous** — they are queued on the GPU and Python continues immediately. However, `.item()`, `.cpu()`, and `print(tensor)` force **synchronization**: Python blocks until all queued GPU operations complete.

```python
# BAD — synchronization at every step
for batch in dataloader:
    loss = model(batch).sum()
    print(f"Loss: {loss.item()}")  # ← blocks GPU pipeline!
```

```python
# GOOD — log less frequently, or accumulate on GPU
if step % 100 == 0:
    print(f"Loss: {loss.item()}")
```

### `torch.profiler` Basics

PyTorch's built-in profiler traces both CPU and CUDA operations:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    # your code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### NVTX Ranges (CUDA-only)

Custom annotation ranges for NVIDIA profiling tools:

```python
torch.cuda.nvtx.range_push("forward_pass")
output = model(x)
torch.cuda.nvtx.range_pop()
```"""))

# ── Task 5.2: .item() trap ──
cells.append(md("""\
## 5.2 Experiment — Measuring the `.item()` Cost

Quantify the overhead of calling `.item()` in a training loop.

**Tasks**:
1. Write a minimal training loop (fake data, simple model)
2. Version A: call `loss.item()` every step
3. Version B: call `loss.item()` every 100 steps
4. Version C: never call `.item()` (accumulate loss tensor on GPU)
5. Benchmark all three versions
6. Report the overhead as a percentage

> **Note**: This experiment is most dramatic on GPU. On CPU, the difference is minimal. If you only have CPU, still do it — just note that the synchronization cost is negligible on CPU."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════╗
# ║  TODO: Benchmark .item() overhead          ║
# ║                                           ║
# ║  1. Create a simple model + fake data     ║
# ║  2. Loop N steps for each version          ║
# ║  3. Time each version                      ║
# ║  4. Compare with plot_benchmark_comparison ║
# ║                                           ║
# ║  Hint: Use benchmark_fn or manual timing  ║
# ╚═══════════════════════════════════════════╝
"""))

# ── Task 5.3: Profiler ──
cells.append(md("""\
## 5.3 Implementation Task — Profile a Training Step

Use `torch.profiler` to profile a single training step and identify the most expensive operations.

**Tasks**:
1. Create a model and optimizer
2. Wrap a forward + backward + optimizer step in `torch.profiler.profile`
3. Print the profiler table sorted by total time
4. (If GPU available) Add NVTX ranges for forward, backward, and optimizer steps
5. Identify the single most expensive operation"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════════╗
# ║  TODO: Profile a training step                             ║
# ║                                                           ║
# ║  1. model = TinyMLP(...)  (or any model from above)        ║
# ║  2. optimizer = torch.optim.SGD(model.parameters(), lr=.01)║
# ║  3. Profile with torch.profiler.profile(...)               ║
# ║  4. Inside the context: forward, loss, backward, step      ║
# ║  5. Print prof.key_averages().table(...)                   ║
# ║  6. Identify the top bottleneck                            ║
# ║                                                           ║
# ║  Optional (GPU only):                                      ║
# ║  - Add torch.cuda.nvtx.range_push/pop around phases        ║
# ╚═══════════════════════════════════════════════════════════╝
"""))

# ═══════════════════════════════════════════════════════
# FINAL CHALLENGE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# 🧪 Final Challenge — Tensor Autograd Debugger

## Objective

Implement the `AutogradDebugger` class defined in `src/grad_diagnostics.py`.

This is a **complete, reusable tool** that you can attach to any `nn.Module` to automatically diagnose training issues.

## Requirements

The debugger must:

1. **Accept any `nn.Module`** — no model-specific code
2. **Attach forward & backward hooks** to every layer
3. Log per-layer:
   - Gradient norms, means, stds
   - Activation means, stds, mins, maxs
4. **Detect NaN/Inf** in both activations and gradients
5. **Warn about non-contiguous tensors** where they might cause performance issues
6. **Report gradient health**: flag layers as HEALTHY / VANISHING / EXPLODING
7. **Provide clean summary reports** via `report_gradient_health()` and `report_activation_health()`

## Skeleton

The class skeleton is already in `src/grad_diagnostics.py`. You must implement:

| Method | Purpose |
|--------|---------|
| `attach()` | Register hooks on all layers |
| `detach()` | Remove all hooks |
| `_forward_hook_fn()` | Collect activation stats |
| `_backward_hook_fn()` | Collect gradient stats |
| `report_gradient_health()` | Print/return gradient diagnostics |
| `report_activation_health()` | Print/return activation diagnostics |
| `check_contiguity_warnings()` | Flag non-contiguous activations |

## Validation

After implementing, test your debugger:

```python
model = TinyMLP(n_layers=8, activation='relu')
debugger = AutogradDebugger(model).attach()

x = torch.randn(16, 32)
loss = model(x).sum()
loss.backward()

debugger.report_gradient_health()
debugger.report_activation_health()
debugger.check_contiguity_warnings()
debugger.log.summarize()
debugger.detach()
```

No solution is provided. Only the requirements above."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════════╗
# ║  FINAL CHALLENGE: Implement AutogradDebugger              ║
# ║                                                           ║
# ║  1. Open src/grad_diagnostics.py                           ║
# ║  2. Implement all TODO methods in AutogradDebugger         ║
# ║  3. Test it here:                                          ║
# ╚═══════════════════════════════════════════════════════════╝

# After implementing, uncomment and run:

# from grad_diagnostics import AutogradDebugger, DiagnosticLog
#
# model = TinyMLP(n_layers=8, activation='relu')
# debugger = AutogradDebugger(model).attach()
#
# x = torch.randn(16, 32)
# loss = model(x).sum()
# loss.backward()
#
# print("\\n=== GRADIENT HEALTH ===")
# debugger.report_gradient_health()
#
# print("\\n=== ACTIVATION HEALTH ===")
# debugger.report_activation_health()
#
# print("\\n=== CONTIGUITY WARNINGS ===")
# debugger.check_contiguity_warnings()
#
# print("\\n=== DIAGNOSTIC LOG ===")
# debugger.log.summarize()
#
# debugger.detach()
"""))

# ═══════════════════════════════════════════════════════
# SUMMARY CHECKLIST
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# ✅ Summary Checklist

Use this to verify you've internalized all concepts:

| # | Concept | Confident? |
|---|---------|-----------|
| 1 | I can explain the difference between a Tensor and its Storage | ☐ |
| 2 | I can predict strides after transpose/permute without running code | ☐ |
| 3 | I know exactly when `.view()` will fail and why | ☐ |
| 4 | I can draw the autograd DAG for any simple computation | ☐ |
| 5 | I understand leaf vs non-leaf tensors and when `.grad` is populated | ☐ |
| 6 | I can explain gradient accumulation and the `zero_grad` pattern | ☐ |
| 7 | I know why in-place ops break autograd (version counter) | ☐ |
| 8 | I can register forward/backward hooks and use them for diagnostics | ☐ |
| 9 | I understand the `.item()` synchronization cost on CUDA | ☐ |
| 10 | I can use `torch.profiler` to identify training bottlenecks | ☐ |
| 11 | I built a reusable AutogradDebugger tool | ☐ |

---

### 🔜 Next: Group 2 — Training Dynamics & Stability

Initialization strategies, normalization layers, loss function internals, reproducibility, and deterministic training."""))

# ═══════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "Group_1_Tensor_Autograd", "notebooks")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "01_tensor_autograd_lab.ipynb")
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Notebook generated: {out_path}")
print(f"   Cells: {len(cells)}")
