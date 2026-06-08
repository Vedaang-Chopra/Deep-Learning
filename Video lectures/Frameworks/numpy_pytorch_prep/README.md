# NumPy & PyTorch Interview Prep

**Estimated total practice time: 6–8 hrs**
**Recommended session:** one sitting, phases in order

---

## Integrity Notes

> These notebooks contain **reference implementations** in all TODO cells.
> Every problem is already solved — your job is to **clear each cell, implement from memory,
> then compare**. Self-discipline is the only enforcement mechanism.
>
> **Workflow:** read the problem heading → close the cell → implement in a scratch block
> or clear the cell → run asserts to verify → only then reveal the reference.

### Known Issues

| File | Issue |
|------|-------|
| `01_bridge/bridge_01_indexing_shape.ipynb` | **MISSING** — no source file found |
| `01_bridge/bridge_02_einsum_linalg.ipynb` | **MISSING** — no source file found |
| `01_bridge/bridge_03_coords_conv.ipynb` | **MISSING** — no source file found |
| `02_numpy_challenges/numpy_dojo_03_linalg_optimizers.ipynb` | Corrupted JSON metadata (line 583) — opens fine in JupyterLab, fails strict JSON parse |
| `02_numpy_challenges/numpy_dojo_05_operations.ipynb` | Corrupted JSON metadata (line 697) — same |
| `02_numpy_challenges/numpy_dojo_06_algorithms.ipynb` | Corrupted JSON metadata (line 555) — same |
| `02_numpy_challenges/numpy_dojo_07_av_specific.ipynb` | Corrupted JSON metadata (line 612) — same |

The corrupted metadata affects only the kernel info block, not cell content. JupyterLab
will load these files and may prompt you to select a kernel manually.

---

## 1. Learning Objectives

**By the end of these notebooks you will be able to:**

### NumPy
- Create, index, slice, reshape, and broadcast arrays without looking up syntax
- Implement standardization, min-max normalization, and image preprocessing pipelines
- Write all standard activation functions (ReLU, sigmoid, tanh, GELU, swish) and their gradients from scratch
- Implement softmax with numerical stability and cross-entropy loss
- Compute pairwise distance matrices, PCA, and SVD-based low-rank approximation without a loop
- Use `np.pad`, `np.cumsum`, `np.diff`, `np.percentile`, `np.bincount`, `np.unique`, `np.argsort`, `np.searchsorted`, `np.roll`, `np.meshgrid`, `np.einsum` fluently
- Implement SGD with momentum and Adam optimizer from scratch
- Build a 1-hidden-layer MLP with He initialization, forward pass, backpropagation, and gradient verification via finite differences
- Implement 2D convolution, K-means, Precision/Recall/F1, L1/L2 regularization, and SVD reconstruction
- Compute IoU and NMS for object detection; build rotation matrices in 2D and 3D; transform point clouds; voxelize a LiDAR scan into a BEV feature map

### PyTorch
- Create tensors from NumPy, factory functions, and datasets; manage dtypes and devices
- Index, slice, mask, and fancy-index tensors identically to NumPy
- Reshape, permute, transpose, concatenate, and stack tensors
- Apply element-wise math, reductions, argmax, topk, softmax, cumsum
- Perform matrix multiply (`@`), batched matmul (`bmm`), and `einsum`
- Use `requires_grad`, `loss.backward()`, `optimizer.step()`, `zero_grad()`, `no_grad()`, and `detach()`
- Build models with `nn.Linear`, `nn.ReLU`, `nn.Sequential`, and custom `nn.Module`
- Write a complete training loop with DataLoader, train/val split, and model checkpointing

---

## 2. Assignment Structure

| # | File | Tier | Topics | Problems | Est. Time | Status |
|---|------|------|--------|----------|-----------|--------|
| 0 | `00_pytorch_intro/pytorch_numpy_fundamentals.ipynb` | Intro (ref) | All core ops, NumPy+PyTorch side-by-side | Sections 1–5 | 20 min | Read & run only |
| 1 | `00_pytorch_intro/01_creation_and_types.ipynb` | Intro | Array creation, dtypes, devices | P1–P5 | 10–15 min | |
| 2 | `00_pytorch_intro/02_indexing_and_slicing.ipynb` | Intro | Slicing, boolean mask, fancy index, masked_fill | P1–P6 | 10–15 min | |
| 3 | `00_pytorch_intro/03_shape_and_broadcasting.ipynb` | Intro | Reshape, permute, cat/stack, broadcasting | P1–P6 | 10–15 min | |
| 4 | `00_pytorch_intro/04_math_and_reductions.ipynb` | Intro | Stats, normalization, argmax, softmax | P1–P6 | 10–15 min | |
| 5 | `00_pytorch_intro/05_matmul_and_aggregations.ipynb` | Intro | matmul, pairwise similarity, bmm, einsum, SVD | P1–P5 | 10–15 min | |
| 6 | `00_pytorch_intro/06_autograd_and_nn.ipynb` | Intro | autograd, no_grad, detach, nn.Linear, optimizer | P1–P6 | 10–15 min | |
| 7 | `01_bridge/bridge_01_indexing_shape.ipynb` | Bridge | Advanced indexing, shape ops | — | 20–30 min | **MISSING** |
| 8 | `01_bridge/bridge_02_einsum_linalg.ipynb` | Bridge | einsum, linear algebra | — | 20–30 min | **MISSING** |
| 9 | `01_bridge/bridge_03_coords_conv.ipynb` | Bridge | Coordinate transforms, convolution | — | 20–30 min | **MISSING** |
| 10 | `02_numpy_challenges/numpy_dojo_01_preprocessing.ipynb` | Challenge | Standardize, min-max, image preprocess, train/val split | P1–P4 | 25–35 min | |
| 11 | `02_numpy_challenges/numpy_dojo_02_activations_losses.ipynb` | Challenge | ReLU/sigmoid/tanh/GELU, softmax, cross-entropy | P5–P7 | 25–35 min | |
| 12 | `02_numpy_challenges/numpy_dojo_03_linalg_optimizers.ipynb` | Challenge | Pairwise distance, PCA, SVD, SGD+momentum, Adam | P8–P12 | 35–45 min | |
| 13 | `02_numpy_challenges/numpy_dojo_04_mlp_pipeline.ipynb` | Challenge | He init, forward, backprop, finite-diff check, training loop | P13–P17 | 35–45 min | |
| 14 | `02_numpy_challenges/numpy_dojo_05_operations.ipynb` | Challenge | pad, cumsum, diff, roll, meshgrid, einsum, argsort | P1–P8 | 35–50 min | |
| 15 | `02_numpy_challenges/numpy_dojo_06_algorithms.ipynb` | Challenge | 2D conv, K-means, Precision/Recall/F1, L1/L2 reg, SVD reconstruct | P1–P5 | 35–50 min | |
| 16 | `02_numpy_challenges/numpy_dojo_07_av_specific.ipynb` | Challenge | IoU, NMS, rotation matrices, point cloud transforms, BEV voxelization | P1–P5 | 35–50 min | |

**Tier definitions:**
- **Intro** — hints + DRILL examples provided; NumPy reference shown alongside PyTorch implementation
- **Bridge** — isolated mechanic drills, 3–8 lines each, no hints *(files missing — skip or create your own)*
- **Challenge** — full algorithms from memory, no hints, assert-verified

---

## 3. Recommended Order

```
PHASE 1 — Syntax warmup  (1.5–2 hrs)
─────────────────────────────────────
  00_pytorch_intro/pytorch_numpy_fundamentals.ipynb   ← read + run, do not implement
  00_pytorch_intro/01_creation_and_types.ipynb
  00_pytorch_intro/02_indexing_and_slicing.ipynb
  00_pytorch_intro/03_shape_and_broadcasting.ipynb
  00_pytorch_intro/04_math_and_reductions.ipynb
  00_pytorch_intro/05_matmul_and_aggregations.ipynb
  00_pytorch_intro/06_autograd_and_nn.ipynb

PHASE 2 — Bridge  (SKIPPED — source files missing)
────────────────────────────────────────────────────
  01_bridge/ files not present; proceed to Phase 3

PHASE 3 — Challenges  (3.5–4.5 hrs)
──────────────────────────────────────
  02_numpy_challenges/numpy_dojo_01_preprocessing.ipynb
  02_numpy_challenges/numpy_dojo_02_activations_losses.ipynb
  02_numpy_challenges/numpy_dojo_03_linalg_optimizers.ipynb
  02_numpy_challenges/numpy_dojo_04_mlp_pipeline.ipynb       ← most important, do not skip
  02_numpy_challenges/numpy_dojo_05_operations.ipynb
  02_numpy_challenges/numpy_dojo_06_algorithms.ipynb
  02_numpy_challenges/numpy_dojo_07_av_specific.ipynb        ← AV-specific, do last
```

---

## 4. Rules

### Phase 1 (Intro)
- Open the notebook and **clear all outputs** (`Kernel → Restart & Clear Output`)
- Read the NumPy reference cell first — understand the pattern
- Read the `# DRILL:` cell — this is a minimal worked example of the key API
- **Close or scroll past** the TODO cell, implement in a fresh scratch cell, then compare
- Do not look at the reference while you implement the PyTorch version

### Phase 2 (Bridge — files missing)
- Each problem would have been 3–8 lines
- If you create your own bridge exercises: write with a loop first, get the assert green, then vectorize
- Do not search docs mid-problem; note the gap and look it up after the assert passes

### Phase 3 (Challenges)
- No hints. No docs. Implement from math and memory.
- If you can't remember a function name, derive it from primitives:
  *Can't recall `np.linalg.norm`? Use `np.sqrt((x**2).sum())`*
- The finite difference gradient check in `dojo_04` P15 is the single most important cell
  in all notebooks. If it passes (rel\_error < 1e-4), your backprop is correct.
- For `dojo_07`: IoU and NMS are the most likely to appear in the interview.
  Prioritize those if time is short.

---

## 5. Quick Reference: Operations by Notebook

| Operation | NumPy | PyTorch |
|-----------|-------|---------|
| Create from list | `np.array([1,2,3])` | `torch.tensor([1,2,3])` |
| Create from NumPy | — | `torch.from_numpy(arr)` |
| Shape | `arr.shape` | `t.shape` |
| Dtype | `arr.dtype` | `t.dtype` |
| Cast | `arr.astype(np.float32)` | `t.float()` / `t.to(torch.float32)` |
| Zeros | `np.zeros((3,4))` | `torch.zeros(3,4)` |
| Ones | `np.ones((3,4))` | `torch.ones(3,4)` |
| Random normal | `np.random.randn(3,4)` | `torch.randn(3,4)` |
| Arange | `np.arange(10)` | `torch.arange(10)` |
| Slice | `arr[2:5]` | `t[2:5]` |
| Boolean mask | `arr[arr > 0]` | `t[t > 0]` |
| Fancy index | `arr[[0,2,4]]` | `t[[0,2,4]]` |
| Where | `np.where(cond, a, b)` | `torch.where(cond, a, b)` |
| Masked fill | — | `t.masked_fill(mask, val)` |
| Reshape | `arr.reshape(N, -1)` | `t.reshape(N, -1)` / `t.view(N,-1)` |
| Flatten | `arr.flatten()` / `arr.ravel()` | `t.flatten()` |
| Transpose 2D | `arr.T` | `t.T` / `t.transpose(0,1)` |
| Permute axes | `np.transpose(arr, (2,0,1))` | `t.permute(2,0,1)` |
| Add dim | `arr[np.newaxis]` / `np.expand_dims` | `t.unsqueeze(0)` |
| Remove dim | `np.squeeze(arr)` | `t.squeeze()` |
| Concat | `np.concatenate([a,b], axis=0)` | `torch.cat([a,b], dim=0)` |
| Stack | `np.stack([a,b], axis=0)` | `torch.stack([a,b], dim=0)` |
| Element-wise mul | `a * b` | `a * b` |
| Matrix mul | `a @ b` / `np.matmul` | `a @ b` / `torch.matmul` |
| Batched matmul | `np.einsum('bij,bjk->bik',a,b)` | `torch.bmm(a,b)` |
| Einsum | `np.einsum('ij,jk->ik', a, b)` | `torch.einsum('ij,jk->ik', a, b)` |
| Sum | `arr.sum(axis=0)` | `t.sum(dim=0)` |
| Mean | `arr.mean(axis=0)` | `t.mean(dim=0)` |
| Std | `arr.std(axis=0)` | `t.std(dim=0)` |
| Max value | `arr.max(axis=0)` | `t.max(dim=0).values` |
| Argmax | `arr.argmax(axis=0)` | `t.argmax(dim=0)` |
| Topk | — | `t.topk(k, dim=-1)` |
| Clip | `np.clip(arr, 0, 1)` | `t.clamp(0, 1)` |
| Abs | `np.abs(arr)` | `t.abs()` |
| Exp | `np.exp(arr)` | `t.exp()` |
| Log | `np.log(arr)` | `t.log()` |
| Sqrt | `np.sqrt(arr)` | `t.sqrt()` |
| Sign | `np.sign(arr)` | `t.sign()` |
| L2 norm | `np.linalg.norm(arr, axis=1)` | `t.norm(dim=1)` / `torch.linalg.norm` |
| SVD | `np.linalg.svd(A, full_matrices=False)` | `torch.linalg.svd(A, full_matrices=False)` |
| Pad | `np.pad(arr, pad_width)` | `torch.nn.functional.pad(t, pad)` |
| Roll | `np.roll(arr, k)` | `torch.roll(t, k)` |
| Cumsum | `np.cumsum(arr, axis=0)` | `t.cumsum(dim=0)` |
| Diff | `np.diff(arr)` | `t.diff()` |
| Unique | `np.unique(arr, return_counts=True)` | `torch.unique(t, return_counts=True)` |
| Bincount | `np.bincount(arr, minlength=k)` | `torch.bincount(t, minlength=k)` |
| Argsort | `np.argsort(arr)[::-1]` | `t.argsort(descending=True)` |
| Searchsorted | `np.searchsorted(edges, vals)` | `torch.searchsorted(edges, vals)` |
| Meshgrid | `np.meshgrid(x, y)` | `torch.meshgrid(x, y, indexing='xy')` |
| view vs reshape | reshape (may copy) | view (contiguous required) / reshape |
| Contiguous | — | `t.contiguous()` |
| Detach | — | `t.detach()` |
| No grad | — | `with torch.no_grad():` |
| To numpy | — | `t.detach().cpu().numpy()` |
| Gradient | — | `loss.backward()` |
| Zero grad | — | `optimizer.zero_grad()` |
| Softmax | `np.exp(x)/np.exp(x).sum()` | `torch.softmax(t, dim=-1)` |
| Log softmax | custom | `torch.log_softmax(t, dim=-1)` |
| Scatter add | `np.add.at(out, idx, vals)` | `out.scatter_add_(0, idx, vals)` |
| Scatter max | `np.maximum.at(out, idx, vals)` | no direct; use `scatter_reduce` |

---

## 6. Things to Do Without Looking Up

Star any you got wrong during practice.

### NumPy — write these cold

- [ ] Standardize `(N,D)` matrix per column, guard std=0
- [ ] Min-max normalize to `[-1, 1]`, guard range=0
- [ ] Rotate `(B,H,W,C)` batch 90° CW using only indexing; convert to `(B,C,H,W)`
- [ ] Numerically stable softmax on `(N,C)` logits
- [ ] Cross-entropy loss from logits + integer labels
- [ ] Gradient of cross-entropy: `(softmax - one_hot) / N`
- [ ] Pairwise L2 distance matrix `(N,M)` using identity, no loops
- [ ] He initialization: `W ~ N(0, sqrt(2/fan_in))`
- [ ] Adam one step from scratch (m, v, bias correction, update)
- [ ] SVD low-rank reconstruction: `U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]`
- [ ] Rolling sum from cumsum: `cs = np.concatenate([[0], cumsum(x)]); cs[w:] - cs[:-w]`
- [ ] Log-sum-exp: `m + log(sum(exp(x - m)))` where `m = max(x)`
- [ ] 2D rotation matrix: `[[cos,-sin],[sin,cos]]`
- [ ] Homogeneous coords: `np.hstack([pts, np.ones((N,1))])`
- [ ] Flat index: `row * W + col`; reverse: `flat // W`, `flat % W`
- [ ] IoU: `inter / (area_a + area_b - inter)`
- [ ] NMS: sort by score, greedily keep, suppress IoU > threshold
- [ ] Voxelize: `np.floor(pts / cell_size).astype(int)`; count with `np.bincount` on flat index

### PyTorch — write these cold

- [ ] `torch.tensor` vs `torch.from_numpy` (copy vs shared memory)
- [ ] `t.view(-1, 64)` vs `t.reshape(-1, 64)` (contiguous requirement)
- [ ] `t.unsqueeze(1)` vs `t[:, None]`
- [ ] `t.permute(0, 2, 3, 1)` for BCHW→BHWC
- [ ] `torch.cat` vs `torch.stack` (existing dim vs new dim)
- [ ] `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()` order
- [ ] `with torch.no_grad():` for inference
- [ ] `t.detach().cpu().numpy()` to get numpy array
- [ ] `nn.Module` skeleton: `__init__` with `super().__init__()`, `forward`
- [ ] `torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)`
- [ ] `t.masked_fill(mask, float('-inf'))` for attention masking

---

## 7. Missing Files

The following files were not found in any source directory and could not be copied:

```
01_bridge/bridge_01_indexing_shape.ipynb   — NOT FOUND
01_bridge/bridge_02_einsum_linalg.ipynb   — NOT FOUND
01_bridge/bridge_03_coords_conv.ipynb     — NOT FOUND
```

The `01_bridge/` directory exists but is empty. Skip Phase 2 or create your own
bridge exercises as warm-ups before the challenge notebooks.
