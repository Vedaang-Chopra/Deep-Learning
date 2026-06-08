#!/usr/bin/env python3
"""Generate the Group 11 Data-Centric Experiment Lab notebook."""
import json, os

def md(source):
    return {"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in source.split("\n")[:-1]]+[source.split("\n")[-1]]}

def code(source):
    return {"cell_type":"code","metadata":{},"source":[l+"\n" for l in source.split("\n")[:-1]]+[source.split("\n")[-1]],"execution_count":None,"outputs":[]}

cells = []

# ═══ HEADER ═══
cells.append(md("""\
# 🔬 Notebook 11 — Data-Centric Experiment Lab
## Versioned Datasets, Leakage Detection, Ablations, and Error Analysis

**Group 11 — Data-Centric ML + Experiment Engineering**

---

### 🎯 Learning Objectives

1. Build dataset manifests for reproducible versioning
2. Detect data leakage (exact + near-duplicate)
3. Run data quality checks (balance, corruption, label noise)
4. Log experiments with reproducible run bundles
5. Run structured ablations with multi-seed aggregation
6. Perform error analysis: confusion, calibration, slice metrics
7. Monitor distribution drift with PSI and JS divergence
8. Produce a paper-grade "Run Bundle" directory

### 📂 File Structure

```
Group_11_Data_Centric/
├── notebooks/11_data_experiment_rigor_lab.ipynb   ← you are here
└── src/
    ├── dataset_manifest.py   ← versioning via hashes + metadata
    ├── splitting.py          ← stratified splits + leakage detection
    ├── data_checks.py        ← class balance, corruption, label noise
    ├── run_logging.py        ← run dirs, config, metrics, checkpoints
    ├── ablations.py          ← grid generation, sweep runner
    ├── error_analysis.py     ← confusion, calibration, slicing
    └── drift_monitoring.py   ← PSI, JS divergence, drift reports
```

> 📌 **No black boxes**: You build the versioning, logging, and analysis tools yourself."""))

# ═══ SETUP ═══
cells.append(md("## 0 — Environment Setup"))

cells.append(code("""\
import sys, os, copy, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.pardir, "src"))

from dataset_manifest import compute_file_hash, compute_tensor_hash, build_dataset_manifest, save_manifest
from splitting import make_split, make_stratified_split, leakage_check_exact_duplicates, leakage_check_near_duplicates, print_split_stats, print_leakage_report
from data_checks import check_class_balance, check_missing_or_corrupt, check_label_noise_suspects, plot_class_balance, plot_label_noise_gallery
from run_logging import create_run_dir, save_config, log_metrics, save_model, capture_env, print_run_tree, load_metrics
from ablations import generate_grid, run_sweep, summarize_results, plot_ablation_results
from error_analysis import confusion_matrix, per_class_metrics, calibration_curve, slice_by_feature, slice_metrics, plot_confusion_matrix, plot_calibration, plot_slice_metrics
from drift_monitoring import compute_feature_stats, psi, js_divergence, drift_report, plot_drift_comparison, plot_drift_summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
print(f"PyTorch: {torch.__version__}, Device: {DEVICE}")"""))

cells.append(code("""\
# ── Load CIFAR-10 ──
transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
train_full = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

all_labels = [train_full[i][1] for i in range(len(train_full))]
print(f"Train: {len(train_full)}, Test: {len(test_set)}, Classes: {len(CIFAR10_CLASSES)}")"""))

# ═══ SECTION 1 — WHY ═══
cells.append(md("""\
---
# Section 1 — Why Data-Centric and Experiment Rigor Matters

### The Uncomfortable Truth

| Improvement Source | Frequency | Effort |
|-------------------|-----------|--------|
| Better architecture | ~20% of wins | High (research) |
| Better data/evaluation | ~60% of wins | Medium (engineering) |
| Better hyperparams | ~20% of wins | Low (sweeps) |

### Quiet Dataset Failures

| Failure | Symptom | Fix |
|---------|---------|-----|
| Train/test leakage | Inflated accuracy | Leakage checks |
| Label noise | Ceiling on accuracy | Noise suspects |
| Class imbalance | Poor minority class recall | Stratified splits |
| Distribution shift | Accuracy drops in production | Drift monitoring |
| Unreproducible runs | "It worked on my machine" | Run bundles |

### What You Will Build

```
runs/final_run/
├── manifest.json       ← exactly which data was used
├── config.json         ← full experiment config
├── environment.json    ← python, torch, CUDA versions
├── metrics.jsonl       ← per-epoch metrics
├── model.pt            ← checkpoint
├── evaluation.json     ← error analysis report
└── plots/              ← saved figures
```"""))

# ═══ SECTION 2 — MANIFEST ═══
cells.append(md("""\
---
# Section 2 — Dataset Versioning via Manifest

### Why "I used CIFAR-10" is insufficient

```
Saying "I used CIFAR-10" doesn't specify:
  - Which split? (downloaded when? from where?)
  - What preprocessing? (normalization? augmentation?)
  - Which subset? (full? random 10%? stratified?)
  - Seed for subset selection?
  - Were any samples excluded?

A MANIFEST records ALL of this.
```"""))

cells.append(md("## 2.1 Tasks\n\nOpen `src/dataset_manifest.py` and implement:\n\n1. **`compute_file_hash()`** — SHA-256 hash\n2. **`compute_tensor_hash()`** — hash of tensor bytes\n3. **`build_dataset_manifest()`** — full manifest with metadata\n4. **`save_manifest()` / `load_manifest()`** — JSON I/O"))

cells.append(code("""\
# ╔═════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Create a dataset manifest               ║
# ║                                                     ║
# ║  TODO:                                               ║
# ║  1. Build sample list: [{index, label, hash}, ...]  ║
# ║  2. build_dataset_manifest(samples, ...)             ║
# ║  3. Save to runs/<run_id>/manifest.json              ║
# ╚═════════════════════════════════════════════════════╝

# TODO:
# samples = []
# for i in range(min(1000, len(train_full))):  # subset for speed
#     img, label = train_full[i]
#     h = compute_tensor_hash(img)
#     samples.append({'index': i, 'label': int(label), 'hash': h})
#
# manifest = build_dataset_manifest(
#     samples, dataset_id='cifar10-train-subset',
#     split_name='train', transforms_spec='ToTensor+Normalize',
#     seed=SEED
# )
# print(f"Manifest: {manifest['dataset_id']}, {manifest['num_samples']} samples")
# print(f"Content hash: {manifest['content_hash'][:16]}...")
"""))

# ═══ SECTION 3 — SPLITTING + LEAKAGE ═══
cells.append(md("""\
---
# Section 3 — Splitting Strategies + Leakage Detection

### Leakage Examples

| Type | How it happens | Effect |
|------|---------------|--------|
| Exact duplicates | Same image in train+test | Inflated accuracy |
| Near-duplicates | Augmented versions leak | Subtle inflation |
| Group leakage | Same patient/user in both | Systemic bias |
| Temporal leakage | Future data in training | Unrealistic perf |"""))

cells.append(md("## 3.1 Tasks\n\nOpen `src/splitting.py` and implement:\n\n1. **`make_split()`** — random split with seed\n2. **`make_stratified_split()`** — class-balanced split\n3. **`leakage_check_exact_duplicates()`** — find identical samples\n4. **`leakage_check_near_duplicates()`** — embedding similarity"))

cells.append(code("""\
# ╔═════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Leakage failure → detection → fix      ║
# ║                                                     ║
# ║  TODO:                                               ║
# ║  1. Make stratified split                            ║
# ║  2. INJECT duplicates (copy train→val intentionally) ║
# ║  3. Run leakage check → should FAIL                 ║
# ║  4. Fix the split (remove duplicates)               ║
# ║  5. Re-run leakage check → should PASS              ║
# ╚═════════════════════════════════════════════════════╝

# TODO:
# splits = make_stratified_split(all_labels, seed=SEED, ratios=(0.8, 0.1, 0.1))
# print_split_stats(splits, all_labels)
#
# # Inject leakage
# leaked_val = splits['val'][:100]
# splits_bad = copy.deepcopy(splits)
# splits_bad['train'].extend(leaked_val)
#
# hashes_train = [compute_tensor_hash(train_full[i][0]) for i in splits_bad['train'][:500]]
# hashes_val = [compute_tensor_hash(train_full[i][0]) for i in splits_bad['val'][:500]]
# report = leakage_check_exact_duplicates(hashes_train, hashes_val, [])
# print_leakage_report(report)  # should FAIL
"""))

# ═══ SECTION 4 — DATA QUALITY ═══
cells.append(md("""\
---
# Section 4 — Data Quality Checks

### Data Checks as Unit Tests

```python
# Think of data checks like unit tests:
assert check_class_balance(labels)['is_balanced']     # no extreme imbalance
assert check_missing_or_corrupt(dataset)['passed']    # no corrupt samples
assert len(noise_suspects) < threshold                # reasonable noise level
```"""))

cells.append(md("## 4.1 Tasks\n\nOpen `src/data_checks.py` and implement:\n\n1. **`check_class_balance()`** — class distribution + imbalance ratio\n2. **`check_missing_or_corrupt()`** — catch bad samples\n3. **`check_label_noise_suspects()`** — high-confidence wrong = mislabeled?"))

cells.append(code("""\
# ╔═════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Inject label noise → detect suspects    ║
# ║                                                     ║
# ║  TODO:                                               ║
# ║  1. Check class balance on clean labels              ║
# ║  2. Inject 5% random label noise                    ║
# ║  3. Train a small CNN on noisy data                  ║
# ║  4. Get model predictions on training data           ║
# ║  5. check_label_noise_suspects() → find the fakes   ║
# ║  6. Visualize suspect gallery                        ║
# ╚═════════════════════════════════════════════════════╝

# TODO:
# balance = check_class_balance(all_labels, class_names=CIFAR10_CLASSES)
# plot_class_balance(balance)
#
# # Inject noise
# noisy_labels = all_labels.copy()
# noise_rate = 0.05
# n_flip = int(len(noisy_labels) * noise_rate)
# flip_indices = np.random.choice(len(noisy_labels), n_flip, replace=False)
# for i in flip_indices:
#     noisy_labels[i] = (noisy_labels[i] + np.random.randint(1, 10)) % 10
"""))

# ═══ SECTION 5 — RUN LOGGING ═══
cells.append(md("""\
---
# Section 5 — Run Logging: Reproducible Artifacts

### The Minimum Viable Run Bundle

Every experiment run must record:

| Artifact | Why |
|----------|-----|
| `config.json` | What hyperparameters were used |
| `manifest.json` | Exactly which data was used |
| `environment.json` | Python/PyTorch/CUDA versions |
| `metrics.jsonl` | Per-epoch loss and accuracy |
| `model.pt` | Checkpoint for reproduction |"""))

cells.append(md("## 5.1 Tasks\n\nOpen `src/run_logging.py` and implement:\n\n1. **`create_run_dir()`** — timestamped directory\n2. **`save_config()`** — experiment config\n3. **`log_metrics()`** — append to JSONL\n4. **`save_model()`** / **`load_model()`** — checkpoints\n5. **`capture_env()`** — environment snapshot"))

cells.append(code("""\
# ╔═════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Train + Log Everything                  ║
# ║                                                     ║
# ║  TODO:                                               ║
# ║  1. create_run_dir()                                 ║
# ║  2. save_config() with all hyperparams               ║
# ║  3. capture_env()                                    ║
# ║  4. Train small CNN for 5 epochs                     ║
# ║  5. log_metrics() each epoch                         ║
# ║  6. save_model() final checkpoint                    ║
# ║  7. print_run_tree() to see the bundle               ║
# ╚═════════════════════════════════════════════════════╝

# Simple CNN (provided)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(nn.Linear(64*4*4, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

# TODO: implement training loop with full logging
# run_dir = create_run_dir("runs", prefix="baseline")
# config = {'model': 'SimpleCNN', 'optimizer': 'adamw', 'lr': 1e-3, 'epochs': 5, 'seed': SEED}
# save_config(config, run_dir)
# capture_env(run_dir)
# ...training loop with log_metrics()...
# save_model(model, run_dir)
# print_run_tree(run_dir)
"""))

# ═══ SECTION 6 — ABLATIONS ═══
cells.append(md("""\
---
# Section 6 — Structured Ablations

### Ablation ≠ Random Tuning

| Bad Practice | Good Practice |
|-------------|--------------|
| Change one thing, run once | Grid of configs × multiple seeds |
| "I tried Adam, it was better" | Mean ± std across 3 seeds |
| Notebook-only experiment | Logged configs + metrics |"""))

cells.append(md("## 6.1 Tasks\n\nOpen `src/ablations.py` and implement:\n\n1. **`generate_grid()`** — expand config space\n2. **`run_sweep()`** — run each config × seed\n3. **`summarize_results()`** — mean/std/best"))

cells.append(code("""\
# ╔═════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Small Ablation Sweep                    ║
# ║                                                     ║
# ║  TODO:                                               ║
# ║  1. Define config space                              ║
# ║  2. generate_grid()                                  ║
# ║  3. Write train_fn(config, seed) -> metrics          ║
# ║  4. run_sweep() with seeds=[42, 123, 456]            ║
# ║  5. summarize_results()                              ║
# ║  6. plot_ablation_results()                          ║
# ╚═════════════════════════════════════════════════════╝

# TODO:
# config_space = {
#     'optimizer': ['sgd', 'adamw'],
#     'augmentation': ['none', 'basic'],
#     'label_smoothing': [0.0, 0.1],
# }
# configs = generate_grid(config_space)
# print(f"Total configs: {len(configs)}")
#
# def train_fn(config, seed):
#     # Short training (2 epochs, subset)
#     # Return {'accuracy': ..., 'loss': ...}
#     pass
#
# results = run_sweep(configs, train_fn, seeds=[42, 123, 456])
# summary = summarize_results(results, metric_key='accuracy')
# plot_ablation_results(summary)
"""))

# ═══ SECTION 7 — ERROR ANALYSIS ═══
cells.append(md("""\
---
# Section 7 — Error Analysis + Slice Metrics

### Why Accuracy Hides Failures

```
Overall accuracy: 92%   ← looks great!

Per-class breakdown:
  airplane: 96%  ← fine
  cat:      74%  ← terrible!
  dog:      71%  ← even worse!

Slice by brightness:
  dark images:  81%  ← significant drop
  bright images: 95%  ← much better

Without slicing, you'd ship a biased model.
```"""))

cells.append(md("## 7.1 Tasks\n\nOpen `src/error_analysis.py` and implement:\n\n1. **`confusion_matrix()`** — from scratch\n2. **`per_class_metrics()`** — precision/recall/F1\n3. **`calibration_curve()`** — reliability diagram\n4. **`slice_by_feature()`** — bin by brightness, etc.\n5. **`slice_metrics()`** — accuracy per slice"))

cells.append(code("""\
# ╔═════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Full Error Analysis                     ║
# ║                                                     ║
# ║  TODO:                                               ║
# ║  1. Get model predictions on test set                ║
# ║  2. confusion_matrix() + plot                        ║
# ║  3. per_class_metrics() → find worst class           ║
# ║  4. calibration_curve() → reliability diagram        ║
# ║  5. Slice by brightness → compare metrics            ║
# ╚═════════════════════════════════════════════════════╝

# TODO:
# model.eval()
# all_preds, all_labels, all_probs = [], [], []
# with torch.no_grad():
#     for images, labels in test_loader:
#         logits = model(images.to(DEVICE))
#         probs = F.softmax(logits, dim=1)
#         all_preds.extend(logits.argmax(1).cpu().numpy())
#         all_labels.extend(labels.numpy())
#         all_probs.append(probs.cpu().numpy())
#
# cm = confusion_matrix(np.array(all_labels), np.array(all_preds), 10)
# plot_confusion_matrix(cm, class_names=CIFAR10_CLASSES)
#
# metrics = per_class_metrics(cm, class_names=CIFAR10_CLASSES)
# for m in sorted(metrics, key=lambda x: x['f1']):
#     print(f"  {m['class']:<12} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
"""))

# ═══ SECTION 8 — DRIFT ═══
cells.append(md("""\
---
# Section 8 — Drift Monitoring Signals

### Three Types of Drift

| Type | What Changes | Signal |
|------|-------------|--------|
| **Covariate** | Input distribution P(X) | Feature stats, PSI |
| **Label** | Label distribution P(Y) | Class frequency shift |
| **Concept** | Relationship P(Y|X) | Accuracy drop |

### PSI Interpretation

```
PSI < 0.1     → No significant drift
PSI 0.1–0.25  → Moderate drift (investigate)
PSI > 0.25    → Significant drift (retrain!)
```"""))

cells.append(md("## 8.1 Tasks\n\nOpen `src/drift_monitoring.py` and implement:\n\n1. **`compute_feature_stats()`** — summary statistics\n2. **`psi()`** — Population Stability Index\n3. **`js_divergence()`** — Jensen-Shannon divergence\n4. **`drift_report()`** — comprehensive report"))

cells.append(code("""\
# ╔═════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Simulate Drift → Detect → Measure      ║
# ║                                                     ║
# ║  TODO:                                               ║
# ║  1. Extract embeddings from test set (reference)     ║
# ║  2. Apply strong augmentation → "drifted" data       ║
# ║  3. Extract drifted embeddings                       ║
# ║  4. drift_report() → PSI, JS divergence              ║
# ║  5. Correlate drift magnitude with accuracy drop     ║
# ╚═════════════════════════════════════════════════════╝

# TODO:
# # Extract features using model's feature extractor
# def get_embeddings(model, loader, device):
#     model.eval()
#     embs = []
#     with torch.no_grad():
#         for imgs, _ in loader:
#             feats = model.features(imgs.to(device)).flatten(1)
#             embs.append(feats.cpu().numpy())
#     return np.concatenate(embs)
#
# ref_emb = get_embeddings(model, test_loader, DEVICE)
#
# # Simulate drift: strong color jitter + blur
# drift_transform = T.Compose([
#     T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
#     T.GaussianBlur(5, sigma=(1.0,2.0)),
#     T.ToTensor(),
#     T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
# ])
#
# report = drift_report(ref_emb, drifted_emb)
# plot_drift_summary(report)
"""))

# ═══ FINAL CHALLENGE ═══
cells.append(md("""\
---
# 🧪 Final Challenge — Paper-Grade Run Bundle

## Produce a complete run directory:

```
runs/final_run/
├── manifest.json       ← dataset versioning
├── config.json         ← experiment config
├── environment.json    ← reproducibility
├── metrics.jsonl       ← training log
├── model.pt            ← checkpoint
├── evaluation.json     ← error analysis report
└── plots/              ← saved figures
```

### Reproducibility Checklist

| Check | Status |
|-------|--------|
| Dataset manifest saved and versioned | ☐ |
| No train/test leakage (verified) | ☐ |
| Data quality checks passed | ☐ |
| Config fully specified (no hardcoded values) | ☐ |
| Environment captured | ☐ |
| Per-epoch metrics logged | ☐ |
| Error analysis: confusion + calibration + slices | ☐ |
| Drift monitoring baseline established | ☐ |
| Model checkpoint saved | ☐ |
| Can another person reproduce from this bundle? | ☐ |"""))

cells.append(code("""\
# ╔═════════════════════════════════════════════════════╗
# ║  FINAL: Build the complete run bundle                ║
# ╚═════════════════════════════════════════════════════╝

# TODO: Create final run directory with all artifacts
# final_run = create_run_dir("runs", prefix="final")
# ... save all artifacts ...
# print_run_tree(final_run)
"""))

cells.append(code("""\
# ╔═════════════════════════════════════════════════════╗
# ║  REFLECTION                                         ║
# ╚═════════════════════════════════════════════════════╝

REFLECTION = \"\"\"
# Data-Centric Experiment Reflection

## 1. Top 3 Failure Modes Found
TODO: What data/evaluation issues did you catch?

## 2. Mandatory Checks for Every Future Project
TODO: Which checks are now non-negotiable?

## 3. Most Useful Drift Signal
TODO: Which drift metric was most informative?

## 4. What Would You Add to the Run Bundle?
TODO: What's missing for full reproducibility?
\"\"\"
print(REFLECTION)"""))

cells.append(md("""\
---
# ✅ Summary Checklist

| # | Competency | Confident? |
|---|-----------|-----------|
| 1 | I can create versioned dataset manifests with content hashes | ☐ |
| 2 | I can detect exact and near-duplicate leakage across splits | ☐ |
| 3 | I can run data quality checks (balance, corruption, noise) | ☐ |
| 4 | I can log reproducible run bundles (config, env, metrics, model) | ☐ |
| 5 | I can run structured ablations with multi-seed aggregation | ☐ |
| 6 | I can compute confusion matrices, per-class F1, calibration curves | ☐ |
| 7 | I can slice metrics by features and identify failure modes | ☐ |
| 8 | I can detect drift using PSI and JS divergence | ☐ |
| 9 | I can produce a paper-grade run bundle | ☐ |
| 10 | I understand why data-centric engineering matters more than architecture | ☐ |"""))

# ═══ BUILD ═══
nb = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10.0"}},"cells":cells}
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir,"notebooks","11_data_experiment_rigor_lab.ipynb"))
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w") as f: json.dump(nb,f,indent=1)
mc = sum(1 for c in cells if c["cell_type"]=="markdown")
cc = sum(1 for c in cells if c["cell_type"]=="code")
print(f"Notebook: {out}\nCells: {len(cells)} (markdown: {mc}, code: {cc})")
