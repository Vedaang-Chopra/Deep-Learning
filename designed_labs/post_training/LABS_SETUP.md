# LABS_SETUP.md — Infrastructure & Ops

> Companion to `POST_TRAINING_LAB_CURRICULUM_PLAN.md` §7 ("Infrastructure & ops — every lab inherits this").
> One-time setup per environment; every lab in the §5 dependency chain assumes this file is done.
> **This file contains infrastructure commands only — no lab solutions.** Lab-specific logic lives in each lab's starter/notebook per the plan's no-solutions rule (§9).

---

## 0. Environments at a glance

| Tier | Where | Hardware | Role |
|---|---|---|---|
| **Local** | Laptop (macOS) | CPU | Read the answer key (`rlhf-book/code/`); write/inspect code |
| **A — prototype** | Google Colab | T4, 16 GB VRAM | Labs 00–03 at 0.6B; Lab 05 (CPU); smoke-tests of everything |
| **B — train** | GT CoC (`mal`/`inara`/`jayne`/`shepherd`) | 8× A40, 46 GB | Full-FT 1.7B (SFT/DPO/RM/GRPO); 4B with LoRA; SDPO reference runs |
| **C — data prep only** | `rivertam`/`simontam` | 2080 Ti, 11 GB | Tokenization / data prep. **Never training** (too small for training batches) |

Lab→tier mapping is the "Compute tier" column of the §5 curriculum table. When a lab says A/B, prototype on Colab, run the long version on the cluster.

---

## 1. Local / laptop — answer-key reading only

The repo (`natolambert/rlhf-book`) is the **answer key**. Per design principle 1, open `code/` only *after* your version trains. The local clone is for reading, grepping, and `uv sync` so you can type-check/step through the reference implementations.

```bash
# one-time
cd "$HOME/all_data/complete_technical_work/Reference and Learning Content/Model_Training/Post_Training"
git clone https://github.com/natolambert/rlhf-book.git

cd rlhf-book
uv sync          # installs locked deps from uv.lock into .venv
source .venv/bin/activate
python -c "import torch; print(torch.__version__)"   # sanity check
```

Checklist:

- [ ] Clone exists and `uv sync` completed without errors.
- [ ] You can locate the compare-against targets, e.g. `code/instruction_tuning/utils.py`, `code/policy_gradients/loss.py::GRPOLoss`, `code/direct_alignment/loss.py::DPOLoss`.
- [ ] **Discipline:** you have NOT opened the lab's answer-key file before your own version trains.

---

## 2. Colab profile (Tier A — T4, 16 GB)

One Colab notebook cell block per lab notebook (pinned, T4-safe). Run before any other cell.

### 2.1 Pinned deps

```python
# Colab cell 1 — pinned install (~2 min). Re-run per fresh session.
%pip install -q \
    "torch==2.6.0" \
    "transformers==4.51.3" \
    "datasets==3.5.0" \
    "accelerate==1.6.0" \
    "numpy<2.2" \
    "matplotlib" \
    "pyyaml" \
    "tensorboard"
# TRL is used only as a post-hoc cross-check (Lab 04 optional); do not pre-install.
```

### 2.2 Secrets

Colab → 🔑 **Secrets** panel → add (do **not** hardcode in cells):

| Secret | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | **yes** | Pull Qwen3/OLMo-2 weights + No Robots / UltraFeedback / GSM8K datasets |
| `WANDB_API_KEY` | optional | Only if you enable wandb (metrics convention §6 works without it) |

```python
# Colab cell 2 — load secrets
from google.colab import userdata, runtime
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
# optional:
# os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

### 2.3 Data: Drive mount OR streaming

```python
# Colab cell 3a — option A: mount Drive (persistent across sessions; good for checkpoints)
from google.colab import drive
drive.mount("/content/drive")
LAB_DIR = "/content/drive/MyDrive/post_training_labs/lab00"   # adjust per lab
```

```python
# Colab cell 3b — option B: stream datasets (no disk use; good for big mixes)
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True)
```

Checklist: pick **one** per lab and record which in the lab README. Drive for anything you checkpoint; streaming for read-only mixes (SmolTalk, Tülu 3).

### 2.4 T4-safe settings (16 GB — non-negotiable)

```python
# T4 does NOT support bf16. Always:
import torch

AMP_DTYPE = torch.float16          # bf16 OFF on T4
assert not torch.cuda.is_bf16_available() or AMP_DTYPE == torch.float16

# batch sizing starting points for 0.6B (Qwen3-0.6B-Base) full-FT on T4:
#   per_device_batch=1  grad_accum=16  seq_len=1024   (SFT / RM)
#   DPO: per_device_batch=1 PAIR (policy+ref forwards → effective 4 fwd/bwd)
# If OOM: seq_len ↓ first, then grad_accum ↑. Never reduce below fp16 precision.
```

- [ ] fp16 autocast (never bf16), with `GradScaler` if you scale loss manually.
- [ ] Batch sizing recipe above; OOM handled by seq-len/accum, not precision.
- [ ] Colab disconnects → rerun from the pinned-install cell; checkpoints live in Drive if you used option 3a.

---

## 3. GT cluster profile (Tier B — A40 46 GB)

Verified facts baked in below: `/coc/scratch` is **one NFS volume visible identically on all six nodes**; **no sudo**; **foreign jobs share nodes**.

### 3.1 ⛔ HARD RULE — read before every launch

> **NO UNATTENDED JOB STARTS WITHOUT EXPLICIT USER GO-AHEAD.**
> A cluster launch (tmux session starting training) is a state-changing action on shared hardware. The user says "go" / "launch it" — not "this looks right." Every checklist item below up to and including §3.5 must be green, and the user must approve, before `tmux new-session` fires.

### 3.2 Shared venv (build once, reuse across ALL labs)

```bash
ssh vchopra@<node>   # mal | inara | jayne | shepherd for Tier B

# one-time: venv on scratch (no sudo anywhere)
python3 -m venv /coc/scratch/vchopra/venvs_post_training
source /coc/scratch/vchopra/venvs_post_training/bin/activate

# torch from the cu126 wheel index (A40 driver generation)
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install \
    "transformers==4.51.3" \
    "datasets==3.5.0" \
    "accelerate==1.6.0" \
    "numpy<2.2" \
    "matplotlib" \
    "pyyaml" \
    "tensorboard"

python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY   # expect: 2.x+cu126 True 8
```

- [ ] venv at `/coc/scratch/vchopra/venvs_post_training/` exists; torch reports the `cu126` build.
- [ ] All labs activate **this same venv** — never a second `~/venvs/`.

### 3.3 Directory layout (node-agnostic by construction)

Everything under `/coc/scratch/vchopra/post_training_labs/<lab>/` — written once, instantly cluster-global:

```bash
LAB=<labNN_slug>   # e.g. lab01_sft
mkdir -p /coc/scratch/vchopra/post_training_labs/$LAB/{data,checkpoints,logs,artifacts}
```

```
/coc/scratch/vchopra/post_training_labs/
└── <lab>/
    ├── data/            # datasets, tokenized caches
    ├── checkpoints/     # resumable checkpoints (§7)
    ├── logs/            # tmux/tee'd stdout, TensorBoard event files
    └── artifacts/       # JSONL metrics + samples, pulled back to notebook
```

Checklist:

- [ ] Data/checkpoints/logs are NEVER written to `$HOME` (node-local, invisible elsewhere).
- [ ] Results land in `artifacts/` so the notebook can plot from files regardless of platform (§6 contract).

### 3.4 Pre-launch occupancy check — EVERY time

Expect foreign jobs at any moment; check immediately before launch, on the exact node you'll use:

```bash
# quick overview (your ~/bin/gpu-status helper)
~/bin/gpu-status

# authoritative: what processes hold each GPU right now
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv

# pick a GPU with zero compute apps and note its index
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Checklist:

- [ ] `~/bin/gpu-status` run.
- [ ] `nvidia-smi --query-compute-apps` run **immediately before launch** — chosen GPU shows no compute apps.
- [ ] Acknowledged: **foreign jobs may land mid-run on your GPU.** Plan for modest batch sizes; if throughput suddenly halves, re-check the query before assuming a training bug.

### 3.5 Launch in tmux with pinned GPU

```bash
LAB=<labNN_slug>
GPU=<index from 3.4>
CFG=/coc/scratch/vchopra/post_training_labs/$LAB/configs/<lab>.yaml

tmux new-session -d -s $LAB
tmux send-keys -t $LAB "source /coc/scratch/vchopra/venvs_post_training/bin/activate" C-m
tmux send-keys -t $LAB "CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CFG \
    2>&1 | tee -a /coc/scratch/vchopra/post_training_labs/$LAB/logs/run_$(date +%Y%m%d_%H%M).log" C-m
tmux detach -t $LAB
```

Why each piece: `tmux` survives ssh disconnects · `CUDA_VISIBLE_DEVICES` pins the one GPU you verified empty (pinning, not "first free") · `tee` mirrors stdout into the log you poll.

### 3.6 Log polling (ssh tail)

```bash
# from laptop — poll the run
ssh vchopra@<node> "tail -n 40 /coc/scratch/vchopra/post_training_labs/$LAB/logs/run_*.log"

# watch live
ssh vchopra@<node> "tail -f /coc/scratch/vchopra/post_training_labs/$LAB/logs/run_*.log"

# is it still alive?
ssh vchopra@<node> "tmux ls; nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
```

Cadence: poll at ~30 min, then every few hours for multi-day runs (Labs 06/07/10). Every poll is also a foreign-jobs check.

### 3.7 TensorBoard port-forward recipe

```bash
# laptop terminal 1 (leave open)
ssh -N -L 6006:localhost:6006 vchopra@<node>
```

```bash
# on the node (inside your existing tmux or a second ssh session)
source /coc/scratch/vchopra/venvs_post_training/bin/activate
tensorboard --logdir=/coc/scratch/vchopra/post_training_labs/$LAB/logs --port=6006
```

Then open `http://localhost:6006` locally. (The JSONL metrics in `artifacts/` are the primary record per §6; TensorBoard is the live window.)

### 3.8 Tier C — `rivertam`/`simontam` (2080 Ti, 11 GB)

- **Tokenization / data prep only. Never training.** No training batch fits well in 11 GB; use these nodes for dataset downloads, tokenized-cache builds, and decontamination scans (e.g. Lab 02's 8-gram overlap).
- Same venv, same `/coc/scratch` layout — the NFS volume makes a tokenized cache built on Tier C instantly available to Tier B.

```bash
ssh vchopra@rivertam
source /coc/scratch/vchopra/venvs_post_training/bin/activate
mkdir -p /coc/scratch/vchopra/post_training_labs/$LAB/data
# ... run tokenization / data-prep scripts writing into .../$LAB/data/
```

---

## 4. Notebook ↔ script contract (every training lab)

Every training lab ships **both** of these, sharing **one** YAML config:

| Artifact | Purpose | Where |
|---|---|---|
| `notebook.ipynb` | Prototype, small-run sanity checks, plotting from artifacts | Colab (Tier A) |
| `train.py` | Exportable, config-driven, resumable cluster run | GT cluster (Tier B) |
| `configs/<lab>.yaml` | The single source of hyperparameters for both | repo / scratch |

```bash
# config-driven run — the ONLY supported invocation on the cluster
python train.py --config /coc/scratch/vchopra/post_training_labs/$LAB/configs/<lab>.yaml
```

Contract checklist:

- [ ] `train.py` reads **all** hyperparameters from the YAML; no magic numbers in code.
- [ ] The notebook imports the same config loader; a "small run" is a config with fewer steps/rows — not different code.
- [ ] After a cluster run, the notebook plots from `artifacts/` (JSONL + samples), per the Colab↔cluster contract in plan §3.

---

## 5. Metrics convention — JSONL per run (wandb optional)

Every run — notebook or `train.py`, any platform — **appends** one JSON object per log step to:

```
/coc/scratch/vchopra/post_training_labs/<lab>/artifacts/metrics_<run_name>.jsonl     # cluster
<Drive>/post_training_labs/<lab>/artifacts/metrics_<run_name>.jsonl                  # Colab
```

```python
# the one required helper pattern (both notebook and train.py use it)
import json, time
from pathlib import Path

METRICS_PATH = Path(ARTIFACTS_DIR) / f"metrics_{RUN_NAME}.jsonl"

def log_metrics(step: int, **metrics):
    record = {"step": step, "time": time.time(), **metrics}
    with open(METRICS_PATH, "a") as f:      # APPEND — never rewrite the file
        f.write(json.dumps(record) + "\n")
# usage: log_metrics(step, loss=..., grad_norm=..., val_loss=...)
```

- [ ] Append-only JSONL exists for **every** run (this is the "observability or it didn't happen" rule, principle 5).
- [ ] wandb is **optional** garnish on top — the JSONL is the platform-independent record, so notebooks always plot from artifacts.
- [ ] Each lab's "Metrics" list (§6 of the plan) defines which keys must move before the lab is done.

---

## 6. Checkpoint/resume + seeds

### 6.1 Fixed-seed protocol

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# call at the top of BOTH notebook and train.py; seed lives in the lab YAML
```

- [ ] Seed is a YAML field, identical across notebook and script, recorded in the run name: e.g. `metrics_sft_lr5e-6_seed42.jsonl`.
- [ ] Data order is seeded (DataLoader `shuffle=True` + `generator=torch.Generator().manual_seed(seed)`), so a rerun with the same config is comparable.

### 6.2 Checkpoint + resume (config-driven, on scratch/Drive)

```bash
# checkpoints land in (per §3.3)
/coc/scratch/vchopra/post_training_labs/<lab>/checkpoints/<run_name>/step_<N>/
```

```python
# required checkpoint payload (write periodically from train.py)
# torch.save({
#     "model": model.state_dict(),
#     "optimizer": optimizer.state_dict(),
#     "scheduler": scheduler.state_dict() if scheduler else None,
#     "step": step,
#     "config": cfg_dict,
# }, ckpt_path)
```

```bash
# resume — same config, checkpoint path added
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CFG --resume \
    /coc/scratch/vchopra/post_training_labs/$LAB/checkpoints/<run_name>/step_<N>.pt
```

- [ ] Resume restores model **and optimizer (and scheduler)** state — resume must continue metrics on the same JSONL (or a continuation file), not restart loss at step 0.
- [ ] **Resume-from-checkpoint must be exercised at least once in Labs 01, 06, and 07** (the long multi-day runs where interruption is likely). Lab 01 = first SFT run; Lab 06 = first RL loop; Lab 07 = capstone.

---

## 7. Setup checklist by lab (cross-reference: §5 dependency chain)

```
00 → 01 → 02 → 03 → 04 ─┐
            └──────────→ 05 → 06 → 07 → 08
                         03+06 → 09       07 → 10        03(+04) → 11 → 12
```

| When you reach | Do first |
|---|---|
| **Lab 00** (entry) | §1 (local clone), §2 (Colab profile + secrets) — Lab 00 is CPU/Tier A |
| **Lab 01** | §3.1 hard rule, §3.2 venv, §3.3 layout · §4 contract · §6 resume exercised here · §5 JSONL |
| **Lab 02** | Tier C (§3.8) is fine — CPU pandas/decontamination |
| **Lab 03 → 04** | Full §3 cluster launch checklist per run |
| **Lab 05** | CPU only — local or Colab; no cluster needed |
| **Lab 06 → 07 → 08** | Full §3 checklist; §6 resume exercised in 06 and 07; expect multi-day polling (§3.6) |
| **Lab 09 → 10 → 11 → 12** | Optional tier; §3 checklist applies to every Tier B run; Lab 11 loads ALL prior checkpoints — so §3.3 layout must have been kept clean throughout |

**Before any cluster launch, the compressed go/no-go:**
1. `~/bin/gpu-status` + `nvidia-smi --query-compute-apps` → target GPU empty.
2. tmux session named for the lab, `CUDA_VISIBLE_DEVICES` pinned, output tee'd to `logs/`.
3. Artifacts dir + JSONL path set; config YAML is the single source of truth.
4. **Explicit user go-ahead received.**
