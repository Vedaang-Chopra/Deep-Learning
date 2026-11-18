# GPT-2 LoRA From Scratch — Research Experiment Handbook

**Project ID:** `LORA-GPT2-01`  
**Level:** Post-training research engineer (P0 foundations → P1 controlled experiments → P2 original extension)  
**Model:** pretrained GPT-2 Small (`openai-community/gpt2`)  
**Main tools:** PyTorch, Hugging Face `transformers` for loading the model/tokenizer, `datasets` for data loading; PEFT **only after** the independent implementation works  
**Primary deliverable:** Reproducible code, tested LoRA implementation, adapter checkpoints, measured comparisons, and a research report.

> **Main question:** Under which conditions does LoRA approach full fine-tuning on GPT-2, and how do adapter rank, placement, learning rate, and dataset size change quality and training efficiency?

## 0. What to complete, in order

| Level | Step | Build / experiment | Completion evidence |
|---|---|---|---|
| P0 | 1 | Define hypotheses, outcomes, and scope | Research plan + fixed metrics |
| P0 | 2 | Set up repository and reproducibility | Environment lock + run manifests |
| P0 | 3 | Select, split, and tokenize data | Frozen splits + supervised-token masks |
| P0 | 4 | Inspect GPT-2 projection shapes | Architecture/parameter audit |
| P0 | 5 | Implement and unit-test `LoRAConv1D` | Output, gradient, freezing, merge tests |
| P0 | 6 | Inject independent adapters into GPT-2 | Correct adapter counts and zero-update equivalence |
| P0 | 7 | Build a PyTorch SFT loss and trainer | Correct causal shift, masking, optimizer step |
| P0 | 8 | Overfit 16–32 samples | Small-sample learning test |
| P0 | 9 | Run first full LoRA SFT | Adapter checkpoint + held-out evaluation |
| P0 | 10 | Save, reload, disable, and merge adapters | Round-trip / numerical equivalence checks |
| P0 | 11 | Validate against PEFT | Matched trainable counts and forward behavior |
| P1 | 12 | Establish frozen/full-FT/LoRA baselines | Comparable validation and test results |
| P1 | 13 | Tune learning rates separately | LR–NLL curves |
| P1 | 14 | Sweep LoRA rank | Rank vs accuracy/efficiency plot |
| P1 | 15 | Compare attention, MLP, and all placements | Parameter-matched layer-placement analysis |
| P1 | 16 | Measure GPU memory and throughput | Measured efficiency report |
| P2 | 17 | Investigate rank × data-size interaction | Controlled ablation + uncertainty |
| P2 | 18 | Optional rsLoRA/DoRA extension | Verified extension with fair comparison |
| Final | 19–25 | Budget plan, evaluation contract, failure analysis, final report, references, first sessions | Reproducible repository and research write-up |

**Stop rules:** Finish P0 before any P1 sweep. Run P2 only if P1 yields a reliable baseline and sufficient GPU budget.

---

## 1. Decide the research question before coding

### 1.1 Main hypotheses

- **H1 — Performance:** With the right rank, placement, and learning rate, LoRA can achieve held-out loss close to full fine-tuning on a modest SFT dataset.
- **H2 — Placement:** Attention-only adapters will underperform all-projection adapters; MLP placement may contribute more improvement than attention placement.
- **H3 — Capacity:** Small ranks may eventually learn more slowly than larger ranks as training data increases.
- **H4 — Efficiency:** LoRA will reduce trainable parameters and optimizer-state memory; total GPU memory and wall time depend on activation storage, batch size, precision, and implementation.

These are **hypotheses**, not guaranteed outcomes. They are motivated by *LoRA Without Regret*, which studied larger Llama/Qwen models. GPT-2 is a replication at a different scale, **not** a claim that those findings must hold.

### 1.2 Primary and secondary outcomes

- **Primary metric:** Held-out, assistant-completion-only negative log-likelihood (NLL), averaged **per supervised token**.
- **Secondary metrics:** Completion perplexity (`exp(NLL)`), train NLL, trainable parameters, peak GPU allocated memory, tokens per second, wall time, checkpoint bytes.
- **Behavior checks:** Fixed prompt set with generated completions, manually labeled error classes, and optional objectively graded task suite.
- **Never interpret a lower training loss alone as successful instruction following.**

### 1.3 Scope limits

- Start with **standard LoRA**; do not mix in QLoRA, DoRA, rsLoRA, or DPO at baseline.
- GPT-2 Small is a **base** model, not an instruction-tuned model. It may have weak conversational performance even after short instruction SFT.
- The primary research claim should concern held-out NLL and resource/optimization trade-offs. Report behavioral gains only if separately measured.

**Acceptance:** One-page `research_plan.md` containing questions, hypotheses, metric definitions, primary dataset, and comparison rules.

---

## 2. Set up a reproducible repository

```text
gpt2-lora-research/
├── README.md
├── requirements.txt               # pin versions or save a lockfile
├── configs/
│   ├── baseline.yaml
│   ├── lora_attn_r8.yaml
│   └── sweeps/
├── src/
│   ├── inspect_gpt2.py
│   ├── lora.py                    # custom modules
│   ├── inject_lora.py             # explicit GPT-2 target placement
│   ├── data.py                    # formats prompt, answer, masks
│   ├── loss.py                    # manual causal LM loss
│   ├── train.py                   # PyTorch-only optimization loop
│   ├── evaluate.py                # token-weighted eval + generations
│   ├── adapters.py                # save/load/merge/disable
│   └── profile.py                 # memory and throughput
├── tests/
│   ├── test_lora_math.py
│   ├── test_lora_gradients.py
│   ├── test_gpt2_injection.py
│   ├── test_masking.py
│   └── test_checkpoint.py
├── data/manifests/                # fixed IDs, hashes, filtered rows
├── runs/                          # per-run configs and metrics
├── figures/
└── reports/final.md
```

**Setup checklist**

- [ ] Pick compute device and record GPU, VRAM, PyTorch/CUDA versions.
- [ ] Pin package versions and pretrained model revision.
- [ ] Use a fixed seed for Python, NumPy (if used), and PyTorch. Document residual non-determinism.
- [ ] Write one command for a dry run (`--max_steps 2`) and one for evaluation.
- [ ] Save config + git commit + seeds with **every** run.

**Acceptance:** A fresh environment can load GPT-2 and run a forward pass without training.

---

## 3. Choose and freeze a dataset

### Primary study: instruction SFT on a fixed Alpaca subset

Use a reproducibly selected subset of `yahma/alpaca-cleaned` (subject to verifying its actual schema, terms, and accessibility). Use the fields `instruction`, optional `input`, and `output` to build prompt/completion examples.

**Starting sizes (budget-dependent, not universal requirements):**

| Split | Suggested count | Purpose |
|---|---:|---|
| Train | 4,000 | Main P0/P1 training |
| Validation | 500 | Learning-rate/model selection |
| Test | 500 | Final comparisons only |
| Pilot | 32 from train | Debug overfitting |

For the data-size study (Step 15), choose **nested train subsets** such as 500, 2,000, and 4,000 from the same shuffled train manifest. Never add validation or test examples to training.

**Prompt template (avoid adding new vocabulary tokens at first):**

```text
### Instruction:
{instruction}

### Input:
{input}                # omit this section when blank

### Response:
{output}<|endoftext|>
```

**Important details:**

1. Construct prompt IDs and answer IDs separately, then concatenate; this gives an exact prompt/answer token boundary for loss masking.
2. Add EOS at the end of the response.
3. Right-pad for training, set the tokenizer's `pad_token` to GPT-2 EOS **without adding new embedding rows**, and derive masks from sequence lengths / attention masks, not token-ID equality.
4. Build `labels` with `-100` on instruction/input/prompt tokens and on padding; retain answer/EOS token IDs as labels.
5. Keep max sequence length fixed (start with 256; adjust for truncation). Record how many answers are truncated.
6. Save **example IDs and rendered token sequences**. Check duplicates or near-duplicates across splits.
7. For generation evaluation, be consistent with padding side and generation settings; batching with GPT-2 often needs left padding or per-example generation.

**Separate optional validity suite:** A small synthetic instruction-to-JSON task with an exact checker. Treat it as a **separate task/branch**, not evidence that Alpaca SFT gained generic instruction-following ability.

**Acceptance:** Print 20 examples showing text, `input_ids`, supervised positions, EOS, and padding; all masking invariants pass.

---

## 4. Inspect GPT-2: know exactly where LoRA goes

Load `GPT2LMHeadModel` / `AutoModelForCausalLM` using Transformers. Do **not** use a high-level Trainer.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

for name, module in model.named_modules():
    if name.startswith("transformer.h.0"):
        print(name, type(module).__name__)
```

GPT-2 Small has **12 transformer blocks**. For each block, inspect:

| GPT-2 module | Role | Base weight shape (`in`, `out`) |
|---|---|---|
| `block.attn.c_attn` | Fused Q, K, V projection | `(768, 2304)` |
| `block.attn.c_proj` | Attention output projection | `(768, 768)` |
| `block.mlp.c_fc` | MLP expansion | `(768, 3072)` |
| `block.mlp.c_proj` | MLP contraction | `(3072, 768)` |

**Critical:** Hugging Face GPT-2 uses its `Conv1D` projection class with weights stored as `(in_features, out_features)`. This differs from `nn.Linear`, which stores `(out_features, in_features)`.

**Critical:** There are two different modules named `c_proj` in each block. Do **not** select all `c_proj` modules with an ambiguous string match when you want attention-only training.

**First task:** Record module path, input/output dimension, total base parameters, and intended adapter count.

**Acceptance:** `inspect_gpt2.py` creates a table and checks shape expectations before adapting anything.

---

## 5. Implement `LoRAConv1D` from scratch (P0 core)

### 5.1 Derive the equation

For GPT-2's `(in, out)` convention:

\[
Y = XW + b + \frac{\alpha}{r}(XA)B
\]

where

- `W`: frozen pretrained `(d_in, d_out)` matrix;
- `A`: trainable `(d_in, r)` matrix;
- `B`: trainable `(r, d_out)` matrix;
- `r`: low rank;
- `alpha / r`: standard LoRA scaling.

Equivalently, `W_effective = W + (alpha/r) * (A @ B)`.

### 5.2 A minimal implementation to write and test

```python
class LoRAConv1D(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base                       # GPT-2 Conv1D
        self.base.requires_grad_(False)
        d_in, d_out = base.weight.shape
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, d_out))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scaling * ((x @ self.A) @ self.B)
```

Use `torch.nn` and `torch`; add explicit rank/dtype/device checks in the production version. To keep device/dtype correct, inject before moving the full model to the training device or explicitly create adapters matching the base's device/dtype. Start with float32 and `lora_dropout=0` for unit tests.

### 5.3 Required unit tests

- [ ] `A` is `(in, rank)`; `B` is `(rank, out)`.
- [ ] Number of trainable parameters is `rank * (in + out)`.
- [ ] `B == 0` initially, so original and adapted outputs agree in **evaluation mode**.
- [ ] `base.weight.requires_grad == False` and `base.bias.requires_grad == False`.
- [ ] With a nonzero supervised loss, initial `B.grad` is generally nonzero; `A.grad` starts at zero because `B` is zero.
- [ ] After an optimizer update changes `B`, `A` can acquire nonzero gradients.
- [ ] The original `W` is bitwise unchanged after multiple training updates.
- [ ] Disabling LoRA restores the base result; compare in eval mode to remove dropout noise.
- [ ] Merging `W + scaling * A @ B` into `Conv1D` matches the unmerged result within an appropriate FP32 tolerance.
- [ ] Saving and restoring only `A`, `B`, and adapter metadata reproduces outputs.

**Do not wrap the base branch in `torch.no_grad()` during training:** Even frozen modules may need to propagate gradients to earlier trainable adapters.

**Acceptance:** `pytest tests/test_lora_math.py tests/test_lora_gradients.py` passes independently of GPT-2.

---

## 6. Inject adapters into the pretrained GPT-2

**Start with fused QKV-only** to understand the mechanism, then implement named placement presets.

```python
# Freeze *all* pretrained parameters first.
for p in model.parameters():
    p.requires_grad_(False)

for block in model.transformer.h:
    block.attn.c_attn = LoRAConv1D(block.attn.c_attn, rank=8, alpha=16)

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=3e-4)
```

### Placement presets

- `qkv_only`: `attn.c_attn` — **12 adapters**.
- `attention`: `attn.c_attn` + `attn.c_proj` — **24 adapters**.
- `mlp`: `mlp.c_fc` + `mlp.c_proj` — **24 adapters**.
- `all`: both attention and MLP sets — **48 adapters**.

Each adapted weight matrix gets a **different, independently learned A/B pair**, but there is **one final language-modeling loss** and one joint optimizer step.

**Acceptance:** Confirm wrapper counts and trainability. Verify base vs LoRA token logits match (with dropout disabled) immediately after adapter insertion. `lm_head`, embeddings, layer norms, and biases should remain frozen under the initial setup.

---

## 7. Write the training objective in PyTorch

### 7.1 Completion-only causal LM loss

The model receives the full instruction + answer token sequence. Only the answer (and EOS) contributes to the loss.

```python
logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
loss = F.cross_entropy(
    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
    labels[:, 1:].contiguous().view(-1),
    ignore_index=-100,
)
```

This explicitly performs GPT-style next-token shifting. **Do not additionally pass `labels=labels` and then shift the returned loss again:** `GPT2LMHeadModel` already handles causal shifting when provided labels.

### 7.2 Minimal training loop

```python
for batch in train_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    loss = completion_loss(logits, batch["labels"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
```

Build `completion_loss` from the previous cell. Extend this loop with gradient accumulation, evaluation intervals, and checkpointing; when accumulating, scale loss by accumulation steps and step only at the effective-batch boundary.

**Required training instrumentation**

- [ ] Training loss and validation loss (correct token-weighted aggregate).
- [ ] Number of supervised completion tokens per batch.
- [ ] Norms of `A`, `B`, gradients of `A`/`B`, and effective `A @ B` update.
- [ ] Optimizer LR and update count.
- [ ] Tokens/sec and elapsed time.
- [ ] Peak GPU memory (`torch.cuda.max_memory_allocated` after resetting stats).
- [ ] All parameters with `requires_grad=True` are in the optimizer; all base params remain frozen.

**Acceptance:** One forward/backward/optimizer step executes with finite loss and the expected gradients.

---

## 8. Overfit 16–32 examples before any real run

This is an engineering test, not a research result.

1. Select 16–32 fixed, short examples from the training set.
2. Disable LoRA dropout; keep sequence length short and verify supervision masks.
3. Train `qkv_only`, rank 8, for enough steps to make a visible loss reduction.
4. Generate on the **same training prompts** without sampling (`do_sample=False`).
5. If the model fails to memorize, inspect shifts, EOS handling, gradients, target modules, and LR before increasing dataset size.

**Acceptance:** Strong memorization of the tiny set or a clear, investigated architectural limitation; no unexplained zero gradients or NaNs.

---

## 9. First complete LoRA fine-tuning run

Use one fixed, documented configuration. These settings are **starting points to test**, not known optima:

| Setting | Initial choice |
|---|---|
| Model | `openai-community/gpt2` |
| Data | 4,000 train / 500 val / 500 test |
| Target modules | `attention` (24 modules) |
| Rank `r` | 8 |
| Alpha | 16 |
| LoRA dropout | 0.0 initially |
| LR | `3e-4` (pilot; tune in P1) |
| Optimizer | AdamW (explicitly log `betas`, `eps`, weight decay) |
| Max sequence length | 256 |
| Effective batch size | 32 when hardware permits |
| Epochs | Start with 1–3; validate regularly |
| Selection | Best validation completion NLL, not test |

Record a frozen-GPT-2 baseline before training. Save the LoRA adapter and compare:

1. Frozen pretrained model on fixed test set.
2. Adapted GPT-2 on the **same** examples.
3. Fixed-prompt before/after generations with identical decoding settings.

**Acceptance:** Adapter trains reliably; base weights unchanged; validation metric and generations recorded; no claim of improvement without held-out evidence.

---

## 10. Implement robust adapter checkpointing and weight merging

Build `adapters.py` with the following API:

```text
save_adapter(model, path, config, base_model_revision)
load_adapter(base_model, path)
disable_adapter(model)
enable_adapter(model)
merge_adapter(model)           # returns/effectively constructs merged base weights
```

Tests:

- [ ] Adapter-only checkpoint includes the A/B tensors, module paths, rank, alpha, and revision metadata.
- [ ] Reload the **same** base model revision, inject identical modules, restore A/B, and reproduce logits.
- [ ] Disable → reproduce frozen base logits in eval mode.
- [ ] Enable → reproduce adapter logits in eval mode.
- [ ] Merge → reproduce unmerged adapter logits within tolerance.
- [ ] After merge, verify no remaining adapter branch is added a second time.
- [ ] Save/restore on a new Python process, not only the original notebook kernel.

**Acceptance:** No silent mismatch in weights, names, dtypes, or inference results.

---

## 11. Validate against Hugging Face PEFT

Do this **after** your own implementation passes tests.

1. Create a fresh GPT-2 model from the same revision.
2. Add PEFT LoRA with the **same** rank, alpha, targeted modules, dropout, and bias policy.
3. For GPT-2 `Conv1D`, verify `fan_in_fan_out=True` behavior/weight orientation in PEFT.
4. Compare total trainable parameter counts.
5. Copy or match the adapter tensors (take care with transposes and naming) and compare forward logits in eval mode.
6. Optionally compare loss and gradients on one identical batch in float32.
7. Compare a full training run only after the one-batch equivalence passes.

**Acceptance:** Explain any observed output/gradient differences; avoid expecting bitwise equality when kernels and training order differ.

---

# P1 — Controlled experimental study

## 12. Frozen, full fine-tuning, and LoRA baselines

Run from the **same untouched pretrained checkpoint**, on the same fixed dataset and evaluation protocol:

- **B0 — Frozen GPT-2:** Evaluation only.
- **B1 — Full FT:** All pretrained parameters trainable, including GPT-2's LM head and embeddings as normally tied/configured.
- **B2 — Attention LoRA:** Frozen base + attention projection adapters.
- **B3 — All-projection LoRA:** Frozen base + all 48 projection adapters.

**Fairness:** Full FT and LoRA usually have different optimal LRs. Use the next step's LR sweeps before interpreting results; do not claim LoRA is worse because both were run at one arbitrary LR.

If full FT does not fit in memory, use smaller microbatches, gradient accumulation, mixed precision, and/or activation checkpointing, recording the differences. If full FT still cannot run, **label the baseline unavailable** rather than substituting an incomparable result.

**Acceptance:** A baseline table with starting checkpoint, data version, train tokens, validation/test NLL, runtime, memory, and checkpoint size.

## 13. Learning-rate sweep (remove the biggest confound)

**Question:** Does the best learning rate differ for GPT-2 full FT and LoRA?

Suggested pilot grids:

| Method | Candidate LRs |
|---|---|
| Full FT | `1e-5`, `3e-5`, `1e-4` |
| LoRA | `1e-4`, `3e-4`, `1e-3` |

Use a short pilot with fixed train tokens. Drop unstable settings; refine around the best **validation** result. Preserve the same optimizer family, precision, batch, and evaluation metric; record any method-specific tuning decisions.

**Outputs:** LR vs validation NLL plot, best LR for each method, early loss-vs-steps curves.

**Acceptance:** Baseline comparisons use tuned LRs or explicitly state they are untuned matched-recipe comparisons.

## 14. Rank sweep (capacity experiment)

**Question:** At what rank does extra capacity stop improving held-out loss for this dataset?

1. Fix placement to `all` (or keep `attention` as a separately named comparison).
2. Test ranks **2, 8, 32** first. Expand to 1, 4, 16, 64 if results justify the cost.
3. Start with a fixed `alpha=16`, standard scaling `alpha/r`, and the tuned/validated LR protocol.
4. Hold dataset IDs, evaluation, training-token budget, sequence length, and effective batch fixed.
5. Log trainable parameters, validation NLL, best-checkpoint test NLL, peak memory, tokens/s, and elapsed time.

**Outputs:**

- NLL vs rank.
- NLL vs number of trainable parameters.
- NLL vs GPU-hours (or runtime with exact GPU named).
- Loss trajectories, not just final scores.

**Interpretation:** More capacity does not necessarily imply better held-out results; assess overfitting and LR sensitivity.

## 15. Layer-placement experiment (the main paper-inspired ablation)

**Question:** Does MLP LoRA outperform attention-only LoRA on GPT-2 at a matched adapter parameter budget?

For GPT-2 Small, with the **four weight types per transformer block** and no adapted biases:

| Placement | Adapters | Trainable adapter parameters per rank `r` |
|---|---:|---:|
| Fused QKV only | 12 | `36,864 × r` |
| Attention (`c_attn` + attention `c_proj`) | 24 | `55,296 × r` |
| MLP (`c_fc` + MLP `c_proj`) | 24 | `92,160 × r` |
| All four | 48 | `147,456 × r` |

**Equal-rank comparison:** Attention vs MLP vs all, all at `r=8`. This isolates the *configuration at equal rank* but **not** equal parameter budget.

**Approximately budget-matched comparison:**

- Attention, `r=20`: **1,105,920** adapter parameters.
- MLP, `r=12`: **1,105,920** adapter parameters.
- All projections, `r=8`: **1,179,648** adapter parameters (about 6.7% more).

The first two are **exactly parameter-matched** under these projection shapes. The third is a near-match; disclose the difference.

Use independently tuned LRs for important placement comparisons; do not present unmatched parameter counts as evidence that layer location alone caused the result.

**Outputs:** Validation/test NLL; NLL per million adapter parameters; loss curves; gradient/update magnitude per module type.

**Acceptance:** A defensible conclusion about which placement worked best *for GPT-2, this dataset, and this budget*. Compare directionally against Schulman's larger-model study without asserting generality.

## 16. Compute, memory, and optimizer study

**Question:** What does LoRA actually save on this hardware?

- Keep training examples, max length, effective batch, precision, and reporting method fixed.
- Measure allocated peak GPU memory after warm-up and resetting peak counters; do not conflate reserved and allocated memory.
- Count trainable parameters and optimizer state tensors.
- Compare adapter-only and full-model checkpoint bytes.
- Measure training tokens/sec and wall time, including vs excluding evaluation/checkpoint overhead as separate figures.
- Repeat with activation checkpointing enabled as a **separate factor**. LoRA and checkpointing save different forms of memory.
- Record whether optimizer state, activations, or temporary buffers dominate; do not infer memory from parameter count alone.

**Outputs:** Bar charts for memory and speed, plus a clear hardware/software/config table.

**Acceptance:** Explain why fewer trainable parameters do **not** imply proportionally lower total GPU memory or proportionally higher throughput.

---

# P2 — A distinct research contribution

## 17. Rank × dataset-size interaction

**Specific research hypothesis:** Low-rank adapters stop reducing training loss efficiently at smaller data scales than high-rank adapters; the threshold depends on whether adapters cover attention alone or all projections.

**Minimal feasible design (expand only if compute permits):**

- Dataset sizes: **500, 2,000, 4,000** nested examples from one fixed train split.
- Placements: **attention**, **all**.
- Ranks: **4**, **32**.
- Same validation/test sets across all cells.
- Control evaluation frequency, prompt formatting, training-token budget **within each size**, and LR-selection procedure.
- Run important comparisons with **2–3 random seeds** if feasible; report the number actually used.

This is 3 × 2 × 2 = **12 main configurations** before extra seeds and LR pilots. If this is too expensive, begin with attention/all at ranks 4/32 using only small and large data (8 configurations), then expand based on the result.

**Primary plots:**

1. Train NLL vs number of **supervised tokens**.
2. Test NLL vs train-set size, grouped by rank and placement.
3. Gap to tuned full FT (where full FT exists) vs dataset size.
4. Training loss curves annotated with optimizer budget and LR.

**How to interpret outcomes:**

- Train loss stalls and higher rank improves train loss → possible capacity/optimization bottleneck; check whether LR is adequately tuned.
- Train loss improves, test loss worsens → generalization/overfitting issue.
- Differences vanish after LR sweep → original conclusion was likely an optimization confound.
- No difference at GPT-2 scale → useful negative finding; report measurement uncertainty rather than inventing a trend.

**Acceptance:** At least one falsifiable conclusion with counterexamples and a follow-up experiment that distinguishes alternative explanations.

## 18. Optional method extension — only after standard LoRA is validated

Pick **one**, not every method:

- **rsLoRA:** Replace scaling `alpha/r` with `alpha/sqrt(r)`; at different ranks, retune LR and disclose the changed scaling so it is not confused with a rank-only experiment.
- **DoRA:** Add a learned magnitude component and adapt weight direction; independently verify initialization, normalization axis, gradients, and merge correctness before comparison. Use the Raschka article as implementation study material, not as evidence that DoRA must win on GPT-2.
- **Adapter initialization:** Compare Gaussian/Kaiming-based A initialization under controlled seeds and norms.

**Acceptance:** Tests + one controlled ablation, not just a working library option.

---

## 19. Run plan: do not launch an uncontrolled 50-run sweep

### Minimum credible completion (P0)

1. One frozen baseline evaluation.
2. One `qkv_only`, rank-8 overfit test.
3. One `attention`, rank-8 main SFT run.
4. One adapter save/load/disable/merge verification.
5. One PEFT parity check.

### Strong research-engineer project (P0 + P1)

1. Tuned full-FT baseline if hardware permits.
2. Tuned attention and all-projection LoRA runs.
3. Three-rank all-projection sweep.
4. Equal-parameter attention-vs-MLP placement comparison.
5. Performance/resource dashboard, two or more repeat seeds for headline claims.

### Differentiated extension (P2)

Run the reduced or full rank × data-size study **only after** P1. A paper reproduction with a careful negative result is better than an unfocused collection of variants.

---

## 20. Evaluation contract and logging schema

**Before running any experiment:** freeze `test_manifest.json`, tokenization/template code version, generation settings, and metric definitions. Select the checkpoint by validation NLL **only**.

### Per-run manifest (example)

```yaml
run_id: attn_r8_seed42
base_model: openai-community/gpt2
base_revision: <exact commit SHA>
dataset: yahma/alpaca-cleaned
dataset_manifest: data/manifests/v1.json
seed: 42
method: lora
placement: attention
rank: 8
alpha: 16
scaling: alpha_over_rank
max_length: 256
learning_rate: 0.0003
optimizer: adamw
effective_batch: 32
train_token_budget: <measured count>
checkpoint_selection: best_validation_completion_nll
```

### Results table (fill with measured results only)

| Run | Method / placement | Rank | LR | Trainable params | Val NLL ↓ | Test NLL ↓ | Peak VRAM | Tokens/s | Seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | GPT-2 | — | — | 0 | TBD | TBD | TBD | TBD | — |
| FullFT | All trainable | — | tuned | TBD | TBD | TBD | TBD | TBD | 42 |
| Attn8 | LoRA / attention | 8 | tuned | 442,368 | TBD | TBD | TBD | TBD | 42 |
| MLP12 | LoRA / MLP | 12 | tuned | 1,105,920 | TBD | TBD | TBD | TBD | 42 |
| Attn20 | LoRA / attention | 20 | tuned | 1,105,920 | TBD | TBD | TBD | TBD | 42 |
| All8 | LoRA / all | 8 | tuned | 1,179,648 | TBD | TBD | TBD | TBD | 42 |

**NLL accounting:** For each supervised target token, accumulate summed negative log-probability across all eval batches and divide by the total number of supervised tokens. Do not average per-batch losses equally when batches contain different supervised-token counts.

**Generation review:** Fix 30–100 prompts and decoding settings, save outputs from all methods, annotate correctness/format/irrelevance/repetition. Do not declare a win from hand-picked examples.

**Repeatability:** Repeat meaningful comparisons over seeds; for small differences, give uncertainty or say the difference is inconclusive.

---

## 21. Key failure modes to deliberately test

| Symptom | First diagnostic |
|---|---|
| LoRA outputs differ from frozen GPT-2 at initialization | `B=0`, wrapper bias/base call, eval mode, dtype |
| Shape mismatch in GPT-2 | `Conv1D` stores `(in, out)`; not `nn.Linear` convention |
| More adapters than expected | `c_proj` exists in both attention and MLP |
| A gradient is zero at the first step | Expected when `B` initializes to zero |
| All A gradients stay zero | Check B updates, optimizer param list, loss masking |
| Frozen `W` changes after optimizer step | Freeze audit; check accidental merge/in-place edits |
| Loss near zero immediately | Check target leakage, prompt/answer masks, duplicate data |
| Loss decreases, generation remains poor | Instruction capability, teacher forcing vs generation, EOS and decoding |
| Full FT is much worse than LoRA | Tune LRs independently; check schedules and overfitting |
| Adapter merge changes output greatly | Transpose/scale/bias mistake; compare eval-mode FP32 outputs |
| Memory barely decreases with LoRA | Activations, sequence length, checkpointing, optimizer-state breakdown |
| Model appears better on test after tuning | Test leakage; move selection back to validation |

---

## 22. Research report template

Create `reports/final.md` with exactly these substantive sections:

1. **Research question and hypotheses.** What did you expect? Why?
2. **Relevant prior work.** Original LoRA, Raschka's implementation discussion, *LoRA Without Regret*.
3. **System under study.** Model revision, GPT-2 projections, LoRA equations, parameter formula, software/hardware.
4. **Data and evaluation.** Manifest, split policy, formatting, token masks, truncation, loss aggregation.
5. **Implementation correctness.** Tests, gradient checks, frozen weights, merged equivalence, PEFT parity.
6. **Experimental design.** Controls, intervention, learning-rate policy, compute budget, seeds.
7. **Results.** All curves and tables, including failed experiments.
8. **Failure analysis.** Inspect at least 20 output examples and training anomalies.
9. **Interpretation.** Is the hypothesis supported? What confounds remain?
10. **Limitations.** GPT-2 scale, instruction-following weakness, limited benchmark and compute.
11. **Reproduction.** Exact commands, configs, git SHA, model revision, data manifest, result paths.
12. **Next research question.** The smallest follow-up that resolves the biggest uncertainty.

**Resume/interview evidence:** Be able to explain (a) why `A` initially has zero gradients, (b) why GPT-2 `Conv1D` needs careful orientation, (c) why layer-matched rank is not parameter-matched, and (d) why rank comparisons without LR tuning can mislead.

---

## 23. Definition of done

### P0 — can implement and debug LoRA

- [ ] Custom `LoRAConv1D` passes all shape, gradient, freezing, and merge tests.
- [ ] Correct module injection in GPT-2; trainable count matches analytical formula.
- [ ] Custom, correctly masked PyTorch SFT loop trains GPT-2.
- [ ] Tiny overfit works; held-out evaluation runs.
- [ ] Adapter save/load/disable/merge works across processes.
- [ ] PEFT configuration/output parity explained.

### P1 — can design a rigorous comparison

- [ ] Full FT vs LoRA baselines measured with reasonable LR tuning.
- [ ] Rank sweep completed and plotted.
- [ ] Layer-placement study includes parameter-matched attention/MLP comparison.
- [ ] Memory and throughput measured rather than estimated.
- [ ] Results tied to named seeds, configs, fixed splits, and examples.

### P2 — can demonstrate research thinking

- [ ] Test a new rank × data-size or scaling hypothesis.
- [ ] Include at least one unexpected result or negative finding.
- [ ] Rule out one plausible confound with a targeted follow-up.
- [ ] Write a concise report that clearly limits claims to supported evidence.

---

## 24. Sources and reading order

1. **Umar Jamil's LoRA implementation / PyTorch parametrization notebook** (the MNIST-based notebook supplied with this project): study parameterization, freezing, trainability, toggling adapters. Re-implement the same ideas in GPT-2 `Conv1D`, not MNIST.
2. **Original LoRA paper — Hu et al.** https://arxiv.org/abs/2106.09685 — equation, freezing, rank, scaling, initialization.
3. **Sebastian Raschka — Implementing LoRA and DoRA from scratch** https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch — wrapper pattern, merge, DoRA extension.
4. **Thinking Machines Lab — LoRA Without Regret (Schulman et al., 2025)** https://thinkingmachines.ai/blog/lora/ — rank, attention vs MLP placement, data capacity, LR and batch effects. Use as a *hypothesis source*, not as GPT-2 results.
5. **Hugging Face GPT-2** https://huggingface.co/docs/transformers/model_doc/gpt2 — model/tokenization conventions.
6. **Hugging Face PEFT LoRA reference** https://huggingface.co/docs/peft/main/package_reference/lora — target modules, `fan_in_fan_out`, scaling, adapter controls.
7. **PyTorch parametrizations tutorial** https://pytorch.org/tutorials/intermediate/parametrizations.html — optional alternative injection approach.
8. **Project handbook:** `post_training_experiment_handbook.md` → E09 (LoRA), E10 (Full FT vs LoRA), E12/E13 (evaluation, profiling, reproducibility).

**Recommended order of study:** LoRA paper §4 → independent `LoRAConv1D` and tests → GPT-2 integration → SFT loop → PEFT verification → *LoRA Without Regret* sections on rank, layer placement, and LR → research ablations → optional DoRA.

---

## 25. First three working sessions

### Session 1 — inspect and understand

- [ ] Load GPT-2 and print all attention/MLP projection names and shapes.
- [ ] Derive `r * (in + out)` for each layer type and total for each placement preset.
- [ ] Write the LoRA forward equation using GPT-2's `(in, out)` convention.
- [ ] Save an untouched baseline checkpoint reference and sample logits.

**Exit:** `inspect_gpt2.py` output and a short note: *Exactly what will change under LoRA?*

### Session 2 — implement and prove correctness

- [ ] Write `LoRAConv1D` without PEFT.
- [ ] Unit-test zero-update equivalence, A/B gradients, frozen W, and weight merging.
- [ ] Wrap `attn.c_attn` in all 12 transformer blocks.
- [ ] Show 12 separate A/B pairs and optimizer parameter list.

**Exit:** All LoRA math/gradient/injection tests pass.

### Session 3 — train and debug

- [ ] Implement tokenization and assistant-only label masks.
- [ ] Compare your explicit shifted CE with Hugging Face's built-in GPT-2 loss on one identical batch.
- [ ] Train on 16–32 examples until memorized.
- [ ] Evaluate the frozen and adapted checkpoints on the same prompts.

**Exit:** Verified end-to-end training; only then move to the full 4,000-example run.

> **Rule for the whole project:** Derive → implement → unit-test → train → evaluate → ablate → explain. Do not advance based only on a notebook that executes without errors.
