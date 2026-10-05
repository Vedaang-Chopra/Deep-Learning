# Lab 01 — SFT From Scratch (★★☆ · CORE)

> Build the base→assistant transition yourself: prompt-masked causal-LM fine-tuning with a
> hand-written training loop, plus two from-scratch checkpoints — **a LoRA layer you implement
> by hand** and (optional) **naive sequence packing**. No HF `Trainer`.

**Derived from:** the answer key at `rlhf-book/code/instruction_tuning/` (`train.py`, `utils.py`,
`configs/sft_olmo2_1b.yaml`). Do not open it until your version trains.

---

## 1. Why this lab

Every later lab in this curriculum assumes the base→assistant transition is something you *caused*.
SFT is supervised learning, so the mechanism is simple; what matters is owning the three details
that most post-training bugs trace back to:

1. **Which token positions carry gradient** (prompt masking, EOS supervision).
2. **The practical loop mechanics** (grad accumulation, clip-grad-norm, bf16 autocast, eval).
3. **Parameter-efficient adaptation** (LoRA from scratch: what A·B actually buys and costs).

## 2. Prerequisites

- **Lab 00 — Data & Chat Templates** (prompt-masked labels intuition must already exist).
- **Lecture 2 part 1 (IFT)** — Nathan Lambert's course:
  [course page](https://rlhfbook.com/course/) ·
  [Lecture 2 recording](https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3).
- **Chapter 4 — Instruction Fine-Tuning**: <https://rlhfbook.com/c/04-instruction-tuning>.

## 3. Assignment checklist

Implement everything from scratch — **no HF `Trainer`, no TRL**:

- [ ] Dataset + collate + DataLoader with prompt-masked batching
      (`build_prompt_masked_labels`, `SFTDataset`, `collate_sft` in `starter.py`).
- [ ] bf16 autocast training loop: gradient accumulation, clip-grad-norm,
      periodic val loss, JSONL metrics, fixed 6-prompt generation panels
      (`train_step`, `run_training_loop`).
- [ ] **From-scratch checkpoint:** implement a LoRA Linear layer yourself
      (`LoRALinear`) — low-rank A·B wrap of a frozen linear with scaling α/r.
      Verify it matches frozen-linear + adapter on a random input, then
      SFT Qwen3-0.6B with your own LoRA on Colab.
- [ ] Greedy vs sampled generation harness; step-0 vs final panel on the fixed prompts.
- [ ] *(optional)* Naive sequence packing (`pack_sequences`): concatenate examples with
      position_ids reset per document; measure throughput gain vs unpadded batching and
      write up cross-contamination risk (documents attending across boundaries when
      block-diagonal attention is skipped).

Each function carries its full contract in its docstring in `starter.py`. The starter contains
**no solutions** — only signatures, contracts, TODOs, shape comments, and
`raise NotImplementedError`.

## 4. Model

| Tier | Model | Notes |
|---|---|---|
| A (Colab T4) | `Qwen/Qwen3-0.6B-Base` | Full FT feasible; LoRA trivial |
| B (GT A40) | `Qwen/Qwen3-1.7B-Base` | Full FT / overnight runs |
| Answer-key parity | `allenai/OLMo-2-0425-1B` (+ `-SFT` template donor) | Only for diffing against the reference W&B run |

Set via `model_name` / `chat_template_source` in `configs/01_sft_from_scratch.yaml`.
Note the deviation-vs-answer-key record: our default is Qwen3 (per §4 register), OLMo-2 is kept
as the comparable arm.

## 5. Dataset

- Primary: `HuggingFaceH4/no_robots` (~9.5k human-written rows) — subset `500 → 2k → 9.5k`.
- Extension: `HuggingFaceTB/smoltalk` (`smol-magpie-ultra[:20000]`) for scale comparison.
- Every conversation row uses the `messages` field; the final turn must be `role == "assistant"`
  or the row is skipped (no trainable tokens).

## 6. Components (in `starter.py`)

| Component | Contract (summary) | Shapes |
|---|---|---|
| `IGNORE_INDEX` | label value at positions that get no gradient | `-100` |
| `DEFAULT_SAMPLE_PANEL_PROMPTS` | the fixed 6-prompt panel used across the run | 6 strings |
| `SFTBatch` | immutable container: `input_ids`, `attention_mask`, `labels` | `(B,T)` long each |
| `build_prompt_masked_labels` | render chat template twice (prompt-only w/ generation prompt, full); mask every position ≤ prompt length to -100; supervise assistant tokens incl. EOS | ids/labels length `T_full` |
| `collate_sft` | pad batch to max T, rectangular output; attention_mask 0 at pads; labels -100 at pads | see docstring |
| `make_dataloader` | wire dataset+collate over the config | — |
| `compute_loss` | shifted CE over unmasked label positions only (logits[:, :-1] vs labels[:, 1:], ignore_index=-100). **Do not compute CE over masked positions.** | logits `(B,T,V)` → scalar |
| `LoRALinear` | frozen base linear `W` (no grad) + trainable low-rank adapters A∈(r,in), B∈(out,r), forward adds scaled α/r·B(A(x)); init per contract docstring; optional merge | x `(N,in)` → `(N,out)` |
| `apply_lora` | swap target nn.Linear modules for wrapped versions, freeze originals | — |
| `pack_sequences` | *(optional)* greedy concatenate encoded examples into `max_packed_len` bins, emit position_ids restarting at each boundary | see docstring |
| `train_step` | one optimizer step worth of micro-batches: accumulate grads ÷ accumulation steps, clip to `max_grad_norm`, step opt+scheduler, return avg loss & grad norm | scalar loss |
| `evaluate_val_loss` | same loss math under no_grad over val split | float |
| `generation_panel` | render+generate on `DEFAULT_SAMPLE_PANEL_PROMPTS`, greedy vs sampled | text list |
| `run_training_loop` | epochs × dataloader calling `train_step`; JSONL metrics line per optimizer step; periodic panels + val loss | metrics path |

## 7. Experiments

From the plan §6 Lab 01:

- lr ∈ {`1e-6`, `5e-6`, `2e-5`}
- epochs ∈ {1, 3}
- data size ∈ {500, 2000, 9500} rows
- LoRA r ∈ {8, 32} vs full fine-tune (0.6B on Colab)
- *(optional)* packed vs unpacked throughput under identical token budget

Metrics that must move: train/val loss, grad norm, response length, generation panels.

## 8. Questions (answer in notebook markdown)

1. Why mask prompt tokens instead of computing CE over the whole sequence?
2. What is the early loss cliff (first ~50 steps) composed of — what tokens dominate it?
3. How does LoRA capacity (rank r) limit how far the behavior shifts? Rank 8 vs 32?
4. When is packing unsafe? What would a model learn from documents bleeding into each other?
5. What breaks if EOS isn't part of the supervised positions?

## 9. Debugging challenges

- Generations never stop → which position did you forget to supervise?
- Loss ≈ 0 after warmup → which positions did you mask away by accident?
- NaNs mid-run → order of autocast vs backward vs accumulation?
- Val loss rises while train loss falls → data size vs epoch tradeoff?

## 10. Completion criteria

Per plan §6 Lab 01 "done when":

- Base-vs-SFT transcripts on held-out prompts.
- A narrated loss plot (annotate the early cliff and any instability).
- LoRA-vs-full-FT comparison table (lr, steps, val loss, panel quality verdict).

## 11. Compare against the answer key

Open **only after your version trains**:

- `rlhf-book/code/instruction_tuning/train.py` — accumulation/clipping/scheduler structure
- `rlhf-book/code/instruction_tuning/utils.py` — `_encode_batch` (lines ~148–188),
  `_collate` (202–219), `compute_loss` (136–145), `make_lr_scheduler` (346–357)
- `rlhf-book/code/instruction_tuning/configs/sft_olmo2_1b.yaml`

Diff your decisions: where did you choose differently and why? Record deviations here.

## Layout

```
01_sft_from_scratch/
├── README.md                     ← this file
├── starter.py                    ← stubs to fill (no solutions inside)
├── notebook.ipynb                ← prototype/small-run workspace
├── configs/
│   └── 01_sft_from_scratch.yaml  ← single config shared by notebook & exported train.py
└── tests/
    ├── conftest.py               ← fake tokenizer fixture (numpy only)
    └── test_01_sft_from_scratch.py  ← contract tests (pass before AND during implementation)
```

Run tests locally (no GPU/network needed):

```bash
python3 -m pytest tests/ -q
```
