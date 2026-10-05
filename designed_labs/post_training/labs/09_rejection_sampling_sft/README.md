# Lab 09 — Rejection Sampling → SFT

**Difficulty:** ★★☆ · ~1–2 days · **optional**
**Concepts:** best-of-N selection, reward-model-based data filtering vs. matched random controls, rejection-sampling SFT (RFT) loops, exact-match evaluation.

> [!NOTE]
> This directory is a **scaffold**. `starter.py`, `notebook.ipynb`, and
> `configs/09_rejection_sampling_sft.yaml` define signatures, contracts, and
> fixtures — every function marked `TODO` raises `NotImplementedError`. You
> implement the mechanisms. The reference implementation lives in the RLHF Book
> repo (`rlhf-book/code/rejection_sampling/`) and is meant to be opened **only
> after your version trains** (design principle #1 of the curriculum plan).

## Prerequisites

- **Lecture 2 — RS section** ("Instruction tuning → Rejection sampling"), course deck
  [`lec2-chap4-5-9`](https://rlhfbook.com/course) on rlhfbook.com/course.
- **Chapter 9 — Rejection Sampling**, <https://rlhfbook.com/c/09-rejection-sampling.html>
  (selection rules @eq:rs_selection_per_prompt and @eq:rs_topk_selection).
- **Your own Lab 03 Bradley-Terry RM is the scorer.** This lab depends on the
  checkpoint you trained in Lab 03 (`Qwen3-0.6B` backbone + BT head). Using a
  scorer *you* built makes scoring failures diagnosable — you know its biases
  from the Lab 02 audit (length bias, near-ties, formatting artifacts).
  An optional production-RM comparison arm is included in the config
  (`nvidia/AceMath-7B-RM`, the answer key's choice).
- **Lab 06/07 rollout habits** are helpful but not required: this lab generates
  its own N=8 rollouts per prompt rather than reusing RL rollouts.

## What you build

The pipeline (mirrors the answer key's five stages):

1. **Generate** `N = 8` completions per GSM8K train prompt (1k prompts) with
   `Qwen/Qwen3-1.7B` (or 0.6B for Tier A — record the deviation here if you swap).
2. **Score** every completion with **your Lab 03 RM** (optional second pass:
   AceMath-7B-RM). Stage 1+2 are shared across all four arms → one cache file.
3. **Select** (prompt, completion) pairs via each of the four strategies below.
4. **SFT** the base model on each selected subset (identical SFT hyperparameters).
5. **Evaluate** greedy exact-match accuracy on the GSM8K test slice (200 prompts)
   and produce the **strategy-vs-matched-random-control table**.

### The four arms (all four configs; shared generation parameters)

| Arm | Keeps | Intuition |
|---|---|---|
| `top_per_prompt` | Argmax-reward completion per prompt (M pairs) | Classic RS: one chosen demo per question, full prompt coverage |
| `random_per_prompt` | One completion per prompt, uniform at random, seeded | Matched control: same size & coverage, ignores the RM |
| `top_k_overall` | Top-K over the flat M×N matrix (K pairs; repeats prompts allowed) | Lets the RM concentrate data on easiest-to-score prompts |
| `random_k_overall` | K uniform-random pairs from the flat pool, same K, seeded | Matched control: same budget & structure, ignores the RM |

Each `top_*` arm is paired with a `random_*` arm identical in sample budget and
structural shape, so any accuracy gap isolates exactly one thing: **does the RM
actually know which completion is good?**

### Cache contract

Stage 1/2 output one JSONL cache; each line is one scored prompt record:

```json
{"question": "...", "answer": "72", "completions": ["...", "..."], "rewards": [0.71, ...]}
```

Any change to policy model, scorer, dataset slice, sampling params, seed, or
`num_completions_per_prompt` invalidates the cache (hash the params, like the
answer key does). `starter.validate_rollout_records` enforces the record shape.

### Deliverables

- [ ] Working scorer bridge: your Lab 03 RM scores a batch of (prompt, completion) pairs.
- [ ] All four selection strategies + seeded controls implemented from scratch.
- [ ] GSM8K answer extraction + exact-match evaluator (greedy decode, no network surprises).
- [ ] **Strategy-vs-control table**: per arm — #pairs, SFT final/loss, test exact-match, Δ vs matched control.
- [ ] Written explanation of the repo's finding and why it happens (see Motivation below).
- [ ] Metrics appended as JSONL per run (curriculum convention §7.5) so the notebook plots from artifacts.

## Scoring diagnostics (methodology from `diagnostics.py`)

Before training, sanity-check that your RM can rank completions at all:

1. **Reward histogram** — do correct and incorrect completions occupy different
   regions of reward space?
2. **Per-row winrate on decidable prompts** — prompts with a mix of correct/incorrect
   completions only; how often does argmax(reward) pick a correct completion vs a
   random pick?
3. **Best-of-N sweep (N = 1..K)** — fraction of prompts with ≥1 correct completion
   in the top-N by reward, vs a random baseline.
4. **`decidable_fraction`** — share of prompts where within-row selection can matter.
   On strong-policy / easy-task slices most prompts are all-correct (or all-wrong),
   so the ceiling on `top_per_prompt`'s advantage is **data-side, not RM-side**.
   Report it alongside the winrates to separate RM quality from data headroom.

Reproduce this analysis on your own cache before drawing conclusions from the
final table.

## Motivation (read before running; results belong to you after)

The answer key's published result on its 1k-train / 200-test GSM8K slice:
**`top_k_overall` beat its matched random baseline, while `top_per_prompt` tied
`random_per_prompt`.** Your job is to reproduce or refute this and explain it —
candidate mechanisms: overall top-K lets many pairs come from prompts whose
completions are high-reward *and correct*, concentrating signal; per-prompt argmax
is capped by the decidable fraction and inherits one possibly-wrong pick per
prompt; small-slice noise can flip either conclusion, which is exactly why the
matched-random control design exists.

Open the answer key (`rlhf-book/code/rejection_sampling/`: `selection.py`,
`config.py`, all four `configs/*.yaml`, `diagnostics.py`) **only after your table exists.**

## Files

| File | Role |
|---|---|
| `starter.py` | Signatures + docstrings + fixtures glue; selection/scoring/eval/table builders are stubs |
| `notebook.ipynb` | Prototype walkthrough: stage-by-stage skeleton (no solutions) |
| `configs/09_rejection_sampling_sft.yaml` | One config shared by all four arms (answer-key style); set `scorer_rm_checkpoint` to your Lab 03 path |
| `tests/test_09_rejection_sampling_sft.py` | CPU-only structural tests: container invariants, config parsing, stubs raise `NotImplementedError` |

```bash
python3 -m pytest tests/ -q     # must pass green before any implementation work
```

## Hardware notes

- Tier A (Colab T4): 0.6B model, reduce `num_completions_per_prompt` to 4–8,
  shorten `max_new_tokens`; expected wall-clock fine within free tier.
- Tier B (A40): 1.7B, four SFT runs of 2 epochs each ≈ an evening including evals.
- No unattended cluster launches without explicit go-ahead (§7 rule).
