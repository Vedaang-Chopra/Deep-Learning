# Lab 03b — ORM vs PRM (optional comparison lab)

> ★★☆→★★★ · **OPTIONAL** · est. build 1–2 days · Scaffold status: **stubs only** (`starter.py` raises `NotImplementedError` everywhere)

**Goal:** train the two flavors of reward model on *the same kind of artifact* — model-generated solutions — and put them head-to-head: a **binary correctness head (ORM)** over GSM8K rollouts labeled only by final-answer match, versus a **step-level `{-1, 0, 1}` classifier (PRM)** over PRM800K-style step annotations. Then deliver the payoff: a **disagreement case-study table on ≥20 solutions**, isolating the canonical failure mode — *correct answer, wrong reasoning* — where the ORM's binary signal is blind and the PRM is not.

---

## Prerequisites

- **Lab 03 — Bradley-Terry reward model** (required): you reuse its paired-batching instincts and last-non-pad pooling discipline; this lab swaps the loss targets, not the backbone plumbing.
- **Lab 06 — REINFORCE/RLOO** (optional but recommended): it makes concrete *what consumes* these rewards during RL, which sharpens why reward granularity matters.
- **Lecture 2, part 2 (Reward models)** — watch before building: <https://rlhfbook.com/course/>
- **Chapter 5, ORM + PRM sections** — read for the two supervision signals you'll implement: <https://rlhfbook.com/c/05-reward-models.html>
- Local setup: see `labs_curriculum/LABS_SETUP.md` (Colab Tier-A profile, GT-cluster profile, JSONL metric convention).

## Why

ORMs are cheap to label (run the solution, check the answer) but throw away everything between "start" and "final answer"; PRMs need human-quality step annotations but can localize *where* reasoning went wrong. This lab forces you to feel that trade-off empirically instead of repeating it from the blog posts: by scoring the **same held-out solutions with both models** and tabulating disagreements, you will find cases the ORM scores confidently-correct while the PRM flags a poisoned step — and quantify how often those cases still reach the right answer.

## Assignment checklist

- [ ] Load & normalize GSM8K rollouts (`load_rollout_dataset`, `normalize_rollout_records`) into the declared record schema; verify every label ∈ {`correct`, `incorrect`}
- [ ] Shape ORM examples (`shape_orm_examples`): prompt+solution concatenated, right-padding collate (`orm_collate_fn`)
- [ ] **ORM binary head wrapper** (`OrmBinaryHead`): backbone + `Linear(hidden_size, 1)` pooled at last non-pad token → logit; same pooling discipline as Lab 03
- [ ] ORM training loop (`run_orm_training_loop`) with hand-written `orm_correctness_loss` — numerically stable, derivation written out first
- [ ] Load the PRM800K slice (`load_prm800k_slice`) into step records `{prompt, steps[], step_labels[]}` with labels ⊆ {−1, 0, +1}
- [ ] **Step-label collation** (`collate_prm_steps`): per-token label expansion aligned to step boundaries, `pad`/`ignore_index` handled so ambiguous steps (0) don't become negatives
- [ ] PRM training loop (`run_prm_training_loop`) with `prm_step_loss`
- [ ] Score one shared eval pool of solutions under BOTH models (`score_solutions_both_models`)
- [ ] Select disagreement cases (`select_disagreement_cases`) honoring the ≥20-solution floor
- [ ] **Deliverable:** `build_disagreement_table` → case-study table exported as JSONL/CSV with categories (correct-answer-wrong-reasoning, both-agree-wrong, etc.), then write up ≥5 hand-read rows in the notebook

## Model

`Qwen/Qwen3-0.6B` — same backbone for both heads so score comparisons are apples-to-apples. Full fine-tune default; frozen-backbone arm E1 optional.

## Dataset

Two sources, deliberately different supervision:

| Source | Role | Label space |
|---|---|---|
| `RLHF-Book/gsm8k-qwen3-0.6B-rollouts` | ORM training/eval: 100 rollouts per GSM8K prompt, auto-labeled by answer match | {correct, incorrect} |
| `tasksource/PRM800K` slice | PRM training/eval: human step-level annotations | {−1, 0, +1} per step |

For the case study, hold out a **shared pool of ≥20 complete solutions** scored by both heads. Colab-feasible (Tier A) at reduced rollouts-per-prompt and a small PRM800K slice; full scale targets Tier B (A40).

## Components ↔ starter.py map

| Component | Function/class |
|---|---|
| rollout dataset loading/shaping | `load_rollout_dataset`, `normalize_rollout_records`, `shape_orm_examples`, `orm_collate_fn` |
| ORM binary head wrapper | `OrmBinaryHead` |
| ORM loss + loop | `orm_correctness_loss`, `run_orm_training_loop` |
| PRM800K slice loading | `load_prm800k_slice` |
| step-label collation | `collate_prm_steps` |
| PRM loss + loop | `prm_step_loss`, `run_prm_training_loop` |
| dual scoring + disagreement study | `score_solutions_both_models`, `select_disagreement_cases`, `build_disagreement_table` |

Config mirror: `configs/03b_orm_vs_prm.yaml` ⇄ `OrmPrmLabConfig`.

## Compute

~1–2 h on an A40 (Tier B) for both heads at default scale; Colab T4/Tier A feasible with `rollouts_per_prompt_cap ≤ 8` and a ≤1k-row PRM slice. Cluster paths: `/coc/scratch/vchopra/post_training_labs/lab03b_orm_vs_prm/`. All long runs follow LABS_SETUP ops rules (tmux, pinned GPU, occupancy check, **no unattended starts**).

## Deliverable — disagreement case-study table

The graded artifact is a table with one row per held-out solution:

```
solution_id | orm_logit | orm_verdict | prm_min_step_label | prm_flagged_step_idx | category | ground_truth_answer | predicted_answer
```

Categories: `agree_correct`, `agree_incorrect`, `orm_right_prm_flags_reasoning` (**the money row — correct answer, wrong reasoning**), `prm_right_orm_wrong`. Require ≥20 rows, ≥3 of the money category if your models trained to sane accuracy; report counts by category in the notebook and hand-read at least 5 flagged solutions, quoting the exact poisoned step.

## Experiments

- **E1:** freeze vs full fine-tune the backbone (both heads) — does freezing hurt PRM more than ORM?
- **E2:** class balance sweep for ORM rollouts (1:1 vs natural skew) — effect on false-positive correctness.
- **E3:** PRM ambiguity handling: drop 0-labels vs train them as a third class.
- **E4:** downsample PRM slice ×4 — how far does step supervision survive scarcity?

## Answer key & references

After **your own version trains**, compare against (READ-ONLY, do not open earlier):

- `rlhf-book/code/reward_models/train_orm.py` — ORM training
- `rlhf-book/code/reward_models/train_prm.py` — PRM training

Course/book links: <https://rlhfbook.com/course/> · <https://rlhfbook.com/c/05-reward-models.html>
