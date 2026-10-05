# Lab 03 — Bradley-Terry Reward Model (from scratch)

> ★★☆ · **CORE** · est. build 1–2 days · Scaffold status: **stubs only** (`starter.py` raises `NotImplementedError` everywhere)

**Goal:** build and train a preference reward model completely from scratch — backbone + `Linear(hidden, 1)` head pooled at the last non-pad token, paired batching, a **hand-written Bradley-Terry loss**, pairwise accuracy / margin metrics, and margin-bucketed calibration analysis — then use it to re-score your own Lab 02 near-tie judgments. No TRL.

---

## Prerequisites

- **Lab 02 — Preference-data forensics**: you need your *cleaned* UltraFeedback subset (2–5k pairs) and the near-ties you judged by hand.
- **Lecture 2, part 2 (Reward models)** — watch before building: <https://rlhfbook.com/course/> (deck `teach/course/lec2-chap4-5-9.md`).
- **Chapter 5, Bradley-Terry section** — read for the derivation you'll implement: <https://rlhfbook.com/c/05-reward-models.html>
- Local setup: see `labs_curriculum/LABS_SETUP.md` (Colab Tier-A profile, GT-cluster profile, JSONL metric convention).

## Why

RMs sit upstream of everything else in post-training: RL labs optimize against their scores and rejection sampling trusts their ranking. This lab forces you to own the mechanism — where gradients come from in a paired objective, why pooling position can silently destroy learning, and why *scale-free* accuracy coexists with *meaningful* margins.

## Assignment checklist

- [ ] Schema-map your Lab 02 cleaned subset → `tokenize_pair`; paired right-padding collate (`paired_collate_fn`)
- [ ] Paired DataLoader, seeded shuffling (`build_dataloader`)
- [ ] Backbone + `Linear(hidden_size, 1)` head (`BradleyTerryRewardModel.__init__`), incl. `freeze_backbone` support and trainable-param report
- [ ] **Pooling at last non-pad token** (`pool_last_non_pad_token`) + decode-based proof you pooled real tokens
- [ ] **Hand-written BT loss** (`bradley_terry_loss`) — numerically stable, derivation written out first, gradient sanity-checked; no library log-sigmoid shortcuts
- [ ] Metrics: `pairwise_accuracy` (documented tie convention), `mean_margin`, reward histogram stats, `accuracy_by_margin_bucket` calibration buckets
- [ ] `compute_eval_metrics` / `evaluate` aggregated over val loader, appended as JSONL every `eval_every_steps`
- [ ] `run_training`: AdamW (trainable params only), grad accum, clip-grad-norm, warmup, AMP flag, checkpoints; smoke-run ≈100 steps @ len 512 before anything long
- [ ] Trace **one pair end-to-end naming every tensor shape** (the acceptance artifact)
- [ ] Score Lab 02 near-ties (`score_lab02_near_ties`) and build the human-vs-RM verdict table

## Model

`Qwen/Qwen3-0.6B-Base` — full fine-tune default; freezing the backbone is experiment arm E1 (register rationale: current-gen, 0.6B trains on free Colab).

## Dataset

Your **Lab 02 cleaned UltraFeedback-binarized-preferences-cleaned subset** (2–5k rows), plus its near-tie export. Size-vs-quality arm uses {1k, 2k, 5k} slices.

## Components ↔ starter.py map

| Component | Function/class |
|---|---|
| pair tokenization | `tokenize_pair` |
| paired batching | `paired_collate_fn`, `build_dataloader` |
| model + head | `BradleyTerryRewardModel` |
| last-non-pad pooling | `pool_last_non_pad_token` |
| BT loss | `bradley_terry_loss` |
| metrics + calibration | `pairwise_accuracy`, `mean_margin`, `accuracy_by_margin_bucket`, `compute_eval_metrics`, `evaluate` |
| orchestration | `run_training`, `score_lab02_near_ties` |

Config mirror: `configs/03_bradley_terry_rm.yaml` ⇄ `RewardModelConfig`.

## Compute

~1–2 h on an A40 (Tier B); feasible on Colab T4 (Tier A) at 0.6B. Cluster paths: `/coc/scratch/vchopra/post_training_labs/lab03_bt_rm/`. All long runs follow LABS_SETUP ops rules (tmux, pinned GPU, occupancy check, **no unattended starts**).

## Experiments

| # | Question | Setting |
|---|---|---|
| E1 | Does the backbone need gradients at all? | `freeze_backbone` True vs False |
| E2 | LR sensitivity near instability | {1e-6, 5e-6, 2e-5} |
| E3 | Data size vs accuracy | {1k, 2k, 5k} rows |

Metrics that must move: pairwise accuracy above chance on val; margins growing early then stabilizing; calibration buckets reporting non-degenerate counts.

## Questions

1. Why sigmoid(diff)? (derive from BT likelihood, once, in symbols)
2. Why is absolute reward scale meaningless? Demonstrate the invariance with a tensor.
3. Accuracy↑ while margins collapse — what does that predict for downstream RL/best-of-N?
4. Where does gradient signal come from when neither completion is "correct", merely preferred?

## Debugging challenges

- Pairwise accuracy stuck at 50% → check **pooling index & pad side** first.
- Margins explode while accuracy stays flat → what is BT actually optimizing then?
- Untrained loss starts at ln(2) — verify you can explain why before training.

## Completion criteria ("done when")

One narrated trace of a single pair through template→tokens→hidden states→pooling→head→loss→backward, every intermediate shape named, plus (a) calibration figure, (b) margin trajectories across E1–E3, (c) the near-tie verdict table.

## Compare against (answer key — open ONLY after your version trains)

- `rlhf-book/code/reward_models/base.py` — `BaseRewardModel` (loading/freeze idiom, `_build_head`, `get_hidden_states`), `pad_sequences`, `create_collate_fn`
- `rlhf-book/code/reward_models/train_preference_rm.py` — `PreferenceRewardModel.get_reward` / `.forward` (its BT loss, one-liner around a library log-sigmoid helper at line 228 — compare against YOUR hand-written version), `evaluate_preference_rm`, `train_preference_rm`

Deviation policy note (per plan §4): none currently — model/dataset match the register defaults. Record any future deviations here.
