# Lab 06 — REINFORCE / RLOO on an LLM (★★★ · CORE)

Minimal RL loop on a real LLM with verifiable rewards: your own rollout engine, your own
verifiers, REINFORCE-no-baseline then RLOO advantages, and a KL drift monitor.
This directory is the **scaffold** — every algorithmic body is yours to implement.

> **No-solutions rule.** `starter.py` contains signatures, docstring contracts, config
> plumbing and `raise NotImplementedError` stubs only. No baseline-subtraction arithmetic,
> no advantage math, and the GSM8K answer-extraction regex is deliberately absent.
> The answer key exists for post-hoc diffing only (see Compare below).

---

## Why this lab exists

Lab 05 gave you policy gradients on a toy MDP you could fully introspect. Here the same
objective meets real LLM rollouts: generation cost, sampling geometry (temperature,
top-p/k), sequence-level rewards from *verifiers* rather than environments, and group-based
variance reduction (RLOO). These are exactly the mechanisms the repo's
`policy_gradients/` module runs on 1.7B models — you build them first at inspectable scale,
then diff against upstream.

## Prerequisites

| Prerequisite | Where |
|---|---|
| **Lab 05 — Policy Gradients on a Toy MDP** | sibling lab dir `labs/05_*` (REINFORCE by hand, baseline variance study) |
| **Lecture 4 — RL Implementation & Practice** | [watch](https://www.youtube.com/watch?v=i-AIMpZHgeg) · [slides](https://rlhfbook.com/teach/course/lec4-chap6-p2/) |
| **Chapter 6 — Policy Gradients (implementation sections)** | https://rlhfbook.com/c/06-policy-gradients.html |

You should already be able to: implement REINFORCE from scratch, explain what a baseline does
and does not change, and trace log-prob → loss → gradient in Lab 05's char-level MDP.

## Assignment checklist

- [ ] **Rollout engine**: K completions per prompt (K = 4–8), old log-probs recorded;
      left-padding + pad masking handled; decoded completions stripped of the prompt.
- [ ] **Your own verifiers**: format gate (`verify_format`), outcome check
      (`verify_correctness`) for spell_backward string-match AND GSM8K numeric exact-match,
      answer extraction (`extract_gsm8k_answer`) — *write the regex yourself*, unit-test it
      against ≥20 messy real completions.
- [ ] **Advantages**: `compute_reinforce_advantages` (no baseline, high variance — keep it)
      then `compute_rloo_advantages` (leave-one-out within each K-group). Combine into the
      `-A·log π(completion)` loss and update.
- [ ] **On-policy single epoch first**; optional extension: reuse an epoch and watch the
      importance-ratio distribution drift away from 1.0 (`compute_importance_ratios`).
- [ ] **Monitoring**: avg correctness / format / total reward, response length, group contrast,
      approx-KL(k1) vs the initial policy snapshot (`approx_kl_k1`, `check_kl_drift`),
      appended as JSONL via `MetricsLogger`.

## Model · dataset · compute

| | Spec | Notes |
|---|---|---|
| Model dev | `Qwen/Qwen3-0.6B` | documented deviation: repo default is Qwen3-1.7B; upgrade to 1.7B on the A40 if time allows |
| Task A | `spell_backward` procedural pool, size 15000, word len 3–10 | fully inspectable reward |
| Task B | GSM8K train slice, 1000 examples | real-world exact-match verifier |
| Sampling | temp 0.6, top_p 0.95, top_k 20, max_new_tokens 512 | mirrors both answer-key configs |
| Compute | laptop smoke run CPU-possible with tiny samples; overnight OK on Colab/A40 | prototype here in `notebook.ipynb`, export long runs to scripts |

Config: [`configs/06_reinforce_rloo_llm.yaml`](configs/06_reinforce_rloo_llm.yaml) — copy it and flip
`loss_mode` + `num_rollouts` per arm (REINFORCE arm: `num_rollouts=1`; RLOO arms: K ∈ {4, 8}).

## Required experiments

1. REINFORCE-no-baseline vs RLOO, same prompts/seed/temp — learning curves overlaid.
2. Temperature {0.6, 1.0}.
3. K ∈ {4, 8} under RLOO.
4. Insight transfer: does what worked on spell_backward transfer to GSM8K? Say why/why not.

## Metrics convention

Every step appends one JSON line to `metrics_path` (`runs/metrics.jsonl`): step, ts, plus
avg_correctness, avg_format, avg_total_reward, group_contrast, mean_length, KL(k1)-to-init,
loss_mode, k, temperature. Notebooks plot straight from JSONL; wandb stays optional.

## Questions to answer

- Why is RLOO's leave-one-out baseline unbiased while using the full-group mean inside its own
  member is not?
- What does a zero-contrast group (all rewards tie) contribute mechanically? Predict before you
  observe — Lab 07 hits this again in GRPO.
- In REINFORCE-no-baseline, why can average reward rise while correctness flatlines?
- What exactly does approx-KL(k1) estimate, and why monitor against the *initial* snapshot?

## Debugging challenges (symptoms only — diagnosis is the assignment)

- Correctness stays flat while format score climbs steadily — name the failure mode and
  quantify how much true accuracy you're leaving on the table.
- KL(k1)-to-init explodes mid-run after looking stable — reconstruct the timeline from your
  own panels and predict the collapse that follows.
- Everything works on spell_backward, nothing moves on GSM8K — instrument extraction first.

## Completion criteria ("done when…")

- Correctness beats the base rate you measured BEFORE training, on held-out prompts.
- You can produce an annotated dump: prompt → K rollouts → verifier decisions → rewards →
  advantages → loss coefficient, explaining every number.
- KL drift panel exists and your threshold choice is justified in writing.

## Layout & how to test

```
06_reinforce_rloo_llm/
├── README.md                      ← this file
├── starter.py                     ← importable scaffold (torch-free)
├── notebook.ipynb                 ← prototype/dump/plot skeleton
├── configs/06_reinforce_rloo_llm.yaml
└── tests/test_06_reinforce_rloo_llm.py   # numpy-only fixtures; also enforce the no-solutions rule
```

```bash
python3 -m pytest tests/ -q        # must exit 0 on a bare laptop (no torch, no network)
```

## Compare (answer key — read AFTER yours works)

- `../../../rlhf-book/code/policy_gradients/rollout.py` — group generation loop, buffers, filters
- `../../../rlhf-book/code/policy_gradients/utils.py` — `compute_loo_advantages`,
  `compute_rewards`, `_format_reward`
- `../../../rlhf-book/code/policy_gradients/configs/reinforce.yaml` · `rloo.yaml`
  (note `num_rollouts`: 1 vs 8, and `prompts_per_step`: 8 vs 4)

Diff deliberately: what did they simplify that you over-engineered? What edge case did they
handle that you missed? Write it down — that list is interview gold.
