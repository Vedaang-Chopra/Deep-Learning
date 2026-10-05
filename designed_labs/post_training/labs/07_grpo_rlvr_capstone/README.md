# Lab 07 — GRPO / RLVR (★★★ · CORE CAPSTONE)

Group-relative policy optimization on a real LLM with verifiable rewards: your own
group sampler, hand-implemented GRPO **and** Dr.GRPO advantages, a clipped update
with asymmetric eps, **k1/k2/k3 KL estimators implemented by hand and compared on
the same batch**, and a justified zero-contrast-group policy — all wired into a
diagnostic panel that can reconstruct a training collapse after the fact.
This directory is the **scaffold** — every algorithmic body is yours to implement.

> **No-solutions rule (contract §9.4, worked example).** `starter.py` contains
> signatures, contract docstrings (shapes like `advantages [N, K]`), config
> plumbing and `raise NotImplementedError` stubs ONLY. No advantage arithmetic
> (no group normalization, no Dr.GRPO variant math), no clipped-surrogate
> formula, no KL estimator formulas (k1/k2/k3 all stubbed), and no
> zero-contrast-group handling logic. The answer key exists for post-hoc
> diffing only (see Compare below).

---

## Why this lab exists

Lab 06 gave you REINFORCE/RLOO over real LLM rollouts. GRPO is the industry's
default RLVR workhorse (DeepSeek-R1 lineage), and it changes three things at once:
advantages become **group-relative and standardized**, the update becomes a
**clipped importance-weighted surrogate**, and regularization moves into cheap
**Monte-Carlo KL estimators** instead of a value network. Each change has a known
failure mode — advantage saturation on tied groups, off-policy ratio blow-ups,
KL-estimator bias — and this capstone makes you build the instruments that see
them coming. You finish by diffing your by-hand version against the repo's
`policy_gradients/` implementation on 1.7B models.

## Prerequisites

| Prerequisite | Where |
|---|---|
| **Lab 06 — REINFORCE / RLOO on an LLM** | sibling lab dir `labs/06_reinforce_rloo_llm/` (rollout engine + verifiers are reused, not rewritten) |
| **Lecture 3 — Policy Gradients I** (revisited) | [slides](https://rlhfbook.com/teach/course/lec3-chap6-p1/) |
| **Lecture 4 — RL Implementation & Practice** (revisited) | [watch](https://www.youtube.com/watch?v=i-AIMpZHgeg) · [slides](https://rlhfbook.com/teach/course/lec4-chap6-p2/) |
| **Lecture 10 — Regularization & KL** | [slides](https://rlhfbook.com/teach/course/lec10-chap15-regularization/) |
| **Chapter 6 — Policy Gradients, GRPO section** | https://rlhfbook.com/c/06-policy-gradients.html |
| **KL estimators (reading)** | Schulman, *approximating KL divergence* — http://joschu.net/blog/kl-approx.html |

You should already be able to: run a K-completion rollout group through your own
verifiers, implement RLOO leave-one-out advantages, and explain what a baseline
does and does not change.

## Assignment checklist

- [ ] **Group sampling (N=8) + verifiers**: compose your Lab 06 engine;
      `spell_backward` AND GSM8K arms; `GroupRecord` per prompt; packed [N, K]
      reward matrices.
- [ ] **Standardized (GRPO) AND non-standardized (Dr.GRPO) group advantages —
      implement both** (`compute_grpo_advantages`, `compute_drgrpo_advantages`),
      derive each from the objective, and measure the std-normalization bias
      the Questions section asks about.
- [ ] **Clipped surrogate with asymmetric eps support**
      (`compute_clipped_surrogate`, eps_lo ≠ eps_hi) + full loss assembly
      (`grpo_loss`) with both sequence-mean and token-level normalization.
- [ ] **KL estimators k1/k2/k3 implemented by hand** (`approx_kl_k1/k2/k3`) and
      **plotted on the same batch** (`compare_kl_estimators`) with a written
      bias/variance verdict per estimator.
- [ ] **Zero-contrast group handling** (`handle_zero_contrast_groups`): all
      three modes (skip / weight / error) implemented, production choice
      **justified** in the README.
- [ ] **Diagnostic panel** via `MetricsLogger.summarize_batch`: correctness /
      format / group-contrast / KL / entropy / length / grad-norm — every
      quantity lands in the JSONL.

## Model · dataset · compute

| | Spec | Notes |
|---|---|---|
| Model dev | `Qwen/Qwen3-0.6B` | documented deviation: repo default is Qwen3-1.7B; upgrade on the A40 |
| Task A | `spell_backward` procedural pool, size 3000 | fully inspectable reward (grpo.yaml parity) |
| Task B | GSM8K train slice, 1000 examples | transfer of Lab 06 insights |
| Sampling | temp 0.6, top_p 0.95, top_k 20, max_new_tokens 512 | mirrors grpo.yaml |
| Compute | laptop smoke run CPU-possible with tiny samples; multi-day with sweeps on A40 | prototype here in `notebook.ipynb`, export long runs to `train.py` |

## Experiments

- N ∈ {4, 8, 16} (group size — `num_rollouts`)
- β ∈ {0, 0.001, 0.04} (KL coefficient — `beta`)
- eps sweep incl. asymmetric hi > lo (DAPO's clip-higher, post-hoc)
- token-level vs sequence-mean normalization (`loss_normalization`)
- GRPO vs Dr.GRPO arms (`loss_mode`) — same runs, different advantage estimator

## Questions to answer in your write-up

1. **What breaks when all rewards in a group tie — mechanically AND
   statistically?** Trace your estimator's output on a tied group by hand first;
   then explain what `zero_contrast_groups: skip` vs `weight` does to the
   estimator's bias.
2. **Std-normalization bias (GRPO vs Dr.GRPO)?** Which component of the
   standardized estimator introduces a reward-scale-dependent gradient, and what
   does removing it cost?
3. **Why GRPO over PPO for LLMs?** Quantify the value-net cost argument with
   your 0.6B run's memory/time numbers.
4. **Token-level normalization and length behavior?** Which aggregation rewards
   verbosity, and what did your length panel show?

## Debugging scenarios (diagnose from the JSONL panel)

- **"all advantages ≈ 0"** — investigate saturation: group contrast and
  frac_zero_contrast are your first two suspects. Reconstruct from the panel.
- **reward ↑, KL ↑, sudden gibberish collapse** — rebuild the timeline from
  panels; identify the earliest warning metric and say WHY it fired first.

## Done when

Learning curve above base rate + annotated group dump (prompt → K rollouts →
rewards → advantages) + KL-estimator comparison plot (k1/k2/k3, same batch).

## Compare (answer key — post-hoc only)

| Artifact | What to diff |
|---|---|
| `policy_gradients/loss.py::GRPOLoss` | your `grpo_loss`: ratio, clipping, KL wiring, masking |
| `policy_gradients/utils.py::compute_standardized_advantages` | your `compute_grpo_advantages` (note the eps placement) |
| `policy_gradients/utils.py::compute_nonstandardized_advantages` | your `compute_drgrpo_advantages` |
| `policy_gradients/configs/grpo.yaml` | your `configs/07_grpo_rlvr_capstone.yaml` |
| **After the capstone:** `loss.py::GSPOLoss`, `loss.py::DAPOLoss` | ratio granularity (sequence-level vs token-level) and normalization choices vs yours |
