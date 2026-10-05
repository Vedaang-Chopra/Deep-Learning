# Lab 08 — Reward Hacking & KL Regularization

**Difficulty:** ★★★ · optional · Compute tier: **B (reuses Lab 07 runs)**
**Scaffold version:** starter only — all implementation is yours.

> This is the recommended **first optional lab after the core chain** (plan §10):
> it is the cheapest optional lab because it reuses Lab 07's GRPO/RLVR runs and
> infrastructure directly, and it has high interview value since "reward hacking"
> and Goodhart scaling curves are frequent discussion topics.

---

## 0. Prerequisites

- **Lab 07 — GRPO/RLVR (capstone) completed**, with its metric JSONL artifacts
  (correctness/format/KL panels) retained under
  `/coc/scratch/vchopra/post_training_labs/07_grpo_rlvr/<run>/metrics*.jsonl`.
  Lab 08 does not retrain from a cold start conceptually — every sweep here is a
  rerun or post-hoc analysis of a Lab 07-shaped run.
- **Lecture 9 (Over-optimization)** — watched at
  <https://rlhfbook.com/course/lec9-chap14-appb-overoptimization.html>
- **Book Chapter 14 — Over-Optimization**
  <https://rlhfbook.com/c/14-over-optimization>
- **Book Chapter 15 — Regularization** <https://rlhfbook.com/c/15-regularization>
- Working knowledge of the KL estimators k1/k2/k3 you implemented by hand in
  Lab 07 (`policy_gradients/loss.py::GRPOLoss` compare-against).

## 1. Why this lab

Reward models and verifiable rewards are *proxies* for what we actually want.
Optimize any proxy hard enough and the true objective degrades while the proxy
keeps climbing — Goodhart's law, with training curves as evidence. The most
inspectable failure mode in an RLVR loop is partial reward hacking: **format
compliance climbs faster than correctness**, so a carelessly weighted composite
reward can literally pay the model to abandon real accuracy. This lab makes you
*cause* that failure on purpose, watch proxy reward rise while true accuracy
falls (the proxy-up / true-down divergence plot), and then characterize which
monitored signal warned you first and how the KL coefficient shapes the fall.

## 2. Model

Same register entries as Lab 07 (no new model):

| Role | Choice | Notes |
|---|---|---|
| Policy | `Qwen/Qwen3-0.6B` (dev) → `Qwen/Qwen3-1.7B` (Tier B A40) | identical to your Lab 07 configuration |
| Tasks | Reasoning-Gym `spell_backward` + `openai/gsm8k` exact-match | deterministic verifiers = trustworthy "true" metric |

**Deviation policy note:** if you sweep only at 0.6B, record it here per the
curriculum's deviation-documentation rule.

## 3. Dataset / data artifacts

No new datasets. You consume:

1. Your existing Lab 07 rollout/metrics artifacts (JSONL convention from §7.5
   of the plan), and
2. Fresh reruns of the same `spell_backward` + GSM8K subset produced by this
   lab's sweep runner (`starter.py::run_kl_sweep`) using a **modified proxy
   reward config** (`configs/08_reward_hacking_kl.yaml`).

All checkpoints/logs stay under `/coc/scratch/vchopra/post_training_labs/08_reward_hacking_kl/`.

## 4. Assignment checklist

- [ ] **Proxy reward misaligned with the true goal** — implement
      `build_proxy_reward`: composite reward = small weight × verifier
      correctness + **large format bonus** weighted by `proxy_reward_weight`
      (config key). The point of the lab is the deliberate misalignment; sanity
      check that a format-perfect/wrong-answer rollout scores near the top.
- [ ] **Rerun the Lab 07 config** under the hacked proxy via `run_kl_sweep`
      over the `kl_coef_grid` — do not fork the trainer; reuse your Lab 07
      engine/config paths and only swap the reward construction.
- [ ] **Track true accuracy separately from proxy reward** — both series must
      appear in every metrics JSONL row: `proxy_reward`, `true_accuracy`,
      `format_rate`, `approx_kl`, plus whatever else you monitor.
- [ ] **Produce the proxy-up / true-down divergence plot** — implement
      `prepare_divergence_plot_data`: given ≥2 runs' step-series, emit the dict
      the notebook plots (x = steps or compute units, y = normalized proxy vs
      true series). The classic figure shows proxy still rising after true
      accuracy has peaked and begun falling.
- [ ] **Sweep the KL coefficient** — for each value in `kl_coef_grid`, record
      where divergence starts and how steeply truth decays; relate β to how
      much drift from the reference policy the policy is allowed to spend.
- [ ] **Identify the earliest warning metric** — implement
      `select_earliest_warning_metric`: across candidate monitored series,
      return the one whose excursion precedes the `true_accuracy` drop by the
      largest margin. Candidates at minimum: `format_rate`, `response_length`,
      `approx_kl`, group reward variance.

## 5. Components (what lives where)

| File | Role |
|---|---|
| `configs/08_reward_hacking_kl.yaml` | single shared config: hacked-proxy weights, `kl_coef_grid`, paths to reused Lab 07 assets |
| `starter.py` | four stubs to implement: `build_proxy_reward`, `prepare_divergence_plot_data`, `run_kl_sweep`, `select_earliest_warning_metric` |
| `notebook.ipynb` | prototype + plotting: load JSONL → divergence plot → warning-metric timeline |
| `tests/test_08_reward_hacking_kl.py` | structural invariants only (shapes/keys/raise-on-call); no GPU, no network |

There is no separate `train.py` export here by design: the long-running job is
your Lab 07 trainer invoked with this lab's config; the cluster workflow is
unchanged (tmux wrap, pinned `CUDA_VISIBLE_DEVICES`, occupancy check before
launch, no unattended starts).

## 6. Experiments

1. Baseline arm: untouched Lab 07 verifier-only reward (proxy == truth).
2. Hacked arm(s): `proxy_reward_weight` large enough that format bonus dominates
   correctness within ~20 steps.
3. For each KL coefficient in `kl_coef_grid` (e.g. {0, 0.001, 0.01, 0.04, 0.2}):
   divergence onset step, peak-true-vs-final-true gap, max approx-KL reached.
4. Warning-metric lead-time table: for each arm, how many steps before the
   truth drop did each candidate metric first cross its alert threshold?

## 7. Questions to answer in the notebook

- Why is `spell_backward`'s deterministic verifier the right environment for
  studying hacking — what would a noisy reward model confound?
- In Ch. 14's terminology (Goldman levels / Goodhart taxonomy), which variant
  are you inducing with a big format bonus — regressional, extremal, or causal?
- KL regularization constrains movement away from the *reference policy*, not
  toward truth. Explain why that still blunts hacking in practice — and when it
  fails anyway (reference itself already hacks).
- Which metric warned earliest, and why is length often the fastest tell?
- If format bonus were replaced by a learned-RM score, what changes about your
  monitoring strategy?

## 8. Debugging challenges

- Proxy plateaus but truth stays flat too — your format bonus probably doesn't
  dominate; recheck the arithmetic of `build_proxy_reward`.
- Divergence plot shows truth rising with proxy — congratulations, you failed
  to hack; the misalignment isn't strong enough yet (this is the expected first
  iteration).
- KL explodes before any behavioral change — inspect whether the estimator you
  plotted is k1 vs k3; see Lab 07's estimator comparison panel.

## 9. Completion criteria

Done when you can show:

1. A divergence plot with ≥2 arms where the hacked arm displays proxy-up /
   true-down separation (or a documented explanation of why your arm didn't hack).
2. A KL-sweep table tying β to divergence onset.
3. A written earliest-warning-metric verdict with the lead-time table behind it,
   and every claim traceable to a metrics JSONL artifact.

## 10. Compare against (after yours works)

- Your own Lab 07 trainer/engine — no repo code is newly compared here, but
  revisit `policy_gradients/loss.py::GRPOLoss` KL-estimator section and
  `configs/grpo.yaml` once your sweeps produce curves.
- Book chapters cited above discuss expected scaling behavior; check your
  empirical divergence shape against Ch. 14's figures.

---

*Scaffold generated from `POST_TRAINING_LAB_CURRICULUM_PLAN.md` §6 (Lab 08 spec)
and §9 contract. Starter contains signatures/docstrings/TODOs only.*
