# Lab 05 — Policy Gradients on a Toy MDP (★☆→★★☆ · CORE · gap lab)

> **Difficulty:** ★★☆ · **CORE** · est. build 1 day · Scaffold status: **stubs only** (`starter.py` raises `NotImplementedError` for every mechanism) · Compute: **CPU** (deliberately — nothing here needs a GPU)

## Prerequisites

- **Lecture 3 (RL theory)** — watch before building: <https://rlhfbook.com/course/>
- **Chapter 6, §§1–3** (policy-gradient theorem, trajectory notation): <https://rlhfbook.com/c/06-policy-gradients.html>
- No lab prerequisites — this is the RL on-ramp the repo skips.
- Local setup: see `labs_curriculum/LABS_SETUP.md`.

## Why

The rlhf-book repo jumps straight to LLM rollouts on a 1.7B model — you never actually *see* a trajectory get scored before a tokenizer enters the room. This gap lab builds the missing on-ramp: a <1M-parameter char-level policy emitting arithmetic expressions (`a+b` evaluating to target N), where you can verify every reward by hand and plot every variance artifact. Everything here (trajectory → log-prob → advantage → surrogate loss → gradient variance) is the exact machinery Labs 06–08 reuse at LLM scale.

## Assignment checklist

- [ ] `ToyArithmeticEnv.reset/step` — episode terminates on `'='` or 12 chars; terminal-only reward = +1 iff the emitted string parses via the given oracle `expression_value` and equals target N
- [ ] `CharPolicyNet.init_params` + `logits_for_prefix` — forward pass conditioning on (target N, emitted prefix); assert total params < 1M; document your prefix encoding
- [ ] **Hand-written `softmax` / `log_softmax`** (shift-by-max stability; no scipy/torch)
- [ ] `sample_trajectory` — per-char log-probs recorded alongside emitted chars into the given `Trajectory` container
- [ ] **REINFORCE by hand**: derive ∇E[R] ≈ E[Σₜ ∇log π(aₜ|sₜ)·A(τ)] in the notebook FIRST, then implement `reinforce_loss`
- [ ] Moving-average baseline (`mode='moving_avg'`, EMA α=0.9) — maintain state across updates
- [ ] Leave-one-out baseline across K samples (`mode='loo'`: Aᵢ = rᵢ − mean(r₋ᵢ))
- [ ] Variance comparison plots: baseline none/moving_avg/loo × K∈{2,4,16} via `run_variance_experiment` + `gradient_variance_estimate`
- [ ] Entropy collapse observation: `observe_entropy_collapse` trace while training

## Components ↔ starter.py map

| Component | Function/class |
|---|---|
| env definition | `ToyArithmeticEnv` (reward/episode logic = student work) |
| verifier oracle (given plumbing) | `expression_value` |
| trajectory sampling w/ log-prob tracking | `sample_trajectory`, `Trajectory` (container given) |
| policy net (<1M params) | `CharPolicyNet.init_params`, `.logits_for_prefix` |
| distributions by hand | `softmax`, `log_softmax` |
| REINFORCE objective | `reinforce_loss`, `trajectory_log_prob` |
| advantage family | `compute_advantages` (none / moving_avg / loo) |
| entropy & collapse diagnostics | `entropy_of_policy`, `observe_entropy_collapse` |
| experiment driver | `run_variance_experiment` |

Config mirror: `configs/05_policy_gradients_toy_mdp.yaml` ⇄ `ToyMDPConfig`.

## Model

Char-level MLP policy, hidden 128 default, vocab = `CHAR_ACTIONS` (10 digits + `+` + `=` + space). **<1M parameters hard cap** — assert it in `init_params`. CPU throughout.

## Dataset

None in the dataset sense: targets N are sampled uniformly in `[0, 99]` by the env (`sample_target`). The "data" is self-generated experience.

## Experiments

- **E1:** baseline on/off — moving-average vs leave-one-out vs raw rewards: gradient-variance and success-rate plots
- **E2:** group size K ∈ {2, 4, 16} for the LOO arm
- **E3 (discussion):** shaped vs sparse+1 reward — document as an explicit deviation if used

## Questions (answer in README/notebook write-up)

1. Why log-probability, not probability, in the policy-gradient theorem?
2. What does a baseline change — and what does it *not* change? (Hint: bias/variance of the estimator vs. the gradient's expectation.)
3. Why is gradient variance THE practical problem of policy gradients?
4. Your LOO estimate with identical group rewards: why is the *loss* still nonzero and what does that imply?

## Debugging challenges

1. **Collapse to repeated action** (e.g., always emits `'='` immediately). Name the failure mode, give two countermeasures (entropy bonus / temperature / re-init — pick two and test one).
2. **Nonzero gradient though all rewards in a group are identical** — trace through your `compute_advantages(mode='loo')` + `reinforce_loss` combination to explain it.

## Done when

Working REINFORCE run on CPU + variance-comparison plots **explained line by line** + entropy-collapse diagnostic demonstrated + the four questions answered.

## Answer key & references

After **your own version works**, compare against (READ-ONLY until then):

- `rlhf-book/code/policy_gradients/loss.py::ReinforceLoss` — diff conventions, not implementations
- Later context: `policy_gradients/utils.py` advantage dispatch mirrors your three modes at LLM scale

Course/book links: <https://rlhfbook.com/course/> · <https://rlhfbook.com/c/06-policy-gradients.html>
