# REPOSITORY_POST_TRAINING_MAP.md

> Audit of the local clone at `.../Post_Training/rlhf-book` (upstream: `github.com/natolambert/rlhf-book`, default branch `main`).
> Purpose: ground truth for designing the hands-on post-training lab curriculum. No labs were created during this audit.

---

## 1. What is actually present in this clone

```
rlhf-book/
├── README.md, Makefile, pyproject.toml     # site build (pandoc/make); NOT python project root
├── AGENTS.md / CLAUDE.md                   # build conventions (uv run, diagram workflow)
├── book/
│   ├── chapters/                           # 17 chapters + 4 appendices (markdown source of rlhfbook.com)
│   ├── images/, templates/, scripts/, data/# site assets/build tooling (not learning content)
│   └── rl-cheatsheet/                      # RL notation cheatsheet page
├── code/                                   # ← THE implementation library (~8,800 LOC Python)
│   ├── instruction_tuning/                 # SFT (Ch. 4)
│   ├── reward_models/                      # Preference RM, ORM, PRM (Ch. 5)
│   ├── policy_gradients/                   # REINFORCE/RLOO/PPO/GRPO + 5 modern variants (Ch. 6)
│   ├── direct_alignment/                   # DPO + 8 variants (Ch. 8)
│   ├── rejection_sampling/                 # Best-of-N → SFT pipeline (Ch. 9)
│   ├── distillation/                       # SDPO on-policy self-distillation (Ch. 12)
│   └── tests/                              # smoke tests only (imports, KL stability)
├── teach/
│   ├── course/                             # 14 lecture decks (lec0–lec13) + 3 Q&A decks + 1 conversation
│   └── extras/                             # 01-shoggoths.md (history essay)
├── diagrams/                               # TikZ/matplotlib sources for book figures
└── LICENSE-CHAPTERS (CC-BY-NC-SA) / LICENSE-CODE (MIT/Apache-2.0 per module)
```

Environment facts (`code/pyproject.toml`): Python ≥3.12, torch ≥2.6, transformers ≥4.40, datasets, accelerate, wandb, rich, **reasoning-gym ≥0.1.24** (procedural RLVR tasks), tensordict. Flash-attn optional with automatic SDPA fallback. Everything runs via `uv run` from `code/`. Reference hardware is a DGX Spark / single 24 GB consumer GPU; memory notes: 0.6B ≈ 4–6 GB, 1.7B ≈ 10–16 GB full fine-tune.

---

## 2. Book chapters ↔ lectures ↔ code relationships

| Chapter | Course lecture | Code module | Status |
|---|---|---|---|
| Ch. 1 Introduction | Lec 1 (Overview) | — | conceptual |
| Ch. 2 Related works | Lec 1 | — | conceptual |
| Ch. 3 Training overview | Lec 1 (+ Lec 0 prereq deck: cross-entropy, LM head, softmax, log-probs, KL, MDP framing) | — | conceptual/foundational |
| Ch. 4 Instruction tuning | Lec 2 | `instruction_tuning/` | **implementation-oriented** |
| Ch. 5 Reward models | Lec 2 | `reward_models/` (BT preference RM, ORM, PRM) | **implementation-oriented** |
| Ch. 6 Policy gradients | Lec 3 (RL theory) + Lec 4 (RL implementation & practice) | `policy_gradients/` (10 algorithms) | **implementation-oriented**, largest module |
| Ch. 7 Reasoning | Lec 5 (Rise of reasoning models) | (uses policy_gradients envs) | mostly conceptual |
| Ch. 8 Direct alignment | Lec 6 (DPO) | `direct_alignment/` (9 losses) | **implementation-oriented** |
| Ch. 9 Rejection sampling | Lec 2 (second half) | `rejection_sampling/` | implementation-oriented |
| Ch. 10 Preferences / Ch. 11 Preference data | Lec 8 | (data handling inside RM/DPO modules) | conceptual + data-practical |
| Ch. 12 Synthetic data & distillation | Lec 7 | `distillation/` (SDPO) | implementation + conceptual |
| Ch. 13 Tools & agents | Lec 11 | — | **gap: no code** |
| Ch. 14 Over-optimization (+ App. B style) | Lec 9 | — | conceptual; experiments must be designed by us |
| Ch. 15 Regularization (KL etc.) | Lec 10 | `policy_gradients/utils.py` KL estimators k1/k2/k3 | partially implemented |
| Ch. 16 Evaluation | Lec 12 | `rejection_sampling/diagnostics.py` only | **mostly a gap** |
| Ch. 17 Product / character training | Lec 13 | — | conceptual |
| App. A definitions / App. C practical | referenced across lectures | — | reference |

Note: the web book mentions end-of-chapter exercises for Ch. 4/5/6/8/9/12, but in this clone they exist only as brief "learning exercise" pointers (e.g., `04:193`, `05:580`) — there are no structured student exercises in the repo. The curriculum we design fills that hole.

---

## 3. Implementation inventory (what each module actually does)

### 3.1 `code/instruction_tuning/` — SFT (≈560 LOC)
- **Model/dataset:** `allenai/OLMo-2-0425-1B` (base) + `HuggingFaceH4/no_robots` (~9.5k rows). Chat template borrowed from `OLMo-2-0425-1B-SFT` because the base tokenizer has none (`utils.load_model`).
- **Key mechanics:** manual PyTorch loop (no Trainer); renders conversations with the chat template and masks everything except the final assistant turn via `labels = IGNORE_INDEX(-100)` prefix (`utils.py:153-186`); pads right; `F.cross_entropy(..., ignore_index=-100)`; bf16 + gradient checkpointing; effective batch 32 (bs 4 × grad-accum 8), lr 5e-6, 3 epochs; in-loop sample panels every 50 steps showing the base-model rambling → answer-and-stop transition.
- **Pedagogically valuable:** exact tensor anatomy of what enters SFT training (prompt-masked labels, template tokens, EOS termination).
- Config: `configs/sft_olmo2_1b.yaml`.

### 3.2 `code/reward_models/` — three RMs (≈2,300 LOC)
- **Shared:** `base.BaseRewardModel` = causal LM backbone + scalar `nn.Linear(hidden, 1)` reward head read at the last token; full fine-tune (freeze option exists); FP32 weights + bf16 autocast.
- **Preference RM** (`train_preference_rm.py`): Bradley-Terry on `argilla/ultrafeedback-binarized-preferences-cleaned` (5k pairs, Qwen3-0.6B-Base). Loss `-F.logsigmoid(r_chosen - r_rejected)`; drops tied/degenerate pairs; metrics: val loss, pairwise accuracy `(margin > 0)`, mean margin.
- **ORM** (`train_orm.py`): binary correctness classification on `RLHF-Book/gsm8k-qwen3-0.6B-rollouts` (a released rollout dataset: 100 sampled solutions/prompt, labeled correct/incorrect); Qwen3-0.6B.
- **PRM** (`train_prm.py`): step-level {-1, 0, 1} classification on `tasksource/PRM800K`, chunked to ≤12 steps/problem; Qwen3-0.6B.
- Marked *experimental* in the README ("needs tuning of hyperparameters… contributions welcome"); no standalone eval scripts yet (open TODO).

### 3.3 `code/policy_gradients/` — the RL workhorse (≈1,420 LOC, adapted from zafstojano/policy-gradients, Apache-2.0)
- **Task:** Reasoning-Gym `spell_backward` (reverse each word; deterministic verifier) — pure RLVR, no external reward model.
- **Data flow:** `RolloutEngine` samples N completions per prompt (`num_rollouts=8`) → `Experience` dataclass (sequence_ids, attention_mask, action_mask, advantages, log_probs_old, log_probs_ref, values_old, TensorDict rewards with components total/correctness/format/penalty/binary) → `ReplayBuffer` → micro-batched updates.
- **Rewards (`utils.compute_rewards`)**: composite = correctness (verifier) + format (tag counting) − DAPO-style length penalty; MaxRL uses strict binary r = correctness ∧ format.
- **Advantage family (`compute_advantages` dispatch):**
  - GRPO/GSPO/CISPO/SAPO/DAPO → standardized group advantages (mean/std over the group);
  - Dr. GRPO → non-standardized group-centered;
  - RLOO → leave-one-out baseline;
  - PPO → GAE(γ, λ) with a separate value model;
  - REINFORCE/MaxRL → raw reward.
- **Losses (`loss.py`)**: REINFORCE, RLOO, PPO (clipped ratio + clipped value loss), GRPO (clipped token-level ratio + β·KL), GSPO (sequence-level ratio), CISPO (stop-gradient clipped ratio × log-prob), SAPO (soft sigmoid gate), DAPO (token-level global normalization, clip-higher, no KL), MaxRL. KL estimators k1/k2/k3 (joschu.net approximations); masked_mean utilities.
- **Default config:** Qwen/Qwen3-1.7B, lr 5e-6, temp 0.6, 512 new tokens, prompts_per_step 4, ~16 GB VRAM. Metrics logged: avg_correctness, avg_format, avg_binary, group contrast.

### 3.4 `code/direct_alignment/` — DPO & friends (≈2,300 LOC)
- **Setup:** `allenai/OLMo-2-0425-1B-SFT` (an already-SFT'd model — note: DPO starts from SFT checkpoint, not base) on UltraFeedback-binarized-cleaned, 6.4k pairs, effective batch 64, 3 epochs, lr 5e-6, β=0.1.
- **`data.py`:** builds paired batches (chosen_input_ids/rejected_input_ids + response_mask separating prompt from response tokens); chat-template formatting helpers.
- **`loss.py`:** clean, textbook-quality loss classes — DPOLoss (incl. label-smoothing = cDPO), IPOLoss, SimPOLoss (length-normalized, reference-free), ORPOLoss (SFT NLL + odds-ratio), KTOLoss (treatment/control, independent pairs), APOZero/APODown; `get_loss_function` dispatcher; metrics: chosen/rejected implicit rewards, margins, accuracy.
- Known-noisy references: SimPO/ORPO runs flagged as untuned (`ORPO_SIMPO.md`, issue #358). DPO/IPO/KTO/APO validated.

### 3.5 `code/rejection_sampling/` — Best-of-N → SFT (≈1,270 LOC)
- Three stages with a shared rollout cache: (1) generate N=8 completions/prompt on GSM8K (1k train / 200 test) with Qwen3-1.7B; (2) score with an external RM (`nvidia/AceMath-7B-RM`); (3) select (top_per_prompt / top_k_overall / matched random controls) and SFT.
- `diagnostics.py` (matplotlib/pandas extra) plots reward distributions and accuracy vs. random baselines. Reference result: `top_k_overall` beat its random control; `top_per_prompt` was tied — a genuinely interesting negative result to study.

### 3.6 `code/distillation/` — SDPO on-policy distillation (≈550 LOC)
- Self-distillation: model samples a group of rollouts on `spell_backward`; a demonstration-conditioned copy of the same model (given a correct sibling rollout) becomes teacher; student distilled via **top-K reverse-KL** (tail-bucket trick in `loss.add_tail`); groups with no correct sample are skipped/refilled (`skipped` metric). Qwen3-1.7B, ~20 h on a 24 GB GPU for the reference run.

---

## 4. Classification against the five categories

1. **Conceptual/theoretical:** chapters 1–3, 7, 10, 11, 13, 14, 15, 16, 17, appendices; lecture decks 0, 1, 5, 8–13; `book/rl-cheatsheet`.
2. **Implementation-oriented:** `instruction_tuning`, `reward_models`, `policy_gradients` (losses/buffer/rollout), `direct_alignment` (losses/data), `rejection_sampling` (selection), `distillation` (loss).
3. **Experimental (runnable reference experiments with published W&B curves at wandb.ai/rlhf-book/core):** all six modules; RM module explicitly flagged experimental/untuned; SimPO/ORPO flagged noisy.
4. **Useful as-is for student exercises:** chapter-end exercise pointers (thin), the Reader Experiment Path table in `code/README.md` (7 suggested starting experiments + one-variable sweeps), configs as hyperparameter ground truth.
5. **Too advanced/expensive for immediate goals:** full PPO with value model on 1.7B (~16 GB + value net), SDPO reference run (~20 h GPU), AceMath-7B-RM scoring stage, the 8 non-core policy-gradient variants (GSPO/CISPO/SAPO/DAPO/MaxRL/Dr.GRPO beyond reading), PRM800K at scale. All have smaller equivalents suitable for learning.

## 5. Gaps in the repository (things the curriculum must design itself)

1. **No policy-gradient-on-toy-MDP lab** — the repo jumps straight to LLM rollouts; there is no computationally trivial place to first meet log-probs/trajectory/reward/advantage. We must build one (cheap: bandit / tiny char-level model / fixed network).
2. **No evaluation lab** — Ch. 16/Lec 12 has no code; only `rejection_sampling/diagnostics.py` touches eval methodology. An eval/LLM-as-judge exercise must be designed.
3. **No reward-hacking / over-optimization experiment** — Ch. 14/Lec 9 is purely conceptual. A hackable proxy-reward environment (format reward gamed while correctness stalls) must be constructed.
4. **No preference-data quality exercise** — Ch. 10/11 discuss length bias, sycophancy, noise; no notebook/script audits a preference dataset.
5. **No tool-use post-training code** — Ch. 13 has nothing executable.
6. **RM evaluation scripts** are an explicit upstream TODO; our labs should include RM eval (accuracy/margins/calibration) as a first-class deliverable.
7. **Apple-silicon/MPS reality:** all reference runs assume NVIDIA CUDA; a learner on a Mac needs downsized models (Qwen3-0.6B, SmolLM2-class) and reduced rollouts — deviations documented per lab.
