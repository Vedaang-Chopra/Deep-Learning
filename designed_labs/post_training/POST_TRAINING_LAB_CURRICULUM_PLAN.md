# POST_TRAINING_LAB_CURRICULUM_PLAN.md — v2

> Companion to `REPOSITORY_POST_TRAINING_MAP.md`. Phase 2 of the assignment.
> **v2 changes:** watch-then-build syllabus added; hardware replanned for Colab + GT CoC cluster (A40 46GB); model/dataset selections refreshed to current standards with rationale; SDPO distillation lab added; LoRA-from-scratch, sequence packing, and decontamination exercises added; infrastructure/setup section added; parallel-subagent scaffolding contract added.
>
> **No labs are scaffolded yet.** Execution happens in a separate session with parallel subagents under the contract in §9.

---

## 0. How you learn (drives everything below)

You watch Nathan Lambert's lecture first, then implement the concept yourself in a lab. Therefore every lab is bound to specific lectures, and §1 gives the strict viewing→building order. Labs are notebook-friendly (prototype on Colab) with long runs exported to scripts and launched on the GT cluster in tmux — matching the established workflow: *prototype in notebook → export to train.py → launch in tmux*.

---

## 1. Watch-then-implement syllabus (the master schedule)

| Step | Watch (rlhfbook.com/course) | Then build | Est. build time |
|---|---|---|---|
| 0 | Lecture 0 (prereq review: cross-entropy, log-probs, KL, MDP) | Lab 00 — Data & chat templates | ½ day |
| 1 | Lecture 1 (Overview) + skim Ch. 1–3 | *(no lab — orient)* | — |
| 2 | Lecture 2 part 1 (IFT) + Ch. 4 | Lab 01 — SFT from scratch | 2 days |
| 3 | Lecture 8 (Preferences & preference data) + Ch. 10–11 | Lab 02 — Preference-data forensics | ½ day |
| 4 | Lecture 2 part 2 (Reward models) + Ch. 5 BT section | Lab 03 — Bradley-Terry reward model | 1–2 days |
| 5 | Lecture 6 (DPO) + Ch. 8 | Lab 04 — DPO from scratch | 2–3 days |
| 6 | Lecture 3 (RL theory) + Ch. 6 §§1–3 | Lab 05 — Policy gradients on toy MDP | 1 day |
| 7 | Lecture 4 (RL implementation & practice) + Ch. 6 impl. sections | Lab 06 — REINFORCE/RLOO on an LLM | 2 days |
| 8 | Lecture 5 (Reasoning models) + Ch. 6 GRPO section + Lecture 10 (regularization/KL) | Lab 07 — GRPO/RLVR (capstone) | 3 days |
| 9 | Lecture 9 (Over-optimization) + Ch. 14–15 | Lab 08 — Reward hacking & KL | 1 day |
| 10 | Lecture 2 RS section + Ch. 9 | Lab 09 — Rejection sampling → SFT | 1–2 days |
| 11 | Lecture 7 (Synthetic data) + Ch. 12 | Lab 10 — SDPO on-policy distillation | 2 days |
| 12 | Lecture 12 (Evaluations) + Ch. 16 | Lab 11 — Evaluation & LLM-as-judge | 1 day |
| 13 | Lecture 13 (Character training) + Ch. 13, 17 | *(stretch: persona-SFT extension; tool-use remains documented gap)* | optional |

Q&A decks 1–3 and the conversation deck are watch-anytime supplements.

---

## 2. Design principles

1. **You implement the mechanism; the repo is the answer key.** Every lab has a "compare against" pointer into `rlhf-book/code/` opened only after your version trains.
2. **Real data, current methods.** Primary datasets are ones actual 2024–2026 post-training pipelines use (No Robots, UltraFeedback, GSM8K, SmolTalk, Tülu 3 mixtures). The only procedural environment (`spell_backward`) is kept deliberately — deterministic rewards make RL mechanics inspectable — and every RL lab pairs it with a real math task (GSM8K exact-match).
3. **Small-but-real models.** Qwen3 (0.6B/1.7B/4B) and OLMo-2 families — current, openly licensed, and sized to your hardware. Nothing smaller than 0.5B except Lab 05's deliberate toy.
4. **From-scratch checkpoints before frameworks** — including a LoRA layer implemented by hand (Lab 01) and all core losses/objectives.
5. **Observability or it didn't happen.** Every lab defines metrics that must move.
6. **Cluster discipline.** All long runs follow the GT-cluster workflow (§7): tmux, pinned GPUs, pre-launch occupancy checks, checkpoints on `/coc/scratch`, artifacts pulled back for notebook analysis.

**Difficulty scale:** ★☆☆ ≤ half day · ★★☆ ~1 day · ★★★ multi-day / subtle debugging

---

## 3. Hardware plan (verified environments)

| Tier | Where | VRAM | Runs |
|---|---|---|---|
| **A — prototype** | Colab (T4) | 16 GB | Labs 00–03 at 0.6B; Lab 05 (CPU); quick smoke-tests of everything |
| **B — train** | GT CoC `mal`/`inara`/`jayne`/`shepherd` (8× A40) | 46 GB | Full-finetune 1.7B (SFT/DPO/RM/GRPO); 4B with LoRA; SDPO reference-scale runs |
| **C — avoid for training** | `rivertam`/`simontam` (2080 Ti) | 11 GB | Only tokenization/data prep; too small for training batches |

Cluster facts that shape lab design (from the verified handoff):
- `/coc/scratch` is ONE NFS volume visible identically on all six nodes → **datasets, tokenized caches, checkpoints written once are instantly cluster-global.** Each lab stores data/checkpoints under `/coc/scratch/vchopra/post_training_labs/<lab>/`.
- No sudo → per-lab venv instructions target `~/venvs/` or a shared venv on scratch.
- Foreign jobs share nodes → labs mandate `nvidia-smi --query-compute-apps` immediately before launch, modest batch sizes, `CUDA_VISIBLE_DEVICES` pinning, and tmux wrapping. **No unattended job starts without your explicit go-ahead.**
- Colab ↔ cluster contract: every training lab ships a notebook (prototype/small-run) AND a `train.py` export (cluster run), sharing one config file. Results (JSON metrics + samples) always come back to the notebook for plotting.

Per-lab wall-clock below assumes Tier B for 1.7B runs.

---

## 4. Model & dataset register (with rationale)

| Role | Primary choice | Why this one | Modern extension |
|---|---|---|---|
| Base model for SFT | `Qwen/Qwen3-0.6B-Base` (Tier A) / `Qwen/Qwen3-1.7B-Base` (Tier B) | Current-gen, Apache-adjacent licensing, excellent tokenizer/tooling support; 0.6B trains on free Colab, 1.7B on A40 shows clearer quality gains | `HuggingFaceTB/SmolLM2-360M` for ultra-fast iteration |
| Answer-key comparable SFT | `allenai/OLMo-2-0425-1B` (+ `-SFT` template donor) | Exact match to repo reference run; lets you diff your loss curve against the published W&B run | — |
| SFT dataset | `HuggingFaceH4/no_robots` (9.5k, human-written) | Human-authored (no model artifacts to confound), small enough to overfit deliberately, repo-comparable | `HuggingFaceTB/smoltalk` subset (e.g., `smol-magpie-ultra[:20000]`) — the corpus behind SmolLM2, representative of modern synthetic SFT mixes |
| Preference dataset (Labs 02–04) | `argilla/ultrafeedback-binarized-preferences-cleaned` | Still the canonical educational preference set; its *documented noise/bias* is precisely what Lab 02 teaches you to find; repo-comparable | `allenai/tulu-3-preference-mixture` subset — what a 2025 lab actually trained on |
| RLVR task | Reasoning-Gym `spell_backward` + `openai/gsm8k` (exact-match verifier) | Deterministic verifiers; string task makes failures human-inspectable, GSM8K makes results feel real; both are the repo's own choices | `reasoning_gym` additional tasks (`rectangle_count`, etc.) for generalization checks |
| ORM data | `RLHF-Book/gsm8k-qwen3-0.6B-rollouts` (released rollout set, 100/prompt) | Pre-generated+labeled → no generation step needed to study outcome supervision | Self-generated rollouts from your Lab 06 engine |
| PRM data | `tasksource/PRM800K` slice | The original OpenAI process-supervision labels | Math-Shepherd (auto-labeled) for contrast |
| Rejection-sampling scorer | **Your own Lab 03 RM** | Pedagogically superior to the repo's AceMath-7B-RM: you trained the scorer, so scoring failures are diagnosable | `nvidia/AceMath-7B-RM` as a "production RM" comparison arm |
| LLM-as-judge (Lab 11) | `Qwen/Qwen3-4B` (Tier B, fits easily in 46 GB) | Big enough to judge sensibly, small enough to run locally — no API dependency | API judge (any frontier model) for bias comparison |

**Deviation policy:** wherever we substitute (model size down, dataset subset, RM swapped), the lab README records the deviation and why, per your requirement to document departures from Lambert's setups.

---

## 5. Curriculum at a glance

| # | Lab | Concepts | Derived from | Difficulty | Compute tier | Core? |
|---|-----|----------|--------------|------------|--------------|-------|
| 00 | Data & chat templates | tokenization, masking, labels | `instruction_tuning/utils.py` | ★☆☆ | A/CPU | CORE |
| 01 | SFT from scratch (+LoRA-from-scratch, packing) | masked CE training loop | `instruction_tuning/` | ★★☆ | A/B | CORE |
| 02 | Preference-data forensics (+ decontamination) | bias, noise, contamination | Ch. 10/11 + repo pair-filtering | ★☆☆ | A/CPU | CORE |
| 03 | Bradley-Terry reward model | BT loss, margins, calibration | `reward_models/train_preference_rm.py` | ★★☆ | A/B | CORE |
| 03b | ORM vs PRM | outcome/process supervision | `train_orm.py`, `train_prm.py` | ★★★ | B | opt |
| 04 | DPO from scratch | ref model, log-ratios, β | `direct_alignment/loss.py`, `data.py` | ★★★ | B | CORE |
| 05 | Policy gradients on toy MDP | trajectory, advantage, variance | *(gap lab)* + reading repo losses | ★★☆ | CPU | CORE |
| 06 | REINFORCE/RLOO on an LLM | rollouts, baselines, KL drift | `policy_gradients/` reinforce/rloo | ★★★ | B | CORE |
| 07 | GRPO/RLVR | groups, advantages, clipping, KL estimators | `policy_gradients/loss.py::GRPOLoss` | ★★★ | B | CORE (capstone) |
| 08 | Reward hacking & KL | proxy-vs-true reward, Goodhart | Ch. 14/15 + reward components | ★★★ | B (reuses 07) | opt |
| 09 | Rejection sampling → SFT | best-of-N, controls | `rejection_sampling/` | ★★☆ | B | opt |
| 10 | SDPO on-policy distillation | teacher/student, reverse top-K KL | `distillation/` | ★★★ | B | opt |
| 11 | Evaluation & LLM-as-judge | win-rates, judge bias | *(gap lab)* + `diagnostics.py` | ★☆☆ | B | opt |
| 12 | Synthetic preference data | generate→judge→filter loop | Ch. 12 concepts | ★★☆ | B | opt |

Dependency chain:
```
00 → 01 → 02 → 03 → 04 ─┐
            └──────────→ 05 → 06 → 07 → 08
                         03+06 → 09       07 → 10        03(+04) → 11 → 12
```

---

## 6. Lab specifications

Each lab below keeps the required ten sections (why / prerequisites / assignment checklist / model / dataset / components / experiments / questions / debugging challenges / completion criteria) plus a **compare-against** pointer. Starter code in scaffolds will contain only signatures, docstrings, TODOs, and `raise NotImplementedError`.

---

### Lab 00 — Data & Chat Templates (★☆☆ · CORE)
- **Prerequisites:** Lecture 0 + Lecture 1; Chapter 3.
- **Why:** every later lab depends on knowing which token positions carry gradient; most post-training bugs are mask bugs.
- **Assignment:** inspection toolkit for `Qwen/Qwen3-0.6B-Base`, `OLMo-2-0425-1B` (+`-SFT` donor), `SmolLM2-360M-Instruct`:
  - [ ] tokenize a 2-turn conversation raw vs chat-templated; diff them; render special tokens/IDs
  - [ ] build prompt-masked labels (-100 outside final assistant turn); verify by decoding only unmasked positions
  - [ ] demonstrate base continuation vs instruct answer-and-stop on 6 fixed prompts
  - [ ] right-pad a batch; show where naive pooling corrupts if attention mask ignored
- **Model:** see register. **Dataset:** no_robots (first 50 rows). **Compute:** CPU/Tier C.
- **Experiments:** decode-unmasked for 10 samples; length distributions base-vs-instruct.
- **Questions:** Why does OLMo base need a template donor? What breaks if EOS isn't supervised? When does pad side matter?
- **Debugging:** unmasked decode includes EOS+padding — why? Mask includes assistant header token — correct?
- **Done when:** any conversation printed as `(token, id, label)` triples, every row explained.
- **Compare:** `code/instruction_tuning/utils.py:153-244`.

### Lab 01 — SFT From Scratch (★★☆ · CORE)
- **Prerequisites:** Lab 00; Lecture 2 pt.1; Chapter 4.
- **Why:** the base→assistant transition must be something you caused.
- **Assignment:** no HF Trainer:
  - [ ] dataset/collate/DataLoader with prompt-masked batching
  - [ ] bf16 autocast loop: grad accumulation, clip-grad-norm, val loss, periodic generation panels
  - [ ] **from-scratch checkpoint: implement a LoRA Linear layer yourself** (A·B low-rank wrap, scaling α/r), verify it matches a frozen-linear + adapter forward, then SFT 0.6B with your own LoRA on Colab
  - [ ] optional: naive sequence packing (concatenate examples, position_ids reset) — measure throughput gain and discuss cross-contamination risk
  - [ ] greedy vs sampled generation harness; fixed 6-prompt panel step-0 vs final
- **Model:** Qwen3-0.6B-Base (Colab full-FT) → Qwen3-1.7B-Base (A40). **Dataset:** no_robots 2k→9.5k rows; extension SmolTalk 20k subset. **Compute:** few hours (0.6B) / overnight (1.7B).
- **Experiments:** lr {1e-6, 5e-6, 2e-5}; epochs {1,3}; sizes {500, 2k, 9.5k}; LoRA r∈{8,32} vs full-FT.
- **Metrics:** train/val loss, grad norm, response length, panels.
- **Questions:** Why mask prompts? What's the early loss cliff? How does LoRA capacity limit the behavior shift? When is packing unsafe?
- **Debugging:** generations never stop — what did you not supervise? Loss ≈0 after warmup — what did you mask away?
- **Done when:** base-vs-SFT transcripts on held-out prompts + narrated loss plot + LoRA-vs-full comparison table.
- **Compare:** `instruction_tuning/train.py`, `configs/sft_olmo2_1b.yaml`.

### Lab 02 — Preference Data Forensics (★☆☆ · CORE)
- **Prerequisites:** Lecture 8; Chapters 10–11.
- **Why:** RMs inherit every artifact in the pairs; audit precedes modeling.
- **Assignment:** analyze ≥3k pairs of UltraFeedback-binarized-cleaned:
  - [ ] schema mapping under real field names; length-bias stats (P(chosen longer)); per-source breakdown
  - [ ] manual noise audit: 100 pairs vs your own judgment; near-tie detection
  - [ ] formatting-artifact analysis (lists/markdown win rates)
  - [ ] cleaned-subset builder with explicit filter rules + removal report
  - [ ] **decontamination check: 8-gram overlap of prompts vs GSM8K test set** — quantify benchmark leakage and write a decontamination filter
  - [ ] extension: load 500 Tülu-3 preference rows; compare bias profile vs UltraFeedback
- **Model:** none. **Dataset:** UF-cleaned ≥3k (+Tülu-3 ext). **Compute:** CPU/pandas.
- **Questions:** If 60% of chosen are longer, what else does the RM learn? How did UF's 4-resp→1-pair construction shape it? Why does industry decontaminate against test sets?
- **Done when:** 4-figure written audit + statement of biases inherited by Lab 03 + leakage number.
- **Compare:** pair-filtering in `reward_models/train_preference_rm.py:84-190`.

### Lab 03 — Bradley-Terry Reward Model (★★☆ · CORE)
- **Prerequisites:** Lab 02; Lecture 2 pt.2; Chapter 5 BT.
- **Why:** own the derivation of r(chosen) > r(rejected).
- **Assignment:** from-scratch RM (no TRL): backbone + `Linear(hidden,1)` head pooled at last non-pad token; paired batching; **hand-written BT loss**; pairwise accuracy, mean margin, reward histograms; margin-bucketed accuracy (calibration); score your Lab 02 near-ties and compare to your human judgments.
- **Model:** Qwen3-0.6B-Base. **Dataset:** your cleaned UF subset (2–5k). **Compute:** ~1–2 h on A40; feasible on Colab.
- **Experiments:** freeze-backbone vs full-FT; lr sweep; accuracy vs data {1k, 2k, 5k}.
- **Questions:** Why sigmoid(diff)? Why is absolute reward scale meaningless? Accuracy↑ while margins collapse — what does that mean for downstream RL?
- **Debugging:** accuracy stuck 50% — check pooling index & pad side. Margins explode, accuracy flat — what is BT optimizing then?
- **Done when:** trace one pair end-to-end naming every tensor shape.
- **Compare:** `reward_models/base.py`, `train_preference_rm.py`.

### Lab 03b — ORM vs PRM (★★★ · optional)
- Binary correctness head on GSM8K rollouts (released rollout set or your Lab 06 rollouts); {-1,0,1} step classifier on PRM800K slice. Deliverable: disagreement case-study table on ≥20 solutions (correct-answer-wrong-reasoning cases). **Models:** Qwen3-0.6B. **Compare:** `reward_models/train_orm.py`, `train_prm.py`.

### Lab 04 — DPO From Scratch (★★★ · CORE)
- **Prerequisites:** Lab 03; Lecture 6; Chapter 8.
- **Why:** sequence log-probs become load-bearing; derive the objective once, never treat trainers as black boxes again.
- **Assignment:**
  - [ ] `sequence_logprob(model, ids, response_mask)` — verify against `F.cross_entropy(reduction='none')`
  - [ ] paired chosen/rejected through policy AND frozen ref (4 forwards)
  - [ ] **hand-written DPO loss**(policy_c, policy_r, ref_c, ref_r, β)
  - [ ] implicit rewards/margins/accuracy/length metrics; train OLMo-2-1B-SFT on your subset
  - [ ] β sweep {0.05, 0.1, 0.5}: margin growth vs mean |log-ratio| drift
  - [ ] optional: TRL DPOTrainer few-hundred-step cross-check
- **Model:** `allenai/OLMo-2-0425-1B-SFT` (repo parity) or Qwen3-1.7B-Instruct-variant for modernity. **Dataset:** Lab 02 cleaned subset (1–3k). **Compute:** A40, several hours.
- **Questions:** Why is the reference model required? β→∞ / β→0 limits? Why can rejected log-probs fall while accuracy rises? Where's the gradient signal with zero sampling?
- **Debugging:** accuracy→1.0 in 50 steps + degraded generations — diagnose. Margin grows while chosen implicit reward negative — interpret.
- **Done when:** narrate pair through template→logprobs→log-ratios→loss→backward; β-sweep figure.
- **Compare:** `direct_alignment/loss.py::DPOLoss`, `data.py`, `configs/dpo.yaml`; read IPOLoss/SimPOLoss after.

### Lab 05 — Policy Gradients on a Toy MDP (★★☆ · CORE · gap lab)
- **Prerequisites:** Lecture 3; Chapter 6 early sections. *(Repo jumps straight to 1.7B rollouts — this on-ramp is ours.)*
- **Assignment:** char-level policy net (<1M params, CPU) on a hand-verifiable task (e.g., emit arithmetic expression evaluating to N):
  - [ ] env definition; trajectory sampling with log-prob tracking
  - [ ] **REINFORCE by hand** incl. own softmax/log-softmax; −log π(τ)·A
  - [ ] moving-average baseline; leave-one-out baseline across K samples; variance comparison plots
  - [ ] entropy collapse observation
- **Experiments:** baseline on/off; K∈{2,4,16}.
- **Questions:** Why log-prob not prob in the theorem? What does a baseline change — and not change? Why is variance THE practical problem?
- **Debugging:** collapse to repeated action — name it + two countermeasures. Nonzero gradient though identical group rewards — trace it.
- **Done when:** working REINFORCE + variance plots explained line-by-line.
- **Compare:** `policy_gradients/loss.py::ReinforceLoss` (after yours works).

### Lab 06 — REINFORCE/RLOO on an LLM (★★★ · CORE)
- **Prerequisites:** Lab 05; Lecture 4; Chapter 6 implementation sections.
- **Assignment:** minimal RL loop on `spell_backward` (inspectable) **plus GSM8K-1k exact-match** (real):
  - [ ] rollout engine: K completions/prompt (K=4–8), old log-probs recorded
  - [ ] your own verifiers (string match + format; GSM8K answer extraction)
  - [ ] REINFORCE-no-baseline then RLOO advantages
  - [ ] on-policy single epoch first; optional reuse epoch → ratio-statistics drift check
  - [ ] monitor avg correctness/format, length, approx-KL(k1) to init policy
- **Model:** Qwen3-0.6B (dev) → Qwen3-1.7B (A40; deviation from repo's 1.7B-default documented if 0.6B used). **Compute:** overnight OK.
- **Experiments:** REINFORCE vs RLOO; temp {0.6, 1.0}; K {4,8}; spell_backward vs GSM8K transfer of insights.
- **Debugging:** correctness flat while format climbs — partial hack. KL explodes mid-run — predict the failure mode.
- **Done when:** correctness beats base rate + annotated dump: prompt→K rollouts→rewards→advantages.
- **Compare:** `policy_gradients/rollout.py`, `utils.py`, configs `reinforce.yaml`/`rloo.yaml`.

### Lab 07 — GRPO/RLVR (★★★ · CORE CAPSTONE)
- **Prerequisites:** Lab 06; Lectures 3–4 revisited + Lecture 10; Chapter 6 GRPO.
- **Assignment:**
  - [ ] group sampling (N=8) + verifiers (spell_backward + GSM8K)
  - [ ] standardized (GRPO) AND non-standardized (Dr.GRPO) group advantages — implement both
  - [ ] clipped surrogate, asymmetric eps support
  - [ ] **KL estimators k1/k2/k3 implemented by hand**, plotted on same batch
  - [ ] zero-contrast group handling (skip vs weight) — implement, justify
  - [ ] diagnostic panel: correctness/format/group-contrast/KL/entropy/length/grad-norm
- **Model:** Qwen3-0.6B→1.7B. **Compute:** A40; multi-day with sweeps.
- **Experiments:** N∈{4,8,16}; β∈{0, 0.001, 0.04}; eps sweep; token-level vs sequence-mean normalization.
- **Questions:** What breaks when all rewards in a group tie (mechanically + statistically)? Std-normalization bias (GRPO vs Dr.GRPO)? Why GRPO over PPO for LLMs (value-net cost)? Token-level normalization and length behavior?
- **Debugging:** "all advantages ≈ 0" — investigate saturation/std blow-up. Reward↑, KL↑, sudden gibberish collapse — reconstruct timeline from panels.
- **Done when:** learning curve above base rate + annotated group dump + KL-estimator comparison plot.
- **Compare:** `policy_gradients/loss.py::GRPOLoss`, `compute_standardized_advantages`, `configs/grpo.yaml`; then GSPOLoss/DAPOLoss readings.

### Lab 08 — Reward Hacking & KL Regularization (★★★ · optional)
- Design proxy reward misaligned with true goal (large format bonus); rerun Lab 07 config; track true accuracy separately; sweep KL coefficient; produce proxy-up/true-down divergence plot; identify earliest warning metric. **Reuses Lab 07 runs.**

### Lab 09 — Rejection Sampling → SFT (★★☆ · optional)
- Best-of-N on GSM8K (N=8), scored by **your Lab 03 RM** (+AceMath-7B-RM comparison arm); select top-per-prompt & top-k-overall; SFT each; evaluate exact-match against matched random controls (the repo's key control). Deliverable: strategy-vs-random table; explain repo's finding (top_k_overall wins, top_per_prompt ties). **Model:** Qwen3-0.6B/1.7B. **Compare:** `rejection_sampling/` all four configs.

### Lab 10 — SDPO On-Policy Distillation (★★★ · optional, NEW)
- **Prerequisites:** Lab 07; Lecture 7; Chapter 12 (incl. OPSD section).
- **Why:** most current technique in the repo (self-distillation for reasoning); previously dropped, now restored.
- **Assignment:**
  - [ ] group rollouts on `spell_backward`; skip-and-refill groups with zero correct samples (watch the `skipped` metric)
  - [ ] demonstration-conditioned teacher = same model reprompted with correct sibling rollout
  - [ ] **top-K reverse-KL distillation loss implemented by hand**, incl. tail-bucket trick for non-top-K mass
  - [ ] track reward, distill loss, skipped-rate as the loop converges
- **Model:** Qwen3-1.7B (repo parity; A40 handles the ~20 h reference-scale run comfortably, or shorten num_steps). **Compare:** `distillation/loss.py` (after yours), `configs/sdpo.yaml`.
- **Key questions:** Reverse vs forward KL — which modes get preserved? Why condition the teacher on a sibling demo rather than gold answers? Why skip empty groups?

### Lab 11 — Evaluation & LLM-as-Judge (★☆☆ · optional, gap lab)
- Mini-eval suite over ALL saved checkpoints (base/SFT/DPO/RL): fixed prompt set; win-rate matrix via (a) your RM, (b) Qwen3-4B judge; quantify judge position/length/self-preference biases by order-swapping flip rates; compare verdicts to verifiable ground truth. Extension: synthetic preference generation → judge-filter → measure introduced bias (absorbs old "Lab 12"). **Compare:** `rejection_sampling/diagnostics.py` methodology.

---

## 7. Infrastructure & ops (every lab inherits this)

**LABS_SETUP.md will cover (scaffold phase):**
1. **Local/laptop:** `uv sync` clone of repo for answer-key reading only.
2. **Colab profile:** pip install pinned deps; `HF_TOKEN` + optional `WANDB_API_KEY` secrets; mount Drive or stream datasets; T4-safe settings (bf16 off → fp16, batch sizing).
3. **GT cluster profile (per the verified handoff):**
   - venv at `/coc/scratch/vchopra/venvs_post_training/` (torch cu126 wheel index), reused by all labs
   - all data/checkpoints/logs under `/coc/scratch/vchopra/post_training_labs/<lab>/` (NFS-shared → node-agnostic)
   - `~/bin/gpu-status` + `nvidia-smi --query-compute-apps` BEFORE launch; expect foreign jobs mid-run
   - tmux-wrapped launches with `CUDA_VISIBLE_DEVICES` pinned; log polling via ssh tail; TensorBoard port-forward recipe
   - **hard rule: no unattended job starts without explicit user go-ahead**
4. **Notebook↔script contract:** every training lab = `notebook.ipynb` (prototype, small-run, plotting) + `train.py` (exportable, config-driven, resumable checkpoints) sharing one YAML config.
5. **Metrics convention:** every run appends JSONL metrics (wandb optional), so notebooks plot from artifacts regardless of platform.
6. **Checkpoint/resume + seeds:** fixed seed protocol; resume-from-checkpoint exercised at least once in Labs 01/06/07.

---

## 8. From-scratch checkpoints (summary)

Sequence log-probs · masked cross-entropy · **LoRA layer** · Bradley-Terry loss · DPO loss · REINFORCE objective · LOO & group-relative advantages · KL estimators k1/k2/k3 · top-K reverse-KL (SDPO) · verifiers (string-match, GSM8K extraction). Frameworks (TRL etc.) enter only as post-hoc cross-checks.

## 9. Parallel-scaffolding contract (for the execution session)

When subagents build the labs in parallel, each receives:
1. This document + `REPOSITORY_POST_TRAINING_MAP.md` (self-contained context).
2. One lab spec from §6 — build ONLY that lab under `labs_curriculum/labs/<NN>_<slug>/`.
3. Uniform structure: `README.md` (spec expanded, with prerequisites/lectures links), `starter.py` + `notebook.ipynb` skeleton, `tests/test_<slug>.py`, `configs/<slug>.yaml`.
4. **No-solutions rule:** signatures, docstrings describing contracts, TODO markers, shape comments, `raise NotImplementedError`. Assertions/tests verify shapes/invariants (e.g., "labels at prompt positions are all -100"), never implementations. Forbidden-pattern review: e.g., a DPO starter must not contain `logsigmoid`; a GRPO starter must not contain advantage arithmetic.
5. Tests must collect/run WITHOUT GPU or network (CPU tensor fixtures only).
6. Review gate per lab: second agent verifies no-solutions compliance + test runnability before merge into the labs tree.
7. `labs_curriculum/LABS_SETUP.md` built once (single agent), per §7.

## 10. Resolved questions & remaining decisions

- Hardware: **resolved** — Colab (Tier A) + GT A40 cluster (Tier B); 2080Ti nodes data-prep only.
- Tests: **yes**, pytest per lab, GPU-free (contract §9.5).
- First optional lab after core: recommendation **Lab 08 (reward hacking)** — cheapest (reuses Lab 07 runs) and highest interview value; then 09, 10, 11.
- Remaining documented gaps (accepted): tool-use post-training (Ch. 13) has no executable upstream code and stays a stretch goal; multi-GPU/FSDP out of scope.
