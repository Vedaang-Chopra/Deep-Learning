# Lab 11 — Evaluation & LLM-as-Judge

> ★☆☆ · **optional · GAP LAB** · est. build 1 day · Scaffold status: **stubs only** (`starter.py` raises `NotImplementedError` everywhere)
>
> **Status note:** this lab has **no single module in `rlhf-book/code/`** — it is a deliberate gap lab built from Ch. 16 concepts plus the *methodology* of `rejection_sampling/diagnostics.py` (reward-vs-correctness diagnostics, per-row win-rate vs random baseline, best-of-N sweep, `decidable_fraction` framing). You reuse that measurement discipline, not its code. It also **absorbs the old "Lab 12" (synthetic preference data)** as its Part 5 extension, so the curriculum table's Lab 12 row is retired into this lab.

**Goal:** build a mini-eval suite that runs over **ALL of your saved checkpoints** (base / SFT / DPO / RL) on one frozen prompt set, produces win-rate matrices scored by two judge arms — (a) your Lab 03 Bradley-Terry RM and (b) a local LLM judge (`Qwen/Qwen3-4B`) — then *audits the judge*: position / length / self-preference bias via order-swap flip rates, and verdict-vs-ground-truth agreement on verifiable prompts. Extension: a generate → judge-filter synthetic preference loop and a measurement of the bias the loop itself introduces.

---

## Prerequisites

- **Labs 00–04 runs completed** — you need their saved checkpoints (base, SFT, DPO) and your Lab 03 RM; RL checkpoints from Labs 06/07 join the matrix if you ran them.
- **Lecture 12 (Evaluations)** — watch before building: <https://rlhfbook.com/course/>
- **Chapter 16** — read for the preference-bias taxonomy this lab measures: <https://rlhfbook.com/c/16.html>
- Lab 03's margin lesson (scale-free accuracy, margins decide) is used constantly.
- Local setup: see `labs_curriculum/LABS_SETUP.md` (Tier-B profile for the 4B judge, JSONL metric convention).

## Why

After every training lab you have asked some variant of "is it *better*?" and answered with a loss curve. Loss curves are not evaluations. This lab forces the harder question: *who says so, and can that referee be trusted?* You will find that judge rankings move when you swap presentation order, that RMs inherit the biases of their preference data, and that "both models were right anyway" (undecidable prompts) silently caps how much any judge comparison can tell you — the same headroom lesson as `rejection_sampling/diagnostics.py`'s `decidable_fraction`.

## Assignment checklist

- [ ] `load_prompt_set` — frozen JSONL prompt set (`prompt_id`, `prompt`, `category`); duplicate/missing-key validation; write down what breaks if you edit prompts between eval runs
- [ ] `build_head_to_head_pairs` — pairwise comparison units with token-length metadata; verify identical prompt-set coverage across checkpoints; seeded capping
- [ ] Two judge arms: `rm_judge_verdict` (Lab 03 RM, documented tie band, margin) and `llm_judge_verdict` (Qwen3-4B, forced parseable output, `unparseable` counted separately, temperature 0, randomized display order)
- [ ] `discover_checkpoints` + `run_checkpoint_suite` — generate → judge → matrices → bias audit → JSONL summary, smoke-run (8 prompts × 2 checkpoints × both arms) before the full sweep
- [ ] **`compute_win_rate_matrix`** — documented tie convention and diagonal; order-swap records un-flipped before aggregation; assert `win_rates[i][j] + win_rates[j][i] == 1.0` on real data
- [ ] **`order_swap_flip_rate`** — aligned two-order verdict lists; flip vs tie-conflict buckets; direction split (toward first vs second position)
- [ ] Bias reports: `position_bias_report` (P(first-position win) ≈ 0.5 for a symmetric judge), `length_bias_report` (P(longer wins) by gap bucket), `self_preference_bias_report` (lineage delta)
- [ ] **`verdict_vs_ground_truth`** — GSM8K-style exact match via your Lab 06 verifier; `decidable_fraction` reported separately from judge agreement; confusion counts
- [ ] Extension (old Lab 12): `generate_synthetic_preferences` → `judge_filter_preferences` (kept + rejected with reasons, `keep_rate`) → `measure_introduced_bias` (length preference, position leakage, margin-distribution shift)

## Model & judge

- Evaluated checkpoints: **your own** Labs 00–04(/06/07) artifacts.
- Judge arm (b): `Qwen/Qwen3-4B` — big enough to judge sensibly, small enough to run locally on Tier B; **no API dependency**. Register rationale per §4.
- Judge arm (a): your Lab 03 RM — pedagogically superior because scoring failures are diagnosable.
- Bias-comparison arm (optional): any frontier API judge, contrasted against the local judge's bias profile.

## Dataset

One **frozen prompt set** you assemble (JSONL): a few hundred prompts mixing `gsm8k_verifiable` (ground truth carried in the row), `open_ended`, and `instruction_following` categories. Verifiable rows power Part 4; the rest power the matrix. The set is written once and never edited mid-suite.

## Components ↔ starter.py map

| Component | Function/class |
|---|---|
| prompt set + pairing | `load_prompt_set`, `build_head_to_head_pairs` |
| judge arms | `rm_judge_verdict`, `llm_judge_verdict` |
| checkpoint discovery + suite | `discover_checkpoints`, `run_checkpoint_suite` |
| win-rate matrix | `compute_win_rate_matrix` |
| bias audit (order-swap) | `order_swap_flip_rate`, `position_bias_report`, `length_bias_report`, `self_preference_bias_report` |
| ground-truth comparison | `verdict_vs_ground_truth` |
| synthetic extension (old Lab 12) | `generate_synthetic_preferences`, `judge_filter_preferences`, `measure_introduced_bias` |

Config mirror: `configs/11_eval_llm_judge.yaml` ⇄ `EvalConfig`.

## Compute

Judge cost is O(#checkpoints² × #prompts) generations — with 4 checkpoints × 200 prompts × both orders ≈ 4,800 judge calls; at Qwen3-4B on an A40 (Tier B) budget ~2–4 h for the full suite. Smoke-run first: 8 prompts × 2 checkpoints × both arms. Cluster paths: `/coc/scratch/vchopra/post_training_labs/lab11_eval_judge/`. All long runs follow LABS_SETUP ops rules (tmux, pinned GPU, occupancy check, **no unattended starts**).

## Experiments

- **E1 — Two arms, one matrix.** Win-rate matrix from the RM vs from the LLM judge over the same checkpoints. Where do the rankings disagree, and does the disagreement concentrate on near-tie cells?
- **E2 — Order-swap flip rate.** Flip rate vs verdict margin: flips should concentrate on near-ties. Direction split (first vs second position) quantifies position bias.
- **E3 — Length bait.** Pair a strong short answer against a padded verbose answer (constructed subset): which arm falls for length? Contrast RM vs LLM-judge vs (optional) API judge.
- **E4 — Self-preference.** If the judge shares lineage with an evaluated checkpoint, measure the lineage delta; compare with an out-of-family judge.
- **E5 — Decidability ceiling.** `decidable_fraction` on your prompt set vs judge agreement on decidable prompts only — separate data headroom from judge skill.
- **E6 (extension) — Filtered-loop bias.** Generate → judge-filter synthetic preferences, then `measure_introduced_bias`: length preference and position leakage that survive the filter, and which bias a DPO model trained on them (Lab 04 loss) would inherit.

## Questions

1. Why is the win-rate matrix only meaningful under a *frozen* prompt set, and what must you re-run if you edit it?
2. Your judge flips verdict on 12% of order swaps. How does that number change your confidence in a 55%-vs-45% matrix cell? What extra measurement resolves it?
3. The RM and the LLM judge disagree on the *direction* of length preference. Which is measuring quality and which is confounded — and how does `verdict_vs_ground_truth` arbitrate on verifiable prompts?
4. Why must order-swapped verdicts be un-flipped before aggregation, and what does a bug there do to the matrix (hint: it mirrors it)?
5. `decidable_fraction` is 0.4 on your set. What is the maximum a perfect judge could ever show as a headline separation, and where does the rest of the spread come from?
6. In the synthetic loop, the judge is also the filter. Which of the three audited biases does this *necessarily* inject into the kept set, and how would Lab 04's DPO training amplify it?

## Debugging challenges

- Matrix asymmetry violation (`win_rates[i][j] + win_rates[j][i] != 1`): almost always an un-flipped order-swap record or an inconsistent tie convention between arms.
- High unparseable rate: the judge prompt's output format is under-specified — a parse failure is not a tie; count it.
- All cells ≈ 0.5: check whether your generations actually differ (decode two checkpoints side by side) before blaming the judge.
- RM arm ties everywhere: your Lab 03 tie band ε may swallow the whole margin distribution — plot margins first.
- Order-swap alignment errors: validate by (model_a, model_b, prompt_id) before computing flip rates.

## Done when

- Win-rate matrices from BOTH judge arms over ≥3 checkpoints on the frozen prompt set, with the symmetry assertion passing on real data.
- Order-swap flip rate + the three bias reports produced for the LLM judge, each with its counts and a written interpretation.
- `verdict_vs_ground_truth` table with `decidable_fraction` reported separately from judge agreement.
- Extension (recommended): a kept/rejected synthetic-pair set plus the introduced-bias measurements.
- Notebook narrates one full comparison end-to-end, naming every artifact written to `output_dir/metrics.jsonl`.

## Compare against (open only after your version works)

`rlhf-book/code/rejection_sampling/diagnostics.py` — reward-vs-correctness diagnostic: histogram separation, per-row win-rate vs random baseline, best-of-N sweep, and the `decidable_fraction` headroom framing. This lab borrows the *methodology* (measure against a baseline; report headroom separately from skill), not the code — there is no judge/eval module to copy in the repo, which is exactly why this is a gap lab.
