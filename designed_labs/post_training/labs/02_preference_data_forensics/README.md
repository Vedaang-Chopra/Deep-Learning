# Lab 02 — Preference Data Forensics

> ★☆☆ · **CORE** · est. build ½ day · Scaffold status: **stubs only** (`starter.py` raises `NotImplementedError` everywhere)

**Goal:** audit ≥3k pairs of `argilla/ultrafeedback-binarized-preferences-cleaned` *before* any reward modeling — schema mapping under real field names, length-bias statistics, near-tie detection, a 100-pair manual noise audit, formatting-artifact analysis, a cleaned-subset builder with explicit filter rules + removal report, and an **8-gram decontamination check of prompts against the GSM8K test set**. RMs inherit every artifact in the pairs; audit precedes modeling.

---

## Prerequisites

- **Lecture 8 (Preferences & preference data)** — watch before starting: <https://rlhfbook.com/course/>
- **Chapter 10, Preferences** — read for what a preference pair *is* and where labels come from: <https://rlhfbook.com/c/10-preferences.html>
- **Chapter 11, Preference Data** — read for dataset construction and its documented biases: <https://rlhfbook.com/c/11-preference-data.html>
- Local setup: see `labs_curriculum/LABS_SETUP.md`.

## Why

Reward models sit upstream of everything else in post-training (Labs 03–04, 07–09 all consume preference data or RM scores), and they inherit every artifact in the pairs they train on. UltraFeedback's 4-responses→1-pair construction, its length asymmetries, and its formatting quirks are precisely what your Lab 03 RM will learn. This lab makes those artifacts visible and quantified — and teaches the decontamination discipline every industry lab applies against benchmark test sets.

## Assignment checklist

- [ ] Schema-map raw rows → canonical pairs under **real field names** (`normalize_pair`, `schema_report`) — handle both string and conversation-format `chosen`/`rejected`, skip unusable rows
- [ ] Length-bias stats: **P(chosen longer)** in chars and words, means/medians (`length_bias_stats`); per-source breakdown (`per_source_breakdown`)
- [ ] Manual noise audit: deterministic 100-pair worksheet (`sample_manual_audit`), judge them **yourself**, then score agreement (`audit_agreement`)
- [ ] Near-tie detection: char n-gram Jaccard similarity, no `difflib` shortcuts (`detect_near_ties`)
- [ ] Formatting-artifact analysis: list/markdown marker win rates (`formatting_win_rates`)
- [ ] Cleaned-subset builder with explicit named filter rules + **removal report** (`build_cleaned_subset`)
- [ ] **Decontamination check: 8-gram overlap of prompts vs GSM8K test set** — quantify leakage (`ngrams`, `decontamination_check`) and write the decontamination filter (`decontamination_filter`)
- [ ] Extension: load 500 Tülu-3 preference rows (`load_pairs_jsonl`), compare bias profiles vs UltraFeedback (`compare_bias_profiles`)

## Dataset

`argilla/ultrafeedback-binarized-preferences-cleaned`, ≥3k pairs (its *documented noise/bias* is exactly what this lab teaches you to find). GSM8K test prompts loaded **locally** (no network during tests). Extension: `allenai/tulu-3-preference-mixture` subset (500 rows).

## Model / Compute

None — this is a data lab. CPU + stdlib python (pandas optional, in the notebook only; `starter.py` and all tests are stdlib-only and run with **no network**).

## Components ↔ starter.py map

| Component | Function(s) |
|---|---|
| loading + schema mapping | `normalize_pair`, `schema_report`, `load_pairs_jsonl` |
| length bias + per-source | `length_bias_stats`, `per_source_breakdown` |
| near-tie detection | `detect_near_ties` |
| manual noise audit | `sample_manual_audit`, `audit_agreement` |
| formatting artifacts | `formatting_win_rates` |
| cleaned subset + removal report | `build_cleaned_subset` |
| 8-gram decontamination vs GSM8K | `ngrams`, `decontamination_check`, `decontamination_filter` |
| Tülu-3 extension | `compare_bias_profiles` |

Config mirror: `configs/02_preference_forensics.yaml` ⇄ `ForensicsConfig`.

## Questions

1. If 60% of chosen are longer, what else does the RM learn — and how would you tell preference signal from length signal apart?
2. How did UltraFeedback's 4-responses→1-pair construction shape the label noise you found in your 100-pair audit?
3. Why does industry decontaminate against test sets? What exactly does a shared 8-gram between a training prompt and GSM8K test set evidence?
4. Your cleaned subset drops X% of pairs — which filter rule removed the most, and was each removal actually justified?

## Done when

A **4-figure written audit** (length-bias, per-source, formatting win rates, near-tie/decontamination) + a **statement of the biases your Lab 03 RM will inherit** + the **GSM8K leakage number** (flag rate) for your slice.

## Compare against (answer key — open ONLY after your audit pipeline works)

- `rlhf-book/code/reward_models/train_preference_rm.py:84-190` — `build_preference_dataset`: schema handling under real field names (string vs conversation `chosen`/`rejected`), skipping unusable rows, and dropping token-identical pairs (constant BT loss, zero gradient). Compare its pair-filtering against YOUR cleaned-subset rules.

Deviation policy note (per plan §4): none currently — dataset matches the register default. Record any future deviations here.
