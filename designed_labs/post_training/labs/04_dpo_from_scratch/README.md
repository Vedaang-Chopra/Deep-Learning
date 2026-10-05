# Lab 04 — DPO From Scratch (★★★ · CORE)

> **Build time:** 2–3 days · **Compute tier:** B (A40), prototype on Tier A
> **Contract:** you implement every mechanism below yourself. The rlhf-book repo is the **answer key** — open it only after your version trains, then diff.

---

## Why this lab exists

Sequence log-probs become load-bearing here: DPO's entire objective is built from the log-probability of full responses under two models (policy and reference). After this lab, never again treat a preference-optimization trainer as a black box — you will have derived the objective once, by hand, from four forward passes and a scalar β.

## Prerequisites

- **Lab 03** — Bradley-Terry reward model completed (you already own paired batching and sequence-level pooling).
- **Lecture 6** (DPO) at [rlhfbook.com/course](https://rlhfbook.com/course).
- **Chapter 8** — Direct Alignment, [rlhfbook.com/c/08-direct-alignment.html](https://rlhfbook.com/c/08-direct-alignment.html). Derive the objective on paper *before* writing code.

## Assignment checklist

Implement in `starter.py` (bodies are `TODO`s / `NotImplementedError` on purpose):

- [ ] `sequence_logprob(logits, input_ids, response_mask)` — sum of per-token log-probs over response positions only. Handle the autoregressive shift. **Verify against `F.cross_entropy(reduction='none')`** on a hand-built batch.
- [ ] Paired batching helpers (`tokenize_preference_pair`, `collate_preference_batch`) producing `PreferenceBatch` with prompt/response masks that separate prompt tokens from supervised positions.
- [ ] Four forward passes per step: chosen/rejected through the **policy**, chosen/rejected through the **frozen reference model**.
- [ ] `dpo_loss(policy_c, policy_r, ref_c, ref_r, beta)` — hand-written, no shortcuts. If your first draft reaches for `torch.nn.functional.logsigmoid`, stop and derive the objective yourself from the chapter; the point of this lab is that derivation.
- [ ] Metrics: implicit rewards (chosen & rejected), margin, pairwise accuracy, response length. Nothing here may be implemented before the loss works.
- [ ] Train OLMo-2-1B-SFT on your Lab 02 cleaned subset.
- [ ] β sweep {0.05, 0.1, 0.5}: plot margin growth vs mean |log-ratio| drift.
- [ ] Optional: TRL `DPOTrainer` few-hundred-step cross-check against your numbers.

## Model

| Choice | Model | Rationale |
|---|---|---|
| Primary | `allenai/OLMo-2-0425-1B-SFT` | Exact parity with the answer key's reference run |
| Modern alt | `Qwen/Qwen3-1.7B` (instruct variant) | Current-generation comparison |

The reference model is an additional frozen copy of the policy's initial weights — same architecture, twice the memory. Budget for it.

## Dataset

Your **Lab 02 cleaned subset** of `argilla/ultrafeedback-binarized-preferences-cleaned` (1–3k pairs). Deviation note: the answer key uses 6400 pairs at effective batch 64 (~300 steps); we shrink to fit A40 wall-clock and to make overfitting visible faster.

## Required experiments

1. Baseline training run, all metrics logged per §2 rule 5 (`metrics.jsonl` + optional wandb).
2. β ∈ {0.05, 0.1, 0.5} — three runs, otherwise identical config.
3. Verification experiment: your `sequence_logprob` vs `F.cross_entropy(reduction='none')` on identical batches — must match to float tolerance.

## Questions to answer

1. Why is the reference model required? What degenerate solution appears without it?
2. What are the β → ∞ and β → 0 limits?
3. Why can rejected log-probs fall while accuracy rises? Trace it through the loss.
4. Where does gradient signal come from when there is zero sampling in the loop?

## Debugging challenges (symptoms only — diagnose yourself)

- Accuracy hits 1.0 within ~50 steps and generations visibly degrade. Diagnose.
- Margin grows steadily while the chosen implicit reward stays negative. Interpret.

## Completion criteria

Done when you can narrate one pair end-to-end — chat template → tokenized tensors → four forward passes → log-probs → log-ratios → loss → backward — naming every tensor's shape and why it exists, **and** produce the β-sweep figure.

## Compare against the answer key

Only after your version trains:

- `rlhf-book/code/direct_alignment/loss.py::DPOLoss`
- `rlhf-book/code/direct_alignment/data.py`
- `rlhf-book/code/direct_alignment/configs/dpo.yaml`

Post-hoc reading (extensions of the same family): `IPOLoss` and `SimPOLoss` in the same file — note how IPO swaps classification for regression to a target margin, and SimPO drops the reference model entirely in favor of length-normalized log-probs.

## Files

```
04_dpo_from_scratch/
├── README.md                        # this file
├── starter.py                       # signatures + docstrings + TODOs (no solutions)
├── notebook.ipynb                   # prototype/small-run skeleton (Colab-friendly)
├── configs/
│   └── 04_dpo_from_scratch.yaml     # one config shared by notebook + train export
└── tests/
    └── test_04_dpo_from_scratch.py  # structural-invariant tests (GPU/network-free)
```

Run tests locally (they skip gracefully if torch isn't installed):

```bash
cd labs_curriculum/labs/04_dpo_from_scratch
python3 -m pytest tests/ -q
```
