# Lab 10 — SDPO On-Policy Distillation (★★★ · optional)

On-policy self-distillation for reasoning: the same model plays both roles. The
**student** samples a group of rollouts per prompt; when at least one rollout is
verifiably correct, that sibling demo is spliced back into the prompt and the
model is **reprompted as its own teacher**, conditioned on the demonstration.
The student's on-policy logits are distilled toward the teacher's via a
hand-written **top-K reverse-KL** loss, with a **tail bucket** carrying all
non-top-K probability mass so the K+1 distribution is valid.

Everything here is built from scratch: you will not import the repo's
`distillation` module until after your implementation works.

## Prerequisites

Before starting this lab you should have finished and understood:

| Prereq | Why it matters here | Link |
| --- | --- | --- |
| **Lab 07 — GRPO/RLVR** | Group rollouts, verifiable reward, `spell_backward`, skip-empty-group reasoning | [labs/07_grpo_rlvr](../07_grpo_rlvr/) |
| **Lecture 7 — Synthetic data & distillation** | The SDPO / OPSD framing: teacher = same model reprompted with a correct demo | [rlhfbook.com/teach/course/lec7-chap12-synthetic-data](https://rlhfbook.com/teach/course/lec7-chap12-synthetic-data/) |
| **Chapter 12 (incl. the OPSD section)** | Motivation, why sibling demos beat gold answers, top-K reverse-KL design | [rlhfbook.com/c/12-synthetic-data](https://rlhfbook.com/c/12-synthetic-data) |

Chapter 8 of RLHF (reverse vs forward KL mode-seeking behavior) is useful
background for the "key questions" below.

## Assignment checklist

- [ ] **Group rollouts** on `spell_backward`: sample `num_rollouts` completions
      per prompt at temperature `0.6`, score each with the string verifier.
- [ ] **Skip-and-refill:** when *zero* rollouts in a group meet
      `success_reward_threshold: 1.0`, discard the group, count it under the
      `skipped` metric, and poll a fresh prompt until
      `prompts_per_step` full groups are collected (with a hard polling bound so
      an impossible task fails fast instead of hanging).
- [ ] **Demonstration-conditioned teacher:** take one correct sibling rollout,
      build the teacher prompt `question + "Correct solution:\n\n" + demo +
      "Correctly solve the original question."`, and rerun the *same* model on
      it to obtain teacher logits over exactly the student's completion tokens.
- [ ] **Top-K reverse-KL distillation loss, by hand:** project both students'
      and teacher's log-distributions onto the student's top-`kl_top_k` tokens;
      close each distribution with a tail bucket covering the non-top-K mass
      (`add_tail_bucket`) so K+1 probabilities sum to 1; compute reverse KL —
      KL(student ‖ teacher) with the gradient flowing through the student side.
      Chunk rollouts (`rollout_chunk`) with backward per chunk so peak memory
      stays bounded; chunk losses divide by the global action-token count so
      gradients accumulate to the full-group gradient.
- [ ] **Metrics loop:** track `reward`, `distill_loss`, and skipped-rate while
      the loop converges. Append JSONL metrics every step so the notebook can
      plot from artifacts.

## Key questions (answer these in the notebook)

1. Reverse vs forward KL — which modes get preserved? What does the reverse-KL
   direction do to low-probability nonsense modes of the teacher?
2. Why condition the teacher on a *sibling* demo rather than gold answers?
   (Hint: think about token-level path/trace alignment.)
3. Why skip empty groups entirely? What would distilling from an all-wrong
   group teach the model? Watch how `skipped` decays as training converges.
4. The tail bucket means gradient never reaches teacher tokens outside the top
   K. When does that matter for the `spell_backward` vocabulary?

## Files

```
10_sdpo_distillation/
├── README.md                       # this file
├── starter.py                      # implement all TODO stubs here (torch-free import)
├── notebook.ipynb                  # prototype small runs + plot reward / loss / skipped-rate
├── tests/
│   └── test_10_sdpo_distillation.py  # structural invariants + stub behavior (CPU-only)
└── configs/
    └── 10_sdpo_distillation.yaml    # your run config (mirrors sdpo.yaml defaults)
```

## Running

```bash
# In the lab directory:
python3 -m pytest tests/ -q          # structural tests pass against the untouched starter

# Training (GPU machine only; see LABS_SETUP.md for cluster profile):
python3 train.py --config configs/10_sdpo_distillation.yaml   # once you finish the stubs
```

## Model & scale

Qwen3-1.7B (repo parity). The A40 cluster handles the ~20 h reference-scale run
comfortably; shorten `num_steps` first. Chat template: thinking enabled
(`enable_thinking: true`), prompts capped at `max_prompt_len: 512`, teacher
re-prompt capped at `max_reprompt_len: 1024` (problem + full sibling demo).

## Compare (only after yours works)

- [`rlhf-book/code/distillation/loss.py`](../../../rlhf-book/code/distillation/loss.py)
  — note especially the `add_tail` tail-bucket trick and the chunked-forward /
  backward-per-chunk memory story.
- [`rlhf-book/code/distillation/configs/sdpo.yaml`](../../../rlhf-book/code/distillation/configs/sdpo.yaml)

Report any divergence between your hand-written loss and theirs in the notebook.
