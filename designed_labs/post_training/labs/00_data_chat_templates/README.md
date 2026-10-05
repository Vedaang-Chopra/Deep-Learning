# Lab 00 — Data & Chat Templates (★☆☆ · CORE)

> **Difficulty:** ≤ half day · **Compute:** CPU only (Tier C is fine) · **First lab of the sequence.**

## 0. Watch first

- **Lecture 0** — prereq review: cross-entropy, log-probs, KL, MDP → <https://rlhfbook.com/course/>
- **Lecture 1** — Overview → <https://rlhfbook.com/course/>
- **Chapter 3** of the RLHF Book → <https://rlhfbook.com/>

## 1. Why this lab exists

Every later lab depends on knowing **which token positions carry gradient**. Most
post-training bugs are mask bugs: a loss computed over the wrong positions silently
trains the model to imitate user prompts, or to never produce EOS. Before you train
anything (Lab 01+), you will be able to print, position by position, exactly what a
causal-LM loss would learn from a chat-formatted example — and explain every token.

## 2. Prerequisites

- Lecture 0 + Lecture 1 watched; Chapter 3 read.
- Comfort with tokenizers (`tokenizer(text)`, `apply_chat_template`), and the idea
  that `F.cross_entropy(..., ignore_index=-100)` drops positions whose *label* is -100.
- Environment: `transformers` (tokenizers only — no GPU), `datasets`, `numpy`,
  `pytest`. The scaffold's tests run with **numpy + pytest only** (no torch needed).

## 3. Assignment checklist

Build an inspection toolkit for `Qwen/Qwen3-0.6B-Base`, `OLMo-2-0425-1B`
(with `-SFT` as chat-template donor), and `SmolLM2-360M-Instruct`:

- [ ] **Raw vs templated.** Tokenize a 2-turn conversation *raw* (plain text) vs
      *chat-templated*; diff the two; render every special token with its ID.
- [ ] **Prompt-masked labels.** Build labels with `-100` everywhere outside the
      **final assistant turn**; verify by decoding *only* the unmasked positions.
- [ ] **Base vs instruct.** Demonstrate base *continuation* vs instruct
      *answer-and-stop* on the 6 fixed prompts in `starter.FIXED_PANEL_PROMPTS`.
- [ ] **Padding & pooling corruption.** Right-pad a batch; show where naive pooling
      corrupts if the attention mask is ignored (and what left-padding changes).

Work in `notebook.ipynb` first, then paste implementations into `starter.py`.

## 4. Models

| Role | Model | Note |
|---|---|---|
| Base (has template) | `Qwen/Qwen3-0.6B-Base` | primary tokenizer for the exercises |
| Base (no template) | `allenai/OLMo-2-0425-1B` | template donor: `allenai/OLMo-2-0425-1B-SFT` |
| Instruct (small) | `HuggingFaceTB/SmolLM2-360M-Instruct` | answer-and-stop behavior |

## 5. Dataset

`HuggingFaceH4/no_robots`, split `train`, **first 50 rows** (CPU-friendly; the
conversations are human-written 1–3 turn dialogs). No GPU anywhere in this lab.

## 6. Components (in `starter.py` — all stubs, you implement)

`inspect_tokenizer` · `tokenize_plain` · `tokenize_chat` · `diff_tokenizations` ·
`build_prompt_masked_labels` · `make_triples` · `decode_unmasked_only` ·
`collate_right_pad` · `pooling_corruption_demo` · `generation_comparison` ·
`length_stats` · `load_no_robots_sample`.

Contract reminders while implementing:

- The mask boundary is *your* arithmetic to write. Compare against the answer key
  **only after** your version passes your own decode-unmasked check (§8).
- One correct way to find the boundary: render the conversation *without* the final
  assistant message (with generation prompt) and the full conversation, then align.
  Think about whether the assistant turn's header tokens belong in the loss.
- `IGNORE_INDEX = -100` is the only label value that `cross_entropy` drops.

## 7. Experiments

1. **Decode-unmasked ×10:** for 10 no_robots samples, decode only unmasked positions;
   confirm you recover *exactly* the final assistant turn (answer + stop token).
2. **Length distributions:** token-length distributions of raw vs templated renders
   for base vs instruct tokenizers (templates add header/turn/EOS tokens).

## 8. Questions to answer in your notes

1. Why does OLMo base need a *template donor*, and what exactly does the donor supply?
2. What breaks at inference if EOS is **not** supervised during SFT?
3. When does pad *side* matter (training vs batched inference)? What corrupts if
   `attention_mask` is ignored during pooling?

## 9. Debugging challenges

- Your unmasked decode includes EOS **and** padding — why? (Hint: what did you set
  labels to at padded positions, and what does `decode` do with pad tokens?)
- Your mask includes the assistant turn's *header* tokens (e.g. `<|im_start|>assistant`)
  — is that correct? Argue both sides; check what the answer key does.
- A batch you right-padded produces different outputs than the same examples run
  individually — where did padding leak into positions that "should not matter"?

## 10. Completion criteria ("done when")

- Any conversation printed as `(token, id, label)` **triples** via
  `starter.make_triples`, with **every row explained** in a markdown cell.
- `decode_unmasked_only` recovers the final assistant turn on the 10-sample check.
- The 6-prompt base-vs-instruct panel is saved with one-line commentary per prompt.
- `cd labs/00_data_chat_templates && python3 -m pytest tests/ -q` passes locally.

## 11. Compare against (answer key — open only after your version works)

`rlhf-book/code/instruction_tuning/utils.py` **lines 153–244**: `_encode_batch`
(template render → prompt-length alignment → `-100` mask) and `_collate`
(right-pad `input_ids` / `attention_mask` / `labels`).

Diff your mask boundary against theirs; explain any off-by-one or header-token
difference you find in writing.
