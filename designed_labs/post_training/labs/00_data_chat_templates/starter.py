"""Lab 00 — Data & chat templates (starter).

Stubs only: signatures, contract docstrings, TODO markers, and
``raise NotImplementedError``. Implement the bodies yourself in the notebook
first, then paste back here. This is an inspection lab — nothing here trains.

NO-SOLUTIONS RULE reminders while you work:
- The mask boundary is *your* arithmetic: do NOT paste the answer key's
  alignment logic from rlhf-book/code/instruction_tuning/utils.py:153-244
  until your own decode-unmasked check passes (then diff and explain).
- No chat-template branching in this file: every template render goes through
  the tokenizer object you pass in.
- Every function must keep raising NotImplementedError until YOU replace it.

Answer key (open ONLY after your version passes your own checks):
    rlhf-book/code/instruction_tuning/utils.py lines 153-244
    (_encode_batch: template render -> boundary alignment -> masked labels;
     _collate: right-pad ids/mask/labels)

Import contract: this module imports cleanly WITHOUT torch installed
(numpy-only environments are fine). Tokenizers/datasets are always passed in
or loaded lazily inside function bodies — no network at import time.

Local, CPU-only test run:
    python3 -m pytest tests/ -q
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100  # the only label value torch cross_entropy drops

# Fixed 6-prompt pool for the base-continuation vs instruct answer-and-stop demo.
# Do not change these without re-reading the README completion criteria.
FIXED_PANEL_PROMPTS: List[str] = [
    "What is the capital of France?",
    "Explain recursion in one sentence.",
    "Write a two-line poem about the sea.",
    "Summarize why the sky is blue.",
    "Convert 72 Fahrenheit to Celsius.",
    "Name three sorting algorithms.",
]


# ---------------------------------------------------------------------------
# A. Tokenizer inspection (assignment item 1)
# ---------------------------------------------------------------------------


def inspect_tokenizer(tokenizer: Any) -> Dict[str, Any]:
    """Render the tokenizer's special-token surface.

    Args:
        tokenizer: any HF tokenizer (base OR template-donor chat tokenizer).

    Returns:
        Dict with (at least):
          - "special_tokens_map": {token_string: token_id} for every special token
          - "pad_token": the pad token string (None if unset — explain why that matters)
          - "pad_token_id": int or None
          - "padding_side": "left"|"right"
          - "has_chat_template": bool
        TODO(00.a): implement and eyeball-diff the three lab tokenizers.
    """
    raise NotImplementedError(
        "TODO(00.a): build the special-token/pad/padding-side report for `tokenizer`."
    )


def tokenize_plain(tokenizer: Any, text: str) -> List[int]:
    """Tokenize raw text WITHOUT any chat template.

    Args:
        tokenizer: HF tokenizer.
        text: raw conversation text (you choose the join formatting — record it!).

    Returns:
        List[int] of token ids. TODO(00.a): implement; keep special-token
        addition explicit so the raw-vs-templated diff is interpretable.
    """
    raise NotImplementedError("TODO(00.a): plain-text tokenization, no chat template.")


def tokenize_chat(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> List[int]:
    """Render `messages` through the tokenizer's chat template and tokenize.

    Args:
        tokenizer: HF tokenizer whose `chat_template` is set (use the OLMo -SFT
            donor when inspecting OLMo base).
        messages: [{"role": ..., "content": ...}, ...] in conversation order.
        add_generation_prompt: if True, end the render at the assistant-turn
            header (prompt-side render; you need this for build_prompt_masked_labels).

    Returns:
        List[int] of token ids (truncation is NOT applied here).
        TODO(00.a): implement; the template is the tokenizer's job, not yours.
    """
    raise NotImplementedError("TODO(00.a): chat-template render -> token ids.")


def diff_tokenizations(
    plain_ids: Sequence[int],
    chat_ids: Sequence[int],
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """Summarize how the chat template changed tokenization.

    Args:
        plain_ids: ids from tokenize_plain.
        chat_ids: ids from tokenize_chat for the same conversation.
        tokenizer: optional, used only to decode for human-readable output.

    Returns:
        Dict with (at least):
          - "n_plain", "n_chat": lengths
          - "n_extra": chat length minus plain length
          - "extra_prefix_tokens": decoded ids that chat adds BEFORE the text
          - "extra_suffix_tokens": decoded ids chat adds AFTER the text
        TODO(00.a): implement; this is descriptive bookkeeping, no masking here.
    """
    raise NotImplementedError("TODO(00.a): quantify plain-vs-template token diff.")


# ---------------------------------------------------------------------------
# B. Prompt-masked labels (assignment item 2 — THE core exercise)
# ---------------------------------------------------------------------------


def build_prompt_masked_labels(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    max_length: int = 512,
) -> Tuple[List[int], List[int]]:
    """Full-sequence ids + labels masked to the FINAL assistant turn only.

    Contract:
        - input_ids: token ids of the FULL templated conversation (with the
          final assistant turn present), truncated to at most `max_length`.
        - labels: same length as input_ids; IGNORE_INDEX at every position the
          loss must NOT train on (everything before the final assistant turn's
          supervised content, plus padded positions later), and equal to the
          corresponding input id at supervised positions.

    Semantics you must decide (and defend in the notebook):
        - Is the assistant turn's HEADER supervised? (Check the answer key.)
        - Is the EOS after the answer supervised? (Argue why it must be.)

    Args:
        tokenizer: chat-capable tokenizer.
        messages: must end with an {"role": "assistant", ...} turn.
        max_length: hard truncation ceiling on the full render.

    Returns:
        (input_ids, labels) — equal-length lists of ints.
        TODO(00.b): implement; the boundary arithmetic is YOUR work.
    """
    raise NotImplementedError(
        "TODO(00.b): render with/without the final assistant turn, align, mask."
    )


def make_triples(
    tokenizer: Any,
    input_ids: Sequence[int],
    labels: Sequence[int],
) -> List[Tuple[str, int, int]]:
    """One row per token: (decoded_token, token_id, label) — the 'done when' view.

    Args:
        tokenizer: for decoding each id individually (use special-token-safe
            decoding so headers/EOS print, not vanish).
        input_ids: full-sequence ids.
        labels: mask from build_prompt_masked_labels (padded rows may be -100).

    Returns:
        List of (token_str, token_id, label) triples, one per position.
        TODO(00.b): implement; print these and explain every row in markdown.
    """
    raise NotImplementedError("TODO(00.b): per-position (token, id, label) table.")


def decode_unmasked_only(
    tokenizer: Any,
    input_ids: Sequence[int],
    labels: Sequence[int],
) -> str:
    """Decode ONLY the positions where labels != IGNORE_INDEX.

    This is your self-verification hook: for a well-built mask it must return
    exactly the final assistant turn (answer text + its stop token), nothing
    from the system/user turns, nothing padded.

    Returns:
        Decoded string of the unmasked span. TODO(00.b): implement.
    """
    raise NotImplementedError("TODO(00.b): select unmasked ids, decode, return.")


# ---------------------------------------------------------------------------
# C. Batching & the padding corruption demo (assignment item 4)
# ---------------------------------------------------------------------------


def collate_right_pad(
    examples: Sequence[Tuple[Sequence[int], Sequence[int]]],
    pad_token_id: int,
) -> Dict[str, List[List[int]]]:
    """Right-pad a list of (input_ids, labels) pairs into rectangular arrays.

    Args:
        examples: sequence of (input_ids, labels) with per-example lengths.
        pad_token_id: id used to fill input_ids tails.

    Returns:
        Dict with three equal-shape (B, T) int lists:
          - "input_ids": original ids + pad_token_id tails
          - "attention_mask": 1 for real tokens, 0 at pad positions
          - "labels": original labels, IGNORE_INDEX at pad positions
        TODO(00.c): implement. Also record what you'd change for left padding.
    """
    raise NotImplementedError("TODO(00.c): right-pad to the batch max length.")


def pooling_corruption_demo(
    padded_batch: Dict[str, List[List[int]]],
) -> Dict[str, Any]:
    """Demonstrate what breaks when pad positions are pooled as if real.

    Args:
        padded_batch: output of collate_right_pad.

    Returns:
        Dict with (at least):
          - "corrupt_positions": [row, col] entries where naive mean/max pooling
            over the full padded length differs from masking pad positions out
          - "explanation": one-paragraph written diagnosis (edit after running)
        TODO(00.c): implement; this feeds README §9's third debugging challenge.
    """
    raise NotImplementedError("TODO(00.c): locate pad-corrupted pooling positions.")


# ---------------------------------------------------------------------------
# D. Base vs instruct generation panel (assignment item 3)
# ---------------------------------------------------------------------------


def generation_comparison(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    mode: str,
    generation_kwargs: Dict[str, Any] | None = None,
) -> List[str]:
    """Generate from `prompts` in one of two modes for the fixed panel.

    Args:
        model: causal LM on CPU (small models only — this lab is Tier C).
        tokenizer: the model's tokenizer.
        prompts: use starter.FIXED_PANEL_PROMPTS (len 6).
        mode: "base" — feed raw prompt text, plain continuation, no template;
              "instruct" — wrap each prompt in the chat template with a
              generation prompt, generate, then STOP at the EOS turn header.
        generation_kwargs: passed through to model.generate (greedy default).

    Returns:
        List[str] of length len(prompts): decoded continuations/answers.
        TODO(00.d): implement; save the panel + one-line commentary per prompt.
    """
    raise NotImplementedError(
        "TODO(00.d): base raw-continuation vs instruct template+stop generation."
    )


def length_stats(lengths: Sequence[int]) -> Dict[str, float]:
    """Descriptive stats for the base-vs-instruct length-distribution experiment.

    Returns:
        Dict with keys "n", "min", "max", "mean", "median".
        TODO(00.d): implement; compare raw vs templated and base vs instruct.
    """
    raise NotImplementedError("TODO(00.d): n/min/max/mean/median of `lengths`.")


# ---------------------------------------------------------------------------
# E. Data access (network/datasets lazily — never at import time)
# ---------------------------------------------------------------------------


def load_no_robots_sample(n: int = 50) -> List[List[Dict[str, str]]]:
    """Load the first `n` conversations of HuggingFaceH4/no_robots (train).

    Requires network + the `datasets` library — call from the notebook, never
    at module import. Return conversations as lists of {"role", "content"}.
    TODO(00.e): implement (filter to conversations ending in an assistant turn
    for the masking exercise, and say why you filtered).
    """
    raise NotImplementedError("TODO(00.e): datasets.load_dataset slice, n rows.")
