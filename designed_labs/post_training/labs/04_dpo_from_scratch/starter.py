"""Lab 04 — DPO From Scratch (starter code).

Every function body below is YOURS to implement. Only signatures,
docstrings, shape notes, and TODO markers are provided. Read Chapter 8
(https://rlhfbook.com/c/08-direct-alignment.html) and derive each piece on
paper before typing it.

Ground rules for this lab:
  - No solution code lives here, and it must not be smuggled in: the objective
    must NOT be a one-line call to any prepackaged log-sigmoid-of-scaled-logits
    loss helper — implement it from the derivation you did for the README.
  - The four sequence log-probs fed to ``dpo_loss`` come from four separate
    forward passes (chosen/rejected x policy/reference). Keep that plumbing
    explicit; don't fuse them.
  - The repo under rlhf-book/code/direct_alignment/ is the ANSWER KEY. Open
    loss.py::DPOLoss and data.py only after your version trains.

This module imports cleanly on a machine without torch (bodies raise
NotImplementedError before any tensor work happens).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# torch is intentionally imported lazily/guarded so this file is inspectable
# (and testable) on CPU-only machines without torch installed.
try:  # pragma: no cover - exercised implicitly by tests
    import torch
    import torch.nn.functional as F  # noqa: F401  (you will need it)

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Part 1 — Preference data plumbing
# ---------------------------------------------------------------------------


@dataclass
class PreferenceBatch:
    """A paired batch of preference examples, padded to a common length.

    Fields (all torch.LongTensor unless noted):
        chosen_input_ids:       (batch, seq_len)
        chosen_attention_mask:  (batch, seq_len)      1 = real token
        chosen_response_mask:   (batch, seq_len)      1 = supervised response position
        rejected_input_ids:     (batch, seq_len)
        rejected_attention_mask:(batch, seq_len)
        rejected_response_mask: (batch, seq_len)

    Contract on the response masks:
        * 0 at every prompt position AND every padding position;
        * 1 only at positions belonging to the assistant response (+ EOS).
      If those two claims are not true of your batches, everything downstream
      is wrong in a way that still trains "fine". Test yourself accordingly
      (see tests/test_04_dpo_from_scratch.py).
    """

    chosen_input_ids: Any
    chosen_attention_mask: Any
    chosen_response_mask: Any
    rejected_input_ids: Any
    rejected_attention_mask: Any
    rejected_response_mask: Any


def tokenize_preference_pair(
    prompt: str,
    chosen_response: str,
    rejected_response: str,
    tokenizer: Any,
    max_length: int = 2048,
) -> Dict[str, Any]:
    """Tokenize one preference pair into aligned chosen/rejected tensors.

    TODO:
      - Render prompt+response with the tokenizer's chat template (Lab 00).
      - Tokenize prompt-only separately to locate where the response begins.
      - Right-pad to max_length with the pad token id; attention mask marks
        real tokens; response mask marks ONLY response tokens (prompt + pad excluded).
      - Decide and document a truncation policy when the pair exceeds
        max_length (hint: which side can you afford to cut?).

    Returns: dict of six tensors matching the fields of PreferenceBatch
             (batch dimension omitted — single example).
    """
    raise NotImplementedError("TODO(lab04): tokenize a preference pair")


def collate_preference_batch(examples: List[Dict[str, Any]]) -> PreferenceBatch:
    """Stack per-example dicts from tokenize_preference_pair into a batch.

    TODO:
      - Verify all examples share seq_len after padding (pad to the longest
        in-batch if you support dynamic lengths).
      - Return a single PreferenceBatch.
    """
    raise NotImplementedError("TODO(lab04): collate paired examples")


def make_paired_dataloader(dataset: Any, tokenizer: Any, config: Dict[str, Any]) -> Any:
    """Wrap a normalized preference dataset (columns: prompt/chosen/rejected)
    in a DataLoader yielding PreferenceBatch objects.

    TODO: shuffle with a seeded generator, pin_memory off for CPU debugging,
          batch_size from config["batch_size"].
    """
    raise NotImplementedError("TODO(lab04): build the paired DataLoader")


# ---------------------------------------------------------------------------
# Part 2 — Sequence log-probs
# ---------------------------------------------------------------------------


def sequence_logprob(logits: Any, input_ids: Any, response_mask: Any) -> Any:
    """Total log-probability of the response tokens under the model.

    Args:
        logits:       (batch, seq_len, vocab_size) — raw logits from one forward pass.
        input_ids:    (batch, seq_len)
        response_mask:(batch, seq_len) — 1 at supervised positions, else 0.

    Returns:
        (batch,) tensor: sum over RESPONSE positions of log P(token_i | prefix),
        computed once per sequence.

    TODO:
      - Handle the autoregressive shift yourself: position t's probability
        comes from logits at t-1. Get the alignment wrong and the loss trains
        on garbage that looks normal.
      - Restrict contribution to positions where response_mask == 1 AFTER the
        shift (think carefully about which mask positions survive shifting).
      - VERIFY: on a small hand-built batch, compare against an equivalent
        F.cross_entropy(reduction="none") computation. They must agree.
    """
    raise NotImplementedError("TODO(lab04): sum masked per-token log-probs")


# ---------------------------------------------------------------------------
# Part 3 — The DPO objective
# ---------------------------------------------------------------------------


def dpo_loss(policy_c: Any, policy_r: Any, ref_c: Any, ref_r: Any, beta: float) -> Any:
    """Hand-written DPO loss (Rafailov et al., 2023).

    Args:
        policy_c: (batch,) sequence log-probs of CHOSEN responses under the policy.
        policy_r: (batch,) sequence log-probs of REJECTED responses under the policy.
        ref_c:    (batch,) sequence log-probs of CHOSEN responses under the frozen reference.
        ref_r:    (batch,) sequence log-probs of REJECTED responses under the frozen reference.
        beta:     temperature controlling how strongly the objective anchors the
                  policy to the reference.

    Returns:
        Scalar tensor loss, averaged over the batch.

    IMPORTANT — the whole point of this lab:
      Derive the objective from the Bradley-Terry reparameterization in Ch. 8
      BEFORE writing this function. Do NOT reach for a prepackaged
      log-sigmoid-of-scaled-logits helper or copy the closed form from memory
      of someone else's code; if your implementation contains such a one-line
      call instead of something you built from the derivation, redo it.
      Show your derivation as comments.
    """
    raise NotImplementedError("TODO(lab04): implement DPO from your derivation")


# ---------------------------------------------------------------------------
# Part 4 — Metrics / observability
# ---------------------------------------------------------------------------


def compute_metrics(policy_c: Any, policy_r: Any, ref_c: Any, ref_r: Any, beta: float) -> Dict[str, float]:
    """Per-step training diagnostics appended to metrics.jsonl by train.py.

    TODO (implement AFTER your loss passes verification):
      - implicit reward of chosen and rejected sequences (the quantities DPO
        actually moves; note they should be detached — logging must not leak
        gradient);
      - margin between the two implicit rewards;
      - pairwise accuracy: fraction of pairs currently ranked correctly;
      - mean response length on both sides (length bias shows up fast).
    Returns: flat dict of floats.
    """
    raise NotImplementedError("TODO(lab04): training diagnostics")


@torch.no_grad() if HAS_TORCH else (lambda fn: fn)  # noqa: B008  (decorator applies only with torch)
def forward_reference_model(ref_model: Any, batch: PreferenceBatch) -> Sequence[Any]:
    """Frozen-reference forward passes used alongside the two policy passes.

    Args:
        ref_model: the frozen reference (policy_init) language model.
        batch:     a PreferenceBatch.

    Returns:
        (ref_chosen_logps, ref_rejected_logps), each shaped (batch,), obtained
        via your own sequence_logprob.

    TODO:
      - Confirm no gradients flow out of this function (it decorates itself
        with no_grad when torch is present).
      - Together with the policy's two forwards, this makes FOUR forward
        passes per step — state in a comment why all four are needed.
    """
    raise NotImplementedError("TODO(lab04): reference-model forwards")
