"""Lab 03 — Bradley-Terry reward model: student starter code.

You build a preference reward model FROM SCRATCH (no TRL) and train it with a
hand-written Bradley-Terry loss:

    backbone (Qwen/Qwen3-0.6B-Base) -> Linear(hidden_size, 1) head
    pooled at the LAST NON-PAD TOKEN -> scalar reward per sequence

NO-SOLUTIONS RULE: every function below is a signature + docstring contract
and raises NotImplementedError. Implement them one at a time against the
shape contracts in the docstrings. Do not read the answer key until your own
version trains:

    rlhf-book/code/reward_models/base.py                  (BaseRewardModel, pooling utilities)
    rlhf-book/code/reward_models/train_preference_rm.py   (BT loss, training loop)

Environment note: this module imports cleanly WITHOUT torch installed (the
torch import is guarded), so the local structural tests can run on a bare
Python 3.9 interpreter. On a machine with torch, the extra pieces activate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # torch stays optional at import time so the scaffold is inspectable anywhere
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - hit on machines without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


# =============================================================================
# Config (provided — mirrors configs/03_bradley_terry_rm.yaml)
# =============================================================================


@dataclass
class RewardModelConfig:
    """Configuration for Lab 03. Kept in sync with configs/03_bradley_terry_rm.yaml."""

    model_id: str = "Qwen/Qwen3-0.6B-Base"
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned"
    # Lab 02 artifact: your cleaned subset (+ near-tie export) lives locally
    cleaned_subset_path: Optional[str] = None       # e.g. labs/02_.../outputs/cleaned_subset.jsonl
    near_ties_path: Optional[str] = None            # Lab 02 near-tie pairs you scored by hand
    subset_sizes_to_sweep: List[int] = field(default_factory=lambda: [1000, 2000, 5000])
    val_fraction: float = 0.05
    max_length: int = 1024
    batch_size: int = 4
    grad_accum_steps: int = 8
    epochs: int = 1
    learning_rate: float = 5e-6                     # sweep {1e-6, 5e-6, 2e-5}
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    freeze_backbone: bool = False                   # experiment: True vs False
    use_amp: bool = True
    eval_every_steps: int = 50
    seed: int = 42
    # Sorted ascending edges for calibration analysis over margins (units of
    # the score difference); bucket i covers edges[i] up to edges[i+1] exclusive.
    margin_bucket_edges: List[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 4.0])
    output_dir: str = "runs/lab03_bt_rm"


# =============================================================================
# Part 1 — Data: paired preference batching
# =============================================================================


def tokenize_pair(
    example: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, List[int]]:
    """Tokenize ONE preference pair into token-id lists.

    TODO(Lab03): implement.

    Contract
    --------
    example
        Mapping describing one pair. It minimally carries a shared prompt and
        two candidate completions; adapt the exact field names to YOUR Lab 02
        cleaned-subset schema and write down the mapping in your notes.
    tokenizer
        An HF fast tokenizer. If you use its chat template, apply it
        consistently so both completions share identical prompt tokens.
    max_length
        Hard cap applied to EACH id list (right truncation).

    Returns a dict with EXACTLY these keys:
        input_ids_chosen    List[int], len <= max_length
        input_ids_rejected  List[int], len <= max_length

    No padding here: sequences may differ in length within a batch. Padding
    belongs in ``paired_collate_fn``.
    """
    raise NotImplementedError("TODO(Lab03): implement tokenize_pair")


def paired_collate_fn(
    batch: Sequence[Mapping[str, Sequence[int]]],
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """Right-pad a list of tokenized pairs into rectangular batch tensors.

    TODO(Lab03): implement.

    Contract
    --------
    batch : sequence produced by ``tokenize_pair``
    pad_token_id : value used to extend short id sequences

    Returns a dict with keys (dtype long everywhere):
        input_ids_chosen      [B, L]   padded with pad_token_id
        attention_mask_chosen [B, L]   1 on real tokens, 0 on pad positions
        input_ids_rejected    [B, L]
        attention_mask_rejected[B, L]

    where L = max sequence length in THIS batch and B = len(batch).

    Convention adopted for this lab: RIGHT padding. State explicitly in your
    notebook what would break under LEFT padding and why pooling logic must
    match the pad side (this was Lab 00's pad-side lesson).
    """
    raise NotImplementedError("TODO(Lab03): implement paired_collate_fn")


def build_dataloader(examples: Sequence[Any], tokenizer: Any, config: RewardModelConfig) -> Any:
    """Wrap tokenized pairs in a DataLoader using ``paired_collate_fn``.

    TODO(Lab03): implement.

    Contract
    --------
    Deterministic shuffling seeded from ``config.seed``; batch size and other
    loop knobs come from ``config``. Returns a torch DataLoader yielding dicts
    shaped like ``paired_collate_fn`` output.
    """
    raise NotImplementedError("TODO(Lab03): implement build_dataloader")


if _HAS_TORCH:

    # =========================================================================
    # Part 2 — Model: backbone + Linear head + last-non-pad pooling
    # =========================================================================

    class BradleyTerryRewardModel(nn.Module):
        """Scalar reward model: transformer backbone plus Linear(hidden, 1) head.

        TODO(Lab03): implement __init__, pooling wiring and forward.

        Architecture contract
        ---------------------
        __init__(model_id, device, freeze_backbone=False) must set up:
          self.backbone — the causal LM loaded FP32 with caching disabled
              (see rlhf-book/code/reward_models/base.py BaseRewardModel for the
              loading idiom; expose final hidden states of shape [B, T, H]).
          self.head — torch.nn.Linear(hidden_size, 1), bias allowed either way;
              document your choice.
          freeze_backbone=True must mark every backbone parameter
          non-trainable while keeping self.head trainable. Report trainable
          param counts in your notebook for the freeze-vs-full experiment.

        forward(input_ids, attention_mask) -> rewards
            input_ids      LongTensor [B, T]
            attention_mask LongTensor [B, T]
            rewards        FloatTensor [B]  (one scalar per sequence)

        Internals you must wire (in order):
            hidden states [B, T, H] -> pool_last_non_pad_token [B, H]
            -> self.head [B, 1] -> squeeze to [B]
        """

        def __init__(
            self,
            model_id: str,
            device: str = "cpu",
            freeze_backbone: bool = False,
        ):
            raise NotImplementedError("TODO(Lab03): implement __init__")

        def forward(self, input_ids, attention_mask):
            """Return scalar rewards, shape [B] — see class docstring for shapes."""
            raise NotImplementedError("TODO(Lab03): implement forward")


def pool_last_non_pad_token(last_hidden_state, attention_mask):
    """Gather, per row, the hidden vector at the LAST NON-PAD position.

    TODO(Lab03): implement. This is the single most bug-prone spot of the lab
    (accuracy stuck at chance usually lives here).

    Args
    ----
    last_hidden_state : FloatTensor [B, T, H]  final-layer states
    attention_mask    : LongTensor  [B, T]     1 real token, 0 padding

    Returns
    -------
    FloatTensor [B, H] — one pooled vector per sequence.

    Hints (not solutions):
        - derive the last valid index per row from attention_mask alone;
        - no Python loop over the batch required;
        - say in your notebook what this function silently returns if given an
          all-zeros mask row.
    """
    raise NotImplementedError("TODO(Lab03): implement pool_last_non_pad_token")


# =============================================================================
# Part 3 — Loss: hand-written Bradley-Terry objective
# =============================================================================


def bradley_terry_loss(scores_chosen, scores_rejected):
    """Hand-written Bradley-Terry negative log-likelihood. THE core task.

    TODO(Lab03): implement — no solution shortcuts allowed:
        - do NOT reach for the library's ready-made log-sigmoid helper
          (F.logsig*id or similar); write a numerically stable version yourself;
        - derive P(chosen preferred | s_chosen, s_rejected) under the BT model
          in your notebook before coding it (Ch. 5 BT section);
        - explain WHY your implementation stays finite for large |scores| while
          the naive formulation does not.

    Args
    ----
    scores_chosen   : FloatTensor [B]
    scores_rejected : FloatTensor [B]

    Returns
    -------
    Scalar FloatTensor [()] — mean over the batch of the per-pair NLL.

    Failure modes to know how to diagnose (see README Debugging): a constant
    offset added to ALL scores leaves this loss unchanged — why?
    """
    raise NotImplementedError("TODO(Lab03): implement bradley_terry_loss")


# =============================================================================
# Part 4 — Metrics: accuracy, margins, calibration buckets, histograms
# =============================================================================


def pairwise_accuracy(scores_chosen, scores_rejected):
    """Fraction of the batch scored correctly: chosen ranked above rejected.

    TODO(Lab03): implement.

    Args: scores_chosen [B], scores_rejected [B].
    Returns: float in [0, 1]. Decide and DOCUMENT the tie convention (equal
    scores count as correct or incorrect?), then keep it fixed across runs.
    """
    raise NotImplementedError("TODO(Lab03): implement pairwise_accuracy")


def mean_margin(scores_chosen, scores_rejected):
    """Mean margin across the batch.

    TODO(Lab03): implement.

    Margin definition (per Ch. 5): the signed difference between the chosen
    and rejected scores for one pair. Compute it for every pair and average.

    Args: scores_chosen [B], scores_rejected [B].
    Returns: float. Note in your notebook that accuracy is scale-free while
    margins are not — what does a shrinking-margin/rising-accuracy regime
    predict about downstream best-of-N behavior (Lab 09)?
    """
    raise NotImplementedError("TODO(Lab03): implement mean_margin")


def accuracy_by_margin_bucket(scores_chosen, scores_rejected, bucket_edges):
    """Margin-bucketed accuracy (calibration analysis).

    TODO(Lab03): implement.

    Partition pairs by margin into consecutive buckets bounded by the sorted
    ``bucket_edges`` (bucket i covers [edges[i], edges[i+1]) ), plus an
    open-ended top bucket holding everything at or above the largest edge.
    Margins can be negative; include a bottom bucket accordingly and document
    your labeling scheme, e.g. "<0.0", "[0.0,1.0)", ">=4.0".

    Args
    ----
    scores_chosen   : FloatTensor [B]
    scores_rejected : FloatTensor [B]
    bucket_edges    : ascending sequence of floats (never empty)

    Returns
    -------
    (labels, accuracies, counts) — parallel lists, one entry per bucket:
        labels     List[str]        your documented interval labels
        accuracies List[float]      pairwise_accuracy restricted to the bucket
        counts     List[int]        number of pairs in the bucket (sum == B)
    """
    raise NotImplementedError("TODO(Lab03): implement accuracy_by_margin_bucket")


def compute_eval_metrics(model, batch, bucket_edges) -> Dict[str, Any]:
    """Forward one collated batch through ``model`` and compute all metrics.

    TODO(Lab03): implement.

    batch : dict from ``paired_collate_fn`` (input_ids_* / attention_mask_*)
    Returns a flat dict:
        accuracy            float           pairwise_accuracy
        mean_margin         float           mean_margin
        reward_chosen_mean  float           histogram material
        reward_chosen_std   float
        reward_rejected_mean float
        reward_rejected_std float
        buckets             {label: {"acc": float, "count": int}}

    These are appended to the JSONL metrics log verbatim (names and casing).
    """
    raise NotImplementedError("TODO(Lab03): implement compute_eval_metrics")


def evaluate(model, loader, device: str = "cpu") -> Dict[str, Any]:
    """Aggregate ``compute_eval_metrics`` over an evaluation loader.

    TODO(Lab03): implement.

    No gradients may flow (guard the whole pass). Merge counts across batches
    BEFORE dividing so bucket accuracies stay correct under unequal batches,
    and return the same key structure as ``compute_eval_metrics`` aggregated
    over the whole dataset.
    """
    raise NotImplementedError("TODO(Lab03): implement evaluate")


# =============================================================================
# Part 5 — Orchestration
# =============================================================================


def run_training(config: RewardModelConfig) -> Any:
    """End-to-end training entry point driven entirely by ``config``.

    TODO(Lab03): implement.

    Required behavior (check off in README as you go):
      1. load tokenizer + backbone (seeded from config.seed),
      2. build train/val loaders from YOUR cleaned Lab 02 subset,
      3. optimize ``bradley_terry_loss`` with AdamW on trainable params only,
         honoring grad accumulation, clip-grad-norm, linear warmup, AMP flag,
      4. every eval_every_steps append JSONL metrics (accuracy, mean_margin,
         histograms, buckets) under config.output_dir — notebooks plot from
         these artifacts, not from wandb,
      5. save a checkpoint per epoch,
      6. finally run ``score_lab02_near_ties`` and store its table.

    Returns whatever your checkpoint/artifact layout needs downstream; the
    acceptance criterion is a narrated end-to-end shape trace in the notebook.
    """
    raise NotImplementedError("TODO(Lab03): implement run_training")


def score_lab02_near_ties(model, near_ties_path: str, tokenizer: Any, device: str = "cpu"):
    """Score the Lab 02 near-tie export and emit the human-agreement table.

    TODO(Lab03): implement.

    Reads the JSONL you exported in Lab 02 (pairs you judged near-ties by
    hand), scores each side with ``model``, and builds a comparison table
    mapping: your human verdict vs the RM's verdict vs the RM's margin — the
    raw material for the disagreement case-study deliverable.

    Returns a list of row dicts, at minimum:
        {"pair_id": ..., "human_verdict": ..., "rm_verdict": ..., "rm_margin": float}
    """
    raise NotImplementedError("TODO(Lab03): implement score_lab02_near_ties")
