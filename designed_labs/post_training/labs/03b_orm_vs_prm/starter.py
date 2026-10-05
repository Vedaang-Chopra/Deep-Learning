"""Lab 03b - ORM vs PRM (scaffold).

Two supervision signals head-to-head on the SAME kind of artifact (model
solutions): an outcome-reward model (ORM = binary correctness head over GSM8K
rollouts labeled by final-answer match) versus a process-reward model
(PRM = per-step {-1, 0, +1} classifier over PRM800K-style step annotations).
The payoff deliverable is a disagreement case-study table on >= 20 held-out
solutions scored by BOTH heads, isolating the canonical failure mode:
correct answer, wrong reasoning.

Prerequisites: Lab 03 (Bradley-Terry RM) required; Lecture 2 part 2 + Ch. 5
(https://rlhfbook.com/c/05-reward-models.html).

NO-SOLUTIONS SCAFFOLD: this file defines contracts only. Every function whose
body is a mechanism you are meant to implement (dataset shaping, both training
loops, both losses, dual scoring, disagreement selection) is marked with TODO
and raises NotImplementedError. Implemented helpers below are pure plumbing:
record-shape validation. Do not peek at the answer key
(rlhf-book/code/reward_models/train_orm.py / train_prm.py) until your version
trains.

Runs torch-free at import time: module-level dependencies are numpy + PyYAML +
stdlib, so the tests collect on a CPU-only machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Types & constants (given)
# ---------------------------------------------------------------------------

Record = Dict[str, Any]

#: One normalized GSM8K rollout record has exactly these top-level keys.
#: ``label`` is one of ORM_LABELS; ``rollout_idx`` in [0, rollouts_per_prompt).
ROLLOUT_RECORD_KEYS = (
    "prompt_id",
    "question",
    "gold_answer",
    "solution",
    "predicted_answer",
    "label",
    "rollout_idx",
)

#: Valid outcome labels for a GSM8K rollout.
ORM_LABELS = ("correct", "incorrect")

#: One PRM800K slice record has exactly these top-level keys.
#: ``step_labels[i]`` scores ``steps[i]`` and must be one of PRM_LABELS.
PRM_RECORD_KEYS = ("problem_id", "problem", "steps", "step_labels")

#: Step-quality label space from PRM800K.
PRM_LABELS = (-1, 0, 1)

#: Disagreement-table categories for the case-study deliverable.
#: The money category: right answer flagged for poisoned reasoning.
DISAGREEMENT_CATEGORIES = (
    "agree_correct",
    "agree_incorrect",
    "orm_right_prm_flags_reasoning",
    "prm_right_orm_wrong",
)

#: One case-study row has exactly these keys (see README "Deliverable").
CASE_ROW_KEYS = (
    "solution_id",
    "orm_logit",
    "orm_verdict",
    "prm_min_step_label",
    "prm_flagged_step_idx",
    "category",
    "ground_truth_answer",
    "predicted_answer",
)


def paired_none() -> None:
    """Placeholder for symmetric optional imports (torch) kept out of module scope.

    torch is imported INSIDE stub bodies that need it once you implement them,
    via ``pytest.importorskip``-style guards or plain import in the function.
    Module stays importable on CPU-only machines either way.
    """
    return None


# ---------------------------------------------------------------------------
# Record-shape validation (implemented plumbing)
# ---------------------------------------------------------------------------


def validate_rollout_records(records: Sequence[Record]) -> None:
    """Raise ValueError if rollout records violate the schema contract.

    Checks (all structural, no algorithm content):
    * every record is a dict with exactly :data:`ROLLOUT_RECORD_KEYS`
    * ``label`` in :data:`ORM_LABELS`
    * ``rollout_idx`` int >= 0
    * non-empty strings for question/solution/gold/predicted answers
    """
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"record {i}: expected dict, got {type(rec).__name__}")
        if tuple(sorted(rec.keys())) != tuple(sorted(ROLLOUT_RECORD_KEYS)):
            raise ValueError(
                f"record {i}: keys {sorted(rec.keys())} != {sorted(ROLLOUT_RECORD_KEYS)}"
            )
        if rec["label"] not in ORM_LABELS:
            raise ValueError(f"record {i}: label {rec['label']!r} not in {ORM_LABELS}")
        idx = rec["rollout_idx"]
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise ValueError(f"record {i}: rollout_idx must be int >= 0")
        for k in ("question", "solution", "gold_answer", "predicted_answer"):
            if not isinstance(rec[k], str) or not rec[k].strip():
                raise ValueError(f"record {i}: {k!r} must be a non-empty string")


def validate_prm_records(records: Sequence[Record]) -> None:
    """Raise ValueError if PRM records violate the schema contract.

    Checks (all structural, no algorithm content):
    * every record is a dict with exactly :data:`PRM_RECORD_KEYS`
    * ``steps`` and ``step_labels`` are lists of equal positive length
    * every step label in :data:`PRM_LABELS`
    """
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"record {i}: expected dict, got {type(rec).__name__}")
        if tuple(sorted(rec.keys())) != tuple(sorted(PRM_RECORD_KEYS)):
            raise ValueError(
                f"record {i}: keys {sorted(rec.keys())} != {sorted(PRM_RECORD_KEYS)}"
            )
        steps, labels = rec["steps"], rec["step_labels"]
        if not isinstance(steps, list) or not isinstance(labels, list):
            raise ValueError(f"record {i}: steps/step_labels must be lists")
        if len(steps) == 0 or len(steps) != len(labels):
            raise ValueError(
                f"record {i}: need len(steps) == len(step_labels) > 0"
            )
        for j, lab in enumerate(labels):
            if lab not in PRM_LABELS:
                raise ValueError(
                    f"record {i} step {j}: label {lab!r} not in {PRM_LABELS}"
                )


# ---------------------------------------------------------------------------
# Config loading (given plumbing)
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the lab YAML into a plain dict (see configs/03b_orm_vs_prm.yaml)."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ORM side -- all student work below
# ---------------------------------------------------------------------------


@dataclass
class OrmPrmLabConfig:
    """Mirror of configs/03b_orm_vs_prm.yaml (kept in sync by contract).

    Shapes/comments note units; nothing here implements training logic.
    """

    model_id: str
    freeze_backbone: bool
    rollout_dataset_name: str
    prm_dataset_name: str
    rollouts_per_prompt_cap: Optional[int]
    max_rows_orm: Optional[int]
    val_fraction: float
    max_length_orm: int
    max_length_prm: int
    batch_size: int
    grad_accum_steps: int
    epochs: int
    learning_rate: float
    warmup_ratio: float
    max_grad_norm: float
    use_amp: bool
    eval_every_steps: int
    disagreement_pool_size: int
    min_cases_required: int
    orm_positive_threshold: float

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "OrmPrmLabConfig":
        """Build from the parsed YAML dict; nested sections flattened."""
        m, d, t, e = cfg["model"], cfg["data"], cfg["train"], cfg["eval"]
        return cls(
            model_id=m["model_id"],
            freeze_backbone=m["freeze_backbone"],
            rollout_dataset_name=d["rollout_dataset_name"],
            prm_dataset_name=d["prm_dataset_name"],
            rollouts_per_prompt_cap=d.get("rollouts_per_prompt_cap"),
            max_rows_orm=d.get("max_rows_orm"),
            val_fraction=d["val_fraction"],
            max_length_orm=d["max_length_orm"],
            max_length_prm=d["max_length_prm"],
            batch_size=t["batch_size"],
            grad_accum_steps=t["grad_accum_steps"],
            epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            warmup_ratio=t["warmup_ratio"],
            max_grad_norm=t["max_grad_norm"],
            use_amp=t["use_amp"],
            eval_every_steps=t["eval_every_steps"],
            disagreement_pool_size=e["disagreement_pool_size"],
            min_cases_required=e["min_cases_required"],
            orm_positive_threshold=e["orm_positive_threshold"],
        )


def load_rollout_dataset(cfg: OrmPrmLabConfig) -> List[Record]:
    """Load the raw GSM8K rollout dataset from the HF hub name in ``cfg``.

    Contract:
      - returns raw rows as dicts (NOT yet validated/shaped)
      - respects ``cfg.rollouts_per_prompt_cap`` (None = all 100/prompt)
      - network access happens here at run time -- keep out of tests

    TODO(student): implement loading (+ split handling).
    """
    raise NotImplementedError("Lab 03b: load_rollout_dataset")


def normalize_rollout_records(raw_rows: List[Dict[str, Any]]) -> List[Record]:
    """Map raw dataset fields onto :data:`ROLLOUT_RECORD_KEYS` records.

    Contract:
      - pure transformation of raw -> normalized records; call
        :func:`validate_rollout_records` before returning
      - derives ``predicted_answer`` with the SAME extraction rule your Lab 06
        verifier uses so ORM scores stay comparable to RLVR correctness

    TODO(student): implement field mapping + answer normalization.
    """
    raise NotImplementedError("Lab 03b: normalize_rollout_records")


def shape_orm_examples(records: List[Record], tokenizer: Any, cfg: OrmPrmLabConfig) -> List[Dict[str, Any]]:
    """Tokenize prompt+solution pairs into ORM training examples.

    Contract:
      - each example: {input_ids, attention_mask, label} with truncation to
        ``cfg.max_length_orm`` (right padding assumed)
      - prompt part unmasked conceptually -- masking here mirrors Lab 03's
        last-non-pad pooling, NOT SFT's label masking (this is classification)

    TODO(student): implement tokenization + shaping.
    """
    raise NotImplementedError("Lab 03b: shape_orm_examples")


def orm_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate ORM examples into right-padded tensors for one forward pass.

    Contract:
      - returns {input_ids [B,T], attention_mask [B,T], labels [B]}-style batch
      - pad token id comes from the tokenizer captured in a closure you write

    TODO(student): implement padding + stacking.
    """
    raise NotImplementedError("Lab 03b: orm_collate_fn")


@dataclass
class OrmBinaryHead:
    """Backbone + Linear(hidden_size, 1) pooled at last non-pad position -> logit.

    Mirrors rlhf-book reward_models/base.py pooling discipline (compare AFTER
    yours trains). Attributes declared for shape clarity only.
    """

    hidden_size: int = 896  # Qwen3-0.6B hidden size
    freeze_backbone: bool = False

    # TODO(student): hold backbone/head modules here when implementing.
    # forward(input_ids [B,T], attention_mask [B,T]) -> logits [B]


def orm_correctness_loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Binary correctness loss for the ORM head.

    Contract:
      - logits [B] float64 (numpy view for tests), labels [B] in {0, 1}
      - numerically stable binary cross-entropy-with-logits (derive it; do NOT
        just call torch's BCEWithLogits -- write the stability argument first)

    TODO(student): implement after deriving stability trick on paper.
    """
    raise NotImplementedError("Lab 03b: orm_correctness_loss")


def run_orm_training_loop(model: OrmBinaryHead, train_loader: Any, cfg: OrmPrmLabConfig) -> Any:
    """Full-FT/freeze ORM loop: bf16 autocast, grad accum, clip, eval cadence.

    Contract:
      - appends JSONL metric lines per eval (see LABS_SETUP.md convention)
      - validates batches shaped by :func:`orm_collate_fn`

    TODO(student): implement.
    """
    raise NotImplementedError("Lab 03b: run_orm_training_loop")


# ---------------------------------------------------------------------------
# PRM side -- all student work below
# ---------------------------------------------------------------------------


def load_prm800k_slice(cfg: OrmPrmLabConfig) -> List[Record]:
    """Load the tasksource/PRM800K slice into :data:`PRM_RECORD_KEYS` records.

    Contract:
      - chunked per repo convention (<= ~12 steps/problem reference)
      - respects ``cfg.prm_slice_rows``
      - validate via :func:`validate_prm_records` before returning

    TODO(student): implement loading + chunking.
    """
    raise NotImplementedError("Lab 03b: load_prm800k_slice")


def collate_prm_steps(
    examples: List[Dict[str, Any]], step_pad_value: int = -100
) -> Dict[str, Any]:
    """Expand step labels to token positions aligned with step boundaries.

    Contract:
      - returns input_ids [B,T], attention_mask [B,T], step_label_ids [B,T]
      - ``step_pad_value`` (= -100 = ignore_index) fills non-step/pad positions
        so ambiguous 0-labeled steps never silently become negatives
      - you decide & document how multi-token steps share their label

    TODO(student): implement alignment + padding.
    """
    raise NotImplementedError("Lab 03b: collate_prm_steps")


def prm_step_loss(
    step_logits: np.ndarray, step_label_ids: np.ndarray
) -> np.ndarray:
    """Step-classification loss over {-1, 0, +1} -> 3-way CE (ignore_index=-100).

    Contract:
      - step_logits [B,T,3], step_label_ids [B,T] with -100 ignored
      - masked-mean over active step tokens (shape comments mandatory)

    TODO(student): implement.
    """
    raise NotImplementedError("Lab 03b: prm_step_loss")


def run_prm_training_loop(model: Any, train_loader: Any, cfg: OrmPrmLabConfig) -> Any:
    """PRM loop: same discipline as ORM loop but with step-label metrics.

    TODO(student): implement.
    """
    raise NotImplementedError("Lab 03b: run_prm_training_loop")


# ---------------------------------------------------------------------------
# Dual scoring + disagreement case study (the deliverable) -- student work
# ---------------------------------------------------------------------------


def score_solutions_both_models(
    orm_model: OrmBinaryHead,
    prm_model: Any,
    solutions: List[Record],
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Score ONE shared pool of complete solutions under BOTH heads.

    Contract:
      - returns one record per solution:
        {solution_id, orm_logit, orm_prob, prm_min_step_label,
         prm_flagged_step_idx}
      - ``prm_min_step_label`` = most-negative step estimate in the solution;
        ``prm_flagged_step_idx`` = its index (first occurrence wins ties)

    TODO(student): implement dual scoring.
    """
    raise NotImplementedError("Lab 03b: score_solutions_both_models")


def select_disagreement_cases(
    scored: List[Dict[str, Any]],
    min_cases: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    """Pick disagreement rows honoring the >= ``min_cases`` floor from ``cfg``.

    Contract:
      - verdict rule: sigmoid(logit) >= ``threshold`` -> 'correct' else
        'incorrect'; PRM flags if ``prm_min_step_label`` == -1
      - raises ValueError if fewer than ``min_cases`` survive (config floor,
        see eval.min_cases_required)

    TODO(student): implement selection + floor check.
    """
    raise NotImplementedError("Lab 03b: select_disagreement_cases")


def build_disagreement_table(
    scored: List[Dict[str, Any]],
    ground_truth: Dict[str, str],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Assemble final case-study rows keyed by :data:`CASE_ROW_KEYS`.

    Contract:
      - one row per solution; ``category`` from :data:`DISAGREEMENT_CATEGORIES`
        ('orm_right_prm_flags_reasoning' = correct answer + PRM flag = THE row)
      - ground-truth lookup keyed by solution_id (KeyError if missing)

    TODO(student): implement categorization.
    """
    raise NotImplementedError("Lab 03b: build_disagreement_table")


def write_disagreement_table(rows: List[Dict[str, Any]], path: str) -> None:
    """Write case-study rows as JSONL (convention: one JSON object per line).

    TODO(student): validate each row against CASE_ROW_KEYS first, then write
    one JSON object per line (json.dumps, sort_keys).
    """
    raise NotImplementedError("Lab 03b: write_disagreement_table")


# ---------------------------------------------------------------------------
# Metrics plumbing (given)
# ---------------------------------------------------------------------------


def append_metric_line(jsonl_path: str, payload: Dict[str, Any]) -> None:
    """Append one JSON line to the run's metrics file (LABS_SETUP convention).

    NOTE: actually implemented here (pure IO plumbing) so notebooks can start
    plotting infrastructure even before training loops exist.
    """
    import json as _json

    with open(jsonl_path, "a") as f:
        f.write(_json.dumps(payload, sort_keys=True) + "\n")
