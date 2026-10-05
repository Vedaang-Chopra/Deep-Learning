"""Lab 06 — REINFORCE / RLOO on an LLM (scaffold only).

This module contains signatures, docstring contracts, config plumbing and
TODO markers ONLY. Every algorithmic body is yours to implement.

No-solutions rule (contract §9.4):
  - no baseline-subtraction arithmetic is provided anywhere in this file
  - the GSM8K answer-extraction regex is deliberately left as a stub
    (writing it IS the exercise)
  - torch is intentionally NOT imported at module level; this scaffold must
    import cleanly on a laptop with numpy only

Answer key — read it only AFTER your own version works, then diff:
  ../../../rlhf-book/code/policy_gradients/rollout.py   # group generation loop
  ../../../rlhf-book/code/policy_gradients/utils.py     # compute_loo_advantages, compute_rewards
  ../../../rlhf-book/code/policy_gradients/configs/reinforce.yaml
  ../../../rlhf-book/code/policy_gradients/configs/rloo.yaml

Workflow: prototype here / in notebook.ipynb on a small run → export your
training loop to train.py in the same style as the answer key.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np  # noqa: F401  (used by student implementations, not the scaffold)

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LabConfig:
    """Flat config for Lab 06.

    Mirrors the knobs of the answer-key configs reinforce.yaml / rloo.yaml,
    downsized to learning scale (Qwen3-0.6B instead of Qwen3-1.7B — a
    documented deviation per the curriculum plan).

    Field semantics (see README §Experiments for sweeps):
      loss_mode      "reinforce" | "rloo"
      num_rollouts   K completions per prompt.
                     REINFORCE in the answer key uses num_rollouts=1 (no group,
                     reward used directly); for THIS lab you implement both
                     REINFORCE-per-sample and RLOO across a group of K = 4–8.
      beta           KL penalty coefficient added to the scalar reward
                     (0.0 = disabled, matches both answer-key configs).
      kl_estimator   which approx-KL estimator the KL drift monitor uses;
                     this lab uses k1 ("approx-KL(k1) to init policy").
    """

    loss_mode: str = "reinforce"
    model_name: str = "Qwen/Qwen3-0.6B"
    lr: float = 5e-6
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_new_tokens: int = 512
    prompts_per_step: int = 8
    num_rollouts: int = 1
    beta: float = 0.0
    kl_estimator: str = "kl1"
    data_size: int = 15000
    gsm8k_max_examples: int = 1000
    metrics_path: str = "runs/metrics.jsonl"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.loss_mode not in ("reinforce", "rloo"):
            raise ValueError(
                f"loss_mode must be 'reinforce' or 'rloo', got {self.loss_mode!r}"
            )
        if self.num_rollouts < 1:
            raise ValueError("num_rollouts (K) must be >= 1")
        if self.kl_estimator not in ("kl1", "kl2", "kl3"):
            raise ValueError(f"unsupported kl_estimator {self.kl_estimator!r} (this lab uses kl1)")


def load_config(path: str | Path) -> LabConfig:
    """Load a YAML config file into a :class:`LabConfig`.

    Only reads keys that exist on ``LabConfig``; unknown YAML keys are ignored
    so you can annotate configs freely. Requires PyYAML at call time (not at
    module import time) so the rest of this scaffold stays dependency-light.
    """
    try:
        import yaml  # lazy: module-level import stays torch/notebook friendly
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("PyYAML is required to load LabConfig from YAML") from exc
    with open(Path(path), "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    known = {f.name for f in dc_fields(LabConfig)}
    return LabConfig(**{k: v for k, v in raw.items() if k in known})


# ---------------------------------------------------------------------------
# Experience container (text-level mirror of policy_gradients/buffer.py)
# ---------------------------------------------------------------------------


@dataclass
class Experience:
    """One *group*: one prompt plus its K sampled completions and rewards.

    Text-level stand-in for ``policy_gradients/buffer.py::Experience`` —
    instead of token-id tensors it holds decoded strings and scalar rewards
    so groups can be constructed, inspected and tested without torch/GPU.

    Invariants enforced in ``__post_init__``:
      - every reward component has exactly one entry per completion (K each)
      - the group is non-empty

    Fields you fill during your rollout engine:
      old_log_probs  sequence-level sum log-prob of each completion under the
                     policy that generated it (needed later for ratio checks).
      kl_to_init     approx-KL(k1) of this group's actions vs the *initial*
                     policy snapshot — populated by the KL drift monitor.
    """

    prompt: str
    completions: List[str]
    correctness: List[float]
    format_scores: List[float]
    total_rewards: List[float]
    advantages: Optional[List[float]] = None
    old_log_probs: Optional[List[float]] = None
    kl_to_init: Optional[List[float]] = None

    def __post_init__(self) -> None:
        n = len(self.completions)
        if n == 0:
            raise ValueError("Experience group must contain at least one completion")
        for name in ("correctness", "format_scores", "total_rewards"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"reward component {name!r} has {len(getattr(self, name))} entries "
                    f"but there are {n} completions"
                )

    @property
    def k(self) -> int:
        """Number of rollouts in this group."""
        return len(self.completions)


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------


def verify_format(completion: str) -> bool:
    """Return True iff ``completion`` follows the required output format.

    Contract:
      - input is ONE raw generated completion string
      - returns a strict bool (never a score, never None) — RL downstream code
        treats this as a gate, so silent falsy values are bugs
      - the exact tag set / structure you require is YOUR design decision.
        For inspiration on how the answer key scores <think>/<answer> tags see
        ``utils.py::_format_reward`` — read it AFTER your version works.

    TODO(student): decide the tag contract, then implement.
    """
    raise NotImplementedError("Lab 06: verify_format is yours to write")


def verify_correctness(completion: str, answer: str) -> bool:
    """Return True iff ``completion`` solves the task with gold ``answer``.

    Contract:
      - boolean outcome verifier: True iff the extracted final response is
        correct for this entry, False otherwise (strict bool again)
      - for spell_backward compare against the expected reversed string;
        for GSM8K extract the final numeric answer first (see
        :func:`extract_gsm8k_answer`) and compare normalized values
      - string equality alone is NOT enough for numbers across formats
        (think: trailing zeros, commas, "1e5")

    TODO(student): implement per task; keep it deterministic and fast.
    """
    raise NotImplementedError("Lab 06: verify_correctness is yours to write")


def extract_gsm8k_answer(completion: str) -> Optional[str]:
    """Extract the final boxed/claimed numeric answer from a GSM8K completion.

    Contract:
      - takes a raw completion string, returns the candidate answer substring
        or None when nothing plausible can be extracted
      - THE REGEX ITSELF IS SOLUTION CONTENT and is deliberately NOT here —
        writing a robust extraction rule (and watching it fail on real model
        outputs) is part of this lab
      - after extraction you still normalize before comparing (see above)

    TODO(student): design the pattern + fallback chain; unit-test it against
    messy completions you actually sample from the model.
    """
    raise NotImplementedError("Lab 06: the GSM8K extraction regex is the exercise")


# ---------------------------------------------------------------------------
# Advantage estimators (NO baseline-subtraction arithmetic is provided)
# ---------------------------------------------------------------------------


def compute_reinforce_advantages(total_rewards: Sequence[float]) -> List[float]:
    """REINFORCE advantage estimator WITHOUT any baseline.

    Contract:
      - input: sequence-level total rewards (one scalar per rollout)
      - output: same-length list where each position holds the gradient
        coefficient used by the REINFORCE objective, i.e. what multiplies
        ``-log pi(completion)`` inside your loss
      - baseline-free variant: derive the coefficient directly from the
        reward itself (this is the high-variance version — you will measure
        exactly why in the notebook)

    TODO(student): one line of math once you understand what A should be here.
    """
    raise NotImplementedError("Lab 06: REINFORCE-no-baseline advantages")


def compute_rloo_advantages(total_rewards: Sequence[float]) -> List[float]:
    """RLOO leave-one-out advantage estimator over ONE group of K rewards.

    Contract:
      - input: the K total rewards of a single group (order preserved)
      - output: K advantages; each rollout's baseline must be computed from
        the OTHER rollouts' rewards only (leave-one-out), which removes the
        bias a shared-group mean would introduce into its own member
      - grouping semantics: rewards belong to one prompt; averaging across
        prompts is a bug (each group standardizes independently)
      - expect zero advantages wherever the whole group ties — be ready to
        explain mechanically WHY (Lab 07 hits the same wall in GRPO)

    TODO(student): derive the leave-one-out mean and the resulting estimator.
    No scale factor short-cuts: work it out by hand first.
    """
    raise NotImplementedError("Lab 06: RLOO leave-one-out advantages")


def compute_importance_ratios(new_log_probs: Sequence[float], old_log_probs: Sequence[float]) -> List[float]:
    """Per-token importance ratios between current and sampling policies.

    Contract:
      - elementwise exp(new - old) on token-level log-probs; equal lengths are
        required (assert, do not silently truncate)
      - used for the OPTIONAL epoch-reuse extension: with fresh samples every
        step all ratios are exactly 1.0; reused batches let you watch the
        ratio distribution drift away from 1.0 — that statistic IS the point
        of the exercise, so plot its histogram, don't just print a mean

    TODO(student): implement during the optional extension phase.
    """
    raise NotImplementedError("Lab 06: importance-ratio statistics (optional extension)")


# ---------------------------------------------------------------------------
# KL drift monitor
# ---------------------------------------------------------------------------


def approx_kl_k1(log_probs_current: Sequence[float], log_probs_ref: Sequence[float]) -> float:
    """Approximate per-sequence KL divergence using estimator k1.

    Contract:
      - inputs: token-level log-probs of the SAME tokens under the current
        policy and the reference (initial) policy
      - returns a single non-negative float averaged over positions; an empty
        or length-mismatched pair of inputs raises ValueError
      - k1 is the simple ratio-based estimator; Lab 07 makes you implement
        k2/k3 and compare their bias/variance — keep this function isolated so
        you can swap estimators via ``LabConfig.kl_estimator`` later

    TODO(student): implement the k1 form (one expression over log-ratios).
    """
    raise NotImplementedError("Lab 06: approx-KL(k1) monitor math")


@dataclass
class KLDriftRecord:
    """One monitoring datapoint returned by the KL drift check."""

    step: int
    kl: float
    flagged: bool


def check_kl_drift(step: int, kl: float, threshold: float = 10.0) -> KLDriftRecord:
    """Flag a training step whose KL-to-init estimate exceeds ``threshold``.

    Contract:
      - pure bookkeeping wrapper around :func:`approx_kl_k1` outputs; returns a
        record with ``flagged=True`` iff ``kl > threshold``
      - thresholds are config/scale dependent: justify whichever number you
        pick in the README experiments section rather than trusting this default

    TODO(student): wire this into your metrics logging once KL works.
    """
    raise NotImplementedError("Lab 06: KL drift flagging")


# ---------------------------------------------------------------------------
# Metrics logging (JSONL convention — curriculum plan §7.5)
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Append-only JSONL metrics writer.

    Convention (every lab inherits it): each training step appends ONE flat
    JSON object with at least ``{"step": int, "ts": iso-string}`` plus whatever
    scalars you monitored that step (avg correctness/format, reward components,
    response length, grad norm, approx-KL(k1), importance-ratio stats...).
    Notebooks later plot straight from this file regardless of platform —
    wandb stays optional.

    TODO(student): implement ``log`` (open lazily, append, flush) and
    ``summarize_group`` (turn an :class:`Experience` into the scalar dict).
    The constructor is provided so the scaffolding is uniform across labs.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Any = None

    def log(self, record: Mapping[str, Any]) -> None:
        """Append one JSON line (keys/values JSON-serializable) then flush."""
        raise NotImplementedError("Lab 06: JSONL append+flush")

    def summarize_group(self, exp: Experience, step: int) -> Dict[str, Any]:
        """Flatten one Experience group into the per-step metric dict.

        Expected keys (yours may add more): avg_correctness, avg_format,
        avg_total_reward, group_contrast (max-min of rewards within the group),
        mean_length, k. Logging these BEFORE training starts gives you the
        base rate your completion criterion is measured against.
        """
        raise NotImplementedError("Lab 06: group summary flattening")


# ---------------------------------------------------------------------------
# Rollout engine
# ---------------------------------------------------------------------------


class RolloutEngine:
    """Minimal LLM rollout loop yielding one :class:`Experience` group per step.

    Mirrors the shape of ``policy_gradients/rollout.py::RolloutEngine`` but at
    text level: sampling K completions per prompt, scoring them with the
    verifiers above, recording everything into an Experience.

    Generation hygiene to handle yourself (symptoms appear in Debugging §README):
      - left padding + pad-token masking; completions slice off the prompt
      - deterministic seeds per batch so reruns reproduce a fix
      - prompt/K bookkeeping consistent with ``prompts_per_step`` ×
        ``num_rollouts``
    """

    def __init__(self, cfg: LabConfig, model: Any = None, tokenizer: Any = None,
                 task_sources: Optional[Sequence[Any]] = None) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.task_sources = list(task_sources) if task_sources is not None else []

    def build_generation_config(self) -> Dict[str, Any]:
        """Sampling kwargs dict from cfg (temperature/top_p/top_k/min_p/
        max_new_tokens). Pure plumbing — implement freely; nothing secret.
        """
        return {
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "min_p": self.cfg.min_p,
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": True,
        }

    def sample_group(self, prompt: str, k: int) -> List[str]:
        """Generate K independent completions for one prompt.

        Contract:
          - exact count K distinct sampling passes (batched generate fine)
          - returns DECODED text strings with prompt stripped and special
            tokens handled consistently with your tokenizer's protocol
          - TODO(student): load the chat template decision from Lab 00 here
        """
        raise NotImplementedError("Lab 06: K-completion sampling")

    def score_and_pack(self, prompt: str, answer: str, completions: List[str]) -> Experience:
        """Verify completions, combine into total rewards, pack one Experience.

        Contract:
          - calls verify_format / verify_correctness per completion
          - combines scores into scalar totals per your chosen weighting
            (justify weightings; start by matching the repo's spirit:
            correctness dominates, format gates)
          - leaves ``advantages`` unset — those come from the estimators,
            NEVER computed inside scoring
        """
        raise NotImplementedError("Lab 06: score completions and pack a group")

    def collect_batch(self) -> List[Experience]:
        """Draw ``prompts_per_step`` prompts and yield their groups.

        This is the engine iteration used by the training loop; respects
        ``cfg.num_rollouts`` for how many completions live in each group.
        """
        raise NotImplementedError("Lab 06: batch collection loop")
