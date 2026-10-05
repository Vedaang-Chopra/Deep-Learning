"""Lab 07 — GRPO / RLVR capstone (scaffold only).

This module contains signatures, docstring contracts, config plumbing and
TODO markers ONLY. Every algorithmic body is yours to implement.

No-solutions rule (contract §9.4) — applied EXTRA strictly here, because this
lab is the contract's explicit worked example:
  - NO advantage arithmetic anywhere in this file: no group location/scale
    normalization, no Dr.GRPO variant math, and no zero-contrast-group
    handling logic (the decision itself is the exercise)
  - NO clipped-surrogate formula (asymmetric eps support is plumbing only)
  - NO KL estimator formulas — k1, k2 AND k3 are ALL stubbed
  What you get: signatures, rich contract docstrings (shapes like
  ``advantages [N, K]``), config plumbing, and `raise NotImplementedError`.

Answer key — read it only AFTER your own version works, then diff:
  ../../../rlhf-book/code/policy_gradients/loss.py::GRPOLoss
  ../../../rlhf-book/code/policy_gradients/utils.py::compute_standardized_advantages
  ../../../rlhf-book/code/policy_gradients/utils.py::compute_nonstandardized_advantages
  ../../../rlhf-book/code/policy_gradients/configs/grpo.yaml
Post-hoc readings (same file, after the capstone): GSPOLoss, DAPOLoss —
compare their ratio granularity and normalization choices to yours.

Workflow: prototype here / in notebook.ipynb on a small run → export your
training loop to train.py in the same style as the answer key. Reuse your
Lab 06 rollout engine and verifiers rather than forking them.
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
    """Flat config for Lab 07.

    Mirrors the knobs of the answer-key config
    ``policy_gradients/configs/grpo.yaml``, downsized to learning scale
    (Qwen3-0.6B instead of Qwen3-1.7B — documented deviation per the
    curriculum plan) and extended with the lab's sweep axes.

    Field semantics (see README §Experiments for sweeps):
      loss_mode             "grpo" | "drgrpo" — selects WHICH group-advantage
                            estimator feeds the loss; the rest of the loop is
                            shared (the answer key shares GRPOLoss for both).
      num_rollouts          K completions per prompt — the GRPO group size
                            (answer key: 8). N prompts x K rollouts per step.
      clip_eps_lo/hi        asymmetric clipping bounds for the importance
                            update; grpo.yaml uses 0.2/0.2, the eps sweep and
                            DAPO's clip-higher idea use hi > lo.
      beta                  KL penalty coefficient (0 = disabled). Sweep
                            {0, 0.001, 0.04} per the curriculum plan.
      kl_estimator          "kl1" | "kl2" | "kl3" — which hand-implemented
                            estimator the KL panel uses (grpo.yaml: kl3).
      zero_contrast_groups  policy for groups whose K rewards are all equal:
                            "skip" | "weight" | "error". Implement, then
                            JUSTIFY your choice in the README — there is no
                            scaffold default logic for this.
      loss_normalization    "sequence_mean" | "token" — sequence-mean vs
                            token-level normalization experiment axis
                            (compare DAPOLoss's choice post-hoc).
      format_weight         scalar weight of the format reward inside the
                            total reward (correctness dominates by design).
    """

    loss_mode: str = "grpo"
    model_name: str = "Qwen/Qwen3-0.6B"
    lr: float = 5e-6
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_new_tokens: int = 512
    prompts_per_step: int = 4
    num_rollouts: int = 8
    clip_eps_lo: float = 0.2
    clip_eps_hi: float = 0.2
    beta: float = 0.0
    kl_estimator: str = "kl3"
    zero_contrast_groups: str = "skip"
    loss_normalization: str = "sequence_mean"
    format_weight: float = 0.3
    max_norm: float = 1.0
    data_size: int = 3000
    gsm8k_max_examples: int = 1000
    metrics_path: str = "runs/metrics.jsonl"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.loss_mode not in ("grpo", "drgrpo"):
            raise ValueError(f"loss_mode must be 'grpo' or 'drgrpo', got {self.loss_mode!r}")
        if self.num_rollouts < 2:
            raise ValueError("num_rollouts (group size K) must be >= 2 for group-relative advantages")
        if self.kl_estimator not in ("kl1", "kl2", "kl3"):
            raise ValueError(f"unsupported kl_estimator {self.kl_estimator!r} (kl1|kl2|kl3)")
        if self.zero_contrast_groups not in ("skip", "weight", "error"):
            raise ValueError(
                "zero_contrast_groups must be 'skip', 'weight' or 'error', "
                f"got {self.zero_contrast_groups!r}"
            )
        if self.loss_normalization not in ("sequence_mean", "token"):
            raise ValueError(
                f"loss_normalization must be 'sequence_mean' or 'token', got {self.loss_normalization!r}"
            )
        if self.clip_eps_lo <= 0 or self.clip_eps_hi <= 0:
            raise ValueError("clip_eps_lo / clip_eps_hi must be positive")


def load_config(path: str | Path) -> LabConfig:
    """Load a YAML config file into a :class:`LabConfig`.

    Only reads keys that exist on ``LabConfig``; unknown YAML keys are ignored
    so you can annotate configs freely. Requires PyYAML at call time (not at
    module import time) so this scaffold stays dependency-light.
    """
    try:
        import yaml  # lazy: module-level import stays notebook friendly
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("PyYAML is required to load LabConfig from YAML") from exc
    with open(Path(path), "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    known = {f.name for f in dc_fields(LabConfig)}
    return LabConfig(**{k: v for k, v in raw.items() if k in known})


# ---------------------------------------------------------------------------
# Group container (text-level mirror of policy_gradients/buffer.py)
# ---------------------------------------------------------------------------


@dataclass
class GroupRecord:
    """One prompt plus its K sampled completions, rewards and bookkeeping.

    Text-level stand-in for ``policy_gradients/buffer.py::Experience`` —
    decoded strings and scalar rewards so groups can be constructed,
    inspected and tested without torch/GPU.

    Invariants enforced in ``__post_init__``:
      - every reward component has exactly one entry per completion (K each)
      - the group is non-empty and K >= 2 (a group of one has no contrast)

    Fields you fill during the lab:
      advantages   per-completion gradient coefficients — produced ONLY by
                   your advantage estimators, NEVER inside scoring
      old_log_probs sequence-level log-prob of each completion under the
                   policy that generated it (needed for the update's
                   importance weighting)
      kl_to_ref    per-completion approx-KL estimate vs the reference policy
                   — populated by the KL panel
    """

    prompt: str
    completions: List[str]
    correctness: List[float]
    format_scores: List[float]
    total_rewards: List[float]
    lengths: Optional[List[int]] = None
    advantages: Optional[List[float]] = None
    old_log_probs: Optional[List[float]] = None
    kl_to_ref: Optional[List[float]] = None

    def __post_init__(self) -> None:
        n = len(self.completions)
        if n < 2:
            raise ValueError("GroupRecord needs K >= 2 completions for group-relative advantages")
        for name in ("correctness", "format_scores", "total_rewards"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"reward component {name!r} has {len(getattr(self, name))} entries "
                    f"but there are {n} completions"
                )

    @property
    def k(self) -> int:
        """Number of rollouts in this group (the GRPO group size K)."""
        return len(self.completions)

    @property
    def has_contrast(self) -> bool:
        """True iff the K total rewards are not all equal.

        Bookkeeping only: this checks for a tie, it does not decide what to
        DO with a tied group — that policy is ``zero_contrast_groups`` and
        handling it is yours to implement.
        """
        return max(self.total_rewards) != min(self.total_rewards)


# ---------------------------------------------------------------------------
# Verifiers (RLVR) — reuse your Lab 06 implementations
# ---------------------------------------------------------------------------


def verify_format(completion: str) -> bool:
    """Return True iff ``completion`` follows the required output format.

    Contract: strict bool, one raw completion in, gate decision out.
    RECOMMENDED: copy your Lab 06 ``verify_format`` — the capstone grades the
    GRPO machinery, not a second verifier. Keep it identical so Lab 06/07
    metrics stay comparable.

    TODO(student): paste your Lab 06 implementation (or reimplement).
    """
    raise NotImplementedError("Lab 07: reuse/paste your Lab 06 verify_format")


def verify_correctness(completion: str, answer: str) -> bool:
    """Return True iff ``completion`` solves the task with gold ``answer``.

    Contract: strict bool; must cover BOTH arms — spell_backward string-match
    AND GSM8K numeric exact-match after answer extraction.

    TODO(student): paste your Lab 06 implementation (or reimplement).
    """
    raise NotImplementedError("Lab 07: reuse/paste your Lab 06 verify_correctness")


def extract_gsm8k_answer(completion: str) -> Optional[str]:
    """Extract the final claimed numeric answer from a GSM8K completion.

    Contract: candidate substring or None. The extraction rule itself is
    solution content — Lab 06 already made you write it.

    TODO(student): paste your Lab 06 implementation (or reimplement).
    """
    raise NotImplementedError("Lab 07: reuse/paste your Lab 06 extraction rule")


# ---------------------------------------------------------------------------
# Group-advantage estimators (NO advantage arithmetic is provided)
# ---------------------------------------------------------------------------


def compute_grpo_advantages(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardized group-relative advantages (GRPO), vectorized over batches.

    Contract:
      - input ``rewards``: shape [N, K] — N prompts (rows) x K rollouts each
        (columns); row i holds one prompt's group of K total rewards
      - output: advantages [N, K], same shape, each row standardized WITHIN
        the group so that completions of one prompt are ranked only against
        their own siblings — never across prompts
      - ``eps``: stability constant for groups with (near-)zero spread; how
        it enters is part of the estimator you derive
      - this is the GRPO/GSPO/CISPO/DAPO family estimator in the answer key

    Watch the failure mode the README asks about: when a whole group ties,
    what does your formula produce, and is that the gradient signal you want?

    TODO(student): derive the per-row transformation from the GRPO objective.
    """
    raise NotImplementedError("Lab 07: standardized (GRPO) group advantages")


def compute_drgrpo_advantages(rewards: np.ndarray) -> np.ndarray:
    """Non-standardized group-relative advantages (Dr.GRPO), vectorized.

    Contract:
      - same [N, K] -> [N, K] shape contract as :func:`compute_grpo_advantages`
      - the Dr.GRPO variant REMOVES one component of GRPO's per-row transform
        to eliminate a reward-scale bias; derive which component and why
        (Chapter 6 GRPO section + the Dr.GRPO paper) — do NOT guess-and-check
      - no ``eps`` parameter: after your derivation, decide whether one is
        still needed and justify either way in the README

    TODO(student): derive the Dr.GRPO per-row transformation.
    """
    raise NotImplementedError("Lab 07: non-standardized (Dr.GRPO) group advantages")


def handle_zero_contrast_groups(
    advantages: np.ndarray, rewards: np.ndarray, mode: str = "skip"
) -> np.ndarray:
    """Apply the zero-contrast-group policy to a batch of advantages.

    Contract:
      - inputs: ``advantages`` [N, K] (already produced by ONE of the two
        estimators above) and the ``rewards`` [N, K] they came from
      - a row is "zero-contrast" iff its K rewards are all equal (the fixture
        in the tests builds such rows — check your estimator's output on them
        FIRST, before deciding how to treat them)
      - ``mode`` comes from ``LabConfig.zero_contrast_groups``:
          "skip"   — drop those rows from the update (decide the exact
                     semantics yourself: masked-out vs removed, and what the
                     step's normalization then averages over)
          "weight" — keep the rows but attenuate them (you choose the
                     weighting rule and defend it)
          "error"  — raise ValueError, forcing upstream resampling
      - returns advantages [N, K] (or fewer rows if your skip semantics
        remove rows — document WHICH and make tests match your choice)
      - you must JUSTIFY the choice in the README (statistical argument, not
        vibes: what does each option do to the estimator's bias?)

    TODO(student): implement all three modes + the justification.
    """
    raise NotImplementedError("Lab 07: zero-contrast-group policy")


# ---------------------------------------------------------------------------
# Clipped update (NO surrogate formula is provided)
# ---------------------------------------------------------------------------


def compute_clipped_surrogate(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    clip_eps_lo: float,
    clip_eps_hi: float,
) -> np.ndarray:
    """Per-token clipped surrogate terms for the GRPO policy update.

    Contract:
      - inputs: ``log_probs_new`` / ``log_probs_old`` [B, T] token-level
        log-probs of the SAME tokens under current vs sampling policy,
        ``advantages`` [B, T] broadcast per token (sequence-level advantages
        expanded over tokens is fine — say which you did)
      - asymmetric bounds: the lower excursion is limited by
        ``1 - clip_eps_lo``, the upper by ``1 + clip_eps_hi``; grpo.yaml
        uses equal eps, DAPO's clip-higher sets hi > lo (post-hoc reading)
      - output: per-token surrogate contribution [B, T] that the loss will
        aggregate according to ``LabConfig.loss_normalization`` — implement
        BOTH aggregations in your training loop and compare
      - the pessimistic-update structure (what gets compared, what survives)
        is the point of the exercise; derive it from Chapter 6

    TODO(student): derive the clipped surrogate from the PPO/GRPO objective.
    """
    raise NotImplementedError("Lab 07: clipped surrogate with asymmetric eps")


def grpo_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    log_probs_ref: Optional[np.ndarray],
    action_mask: np.ndarray,
    clip_eps_lo: float,
    clip_eps_hi: float,
    beta: float,
    kl_estimator: str,
    loss_normalization: str = "sequence_mean",
) -> float:
    """Full GRPO loss: clipped surrogate + optional KL penalty, aggregated.

    Contract:
      - combines :func:`compute_clipped_surrogate` with the hand-implemented
        KL estimators below; ``beta == 0`` disables the KL term entirely
        (and must short-circuit WITHOUT touching ``log_probs_ref``)
      - ``action_mask`` [B, T] (1 = generated token, 0 = pad/prompt) gates
        every reduction — prompt/pad positions contribute nothing
      - ``loss_normalization``: "sequence_mean" aggregates per sequence then
        across the batch; "token" aggregates over all active tokens at once
        (the two differ when sequences have unequal lengths — that difference
        is the experiment)
      - returns a scalar float
      - mirror of the answer key's ``GRPOLoss.forward`` (loss.py) — diff
        against it AFTER your version works, then read GSPOLoss/DAPOLoss

    TODO(student): assemble the pieces; no formula is given for any of them.
    """
    raise NotImplementedError("Lab 07: full GRPO loss assembly")


# ---------------------------------------------------------------------------
# KL estimators k1 / k2 / k3 — ALL THREE STUBBED (implement by hand)
# ---------------------------------------------------------------------------


def approx_kl_k1(
    log_probs_new: np.ndarray, log_probs_ref: np.ndarray, action_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """KL estimator k1 (Schulman, http://joschu.net/blog/kl-approx.html).

    Contract:
      - inputs: token-level log-probs of the SAME tokens under current and
        reference policies, shape [B, T]; optional ``action_mask`` gates
        positions (masked positions contribute nothing)
      - returns per-position estimate [B, T]; non-negative IN EXPECTATION
        only — whether k1 can dip below zero pointwise, and what that does
        to your panel plots, is exactly the bias/variance question this lab
        makes you answer for all three estimators
      - derive each estimator from the log-of-expectation vs expectation-of-log
        gap (the same source derives all three — read it, then implement)

    TODO(student): implement k1.
    """
    raise NotImplementedError("Lab 07: KL estimator k1")


def approx_kl_k2(
    log_probs_new: np.ndarray, log_probs_ref: np.ndarray, action_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """KL estimator k2 (Schulman, http://joschu.net/blog/kl-approx.html).

    Contract: identical calling convention to :func:`approx_kl_k1`.
    TODO(student): implement k2 after deriving it from the same source.
    """
    raise NotImplementedError("Lab 07: KL estimator k2")


def approx_kl_k3(
    log_probs_new: np.ndarray, log_probs_ref: np.ndarray, action_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """KL estimator k3 (Schulman, http://joschu.net/blog/kl-approx.html).

    Contract: identical calling convention to :func:`approx_kl_k1`. This is
    the answer key's default (grpo.yaml: kl3).
    TODO(student): implement k3 after deriving it from the same source.
    """
    raise NotImplementedError("Lab 07: KL estimator k3")


KL_ESTIMATORS = {"kl1": approx_kl_k1, "kl2": approx_kl_k2, "kl3": approx_kl_k3}


def compare_kl_estimators(
    log_probs_new: np.ndarray,
    log_probs_ref: np.ndarray,
    action_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Evaluate ALL THREE hand-implemented estimators on the SAME batch.

    Contract:
      - thin dispatcher over :data:`KL_ESTIMATORS`; returns
        ``{"kl1": [B, T], "kl2": [B, T], "kl3": [B, T]}``
      - this exists so the notebook can plot the three estimates for the
        SAME batch side by side — the deliverable is that comparison plot,
        plus a written bias/variance verdict per estimator
      - batch statistics (how you summarize [B, T] -> one panel number) are
        your design decision; document them next to the plot
    """
    return {name: fn(log_probs_new, log_probs_ref, action_mask) for name, fn in KL_ESTIMATORS.items()}


# ---------------------------------------------------------------------------
# Diagnostics panel (JSONL convention — curriculum plan §7.5)
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Append-only JSONL metrics writer (same convention as Lab 06).

    Each training step appends ONE flat JSON object with at least
    ``{"step": int, "ts": iso-string}`` plus the diagnostic-panel scalars.
    Notebooks plot straight from this file regardless of platform.

    TODO(student): implement ``log`` and ``summarize_batch``; the constructor
    is provided so scaffolding stays uniform across labs.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Any = None

    def log(self, record: Mapping[str, Any]) -> None:
        """Append one JSON line (keys/values JSON-serializable) then flush."""
        raise NotImplementedError("Lab 07: JSONL append+flush")

    def summarize_batch(
        self,
        rewards: np.ndarray,
        advantages: np.ndarray,
        lengths: Sequence[int],
        kl: Optional[float] = None,
        entropy: Optional[float] = None,
        grad_norm: Optional[float] = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        """Flatten one training batch into the diagnostic-panel dict.

        Expected keys (add more freely): avg_correctness, avg_format,
        avg_total_reward, group_contrast (mean within-group reward spread —
        the saturation tripwire), frac_zero_contrast, avg_advantage,
        kl_to_ref, entropy, mean_length, grad_norm, n_groups, k.
        Logging this BEFORE training starts gives you the base rate.
        """
        raise NotImplementedError("Lab 07: diagnostic-panel flattening")


# ---------------------------------------------------------------------------
# Rollout engine — reuse your Lab 06 engine, add group bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class Lab07ConfigMixin:
    """Reserved for engine-level knobs you add beyond :class:`LabConfig`.

    Deliberately empty: the capstone's config surface is LabConfig. If your
    implementation needs engine knobs (e.g. per-group seeds for reproducing
    a specific failure), add them HERE and keep LabConfig answer-key-shaped.
    """


class GRPORolloutEngine:
    """Group rollout loop yielding one :class:`GroupRecord` per prompt.

    RECOMMENDED: compose your Lab 06 RolloutEngine (sampling + verifiers)
    rather than forking it — the capstone adds GROUP bookkeeping on top:
    K completions per prompt, per-group contrast tracking, old log-probs,
    and packed [N, K] reward matrices ready for the advantage estimators.

    Generation hygiene is unchanged from Lab 06 (left padding + pad masking,
    prompt stripped from decoded text, deterministic per-batch seeds).
    """

    def __init__(
        self,
        cfg: LabConfig,
        model: Any = None,
        tokenizer: Any = None,
        task_sources: Optional[Sequence[Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.task_sources = list(task_sources) if task_sources is not None else []

    def build_generation_config(self) -> Dict[str, Any]:
        """Sampling kwargs dict from cfg (pure plumbing — nothing secret)."""
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

        Contract: exactly K decoded completions, prompt stripped, consistent
        special-token protocol (see Lab 06 for the chat-template decision).

        TODO(student): delegate to your Lab 06 sampler.
        """
        raise NotImplementedError("Lab 07: K-completion group sampling")

    def score_group(self, prompt: str, answer: str, completions: List[str]) -> GroupRecord:
        """Verify completions, combine into total rewards, pack one GroupRecord.

        Contract:
          - calls verify_format / verify_correctness per completion
          - total = correctness + format_weight * format (justify deviations;
            the misalignment games in Lab 08 later abuse this combination)
          - ``advantages`` stays unset — NEVER computed inside scoring
          - record per-completion token ``lengths`` (the length panel needs
            them, and token-level normalization changes with them)

        TODO(student): wire your Lab 06 verifiers in and pack the record.
        """
        raise NotImplementedError("Lab 07: score completions and pack a group")

    def collect_batch(self) -> List[GroupRecord]:
        """Draw ``prompts_per_step`` prompts and yield their groups.

        This is the engine iteration the training loop consumes; K comes
        from ``cfg.num_rollouts``. Also the right place to COUNT zero-contrast
        groups per batch — the frac_zero_contrast panel metric falls out of
        that count for free.

        TODO(student): implement the collection loop.
        """
        raise NotImplementedError("Lab 07: batch collection loop")


# ---------------------------------------------------------------------------
# Training step (export target: train.py)
# ---------------------------------------------------------------------------


def training_step(engine: GRPORolloutEngine, cfg: LabConfig, logger: MetricsLogger, step: int) -> Dict[str, Any]:
    """One GRPO/RLVR update: collect groups -> advantages -> loss -> update.

    Contract:
      - collect_batch -> stack rewards into [N, K] -> choose the estimator
        per ``cfg.loss_mode`` -> apply ``handle_zero_contrast_groups`` ->
        grpo_loss -> optimizer step (grad-norm clipping per ``cfg.max_norm``)
      - appends ONE diagnostic-panel line per step via ``logger``
      - returns the metric dict it logged (handy for notebook assertions)
      - the "all advantages ~ 0" and "reward up, KL up, gibberish" debugging
        scenarios in the README are diagnosed FROM these logged lines — make
        sure every panel quantity survives into the JSONL

    TODO(student): implement; this is the function train.py loops over.
    """
    raise NotImplementedError("Lab 07: one GRPO training step")
