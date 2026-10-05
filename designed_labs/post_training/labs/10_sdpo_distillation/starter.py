"""Lab 10 - SDPO On-Policy Distillation (scaffold).

On-policy self-distillation for reasoning: the SAME model plays both roles.
Per prompt, the **student** samples a group of ``num_rollouts`` rollouts; when
at least one sibling rollout is verifiably correct, that demo is spliced back
into the prompt and the model is **reprompted as its own teacher**. The
student's on-policy logits are distilled toward the teacher's with a
hand-written **top-K reverse-KL** loss whose K+1 distribution is closed with a
**tail bucket** carrying all non-top-K mass. Groups with zero correct rollouts
are skipped and refilled (watch the ``skipped`` metric decay as the loop
converges).

Prerequisites: Lab 07 (group rollouts + verifiable reward) + Lecture 7
(Synthetic data & distillation) + Chapter 12 of the RLHF Book, incl. the OPSD
section (https://rlhfbook.com/c/12-synthetic-data.html).

NO-SOLUTIONS SCAFFOLD: this file defines contracts only. Every function whose
body is a mechanism you are meant to implement -- group rollout orchestration
with skip-and-refill, the demonstration-conditioned teacher reprompt, the
spell_backward verifier bridge, the top-K reverse-KL loss (incl. the tail
bucket) and the metrics/training loop -- is marked with TODO and raises
NotImplementedError. Implemented helpers below are pure plumbing: config
loading/validation, rollout-group container validation, skip bookkeeping
fields, and metrics-record shape validation. Do not peek at the answer key
(rlhf-book/code/distillation/, especially loss.py::add_tail and
configs/sdpo.yaml) until your version trains.

Runs torch-free at import time: module-level dependencies are numpy + PyYAML +
stdlib, so the tests collect on a CPU-only machine. Inside the stubs you will
use torch at *call* time; the loss class never imports it at module level.
"""

from __future__ import annotations

import math  # noqa: F401  (you may need it in your implementations)
import random  # noqa: F401  (seeded sampling -- provided for your implementations)
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Types & constants (given)
# ---------------------------------------------------------------------------

Record = Dict[str, Any]

#: One rollout group has exactly these top-level keys: the prompt, the
#: verifiable target (for spell_backward: the reversed word), the
#: ``num_rollouts`` completions, and their scalar rewards (completions[i] was
#: scored as rewards[i]).
ROLLOUT_GROUP_KEYS = ("question", "target", "completions", "rewards")

#: One JSONL metrics line has exactly these top-level keys (see README
#: "Metrics loop"): reward = mean verifiable reward over trained groups,
#: distill_loss = the scaled loss value returned by the loss accumulation,
#: skipped = groups discarded this step, skipped_rate = skipped / (skipped +
#: prompts_per_step).
METRICS_RECORD_KEYS = ("step", "reward", "distill_loss", "skipped", "skipped_rate")

#: Teacher reprompt skeleton (README "Demonstration-conditioned teacher"):
#: the exact surface strings are fixed so your teacher prompt is comparable to
#: the answer key; ASSEMBLING the three pieces is your TODO.
TEACHER_DEMO_HEADER = "Correct solution:\n\n"
TEACHER_TASK_SUFFIX = "Correctly solve the original question."


# ---------------------------------------------------------------------------
# Config loading (given plumbing)
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the lab YAML into a plain dict.

    The dict is intentionally untyped (no pydantic dependency here); structural
    validation lives in :func:`validate_config`. The loss reads
    ``cfg["kl_top_k"]`` and ``cfg["rollout_chunk"]``; the loop reads
    ``cfg["prompts_per_step"]``, ``cfg["num_rollouts"]`` and
    ``cfg["success_reward_threshold"]``.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Raise ValueError if ``cfg`` violates the lab's structural contract.

    Checks (all structural, no algorithm content):

    * required top-level keys: ``model_name``, ``data``, and the SDPO/
      generation/training scalars read by the stubs below
    * ``data.specs`` contains a ``spell_backward`` entry (the lab's task)
    * ``kl_top_k`` int >= 1 and ``rollout_chunk`` int >= 1
    * ``num_rollouts`` int >= 1 and ``prompts_per_step`` int >= 1
    * ``success_reward_threshold`` numeric in [0, 1]
    * nonnegative generation/training scalars where negative values are
      meaningless (temperature, top_p, max_new_tokens, max_prompt_len,
      max_reprompt_len, lr, num_steps, max_norm)
    """
    for key in ("model_name", "data"):
        if key not in cfg or cfg[key] in (None, ""):
            raise ValueError(f"config missing key: {key!r}")
    data = cfg["data"]
    if not isinstance(data, dict) or not isinstance(data.get("specs"), list):
        raise ValueError("config['data']['specs'] must be a list")
    if not any(
        isinstance(s, dict) and s.get("name") == "spell_backward" for s in data["specs"]
    ):
        raise ValueError("config['data']['specs'] must include 'spell_backward'")

    for key in ("kl_top_k", "rollout_chunk", "num_rollouts", "prompts_per_step"):
        val = cfg.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(f"config[{key!r}] must be an int >= 1")

    thr = cfg.get("success_reward_threshold")
    if isinstance(thr, bool) or not isinstance(thr, (int, float)):
        raise ValueError("config['success_reward_threshold'] must be numeric")
    if not 0.0 <= float(thr) <= 1.0:
        raise ValueError("config['success_reward_threshold'] must be in [0, 1]")

    for key in (
        "temperature",
        "top_p",
        "max_new_tokens",
        "max_prompt_len",
        "max_reprompt_len",
        "lr",
        "num_steps",
        "max_norm",
    ):
        val = cfg.get(key)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"config[{key!r}] must be numeric")
        if val < 0:
            raise ValueError(f"config[{key!r}] must be >= 0")


# ---------------------------------------------------------------------------
# Rollout-group container validation (given plumbing)
# ---------------------------------------------------------------------------


def validate_rollout_group(group: Record) -> None:
    """Assert one rollout group has the container-contract shape.

    Raises ValueError unless the group:

    * is a dict containing exactly :data:`ROLLOUT_GROUP_KEYS` at top level
    * has ``len(completions) == len(rewards) > 0``
    * has float-numeric ``rewards`` entries (verifiable reward is 0.0/1.0 in
      this lab, but the container only demands numeric)
    """
    if not isinstance(group, dict):
        raise ValueError(f"group: expected dict, got {type(group).__name__}")
    keys = tuple(sorted(group.keys()))
    if keys != tuple(sorted(ROLLOUT_GROUP_KEYS)):
        raise ValueError(
            f"group: keys {keys} != container contract {sorted(ROLLOUT_GROUP_KEYS)}"
        )
    comps, rews = group["completions"], group["rewards"]
    if not isinstance(comps, list) or not isinstance(rews, list):
        raise ValueError("group: completions/rewards must be lists")
    if len(comps) != len(rews) or len(comps) == 0:
        raise ValueError(
            f"group: len(completions)={len(comps)} != len(rewards)={len(rews)} or empty"
        )
    for j, r in enumerate(rews):
        if isinstance(r, bool) or not isinstance(r, (int, float)):
            raise ValueError(f"group[{j}]: reward {r!r} is not numeric")


def group_is_full(group: Record, success_reward_threshold: float) -> bool:
    """A group is FULL (trainable) iff at least one rollout meets the threshold.

    Given plumbing: pure predicate over the container -- the skip-and-refill
    *orchestration* that calls this lives in :meth:`GroupCollector`.
    """
    validate_rollout_group(group)
    return max(float(r) for r in group["rewards"]) >= success_reward_threshold


# ---------------------------------------------------------------------------
# Group collector -- skip-and-refill bookkeeping (fields given, refill STUB)
# ---------------------------------------------------------------------------


class GroupCollector:
    """Bookkeeping container for the skip-and-refill rollout loop.

    The collector accumulates FULL groups (>= 1 rollout at or above
    ``success_reward_threshold``) until ``prompts_per_step`` are held; groups
    with zero correct rollouts are discarded and counted under
    :attr:`skipped`. The per-group decision and counters below are given
    plumbing; the *polling orchestration* that samples fresh prompts and
    refills the buffer is your TODO (see :meth:`collect_full_groups`).

    Attributes:
        num_rollouts: rollouts sampled per prompt (group size).
        prompts_per_step: full groups required before one optimizer step.
        success_reward_threshold: verifiable-reward cutoff for "correct".
        max_polls: hard bound on prompts examined per fill attempt, so an
            impossible task fails fast instead of hanging (None = unbounded;
            you decide the default in your implementation).
        skipped: counter of groups discarded for having zero correct
            rollouts (the README's ``skipped`` metric).
        groups: full (trainable) groups collected so far.
    """

    def __init__(
        self,
        num_rollouts: int,
        prompts_per_step: int,
        success_reward_threshold: float,
        max_polls: Optional[int] = None,
    ) -> None:
        if num_rollouts < 1 or prompts_per_step < 1:
            raise ValueError("num_rollouts and prompts_per_step must be >= 1")
        self.num_rollouts = num_rollouts
        self.prompts_per_step = prompts_per_step
        self.success_reward_threshold = float(success_reward_threshold)
        self.max_polls = max_polls
        self.skipped: int = 0
        self.groups: List[Record] = []

    def record_group(self, question: str, target: str, completions: List[str], rewards: List[float]) -> bool:
        """Validate + file one sampled group; returns True iff it was KEPT.

        Given plumbing: validates the container shape, then either appends the
        group to :attr:`groups` (>= 1 correct rollout) or discards it and
        increments :attr:`skipped` (zero correct rollouts). No refill logic
        here -- deciding to sample another prompt is :meth:`collect_full_groups`'s
        job.
        """
        group: Record = {
            "question": question,
            "target": target,
            "completions": list(completions),
            "rewards": [float(r) for r in rewards],
        }
        validate_rollout_group(group)
        if group_is_full(group, self.success_reward_threshold):
            self.groups.append(group)
            return True
        self.skipped += 1
        return False

    def collect_full_groups(
        self,
        generate_fn: Callable[[str, int], List[str]],
        sample_prompt_fn: Callable[[], Record],
    ) -> List[Record]:
        """Poll fresh prompts until ``prompts_per_step`` full groups are held.

        Contract:
          repeatedly ``sample_prompt_fn()`` -> {question, target, ...}, run
          ``generate_fn(question, num_rollouts)`` -> completions, score them
          with your verifier, and file each group via :meth:`record_group`;
          stop when ``len(self.groups) == prompts_per_step``. Discarded groups
          are already counted under :attr:`skipped` by ``record_group``.
          Honor ``max_polls`` as a hard bound across the whole call (raise
          RuntimeError when exceeded -- an impossible task must fail fast,
          not hang). Returns the full-group list.

        See Lab 07 for the group-rollout mechanics and README "Skip-and-refill"
        for the polling loop shape.
        """
        # TODO(Lab 10): implement the poll-until-full loop by hand -- the
        # skip accounting already lives in record_group; the refill is yours.
        raise NotImplementedError("Lab 10 TODO: GroupCollector.collect_full_groups")


# ---------------------------------------------------------------------------
# Verifier bridge (STUB) -- reuse/rewrite your Lab 07 string verifier
# ---------------------------------------------------------------------------


def spell_backward_reward(question: str, completion: str) -> float:
    """Verifiable reward for one spell_backward rollout.

    Contract:
      the question asks the model to spell a word backwards; return 1.0 iff
      the completion contains the reversed target word (case/whitespace
      normalized -- pick and document your normalization ONCE, apply it to
      every rollout in the group) else 0.0. Never fabricate partial credit:
      this lab's skip logic only makes sense with a hard 0/1 verifier.

      The Lab 07 answer is a fine starting point; rewrite it here from memory
      before comparing with ``policy_gradients/`` (Lab 07's compare pointer).
    """
    # TODO(Lab 10): implement the string-match verifier.
    raise NotImplementedError("Lab 10 TODO: spell_backward_reward")


# ---------------------------------------------------------------------------
# Demonstration-conditioned teacher reprompt (STUB)
# ---------------------------------------------------------------------------


def build_teacher_prompt(question: str, demo: str) -> str:
    """Build the teacher prompt that conditions the SAME model on a sibling demo.

    Contract (README "Demonstration-conditioned teacher"):
      return the string ``question + TEACHER_DEMO_HEADER + demo +
      TEACHER_TASK_SUFFIX`` assembled by hand (the constants above pin the
      exact surface form). The caller reruns the *same* model on this prompt
      to obtain teacher logits over exactly the student's completion tokens:
      the student's prompt stays bare while the teacher's prompt carries the
      correct sibling solution, so the two forwards share output-token
      alignment and the per-token KL is well-defined.

      Decide here (and document) how you select WHICH correct sibling to
      condition on when a group has several, and assert the demo itself met
      the threshold before splicing it in.
    """
    # TODO(Lab 10): implement. One line of assembly + the selection/assert
    # policy you documented -- write the policy down before coding it.
    raise NotImplementedError("Lab 10 TODO: build_teacher_prompt")


# ---------------------------------------------------------------------------
# Top-K reverse-KL distillation loss (ALL STUBS -- no KL arithmetic shipped)
# ---------------------------------------------------------------------------


class TopKReverseKL:
    """Hand-written SDPO loss: top-K reverse-KL with a tail-bucket closure.

    Memory story (why the API looks like this): the student logits
    ``[R, A, V]`` (R rollouts, A completion positions, V vocab) and their
    gradient dominate peak memory. :meth:`accumulate` splits the R rollouts
    into ``rollout_chunk``-sized groups and backpropagates each group before
    the next, so peak memory is one chunk, not the whole group. Because each
    chunk loss divides by the GLOBAL action-token count, the chunk gradients
    sum to exactly the full-group gradient.

    Shapes (all docstrings below use these):
        s_ids / t_ids : ``[R, P + A]`` student/teacher token ids (teacher
            prompt is longer by the spliced demo; the same A completion
            positions are scored in both).
        action_mask   : ``[R, A]`` 1.0 on completion tokens, 0.0 on padding.
        s_logits      : ``[R, A, V]`` (projected with ``logits_to_keep``).
        s_topk / idx  : ``[R, A, K]`` top-K logits and their vocab indices
            (K = ``kl_top_k``, taken from the STUDENT side so both sides
            gather the same support).
        s_logp / t_logp: ``[R, A, K]`` log-probabilities on that support.
        after :meth:`add_tail_bucket`: ``[R, A, K + 1]`` -- a valid
            log-distribution whose K+1 probabilities sum to 1.

    No KL arithmetic, top-K masking, or tail math is implemented here: that
    IS the lab. Compare with ``rlhf-book/code/distillation/loss.py``
    (SDPOLoss + add_tail) only after yours trains.
    """

    def __init__(self, kl_top_k: int, rollout_chunk: int = 4) -> None:
        if kl_top_k < 1 or rollout_chunk < 1:
            raise ValueError("kl_top_k and rollout_chunk must be >= 1")
        self.kl_top_k = kl_top_k
        self.rollout_chunk = rollout_chunk

    def add_tail_bucket(self, log_probs):  # noqa: ANN001 -- torch tensor at call time
        """Close a top-K log-distribution with a tail bucket (K+1 column).

        Contract:
          input ``[..., K]`` log-probabilities (natural log); output
          ``[..., K + 1]`` where the appended column is
          ``log(1 - sum(exp(log_probs)))`` -- the log of the TOTAL
          non-top-K probability mass, so the K+1 probabilities sum to 1 and
          both KL arguments are proper distributions. Watch the edge cases:
          the top-K mass can approach 1 (log of ~0 -> -inf) and the sum can
          round to exactly 1 in fp16/bf16 (log of 0 is undefined) -- the
          answer key clamps for a reason; figure out why before reading it.

        Args:
            log_probs: torch tensor ``[..., K]`` of log-probabilities.

        Returns:
            torch tensor ``[..., K + 1]``: valid log-distribution.
        """
        # TODO(Lab 10): implement the tail-bucket trick by hand.
        raise NotImplementedError("Lab 10 TODO: TopKReverseKL.add_tail_bucket")

    def _chunk_loss(self, model, batch: Dict[str, Any], sl: slice, A: int, denom):  # noqa: ANN001
        """Top-K reverse-KL for one rollout slice, normalized by the global token count.

        Contract (shapes above):
          1. student forward on ``batch['s_ids'][sl]`` -> ``[C, A, V]`` logits
             (C = chunk size); take the STUDENT's top-K logits and indices.
          2. teacher forward under ``torch.no_grad()`` on ``batch['t_ids'][sl]``;
             GATHER the teacher's log-probs at the student's indices (shared
             support -- this is the top-K masking logic you owe).
          3. close both with :meth:`add_tail_bucket` -> ``[C, A, K + 1]``.
          4. reverse KL = KL(student || teacher): the STUDENT is the KL
             *target* so the gradient flows through the student side while
             the teacher (KL input) stays detached. Think through which
             argument of the KL goes first BEFORE reading the answer key --
             getting the direction backwards is the classic bug.
          5. mask to action tokens and divide by ``denom`` (the global
             action-token count) so chunk losses sum to the full-group loss.

        Args:
            model: shared teacher/student model.
            batch: dict with ``s_ids`` ``[R, P+A]``, ``t_ids`` ``[R, P'+A]``,
                ``s_mask``/``t_mask`` and ``action_mask`` ``[R, A]``.
            sl: slice selecting this chunk's rollouts.
            A: number of completion (action) positions.
            denom: global action-token count to divide by (clamp >= 1).

        Returns:
            Scalar loss contribution for this chunk.
        """
        # TODO(Lab 10): implement steps 1-5 by hand. No F.kl_div shortcuts
        # until you can expand the reverse-KL formula on paper and say what
        # it reduces to here.
        raise NotImplementedError("Lab 10 TODO: TopKReverseKL._chunk_loss")

    def accumulate(self, model, batch: Dict[str, Any], scale: float = 1.0) -> float:  # noqa: ANN001
        """Accumulate SDPO gradients over rollout chunks, backward per chunk.

        Contract:
          loop ``range(0, R, self.rollout_chunk)``, compute each chunk's
          :meth:`_chunk_loss`, multiply by ``scale`` (e.g.
          ``1 / prompts_per_step`` for gradient accumulation across prompts),
          call ``backward()`` on finite chunks (skip non-finite ones and
          count them), and sum the detached values. Gradients ADD into the
          model parameters; this neither zeroes nor steps the optimizer --
          that is the training loop's job.

        Args:
            model: shared teacher/student model.
            batch: rollout batch (shapes in the class docstring).
            scale: factor applied before each chunk's backward.

        Returns:
            The scaled loss value summed over chunks, as a float.
        """
        # TODO(Lab 10): implement the chunked accumulate loop. Re-derive the
        # "dividing by the global denom makes chunk gradients sum to the
        # full-group gradient" argument in your own words first.
        raise NotImplementedError("Lab 10 TODO: TopKReverseKL.accumulate")


# ---------------------------------------------------------------------------
# Metrics-loop shape validation (given plumbing) + training loop (STUB)
# ---------------------------------------------------------------------------


def validate_metrics_record(record: Record) -> None:
    """Assert one JSONL metrics line has the metrics-contract shape.

    Raises ValueError unless the record:

    * is a dict containing exactly :data:`METRICS_RECORD_KEYS` at top level
    * has int ``step`` >= 0 and int ``skipped`` >= 0
    * has float-numeric ``reward`` / ``distill_loss`` / ``skipped_rate``
    """
    if not isinstance(record, dict):
        raise ValueError(f"metrics record: expected dict, got {type(record).__name__}")
    keys = tuple(sorted(record.keys()))
    if keys != tuple(sorted(METRICS_RECORD_KEYS)):
        raise ValueError(
            f"metrics record: keys {keys} != contract {sorted(METRICS_RECORD_KEYS)}"
        )
    for key in ("step", "skipped"):
        val = record[key]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise ValueError(f"metrics[{key!r}] must be an int >= 0")
    for key in ("reward", "distill_loss", "skipped_rate"):
        val = record[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"metrics[{key!r}] must be numeric")


def run_training_step(model, collector: GroupCollector, loss_fn: TopKReverseKL, cfg: Dict[str, Any], step: int) -> Record:  # noqa: ANN001
    """One SDPO step: fill groups, build teacher batch, accumulate loss, step optimizer.

    Contract (the loop skeleton you owe; README "Assignment checklist"):

    1. ``collector.collect_full_groups(...)`` until the step's groups are full
       (the ``skipped`` counter grows here when the model is still weak).
    2. For each full group: pick a correct sibling rollout, build the teacher
       prompt (:func:`build_teacher_prompt`), rerun the SAME model on it, and
       pack ``s_ids/t_ids/s_mask/t_mask/action_mask`` (shapes in the
       :class:`TopKReverseKL` docstring).
    3. ``loss_fn.accumulate(model, batch, scale=1 / prompts_per_step)`` per
       prompt-batch, then clip to ``max_norm`` and step the optimizer.
    4. Build ONE metrics record: ``step``, mean ``reward`` over the trained
       groups, ``distill_loss`` (the float :meth:`TopKReverseKL.accumulate`
       returned), ``skipped`` (groups discarded this step --
       ``collector.skipped`` delta), ``skipped_rate`` =
       ``skipped / (skipped + prompts_per_step)``; validate with
       :func:`validate_metrics_record` and append as one JSONL line to
       ``cfg['metrics_path']`` so the notebook plots from artifacts.

    Args:
        model: the shared student/teacher model.
        collector: group bookkeeping container (``skipped`` lives there).
        loss_fn: configured :class:`TopKReverseKL` instance.
        cfg: the lab config dict.
        step: 0-indexed optimizer step number.

    Returns:
        The validated metrics record for this step.
    """
    # TODO(Lab 10): implement steps 1-4. Everything it calls is a stub above;
    # this is the last piece -- finish the pieces first.
    raise NotImplementedError("Lab 10 TODO: run_training_step")


def run_training_loop(model, cfg: Dict[str, Any]) -> str:  # noqa: ANN001
    """Drive ``num_steps`` x :func:`run_training_step` with a fixed seed.

    Contract:
      seed everything from ``cfg['seed']`` (torch/numpy/random), construct the
      :class:`GroupCollector` and :class:`TopKReverseKL` from the config, run
      ``cfg['num_steps']`` calls to :func:`run_training_step`, and return the
      ``metrics_path`` the JSONL lines were appended to. Checkpoint/resume is
      NOT required in this lab (it was exercised in Labs 01/06/07) but a
      ``resume_from`` key is welcome.
    """
    # TODO(Lab 10): implement.
    raise NotImplementedError("Lab 10 TODO: run_training_loop")


# ---------------------------------------------------------------------------
# Worked-example fixtures (given, used by tests/notebook)
# ---------------------------------------------------------------------------

#: spell_backward words for the fabricated fixture groups (question asks to
#: spell the word backwards; ``target`` is the reversed word).
SPELL_BACKWARD_WORDS = ("apple", "banana", "cherry")

#: Fabricated group-level correctness pattern (num_rollouts=8 like the config):
#:   group 0: exactly 1 of 8 rollouts correct  -> full (trainable)
#:   group 1: 0 of 8 correct                   -> SKIPPED (skip-and-refill)
#:   group 2: 3 of 8 correct                   -> full (trainable, multi-demo)
#: which siblings are correct is pinned so teacher-selection policies have
#: something deterministic to be tested against.
FIXTURE_CORRECT_ROLLOUTS = ([4], [], [0, 3, 6])

FIXTURE_NUM_ROLLOUTS = 8


def make_fixture_groups() -> List[Record]:
    """Build three rollout groups with the known correctness pattern above.

    Pure fixture data with documented structure -- lets tests pin container
    and skip-bookkeeping invariants without any rollout/verifier logic
    existing yet.
    """
    groups: List[Record] = []
    for g, word in enumerate(SPELL_BACKWARD_WORDS):
        target = word[::-1]
        correct = set(FIXTURE_CORRECT_ROLLOUTS[g])
        completions = [
            (target if j in correct else f"wrong-guess-{g}-{j}") for j in range(FIXTURE_NUM_ROLLOUTS)
        ]
        rewards = [1.0 if j in correct else 0.0 for j in range(FIXTURE_NUM_ROLLOUTS)]
        groups.append(
            {
                "question": f"Spell the word '{word}' backwards.",
                "target": target,
                "completions": completions,
                "rewards": rewards,
            }
        )
    return groups


__all__ = [
    "FIXTURE_CORRECT_ROLLOUTS",
    "FIXTURE_NUM_ROLLOUTS",
    "METRICS_RECORD_KEYS",
    "ROLLOUT_GROUP_KEYS",
    "SPELL_BACKWARD_WORDS",
    "TEACHER_DEMO_HEADER",
    "TEACHER_TASK_SUFFIX",
    "GroupCollector",
    "TopKReverseKL",
    "build_teacher_prompt",
    "group_is_full",
    "load_config",
    "make_fixture_groups",
    "run_training_loop",
    "run_training_step",
    "spell_backward_reward",
    "validate_config",
    "validate_metrics_record",
    "validate_rollout_group",
]


if __name__ == "__main__":
    print("Lab 10 scaffold. Implement the TODOs; see README.md.")
    print("Run the structural tests first:")
    print("    python3 -m pytest tests/ -q")
