"""Lab 09 - Rejection Sampling -> SFT (scaffold).

Best-of-N data filtering on GSM8K, scored by YOUR Lab 03 Bradley-Terry RM:
generate N=8 rollouts per prompt, score them all, select (prompt, completion)
training pairs via four strategies -- each reward-based strategy paired with a
size-matched seeded random control -- SFT the base model on each subset, and
compare greedy exact-match accuracy across arms.

Prerequisites: Lecture 2 (RS section) + Chapter 9 of the RLHF Book
(https://rlhfbook.com/c/09-rejection-sampling.html) + your trained Lab 03 RM.

NO-SOLUTIONS SCAFFOLD: this file defines contracts only. Every function whose
body is a mechanism you are meant to implement (selection, scoring bridge,
exact-match evaluation, comparison table) is marked with TODO and raises
NotImplementedError. Implemented helpers below are pure plumbing: config
loading and record-shape validation. Do not peek at the answer key
(rlhf-book/code/rejection_sampling/) until your version trains.

Runs torch-free at import time: module-level dependencies are numpy + PyYAML +
stdlib, so the tests collect on a CPU-only machine.
"""

from __future__ import annotations

import math  # noqa: F401  (you may need it in your implementations)
import random  # noqa: F401  (seeded controls need it -- provided for your implementations)
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Types & constants (given)
# ---------------------------------------------------------------------------

Record = Dict[str, Any]
TrainingPair = Tuple[str, str]

#: One scored-prompt cache line has exactly these top-level keys
#: (see README "Cache contract"); completions[i] was scored as rewards[i].
CACHE_RECORD_KEYS = ("question", "answer", "completions", "rewards")

#: The four selection arms. Every top_* arm has exactly one matched random_*
#: control with identical sample budget and structural shape.
STRATEGIES = (
    "top_per_prompt",
    "random_per_prompt",
    "top_k_overall",
    "random_k_overall",
)

#: paired_control(strategy) -> the arm it must be compared against.
CONTROL_PAIRS = {
    "top_per_prompt": "random_per_prompt",
    "random_per_prompt": "top_per_prompt",
    "top_k_overall": "random_k_overall",
    "random_k_overall": "top_k_overall",
}

SELECT_TOP_PER_PROMPT = "top_per_prompt"
SELECT_RANDOM_PER_PROMPT = "random_per_prompt"
SELECT_TOP_K_OVERALL = "top_k_overall"
SELECT_RANDOM_K_OVERALL = "random_k_overall"


def paired_control(strategy: str) -> str:
    """Return the matched-control arm for ``strategy`` (raises for unknown)."""
    if strategy not in CONTROL_PAIRS:
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}"
        )
    return CONTROL_PAIRS[strategy]


# ---------------------------------------------------------------------------
# Config loading (given plumbing)
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the lab YAML into a plain dict.

    The dict is intentionally untyped (no pydantic dependency here); structural
    validation lives in :func:`validate_config`. Selection dispatch reads only
    ``cfg["selection"]["strategy"]`` and ``cfg["selection"]["top_k"]``.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Raise ValueError if ``cfg`` violates the lab's structural contract.

    Checks (all structural, no algorithm content):

    * required sections present: ``data``, ``scorer`` (with checkpoint path set
      before any real run), ``selection``
    * ``selection.strategy`` names one of :data:`STRATEGIES`
    * N = ``num_completions_per_prompt`` >= 1
    * nonnegative hyperparameters where negative values are meaningless
      (temperature, max_new_tokens, num_epochs, lr)
    """
    for section in ("data", "scorer", "selection"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"config missing section: {section!r}")
    sel = cfg["selection"]
    if sel.get("strategy") not in STRATEGIES:
        raise ValueError(f"selection.strategy must be one of {STRATEGIES}")
    n = cfg.get("num_completions_per_prompt")
    if not isinstance(n, int) or n < 1:
        raise ValueError("num_completions_per_prompt must be an int >= 1")
    for key in ("temperature", "max_new_tokens", "num_epochs", "lr"):
        val = cfg.get(key)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"config[{key!r}] must be numeric")
        if val < 0:
            raise ValueError(f"config[{key!r}] must be >= 0")


# ---------------------------------------------------------------------------
# Record-shape validation + rewards matrix (given plumbing)
# ---------------------------------------------------------------------------


def validate_rollout_records(records: Sequence[Record]) -> None:
    """Assert every scored-cache record has the cache-contract shape.

    Raises ValueError unless each record:

    * is a dict containing exactly :data:`CACHE_RECORD_KEYS` at top level
    * has ``len(completions) == len(rewards) > 0``
    * has float-numeric ``rewards`` entries

    Mirrors what ``diagnostics.load_rollouts`` would choke on downstream --
    catching malformed caches here, before any expensive stage runs.
    """
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"record {i}: expected dict, got {type(rec).__name__}")
        keys = tuple(sorted(rec.keys()))
        if keys != tuple(sorted(CACHE_RECORD_KEYS)):
            raise ValueError(
                f"record {i}: keys {keys} != cache contract {sorted(CACHE_RECORD_KEYS)}"
            )
        comps, rews = rec["completions"], rec["rewards"]
        if not isinstance(comps, list) or not isinstance(rews, list):
            raise ValueError(f"record {i}: completions/rewards must be lists")
        if len(comps) != len(rews) or len(comps) == 0:
            raise ValueError(
                f"record {i}: len(completions)={len(comps)} "
                f"!= len(rewards)={len(rews)} or empty"
            )
        for j, r in enumerate(rews):
            if isinstance(r, bool) or not isinstance(r, (int, float)):
                raise ValueError(f"record {i}[{j}]: reward {r!r} is not numeric")


def rewards_matrix(records: Sequence[Record]) -> np.ndarray:
    """Stack per-record rewards into an (M, N) float array.

    ``M = len(records)``, ``N = min length``; raises ValueError on ragged rows
    (records of differing completion counts cannot share one matrix). This is
    presentation glue only -- ordering/aggregation logic is yours to write.
    """
    validate_rollout_records(records)
    lengths = [len(r["rewards"]) for r in records]
    if len(set(lengths)) != 1:
        raise ValueError(f"ragged record lengths {set(lengths)}; no rectangular matrix")
    return np.asarray([list(map(float, r["rewards"])) for r in records], dtype=np.float64)


def load_scored_rollouts(path: str) -> List[Record]:
    """Load a Stage 1+2 JSONL cache into records.

    Each line is the JSON object described in the README cache contract;
    validate everything before returning. INVALIDATES nothing here: choosing
    the hash scheme that ties cache files to generation/scoring parameters is
    part of your implementation (README "Cache contract").
    """
    # TODO(Lab 09): json.loads per line, build records, call validate_rollout_records.
    raise NotImplementedError("Lab 09 TODO: load_scored_rollouts")


# ---------------------------------------------------------------------------
# Stage 2 -- scoring interface (STUB)
# ---------------------------------------------------------------------------


class RMScoreBridge:
    """Protocol wrapper around any scorer so arms don't care about backends.

    Your implementation should accept, minimally:

    * your **Lab 03** BT reward model (backbone + pooled ``Linear(hidden, 1)``
      head loaded from its checkpoint) -- the pedagogical scorer
    * optionally ``nvidia/AceMath-7B-RM`` (HF sequence-classification RM) as
      the production-RM comparison arm defined in the config's
      ``reward_model_name``

    Both are called through :meth:`score`, returning one scalar per
    (prompt, completion) pair, aligned with the input order.
    """

    def score(
        self,
        questions: List[str],
        completions: List[str],
        batch_size: int = 2,
    ) -> np.ndarray:
        """Score ``completions[i]`` in the context of ``questions[i]``.

        Contract:
          returns np.ndarray shape ``(len(questions),)`` of float scores,
          higher = better, aligned index-for-index with the inputs;
          must respect ``batch_size`` so the A40/T4 memory notes hold.
        """
        # TODO(Lab 09): implement. Load your Lab 03 checkpoint (or AceMath),
        # tokenize each pair, run the scorer forward pass, pool the head,
        # return the raw scalar rewards. Watch padding side / pooling index --
        # the exact bugs you debugged in Lab 03.
        raise NotImplementedError("Lab 09 TODO: RMScoreBridge.score")


# ---------------------------------------------------------------------------
# Stage 3a -- best-of-N selection strategies (ALL STUBS)
# ---------------------------------------------------------------------------

# Everything from here down either selects pairs or builds the comparison of
# selectors. Per the no-solutions rule these bodies contain NO selection /
# sampling / ranking logic: signature + docstring + shape comment + TODO.


def select_top_per_prompt(records: List[Record]) -> List[TrainingPair]:
    """Classic RS: argmax-reward completion for EVERY prompt.

    Expected output (shape comment only):
      list of M pairs, M = number of records with >=1 completion; each prompt
      contributes exactly one (question, completion) pair -- the one whose
      reward ranks first within its row. Ties broken by your own documented
      convention.

    See Chapter 9 @eq:rs_selection_per_prompt for the selection rule.
    """
    # TODO(Lab 09): implement by hand -- no itertools.argmax-style shortcuts
    # until you can say why they match the equation.
    raise NotImplementedError("Lab 09 TODO: select_top_per_prompt")


def select_random_per_prompt(records: List[Record], seed: int) -> List[TrainingPair]:
    """Matched control for :func:`select_top_per_prompt`.

    Contract: ONE uniformly random completion per prompt, drawn without using
    the rewards at all. Deterministic under ``seed`` (same seed == same
    draws). Same size and prompt coverage as the top arm by construction, so
    any accuracy gap isolates whether reward-based filtering beats a coin flip.
    """
    # TODO(Lab 09): implement with random.Random(seed) -- note the repo's
    # convention: draw indices per record in order, single shared RNG stream.
    raise NotImplementedError("Lab 09 TODO: select_random_per_prompt")


def select_top_k_overall(records: List[Record], k: int) -> List[TrainingPair]:
    """Top-k completions ranked across the ENTIRE M x N reward matrix.

    Expected output (shape comment only): k pairs (or fewer iff the pool is
    smaller than k), sorted however you like but deterministic given input
    order; unlike top_per_prompt, a prompt may contribute several completions
    and others none. This concentration is the mechanism behind the repo's key
    finding -- see README Motivation.
    """
    # TODO(Lab 09): flatten -> rank -> truncate. Decide (and document) how
    # equal-reward ties interact with input order.
    raise NotImplementedError("Lab 09 TODO: select_top_k_overall")


def select_random_k_overall(records: List[Record], k: int, seed: int) -> List[TrainingPair]:
    """Matched control for :func:`select_top_k_overall`.

    Contract: sample k pairs WITHOUT replacement from the flat M x N pool
    (every pair once), ignoring rewards entirely; sample size exactly
    ``min(k, M*N)``; deterministic under ``seed``; same K budget as the top
    arm so the comparison isolates ranking quality, not volume.
    """
    # TODO(Lab 09): rng.sample over the flattened pair space.
    raise NotImplementedError("Lab 09 TODO: select_random_k_overall")


def run_selection(
    records: List[Record],
    strategy: str,
    top_k: Optional[int] = None,
    seed: int = 0,
) -> List[TrainingPair]:
    """Dispatch to the selection function named by ``strategy``.

    Pure routing (implemented): validates ``strategy`` against
    :data:`STRATEGIES`, forwards ``top_k`` only to ``*_k_overall`` arms and
    ``seed`` only to ``random_*`` controls. The actual selection logic lives
    in the four stubs above.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown selection strategy: {strategy!r}; expected one of {STRATEGIES}")
    if strategy == SELECT_TOP_PER_PROMPT:
        return select_top_per_prompt(records)
    if strategy == SELECT_TOP_K_OVERALL:
        if top_k is None:
            raise ValueError(f"{strategy} requires top_k")
        return select_top_k_overall(records, top_k)
    if strategy == SELECT_RANDOM_PER_PROMPT:
        return select_random_per_prompt(records, seed)
    # SELECT_RANDOM_K_OVERALL
    if top_k is None:
        raise ValueError(f"{strategy} requires top_k")
    return select_random_k_overall(records, top_k, seed)


# ---------------------------------------------------------------------------
# Stage 3c -- GSM8K exact-match evaluation (STUBS)
# ---------------------------------------------------------------------------


def extract_gsm8k_answer(completion_text: str) -> Optional[str]:
    """Extract the predicted final numeric answer from a model completion.

    Contract:
      return the canonical string form of the final answer (digits, optional
      leading '-', optional commas stripped) or None when nothing parseable
      is found. GSM8K gold answers live after the '####' marker in the dataset
      but models emit answers in many formats -- decide your extraction rules
      BEFORE running eval, write them down, apply the SAME normalization to
      predictions and gold. The answer-key reference is
      ``rejection_sampling/utils.py::extract_gsm8k_answer`` (read AFTER yours).
    """
    # TODO(Lab 09): implement. No regex shortcuts borrowed beforehand.
    raise NotImplementedError("Lab 09 TODO: extract_gsm8k_answer")


def answers_match(predicted: Optional[str], gold: str) -> bool:
    """Exact-match predicate between extracted prediction and gold answer.

    Contract: True iff normalized forms are equal as NUMBERS where both parse
    numerically (so '1,024' matches '1024' and '72.0' matches '72') -- pick a
    convention, document it, never special-case test items.
    """
    # TODO(Lab 09): implement normalization + comparison.
    raise NotImplementedError("Lab 09 TODO: answers_match")


def evaluate_exact_match(
    generate_fn,  # Callable[[List[str]], List[str]] -- prompts -> completions
    questions: List[str],
    gold_answers: List[str],
) -> Dict[str, float]:
    """Greedy-decode evaluation over the held-out slice.

    Contract:
      calls ``generate_fn(questions)`` once per question batch (greedy per the
      config's eval_temperature: 0.0), extracts answers with
      :func:`extract_gsm8k_answer`, scores with :func:`answers_match`;
      returns {"accuracy": float in [0, 1], "n": int count evaluated}.
      Track and report the fraction of unparsable outputs separately -- it is
      itself a diagnostic (generation degeneration shows up there first).
    """
    # TODO(Lab 09): implement. questions/gold_answers must have equal length.
    raise NotImplementedError("Lab 09 TODO: evaluate_exact_match")


# ---------------------------------------------------------------------------
# Deliverable -- strategy-vs-control comparison table builder (STUB)
# ---------------------------------------------------------------------------


def build_comparison_table(run_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assemble the lab's headline deliverable: strategy-vs-random table.

    Input contract (one entry per completed SFT+eval arm):
      each dict carries at least ``strategy`` (a :data:`STRATEGIES` member),
      ``n_pairs`` (training pairs selected), and ``test_accuracy`` (exact-match
      on the same fixed test slice), plus whatever extra metrics you logged.

    Output contract:
      one row per STRATEGY arm in :data:`STRATEGIES` order with fields:
      ``strategy``, ``paired_control``, ``n_pairs``, ``test_accuracy``,
      ``control_accuracy``, ``delta_vs_control`` (= arm - control), and
      ``budget_matched`` (True iff the pair selected equal-size subsets).
      Raise ValueError for missing/duplicate/unknown arms rather than silently
      dropping rows -- an unfair table is worse than no table.
    """
    # TODO(Lab 09): implement. Pair via CONTROL_PAIRS; do NOT invent metrics.
    raise NotImplementedError("Lab 09 TODO: build_comparison_table")


# ---------------------------------------------------------------------------
# Worked-example fixtures (given, used by tests/notebook)
# ---------------------------------------------------------------------------

#: Chapter 9 worked-example reward matrix (M=5 prompts x N=4 completions).
#: Known ordering properties (asserted by the tests):
#:   row argmaxes  -> [0, 1, 0, 2, 3]   (what top_per_prompt would keep)
#:   flat top-2    -> Q3/c1 (0.9), Q2/c2 (0.8)   (both land on distinct rows)
CHAPTER_REWARD_MATRIX: List[List[float]] = [
    [0.7, 0.3, 0.5, 0.2],
    [0.4, 0.8, 0.6, 0.5],
    [0.9, 0.3, 0.4, 0.7],
    [0.2, 0.5, 0.8, 0.6],
    [0.5, 0.4, 0.3, 0.6],
]

GOLD_ANSWERS: List[str] = ["7", "12", "45", "18", "99"]


def make_fixture_records() -> List[Record]:
    """Build scored-rollout records from :data:`CHAPTER_REWARD_MATRIX`.

    Pure fixture data with known ordering -- lets tests pin container
    invariants without any selection logic existing yet.
    """
    records: List[Record] = []
    for i, row in enumerate(CHAPTER_REWARD_MATRIX):
        records.append(
            {
                "question": f"Q{i + 1}",
                "answer": GOLD_ANSWERS[i],
                "completions": [f"y_{i + 1},{j + 1}" for j in range(len(row))],
                "rewards": list(row),
            }
        )
    return records


__all__ = [
    "CACHE_RECORD_KEYS",
    "CONTROL_PAIRS",
    "SELECT_RANDOM_K_OVERALL",
    "SELECT_RANDOM_PER_PROMPT",
    "SELECT_TOP_K_OVERALL",
    "SELECT_TOP_PER_PROMPT",
    "STRATEGIES",
    "RMScoreBridge",
    "answers_match",
    "build_comparison_table",
    "evaluate_exact_match",
    "extract_gsm8k_answer",
    "load_config",
    "load_scored_rollouts",
    "make_fixture_records",
    "paired_control",
    "rewards_matrix",
    "run_selection",
    "select_random_k_overall",
    "select_random_per_prompt",
    "select_top_k_overall",
    "select_top_per_prompt",
    "validate_config",
    "validate_rollout_records",
]


if __name__ == "__main__":
    print("Lab 09 scaffold. Implement the TODOs; see README.md.")
    print("Run the structural tests first:")
    print("    python3 -m pytest tests/ -q")
