"""Lab 11 — Evaluation & LLM-as-judge: student starter code.

A mini-eval suite over ALL of your saved checkpoints (base / SFT / DPO / RL):

    fixed prompt set -> candidate answers per checkpoint
        -> pairwise win-rate matrix scored by (a) your Lab 03 RM,
           (b) a local LLM judge (Qwen/Qwen3-4B)
    -> judge-bias audit: position / length / self-preference bias measured
       via ORDER-SWAP flip rates
    -> judge verdicts vs verifiable ground truth (GSM8K exact match)
    -> extension (absorbs old Lab 12): synthetic preference data
       generate -> judge-filter -> measure the bias you introduced

NO-SOLUTIONS RULE: every function below is a signature + docstring contract
and raises NotImplementedError. No win-rate or bias arithmetic is provided —
you derive it, one function at a time, against the docstring contracts.

Methodology reference (open ONLY after your own version works):

    rlhf-book/code/rejection_sampling/diagnostics.py
    (reward-vs-correctness diagnostic: histograms, per-row win-rate vs a
    random baseline, best-of-N sweep, decidable_fraction framing — the same
    "measure against a baseline, report headroom separately" discipline you
    should reuse for every matrix and bias number in this lab)

Environment note: this module imports cleanly WITHOUT torch installed (the
torch import is guarded), so the local structural tests run on a bare
interpreter with only numpy present. No network access is needed for tests —
all judge outputs in the tests are FABRICATED fixtures with known flip
patterns, never live model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:  # torch stays optional at import time so the scaffold is inspectable anywhere
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - hit on machines without torch
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# =============================================================================
# Config (provided — mirrors configs/11_eval_llm_judge.yaml)
# =============================================================================


@dataclass
class EvalConfig:
    """Configuration for Lab 11. Kept in sync with configs/11_eval_llm_judge.yaml."""

    # --- checkpoints under evaluation (your Labs 00-04/06 artifacts) ---
    # label -> path to a saved model / checkpoint dir. Include the BASE model
    # and every post-training stage you ran: base, sft, dpo, (rl if Lab 06-07).
    checkpoint_paths: Dict[str, str] = field(default_factory=dict)
    # Path to your Lab 03 Bradley-Terry reward model (judge arm "rm").
    rm_checkpoint_path: Optional[str] = None

    # --- judge arm ---
    judge_model_id: str = "Qwen/Qwen3-4B"       # Tier B; runs locally, no API dependency
    judge_max_new_tokens: int = 256
    judge_temperature: float = 0.0              # deterministic judging for bias audit
    judge_prompt_style: str = "pairwise"        # "pairwise" | "pointwise"
    # API judge (any frontier model) for the bias-comparison arm; None = skip.
    api_judge_name: Optional[str] = None

    # --- prompt set + generation ---
    prompt_set_path: Optional[str] = None       # fixed JSONL prompt set (NEVER regenerated mid-suite)
    verifiable_prompts_path: Optional[str] = None  # subset with ground-truth answers (e.g. GSM8K)
    samples_per_checkpoint: int = 1             # greedy single answer per prompt by default
    max_new_tokens: int = 512
    max_prompt_length: int = 1024

    # --- order-swap / bias audit ---
    swap_orders: bool = True                    # judge every pair in BOTH positions
    n_order_swaps: int = 1                      # extra shuffled repeats beyond the strict swap
    length_gap_tokens_threshold: int = 200      # "long vs short" bucket boundary for length bias
    self_preference_checkpoint: Optional[str] = None  # e.g. "sft" when the judge shares its lineage

    # --- synthetic preference extension (absorbs old Lab 12) ---
    synthetic_teacher_model_id: str = "Qwen/Qwen3-4B"
    n_synthetic_prompts: int = 200
    synthetic_filter_min_margin: float = 0.5    # judge-margin floor for kept pairs
    synthetic_output_path: Optional[str] = None

    seed: int = 42
    output_dir: str = "runs/lab11_eval_judge"


# =============================================================================
# Part 1 — Fixed prompt set + head-to-head pairing
# =============================================================================


def load_prompt_set(path: str) -> List[Dict[str, Any]]:
    """Load the FIXED evaluation prompt set from a JSONL file.

    TODO(Lab11): implement.

    Contract
    --------
    Reads one JSON object per line; every row minimally carries:
        prompt_id   str   stable unique id (used across ALL checkpoints/runs)
        prompt      str   the user prompt text
        category    str   e.g. "gsm8k_verifiable" | "open_ended" | "instruction_following"
    Optional fields you may add: "ground_truth" (for verifiable rows), notes.

    Validation (raise ValueError, never silently repair):
      - duplicate prompt_ids are an error;
      - rows missing any required key are an error.

    Returns a list of row dicts in file order. FREEZE the file once written:
    the win-rate matrix is only comparable across checkpoints evaluated on
    the identical prompt set — say in your notebook what breaks if you edit
    prompts between eval runs.
    """
    raise NotImplementedError("TODO(Lab11): implement load_prompt_set")


def build_head_to_head_pairs(
    answers_by_checkpoint: Mapping[str, Sequence[Mapping[str, Any]]],
    max_pairs_per_checkpoint_pair: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Enumerate the pairwise comparison units for the win-rate matrix.

    TODO(Lab11): implement.

    Contract
    --------
    answers_by_checkpoint
        checkpoint label -> per-prompt answers. Every answer row minimally
        carries {"prompt_id": str, "completion": str, "completion_token_len": int}.
        All labels must be evaluated on the SAME prompt set (verify prompt_id
        coverage matches across labels; raise ValueError on a mismatch).

    Returns one dict per (prompt, checkpoint_a, checkpoint_b) with a < b by
    a documented, deterministic ordering (e.g. sorted label):
        {"prompt_id", "model_a", "model_b",
         "completion_a", "completion_b",
         "len_a", "len_b"}          # token lengths, for the length-bias audit

    max_pairs_per_checkpoint_pair caps comparisons per (model_a, model_b)
    cell (deterministic subsample seeded from config.seed — never random
    without a seed). Judge COST scales as O(#checkpoints^2 x #prompts);
    write down your cell count before launching anything.
    """
    raise NotImplementedError("TODO(Lab11): implement build_head_to_head_pairs")


# =============================================================================
# Part 2 — Judges (two arms) and the checkpoint eval suite
# =============================================================================


def rm_judge_verdict(
    rm: Any,
    prompt: str,
    completion_a: str,
    completion_b: str,
) -> Dict[str, Any]:
    """Score one pair with YOUR Lab 03 Bradley-Terry reward model.

    TODO(Lab11): implement.

    Contract
    --------
    Score each completion independently (context = prompt + completion),
    then decide a verdict from the two scalar rewards.

    Returns a dict:
        {"verdict": "a" | "b" | "tie",
         "score_a": float, "score_b": float,
         "margin": float}           # signed, documented convention

    Decide and DOCUMENT the tie band (|margin| < epsilon for which epsilon?).
    RM scores are only meaningful RELATIVELY (Lab 03 lesson: scale is
    arbitrary, margins decide) — explain why cross-checkpoint score averages
    are not comparable but margins within a pair are.
    """
    raise NotImplementedError("TODO(Lab11): implement rm_judge_verdict")


def llm_judge_verdict(
    judge_model: Any,
    tokenizer: Any,
    prompt: str,
    completion_a: str,
    completion_b: str,
    config: EvalConfig,
) -> Dict[str, Any]:
    """Score one pair with the LLM judge (Qwen/Qwen3-4B) via a chat prompt.

    TODO(Lab11): implement.

    Contract
    --------
    Build a JUDGE PROMPT that presents the prompt + the two completions and
    forces a parseable output format (e.g. final line "ANSWER: A|B|TIE").
    Parse the verdict out of the generation; a generation you cannot parse
    is NOT a tie — return {"verdict": "unparseable", ...} and count it
    separately (report the unparseable rate in every results table).

    Anti-sycophancy notes you must test, not assume:
      - judge_temperature=0 for the main matrix (determinism);
      - randomize WHICH completion is shown as A vs B per comparison
        (this is the position-bias control — see Part 3);
      - strip/normalize any formatting differences between completions
        before judging (markdown-only differences fool judges).

    Returns a dict:
        {"verdict": "a" | "b" | "tie" | "unparseable",
         "raw_judge_output": str,
         "order_shown": "ab" | "ba"}   # which completion was displayed as A
    """
    raise NotImplementedError("TODO(Lab11): implement llm_judge_verdict")


def discover_checkpoints(runs_root: str) -> Dict[str, str]:
    """Find every saved checkpoint from Labs 00-04/06 under your runs root.

    TODO(Lab11): implement.

    Contract
    --------
    Walk ``runs_root`` (your per-lab output dirs), map a human label ->
    checkpoint path for every loadable model, e.g.:
        {"base": ".../base", "sft": ".../lab01_sft/final", "dpo": ".../lab04_dpo/final", ...}
    Labels must be stable across runs (they become matrix row/column names).
    Raise FileNotFoundError if runs_root itself is missing; skip + WARN on
    unreadable subdirs rather than failing the whole suite.
    """
    raise NotImplementedError("TODO(Lab11): implement discover_checkpoints")


def run_checkpoint_suite(config: EvalConfig) -> Any:
    """End-to-end eval entry point: generate -> judge -> matrix -> bias audit.

    TODO(Lab11): implement.

    Required behavior (check off in README as you go):
      1. load the frozen prompt set (``load_prompt_set``);
      2. for every checkpoint in ``config.checkpoint_paths``, generate one
         answer per prompt (greedy, seeded) and store per-prompt outputs
         under ``config.output_dir`` as JSONL — one file per checkpoint;
      3. judge every head-to-head pair with BOTH arms — RM
         (``rm_judge_verdict``) and LLM judge (``llm_judge_verdict``) —
         honoring ``config.swap_orders``;
      4. compute the two win-rate matrices (``compute_win_rate_matrix``);
      5. run the bias audit (``order_swap_flip_rate`` + the three bias
         reports) and the ground-truth comparison
         (``verdict_vs_ground_truth``);
      6. append ONE JSON summary line per stage to
         ``config.output_dir/metrics.jsonl`` — notebooks plot from these
         artifacts, not from wandb.

    Smoke-run FIRST: 8 prompts x 2 checkpoints x both arms before anything
    larger. Report the wall-clock and token cost of the full matrix in your
    notebook BEFORE launching it.
    """
    raise NotImplementedError("TODO(Lab11): implement run_checkpoint_suite")


# =============================================================================
# Part 3 — Win-rate matrix + judge-bias audit (order-swap flip rates)
# =============================================================================


def compute_win_rate_matrix(
    verdicts: Sequence[Mapping[str, Any]],
    checkpoint_labels: Sequence[str],
) -> Dict[str, Any]:
    """Turn a flat list of pair verdicts into a win-rate matrix.

    TODO(Lab11): implement — NO shortcut helpers; derive the aggregation.

    Contract
    --------
    verdicts
        One record per judged comparison, minimally:
        {"model_a", "model_b",          # checkpoint labels in THIS record
         "verdict": "a"|"b"|"tie",      # possibly order-swapped; see below
         "order_shown": "ab"|"ba"}      # display order used by the judge
        Records whose verdict is "unparseable" are excluded from the matrix
        but must be COUNTED (returned separately).

    Returns
    -------
    {"labels": List[str],                       # sorted, matrix is [i][j]
     "win_rates": List[List[float]],            # P(i beats j) from i's row view
     "pair_counts": List[List[int]],            # n judged comparisons per cell
     "n_unparseable": int,
     "n_ties": int}

    Conventions to decide and DOCUMENT (then keep fixed across both arms):
      - the diagonal (i vs i): excluded? 0.5? pick one and state why;
      - ties: 0.5 credit, or excluded from the denominator? State it;
      - order-swap records: a verdict shown as "ba" must be UN-flipped
        before aggregation — i.e. reported in the (model_a, model_b) frame
        where model_a < model_b by the same ordering as
        ``build_head_to_head_pairs``. A swap bug here silently mirrors the
        matrix; verify your un-flipping with the fabricated fixtures in
        tests/test_11_eval_llm_judge.py.

    win_rates[i][j] must satisfy: for i != j with pair_counts>0,
    win_rates[i][j] + win_rates[j][i] == 1.0 exactly (given the tie
    convention). Assert this in your notebook on real data.
    """
    raise NotImplementedError("TODO(Lab11): implement compute_win_rate_matrix")


def order_swap_flip_rate(
    verdicts_first_order: Sequence[Mapping[str, Any]],
    verdicts_swapped_order: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Measure how often the judge's verdict FLIPS when A/B presentation order swaps.

    TODO(Lab11): implement — THE core bias measurement of this lab.

    Contract
    --------
    Two parallel verdict lists for the SAME comparisons: list 1 judged with
    order "ab", list 2 the identical pairs with order "ba" (aligned by
    index — validate the alignment by (model_a, model_b, prompt_id) and
    raise ValueError on any mismatch).

    A "flip" = the verdict strictly favors the SAME COMPLETION both times is
    a non-flip; favoring the OTHER completion is a flip; ties/unparseable
    need their own documented buckets (a tie in one order and a preference
    in the other: flip or not? decide + document).

    Returns
    -------
    {"n_compared": int,
     "n_flips": int,
     "flip_rate": float,                  # n_flips / n_compared
     "flips_toward_first": int,          # position-bias direction split:
     "flips_toward_second": int,         #   which position won the flipped verdict?
     "excluded_tie_conflicts": int}

    Position bias exists in proportion to flip_rate. A well-behaved judge
    still flips ~some% (LLMs are noisy) — the number matters only relative
    to the verdict margin: also note HOW MANY flips happen on pairs the RM
    scores as near-ties (report, don't average away).
    """
    raise NotImplementedError("TODO(Lab11): implement order_swap_flip_rate")


def position_bias_report(
    verdicts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Quantify first-position preference across the whole judging run.

    TODO(Lab11): implement.

    Contract
    --------
    Using the order_shown field, compute P(verdict favors the FIRST-displayed
    completion) over all decisive (a/b) verdicts, plus the same rate per
    prompt category and per model pair.

    Returns
    -------
    {"p_first_position": float,
     "n_decisive": int,
     "p_first_by_model_pair": {pair_key: float},
     "verdict_rate_notation": str}       # one line stating your exact convention

    A symmetric judge gives p_first_position == 0.5. State what a 0.6 rate
    means for every headline win-rate number computed from this judge.
    """
    raise NotImplementedError("TODO(Lab11): implement position_bias_report")


def length_bias_report(
    verdicts: Sequence[Mapping[str, Any]],
    length_gap_threshold: int = 200,
) -> Dict[str, Any]:
    """Do judges prefer LONGER completions, holding quality fixed?

    TODO(Lab11): implement.

    Contract
    --------
    verdicts additionally carry "len_a"/"len_b" (completion token lengths).
    Bucket pairs by |len_a - len_b| (<= threshold vs > threshold — document
    your bucketing); within the gap>threshold bucket compute P(verdict
    favors the longer completion). Compare against the same rate on
    near-equal-length pairs.

    Returns
    -------
    {"p_longer_bucketed": {"small_gap": float, "large_gap": float},
     "counts": {"small_gap": int, "large_gap": int},
     "length_gap_threshold": int}

    Interpretation you must write out: if the RM and the LLM judge DISAGREE
    on the length preference direction, which one is Confounder-length and
    which one is quality? (Hint: Ch. 16 — preference-data biases.)
    """
    raise NotImplementedError("TODO(Lab11): implement length_bias_report")


def self_preference_bias_report(
    verdicts: Sequence[Mapping[str, Any]],
    judge_checkpoint_label: str,
) -> Dict[str, Any]:
    """Does the judge favor candidates from its own model lineage?

    TODO(Lab11): implement.

    Contract
    --------
    judge_checkpoint_label
        e.g. "sft" when the judge (Qwen3-4B fine-tune) shares lineage with
        one of the evaluated checkpoints (or "rm" for the RM arm scoring its
        own training distribution — Lab 03's UltraFeedback pairs).

    Compute P(judge favors candidates generated by its own lineage) vs
    P(judge favors others), restricted to comparisons where the lineage
    candidate actually participates.

    Returns
    -------
    {"p_self_win": float,
     "p_other_win": float,
     "n_self_pairs": int,
     "delta": float}                     # p_self_win - p_other_win (sign matters)

    An honest judge gives delta ~ 0. A large positive delta means the
    win-rate matrix is partially a mirror of the judge, not of quality —
    quantify how much of each row's spread this could explain.
    """
    raise NotImplementedError("TODO(Lab11): implement self_preference_bias_report")


# =============================================================================
# Part 4 — Verdicts vs verifiable ground truth
# =============================================================================


def verdict_vs_ground_truth(
    verdicts: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare judge verdicts to exact-match ground truth on verifiable prompts.

    TODO(Lab11): implement.

    Contract
    --------
    ground_truth
        prompt_id -> {"ground_truth": str, "extract_answer_a": str,
                      "extract_answer_b": str}  (GSM8K-style exact match;
        reuse YOUR Lab 06 exact-match verifier for the extraction/matching
        — do not re-implement it here).

    For every verifiable comparison, derive the CORRECT verdict from the
    ground truth (a correct, b correct, both, neither) and compare with the
    judge's verdict. "Both correct"/"neither correct" pairs have no correct
    verdict — exclude them and report the count (this is the
    ``decidable_fraction`` framing from rejection_sampling/diagnostics.py:
    measure headroom separately from judge skill).

    Returns
    -------
    {"n_comparisons": int,
     "n_decidable": int,
     "decidable_fraction": float,
     "judge_agreement_on_decidable": float,   # P(judge verdict == truth verdict)
     "agreement_by_judge": {"rm": float, "llm": float},  # when both arms present
     "confusions": {"truth_a_judge_b": int, "truth_b_judge_a": int,
                    "truth_decided_judge_tie": int}}

    Decidability is a property of the PROMPT SET, not the judge — report it
    once and carry it into every interpretation of the win-rate matrix.
    """
    raise NotImplementedError("TODO(Lab11): implement verdict_vs_ground_truth")


# =============================================================================
# Part 5 — Extension: synthetic preferences (absorbs old Lab 12)
# =============================================================================


def generate_synthetic_preferences(
    teacher_model: Any,
    tokenizer: Any,
    prompt_set: Sequence[Mapping[str, Any]],
    config: EvalConfig,
) -> List[Dict[str, Any]]:
    """Generate candidate preference pairs from a teacher model.

    TODO(Lab11): implement (extension arm).

    Contract
    --------
    For each prompt: sample K>=2 completions from the teacher at a
    temperature you document, form pairs, and attach teacher-reasoning-free
    metadata {"prompt_id", "completion_a", "completion_b", "temperature",
    "sample_ids"}. Deliberately INCLUDE a quality spread (e.g. one greedy
    candidate + one high-temperature candidate) — a filtered set of
    near-identical candidates teaches nothing about the filter.

    Returns the raw (UNFILTERED) pair list. Filtering is the next function's
    job — keep the stages separable so the filter's effect is measurable.
    """
    raise NotImplementedError("TODO(Lab11): implement generate_synthetic_preferences")


def judge_filter_preferences(
    raw_pairs: Sequence[Mapping[str, Any]],
    judge_fn: Any,
    config: EvalConfig,
) -> Dict[str, Any]:
    """Judge -> filter raw synthetic pairs into a training preference set.

    TODO(Lab11): implement (extension arm).

    Contract
    --------
    judge_fn
        One of your Part-2 judge functions (arm is a config choice — study
        BOTH: filtering by the same judge you will audit is self-preference
        risk made concrete).

    Keep a pair iff the judge's verdict is decisive AND its confidence
    (margin / parseable strength — your documented convention) clears
    ``config.synthetic_filter_min_margin``. Record the REJECTED pairs too,
    with a reject_reason — the discarded mass is where filter bias hides.

    Returns
    -------
    {"kept": List[rows with verdict + margin + order_shown],
     "rejected": List[rows with reject_reason],
     "keep_rate": float}

    THIS is the generate -> judge -> filter loop old Lab 12 existed for.
    """
    raise NotImplementedError("TODO(Lab11): implement judge_filter_preferences")


def measure_introduced_bias(
    filtered: Mapping[str, Any],
    raw_pairs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Measure what bias the generate->judge->filter loop INTRODUCED.

    TODO(Lab11): implement (extension arm).

    Contract
    --------
    Compare kept vs raw pairs along the axes this lab audited for judges:
      - length preference: P(kept pair's chosen side is the longer one) vs
        the same rate over ALL raw pairs (not just judged-kept ones);
      - position leakage: any statistical asymmetry between completion_a
        and completion_b slots that survives into the kept set (the filter
        must be order-invariant — prove it or find the leak);
      - verdict-margin distribution shift: kept pairs' margins vs the raw
        judged margins (selection on margin is fine; selection on LENGTH
        correlated with margin is the failure mode).

    Returns a flat dict of the three contrasts above, each with its counts.
    Interpretation to write in your notebook: if you trained a DPO model on
    these synthetic pairs (Lab 04 loss), which bias would the trained
    policy inherit — and how would you detect it with THIS lab's own
    win-rate + order-swap tooling?
    """
    raise NotImplementedError("TODO(Lab11): implement measure_introduced_bias")
