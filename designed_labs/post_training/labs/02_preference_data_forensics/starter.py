"""Lab 02 — Preference-data forensics: student starter code.

You audit >=3k pairs of `argilla/ultrafeedback-binarized-preferences-cleaned`
BEFORE any reward modeling: schema mapping under real field names, length-bias
statistics, near-tie detection, a manual noise audit, formatting-artifact
analysis, a cleaned-subset builder with explicit filter rules + removal report,
and an 8-gram decontamination check of prompts against the GSM8K test set.

NO-SOLUTIONS RULE: every function below is a signature + docstring contract
and raises NotImplementedError. Implement them one at a time against the
contracts in the docstrings. Reference implementation (open ONLY after your
own audit pipeline works — it shows pair filtering under real field names):

    rlhf-book/code/reward_models/train_preference_rm.py:84-190
    (build_preference_dataset — schema handling, unusable-row skipping,
     token-identical pair dropping)

Environment note: this module imports cleanly with the STANDARD LIBRARY ONLY —
no pandas, numpy, torch, datasets, or network access is required to import it
or to run the structural tests. Write your analysis in pure python first;
reach for pandas only in the notebook if your runtime has it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


# =============================================================================
# Config (provided — mirrors configs/02_preference_forensics.yaml)
# =============================================================================


@dataclass
class ForensicsConfig:
    """Configuration for Lab 02. Kept in sync with configs/02_preference_forensics.yaml."""

    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned"
    split: str = "train"
    limit: int = 3000                  # audit at least this many pairs
    source_field: str = "source"       # per-instruction source in UltraFeedback
    seed: int = 42
    # Manual noise audit
    audit_sample_size: int = 100
    # Near-tie detection: char n-gram Jaccard similarity of chosen vs rejected
    near_tie_ngram_size: int = 4       # char-level n for similarity
    near_tie_similarity_threshold: float = 0.85
    # Formatting-artifact markers (list/markdown win rates)
    formatting_patterns: List[str] = field(
        default_factory=lambda: ["- ", "* ", "1. ", "##", "**", "```"]
    )
    # Decontamination vs GSM8K test prompts: word-level 8-grams
    ngram_n: int = 8
    min_shared_ngrams: int = 1         # flag a pair if it shares >= this many
    gsm8k_prompts_path: Optional[str] = None   # local JSONL of GSM8K test prompts
    output_dir: str = "outputs/lab02_preference_forensics"


# =============================================================================
# 1 · Loading + schema mapping under REAL field names
# =============================================================================


def normalize_pair(raw: Mapping[str, Any], source_field: str = "source") -> Optional[Dict[str, str]]:
    """Map one raw dataset row to the canonical pair schema.

    Contract:
    - Input rows come from `argilla/ultrafeedback-binarized-preferences-cleaned`
      (real field names: `prompt`, `chosen`, `rejected`, plus a per-instruction
      source field when present).
    - `chosen`/`rejected` may arrive EITHER as a plain string OR as a
      conversation list of {"role": ..., "content": ...} messages (the repo's
      build_preference_dataset handles both; so must you).
    - Returns {"prompt": str, "chosen": str, "rejected": str, "source": str}.
      When the source field is absent use the literal string "unknown".
      For conversation format, concatenate the assistant-turn contents with
      "\\n\\n" in order.
    - Return None for rows that cannot yield a usable pair (missing/empty
      prompt or either response) — never raise on messy rows.
    """
    raise NotImplementedError


def schema_report(raw_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize the REAL schema of a batch of raw rows.

    Contract:
    - Returns a dict with at least:
        "total":            number of input rows
        "usable":           count normalize_pair would accept
        "unusable":         total - usable
        "chosen_formats":   {"str": n, "messages": n, "other": n} counts
        "missing_fields":   {field_name: count} for expected fields absent/empty
    - Pure inspection: no row is mutated.
    """
    raise NotImplementedError


def load_pairs_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load an already-normalized pairs file (one JSON object per line).

    Contract:
    - Each line is a JSON object with at least "prompt", "chosen", "rejected".
    - Returns a list of dicts in file order; skips blank lines; raises
      FileNotFoundError for a missing path.
    - Used for locally exported subsets (GSM8K prompts, Tülu-3 extension)
      so the lab runs CPU-only with NO network.
    """
    raise NotImplementedError


# =============================================================================
# 2 · Length-bias statistics + per-source breakdown
# =============================================================================


def length_bias_stats(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Length-bias statistics over normalized pairs.

    Contract:
    - Input: output of normalize_pair (dicts with "chosen"/"rejected" strings).
    - Returns a dict with at least:
        "n_pairs":               int
        "p_chosen_longer_chars": fraction of pairs where len(chosen) > len(rejected)
                                 (strictly greater; ties count AGAINST the bias)
        "p_chosen_longer_words": same on whitespace-token counts
        "mean_chars_chosen" / "mean_chars_rejected": float means
        "median_chars_chosen" / "median_chars_rejected": float medians
    - All fractions in [0.0, 1.0]; empty input returns n_pairs == 0 and every
      fraction as 0.0 (never divide by zero).
    """
    raise NotImplementedError


def per_source_breakdown(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-source version of the length-bias stats.

    Contract:
    - Group pairs by their "source" value (rows from normalize_pair always
      carry one; "unknown" is a valid group).
    - Returns {source: {"n_pairs": int, "p_chosen_longer_chars": float,
                        "mean_chars_chosen": float, "mean_chars_rejected": float}}.
    - Sources may appear in any order; every input pair lands in exactly one
      group.
    """
    raise NotImplementedError


# =============================================================================
# 3 · Near-tie detection (the pairs an RM can barely learn from)
# =============================================================================


def detect_near_ties(
    pairs: Sequence[Mapping[str, Any]],
    ngram_size: int = 4,
    similarity_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Find pairs whose two responses are nearly interchangeable.

    Contract:
    - Similarity metric (implement it yourself — no library similarity
      shortcuts from difflib or anywhere else): character n-gram Jaccard between
      chosen and rejected, i.e.
          |G(chosen) & G(rejected)| / |G(chosen) | G(rejected)|
      where G(text) is the set of case-sensitive character n-grams of length
      `ngram_size`; define Jaccard as 1.0 when both texts are identical and
      0.0 when both n-gram sets are empty.
    - Returns one record per flagged pair:
        {"index": int, "reason": str, "similarity": float}
      where "reason" is "identical" when chosen == rejected verbatim, else
      "near_identical" when similarity >= similarity_threshold.
    - Records appear in increasing "index" order; a pair is flagged at most
      once; indices index into `pairs` as given.
    """
    raise NotImplementedError


# =============================================================================
# 4 · Manual noise audit (100 pairs vs your own judgment)
# =============================================================================


def sample_manual_audit(
    pairs: Sequence[Mapping[str, Any]],
    n: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build the deterministic worksheet for the manual noise audit.

    Contract:
    - Sample `n` distinct indices from `pairs` using `random.Random(seed)` —
      same seed and same pairs MUST give the same worksheet.
    - Returns one record per sampled pair:
        {"index": int, "prompt": str, "chosen": str, "rejected": str,
         "human_verdict": None}
    - `human_verdict` stays None; you fill it in by hand with one of
      "chosen_better" / "tie" / "rejected_better".
    - n is capped at len(pairs); n == 0 returns [].
    """
    raise NotImplementedError


def audit_agreement(audited: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score a filled-in manual-audit worksheet against the dataset's labels.

    Contract:
    - Input: records from sample_manual_audit with "human_verdict" filled in
      ("chosen_better" | "tie" | "rejected_better").
    - Returns:
        {"n_audited": int,
         "verdict_counts": {"chosen_better": n, "tie": n, "rejected_better": n},
         "tie_rate": float,
         "n_missing_verdict": int}   # records you did not judge
    - "tie" means the dataset's chosen/rejected label did not match your
      discrimination; report tie_rate = ties / n_audited (0.0 when empty).
    - Missing/None verdicts never crash the function; they are only counted.
    """
    raise NotImplementedError


# =============================================================================
# 5 · Formatting-artifact analysis (lists/markdown win rates)
# =============================================================================


def formatting_win_rates(
    pairs: Sequence[Mapping[str, Any]],
    patterns: Sequence[str] = ("- ", "* ", "1. ", "##", "**", "```"),
) -> Dict[str, Dict[str, Any]]:
    """Do formatting markers correlate with the chosen label?

    Contract:
    - For each marker string in `patterns`, count pairs where the marker
      occurs in chosen ("n_chosen_has"), in rejected ("n_rejected_has"), in
      both, and in neither (marker occurrence = plain substring test).
    - For each marker also report:
        "win_rate_when_chosen_has":  P(dataset chose that response | marker in
                                      chosen) computed over pairs where the
                                      marker appears in chosen — by
                                      construction this is 1.0 (the label IS
                                      "chosen"); the informative numbers are
                                      the COUNTS and the base rates below
        "base_win_rate":             fraction of all pairs won by the response
                                      containing the marker (either side)
    - Returns {"overall": {"n_pairs": int, ...}, "per_pattern": {marker: {...}}}.
      Every reported rate is in [0.0, 1.0]; denominators of zero yield 0.0.
    """
    raise NotImplementedError


# =============================================================================
# 6 · Cleaned-subset builder with explicit filter rules + removal report
# =============================================================================


def build_cleaned_subset(
    pairs: Sequence[Mapping[str, Any]],
    rules: Sequence[Tuple[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply explicit, named filter rules; keep an auditable removal report.

    Contract:
    - `rules` is an ordered sequence of (rule_name, predicate) where predicate
      maps a pair dict to bool; True = KEEP the pair.
    - Apply rules IN ORDER; a pair removed by an earlier rule is never tested
      again by later rules (first-removal-wins attribution).
    - Never mutate the input pairs; the kept list contains the same dict
      objects (shallow), in original order.
    - Returns (kept_pairs, report) where report has at least:
        {"initial": n_in, "kept": n_kept,
         "removed": {rule_name: count},
         "removed_indices": {rule_name: [original indices]}}
      so every dropped pair is attributable to exactly one named rule.
    """
    raise NotImplementedError


# =============================================================================
# 7 · Decontamination: 8-gram overlap of prompts vs GSM8K test set
# =============================================================================


def ngrams(text: str, n: int = 8) -> Set[str]:
    """Word-level n-grams of a text, normalized for matching.

    Contract:
    - Lowercase the text, strip punctuation (anything not alphanumeric or
      whitespace becomes a space), collapse whitespace, split on whitespace,
      and return the set of all `n`-token tuples-as-strings ("w1 w2 ... wn").
    - Texts with fewer than `n` tokens yield an EMPTY set.
    - Deterministic and pure.
    """
    raise NotImplementedError


def decontamination_check(
    prompts: Sequence[str],
    gsm8k_prompts: Sequence[str],
    n: int = 8,
) -> Dict[str, Any]:
    """Quantify benchmark leakage: how many prompts share an n-gram with GSM8K?

    Contract:
    - Build the n-gram index of `gsm8k_prompts` ONCE (test-set side).
    - A prompt is "flagged" when it shares >= 1 word n-gram with any test-set
      prompt (the count of distinct shared n-grams is the match size).
    - Returns:
        {"n_prompts": len(prompts),
         "n_test_prompts": len(gsm8k_prompts),
         "n_flagged": int,
         "flag_rate": n_flagged / n_prompts (0.0 when empty),
         "flagged_indices": sorted list of flagged indices into `prompts`,
         "shared_ngram_counts": {str(index): int}  # distinct shared n-grams}
    - Empty GSM8K side flags nothing. Never loads anything from disk/network.
    """
    raise NotImplementedError


def decontamination_filter(
    pairs: Sequence[Mapping[str, Any]],
    gsm8k_prompts: Sequence[str],
    n: int = 8,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The write-up deliverable: a decontamination filter over preference pairs.

    Contract:
    - Flag a pair when its "prompt" shares >= 1 word n-gram with any GSM8K
      test prompt (same metric as decontamination_check).
    - Returns (kept_pairs, removed_records):
        kept_pairs:     pairs in original order, decontaminated subset
        removed_records: [{"index": int, "matched_ngrams": sorted list of the
                           shared n-gram strings}] in increasing index order
    - Never mutates the input; kept pairs are the same dict objects.
    """
    raise NotImplementedError


# =============================================================================
# 8 · Extension: Tülu-3 preference rows — compare bias profiles
# =============================================================================


def compare_bias_profiles(
    profile_a: Mapping[str, Any],
    profile_b: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare two bias profiles (e.g. UltraFeedback vs Tülu-3).

    Contract:
    - Inputs are outputs of length_bias_stats (dicts with "n_pairs",
      "p_chosen_longer_chars", "p_chosen_longer_words", mean char lengths).
    - Returns {"n_pairs_a", "n_pairs_b",
               "delta_p_chosen_longer_chars": b - a,
               "delta_p_chosen_longer_words": b - a,
               "delta_mean_chars_chosen": b - a,
               "delta_mean_chars_rejected": b - a}.
    - Deltas are floats signed so that > 0 means profile_b shows MORE of the
      property; pure arithmetic over the two dicts, no I/O.
    """
    raise NotImplementedError
