"""Structural tests for the Lab 02 scaffold (preference-data forensics).

Contract (plan §9):
- no GPU, no network, no pandas/numpy/torch/datasets dependencies — the
  starter and these tests are STANDARD-LIBRARY ONLY (there are no
  pandas-dependent parts to skip);
- pure-python fixtures with KNOWN properties: assertions are STRUCTURAL
  invariants of the scaffold (files exist, starter imports cleanly, fixtures
  have their documented properties, everything is stubbed, forbidden solution
  patterns absent), never implementations.

Run: /opt/homebrew/bin/python3 -m pytest tests/ -q   (exit 0 expected)
"""

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

LAB_DIR = Path(__file__).resolve().parents[1]
STARTER_PATH = LAB_DIR / "starter.py"
NOTEBOOK_PATH = LAB_DIR / "notebook.ipynb"
README_PATH = LAB_DIR / "README.md"
CONFIG_PATH = LAB_DIR / "configs" / "02_preference_forensics.yaml"

# Forbidden library shortcuts, spelled dynamically so this *test file* itself
# never contains the banned tokens:
#  - difflib similarity shortcut for near-tie detection
FORBIDDEN_NEAR_TIE_SHORTCUT = "Sequence" + "Matcher"
#  - set-intersection one-liner for the 8-gram overlap (students build the
#    index and count matches themselves)
FORBIDDEN_SET_SHORTCUT = ".intersection("

REQUIRED_MODULE_FUNCTIONS = {
    "normalize_pair",
    "schema_report",
    "load_pairs_jsonl",
    "length_bias_stats",
    "per_source_breakdown",
    "detect_near_ties",
    "sample_manual_audit",
    "audit_agreement",
    "formatting_win_rates",
    "build_cleaned_subset",
    "ngrams",
    "decontamination_check",
    "decontamination_filter",
    "compare_bias_profiles",
}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def starter_source() -> str:
    return STARTER_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def starter_tree(starter_source) -> ast.Module:
    return ast.parse(starter_source)


@pytest.fixture(scope="module")
def starter_module():
    spec = importlib.util.spec_from_file_location("lab02_starter_under_test", STARTER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("lab02_starter_under_test", module)
    spec.loader.exec_module(module)
    return module


def _function_nodes(tree):
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found.setdefault(node.name, []).append(node)
    return found


# ---------------------------------------------------------------------------
# Scaffold structure
# ---------------------------------------------------------------------------


def test_required_files_exist():
    for path in (STARTER_PATH, NOTEBOOK_PATH, README_PATH, CONFIG_PATH):
        assert path.is_file(), f"missing scaffold file: {path}"


def test_starter_imports_cleanly_stdlib_only(starter_module):
    """starter.py must import on a bare interpreter (no pandas/numpy/torch)."""
    cfg = starter_module.ForensicsConfig()
    assert cfg.dataset_name == "argilla/ultrafeedback-binarized-preferences-cleaned"
    assert cfg.limit == 3000
    assert cfg.audit_sample_size == 100
    assert cfg.ngram_n == 8
    assert cfg.near_tie_similarity_threshold == 0.85
    assert "- " in cfg.formatting_patterns and "##" in cfg.formatting_patterns
    for name in REQUIRED_MODULE_FUNCTIONS:
        assert callable(getattr(starter_module, name)), f"missing function: {name}"


def test_every_required_function_is_a_stub(starter_tree):
    funcs = _function_nodes(starter_tree)

    def _is_not_implemented(node):
        def _is_nie_call(exc):
            if isinstance(exc, ast.Call):
                return getattr(exc.func, "id", "") == "NotImplementedError"
            if isinstance(exc, ast.Name):
                return exc.id == "NotImplementedError"
            return False

        raises = [n for n in ast.walk(node) if isinstance(n, ast.Raise)]
        return bool(raises) and all(_is_nie_call(r.exc) for r in raises)

    missing = [name for name in REQUIRED_MODULE_FUNCTIONS if name not in funcs]
    assert not missing, f"required functions not defined in starter.py: {missing}"

    for name, nodes in funcs.items():
        if name in REQUIRED_MODULE_FUNCTIONS:
            for node in nodes:
                assert _is_not_implemented(node), (
                    f"{name} should raise NotImplementedError (no-solutions rule)"
                )


def test_no_forbidden_solution_patterns(starter_source, starter_tree):
    # 1) the banned library shortcuts must not appear anywhere at all
    assert FORBIDDEN_NEAR_TIE_SHORTCUT not in starter_source
    assert FORBIDDEN_SET_SHORTCUT not in starter_source.lower()

    # 2) no executable n-gram slicing / overlap arithmetic outside docstrings
    lines = starter_source.splitlines(keepends=True)
    drop = set()
    for node in ast.walk(starter_tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ) and isinstance(body[0].value.value, str):
                drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    code = "".join(
        re.sub(r"#.*$", "", line)
        for idx, line in enumerate(lines, start=1)
        if idx not in drop
    )
    code = re.sub(r'"[^"\n]*"', '""', code)
    code = re.sub(r"'[^'\n]*'", "''", code)

    ngram_arith = re.compile(r"\[\s*\w+\s*:\s*\w+\s*\+\s*\w+\s*\]")
    offenders = [ln.strip() for ln in code.splitlines() if ngram_arith.search(ln)]
    assert not offenders, f"n-gram slicing found outside docstrings: {offenders}"


# ---------------------------------------------------------------------------
# Notebook skeleton
# ---------------------------------------------------------------------------


def test_notebook_is_valid_nbformat4_with_no_outputs():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] >= 4
    assert isinstance(nb.get("cells"), list) and len(nb["cells"]) >= 10
    assert "kernelspec" in nb["metadata"]

    md_sources = []
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], "code cells must ship with no outputs"
            assert cell.get("execution_count") is None
        else:
            md_sources.append("".join(cell["source"]))

    joined = "\n".join(md_sources).lower()
    for marker in ("schema", "length", "near-tie", "formatting", "gsm8k", "todo"):
        assert marker in joined, f"notebook markdown missing section hint: {marker}"


# ---------------------------------------------------------------------------
# README & config
# ---------------------------------------------------------------------------


def test_readme_documents_prereqs_links_and_compare_pointers():
    readme = README_PATH.read_text(encoding="utf-8")
    for fragment in (
        "Lecture 8",
        "Chapter 10",
        "Chapter 11",
        "rlhfbook.com/course",
        "10-preferences",
        "11-preference-data",
        "ultrafeedback-binarized-preferences-cleaned",
        "GSM8K",
        "near-tie",
        "train_preference_rm.py",
    ):
        assert fragment in readme, f"README missing expected content: {fragment!r}"


def test_config_parses_and_mirrors_expected_keys():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert data["data"]["dataset_name"] == (
        "argilla/ultrafeedback-binarized-preferences-cleaned"
    )
    assert data["data"]["limit"] >= 3000
    assert data["decontamination"]["ngram_n"] == 8
    assert data["audit"]["sample_size"] == 100
    assert data["near_ties"]["similarity_threshold"] == 0.85
    assert isinstance(data["formatting"]["patterns"], list) and data["formatting"]["patterns"]


# ---------------------------------------------------------------------------
# Pure-python fixtures: synthetic pairs with KNOWN properties
# ---------------------------------------------------------------------------

CH_BODY = "This is the chosen reply for pair {i}, written one way."
RJ_BODY = "This is the rejected reply for pair {i}, written another way."
EXTRA = "Extra elaboration follows to make this side longer on purpose."
SOURCES = ["sharegpt", "evol_instruct", "ultrachat", "false_qa", "flan"]

# Among the 18 "normal" pairs, exactly these have the strictly longer CHOSEN:
LONGER_CHOSEN = {0, 1, 2, 3, 4, 6, 8, 9, 10, 12, 13, 15}
IDENTICAL_PAIR = 5     # chosen == rejected verbatim
NEAR_TIE_PAIR = 7      # responses differ by one word only
CONTAMINATED_PAIR = 11 # prompt shares a word 8-gram with the GSM8K fixture

GSM8K_TEST_PROMPTS = [
    "Janet has four apples and buys two more apples at the market "
    "how many apples does she have now",
    "Tom races his car around the track twice and each lap is 3 kilometers long",
]
CONTAMINATED_PROMPT = (
    "Please solve: Janet has four apples and buys two more apples at the "
    "market. Show your work step by step."
)


def _synthetic_pairs():
    """20 normalized pairs; properties asserted by the fixture tests below."""
    pairs = []
    for i in range(20):
        if i == CONTAMINATED_PAIR:
            prompt = CONTAMINATED_PROMPT
        else:
            prompt = f"Prompt number {i}: explain a simple concept in a few sentences."
        if i == IDENTICAL_PAIR:
            chosen = rejected = f"Pair {i}: identical twins share the same text verbatim."
        elif i == NEAR_TIE_PAIR:
            chosen = (
                "The quick brown fox jumps over the lazy dog beside the river "
                "bank in the morning light."
            )
            rejected = (
                "The quick brown fox jumps over the lazy dog beside the river "
                "bank in the evening light."
            )
        else:
            chosen_body = CH_BODY.format(i=i)
            rejected_body = RJ_BODY.format(i=i)
            if i in LONGER_CHOSEN:
                chosen = chosen_body + " " + " ".join([EXTRA] * 4)
                rejected = rejected_body
            else:
                chosen = chosen_body
                rejected = rejected_body + " " + " ".join([EXTRA] * 4)
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": SOURCES[i % len(SOURCES)],
            }
        )
    return pairs


SYNTHETIC_PAIRS = _synthetic_pairs()


# ---- documented metric re-implementations (fixture side only) ----


def _word_ngrams_fixture(text, n=8):
    words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return {" ".join(words[k : k + n]) for k in range(len(words) - n + 1)}


def _char_ngram_jaccard_fixture(a, b, size=4):
    if a == b:
        return 1.0
    ga = {a[k : k + size] for k in range(max(len(a) - size + 1, 0))}
    gb = {b[k : k + size] for k in range(max(len(b) - size + 1, 0))}
    union = ga | gb
    return len(ga & gb) / len(union) if union else 0.0


def test_fixture_length_bias_invariant():
    """The fixture really has P(chosen strictly longer) == 12/20 == 0.6."""
    n_longer = sum(
        1 for p in SYNTHETIC_PAIRS if len(p["chosen"]) > len(p["rejected"])
    )
    assert n_longer == 12
    assert n_longer / len(SYNTHETIC_PAIRS) == 0.6
    # sources cover five groups, every pair carries one
    assert {p["source"] for p in SYNTHETIC_PAIRS} == set(SOURCES)


def test_fixture_near_tie_invariants():
    """Exactly one identical pair; the near-tie pair scores high Jaccard."""
    identical = [
        i
        for i, p in enumerate(SYNTHETIC_PAIRS)
        if p["chosen"] == p["rejected"]
    ]
    assert identical == [IDENTICAL_PAIR]

    near = SYNTHETIC_PAIRS[NEAR_TIE_PAIR]
    sim = _char_ngram_jaccard_fixture(near["chosen"], near["rejected"])
    assert sim >= 0.85, f"near-tie fixture must exceed the 0.85 threshold, got {sim}"

    control = SYNTHETIC_PAIRS[0]
    sim_c = _char_ngram_jaccard_fixture(control["chosen"], control["rejected"])
    assert sim_c < 0.4, f"control pair must be clearly non-near-tie, got {sim_c}"


def test_fixture_gsm8k_overlap_invariant():
    """Only the contaminated prompt shares a word 8-gram with the GSM8K side."""
    test_grams = set()
    for p in GSM8K_TEST_PROMPTS:
        test_grams |= _word_ngrams_fixture(p)
    assert test_grams, "GSM8K fixture prompts must yield 8-grams"

    flagged = []
    for i, pair in enumerate(SYNTHETIC_PAIRS):
        shared = _word_ngrams_fixture(pair["prompt"]) & test_grams
        if shared:
            flagged.append((i, shared))
    assert [i for i, _ in flagged] == [CONTAMINATED_PAIR]
    # the shared 8-gram really comes from the intended sentence
    assert any(
        "janet has four apples and buys two" in gram
        for _, shared in flagged
        for gram in shared
    )
