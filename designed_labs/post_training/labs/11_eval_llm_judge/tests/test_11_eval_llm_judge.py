"""Structural tests for the Lab 11 scaffold (Evaluation & LLM-as-judge).

Contract (plan §9):
- no GPU, no network, torch-free at module level;
- pure-python (+ numpy available) FABRICATED judge outputs with KNOWN flip
  patterns — assertions are fixture-level invariants of that fabricated data
  plus STRUCTURAL invariants of the scaffold (files exist, starter imports
  cleanly, everything is stubbed, forbidden solution patterns absent);
  never implementations;
- no pandas, no YAML lib requirement, no HF/network imports.

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
CONFIG_PATH = LAB_DIR / "configs" / "11_eval_llm_judge.yaml"

# The forbidden methodology pointer is spelled via parts so this *test file*
# itself never trips naive token greps aimed at the starter.
FORBIDDEN_METHODOLOGY_TOKEN = "diagnos" + "tics.py"

REQUIRED_FUNCTIONS = {
    "load_prompt_set",
    "build_head_to_head_pairs",
    "rm_judge_verdict",
    "llm_judge_verdict",
    "discover_checkpoints",
    "run_checkpoint_suite",
    "compute_win_rate_matrix",
    "order_swap_flip_rate",
    "position_bias_report",
    "length_bias_report",
    "self_preference_bias_report",
    "verdict_vs_ground_truth",
    "generate_synthetic_preferences",
    "judge_filter_preferences",
    "measure_introduced_bias",
}

REQUIRED_DATACLASS = "EvalConfig"
REQUIRED_CONFIG_FIELDS = {
    "checkpoint_paths",
    "rm_checkpoint_path",
    "judge_model_id",
    "judge_temperature",
    "prompt_set_path",
    "swap_orders",
    "length_gap_tokens_threshold",
    "self_preference_checkpoint",
    "synthetic_teacher_model_id",
    "synthetic_filter_min_margin",
    "seed",
    "output_dir",
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
def code_without_docs_and_strings(starter_source, starter_tree) -> str:
    """Source text with docstrings, comments and string literals removed.

    Forbidden-pattern checks run against THIS view so that textual mentions
    inside docstrings (allowed) can never pass as implementation arithmetic.
    """
    lines = starter_source.splitlines(keepends=True)
    drop = set()
    for node in ast.walk(starter_tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ) and isinstance(body[0].value.value, str):
                drop.update(range(body[0].lineno - 1, body[0].end_lineno))
    return "".join(line for i, line in enumerate(lines) if i not in drop)


@pytest.fixture(scope="module")
def starter_module():
    spec = importlib.util.spec_from_file_location("lab11_starter_under_test", STARTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Structural invariants: files, imports, stubs
# ---------------------------------------------------------------------------


def test_required_files_exist():
    assert STARTER_PATH.is_file()
    assert NOTEBOOK_PATH.is_file()
    assert README_PATH.is_file()
    assert CONFIG_PATH.is_file()


def test_starter_imports_cleanly(starter_module):
    assert starter_module is not None
    assert starter_module._HAS_TORCH in (True, False)


def test_required_functions_and_config_present(starter_module):
    for name in REQUIRED_FUNCTIONS:
        assert callable(getattr(starter_module, name, None)), f"missing function: {name}"
    cfg = getattr(starter_module, REQUIRED_DATACLASS, None)
    assert cfg is not None, f"missing dataclass: {REQUIRED_DATACLASS}"
    import dataclasses

    fields = {f.name for f in dataclasses.fields(cfg)}
    missing = REQUIRED_CONFIG_FIELDS - fields
    assert not missing, f"EvalConfig missing fields: {sorted(missing)}"


def test_every_stub_raises_not_implemented(starter_source, starter_tree):
    """Every required function body must be docstring + raise NotImplementedError."""
    funcs = {
        n.name: n
        for n in ast.walk(starter_tree)
        if isinstance(n, ast.FunctionDef) and n.name in REQUIRED_FUNCTIONS
    }
    assert set(funcs) == REQUIRED_FUNCTIONS
    for name, node in funcs.items():
        stmts = [
            s
            for s in node.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]
        assert len(stmts) == 1 and isinstance(stmts[0], ast.Raise), (
            f"{name} must be a pure stub (docstring + raise NotImplementedError)"
        )


@pytest.mark.parametrize("func_name", sorted(REQUIRED_FUNCTIONS))
def test_stub_behavior_not_implemented(starter_module, func_name):
    """Calling each stub with minimal fabricated args raises NotImplementedError."""
    fn = getattr(starter_module, func_name)
    minimal_args = {
        "load_prompt_set": ("dummy.jsonl",),
        "build_head_to_head_pairs": ({},),
        "rm_judge_verdict": (None, "p", "ca", "cb"),
        "llm_judge_verdict": (None, None, "p", "ca", "cb", starter_module.EvalConfig()),
        "discover_checkpoints": ("dummy_root",),
        "run_checkpoint_suite": (starter_module.EvalConfig(),),
        "compute_win_rate_matrix": ([], []),
        "order_swap_flip_rate": ([], []),
        "position_bias_report": ([],),
        "length_bias_report": ([],),
        "self_preference_bias_report": ([], "sft"),
        "verdict_vs_ground_truth": ([], {}),
        "generate_synthetic_preferences": (None, None, [], starter_module.EvalConfig()),
        "judge_filter_preferences": ([], None, starter_module.EvalConfig()),
        "measure_introduced_bias": ({}, []),
    }
    with pytest.raises(NotImplementedError):
        fn(*minimal_args[func_name])


def test_no_solution_arithmetic_in_code(starter_source, code_without_docs_and_strings):
    """No win-rate/bias/aggregation arithmetic may appear outside docstrings.

    These tokens only belong in docstring CONTRACTS; if they show up in the
    stripped code view, arithmetic leaked into the scaffold.
    """
    for token in (
        # tokens that would only make sense as computed OUTPUTS/arithmetic;
        # occurrences inside stub NAMES (e.g. order_swap_flip_rate) are fine
        # because test_every_stub_raises_not_implemented already proves every
        # required body is exactly one `raise NotImplementedError`.
        "win_rates",
        "np.mean",
        ".mean(",
        "/ len(",
        "Counter(",
        "0.5 *",
        "* 0.5",
        "decidable_fraction",
        "p_first_position",
        "p_self_win",
        "keep_rate",
        FORBIDDEN_METHODOLOGY_TOKEN.replace(".py", ""),  # module name, not path text
    ):
        assert token not in code_without_docs_and_strings, (
            f"forbidden implementation pattern in code: {token!r}"
        )
    # The methodology pointer must exist in the DOCSTRINGS (compare-against rule).
    assert "rejection_sampling/diagnostics.py" in starter_source


def test_no_network_or_pandas_imports(starter_tree):
    imported = set()
    for node in ast.walk(starter_tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"requests", "httpx", "urllib", "socket", "pandas", "transformers", "torch.nn", "datasets", "vllm"}
    assert not (imported & banned), f"banned imports found: {sorted(imported & banned)}"
    assert imported <= {"__future__", "dataclasses", "typing", "torch"}, sorted(imported)


def test_config_yaml_mirrors_dataclass_fields():
    yaml_text = CONFIG_PATH.read_text(encoding="utf-8")
    # EvalConfig field -> the yaml key that carries it (sections nest keys).
    yaml_key_for_field = {
        "checkpoint_paths": "checkpoint_paths",
        "rm_checkpoint_path": "rm_checkpoint_path",
        "judge_model_id": "model_id",
        "judge_temperature": "temperature",
        "prompt_set_path": "prompt_set_path",
        "swap_orders": "swap_orders",
        "length_gap_tokens_threshold": "length_gap_tokens_threshold",
        "self_preference_checkpoint": "self_preference_checkpoint",
        "synthetic_teacher_model_id": "teacher_model_id",
        "synthetic_filter_min_margin": "filter_min_margin",
        "seed": "seed",
        "output_dir": "output_dir",
    }
    for field, key in yaml_key_for_field.items():
        # anchored so 'teacher_model_id:' never satisfies a search for 'model_id:'
        assert re.search(rf"^\s*{key}:", yaml_text, re.M), (
            f"yaml missing key {key!r} (EvalConfig.{field})"
        )
    for anchor in ("Qwen/Qwen3-4B", "lab11_eval_judge"):
        assert anchor in yaml_text


def test_readme_contract_markers():
    readme = README_PATH.read_text(encoding="utf-8")
    for marker in (
        "Labs 00",
        "Lecture 12",
        "Chapter 16",
        "gap lab",
        "Lab 12",  # absorbs the old Lab 12
        "rejection_sampling/diagnostics.py",
        "order-swap",
        "Qwen/Qwen3-4B",
    ):
        assert marker in readme, f"README missing required marker: {marker!r}"


def test_notebook_valid_json_no_outputs():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert cells, "notebook must contain cells"
    for i, cell in enumerate(cells):
        assert cell.get("cell_type") in ("code", "markdown"), f"cell {i} bad type"
        if cell.get("cell_type") == "code":
            assert cell.get("outputs") == [], f"code cell {i} has pre-computed outputs"
            assert cell.get("execution_count") is None, f"code cell {i} has execution_count"
        src = "".join(cell.get("source", []))
        assert src.strip(), f"cell {i} is empty"


# ---------------------------------------------------------------------------
# Fabricated judge outputs with KNOWN flip patterns (fixture-level invariants)
# ---------------------------------------------------------------------------

MODELS = ("base", "sft", "dpo")


def _rec(model_a, model_b, prompt_id, verdict, order_shown="ab", **extra):
    row = {
        "model_a": model_a,
        "model_b": model_b,
        "prompt_id": prompt_id,
        "verdict": verdict,       # "a" | "b" | "tie" — as DISPLAYED to the judge
        "order_shown": order_shown,
    }
    row.update(extra)
    return row


def fabricated_two_order_verdicts():
    """12 aligned comparisons judged in BOTH display orders, planted pattern.

    Physical winner per prompt (in the (model_a=model_a, model_b=model_b,
    a-first frame): prompts p0..p6 physically favor model_a; prompts p7..p11
    exercise ties/position bias as labeled below.
    """
    first = []
    swapped = []
    for i in range(7):
        first.append(_rec("sft", "dpo", f"p{i}", "a", "ab"))
        # same physical winner, now displayed second:
        swapped.append(_rec("sft", "dpo", f"p{i}", "b", "ba"))
    # p7..p9: judge always picks the FIRST-DISPLAYED completion (position bias):
    for i in range(7, 10):
        first.append(_rec("sft", "dpo", f"p{i}", "a", "ab"))
        swapped.append(_rec("sft", "dpo", f"p{i}", "a", "ba"))  # winner is physically model_b here
    # p10: tie in one order, decisive in the other (tie conflict, not a flip):
    first.append(_rec("sft", "dpo", "p10", "tie", "ab"))
    swapped.append(_rec("sft", "dpo", "p10", "a", "ba"))
    # p11: mirror-image tie conflict:
    first.append(_rec("sft", "dpo", "p11", "b", "ab"))
    swapped.append(_rec("sft", "dpo", "p11", "tie", "ba"))
    return first, swapped


def test_fixture_two_order_known_flip_pattern():
    """Invariant check on the FABRICATED data: exactly 3 flips, all toward first."""
    first, swapped = fabricated_two_order_verdicts()
    assert len(first) == len(swapped) == 12
    # alignment contract: same (model_a, model_b, prompt_id) per index
    for f, s in zip(first, swapped):
        assert (f["model_a"], f["model_b"], f["prompt_id"]) == (
            s["model_a"], s["model_b"], s["prompt_id"]
        )
        assert {f["order_shown"], s["order_shown"]} == {"ab", "ba"}
    # verdict expressed in the PHYSICAL (model_a-first) frame:
    def _mirror(v):
        return {"a": "b", "b": "a", "tie": "tie"}[v]

    flips = 0
    toward_first = 0
    tie_conflicts = 0
    for f, s in zip(first, swapped):
        # verdict expressed in the PHYSICAL (model_a-first) frame:
        # a record shown as 'ba' displays model_b in slot A -> mirror the verdict
        # into the physical (model_a-first) frame:
        v1 = f["verdict"] if f["order_shown"] == "ab" else _mirror(f["verdict"])
        v2 = _mirror(s["verdict"]) if s["order_shown"] == "ba" else s["verdict"]
        if "tie" in (v1, v2):
            tie_conflicts += 1
            continue
        if v1 != v2:
            flips += 1
            # which DISPLAY position won the conflicting verdicts?
            if f["verdict"] == "a" and s["verdict"] == "a":
                toward_first += 1
    assert flips == 3, "planted pattern: prompts p7-p9 flip toward first position"
    assert toward_first == 3
    assert tie_conflicts == 2, "p10/p11 are tie conflicts, not flips"
    # non-flipping prompts are consistent across orders (p0-p6 favor sft):
    for i in range(7):
        assert first[i]["verdict"] == "a" and swapped[i]["verdict"] == "b"


def test_fixture_order_swap_mirror_property():
    """A consistent judge's two-order verdicts are exact mirror images.

    Fabricated records with order_shown='ba' must, after un-flipping,
    reproduce the a-first verdicts one-to-one — the property
    compute_win_rate_matrix's un-flipping step must preserve.
    """
    first, swapped = fabricated_two_order_verdicts()
    mirror = {"a": "b", "b": "a", "tie": "tie"}
    for i in range(7):  # consistent physical winners only
        v_swapped_physical = mirror[swapped[i]["verdict"]]
        assert v_swapped_physical == first[i]["verdict"]


def test_fixture_win_rate_records_complementarity():
    """Records destined for the matrix: per-prompt winners are well defined
    and complementary — win_rates[i][j] + win_rates[j][i] == 1 given ties=0.5."""
    records = [
        _rec("base", "sft", "q0", "a", "ab"),
        _rec("base", "sft", "q1", "b", "ab"),
        _rec("sft", "base", "q2", "a", "ab"),   # displayed order reversed!
        _rec("base", "sft", "q2", "b", "ba"),   # same comparison, other order
        _rec("base", "dpo", "q3", "tie", "ab"),
    ]
    # un-flip into a canonical (model_a < model_b) frame, then score:
    canonical = {}
    for r in records:
        key = (r["prompt_id"], tuple(sorted((r["model_a"], r["model_b"]))))
        v = r["verdict"]
        if r["order_shown"] == "ba":
            v = {"a": "b", "b": "a", "tie": "tie"}[v]
            # swap which model 'a'/'b' refer to:
            lo, hi = r["model_b"], r["model_a"]
        else:
            lo, hi = r["model_a"], r["model_b"]
        winner = {"a": lo, "b": hi, "tie": None}[v]
        canonical.setdefault(key, []).append(winner)
    assert len(canonical) == 4
    # q2 was judged twice in opposite orders consistently -> same winner:
    assert canonical[("q2", ("base", "sft"))] == ["sft", "sft"]
    # complementary credit invariant (ties = 0.5 each):
    for key, winners in canonical.items():
        a_model, b_model = key[1]
        credit_a = 0.5 * winners.count(None) + winners.count(a_model)
        credit_b = 0.5 * winners.count(None) + winners.count(b_model)
        assert credit_a + credit_b == len(winners)


def test_fixture_self_preference_known_delta():
    """Judge lineage 'sft' participates in 8 pairs and wins 6 -> delta 0.5-? known."""
    recs = []
    n = 0
    for i in range(8):
        winner_is_sft = i < 6
        recs.append(_rec("sft", "dpo", f"s{i}", "a" if winner_is_sft else "b"))
        n += 1
    p_self_win = sum(1 for r in recs if (r["verdict"] == "a")) / n
    p_other_win = 1.0 - p_self_win
    assert abs(p_self_win - 0.75) < 1e-9
    assert abs((p_self_win - p_other_win) - 0.5) < 1e-9


def test_fixture_ground_truth_known_confusions():
    """Verifiable prompts with planted truth-vs-judge disagreements."""
    gt = {
        "g0": {"truth": "a", "judge": "a"},   # agree
        "g1": {"truth": "a", "judge": "b"},   # truth_a_judge_b
        "g2": {"truth": "b", "judge": "a"},   # truth_b_judge_a
        "g3": {"truth": "both_correct", "judge": "a"},  # undecidable, excluded
    }
    confusions = {"truth_a_judge_b": 0, "truth_b_judge_a": 0, "truth_decided_judge_tie": 0}
    decidable = 0
    for row in gt.values():
        if row["truth"] in ("both_correct", "neither_correct"):
            continue
        decidable += 1
        if row["judge"] == "tie":
            confusions["truth_decided_judge_tie"] += 1
        elif (row["truth"], row["judge"]) == ("a", "b"):
            confusions["truth_a_judge_b"] += 1
        elif (row["truth"], row["judge"]) == ("b", "a"):
            confusions["truth_b_judge_a"] += 1
    assert decidable == 3
    assert sum(confusions.values()) == 2
    assert confusions["truth_a_judge_b"] == 1 and confusions["truth_b_judge_a"] == 1


def test_fixture_prompt_set_invariants():
    """Frozen prompt set rows: unique ids, required keys present."""
    rows = [
        {"prompt_id": f"p{i}", "prompt": f"prompt {i}", "category": "gsm8k_verifiable"}
        for i in range(5)
    ] + [{"prompt_id": "x0", "prompt": "open", "category": "open_ended"}]
    ids = [r["prompt_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate prompt_id must be a ValueError"
    for r in rows:
        assert {"prompt_id", "prompt", "category"} <= set(r)
