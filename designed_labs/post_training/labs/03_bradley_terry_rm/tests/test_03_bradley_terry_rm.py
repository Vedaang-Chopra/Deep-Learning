"""Structural tests for the Lab 03 scaffold (Bradley-Terry reward model).

Contract (plan §9):
- no GPU, no network, torch-free at module level;
- pure-python fixtures only — assertions are STRUCTURAL invariants of the
  scaffold (files exist, starter imports cleanly, everything is stubbed,
  forbidden solution patterns absent), never implementations.

Run: python3 -m pytest tests/ -q   (exit 0 expected on a bare Python 3.9)
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
CONFIG_PATH = LAB_DIR / "configs" / "03_bradley_terry_rm.yaml"

# The forbidden library shortcut is spelled dynamically so this *test file*
# itself never contains the banned token.
FORBIDDEN_TOKEN = "log" + "sigmoid"

REQUIRED_MODULE_FUNCTIONS = {
    "tokenize_pair",
    "paired_collate_fn",
    "build_dataloader",
    "pool_last_non_pad_token",
    "bradley_terry_loss",
    "pairwise_accuracy",
    "mean_margin",
    "accuracy_by_margin_bucket",
    "compute_eval_metrics",
    "evaluate",
    "run_training",
    "score_lab02_near_ties",
}

REQUIRED_CLASS = "BradleyTerryRewardModel"
REQUIRED_METHODS = {"__init__", "forward"}


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
                drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    kept = []
    for idx, line in enumerate(lines, start=1):
        if idx in drop:
            continue
        kept.append(re.sub(r"#.*$", "", line))
    text = "".join(kept)
    text = re.sub(r'"[^"\n]*"', '""', text)  # single-line string literals
    text = re.sub(r"'[^'\n]*'", "''", text)
    return text


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


def test_starter_imports_cleanly_without_torch():
    """starter.py must be importable on any machine — including without torch."""
    spec = importlib.util.spec_from_file_location("lab03_starter_under_test", STARTER_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses introspect sys.modules[cls.__module__] for string annotations,
    # so the module must be registered before execution.
    sys.modules.setdefault("lab03_starter_under_test", module)
    spec.loader.exec_module(module)

    assert hasattr(module, "RewardModelConfig")  # plain-python config always available
    cfg = module.RewardModelConfig()
    assert cfg.model_id == "Qwen/Qwen3-0.6B-Base"

    has_torch = bool(getattr(module, "_HAS_TORCH", False))
    if not has_torch:
        assert getattr(module, "BradleyTerryRewardModel", "missing") is None
    else:  # pragma: no cover - machines that do have torch installed
        assert getattr(module, "BradleyTerryRewardModel") is not None


def test_class_wraps_nn_module_when_defined(starter_tree):
    classes = [
        n for n in ast.walk(starter_tree)
        if isinstance(n, ast.ClassDef) and n.name == REQUIRED_CLASS
    ]
    assert classes, f"{REQUIRED_CLASS} must be defined"
    base_names = set()
    for cls in classes:
        for base in cls.bases:
            if isinstance(base, ast.Name):
                base_names.add(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.add(f"{ast.dump(base.value)}|{base.attr}")
    assert any("Module" in b or b == "nn" for b in base_names), (
        "BradleyTerryRewardModel must subclass nn.Module"
    )


def test_every_required_implementation_is_a_stub(starter_tree):
    funcs = _function_nodes(starter_tree)

    def _is_not_implemented(node):
        raises = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Raise)
        ]
        return bool(raises) and all(
            isinstance(r.exc, ast.Call)
            and getattr(r.exc.func, "id", "") == "NotImplementedError"
            for r in raises
        )

    missing = [name for name in REQUIRED_MODULE_FUNCTIONS if name not in funcs]
    assert not missing, f"required functions not defined in starter.py: {missing}"

    for name, nodes in funcs.items():
        if name in REQUIRED_MODULE_FUNCTIONS:
            for node in nodes:
                assert _is_not_implemented(node), f"{name} should raise NotImplementedError"

    class_nodes = [
        n for n in ast.walk(starter_tree)
        if isinstance(n, ast.ClassDef) and n.name == REQUIRED_CLASS
    ]
    methods = {}
    for cls in class_nodes:
        for item in cls.body:
            if isinstance(item, ast.FunctionDef):
                methods.setdefault(item.name, item)
    for method in REQUIRED_METHODS:
        assert method in methods, f"{REQUIRED_CLASS}.{method} must exist"
        assert _is_not_implemented(methods[method]), (
            f"{REQUIRED_CLASS}.{method} should raise NotImplementedError"
        )


def test_no_forbidden_solution_patterns(code_without_docs_and_strings, starter_source):
    # 1) the banned log-sigmoid shortcut must not appear anywhere at all
    assert FORBIDDEN_TOKEN not in starter_source.lower()
    # 2) no executable reward-difference arithmetic outside docstrings/comments
    arith = re.compile(
        r"(chosen|rejected|reward|margin)[A-Za-z_0-9]*\s*-\s*[A-Za-z_(\[]",
        re.IGNORECASE,
    )
    offenders = [
        line.strip()
        for line in code_without_docs_and_strings.splitlines()
        if arith.search(line)
    ]
    assert not offenders, f"reward-difference arithmetic found outside docstrings: {offenders}"
    # 3) docstrings may mention shapes/contract but starter carries no solutions:
    assert "return torch." not in code_without_docs_and_strings.replace('""', "")


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

    joined = "\n".join(md_sources)
    for marker in ("BT loss", "last non-pad", "calibration", "TODO"):
        assert marker.lower() in joined.lower(), f"notebook markdown missing section hint: {marker}"


# ---------------------------------------------------------------------------
# README & config
# ---------------------------------------------------------------------------


def test_readme_documents_prereqs_links_and_compare_pointers():
    readme = README_PATH.read_text(encoding="utf-8")
    for fragment in (
        "Lab 02",
        "Lecture 2",
        "Chapter 5",
        "rlhfbook.com/course",
        "05-reward-models",
        "reward_models/base.py",
        "train_preference_rm.py",
        "Qwen/Qwen3-0.6B-Base",
        "margin",
    ):
        assert fragment in readme, f"README missing expected content: {fragment!r}"


def test_config_parses_and_mirrors_expected_keys():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert data["model"]["model_id"] == "Qwen/Qwen3-0.6B-Base"
    assert "freeze_backbone" in data["model"]
    assert data["data"]["dataset_name"] == (
        "argilla/ultrafeedback-binarized-preferences-cleaned"
    )
    train_keys = {"batch_size", "grad_accum_steps", "learning_rate", "epochs"}
    assert train_keys.issubset(data["train"])
    edges = data["eval"]["margin_bucket_edges"]
    assert edges == sorted(edges), "bucket edges must ascend"


# ---------------------------------------------------------------------------
# Pure-python fixtures: structural invariants of the documented DATA CONTRACT
# ---------------------------------------------------------------------------


PAD_ID = 0


def _fake_paired_batch():
    """Hand-built nested-list fixture standing in for paired_collate_fn output."""
    seqs_chosen = [[5, 6, 7], [8, 9], [10]]
    seqs_rejected = [[11, 12], [13, 14, 15, 16], [17]]

    def pad(seqs):
        width = max(len(s) for s in seqs)
        ids = [s + [PAD_ID] * (width - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
        return ids, mask

    ids_c, mask_c = pad(seqs_chosen)
    ids_r, mask_r = pad(seqs_rejected)
    return ids_c, mask_c, ids_r, mask_r, [len(s) for s in seqs_chosen], [
        len(s) for s in seqs_rejected
    ]


def test_fake_paired_batch_matches_documented_collate_contract():
    ids_c, mask_c, ids_r, mask_r, lens_c, lens_r = _fake_paired_batch()

    B = len(ids_c)
    assert len(ids_r) == B and len(mask_c) == B and len(mask_r) == B

    for ids, mask, lens in ((ids_c, mask_c, lens_c), (ids_r, mask_r, lens_r)):
        L = len(ids[0])
        assert all(len(row) == L for row in ids), "input_ids must be rectangular"
        assert all(len(row) == L for row in mask), "attention_mask must be rectangular"
        for row_ids, row_mask, true_len in zip(ids, mask, lens):
            assert sum(row_mask) == true_len, "mask must count real tokens"
            assert all(tok != PAD_ID or m == 0 for tok, m in zip(row_ids, row_mask)), (
                "pad positions carry PAD_ID and are masked out"
            )
            assert row_mask[:true_len] == [1] * true_len, "right padding assumed"


def test_margin_bucket_fixture_partitions_and_calibration_invariants():
    """Structural rules accuracy_by_margin_bucket's docstring promises."""
    edges = [0.0, 1.0, 2.0, 4.0]
    assert edges == sorted(set(edges)) and len(edges) > 0

    fake_scores_chosen = [[0.5], [2.0], [-0.25]]     # nested lists stand in for [B] tensors
    fake_scores_rejected = [[0.1], [1.5], [3.0]]
    flat_c = [v for row in fake_scores_chosen for v in row]
    flat_r = [v for row in fake_scores_rejected for v in row]
    assert len(flat_c) == len(flat_r) == 3           # shapes align pair-wise

    labels = ["<0.0"] + [f"[{edges[i]},{edges[i+1]})" for i in range(len(edges) - 1)] + [
        f">={edges[-1]}"
    ]
    def label_of(m):
        if m < edges[0]:
            return labels[0]
        if m >= edges[-1]:
            return labels[-1]
        for i in range(len(edges) - 1):
            if edges[i] <= m < edges[i + 1]:
                return labels[i + 1]
        raise AssertionError("unreachable")

    counts = {lab: 0 for lab in labels}
    for c, r in zip(flat_c, flat_r):
        margin = c - r  # fixture-side value only; starter must derive its own
        counts[label_of(margin)] += 1
    assert sum(counts.values()) == 3                 # buckets partition the batch exactly once
