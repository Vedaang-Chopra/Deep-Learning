"""Lab 03b contract tests.

Run WITHOUT GPU/network/torch: ``python3 -m pytest tests/ -q`` (numpy + pytest
only). Torch is imported inside stub bodies only once implemented; these tests
verify the SCAFFOLD: structure, importability, stub discipline, fixture
invariants, and the ORM/PRM record contracts.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

import starter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (fabricated, no network)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg_dict():
    return {
        "model": {"model_id": "Qwen/Qwen3-0.6B", "freeze_backbone": False},
        "data": {
            "rollout_dataset_name": "RLHF-Book/gsm8k-qwen3-0.6B-rollouts",
            "prm_dataset_name": "tasksource/PRM800K",
            "rollouts_per_prompt_cap": None,
            "max_rows_orm": None,
            "val_fraction": 0.05,
            "max_length_orm": 1024,
            "max_length_prm": 1536,
            "step_pad_value": -100,
        },
        "train": {
            "batch_size": 4, "grad_accum_steps": 8, "epochs": 1,
            "learning_rate": 5e-6, "weight_decay": 0.0, "warmup_ratio": 0.03,
            "max_grad_norm": 1.0, "use_amp": True, "eval_every_steps": 50,
        },
        "eval": {
            "disagreement_pool_size": 64, "min_cases_required": 20,
            "orm_positive_threshold": 0.5,
        },
    }


@pytest.fixture(scope="module")
def cfg(cfg_dict):
    return starter.OrmPrmLabConfig.from_dict(cfg_dict)


@pytest.fixture(scope="module")
def rollout_records():
    """Two prompts x two rollouts, balanced labels, schema-exact."""
    recs = []
    for p, pid in enumerate(["p1", "p2"]):
        for r in range(2):
            recs.append({
                "prompt_id": pid,
                "question": f"Q{p}: what is {p + 1}+{r + 1}?",
                "gold_answer": str(p + r + 2),
                "solution": f"Step: count. Answer: {p + r + 2}" if r == 0 else "Answer: 99",
                "predicted_answer": str(p + r + 2) if r == 0 else "99",
                "label": "correct" if r == 0 else "incorrect",
                "rollout_idx": r,
            })
    return recs


@pytest.fixture(scope="module")
def prm_records():
    """Three problems with {-1, 0, +1} step annotations."""
    return [
        {"problem_id": "pr1", "problem": "compute 2+2", "steps": ["add tens", "add ones"], "step_labels": [1, 1]},
        {"problem_id": "pr2", "problem": "compute 7x6", "steps": ["multiply", "state"], "step_labels": [-1, 1]},
        {"problem_id": "pr3", "problem": "compute 9-4", "steps": ["subtract", "round", "state"], "step_labels": [1, 0, 1]},
    ]


# ---------------------------------------------------------------------------
# Structure & importability
# ---------------------------------------------------------------------------

REQUIRED_FUNCTIONS = [
    "load_rollout_dataset", "normalize_rollout_records", "shape_orm_examples",
    "orm_collate_fn", "orm_correctness_loss", "run_orm_training_loop",
    "load_prm800k_slice", "collate_prm_steps", "prm_step_loss",
    "run_prm_training_loop", "score_solutions_both_models",
    "select_disagreement_cases", "build_disagreement_table",
    "write_disagreement_table",
]


def test_required_files_exist():
    assert (LAB_DIR / "README.md").is_file()
    assert (LAB_DIR / "starter.py").is_file()
    assert (LAB_DIR / "notebook.ipynb").is_file()
    assert (LAB_DIR / "configs" / "03b_orm_vs_prm.yaml").is_file()


def test_starter_imports_cleanly():
    assert starter is not None
    for name in REQUIRED_FUNCTIONS:
        assert callable(getattr(starter, name)), f"missing function: {name}"
    assert hasattr(starter, "OrmBinaryHead")
    assert hasattr(starter, "OrmPrmLabConfig")


def test_no_torch_at_module_level():
    src = (LAB_DIR / "starter.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            assert not any(n.split(".")[0] == "torch" for n in names), \
                f"module-level torch import found: {names}"


# ---------------------------------------------------------------------------
# Stub discipline: mechanism functions raise NotImplementedError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", REQUIRED_FUNCTIONS)
def test_stub_raises_not_implemented(name):
    fn = getattr(starter, name)
    # call with minimal plausible args to hit the stub body; any arg-count
    # TypeError would mean signature drift, which we check separately.
    try:
        fn(*([None] * fn.__code__.co_argcount))
    except NotImplementedError:
        return
    except TypeError:
        pytest.fail(f"{name}: signature no longer matches scaffold contract")


def test_dataclass_field_mirrors_yaml(cfg_dict, cfg):
    yaml_text = (LAB_DIR / "configs" / "03b_orm_vs_prm.yaml").read_text()
    parsed = yaml.safe_load(yaml_text)
    assert parsed["model"]["model_id"] == cfg.model_id
    assert parsed["eval"]["min_cases_required"] == cfg.min_cases_required
    assert parsed["train"]["learning_rate"] == cfg.learning_rate


# ---------------------------------------------------------------------------
# Constants/label-domain invariants
# ---------------------------------------------------------------------------

def test_label_domains():
    assert starter.ORM_LABELS == ("correct", "incorrect")
    assert set(starter.PRM_LABELS) == {-1, 0, 1}
    assert "orm_right_prm_flags_reasoning" in starter.DISAGREEMENT_CATEGORIES
    for k in starter.CASE_ROW_KEYS:
        assert isinstance(k, str)


def test_schema_constants_aligned_with_validators():
    assert set(starter.ROLLOUT_RECORD_KEYS) == {
        "prompt_id", "question", "gold_answer", "solution",
        "predicted_answer", "label", "rollout_idx"}
    assert set(starter.PRM_RECORD_KEYS) == {"problem_id", "problem", "steps", "step_labels"}


# ---------------------------------------------------------------------------
# Validator plumbing (implemented) over fabricated fixtures
# ---------------------------------------------------------------------------

def test_validate_rollout_records_accepts_good(rollout_records):
    starter.validate_rollout_records(rollout_records)  # must not raise


def test_validate_rollout_records_rejects_bad(rollout_records):
    bad = dict(rollout_records[0])
    bad["label"] = "maybe"
    with pytest.raises(ValueError):
        starter.validate_rollout_records([bad])

    extra = dict(rollout_records[0]); extra["surprise"] = 1
    with pytest.raises(ValueError):
        starter.validate_rollout_records([extra])

    neg = dict(rollout_records[0]); neg["rollout_idx"] = -1
    with pytest.raises(ValueError):
        starter.validate_rollout_records([neg])


def test_validate_prm_records_accepts_good(prm_records):
    starter.validate_prm_records(prm_records)  # must not raise


def test_validate_prm_records_rejects_bad(prm_records):
    badlabel = dict(prm_records[1]); badlabel["step_labels"] = [-1, 2]
    with pytest.raises(ValueError):
        starter.validate_prm_records([badlabel])

    ragged = dict(prm_records[2]); ragged["steps"] = ragged["steps"][:2]
    with pytest.raises(ValueError):
        starter.validate_prm_records([ragged])


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_load_config_roundtrip(tmp_path, cfg_dict):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg_dict))
    loaded = starter.load_config(str(p))
    assert loaded["eval"]["orm_positive_threshold"] == 0.5
    assert starter.OrmPrmLabConfig.from_dict(loaded).batch_size == 4


# ---------------------------------------------------------------------------
# Metrics plumbing (implemented)
# ---------------------------------------------------------------------------

def test_append_metric_line(tmp_path):
    p = tmp_path / "metrics.jsonl"
    starter.append_metric_line(str(p), {"step": 1, "acc": 0.5})
    starter.append_metric_line(str(p), {"step": 2, "acc": 0.6})
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["step"] == 2


# ---------------------------------------------------------------------------
# Case-row shape contract (fixture-level invariant the student's builder must
# reproduce): every row the student produces later must have exactly
# CASE_ROW_KEYS -- pinned here via the validatorless constant.
# ---------------------------------------------------------------------------

def test_case_row_keys_complete():
    required_categories = {"agree_correct", "agree_incorrect",
                           "orm_right_prm_flags_reasoning", "prm_right_orm_wrong"}
    assert required_categories == set(starter.DISAGREEMENT_CATEGORIES)
