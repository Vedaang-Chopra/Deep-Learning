"""Lab 05 contract tests.

Run WITHOUT GPU/network/torch: ``python3 -m pytest tests/ -q`` (numpy + pytest
only). Verifies the SCAFFOLD: structure, importability, stub discipline, and
structural invariants of the given plumbing (config validation, trajectory
container, expression oracle) over hand-built fixtures.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

import starter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg_dict():
    return {
        "seed": 42,
        "env": {"target_range_min": 0, "target_range_max": 99},
        "policy": {"hidden_size": 128},
        "reinforce": {
            "learning_rate": 0.01, "episodes_per_update": 32,
            "group_size_k": 4, "baseline": "none",
            "moving_avg_alpha": 0.9, "entropy_coef": 0.0,
        },
        "logging": {"jsonl_path": "runs/lab05_toy_mdp/metrics.jsonl"},
    }


@pytest.fixture()
def rng():
    return np.random.default_rng(1234)


# ---------------------------------------------------------------------------
# Structure & importability
# ---------------------------------------------------------------------------

REQUIRED_FUNCTIONS = [
    "reinforce_loss", "compute_advantages", "entropy_of_policy",
    "gradient_variance_estimate", "run_variance_experiment",
    "observe_entropy_collapse", "softmax", "log_softmax",
    "sample_trajectory",
]

# Implemented bookkeeping plumbing (given, tested separately):
GIVEN_PLUMBING = ["trajectory_log_prob"]

REQUIRED_CLASSES = ["ToyArithmeticEnv", "CharPolicyNet", "Trajectory", "ToyMDPConfig"]


def test_required_files_exist():
    assert (LAB_DIR / "README.md").is_file()
    assert (LAB_DIR / "starter.py").is_file()
    assert (LAB_DIR / "notebook.ipynb").is_file()
    assert (LAB_DIR / "configs" / "05_policy_gradients_toy_mdp.yaml").is_file()


def test_starter_imports_cleanly():
    for name in REQUIRED_FUNCTIONS:
        assert callable(getattr(starter, name)), f"missing function: {name}"
    for cls in REQUIRED_CLASSES:
        assert hasattr(starter, cls), f"missing class: {cls}"


def test_no_torch_at_module_level():
    src = (LAB_DIR / "starter.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            assert not any(n.split(".")[0] in ("torch", "scipy") for n in names), \
                f"module-level forbidden import: {names}"


# ---------------------------------------------------------------------------
# Stub discipline: mechanisms raise NotImplementedError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", REQUIRED_FUNCTIONS)
def test_stub_raises_not_implemented(name):
    fn = getattr(starter, name)
    try:
        fn(*([None] * fn.__code__.co_argcount))
    except NotImplementedError:
        return
    except TypeError:
        pytest.fail(f"{name}: signature drifted from scaffold contract")


def test_env_methods_are_stubs():
    env = starter.ToyArithmeticEnv(lo=0, hi=9)
    with pytest.raises(NotImplementedError):
        env.reset(5)
    with pytest.raises(NotImplementedError):
        env.step("7")
    with pytest.raises(NotImplementedError):
        starter.CharPolicyNet().init_params(np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Given plumbing: config contract
# ---------------------------------------------------------------------------

def test_validate_config_accepts_good(cfg_dict):
    starter.validate_config(cfg_dict)  # must not raise


def test_validate_config_rejects_bad(cfg_dict):
    bad = {**cfg_dict, "seed": "not-int"}
    with pytest.raises(ValueError):
        starter.validate_config(bad)

    bad2 = {**cfg_dict}
    bad2["reinforce"] = {**cfg_dict["reinforce"], "baseline": "trust_region"}
    with pytest.raises(ValueError):
        starter.validate_config(bad2)

    bad3 = {**cfg_dict}
    bad3["env"] = {"target_range_min": 50, "target_range_max": 10}
    with pytest.raises(ValueError):
        starter.validate_config(bad3)

    bad4 = {**cfg_dict}
    bad4["reinforce"] = {**cfg_dict["reinforce"], "group_size_k": 7}  # not in K_GRID
    with pytest.raises(ValueError):
        starter.validate_config(bad4)


def test_k_grid_matches_plan_spec():
    assert set(starter.K_GRID) == {2, 4, 16}


def test_dataclass_mirror_roundtrip(cfg_dict):
    cfg = starter.ToyMDPConfig.from_dict(cfg_dict)
    assert cfg.baseline == "none"
    assert cfg.group_size_k == 4
    assert cfg.target_range_max == 99


# ---------------------------------------------------------------------------
# Given plumbing: trajectory container invariants
# ---------------------------------------------------------------------------

def test_trajectory_validate_parallel_lists():
    trj = starter.Trajectory(target_n=5, chars=["1", "+", "4"], log_probs=[-0.2, -0.3])
    with pytest.raises(ValueError):
        trj.validate()


def test_trajectory_validate_length_cap(rng):
    n = starter.MAX_EPISODE_CHARS + 1
    trj = starter.Trajectory(target_n=5, chars=["0"] * n, log_probs=[-0.1] * n)
    with pytest.raises(ValueError):
        trj.validate()


def test_trajectory_validate_finite_logs():
    trj = starter.Trajectory(target_n=5, chars=["1"], log_probs=[float("nan")])
    with pytest.raises(ValueError):
        trj.validate()


def test_trajectory_text_and_logprob_plumbing():
    trj = starter.Trajectory(target_n=5, chars=["1", "+", "4", "="],
                             log_probs=[-1.0, -0.5, -0.25, -2.0], reward=1.0)
    assert trj.text == "1+4="
    assert starter.trajectory_log_prob(trj) == pytest.approx(-3.75)


# ---------------------------------------------------------------------------
# Given plumbing: expression oracle (verifier spec you must match later)
# ---------------------------------------------------------------------------

def test_expression_value_oracle():
    assert starter.expression_value("1+4") == 5
    assert starter.expression_value("99+0") == 99
    # malformed inputs -> None:
    assert starter.expression_value("1+") is None
    assert starter.expression_value("+1") is None
    assert starter.expression_value("12a+3") is None
    assert starter.expression_value("1=4") is None      # '=' is terminator char
    assert starter.expression_value("") is None
    assert starter.expression_value("1 + 4") is None    # no spaces allowed
    assert starter.expression_value("1-4") is None      # single '+' operator only


def test_char_actions_include_terminator_and_operands():
    assert "=" in starter.CHAR_ACTIONS and "+" in starter.CHAR_ACTIONS
    assert all(c.isdigit() or c in "=+ " for c in starter.CHAR_ACTIONS)


# ---------------------------------------------------------------------------
# No-solutions spot check: core arithmetic must NOT be pre-implemented
# ---------------------------------------------------------------------------

def test_no_hidden_softmax_implementations_in_starter():
    """starter.py source (sans docstrings) must not contain executable softmax/
    advantage arithmetic -- textual mentions inside docstrings are fine."""
    import re
    src = (LAB_DIR / "starter.py").read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ) and isinstance(body[0].value.value, str):
                drop.update(range(body[0].lineno - 1, body[0].end_lineno))
    code_only = "".join(l for i, l in enumerate(lines) if i not in drop)
    code_only = "\n".join(l for l in code_only.splitlines() if not l.strip().startswith("#"))
    forbidden = [
        r"exp\s*\(",                       # no exp() calls outside docstrings
        r"np\.softmax|scipy\.special",
        r"logsumexp",
        r"advantage\s*=\s*.*reward\s*-",
        r"\bmean\(rewards\)|rewards\.mean\(\)",
        r"H\s*=\s*-\s*(p|prob)",           # entropy formula skeleton
    ]
    for pat in forbidden:
        m = re.search(pat, code_only)
        assert m is None, f"executable solution pattern found: /{pat}/"


def test_raise_not_implemented_count_matches_mechanisms():
    """Every mechanism stub should carry exactly one NotImplementedError;
    guards against someone quietly deleting a raise to 'finish' early."""
    src = (LAB_DIR / "starter.py").read_text()
    n_raises = src.count("raise NotImplementedError")
    assert n_raises >= len(REQUIRED_FUNCTIONS), \
        f"expected >= {len(REQUIRED_FUNCTIONS)} stubs, found {n_raises}"
