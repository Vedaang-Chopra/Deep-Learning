"""Tests for Lab 08 — Reward hacking & KL regularization.

Contract (plan §9.5): structural invariants ONLY, verified on numpy fixtures —
no GPU, no network, no torch, and never assertions about solution logic.
These tests pass against the untouched scaffold (stubs raise NotImplementedError)
and keep guarding structure once you implement.

Run:  python3 -m pytest tests/ -q
"""

import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(HERE)

STARTER_DIR = LAB_DIR

# ensure `starter` imports regardless of pytest rootdir
import sys

sys.path.insert(0, STARTER_DIR)

import starter  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures: numpy only
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def config():
    """Parsed lab config (yaml if available; else required-key text scan)."""
    path = os.path.join(LAB_DIR, "configs", "08_reward_hacking_kl.yaml")
    assert os.path.isfile(path), f"missing config: {path}"
    try:
        import yaml

        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        assert isinstance(cfg, dict), "config must parse to a top-level mapping"
        return cfg
    except ImportError:  # minimal envs: fall back to flat key scan
        with open(path) as fh:
            lines = fh.read().splitlines()
        flat = {
            ln.split(":", 1)[0].strip(): ln.split(":", 1)[1]
            for ln in lines
            if ":" in ln and not ln.startswith((" ", "#"))
        }
        return {"__flat__": flat}


@pytest.fixture()
def rollout_batch():
    """Numpy fixture: 8-sample verification batch like one GRPO group."""
    return {
        "correctness": np.array([1, 0, 1, 0, 0, 0, 1, 0]),
        "format_ok": np.array([1, 1, 0, 1, 1, 0, 1, 1]),
    }


# ---------------------------------------------------------------------------
# config invariants
# ---------------------------------------------------------------------------
def test_config_parses_and_has_required_sweep_keys(config):
    """plan §9.4 example invariant: config parses + carries the two keys."""
    if "__flat__" in config:  # fallback scanner found flattened YAML
        flat = config["__flat__"]
        assert "proxy_reward_weight" in flat
        assert "kl_coef_grid" in flat
    else:
        sweep = config.get("sweep", {})
        proxy = config.get("proxy_reward", {})
        # required key lives under proxy_reward by convention; accept either
        # nesting as long as it is discoverable exactly once, top-to-bottom.
        flat_cfg = _flatten(config)
        assert "proxy_reward_weight" in flat_cfg
        assert "kl_coef_grid" in flat_cfg
        grid = sweep.get("kl_coef_grid")
        assert isinstance(grid, (list, tuple)) and len(grid) >= 3
        prw = proxy.get("proxy_reward_weight", flat_cfg.get("proxy_reward_weight"))
        assert isinstance(prw, (int, float)) and prw > 0


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
            out[k] = v  # also index by bare leaf name for convenience
        else:
            out[key] = v
            out.setdefault(k, v)
    return out


def test_config_no_torch_dependency(config):
    """Config mentions no torch/accelerate requirement at module level."""
    text = json.dumps(_flatten(config))
    assert "torch" not in text.lower()


# ---------------------------------------------------------------------------
# starter module invariants
# ---------------------------------------------------------------------------
def test_starter_imports_cleanly():
    """Import must not pull torch or execute any heavy machinery."""
    import sys as _sys

    forbidden = [m for m in _sys.modules if m == "torch" or m.startswith("torch.")]
    assert not forbidden, "starter.py dragged torch into a torch-free process"


def test_all_four_stubs_exist():
    for name in (
        "build_proxy_reward",
        "run_kl_sweep",
        "prepare_divergence_plot_data",
        "select_earliest_warning_metric",
    ):
        fn = getattr(starter, name, None)
        assert callable(fn), f"missing scaffold stub {name}"
        doc = getattr(fn, "__doc__") or ""
        assert len(doc.strip()) >= 40, f"{name} lacks its contract docstring"


def test_build_proxy_reward_is_stub(rollout_batch):
    """Untouched stub raises NotImplementedError on numpy inputs."""
    with pytest.raises(NotImplementedError):
        starter.build_proxy_reward(
            rollout_batch["correctness"],
            rollout_batch["format_ok"],
            proxy_reward_weight=0.4,
        )


def test_run_kl_sweep_is_stub(config):
    with pytest.raises(NotImplementedError):
        starter.run_kl_sweep(config)


def test_prepare_divergence_plot_data_is_stub_with_series_input():
    run = {
        "label": "beta=0.001",
        "steps": np.arange(4),
        "proxy_reward": np.linspace(1.0, 9.0, 4),
        "true_accuracy": np.linspace(0.6, 0.2, 4),
    }
    with pytest.raises(NotImplementedError):
        starter.prepare_divergence_plot_data([run])


def test_select_earliest_warning_metric_is_stub():
    run = {
        "steps": np.arange(4),
        "true_accuracy": np.array([0.6, 0.6, 0.4, 0.2]),
        "format_rate": np.array([0.2, 0.8, 0.95, 0.99]),
    }
    with pytest.raises(NotImplementedError):
        starter.select_earliest_warning_metric(
            [run], candidates=("format_rate",), reference_metric="true_accuracy"
        )


def test_no_goodhart_divergence_logic_in_starter():
    """No-solutions guard: divergence/onset detection must not be implemented."""
    src = os.path.join(STARTER_DIR, "starter.py")
    with open(src) as fh:
        body = fh.read()
    banned = [
        "argmax(",
        "np.diff(",
        "np.sign(",
        "onset_step",
        "lead_time =",
        "crossing",
        "goodhart_score",
    ]
    hits = [tok for tok in banned if tok in body]
    assert not hits, f"possible implemented solution logic in starter: {hits}"


# ---------------------------------------------------------------------------
# notebook skeleton invariants
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def notebook():
    path = os.path.join(LAB_DIR, "notebook.ipynb")
    assert os.path.isfile(path), f"missing notebook: {path}"
    with open(path) as fh:
        nb = json.load(fh)
    return nb


def test_notebook_valid_json_with_outputs_cleared(notebook):
    assert notebook.get("nbformat") == 4
    cells = notebook["cells"]
    assert cells, "skeleton notebook should have placeholder cells"
    for cell in cells:
        if cell.get("cell_type") == "code":
            assert cell.get("outputs") == [], (
                f"skeleton code cell must have no outputs: {cell.get('source', '')[:60]!r}"
            )
            assert cell.get("execution_count") is None


def test_notebook_references_the_lab_parts(notebook):
    text = json.dumps(notebook).lower()
    for token in ("divergence", "kl_coef", "proxy_reward"):
        assert token in text, f"notebook skeleton missing concept: {token}"
