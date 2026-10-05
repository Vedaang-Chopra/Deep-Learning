"""Lab 09 structural tests -- CPU-only, numpy-only fixtures, no network.

Asserts container/config invariants and that every teaching stub raises
NotImplementedError. Per contract 9.5: collects and runs without GPU or
network; torch-free at module level (enforced via subprocess import check).

Run:
    python3 -m pytest tests/ -q
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

import starter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (fabricated scores/rewards with known ordering)
# ---------------------------------------------------------------------------

@pytest.fixture
def records() -> list:
    return starter.make_fixture_records()


@pytest.fixture
def config() -> dict:
    cfg_path = LAB_ROOT / "configs" / "09_rejection_sampling_sft.yaml"
    return starter.load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# Module-level hygiene
# ---------------------------------------------------------------------------

def test_starter_imports_torch_free():
    """starter.py imports cleanly in a fresh interpreter and pulls in NO torch
    stack (contract: torch-free at module level)."""
    code = (
        "import sys; sys.path.insert(0, {!r}); "
        "import starter; "
        "assert 'torch' not in sys.modules, 'torch imported at module level'; "
        "assert 'transformers' not in sys.modules, 'transformers imported at module level'"
    ).format(str(LAB_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_strategy_registry_invariants():
    assert len(starter.STRATEGIES) == 4
    assert len(set(starter.STRATEGIES)) == 4
    top_arms = {s for s in starter.STRATEGIES if s.startswith("top_")}
    control_arms = {s for s in starter.STRATEGIES if s.startswith("random_")}
    assert len(top_arms) == len(control_arms) == 2
    # every arm's paired_control is itself a valid strategy and pairing is symmetric
    for s in starter.STRATEGIES:
        ctrl = starter.paired_control(s)
        assert ctrl in starter.STRATEGIES
        assert (ctrl.startswith("top_")) != (s.startswith("top_"))
        assert starter.paired_control(ctrl) == s


# ---------------------------------------------------------------------------
# Fixture ordering sanity (the fabricated data has KNOWN ordering)
# ---------------------------------------------------------------------------

def test_fixture_records_known_ordering(records):
    """The fixture matrix carries documented argmax / ranking structure that
    future implementations must reproduce; pin the data now."""
    mat = np.asarray([r["rewards"] for r in records], dtype=np.float64)
    assert mat.shape == (5, 4)
    # row argmaxes are the chapter's worked example: [0, 1, 0, 2, 3]
    assert [int(np.argmax(row)) for row in mat] == [0, 1, 0, 2, 3]
    # global flat ranking of rewards 1-2 ranks: 0.9 at row 2 col 0, then 0.8 rows 1 & 3
    flat = mat.ravel()
    order = np.argsort(-flat)
    best = np.unravel_index(int(order[0]), mat.shape)
    assert best == (2, 0) and mat[best] == 0.9
    second = np.unravel_index(int(order[1]), mat.shape)
    assert mat[second] == 0.8 and second[0] in (1, 3)
    # cache-contract keys + gold answers present
    for rec in records:
        assert sorted(rec.keys()) == sorted(starter.CACHE_RECORD_KEYS)


# ---------------------------------------------------------------------------
# Record-shape validation (implemented plumbing)
# ---------------------------------------------------------------------------

def test_validate_rollout_records_accepts_good(records):
    starter.validate_rollout_records(records)  # must not raise


def test_validate_rollout_records_rejects_malformed(records):
    bad_len = dict(records[0])
    bad_len["rewards"] = bad_len["rewards"][:-1]
    with pytest.raises(ValueError):
        starter.validate_rollout_records([records[0], bad_len])

    missing = {k: v for k, v in records[0].items() if k != "answer"}
    with pytest.raises(ValueError):
        starter.validate_rollout_records([missing])

    nonnum = dict(records[0])
    nonnum["rewards"] = list(nonnum["rewards"])[:4]
    nonnum["rewards"][2] = "high"
    with pytest.raises(ValueError):
        starter.validate_rollout_records([nonnum])

    empty = dict(records[0])
    empty["completions"], empty["rewards"] = [], []
    with pytest.raises(ValueError):
        starter.validate_rollout_records([empty])


def test_rewards_matrix_shape_and_ragged_guard(records):
    mat = starter.rewards_matrix(records)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (5, 4) and mat.dtype == np.float64
    ragged = [dict(r) for r in records]
    ragged.append(dict(records[0]))
    ragged[-1]["rewards"] = ragged[-1]["rewards"][:3]
    ragged[-1]["completions"] = ragged[-1]["completions"][:3]
    with pytest.raises(ValueError):
        starter.rewards_matrix(ragged)


# ---------------------------------------------------------------------------
# Config parsing + shared-cache invariant
# ---------------------------------------------------------------------------

def test_config_parses_and_sections_valid(config):
    assert config["data"]["name"] == "openai/gsm8k"
    assert config["selection"]["strategy"] in starter.STRATEGIES
    assert config["scorer"]["kind"] in ("lab03_bt", "hf_rm")
    starter.validate_config(config)  # idempotent after load_config's own check


def test_config_shared_generation_params_across_arms(config):
    """The repo's key trick: all four arms share Stage 1+2 so exactly one
    rollout cache exists. The scaffold encodes that as one shared generation
    block + per-arm selection budget metadata."""
    n = config["num_completions_per_prompt"]
    assert isinstance(n, int) and n >= 1
    gen_keys = ("temperature", "top_p", "max_new_tokens")
    for k in gen_keys:
        assert k in config, f"shared generation param {k} missing"
    sel = config["selection"]
    assert {"strategy", "top_k"} <= set(sel.keys())
    assert isinstance(sel["top_k"], int) and sel["top_k"] >= 1
    # dataset slice matches the reference-scale setup (M=1000 train / 200 test)
    assert config["data"]["max_train_samples"] == 1000
    assert config["data"]["max_test_samples"] == 200


def test_validate_config_rejects_bad_input(config):
    bad_strategy = json.loads(json.dumps(config))
    bad_strategy["selection"]["strategy"] = "greedy_gremlin"
    with pytest.raises(ValueError):
        starter.validate_config(bad_strategy)

    no_scorer = json.loads(json.dumps(config))
    del no_scorer["scorer"]
    with pytest.raises(ValueError):
        starter.validate_config(no_scorer)

    bad_n = json.loads(json.dumps(config))
    bad_n["num_completions_per_prompt"] = 0
    with pytest.raises(ValueError):
        starter.validate_config(bad_n)

    bad_temp = json.loads(json.dumps(config))
    bad_temp["temperature"] = -1.0
    with pytest.raises(ValueError):
        starter.validate_config(bad_temp)


def test_run_selection_dispatch_routing(records):
    """Dispatch validates strategies BEFORE touching (stubbed) mechanics."""
    with pytest.raises(ValueError):
        starter.run_selection(records, "greedy_gremlin")  # unknown -> ValueError
    with pytest.raises(NotImplementedError):
        starter.run_selection(records, starter.SELECT_TOP_PER_PROMPT)
    with pytest.raises(NotImplementedError):
        starter.run_selection(
            records, starter.SELECT_TOP_K_OVERALL, top_k=3
        )
    with pytest.raises(ValueError):
        # *_k_overall routing demands an explicit budget before selecting
        starter.run_selection(records, starter.SELECT_TOP_K_OVERALL, top_k=None)
    with pytest.raises(NotImplementedError):
        starter.run_selection(records, starter.SELECT_RANDOM_PER_PROMPT, seed=42)
    with pytest.raises(NotImplementedError):
        starter.run_selection(records, starter.SELECT_RANDOM_K_OVERALL, top_k=3, seed=42)


# ---------------------------------------------------------------------------
# Teaching stubs raise NotImplementedError (no selection logic shipped)
# ---------------------------------------------------------------------------

def test_selection_stubs_raise(records):
    for call in (
        lambda: starter.select_top_per_prompt(records),
        lambda: starter.select_random_per_prompt(records, seed=42),
        lambda: starter.select_top_k_overall(records, k=3),
        lambda: starter.select_random_k_overall(records, k=3, seed=42),
        lambda: starter.load_scored_rollouts("nonexistent.jsonl"),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_scoring_bridge_stub_raises(records):
    bridge = starter.RMScoreBridge()
    questions = [r["question"] for r in records]
    completions = [c for r in records for c in r["completions"]]
    questions_flat = [q for q, c in zip(questions, completions)]
    # scoring interface must raise NotImplementedError, not return fabricated scores
    with pytest.raises(NotImplementedError):
        bridge.score(questions_flat, completions, batch_size=2)


def test_exact_match_stubs_raise():
    with pytest.raises(NotImplementedError):
        starter.extract_gsm8k_answer("The answer is #### 42")
    with pytest.raises(NotImplementedError):
        starter.answers_match("42", "42")
    with pytest.raises(NotImplementedError):
        starter.evaluate_exact_match(lambda qs: ["#### 42"] * len(qs), ["q"], ["42"])


def test_comparison_table_builder_stub_raises(records):
    fake_result = {
        "strategy": starter.SELECT_TOP_PER_PROMPT,
        "n_pairs": 100,
        "test_accuracy": 0.5,
    }
    with pytest.raises(NotImplementedError):
        starter.build_comparison_table([fake_result])
