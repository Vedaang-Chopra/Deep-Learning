"""Lab 10 structural tests -- CPU-only, numpy-only fixtures, no network.

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
# Fixtures (fabricated groups with KNOWN correctness structure)
# ---------------------------------------------------------------------------


@pytest.fixture
def groups() -> list:
    return starter.make_fixture_groups()


@pytest.fixture
def config() -> dict:
    cfg_path = LAB_ROOT / "configs" / "10_sdpo_distillation.yaml"
    return starter.load_config(str(cfg_path))


@pytest.fixture
def collector(config) -> starter.GroupCollector:
    return starter.GroupCollector(
        num_rollouts=config["num_rollouts"],
        prompts_per_step=config["prompts_per_step"],
        success_reward_threshold=config["success_reward_threshold"],
        max_polls=64,
    )


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


# ---------------------------------------------------------------------------
# Config parsing + answer-key parity invariants
# ---------------------------------------------------------------------------


def test_config_parses_and_mirrors_sdpo_defaults(config):
    """The lab config must mirror the repo's sdpo.yaml defaults (values the
    loss/loop stubs are documented against)."""
    assert config["model_name"] == "Qwen/Qwen3-1.7B"
    assert config["loss"] == "sdpo"
    assert config["kl_top_k"] == 20
    assert config["success_reward_threshold"] == 1.0
    assert config["rollout_chunk"] >= 1
    # generation block
    assert config["temperature"] == 0.6
    assert config["max_new_tokens"] == 512
    assert config["max_prompt_len"] == 512  # student prompt cap
    assert config["max_reprompt_len"] == 1024  # teacher reprompt cap
    assert config["enable_thinking"] is True
    # training block
    assert config["num_rollouts"] == 8
    assert config["prompts_per_step"] == 16
    assert config["lr"] == 1.0e-6
    assert config["warmup_ratio"] == 0.0
    assert config["num_steps"] > 0
    assert config["seed"] == 42
    starter.validate_config(config)  # idempotent after load_config's own check


def test_config_data_is_spell_backward(config):
    specs = config["data"]["specs"]
    names = [s["name"] for s in specs]
    assert "spell_backward" in names
    sb = next(s for s in specs if s["name"] == "spell_backward")
    assert sb["config"]["min_word_len"] >= 1
    assert sb["config"]["max_word_len"] >= sb["config"]["min_word_len"]
    # metrics artifact path is configured (notebook plots from JSONL artifacts)
    assert isinstance(config["metrics_path"], str) and config["metrics_path"].endswith(".jsonl")


def test_validate_config_rejects_bad_input(config):
    bad_topk = json.loads(json.dumps(config))
    bad_topk["kl_top_k"] = 0
    with pytest.raises(ValueError):
        starter.validate_config(bad_topk)

    no_model = json.loads(json.dumps(config))
    del no_model["model_name"]
    with pytest.raises(ValueError):
        starter.validate_config(no_model)

    bad_rollouts = json.loads(json.dumps(config))
    bad_rollouts["num_rollouts"] = 0
    with pytest.raises(ValueError):
        starter.validate_config(bad_rollouts)

    bad_threshold = json.loads(json.dumps(config))
    bad_threshold["success_reward_threshold"] = 1.5
    with pytest.raises(ValueError):
        starter.validate_config(bad_threshold)

    bad_temp = json.loads(json.dumps(config))
    bad_temp["temperature"] = -1.0
    with pytest.raises(ValueError):
        starter.validate_config(bad_temp)

    no_task = json.loads(json.dumps(config))
    no_task["data"]["specs"] = [{"name": "gsm8k"}]
    with pytest.raises(ValueError):
        starter.validate_config(no_task)


# ---------------------------------------------------------------------------
# Fixture container invariants (the fabricated data has KNOWN structure)
# ---------------------------------------------------------------------------


def test_fixture_groups_container_and_structure(groups):
    assert len(groups) == len(starter.SPELL_BACKWARD_WORDS) == 3
    for g, word in zip(groups, starter.SPELL_BACKWARD_WORDS):
        assert sorted(g.keys()) == sorted(starter.ROLLOUT_GROUP_KEYS)
        assert g["target"] == word[::-1]  # spell_backward target
        assert len(g["completions"]) == len(g["rewards"]) == starter.FIXTURE_NUM_ROLLOUTS
        # verifiable reward is strictly binary 0/1 in this lab
        rewards = np.asarray(g["rewards"], dtype=np.float64)
        assert rewards.shape == (starter.FIXTURE_NUM_ROLLOUTS,)
        assert set(rewards.tolist()) <= {0.0, 1.0}
        # completions flagged correct carry the target verbatim, wrong ones do not
        for comp, r in zip(g["completions"], g["rewards"]):
            assert (comp == g["target"]) == (float(r) >= 1.0)


def test_fixture_groups_known_skip_pattern(groups):
    """Exactly one fixture group is all-wrong (the skip-and-refill case) and
    the others have the documented correct-sibling counts."""
    n_correct = [int(np.sum(g["rewards"])) for g in groups]
    assert n_correct == [1, 0, 3]
    full_flags = [starter.group_is_full(g, 1.0) for g in groups]
    assert full_flags == [True, False, True]


def test_validate_rollout_group_rejects_malformed(groups):
    starter.validate_rollout_group(groups[0])  # must not raise

    bad_len = dict(groups[0])
    bad_len["rewards"] = bad_len["rewards"][:-1]
    with pytest.raises(ValueError):
        starter.validate_rollout_group(bad_len)

    missing = {k: v for k, v in groups[0].items() if k != "target"}
    with pytest.raises(ValueError):
        starter.validate_rollout_group(missing)

    nonnum = dict(groups[0])
    nonnum["rewards"] = list(nonnum["rewards"])
    nonnum["rewards"][2] = "correct"
    with pytest.raises(ValueError):
        starter.validate_rollout_group(nonnum)

    empty = dict(groups[0])
    empty["completions"], empty["rewards"] = [], []
    with pytest.raises(ValueError):
        starter.validate_rollout_group(empty)


# ---------------------------------------------------------------------------
# GroupCollector: skip-and-refill BOOKKEEPING (fields) vs refill (stub)
# ---------------------------------------------------------------------------


def test_collector_bookkeeping_keeps_full_and_counts_skipped(groups, collector):
    assert collector.skipped == 0  # 'skipped' counter exists as a field
    assert collector.groups == []
    # group 0 (1 correct) and group 2 (3 correct) are kept; group 1 is skipped
    kept = [
        collector.record_group(g["question"], g["target"], g["completions"], g["rewards"])
        for g in groups
    ]
    assert kept == [True, False, True]
    assert len(collector.groups) == 2
    assert collector.skipped == 1  # exactly the all-wrong group


def test_collector_rejects_malformed_group(groups, collector):
    bad = dict(groups[0])
    bad["rewards"] = bad["rewards"][:3]
    with pytest.raises(ValueError):
        collector.record_group(bad["question"], bad["target"], bad["completions"], bad["rewards"])
    assert collector.skipped == 0  # malformed != skipped: rejected before counting
    assert collector.groups == []


def test_skip_and_refill_orchestration_is_stub(collector):
    """The polling/refill logic is unimplemented (no-solutions rule) but the
    plumbing around it -- threshold predicate, skipped counter, prompt budget --
    must be real."""
    generate_fn = lambda q, n: ["x"] * n  # noqa: E731
    sample_prompt_fn = lambda: {"question": "q", "target": "t"}  # noqa: E731
    with pytest.raises(NotImplementedError):
        collector.collect_full_groups(generate_fn, sample_prompt_fn)
    # bookkeeping fields survived the failed call untouched
    assert collector.skipped == 0
    assert collector.groups == []


# ---------------------------------------------------------------------------
# Teaching stubs raise NotImplementedError (no solution logic shipped)
# ---------------------------------------------------------------------------


def test_verifier_stub_raises():
    with pytest.raises(NotImplementedError):
        starter.spell_backward_reward("Spell the word 'apple' backwards.", "elppa")


def test_teacher_prompt_stub_raises():
    with pytest.raises(NotImplementedError):
        starter.build_teacher_prompt("Spell 'apple' backwards.", "elppa")
    # the prompt skeleton constants exist (README pins the surface form)
    assert starter.TEACHER_DEMO_HEADER == "Correct solution:\n\n"
    assert starter.TEACHER_TASK_SUFFIX == "Correctly solve the original question."


def test_topk_reversekl_stubs_raise(config, groups):
    loss = starter.TopKReverseKL(
        kl_top_k=config["kl_top_k"], rollout_chunk=config["rollout_chunk"]
    )
    assert loss.kl_top_k == 20 and loss.rollout_chunk == 4

    # numpy stand-in for a torch [..., K] log-prob tensor: the stub must raise
    # before any (absent) torch machinery is touched
    fake_logp = np.log(np.full((2, 3, 4), 0.25))
    with pytest.raises(NotImplementedError):
        loss.add_tail_bucket(fake_logp)

    fake_batch = {
        "s_ids": np.zeros((2, 6), dtype=np.int64),
        "t_ids": np.zeros((2, 8), dtype=np.int64),
        "s_mask": np.ones((2, 6)),
        "t_mask": np.ones((2, 8)),
        "action_mask": np.ones((2, 3)),
    }
    fake_model = object()
    with pytest.raises(NotImplementedError):
        loss._chunk_loss(fake_model, fake_batch, slice(0, 2), A=3, denom=6.0)
    with pytest.raises(NotImplementedError):
        loss.accumulate(fake_model, fake_batch, scale=1.0)


def test_topk_reversekl_rejects_bad_construction():
    with pytest.raises(ValueError):
        starter.TopKReverseKL(kl_top_k=0)
    with pytest.raises(ValueError):
        starter.TopKReverseKL(kl_top_k=20, rollout_chunk=0)


def test_training_loop_stubs_raise(config, groups, collector):
    loss = starter.TopKReverseKL(kl_top_k=config["kl_top_k"])
    with pytest.raises(NotImplementedError):
        starter.run_training_step(object(), collector, loss, config, step=0)
    with pytest.raises(NotImplementedError):
        starter.run_training_loop(object(), config)


# ---------------------------------------------------------------------------
# Metrics contract: shape validation given, loop stubbed
# ---------------------------------------------------------------------------


def test_metrics_record_validation_accepts_good():
    rec = {
        "step": 0,
        "reward": 0.125,
        "distill_loss": 1.234,
        "skipped": 5,
        "skipped_rate": 5 / 21,
    }
    starter.validate_metrics_record(rec)  # must not raise


def test_metrics_record_validation_rejects_malformed():
    good = {
        "step": 0,
        "reward": 0.125,
        "distill_loss": 1.234,
        "skipped": 5,
        "skipped_rate": 5 / 21,
    }
    missing = dict(good)
    del missing["skipped_rate"]
    with pytest.raises(ValueError):
        starter.validate_metrics_record(missing)

    extra = dict(good)
    extra["wandb"] = "on"
    with pytest.raises(ValueError):
        starter.validate_metrics_record(extra)

    bad_step = dict(good)
    bad_step["step"] = -1
    with pytest.raises(ValueError):
        starter.validate_metrics_record(bad_step)

    nonnum = dict(good)
    nonnum["distill_loss"] = "low"
    with pytest.raises(ValueError):
        starter.validate_metrics_record(nonnum)
