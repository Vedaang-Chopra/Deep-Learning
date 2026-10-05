"""Lab 07 tests — GRPO / RLVR capstone.

GPU-free, network-free, torch-free (contract §9.5): numpy fixtures of fake
reward batches (shape [N, K], fixed values) exercise the scaffold's containers,
config plumbing and signature invariants. Algorithm stubs are asserted to RAISE
NotImplementedError, their bodies are asserted to contain no logic, and the
module source is scanned for forbidden solution patterns — the tests double as
the contract §9.4 no-solutions tripwire.
"""

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import starter
from starter import (
    GRPORolloutEngine,
    GroupRecord,
    KL_ESTIMATORS,
    LabConfig,
    MetricsLogger,
    approx_kl_k1,
    approx_kl_k2,
    approx_kl_k3,
    compare_kl_estimators,
    compute_clipped_surrogate,
    compute_drgrpo_advantages,
    compute_grpo_advantages,
    extract_gsm8k_answer,
    grpo_loss,
    handle_zero_contrast_groups,
    load_config,
    training_step,
    verify_correctness,
    verify_format,
)

LAB_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = LAB_DIR / "configs" / "07_grpo_rlvr_capstone.yaml"
STARTER_PATH = LAB_DIR / "starter.py"

# Relative to this lab dir, the answer key lives at:
#   ../../../rlhf-book/code/policy_gradients/...
ANSWER_KEY_DIR = LAB_DIR / ".." / ".." / ".." / "rlhf-book" / "code" / "policy_gradients"


# ---------------------------------------------------------------------------
# Fixtures: fixed fake rewards [N, K] — values chosen by hand, NOT computed
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_rewards() -> np.ndarray:
    """[N=4 prompts, K=8 rollouts] total rewards, hand-picked.

    Row 0: mixed contrast            (ties exist but the group has spread)
    Row 1: ALL TIED                  (zero-contrast group — the GRPO failure mode)
    Row 2: partial ties              (some spread, repeated values)
    Row 3: two tie-blocks            (contrast across blocks only)
    """
    return np.array(
        [
            [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.5],
            [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
            [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def fake_log_probs() -> np.ndarray:
    """[B=4, T=8] token-level log-probs, current vs reference + action mask."""
    rng = np.random.RandomState(42)  # deterministic: same numbers every run
    new = -np.abs(rng.uniform(0.05, 3.0, size=(4, 8)))
    ref = new + rng.normal(0.0, 0.25, size=(4, 8))
    mask = np.ones((4, 8), dtype=np.float64)
    mask[0, :3] = 0.0  # prompt/pad positions must be gated everywhere
    mask[2, 6:] = 0.0
    return new, ref, mask


# ---------------------------------------------------------------------------
# starter.py imports cleanly
# ---------------------------------------------------------------------------


def test_starter_imports_cleanly():
    """Contract: scaffold is importable on a numpy-only laptop (no torch)."""
    assert starter is not None
    source = STARTER_PATH.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_are_valid(self):
        cfg = LabConfig()
        assert cfg.loss_mode == "grpo"
        assert cfg.num_rollouts == 8
        assert cfg.kl_estimator == "kl3"
        assert cfg.zero_contrast_groups == "skip"
        assert cfg.loss_normalization == "sequence_mean"

    def test_drgrpo_mode_accepted(self):
        """Lab 07 (unlike Lab 06) must accept the Dr.GRPO arm at config level."""
        assert LabConfig(loss_mode="drgrpo").loss_mode == "drgrpo"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"loss_mode": "rloo"},  # that was Lab 06's job
            {"loss_mode": "ppo"},
            {"kl_estimator": "kl4"},
            {"num_rollouts": 1},  # a group of one has no contrast
            {"zero_contrast_groups": "drop"},
            {"loss_normalization": "per_token"},
            {"clip_eps_lo": 0.0},
            {"clip_eps_hi": -0.2},
        ],
    )
    def test_invalid_configs_raise(self, kwargs):
        with pytest.raises(ValueError):
            LabConfig(**kwargs)

    def test_load_config_from_lab_yaml(self):
        pytest.importorskip("yaml")
        cfg = load_config(CONFIG_PATH)
        assert isinstance(cfg, LabConfig)
        assert cfg.loss_mode == "grpo"
        assert cfg.num_rollouts == 8
        assert cfg.prompts_per_step == 4
        assert cfg.kl_estimator == "kl3"
        assert cfg.beta == 0.0
        assert cfg.clip_eps_lo == 0.2 and cfg.clip_eps_hi == 0.2
        assert cfg.zero_contrast_groups == "skip"
        assert cfg.model_name == "Qwen/Qwen3-0.6B"
        assert cfg.seed == 42

    def test_unknown_yaml_keys_ignored(self, tmp_path):
        pytest.importorskip("yaml")
        p = tmp_path / "c.yaml"
        p.write_text("loss_mode: drgrpo\nsome_future_knob: 3\n", encoding="utf-8")
        cfg = load_config(p)
        assert cfg.loss_mode == "drgrpo"


# ---------------------------------------------------------------------------
# Group container
# ---------------------------------------------------------------------------


class TestGroupRecord:
    def _group(self, **overrides):
        base = dict(
            prompt="spell_backward: reverse 'abc'",
            completions=[f"completion-{i}" for i in range(8)],
            correctness=[1.0, 0.0] * 4,
            format_scores=[1.0] * 8,
            total_rewards=[1.0, 0.3] * 4,
        )
        base.update(overrides)
        return GroupRecord(**base)

    def test_valid_group(self):
        g = self._group()
        assert g.k == 8

    def test_reward_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self._group(total_rewards=[1.0, 0.3])

    def test_single_completion_group_raises(self):
        """K=1 has no group contrast — must be rejected at construction."""
        with pytest.raises(ValueError):
            self._group(
                completions=["only-one"],
                correctness=[1.0],
                format_scores=[1.0],
                total_rewards=[1.0],
            )

    def test_has_contrast_flag(self):
        tied = self._group(total_rewards=[0.5] * 8)
        spread = self._group()
        assert tied.has_contrast is False
        assert spread.has_contrast is True


# ---------------------------------------------------------------------------
# Fixture invariants (the fixtures themselves, not any implementation)
# ---------------------------------------------------------------------------


class TestFakeRewardFixtures:
    def test_shape_is_n_times_k(self, fake_rewards):
        assert fake_rewards.shape == (4, 8)

    def test_row1_is_zero_contrast(self, fake_rewards):
        row = fake_rewards[1]
        assert row.max() == row.min()  # all-tied group

    def test_other_rows_have_contrast(self, fake_rewards):
        for i in (0, 2, 3):
            assert fake_rewards[i].max() > fake_rewards[i].min()

    def test_values_are_fixed_not_sampled(self, fake_rewards):
        expected = np.array([1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.5])
        assert np.array_equal(fake_rewards[0], expected)


# ---------------------------------------------------------------------------
# Stub behavior: every algorithmic entry point raises NotImplementedError
# ---------------------------------------------------------------------------

STUB_NAMES = [
    "verify_format",
    "verify_correctness",
    "extract_gsm8k_answer",
    "compute_grpo_advantages",
    "compute_drgrpo_advantages",
    "handle_zero_contrast_groups",
    "compute_clipped_surrogate",
    "grpo_loss",
    "approx_kl_k1",
    "approx_kl_k2",
    "approx_kl_k3",
    "sample_group",
    "score_group",
    "collect_batch",
    "log",
    "summarize_batch",
    "training_step",
]


def _call_stub(name, fake_rewards, fake_log_probs):
    """Invoke each stub with contract-valid inputs; nothing should succeed."""
    new, ref, mask = fake_log_probs
    if name == "verify_format":
        verify_format("</think><answer>abc</answer>")
    elif name == "verify_correctness":
        verify_correctness("answer is 42", "42")
    elif name == "extract_gsm8k_answer":
        extract_gsm8k_answer("... so the answer is 42")
    elif name in ("compute_grpo_advantages", "compute_drgrpo_advantages"):
        getattr(starter, name)(fake_rewards)
    elif name == "handle_zero_contrast_groups":
        handle_zero_contrast_groups(fake_rewards.copy(), fake_rewards, mode="skip")
    elif name == "compute_clipped_surrogate":
        compute_clipped_surrogate(new, ref, np.ones_like(new), 0.2, 0.2)
    elif name == "grpo_loss":
        grpo_loss(
            new, ref, np.ones_like(new), ref.copy(), mask,
            0.2, 0.2, beta=0.0, kl_estimator="kl3",
        )
    elif name.startswith("approx_kl_"):
        getattr(starter, name)(new, ref, mask)
    elif name in ("sample_group", "collect_batch"):
        getattr(GRPORolloutEngine(LabConfig()), name)("prompt: x", 8) if name == "sample_group" else getattr(
            GRPORolloutEngine(LabConfig()), name
        )()
    elif name == "score_group":
        GRPORolloutEngine(LabConfig()).score_group("prompt: x", "gold", ["c1", "c2"])
    elif name in ("log", "summarize_batch"):
        logger = MetricsLogger(tmp_metrics_path())
        if name == "log":
            logger.log({"step": 0})
        else:
            logger.summarize_batch(fake_rewards, fake_rewards.copy(), [8] * 4)
    elif name == "training_step":
        training_step(GRPORolloutEngine(LabConfig()), LabConfig(), MetricsLogger(tmp_metrics_path()), 0)
    else:  # pragma: no cover
        raise AssertionError(f"unrouted stub {name}")


def tmp_metrics_path():
    import tempfile

    return str(Path(tempfile.mkdtemp()) / "metrics.jsonl")


@pytest.mark.parametrize("name", STUB_NAMES)
def test_stub_raises_not_implemented(name, fake_rewards, fake_log_probs):
    with pytest.raises(NotImplementedError):
        _call_stub(name, fake_rewards, fake_log_probs)


def test_kl_dispatcher_hits_all_three_stubs(fake_log_probs):
    """compare_kl_estimators is implemented plumbing, but its three targets are stubs."""
    new, ref, mask = fake_log_probs
    with pytest.raises(NotImplementedError):
        compare_kl_estimators(new, ref, mask)


def test_kl_estimator_registry_is_complete():
    assert set(KL_ESTIMATORS) == {"kl1", "kl2", "kl3"}


# ---------------------------------------------------------------------------
# No-solutions tripwire (contract §9.4)
# ---------------------------------------------------------------------------

FORBIDDEN_SOLUTION_TOKENS = [
    "expm1",        # k3's formula shape
    "clamp",        # torch clipping call
    "log_ratio",    # the log-ratio variable every estimator starts from
    ".std(",        # group scale normalization
    ".mean(",       # group location normalization
    "np.std(",
    "np.mean(",
    "np.exp(",
    "np.log(",
    "np.maximum",
    "np.minimum",
    "softmax",
    "sigmoid",
    "torch.min",
]


class TestNoSolutionsTripwire:
    @pytest.fixture(autouse=True)
    def _source(self):
        self.source = STARTER_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_no_forbidden_tokens_in_source(self):
        hits = [tok for tok in FORBIDDEN_SOLUTION_TOKENS if tok in self.source]
        assert hits == [], f"solution-shaped tokens found in starter.py: {hits}"

    def _functions(self):
        return {node.name: node for node in ast.walk(self.tree) if isinstance(node, ast.FunctionDef)}

    def test_stub_bodies_contain_no_logic(self):
        """Each stub's body (docstring stripped) must be a single bare raise."""
        fns = self._functions()
        missing = [n for n in STUB_NAMES if n not in fns]
        assert missing == [], f"expected stubs absent from starter.py: {missing}"
        for name in STUB_NAMES:
            fn = fns[name]
            body = list(fn.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]  # drop docstring
            assert len(body) == 1 and isinstance(body[0], ast.Raise), (
                f"stub {name} contains executable logic beyond a bare raise"
            )

    def test_signature_contracts_mention_batch_shapes(self):
        """Contract docstrings must carry the [N, K] / [B, T] shape language."""
        for fn_name, needle in [
            ("compute_grpo_advantages", "[N, K]"),
            ("compute_drgrpo_advantages", "[N, K]"),
            ("compute_clipped_surrogate", "[B, T]"),
            ("approx_kl_k1", "[B, T]"),
        ]:
            src = ast.get_source_segment(self.source, self._functions()[fn_name])
            assert needle in src, f"{fn_name} docstring missing shape contract {needle}"
