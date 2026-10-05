"""Lab 06 tests — REINFORCE/RLOO on an LLM.

GPU-free, network-free, torch-free (contract §9.5): numpy fixtures of fake
rollouts (list-of-str with predetermined rewards) exercise the scaffold's
containers, config plumbing and signature invariants. Algorithm stubs are
asserted to RAISE NotImplementedError — the tests are also a no-solutions tripwire.
"""

import inspect
import json
import re
from pathlib import Path
from typing import List, get_type_hints

import pytest

import starter
from starter import (
    Experience,
    LabConfig,
    MetricsLogger,
    RolloutEngine,
    approx_kl_k1,
    check_kl_drift,
    compute_importance_ratios,
    compute_reinforce_advantages,
    compute_rloo_advantages,
    extract_gsm8k_answer,
    load_config,
    verify_correctness,
    verify_format,
)

LAB_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = LAB_DIR / "configs" / "06_reinforce_rloo_llm.yaml"


# ---------------------------------------------------------------------------
# Fixtures: fake rollouts as list-of-str with predetermined outcomes
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_completions() -> list:
    """K=4 completions for one spell_backward prompt with distinct fates."""
    return [
        "<think>reverse each word</think><answer>cba fed</answer>",   # correct + formatted
        "cba fed",                                                      # correct, unformatted
        "<think>...</think><answer>abc def</answer>",                   # wrong but formatted
        "blah blah never closes",                                       # neither
    ]


@pytest.fixture
def fake_group(fake_completions) -> Experience:
    """Experience built from predetermined per-completion outcomes.

    Reward map (fixed in the fixture, NOT computed by any scaffold code):
        index 0: correct + formatted   -> total 1.0
        index 1: correct only          -> total 0.7
        index 2: formatted only        -> total 0.3
        index 3: nothing               -> total 0.0
    """
    correctness = [1.0, 1.0, 0.0, 0.0]
    fmt = [1.0, 0.0, 1.0, 0.0]
    totals = [1.0, 0.7, 0.3, 0.0]
    return Experience(
        prompt="spell_backward: emit 'cba fed' reversed words",
        completions=fake_completions,
        correctness=correctness,
        format_scores=fmt,
        total_rewards=totals,
        old_log_probs=[-18.2, -9.5, -21.0, -12.3],
    )


# ---------------------------------------------------------------------------
# Experience-like container
# ---------------------------------------------------------------------------


class TestExperienceContainer:
    def test_initializes_with_matching_lengths(self, fake_group):
        assert fake_group.k == 4
        assert len(fake_group.completions) == 4
        assert fake_group.advantages is None, "advantages must be produced by estimators, not scoring"
        assert fake_group.old_log_probs == [-18.2, -9.5, -21.0, -12.3]
        for comp in fake_group.completions:
            assert isinstance(comp, str)

    def test_all_reward_components_align(self, fake_group):
        n = fake_group.k
        assert len(fake_group.correctness) == n
        assert len(fake_group.format_scores) == n
        assert len(fake_group.total_rewards) == n

    @pytest.mark.parametrize("field_name", ["correctness", "format_scores", "total_rewards"])
    def test_mismatched_reward_length_rejected(self, fake_group, field_name):
        kwargs = dict(
            prompt="p",
            completions=["a", "b"],
            correctness=[1.0, 0.0],
            format_scores=[1.0, 0.0],
            total_rewards=[1.0, 0.0],
        )
        kwargs[field_name] = [1.0]  # wrong length on purpose
        with pytest.raises(ValueError):
            Experience(**kwargs)

    def test_empty_group_rejected(self):
        with pytest.raises(ValueError):
            Experience(prompt="p", completions=[], correctness=[], format_scores=[], total_rewards=[])


# ---------------------------------------------------------------------------
# Verifier signature invariants ("verifier returns bool")
# ---------------------------------------------------------------------------


class TestVerifierSignatures:
    def test_verify_format_returns_bool_annotation(self):
        hints = get_type_hints(verify_format)
        assert hints["return"] is bool

    def test_verify_correctness_returns_bool_annotation(self):
        hints = get_type_hints(verify_correctness)
        assert hints["return"] is bool

    def test_verify_correctness_takes_completion_and_answer(self):
        params = list(inspect.signature(verify_correctness).parameters)
        assert params[:2] == ["completion", "answer"]

    def test_verify_format_takes_single_completion_string(self):
        sig = inspect.signature(verify_format).parameters["completion"]
        assert get_type_hints(verify_format)["completion"] is str

    def test_extract_gsm8k_answer_optional_str(self):
        from typing import Optional

        hints = get_type_hints(extract_gsm8k_answer)
        assert hints["return"] == Optional[str]

    # --- no-solutions tripwire: the regex itself must not exist yet -------

    def test_gsm8k_extraction_is_stubbed(self):
        with pytest.raises(NotImplementedError):
            extract_gsm8k_answer("<answer>42</answer>")

    def test_verifiers_are_stubbed_until_student_writes_them(self):
        with pytest.raises(NotImplementedError):
            verify_format("<answer>x</answer>")
        with pytest.raises(NotImplementedError):
            verify_correctness("cba fed", "cba fed")


# ---------------------------------------------------------------------------
# Advantage estimators & KL monitor: signatures + stubbing
# ---------------------------------------------------------------------------


class TestEstimatorSignatures:
    def test_reinforce_estimator_shape_contract(self):
        hints = get_type_hints(compute_reinforce_advantages)
        assert hints["return"] is List[float]
        params = inspect.signature(compute_reinforce_advantages).parameters
        assert list(params) == ["total_rewards"]

    def test_rloo_estimator_shape_contract(self):
        hints = get_type_hints(compute_rloo_advantages)
        assert hints["return"] is List[float]
        params = inspect.signature(compute_rloo_advantages).parameters
        assert list(params) == ["total_rewards"]

    def test_kl_monitor_signature(self):
        hints = get_type_hints(approx_kl_k1)
        assert hints["return"] is float
        params = list(inspect.signature(approx_kl_k1).parameters)
        assert params == ["log_probs_current", "log_probs_ref"]

    @pytest.mark.parametrize(
        "fn,args",
        [
            (compute_reinforce_advantages, ([1.0, 0.0],)),
            (compute_rloo_advantages, ([1.0, 0.0],)),
            (approx_kl_k1, ((-0.1, -0.2), (-0.15, -0.2))),
            (compute_importance_ratios, ((-0.1, -0.2), (-0.1, -0.2))),
        ],
    )
    def test_estimators_not_preimplemented(self, fn, args):
        """No-solutions rule: advantage/KL arithmetic ships empty."""
        with pytest.raises(NotImplementedError):
            fn(*args)


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


class TestLabConfig:
    def test_defaults_match_curriculum(self):
        cfg = LabConfig()
        assert cfg.loss_mode == "reinforce"
        assert cfg.kl_estimator == "kl1"
        assert cfg.model_name == "Qwen/Qwen3-0.6B"

    def test_invalid_loss_mode_rejected(self):
        with pytest.raises(ValueError):
            LabConfig(loss_mode="grpo")  # that is Lab 07's job

    def test_yaml_roundtrip(self):
        cfg = load_config(CONFIG_PATH)
        assert cfg.loss_mode == "reinforce"
        assert cfg.num_rollouts >= 1
        assert cfg.data_size == 15000
        assert cfg.gsm8k_max_examples == 1000
        assert cfg.metrics_path.endswith(".jsonl")

    def test_config_file_exists_in_contract_layout(self):
        assert CONFIG_PATH.is_file()


# ---------------------------------------------------------------------------
# Rollout engine scaffolding
# ---------------------------------------------------------------------------


class TestRolloutEngineScaffold:
    def test_builds_sampling_kwargs_from_cfg(self):
        cfg = LabConfig(num_rollouts=4, temperature=1.0)
        gen = RolloutEngine(cfg).build_generation_config()
        assert gen["temperature"] == 1.0
        assert gen["top_p"] == cfg.top_p
        assert gen["max_new_tokens"] == cfg.max_new_tokens
        assert gen["do_sample"] is True

    def test_engine_defaults_are_stubs(self):
        eng = RolloutEngine(LabConfig())
        with pytest.raises(NotImplementedError):
            eng.sample_group("prompt", k=4)

    def test_score_and_pack_does_not_compute_advantages(self):
        """Group packing may verify/score, but advantages come ONLY from estimators."""
        assert not hasattr(RolloutEngine(LabConfig()), "_compute_advantage_arithmetic")


# ---------------------------------------------------------------------------
# Metrics logger convention
# ---------------------------------------------------------------------------


class TestMetricsLoggerConvention:
    def test_record_freshly_flagged_steps_raise_until_implemented(self, tmp_path):
        log = MetricsLogger(tmp_path / "metrics.jsonl")
        with pytest.raises(NotImplementedError):
            log.log({"step": 0})

    def test_tmp_path_ready_for_student_implementation(self, tmp_path):
        """Sanity: the JSONL directory layout the student's log() writes into."""
        log = MetricsLogger(tmp_path / "runs" / "metrics.jsonl")
        assert log.path.parent.name == "runs"


# ---------------------------------------------------------------------------
# No-solutions source scan (forbidden-pattern review aid, contract §9.4)
# ---------------------------------------------------------------------------


# Concrete solution-content tokens that must never appear in the scaffold's
# executable code (a broader heuristic scanner would fight legitimate
# docstring explanations; stubs-raising tests above cover behavior).
FORBIDDEN_IN_STARTER = [
    "import torch",
    "from torch",
    "(K - 1)",
    "/ (K-1)",
    "K/(K",
    "np.std",
    "re.compile",
    ".mean(dim=0)",
    "rewards.mean(",
]


def _starter_source() -> str:
    return (LAB_DIR / "starter.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("token", FORBIDDEN_IN_STARTER)
def test_no_solutions_rule_source_scan(token):
    src = _starter_source()
    # 'import torch' / 'from torch' may legitimately appear ONLY inside the
    # module docstring prose ("torch is intentionally NOT imported"); every
    # other token must be fully absent.
    if token in ("import torch", "from torch"):
        code_only = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith("#") or "torch" not in line.lower()
        )
        body_lines = [ln for ln in code_only.splitlines()]
        stripped = []
        in_doc = False
        for ln in body_lines:
            count = ln.count('"""')
            if count % 2 == 1:
                in_doc = not in_doc
                continue
            if not in_doc:
                stripped.append(ln)
        assert token not in "\n".join(stripped), f"forbidden token {token!r} in executable code"
    else:
        assert token not in src, f"forbidden token {token!r} found anywhere in scaffold"


def test_torch_free_module_level():
    src = _starter_source()
    header = src.split("# ------")[0] if "# ------" in src else src.split("\n\n\n")[0]
    assert not re.search(r"^\s*(import torch|from torch)", src, re.MULTILINE), \
        "starter must stay importable without torch installed (contract §9.5)"


def test_lab_directory_matches_contract_layout():
    expected = {
        "README.md": True,
        "starter.py": True,
        "notebook.ipynb": True,
        "configs/06_reinforce_rloo_llm.yaml": True,
        "tests/test_06_reinforce_rloo_llm.py": True,
    }
    for rel in expected:
        assert (LAB_DIR / rel).is_file(), f"missing contract file {rel}"


def test_notebook_is_valid_json_without_outputs():
    nb = json.loads((LAB_DIR / "notebook.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], "notebook skeleton must ship without outputs"
