"""Tests for Lab 04 — DPO From Scratch.

Contract (curriculum plan §9):
  * GPU-free and network-free.
  * torch-free at module level — the suite imports and runs without torch
    installed (tensor tests self-skip via pytest.importorskip inside fixtures).
  * These tests assert STRUCTURAL INVARIANTS of hand-built fixtures and
    scaffolding integrity only. They never verify algorithm implementations,
    because the starter contains no solutions to verify.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(HERE)
STARTER_PATH = os.path.join(LAB_DIR, "starter.py")
NOTEBOOK_PATH = os.path.join(LAB_DIR, "notebook.ipynb")
CONFIG_PATH = os.path.join(LAB_DIR, "configs", "04_dpo_from_scratch.yaml")


# ---------------------------------------------------------------------------
# Scaffolding integrity (run everywhere: CPU-only, no torch, no network)
# ---------------------------------------------------------------------------


def _load_starter_module():
    spec = importlib.util.spec_from_file_location("lab04_starter_under_test", STARTER_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses/annotations resolve the module correctly.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_starter_imports_cleanly_and_exposes_contract_functions():
    """starter.py must import WITHOUT torch present and expose all stubs."""
    mod = _load_starter_module()

    for name in (
        "PreferenceBatch",
        "tokenize_preference_pair",
        "collate_preference_batch",
        "make_paired_dataloader",
        "sequence_logprob",
        "dpo_loss",
        "compute_metrics",
        "forward_reference_model",
    ):
        assert hasattr(mod, name), f"starter.py is missing required symbol: {name}"


def test_starter_contains_no_solution_code():
    """No-solutions rule: forbidden shortcut patterns must not appear."""
    with open(STARTER_PATH, encoding="utf-8") as fh:
        source = fh.read().lower()
    for forbidden in ("logsigmoid", "softplus"):
        assert forbidden not in source


def test_notebook_is_valid_json_skeleton_with_no_outputs():
    with open(NOTEBOOK_PATH, encoding="utf-8") as fh:
        nb = json.load(fh)
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and nb["cells"]
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("outputs") == [], "code cells must ship with no outputs"
            assert cell.get("execution_count") is None


def _parse_simple_yaml(path):
    """Minimal key/value reader so this test needs no third-party yaml pkg.

    Only understands top-level 'key: value' lines (what our config uses).
    """
    parsed = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            parsed[key.strip()] = value.strip().strip("'\"")
    return parsed


def test_config_defines_required_training_keys():
    cfg = _parse_simple_yaml(CONFIG_PATH)
    for key in (
        "model_name",
        "dataset_name",
        "loss",
        "beta",
        "learning_rate",
        "num_epochs",
        "batch_size",
        "max_length",
        "seed",
    ):
        assert key in cfg, f"config is missing required key: {key}"
    assert cfg["loss"] == "dpo"
    assert float(cfg["beta"]) > 0.0


# ---------------------------------------------------------------------------
# Tensor-fixture invariants (skipped automatically when torch is absent)
# ---------------------------------------------------------------------------

PAD_ID = 0
PROMPT_TOKENS = [10, 11, 12]
HEADER_TOKEN = 99  # e.g. the assistant-turn-start special token from Lab 00
CHOSEN_RESPONSE = [70, 71]
REJECTED_RESPONSE = [80, 81]
EOS_ID = 1

SEQ_LEN = len(PROMPT_TOKENS) + 1 + 2 + 1 + 1  # prompt + header + resp + EOS + pad
PROMPT_LEN = len(PROMPT_TOKENS) + 1  # prompt tokens + shared header token


@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch")


def build_handmade_pair(torch_mod):
    """Hand-built preference pair with every token position known explicitly.

    Layout per side (seq_len=8):
        positions 0..2 : prompt tokens      [10, 11, 12]
        position  3    : shared header token [99] (assistant-turn start)
        positions 4..5 : response tokens (chosen vs rejected differ)
        position  6    : EOS token          [1]
        position  7    : padding            [PAD_ID]
    """
    t = torch_mod

    chosen_ids = t.tensor(PROMPT_TOKENS + [HEADER_TOKEN] + CHOSEN_RESPONSE + [EOS_ID, PAD_ID])
    rejected_ids = t.tensor(PROMPT_TOKENS + [HEADER_TOKEN] + REJECTED_RESPONSE + [EOS_ID, PAD_ID])
    attention = t.tensor([1] * (SEQ_LEN - 1) + [0])
    response_mask = t.tensor([0] * PROMPT_LEN + [1, 1, 1] + [0])

    return {
        "chosen_input_ids": chosen_ids.unsqueeze(0),
        "rejected_input_ids": rejected_ids.unsqueeze(0),
        "chosen_attention_mask": attention.unsqueeze(0),
        "rejected_attention_mask": attention.clone().unsqueeze(0),
        "chosen_response_mask": response_mask.clone().unsqueeze(0),
        "rejected_response_mask": response_mask.clone().unsqueeze(0),
        "prompt_len": PROMPT_LEN,
    }


def test_response_mask_separates_prompt_from_response_positions(torch_mod):
    """Core invariant: mask must cover ONLY response+EOS, never prompt/pad."""
    pair = build_handmade_pair(torch_mod)
    for side in ("chosen", "rejected"):
        ids = pair[f"{side}_input_ids"]
        mask = pair[f"{side}_response_mask"]
        plen = pair["prompt_len"]

        # Prompt region untouched.
        assert (mask[:, :plen] == 0).all(), f"{side}: response mask covers prompt tokens"

        # Padding never supervised.
        for row, col in (ids == PAD_ID).nonzero(as_tuple=False).tolist():
            assert mask[row][col].item() == 0, f"{side}: pad position {col} is masked-in"

        # Everything supervised IS a real token, and includes EOS.
        supervised = ids[0][mask[0].nonzero(as_tuple=True)[0]].tolist()
        assert EOS_ID in supervised, f"{side}: EOS should be supervised"
        assert set(supervised) <= set(CHOSEN_RESPONSE + REJECTED_RESPONSE + [EOS_ID]), (
            f"{side}: mask selects unexpected tokens"
        )


def test_attention_mask_matches_real_tokens_in_fixture(torch_mod):
    pair = build_handmade_pair(torch_mod)
    ids, attn = pair["chosen_input_ids"], pair["chosen_attention_mask"]
    real = (ids != PAD_ID).long()
    assert (attn == real).all(), "attention mask disagrees with non-pad tokens"


def test_pair_shares_prompt_prefix_but_differs_in_response_region(torch_mod):
    """Chosen and rejected must share the exact prompt prefix (identical
    conditioning) and differ after it — otherwise the comparison is meaningless."""
    pair = build_handmade_pair(torch_mod)
    plen = pair["prompt_len"]
    c, r = pair["chosen_input_ids"], pair["rejected_input_ids"]

    assert (c[:, :plen] == r[:, :plen]).all(), "prompt prefixes differ — batch is broken"
    assert not (c[:, plen:] == r[:, plen:]).all(), "responses are identical — not a preference"


def test_paired_batch_shape_contract(torch_mod):
    """Stacking single examples yields consistent (batch, seq_len) tensors —
    the structural guarantee collate_preference_batch must maintain."""
    pair = build_handmade_pair(torch_mod)
    batch_two = {k: torch_mod.cat([v, v], dim=0) for k, v in pair.items() if torch_mod.is_tensor(v)}

    seq_lens = {v.shape[1] for v in batch_two.values()}
    batch_dims = {v.shape[0] for v in batch_two.values()}
    assert len(seq_lens) == 1, "padded tensors disagree on seq_len"
    assert batch_dims == {2}, "paired tensors disagree on batch dim"

    # Chosen and rejected response masks must agree in structure on the shared
    # prompt region even when response texts differ in length (here they don't).
    cpm = pair["chosen_response_mask"]
    rpm = pair["rejected_response_mask"]
    assert (cpm[:, : pair["prompt_len"]] == rpm[:, : pair["prompt_len"]]).all()
