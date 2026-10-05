"""Lab 01 contract tests.

Run WITHOUT GPU/network/torch: ``python3 -m pytest tests/ -q`` (numpy + pytest only).

Two layers of checks:
  A. Scaffold/contract tests that pass NOW (structure, importability, stub
     discipline, fixture invariants, data-convention invariants).
  B. Implementation-grading tests that SKIP while a stub still raises
     NotImplementedError and become real assertions once you implement it.

No-solutions compliance note: these tests never contain lab solutions — layer B
only asserts SHAPE / INVARIANT contracts (rectangularity, pad-side consistency,
mask correctness), never computes losses or adapters itself.
"""

import ast
import json
import pathlib

import numpy as np
import pytest

LAB_DIR = pathlib.Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100

REQUIRED_FILES = [
    "README.md",
    "starter.py",
    "notebook.ipynb",
    "configs/01_sft_from_scratch.yaml",
]

STUB_CALLS = {
    # name -> (args, kwargs) chosen to be inert; every entry must raise NotImplementedError
    "SFTBatch.to": ((None,), {}),
    "build_prompt_masked_labels": (("fake_tokenizer", None), {"max_length": 64}),
    "collate_sft": (((), 2), {}),                      # empty examples tuple, pad id 2
    "make_dataloader": ((None, "fake_tokenizer"), {}),
    "compute_loss": ((None, None), {}),
    "LoRALinear": (("base", 8, 16), {}),
    "apply_lora": ((None, ["q_proj"], 8, 16), {}),
    "pack_sequences": (((), 16), {}),
    "train_step": ((None, (), None, None, 1, 1.0), {}),
    "evaluate_val_loss": ((None, None), {}),
    "generation_panel": ((None, "fake_tokenizer"), {}),
    "run_training_loop": ((None, None, None), {}),
}


# ---------------------------------------------------------------------------
# Layer A — scaffold integrity
# ---------------------------------------------------------------------------


def test_lab_structure_is_uniform():
    missing = [f for f in REQUIRED_FILES if not (LAB_DIR / f).exists()]
    assert not missing, f"missing scaffold files: {missing}"
    assert (LAB_DIR / "tests" / "test_01_sft_from_scratch.py").exists()


def _load_starter(name):
    """Load starter.py as a fresh module (registered in sys.modules so that
    string annotations in its @dataclass resolve under any Python >=3.9)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, LAB_DIR / "starter.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_starter_imports_cleanly():
    """starter must import WITHOUT torch installed (numpy-only environments)."""
    import sys

    mod = _load_starter("starter_l01_import")
    assert hasattr(mod, "IGNORE_INDEX")
    sys.modules.pop("starter_l01_import", None)


def test_starter_exposes_contract_symbols():
    mod = _load_starter("starter_l01")

    required = [
        "IGNORE_INDEX",
        "DEFAULT_SAMPLE_PANEL_PROMPTS",
        "SFTBatch",
        "SFTDataset",
        "build_prompt_masked_labels",
        "collate_sft",
        "make_dataloader",
        "compute_loss",
        "LoRALinear",
        "apply_lora",
        "pack_sequences",
        "train_step",
        "evaluate_val_loss",
        "generation_panel",
        "run_training_loop",
    ]
    for name in required:
        assert hasattr(mod, name), f"starter.py is missing required symbol {name}"

    assert mod.IGNORE_INDEX == IGNORE_INDEX
    assert len(mod.DEFAULT_SAMPLE_PANEL_PROMPTS) == 6


def test_no_solutions_stub_discipline():
    """Every contracted stub must raise NotImplementedError, not return values."""
    mod = _load_starter("starter_l01b")

    for dotted, (args, kwargs) in STUB_CALLS.items():
        obj = mod
        parts = dotted.split(".")
        if parts[0] == "LoRALinear" and len(parts) > 1:
            with pytest.raises(NotImplementedError):
                cls = getattr(mod, "LoRALinear")
                inst = object.__new__(cls)  # bypass __init__ which also raises
                getattr(inst, parts[1])(object())
            continue
        if parts[0] == "SFTBatch":  # method on an instance
            batch = mod.SFTBatch(input_ids=None, attention_mask=None, labels=None)
            bound = lambda *a, **k: batch.to(*a, **k)
        else:
            bound = getattr(mod, parts[0])
        with pytest.raises(NotImplementedError):
            bound(*args, **kwargs)


def test_forbidden_patterns_absent_from_starter():
    """Static guard: the starter source must not embed answer-key mechanics."""
    src = (LAB_DIR / "starter.py").read_text()

    tree = ast.parse(src)
    banned_names = {
        "cross_entropy",     # CE math belongs to the student
        "clip_grad_norm_",   # optimizer-step logic belongs to the student
        "logsigmoid",        # answer-key-style losses have no place in an SFT starter
    }
    called_or_attrs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            called_or_attrs.add(node.attr)
        elif isinstance(node, ast.Name):
            called_or_attrs.add(node.id)
    leaked = banned_names & called_or_attrs
    assert not leaked, f"forbidden solution mechanics found in starter.py: {leaked}"

    # every def/class body contains an explicit TODO/NotImplementedError marker
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):  # internal helpers exempt
                continue
            body_src = ast.get_source_segment(src, node) or ""
            has_marker = (
                "NotImplementedError" in body_src
                or "TODO" in body_src
                or getattr(node, "name", "") == "SFTDataset"
            )
            assert has_marker, f"{node.name} body lacks TODO/NotImplementedError marker"


# ---------------------------------------------------------------------------
# Layer A — fixture invariants the masking convention depends on
# ---------------------------------------------------------------------------


def test_fake_tokenizer_prompt_is_prefix_of_full_render(fake_tokenizer, two_turn_conversation):
    prompt_ids = fake_tokenizer.apply_chat_template(
        two_turn_conversation[:-1], add_generation_prompt=True
    )
    full_ids = fake_tokenizer.apply_chat_template(
        two_turn_conversation, add_generation_prompt=False
    )

    T_p, T_f = len(prompt_ids), len(full_ids)
    assert T_p < T_f, "prompt-only render must be strictly shorter than full render"
    assert full_ids[:T_p] == prompt_ids, "prompt render must be a strict PREFIX of full render"
    assert full_ids[-1] == fake_tokenizer.eos_token_id, "assistant turn ends with EOS"
    assert fake_tokenizer.pad_token_id == fake_tokenizer.eos_token_id  # repo convention


def test_masking_convention_on_fixture(fake_tokenizer, two_turn_conversation):
    """The literal convention Lab 01 encodes, verified against numpy arithmetic."""
    tok = fake_tokenizer
    prompt_ids = tok.apply_chat_template(two_turn_conversation[:-1], add_generation_prompt=True)
    full_ids = tok.apply_chat_template(two_turn_conversation, add_generation_prompt=False)

    prompt_len = len(prompt_ids)
    input_ids = np.asarray(full_ids, dtype=np.int64)
    labels = np.where(
        np.arange(len(input_ids)) < prompt_len, IGNORE_INDEX, input_ids
    )  # supervision starts AFTER the generation-prompt header

    # invariants:
    assert (labels[:prompt_len] == IGNORE_INDEX).all(), "prompt side fully masked"
    supervised = labels[prompt_len:]
    assert supervised.size > 0, "assistant completion must supervise >=1 position"
    assert np.array_equal(supervised, input_ids[prompt_len:]), "labels mirror ids where live"
    assert supervised[-1] == tok.eos_token_id, "EOS is inside the supervised span"
    # decode-by-unmasked sanity analog: reconstruct exactly the tail tokens
    assert np.array_equal(labels[labels != IGNORE_INDEX], input_ids[prompt_len:])


def test_ragged_rows_fixture_shape_invariants(ragged_encoded_rows):
    rows = ragged_encoded_rows
    lens = [len(r["input_ids"]) for r in rows]
    assert lens == sorted(set(lens)) or len(set(lens)) > 1  # genuinely ragged
    for row in rows:
        assert len(row["input_ids"]) == len(row["labels"]), "row must stay paired"
        arr = np.asarray(row["labels"])
        head = arr == IGNORE_INDEX
        tail = ~head
        if tail.any():  # masks are contiguous [pad][supervise] per row
            assert head[:-1].all() or arr[np.argmax(tail) :].tolist().count(IGNORE_INDEX) == 0


# ---------------------------------------------------------------------------
# Layer A — padding conventions (right-side), with literals from the fixtures
# ---------------------------------------------------------------------------


def test_right_pad_batch_convention_literals():
    """Literal miniature of what collate_sft must produce: (3 rows) x ragged lengths."""
    # rows: T=(5,3,7) with supervised tails of 2/1/4 tokens
    input_ids = [
        [100, 101, 102, 103, 104],
        [110, 111, 112],
        [120, 121, 122, 123, 124, 125, 126],
    ]
    labels_in = [[-100, -100, -100, 103, 104], [-100, -100, 112], [-100] * 3 + [123, 124, 125, 126]]
    PAD_ID, T_MAX = 2, max(map(len, input_ids))

    def pad_row(seq, fill):
        return seq + [fill] * (T_MAX - len(seq))

    b_ids = np.asarray([pad_row(r, PAD_ID) for r in input_ids])
    b_labels = np.asarray([pad_row(r, IGNORE_INDEX) for r in labels_in])
    b_mask = np.asarray([pad_row([1] * len(r), 0) for r in input_ids])

    # RECTANGULARITY
    assert b_ids.shape == b_mask.shape == b_labels.shape == (3, T_MAX)
    # PAD-SIDE (right): each row is ones then zeros, nothing interleaved
    for row in b_mask:
        ones = int(row.sum())
        assert np.array_equal(row, np.r_[np.ones(ones, dtype=row.dtype), np.zeros(T_MAX - ones, dtype=row.dtype)])
    # pads carry no signal
    assert (b_labels[b_mask == 0] == IGNORE_INDEX).all()
    assert (b_ids[b_mask == 0] == PAD_ID).all()
    # real-token regions unchanged by padding
    for i, orig in enumerate(input_ids):
        L = len(orig)
        assert b_ids[i, :L].tolist() == orig
        assert b_labels[i, :L].tolist() == labels_in[i]
    # shift alignment: target at t predicts id at t+1 wherever unmasked
    sup = b_labels != IGNORE_INDEX
    aligned = sup[:, :-1] & sup[:, 1:]
    assert np.array_equal(b_labels[:, 1:][aligned], b_ids[:, 1:][aligned])


# ---------------------------------------------------------------------------
# Layer B — grading tests (skip until implemented, then enforce shapes)
# ---------------------------------------------------------------------------


def _call_tolerant(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        return None


def test_build_prompt_masked_labels_contract(fake_tokenizer, two_turn_conversation):
    mod = _load_starter("starter_l01c")

    out = _call_tolerant(
        mod.build_prompt_masked_labels, fake_tokenizer, two_turn_conversation, max_length=64
    )
    if out is None:
        pytest.skip("build_prompt_masked_labels not implemented yet")

    input_ids, labels = out
    assert len(input_ids) == len(labels), "(ids, labels) must be same length"
    full = np.asarray(input_ids)
    lab = np.asarray(labels)
    assert full.ndim == lab.ndim == 1
    assert full[-1] == fake_tokenizer.eos_token_id, "full render must retain EOS"

    # boundary discovered from the fixture itself (oracle lives in the fixtures above)
    prompt_len = len(
        fake_tokenizer.apply_chat_template(two_turn_conversation[:-1], add_generation_prompt=True)
    )
    assert (lab[:prompt_len] == IGNORE_INDEX).all(), "prompt positions must all be masked"
    assert lab[prompt_len:].size > 0, "at least one supervised assistant position"
    assert np.array_equal(lab[lab != IGNORE_INDEX], full[prompt_len:])
    assert len(full) <= 64, "max_length truncation respected"


def test_collate_sft_contract(ragged_encoded_rows):
    mod = _load_starter("starter_l01d")

    PAD_ID = 2
    batch = _call_tolerant(mod.collate_sft, ragged_encoded_rows, PAD_ID)
    if batch is None:
        pytest.skip("collate_sft not implemented yet")

    ids = np.asarray(getattr(batch, "input_ids"))
    mask = np.asarray(getattr(batch, "attention_mask"))
    labels = np.asarray(getattr(batch, "labels"))

    # RECTANGULARITY
    assert ids.shape == mask.shape == labels.shape
    assert ids.shape[0] == len(ragged_encoded_rows)
    # PAD-SIDE CONSISTENCY (right): all rows share one pad column count profile
    row_lengths = mask.sum(axis=1)
    expected = sorted(len(r["input_ids"]) for r in ragged_encoded_rows)
    assert sorted(row_lengths.tolist()) == expected
    assert (ids[mask == 0] == PAD_ID).all(), "pad slots filled with pad_token_id"
    assert (labels[mask == 0] == IGNORE_INDEX).all(), "pads never supervised"
    # no masking corruption of the originals
    for i, row in enumerate(ragged_encoded_rows):
        L = len(row["input_ids"])
        assert ids[i, :L].tolist() == list(row["input_ids"])
        assert labels[i, :L].tolist() == list(row["labels"])


# ---------------------------------------------------------------------------
# Notebook + config validity
# ---------------------------------------------------------------------------


def test_notebook_valid_nbformat4_no_outputs():
    nb = json.loads((LAB_DIR / "notebook.ipynb").read_text())
    assert nb["nbformat"] == 4
    assert isinstance(nb.get("cells"), list) and nb["cells"]
    md_count = 0
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert cell.get("source"), "cells must have content"
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], "skeleton ships with no outputs"
            assert cell.get("execution_count") is None
        else:
            md_count += 1
    assert md_count >= 6, "README-mirroring markdown walkthrough expected"


def test_config_parses_and_matches_lab_contract():
    yaml = pytest.importorskip("yaml", reason="PyYAML not installed; skipping config parse")
    cfg = yaml.safe_load((LAB_DIR / "configs" / "01_sft_from_scratch.yaml").read_text())

    assert cfg["model_name"].startswith("Qwen/")  # §4 register default
    assert cfg["dataset_name"] == "HuggingFaceH4/no_robots"
    assert cfg["lora"]["enabled"] is True and cfg["lora"]["alpha"] % cfg["lora"]["r"] == 0
    assert cfg["packing"]["enabled"] is False  # optional exercise ships off
    assert cfg["batch_size"] * cfg["gradient_accumulation_steps"] == 32  # effective batch
    assert 0.0 <= cfg["warmup_ratio"] < 1.0
