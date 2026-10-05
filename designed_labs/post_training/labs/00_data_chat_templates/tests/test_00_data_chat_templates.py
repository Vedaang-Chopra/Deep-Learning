"""Lab 00 contract tests.

Run WITHOUT GPU/network/torch: ``python3 -m pytest tests/ -q`` (numpy + pytest only).

Two layers:
  A. Scaffold/contract tests that pass NOW (structure, importability, stub
     discipline, no-solutions scan, hand-built mask-invariant fixtures,
     config/notebook integrity).
  B. Implementation-grading hooks that SKIP while a stub still raises
     NotImplementedError (a single marker you can extend once you implement).

No-solutions compliance note: these tests assert SHAPE / INVARIANT contracts
on HAND-BUILT fixtures only (e.g. "labels at prompt positions are all -100");
they never contain mask-building arithmetic or chat-template logic themselves.
Torch is optional: if present, one extra guard runs; otherwise importorskip.
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
    "configs/00_data_chat_templates.yaml",
]

REQUIRED_STARTER_NAMES = [
    "IGNORE_INDEX",
    "FIXED_PANEL_PROMPTS",
    "inspect_tokenizer",
    "tokenize_plain",
    "tokenize_chat",
    "diff_tokenizations",
    "build_prompt_masked_labels",
    "make_triples",
    "decode_unmasked_only",
    "collate_right_pad",
    "pooling_corruption_demo",
    "generation_comparison",
    "length_stats",
    "load_no_robots_sample",
]

# Solution-shaped fragments that must NOT appear in the starter or the
# notebook's code cells (mask-building / template / collate arithmetic).
FORBIDDEN_SOLUTION_PATTERNS = [
    "apply_chat_template(",
    "[IGNORE_INDEX]",
    "prompt_length",
    "+ full_ids",
    "torch.cat(",
    "np.pad(",
    "cross_entropy(",
    "add_generation_prompt=",
]

# Inert calls: every public stub must raise NotImplementedError for these.
STUB_CALLS = {
    "inspect_tokenizer": ((object(),), {}),
    "tokenize_plain": ((object(), "hello"), {}),
    "tokenize_chat": ((object(), ({"role": "user", "content": "hi"}),), {}),
    "diff_tokenizations": (((1, 2), (1, 2, 3)), {}),
    "build_prompt_masked_labels": (
        (
            object(),
            (
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ),
        ),
        {},
    ),
    "make_triples": ((object(), (1, 2), (-100, 2)), {}),
    "decode_unmasked_only": ((object(), (1, 2), (-100, 2)), {}),
    "collate_right_pad": (((( (1, 2), (-100, 2) ),), 0), {}),
    "pooling_corruption_demo": (
        ({"input_ids": [[1, 0]], "attention_mask": [[1, 0]], "labels": [[2, -100]]},),
        {},
    ),
    "generation_comparison": ((None, object(), ["p"], "base"), {}),
    "length_stats": (((1, 2, 3),), {}),
    "load_no_robots_sample": ((50,), {}),
}


# ---------------------------------------------------------------------------
# Layer A — scaffold integrity
# ---------------------------------------------------------------------------


def test_lab_structure_is_uniform():
    missing = [f for f in REQUIRED_FILES if not (LAB_DIR / f).exists()]
    assert not missing, f"missing scaffold files: {missing}"
    assert (LAB_DIR / "tests" / "test_00_data_chat_templates.py").exists()


def _load_starter(name):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, LAB_DIR / "starter.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_starter_imports_cleanly():
    """starter must import WITHOUT torch/network side effects (numpy-only envs)."""
    import sys

    mod = _load_starter("starter_l00_import")
    for name in REQUIRED_STARTER_NAMES:
        assert hasattr(mod, name), f"starter is missing required name: {name}"
    assert mod.IGNORE_INDEX == -100
    assert len(mod.FIXED_PANEL_PROMPTS) == 6
    sys.modules.pop("starter_l00_import", None)


def test_starter_has_no_module_level_torch():
    src = (LAB_DIR / "starter.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:  # module level ONLY
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ",".join(a.name for a in node.names)
            assert "torch" not in mod, f"module-level torch import found: {mod}"


def test_stubs_raise_not_implemented():
    mod = _load_starter("starter_l00_stubs")
    for fname, (args, kwargs) in STUB_CALLS.items():
        fn = getattr(mod, fname)
        with pytest.raises(NotImplementedError):
            fn(*args, **kwargs)


def test_no_solutions_in_starter():
    src = (LAB_DIR / "starter.py").read_text()
    for pat in FORBIDDEN_SOLUTION_PATTERNS:
        assert pat not in src, f"no-solutions violation in starter.py: {pat!r}"


def test_no_solutions_in_notebook_code_cells():
    nb = json.loads((LAB_DIR / "notebook.ipynb").read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        for pat in FORBIDDEN_SOLUTION_PATTERNS:
            assert pat not in code, f"no-solutions violation in code cell {i}: {pat!r}"
        tree = ast.parse(code)  # code cells call stubs; they define nothing
        for node in ast.walk(tree):
            assert not isinstance(node, ast.FunctionDef), (
                f"code cell {i} defines a function — implement in starter.py, not the notebook"
            )


def test_notebook_is_valid_nbformat4_with_no_outputs():
    nb = json.loads((LAB_DIR / "notebook.ipynb").read_text())
    assert nb.get("nbformat") == 4
    assert isinstance(nb.get("metadata"), dict)
    assert nb["cells"], "notebook must contain cells"
    for i, cell in enumerate(nb["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}, f"cell {i}: bad type"
        assert isinstance(cell["source"], list), f"cell {i}: source must be a list"
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], f"code cell {i} must have no outputs"
            assert cell.get("execution_count") is None, f"code cell {i} must be unexecuted"


def test_config_has_required_keys():
    text = (LAB_DIR / "configs" / "00_data_chat_templates.yaml").read_text()
    for key in ["models:", "dataset:", "first_n_rows", "padding_side:",
                "compare_against:", "prerequisites:", "Qwen3-0.6B-Base",
                "OLMo-2-0425-1B-SFT", "SmolLM2-360M-Instruct", "no_robots"]:
        assert key in text, f"config missing expected key/model: {key}"


def test_readme_covers_spec_sections():
    text = (LAB_DIR / "README.md").read_text().lower()
    for section in ["prerequisit", "why", "assignment", "dataset", "questions",
                    "debugging", "completion", "compare"]:
        assert section in text, f"README missing section keyword: {section}"
    assert "rlhfbook.com/course" in text, "README must link the lecture page"


# ---------------------------------------------------------------------------
# Layer A — hand-built mask invariants (the contract Lab 01 depends on)
# ---------------------------------------------------------------------------


def _hand_built_masked_example():
    """Toy example with NO tokenizer: 20 ids, last 7 are 'assistant' tokens."""
    input_ids = list(range(100, 120))
    prompt_len = 13
    labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
    return input_ids, labels, prompt_len


def test_hand_built_labels_mask_prompt_positions():
    input_ids, labels, prompt_len = _hand_built_masked_example()
    assert len(labels) == len(input_ids)
    for i in range(prompt_len):
        assert labels[i] == IGNORE_INDEX, f"position {i} must be masked (-100)"
    for i in range(prompt_len, len(input_ids)):
        assert labels[i] == input_ids[i], f"position {i} must supervise its own id"


def test_hand_built_supervised_span_is_contiguous_and_reaches_the_end():
    """A correct final-turn mask is ONE contiguous supervised suffix."""
    _, labels, _ = _hand_built_masked_example()
    supervised = [i for i, v in enumerate(labels) if v != IGNORE_INDEX]
    assert supervised == list(range(supervised[0], len(labels))), (
        "supervised positions must form a contiguous suffix reaching the last token"
    )


def test_hand_built_shift_semantics():
    """Causal LM: position i predicts token i+1, so trainable predictions are
    exactly the positions whose *next* label is unmasked."""
    input_ids, labels, prompt_len = _hand_built_masked_example()
    shift_labels = labels[1:]
    trainable_pred_positions = [i for i, v in enumerate(shift_labels) if v != IGNORE_INDEX]
    assert trainable_pred_positions == list(range(prompt_len - 1, len(input_ids) - 1))


def test_hand_built_padded_batch_invariants():
    """Right-padded rows: pads are 0-masked, -100-labelled, last axis rectangular."""
    input_ids, labels, _ = _hand_built_masked_example()
    short_ids, short_labels = input_ids[:10], labels[:10]
    max_len = len(input_ids)
    pad_len = max_len - len(short_ids)
    batch_ids = [input_ids + [7] * 0, short_ids + [0] * pad_len]  # pad id 0
    batch_mask = [[1] * len(r) + [0] * (max_len - len(r)) for r in batch_ids]
    batch_labels = [labels, short_labels + [IGNORE_INDEX] * pad_len]
    for row in batch_ids:
        assert len(row) == max_len
    for m_row, l_row in zip(batch_mask, batch_labels):
        assert len(m_row) == len(l_row) == max_len
        for j, (m, l) in enumerate(zip(m_row, l_row)):
            if m == 0:
                assert l == IGNORE_INDEX, f"pad position {j} must carry a -100 label"


def test_wrong_mask_includes_header_violates_invariants():
    """Negative control: supervising one token before the answer span (the
    assistant header) breaks the prompt-side masking invariant AND makes a
    prompt token trainable under the causal shift — i.e. the invariants
    actually catch the README §9 header-token bug."""
    input_ids, labels, prompt_len = _hand_built_masked_example()
    bad = list(labels)
    bad[prompt_len - 1] = input_ids[prompt_len - 1]  # header token supervised
    assert any(l != IGNORE_INDEX for i, l in enumerate(bad) if i < prompt_len), (
        "header supervision must violate 'prompt positions are all -100'"
    )
    trainable = [i for i, v in enumerate(bad[1:]) if v != IGNORE_INDEX]
    assert (prompt_len - 2) in trainable, (
        "header supervision makes position prompt_len-2 predict a PROMPT token"
    )


# ---------------------------------------------------------------------------
# Layer B — implementation-grading hooks (SKIP until implemented)
# ---------------------------------------------------------------------------


def test_implementation_layer_b_placeholder():
    """Once build_prompt_masked_labels is implemented, extend this layer with
    real grading checks (e.g. decode-unmasked == final assistant turn on a
    locally cached tokenizer). SKIP while the stub raises NotImplementedError
    so `pytest -q` stays green for the scaffold-only state."""
    pytest.importorskip("numpy")  # env guard only; no torch/network needed
    pytest.skip("Layer B grading activates after the student implements Lab 00")
