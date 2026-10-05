"""Shared numpy-only fixtures for Lab 01 tests (no GPU, no network, no torch).

FakeTokenizer mimics the pieces of a HF chat tokenizer that Lab 01 depends on:
  - deterministic word->id encoding (hash-free and stable across runs)
  - apply_chat_template(msgs, tokenize=..., add_generation_prompt=...) with
    BOS + role header ids + content ids (+ optional generation prompt)
  - EOS appended when the final message is an assistant turn
  - pad/eos token bookkeeping

The point: prompt-only renderings are STRICT PREFIXES of full renderings —
the invariant prompt masking in Lab 01 is built on.
"""

import pytest


class FakeTokenizer:
    """Deterministic stand-in for PreTrainedTokenizer with a chat template."""

    eos_token_id = 2
    pad_token_id = 2  # answer key sets pad = eos; mirror that convention
    bos_token_id = 1

    ROLE_HEADER_IDS = {  # "<|user|>" style headers
        "user": [10],
        "assistant": [11],
        "system": [12],
    }
    GENERATION_PROMPT_IDS = [11]  # the assistant header itself ends the prompt side

    def __init__(self):
        self.vocab_size = 1000
        self.pad_token = self.eos_token = "<|endoftext|>"

    @staticmethod
    def _encode_text(text):
        # stable word->id in [50, 999); punctuation/space tokens fold into words
        return [50 + (sum(ord(c) for c in word) % 949) for word in text.split()]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize, "tests only exercise tokenized renderings"
        ids = [self.bos_token_id]
        for msg in messages:
            ids.extend(self.ROLE_HEADER_IDS[msg["role"]])
            ids.extend(self._encode_text(msg["content"]))
        if add_generation_prompt:
            ids.extend(self.GENERATION_PROMPT_IDS)  # header present, no content yet
        elif messages[-1]["role"] == "assistant":
            ids.append(self.eos_token_id)  # supervised stop position
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return f"<decoded:{len(ids)}-tokens>"


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def two_turn_conversation():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "Two plus two equals four."},
    ]


@pytest.fixture
def ragged_encoded_rows():
    """Variable-length rows whose labels follow the mask convention already.

    Row layout (T_i, n_unmasked): row0 (5, 2), row1 (3, 1), row2 (7, 4).
    Unmasked label values are the id of the same position (identity targets).
    """
    rows = []
    specs = [(5, 2), (3, 1), (7, 4)]
    start_id = 100
    for t, n_sup in specs:
        ids = list(range(start_id, start_id + t))
        labels = [-100] * (t - n_sup) + ids[t - n_sup:]
        rows.append({"input_ids": ids, "labels": labels})
        start_id += t + 10
    return rows
