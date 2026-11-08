import logging
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)


### Basic Code for Dataset Related Operations
def make_prompt(question: str) -> str:
    """Format one GSM8K question for the model."""
    return (
        "Solve the following math problem.\n"
        "Return only the final numerical answer.\n"
        "Use the format: FINAL_ANSWER: <number>\n\n"
        f"Question: {question}"
    )

def prepare_prompt_batch(
    tokenizer: Any,
    examples: Sequence[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one conversation per example, left-padding the batch."""

    if not examples:
        raise ValueError("examples must contain at least one question")
    tokenizer.padding_side = "left"

    conversations = [
        [{"role": "user", "content": make_prompt(example["question"])}]
        for example in examples
    ]

    inputs = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
