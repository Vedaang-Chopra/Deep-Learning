import logging
import re
from decimal import Decimal
from typing import Sequence

import torch

logger = logging.getLogger(__name__)


### Building our Verifier
def extract_gold_answer(answer: str) -> int:
    """Extract the integer following the GSM8K answer delimiter."""
    value = answer.rsplit("####", 1)[-1].strip()

    if "####" not in answer or not re.fullmatch(
        r"-?(?:\d+|\d{1,3}(?:,\d{3})+)", value
    ):
        raise ValueError(f"Invalid gold answer: {answer!r}")

    return int(value.replace(",", ""))


def extract_prediction(text: str) -> Decimal | None:
    """Extract the first labeled numeric answer from a completion."""
    match = re.search(
        r"FINAL_ANSWER:\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\d,]|\.\d)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return Decimal(match.group(1).replace(",", ""))


def classify_response(prediction: str, gold_answer: str) -> str:
    if not prediction.strip():
        return "empty"

    extracted = extract_prediction(prediction)

    if extracted is None:
        return "invalid_format"

    if extracted != extract_gold_answer(gold_answer):
        return "wrong_answer"

    return "correct"


def compute_reward(prediction: str, gold_answer: str) -> float:
    """
    Return:
        1.0: labeled answer is numerically correct
        0.0: incorrect or missing labeled answer
    """
    status = classify_response(prediction, gold_answer)
    return 1.0 if status == "correct" else 0.0

def compute_group_rewards(
    predictions: Sequence[str],
    gold_answers: Sequence[str],
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Score sampled predictions and return rewards shaped (prompts, group)."""
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    expected_predictions = len(gold_answers) * group_size
    if len(predictions) != expected_predictions:
        raise ValueError(
            f"Expected {expected_predictions} predictions for "
            f"{len(gold_answers)} prompts and group_size={group_size}; "
            f"received {len(predictions)}"
        )

    logger.debug(
        "Computing group rewards: prompts=%s group_size=%s",
        len(gold_answers),
        group_size,
    )
    rewards = [
        [
            compute_reward(prediction, gold_answer)
            for prediction in predictions[
                prompt_index * group_size:(prompt_index + 1) * group_size
            ]
        ]
        for prompt_index, gold_answer in enumerate(gold_answers)
    ]
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    logger.debug("Computed group rewards with shape %s", tuple(reward_tensor.shape))
    return reward_tensor
