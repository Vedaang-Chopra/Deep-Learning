

import logging
import re
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)


def extract_gold_answer(answer: str) -> int:
    value = answer.rsplit("####", 1)[-1].strip()

    if "####" not in answer or not re.fullmatch(
        r"-?(?:\d+|\d{1,3}(?:,\d{3})+)", value
    ):
        raise ValueError(f"Invalid gold answer: {answer!r}")

    return int(value.replace(",", ""))


def extract_prediction(text: str) -> int | None:
    match = re.fullmatch(
        r"FINAL_ANSWER:\s*(-?(?:\d+|\d{1,3}(?:,\d{3})+))",
        text.strip()
    )

    if not match:
        return None

    return int(match.group(1).replace(",", ""))




def compute_reward(prediction: str, gold_answer: str) -> float:
    """
    Return:
        1.0: valid format and numerically correct
        0.0: incorrect answer or invalid format
    """
    if extract_prediction(prediction) == extract_gold_answer(gold_answer):
        return 1.0
    else:
        return 0.0


def sample_group_completions(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Sample and decode a fixed-size completion group for each prompt."""
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    logger.info(
        "Sampling completion groups: prompts=%s group_size=%s",
        input_ids.shape[0],
        group_size,
    )
    generation_kwargs = {}
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask
    else:
        attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )

    prompt_width = input_ids.shape[1]
    predictions = tokenizer.batch_decode(
        output_ids[:, prompt_width:],
        skip_special_tokens=True,
    )
    predictions = [prediction.strip() for prediction in predictions]
    completion_attention_mask = torch.ones_like(output_ids[:, prompt_width:])
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        eos_seen = output_ids[:, prompt_width:].eq(eos_token_id).cumsum(dim=1)
        completion_attention_mask = eos_seen <= 1
    sequence_attention_mask = torch.cat(
        [attention_mask.repeat_interleave(group_size, dim=0), completion_attention_mask],
        dim=1,
    )
    logger.info(
        "Sampled %s completions across %s prompts",
        len(predictions),
        input_ids.shape[0],
    )
    return output_ids, predictions, sequence_attention_mask


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

    logger.info(
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
    logger.info("Computed group rewards with shape %s", tuple(reward_tensor.shape))
    return reward_tensor


def compute_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    rewards: (batch_size, group_size)
    returns: (batch_size, group_size)

    Normalize rewards independently for each prompt.
    Handle groups with identical rewards.
    """

    mean = rewards.mean(dim=1, keepdim=True)

    std = rewards.std(dim=1, keepdim=True, correction=0)
    advantages = (rewards - mean) / (std + eps)

    return advantages


def test_advantages(advantages: torch.Tensor, rewards: torch.Tensor) -> None:
    print(advantages)

    assert advantages.shape == rewards.shape
    assert torch.isfinite(advantages).all()
    assert torch.allclose(
        advantages[0],
        torch.tensor([1., -1., 1., -1.]),
        atol=1e-5
    )
    assert torch.allclose(advantages[1:], torch.zeros_like(advantages[1:]))


def completion_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    prompt_len: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = token_ids.shape

    # 1. Shift
    shifted_logits = logits[:, :-1, :]
    shifted_labels = token_ids[:, 1:]

    # 2. Log probabilities
    log_probs = torch.log_softmax(shifted_logits, dim=-1)

    # 3. Select actual token log probabilities
    token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)

    # 4. Completion mask
    positions = torch.arange(1, T, device=token_ids.device)
    mask = positions >= prompt_len
    mask = mask.unsqueeze(0).expand(B, -1)
    if attention_mask is not None:
        mask = mask & attention_mask[:, 1:].bool()

    return token_logps, mask


def compute_ratio(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
) -> torch.Tensor:
    """
    current_logps: (B, T-1)
    old_logps:     (B, T-1)

    Returns:
        ratio: (B, T-1)
    """
    ratio = torch.exp(
        current_logps - old_logps.detach()
    )
    return ratio
