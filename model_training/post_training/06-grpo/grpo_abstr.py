

import logging
import re
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)


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


def extract_gold_answer(answer: str) -> int:
    """Extract the integer following the GSM8K answer delimiter."""
    value = answer.rsplit("####", 1)[-1].strip()

    if "####" not in answer or not re.fullmatch(
        r"-?(?:\d+|\d{1,3}(?:,\d{3})+)", value
    ):
        raise ValueError(f"Invalid gold answer: {answer!r}")

    return int(value.replace(",", ""))


def extract_prediction(text: str) -> int | None:
    """Extract an integer from a correctly formatted completion."""
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

    logger.debug(
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
    completion_ids = output_ids[:, prompt_width:]
    stop_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if stop_ids is None:
        stop_ids = []
    elif isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    else:
        stop_ids = list(stop_ids)
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    is_stop = torch.zeros_like(completion_ids, dtype=torch.bool)
    for stop_id in stop_ids:
        is_stop |= completion_ids.eq(stop_id)
    # Include the first stop token and exclude every position after it.
    completion_attention_mask = (is_stop.cumsum(dim=1) - is_stop.long()) == 0
    sequence_attention_mask = torch.cat(
        [attention_mask.repeat_interleave(group_size, dim=0), completion_attention_mask],
        dim=1,
    )
    logger.debug(
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
    """Check the original demonstration's expected advantages."""

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
    """Return log probabilities and a mask for the generated tokens."""
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


def grpo_policy_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 0.2,
) -> torch.Tensor:
    """Compute the masked, clipped policy loss for sampled completions."""
    ratio = torch.exp(current_logps.float() - old_logps.detach().float())
    advantage = advantages.detach().reshape(-1, 1)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
    token_loss = -torch.minimum(unclipped, clipped)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1)


def inspect_grpo_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: Sequence[dict[str, Any]],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    clip_eps: float = 0.2,
) -> dict[str, Any]:
    """Inspect one batch through sampling, rewards, advantages and policy loss.

    Returned tensors are detached on the CPU. The model is not updated.
    Records are ordered by example and then completion within that example.
    """
    logger.debug("Inspecting GRPO batch: examples=%s group_size=%s", len(examples), group_size)
    input_ids, attention_mask = prepare_prompt_batch(tokenizer, examples, device)
    output_ids, predictions, sequence_mask = sample_group_completions(
        model, tokenizer, input_ids, group_size, max_new_tokens, temperature, attention_mask
    )
    rewards = compute_group_rewards(
        predictions, [example["answer"] for example in examples], group_size, device
    )
    advantages = compute_advantages(rewards)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        old_logits = model(output_ids, attention_mask=sequence_mask, use_cache=False).logits
        old_logps, mask = completion_token_logprobs(
            old_logits, output_ids, prompt_len, sequence_mask
        )
    del old_logits

    with torch.enable_grad():
        current_logits = model(output_ids, attention_mask=sequence_mask, use_cache=False).logits
        logits_shape = tuple(current_logits.shape)
        current_logps, _ = completion_token_logprobs(
            current_logits, output_ids, prompt_len, sequence_mask
        )
        ratio = compute_ratio(current_logps, old_logps)
        loss = grpo_policy_loss(current_logps, old_logps, advantages, mask, clip_eps)
        current_requires_grad = current_logps.requires_grad

    output_cpu = output_ids.detach().cpu()
    mask_cpu = mask.detach().cpu()
    old_cpu = old_logps.detach().float().cpu()
    current_cpu = current_logps.detach().float().cpu()
    ratio_cpu = ratio.detach().float().cpu()
    rewards_cpu = rewards.detach().cpu()
    advantages_cpu = advantages.detach().cpu()
    batch_loss = loss.detach().item()
    records = []
    for example_index, example in enumerate(examples):
        for completion_index in range(group_size):
            row = example_index * group_size + completion_index
            token_mask = mask_cpu[row]
            records.append({
                "example_index": example_index,
                "question": example["question"],
                "gold_answer": example["answer"],
                "completion_index": completion_index,
                "prediction": predictions[row],
                "reward": rewards_cpu[example_index, completion_index].item(),
                "advantage": advantages_cpu[example_index, completion_index].item(),
                "batch_loss": batch_loss,
                "token_ids": output_cpu[row, 1:][token_mask].numpy().copy(),
                "old_logps": old_cpu[row][token_mask].numpy().copy(),
                "current_logps": current_cpu[row][token_mask].numpy().copy(),
                "ratio": ratio_cpu[row][token_mask].numpy().copy(),
            })

    result = {
        "input_ids": input_ids.detach().cpu(),
        "output_ids": output_cpu,
        "sequence_attention_mask": sequence_mask.detach().cpu(),
        "predictions": predictions,
        "rewards": rewards_cpu,
        "advantages": advantages_cpu,
        "old_logps": old_cpu,
        "current_logps": current_cpu,
        "completion_mask": mask_cpu,
        "ratio": ratio_cpu,
        "logits_shape": logits_shape,
        "loss": batch_loss,
        "old_requires_grad": old_logps.requires_grad,
        "current_requires_grad": current_requires_grad,
        "records": records,
    }
    logger.debug("Inspected GRPO batch: completions=%s loss=%.6f", len(records), result["loss"])
    return result
