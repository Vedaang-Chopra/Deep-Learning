import logging
from typing import Any, Sequence

import torch

from grpo.data import prepare_prompt_batch
from grpo.grpo_loss import (
    completion_token_logprobs,
    compute_advantages,
    compute_ratio,
    grpo_policy_loss,
)
from verifier import compute_group_rewards, classify_response

logger = logging.getLogger(__name__)


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



def generate_rollouts(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: Sequence[dict[str, Any]],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Prepare prompts and generate grouped completions."""
    
    # Step 1: Get the input IDs and attention mask.
    input_ids, attention_mask = prepare_prompt_batch(tokenizer, examples, device)

    # Step 2: Generate the rollouts.
    output_ids, predictions, sequence_mask = sample_group_completions(
        model, tokenizer, input_ids, group_size, max_new_tokens, temperature, attention_mask
    )

    return input_ids, attention_mask, output_ids, predictions, sequence_mask


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

    input_ids, attention_mask, output_ids, predictions, sequence_mask = generate_rollouts(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )

    # Step 3: Generate rewards and advantages.
    rewards = compute_group_rewards(
        predictions, [example["answer"] for example in examples], group_size, device
    )
    advantages = compute_advantages(rewards)
    prompt_len = input_ids.shape[1]



    ## Step-4: - Generate the probability of the computed tokens ? 
    with torch.no_grad():
        old_logits = model(output_ids, attention_mask=sequence_mask, use_cache=False).logits
        old_logps, mask = completion_token_logprobs(
            old_logits, output_ids, prompt_len, sequence_mask
        )
    del old_logits


    ## Step-5: - Compute log probability of new tokens ? 
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
                "failure_reason": classify_response(predictions[row],example["answer"]),
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
