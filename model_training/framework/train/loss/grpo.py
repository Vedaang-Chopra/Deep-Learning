

import logging
import re
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)




### GRPO Loss Work


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
