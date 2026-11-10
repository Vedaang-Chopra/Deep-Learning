import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)



def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute masked next-token cross entropy from SFT logits and labels."""
    logger.debug("sft_loss | logits_shape=%s labels_shape=%s", tuple(logits.shape), tuple(labels.shape))
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    logger.debug("sft_loss | loss_shape=%s requires_grad=%s", tuple(loss.shape), loss.requires_grad)
    return loss
    

def sft_forward(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the SFT model forward and return logits with target token IDs."""
    input_ids, labels = batch

    input_ids = input_ids.to(device)
    labels = labels.to(device)

    logger.debug("sft_forward | input_ids_shape=%s", tuple(input_ids.shape))
    logits = model(input_ids=input_ids).logits
    logger.debug("sft_forward | logits_shape=%s", tuple(logits.shape))
    return logits, labels
        
    
