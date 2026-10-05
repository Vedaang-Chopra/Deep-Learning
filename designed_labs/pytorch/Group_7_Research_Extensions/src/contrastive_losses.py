"""
contrastive_losses.py — Contrastive Learning from Scratch
===========================================================

Student implements:
  - info_nce_loss(): InfoNCE / NT-Xent with numerical stability
  - SimCLRProjector: projection MLP head
  - gather_embeddings_across_ranks(): optional distributed negatives

Implementing Chen et al. 2020 (SimCLR) from the equations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────
# InfoNCE Loss
# ─────────────────────────────────────────────────────

def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute InfoNCE (NT-Xent) contrastive loss.

    Given a batch of N paired embeddings (z1_i, z2_i),
    the positive pair is (z1_i, z2_i) and negatives are all other embeddings.

    Loss for anchor z1_i:
      L_i = -log( exp(sim(z1_i, z2_i)/τ) / Σ_{k≠i} exp(sim(z1_i, z_k)/τ) )

    Parameters
    ----------
    z1 : (N, D) — embeddings from view 1
    z2 : (N, D) — embeddings from view 2
    temperature : float — temperature scaling (τ)

    Returns
    -------
    (loss, info_dict) where info_dict has:
      'accuracy' : float — how often the positive is most similar
      'avg_pos_sim' : float — average positive pair similarity
      'avg_neg_sim' : float — average negative pair similarity

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. L2-normalize z1 and z2                             ║
    ║  2. Concatenate: z = cat([z1, z2], dim=0)  # (2N, D)  ║
    ║  3. Compute similarity matrix:                         ║
    ║     sim = z @ z.T / temperature  # (2N, 2N)           ║
    ║  4. NUMERICAL STABILITY: subtract max per row          ║
    ║     sim = sim - sim.max(dim=1, keepdim=True).values    ║
    ║  5. Create mask to exclude self-similarity:            ║
    ║     mask = ~torch.eye(2N, dtype=bool, device=device)   ║
    ║  6. Identify positive pairs:                           ║
    ║     For z1_i (row i), positive is z2_i (row i+N)       ║
    ║     For z2_i (row i+N), positive is z1_i (row i)       ║
    ║  7. Extract positive logits and negative logits        ║
    ║  8. Loss = -log(exp(pos) / sum(exp(negatives)))        ║
    ║     OR equivalently:                                   ║
    ║     labels = positive indices                          ║
    ║     loss = F.cross_entropy(sim[mask].view(2N,-1), lbl) ║
    ║  9. Compute accuracy and similarity statistics          ║
    ║                                                       ║
    ║  PITFALL: Don't include self-similarity (diagonal)     ║
    ║  as a negative — it's always 1/τ and dominates.        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement info_nce_loss()")


# ─────────────────────────────────────────────────────
# SimCLR Projection Head
# ─────────────────────────────────────────────────────

class SimCLRProjector(nn.Module):
    """
    MLP projection head for contrastive learning.
    Maps encoder features to a lower-dim space for contrastive loss.

    Architecture: Linear → ReLU → Linear

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__(in_dim, hidden_dim=256, out_dim=128):       ║
    ║    self.net = nn.Sequential(                           ║
    ║      nn.Linear(in_dim, hidden_dim),                    ║
    ║      nn.ReLU(inplace=True),                            ║
    ║      nn.Linear(hidden_dim, out_dim))                   ║
    ║                                                       ║
    ║  forward(x):                                          ║
    ║    return self.net(x)                                  ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        raise NotImplementedError("TODO: implement SimCLRProjector.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement SimCLRProjector.forward()")


# ─────────────────────────────────────────────────────
# Distributed Negatives (optional)
# ─────────────────────────────────────────────────────

def gather_embeddings_across_ranks(z: torch.Tensor) -> torch.Tensor:
    """
    All-gather embeddings from all ranks for more negatives.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement (optional, multi-GPU only).          ║
    ║                                                       ║
    ║  1. if not distributed: return z                       ║
    ║  2. Allocate list of tensors (one per rank)            ║
    ║  3. dist.all_gather(tensor_list, z)                    ║
    ║  4. Return torch.cat(tensor_list, dim=0)               ║
    ║                                                       ║
    ║  This gives 2N*world_size negatives instead of 2N.     ║
    ║  Larger effective batch → better contrastive learning. ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement gather_embeddings_across_ranks()")


# ─────────────────────────────────────────────────────
# Augmentation Pipeline (provided)
# ─────────────────────────────────────────────────────

import torchvision.transforms as T


def get_simclr_augmentations(image_size: int = 32):
    """Contrastive learning augmentations (SimCLR-style). Provided."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


class TwoViewDataset(torch.utils.data.Dataset):
    """Wrapper that returns two augmented views of each image. Provided."""

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # base dataset may have already applied transforms,
        # so we use the raw PIL image if available
        if hasattr(self.base, 'data'):
            from PIL import Image
            img = Image.fromarray(self.base.data[idx])
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, label
