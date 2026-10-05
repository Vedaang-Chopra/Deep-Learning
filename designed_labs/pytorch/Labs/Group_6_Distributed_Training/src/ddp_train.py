"""
ddp_train.py — DDP Training Loop
===================================

Student implements:
  - build_dataloader_ddp(): DataLoader with DistributedSampler
  - wrap_model_ddp(): wrap model with DistributedDataParallel
  - train_one_epoch_ddp(): correct DDP training loop

No Lightning. No HuggingFace Trainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple, Optional, Dict


# ─────────────────────────────────────────────────────
# DDP DataLoader
# ─────────────────────────────────────────────────────

def build_dataloader_ddp(
    dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    """
    Build a DataLoader with DistributedSampler.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    batch_size : int — per-GPU batch size
    rank : int — current process rank
    world_size : int — total number of processes
    seed : int — base seed for reproducibility

    Returns
    -------
    (dataloader, sampler)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Create DistributedSampler:                         ║
    ║     sampler = DistributedSampler(dataset,              ║
    ║       num_replicas=world_size, rank=rank,              ║
    ║       shuffle=True, seed=seed)                         ║
    ║  2. Create DataLoader with this sampler:               ║
    ║     - shuffle=False (sampler handles it!)              ║
    ║     - drop_last=True (avoids uneven final batch)       ║
    ║     - pin_memory=True for GPU transfer speed           ║
    ║  3. Return (loader, sampler)                           ║
    ║                                                       ║
    ║  PITFALL: If you set shuffle=True in DataLoader AND    ║
    ║  use a sampler, you get an error. Use the sampler.     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement build_dataloader_ddp()")


def build_dataloader_single(dataset, batch_size, num_workers=2):
    """Non-distributed DataLoader. Provided fallback."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


# ─────────────────────────────────────────────────────
# DDP Model Wrapping
# ─────────────────────────────────────────────────────

def wrap_model_ddp(
    model: nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> DDP:
    """
    Wrap a model with DistributedDataParallel.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Move model to device: model = model.to(device)     ║
    ║  2. Wrap: ddp_model = DDP(model,                       ║
    ║       device_ids=[device.index],                       ║
    ║       output_device=device.index,                      ║
    ║       find_unused_parameters=find_unused_parameters)   ║
    ║  3. Return ddp_model                                   ║
    ║                                                       ║
    ║  NOTE on find_unused_parameters:                       ║
    ║  - False (default): faster, but errors if any param    ║
    ║    doesn't contribute to loss                          ║
    ║  - True: handles unused params but adds overhead       ║
    ║  - Use True only when debugging, then fix your model   ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement wrap_model_ddp()")


# ─────────────────────────────────────────────────────
# DDP Training Loop
# ─────────────────────────────────────────────────────

def train_one_epoch_ddp(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    sampler: DistributedSampler = None,
    grad_clip: Optional[float] = None,
    log_interval: int = 50,
) -> Dict[str, float]:
    """
    Train one epoch with DDP-correct practices.

    Returns
    -------
    dict with 'avg_loss', 'num_samples'

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  CRITICAL STEP: sampler.set_epoch(epoch)               ║
    ║  Without this, every epoch gets the SAME shard order!  ║
    ║  The model sees data in the same order → poor training.║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. if sampler: sampler.set_epoch(epoch)               ║
    ║  2. model.train()                                      ║
    ║  3. For each batch:                                    ║
    ║     a. Move to device                                  ║
    ║     b. optimizer.zero_grad()                           ║
    ║     c. Forward + loss                                  ║
    ║     d. loss.backward()                                 ║
    ║     e. Optional gradient clipping                      ║
    ║     f. optimizer.step()                                ║
    ║     g. Accumulate loss (stay on GPU, no .item()!)      ║
    ║  4. Average loss across batches                        ║
    ║  5. Return results                                     ║
    ║                                                       ║
    ║  PITFALL: Calling loss.item() every step forces a      ║
    ║  CPU-GPU sync. Instead, accumulate as a GPU tensor     ║
    ║  and only .item() once at the end.                     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement train_one_epoch_ddp()")


def evaluate_ddp(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate and return local metrics (aggregate separately).

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  1. model.eval(), torch.no_grad()                      ║
    ║  2. Accumulate correct predictions and total samples   ║
    ║  3. Return {'local_correct': ..., 'local_total': ...,  ║
    ║             'local_loss': ...}                          ║
    ║  4. Do NOT all-reduce here — that's done in            ║
    ║     dist_metrics.compute_global_accuracy()             ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement evaluate_ddp()")
