"""
dist_setup.py — Distributed Process Group Setup
==================================================

Student implements:
  - setup_process_group(): initialize NCCL backend
  - cleanup_process_group(): destroy process group
  - get_local_rank(): read rank info from environment
  - is_distributed(): check if running in distributed mode

Must handle single-GPU gracefully.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────
# Process Group Setup
# ─────────────────────────────────────────────────────

def setup_process_group(
    backend: str = "nccl",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    master_addr: str = "localhost",
    master_port: str = "29500",
) -> Tuple[int, int]:
    """
    Initialize the distributed process group.

    Parameters
    ----------
    backend : str — 'nccl' (GPU), 'gloo' (CPU fallback)
    rank : int or None — if None, read from env RANK
    world_size : int or None — if None, read from env WORLD_SIZE
    master_addr : str — address of rank 0
    master_port : str — port for rendezvous

    Returns
    -------
    (rank, world_size)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. If rank/world_size not provided, read from env:    ║
    ║     rank = int(os.environ.get("RANK", 0))              ║
    ║     world_size = int(os.environ.get("WORLD_SIZE", 1))  ║
    ║  2. Set environment variables:                         ║
    ║     os.environ["MASTER_ADDR"] = master_addr            ║
    ║     os.environ["MASTER_PORT"] = master_port            ║
    ║  3. If world_size > 1:                                 ║
    ║     dist.init_process_group(backend, rank, world_size) ║
    ║     torch.cuda.set_device(get_local_rank())            ║
    ║  4. Print: f"[Rank {rank}/{world_size}] initialized"   ║
    ║  5. Return (rank, world_size)                          ║
    ║                                                       ║
    ║  PITFALL: init_process_group is BLOCKING — all ranks   ║
    ║  must call it or the process hangs.                     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement setup_process_group()")


def cleanup_process_group() -> None:
    """
    Destroy the distributed process group.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  if dist.is_initialized():                             ║
    ║      dist.destroy_process_group()                      ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement cleanup_process_group()")


# ─────────────────────────────────────────────────────
# Rank Helpers
# ─────────────────────────────────────────────────────

def get_local_rank() -> int:
    """
    Get the local rank (GPU index on this machine).

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  Try these env vars in order:                          ║
    ║  1. LOCAL_RANK (set by torchrun)                       ║
    ║  2. RANK (fallback for single-node)                    ║
    ║  3. Default to 0                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement get_local_rank()")


def get_rank() -> int:
    """Get the global rank. Returns 0 if not distributed."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size. Returns 1 if not distributed."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """True if this is rank 0 (or non-distributed)."""
    return get_rank() == 0


def is_distributed() -> bool:
    """True if running with world_size > 1."""
    return dist.is_initialized() and dist.get_world_size() > 1


# ─────────────────────────────────────────────────────
# Launch Helper (provided)
# ─────────────────────────────────────────────────────

def print_launch_instructions():
    """Print instructions for launching distributed training. Provided."""
    print("""
╔═══════════════════════════════════════════════════════╗
║  How to launch distributed training:                  ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║  Single node, 2 GPUs:                                 ║
║    torchrun --nproc_per_node=2 your_script.py          ║
║                                                       ║
║  Single node, 4 GPUs:                                 ║
║    torchrun --nproc_per_node=4 your_script.py          ║
║                                                       ║
║  Multi-node (node 0):                                 ║
║    torchrun --nnodes=2 --node_rank=0                   ║
║             --master_addr=IP --master_port=29500       ║
║             --nproc_per_node=4 your_script.py          ║
║                                                       ║
║  Environment variables set by torchrun:                ║
║    RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, etc.     ║
╚═══════════════════════════════════════════════════════╝
""")
