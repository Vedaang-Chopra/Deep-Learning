"""
dist_debug.py — Distributed Debugging Tools
==============================================

Student implements:
  - distributed_assert(): fail all ranks together
  - sync_barrier(): labeled synchronization point
  - set_nccl_debug_env(): print debug env vars
  - diagnose_hang(): checklist for debugging hangs

Failure injection exercises in the notebook.
"""

import os
import time
import torch
import torch.distributed as dist
from typing import Optional


# ─────────────────────────────────────────────────────
# Distributed Assert
# ─────────────────────────────────────────────────────

def distributed_assert(
    condition: bool,
    message: str = "Assertion failed",
    device: torch.device = None,
) -> None:
    """
    Assert a condition across all ranks. If ANY rank fails,
    ALL ranks raise an error.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Why this matters:                                    ║
    ║  If rank 0 asserts but rank 1 doesn't, rank 1         ║
    ║  continues to the next collective call → DEADLOCK.     ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Create a tensor: 0 if condition is True, 1 if False║
    ║  2. All-reduce SUM across ranks                        ║
    ║  3. If result > 0: at least one rank failed             ║
    ║     → raise RuntimeError(message) on ALL ranks          ║
    ║                                                       ║
    ║  If not distributed, just use normal assert.           ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement distributed_assert()")


# ─────────────────────────────────────────────────────
# Synchronization Barrier
# ─────────────────────────────────────────────────────

def sync_barrier(tag: str = "", timeout_seconds: float = 300.0) -> None:
    """
    Named synchronization barrier with timeout.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. If not distributed, return immediately             ║
    ║  2. Print f"[Rank {rank}] Barrier: {tag}"              ║
    ║  3. dist.barrier()                                     ║
    ║  4. Print f"[Rank {rank}] Past barrier: {tag}"         ║
    ║                                                       ║
    ║  The tag helps identify WHICH barrier caused a hang.   ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement sync_barrier()")


# ─────────────────────────────────────────────────────
# NCCL Debug Environment
# ─────────────────────────────────────────────────────

def set_nccl_debug_env(level: str = "INFO") -> None:
    """
    Set NCCL debug environment variables.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Set the following env vars:                           ║
    ║  NCCL_DEBUG = level (INFO, WARN, or TRACE)            ║
    ║  NCCL_DEBUG_SUBSYS = "ALL"                             ║
    ║  TORCH_DISTRIBUTED_DEBUG = "DETAIL"                    ║
    ║                                                       ║
    ║  Print what was set so the student knows.              ║
    ║                                                       ║
    ║  NOTE: These must be set BEFORE init_process_group().  ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement set_nccl_debug_env()")


# ─────────────────────────────────────────────────────
# Hang Diagnosis Checklist (provided)
# ─────────────────────────────────────────────────────

def print_hang_checklist():
    """Print a debugging checklist for distributed hangs. Provided."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║  DISTRIBUTED HANG DEBUGGING CHECKLIST                     ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  1. Are ALL ranks reaching the same collective call?       ║
║     → Add sync_barrier("before_allreduce") to check        ║
║                                                           ║
║  2. Did one rank crash/OOM silently?                       ║
║     → Check logs on ALL ranks, not just rank 0             ║
║     → Use distributed_assert() before collectives          ║
║                                                           ║
║  3. Is find_unused_parameters needed?                      ║
║     → If model has unused params in some forward paths     ║
║     → DDP hangs waiting for gradients that never come      ║
║                                                           ║
║  4. Is the DataLoader different across ranks?              ║
║     → Different # of batches → one rank finishes early     ║
║     → Use drop_last=True and DistributedSampler            ║
║                                                           ║
║  5. Are you mixing collectives and non-collective code?    ║
║     → if rank == 0: dist.reduce(...)  ← DEADLOCK!          ║
║     → ALL ranks must call collectives                      ║
║                                                           ║
║  6. NCCL timeout?                                         ║
║     → Default is 30 min. Set shorter for debugging:         ║
║     → os.environ["NCCL_TIMEOUT"] = "60"                    ║
║     → Or: dist.init_process_group(...,                     ║
║            timeout=timedelta(seconds=60))                  ║
║                                                           ║
║  7. Network issues (multi-node)?                          ║
║     → NCCL_DEBUG=INFO to see connection attempts            ║
║     → Check firewall, ports, IB vs TCP                     ║
║                                                           ║
║  Quick test: NCCL_DEBUG=INFO torchrun --nproc=2 script.py ║
╚═══════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────
# Failure Simulation Helpers (provided)
# ─────────────────────────────────────────────────────

def simulate_rank_skip_collective(rank: int, skip_rank: int = 1):
    """
    Simulate a bug where one rank skips a collective call.
    Provided for failure injection exercise.
    """
    if not dist.is_initialized():
        print("Not in distributed mode — simulating conceptually")
        print(f"  Rank {skip_rank} would skip all_reduce → hang!")
        return

    tensor = torch.ones(1, device=f"cuda:{rank}")
    if rank != skip_rank:
        # Only non-skipped ranks call all_reduce
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] all_reduce completed: {tensor.item()}")
    else:
        print(f"[Rank {rank}] SKIPPING all_reduce (bug!) → other ranks will hang")


def simulate_oom_one_rank(rank: int, oom_rank: int = 1, gb: float = 100.0):
    """
    Simulate OOM on one rank. Provided for failure injection exercise.
    """
    if rank == oom_rank:
        try:
            size = int(gb * 1024**3 / 4)  # float32
            _ = torch.empty(size, device=f"cuda:{rank}")
        except RuntimeError as e:
            print(f"[Rank {rank}] OOM as expected: {e}")
            return True
    return False
