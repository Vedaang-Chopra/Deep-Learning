"""
run_logging.py — Reproducible Experiment Logging
====================================================

Student implements:
  - create_run_dir(): timestamped run directory
  - save_config() / load_config(): experiment config
  - log_metrics(): append metrics (JSONL format)
  - save_model() / load_model(): checkpoint management
  - capture_env(): environment snapshot

Every run must be reproducible from its bundle.
"""

import os
import json
import time
import platform
from datetime import datetime
from typing import Dict, Optional, Any
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────
# Run Directory
# ─────────────────────────────────────────────────────

def create_run_dir(
    base: str = "runs",
    prefix: str = "run",
) -> str:
    """
    Create a timestamped run directory.

    Returns
    -------
    str — path to the run directory, e.g. runs/run_20240115_143022

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. timestamp = datetime.now().strftime(...)           ║
    ║  2. run_dir = os.path.join(base, f"{prefix}_{ts}")    ║
    ║  3. os.makedirs(run_dir, exist_ok=True)                ║
    ║  4. Print run_dir                                      ║
    ║  5. Return run_dir                                     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement create_run_dir()")


# ─────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────

def save_config(config: Dict, run_path: str) -> str:
    """
    Save experiment config to JSON.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  1. path = os.path.join(run_path, "config.json")       ║
    ║  2. json.dump(config, ..., indent=2)                   ║
    ║  3. Return path                                        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement save_config()")


def load_config(run_path: str) -> Dict:
    """Load config from run directory."""
    path = os.path.join(run_path, "config.json")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────
# Metrics Logging
# ─────────────────────────────────────────────────────

def log_metrics(
    step_or_epoch: int,
    metrics: Dict[str, float],
    run_path: str,
    filename: str = "metrics.jsonl",
) -> None:
    """
    Append metrics to a JSONL file (one JSON object per line).

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. entry = {'step': step_or_epoch, **metrics,         ║
    ║              'timestamp': datetime.now().isoformat()}   ║
    ║  2. path = os.path.join(run_path, filename)            ║
    ║  3. Open file in append mode ('a')                     ║
    ║  4. Write json.dumps(entry) + newline                  ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement log_metrics()")


def load_metrics(run_path: str, filename: str = "metrics.jsonl") -> list:
    """Load all metrics from JSONL file. Provided."""
    path = os.path.join(run_path, filename)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ─────────────────────────────────────────────────────
# Model Checkpointing
# ─────────────────────────────────────────────────────

def save_model(
    model: nn.Module,
    run_path: str,
    filename: str = "model.pt",
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
) -> str:
    """
    Save model checkpoint.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  1. checkpoint = {                                     ║
    ║       'model_state_dict': model.state_dict(),          ║
    ║       'epoch': epoch,                                  ║
    ║     }                                                  ║
    ║  2. If optimizer: add 'optimizer_state_dict'           ║
    ║  3. path = os.path.join(run_path, filename)            ║
    ║  4. torch.save(checkpoint, path)                       ║
    ║  5. Print file size                                    ║
    ║  6. Return path                                        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement save_model()")


def load_model(
    model: nn.Module,
    run_path: str,
    filename: str = "model.pt",
    device: str = "cpu",
) -> Dict:
    """
    Load model checkpoint.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  1. checkpoint = torch.load(path, map_location=device) ║
    ║  2. model.load_state_dict(checkpoint['model_state_dict'])║
    ║  3. Return checkpoint (for optimizer, epoch, etc.)     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement load_model()")


# ─────────────────────────────────────────────────────
# Environment Capture
# ─────────────────────────────────────────────────────

def capture_env(run_path: str) -> str:
    """
    Capture environment info for reproducibility.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  env = {                                               ║
    ║    'python_version': platform.python_version(),        ║
    ║    'torch_version': torch.__version__,                 ║
    ║    'cuda_available': torch.cuda.is_available(),        ║
    ║    'cuda_version': torch.version.cuda,                 ║
    ║    'platform': platform.platform(),                    ║
    ║    'timestamp': datetime.now().isoformat(),            ║
    ║  }                                                     ║
    ║  Save to run_path/environment.json                     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement capture_env()")


# ─────────────────────────────────────────────────────
# Run Directory Viewer (provided)
# ─────────────────────────────────────────────────────

def print_run_tree(run_path: str):
    """Print the contents of a run directory as a tree. Provided."""
    print(f"\n📂 {run_path}/")
    for root, dirs, files in os.walk(run_path):
        level = root.replace(run_path, '').count(os.sep)
        indent = '  ' * (level + 1)
        prefix = '├── ' if level > 0 else '├── '
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"{indent}{prefix}{f} ({size_str})")
