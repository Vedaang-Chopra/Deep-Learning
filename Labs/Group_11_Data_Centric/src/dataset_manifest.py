"""
dataset_manifest.py — Dataset Versioning via Manifests
=========================================================

Student implements:
  - compute_file_hash(): SHA-256 hash of a file or bytes
  - build_dataset_manifest(): create a full dataset manifest
  - save_manifest() / load_manifest(): JSON serialization

Every experiment should record EXACTLY which data was used.
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


# ─────────────────────────────────────────────────────
# File / Sample Hashing
# ─────────────────────────────────────────────────────

def compute_file_hash(
    path_or_bytes: Union[str, bytes],
    algo: str = "sha256",
) -> str:
    """
    Compute a cryptographic hash of a file or raw bytes.

    Parameters
    ----------
    path_or_bytes : str (file path) or bytes
    algo : hash algorithm ('sha256', 'md5')

    Returns
    -------
    hex digest string

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. h = hashlib.new(algo)                              ║
    ║  2. If path_or_bytes is str:                           ║
    ║       Open file in binary mode, read in 8KB chunks     ║
    ║       h.update(chunk) for each                         ║
    ║     Else:                                              ║
    ║       h.update(path_or_bytes)                          ║
    ║  3. Return h.hexdigest()                               ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compute_file_hash()")


def compute_tensor_hash(tensor, algo: str = "sha256") -> str:
    """
    Compute hash of a torch tensor (for in-memory datasets).

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  1. Convert tensor to bytes: tensor.numpy().tobytes()  ║
    ║  2. Return compute_file_hash(bytes_data, algo)         ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compute_tensor_hash()")


# ─────────────────────────────────────────────────────
# Manifest Builder
# ─────────────────────────────────────────────────────

def build_dataset_manifest(
    samples: List[Dict],
    dataset_id: str,
    split_name: str = "full",
    transforms_spec: str = "",
    seed: Optional[int] = None,
    extra_meta: Optional[Dict] = None,
) -> Dict:
    """
    Build a versioned dataset manifest.

    Parameters
    ----------
    samples : list of dicts, each having at least:
        {'index': int, 'label': int/str, 'hash': str (optional)}
    dataset_id : str — identifier like 'cifar10-v1'
    split_name : str — 'train', 'val', 'test', or 'full'
    transforms_spec : str — description of preprocessing
    seed : int — random seed used
    extra_meta : dict — any additional metadata

    Returns
    -------
    manifest dict with:
      'dataset_id', 'split', 'created_at', 'num_samples',
      'seed', 'transforms', 'git_sha' (if available),
      'samples' (list), 'content_hash' (hash of all sample hashes)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Try to get git SHA:                                ║
    ║     subprocess.check_output(['git','rev-parse','HEAD']) ║
    ║     (catch exceptions → None)                          ║
    ║  2. Build manifest dict with all fields                ║
    ║  3. Compute content_hash:                              ║
    ║     hash of concatenated sample hashes                 ║
    ║     (deterministic fingerprint of the dataset)         ║
    ║  4. Return manifest                                    ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement build_dataset_manifest()")


# ─────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────

def save_manifest(manifest: Dict, path: str) -> str:
    """
    Save manifest to JSON.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  1. os.makedirs(os.path.dirname(path), exist_ok=True)  ║
    ║  2. json.dump(manifest, open(path,'w'), indent=2)      ║
    ║  3. print(f"Manifest saved: {path}")                   ║
    ║  4. Return path                                        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement save_manifest()")


def load_manifest(path: str) -> Dict:
    """
    Load manifest from JSON.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement.                                     ║
    ║                                                       ║
    ║  Return json.load(open(path))                          ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement load_manifest()")
