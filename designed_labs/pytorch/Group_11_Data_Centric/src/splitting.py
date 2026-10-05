"""
splitting.py — Train/Val/Test Splitting + Leakage Detection
=============================================================

Student implements:
  - make_split(): random split with seed
  - make_stratified_split(): class-balanced split
  - leakage_check_exact_duplicates(): find identical samples across splits
  - leakage_check_near_duplicates(): find near-duplicate embeddings

Leakage is the silent killer of ML evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter


# ─────────────────────────────────────────────────────
# Basic Splitting
# ─────────────────────────────────────────────────────

def make_split(
    indices: List[int],
    seed: int,
    ratios: Tuple[float, ...] = (0.8, 0.1, 0.1),
) -> Dict[str, List[int]]:
    """
    Random split of indices into train/val/test.

    Parameters
    ----------
    indices : list of sample indices
    seed : random seed for reproducibility
    ratios : (train, val, test) fractions — must sum to 1.0

    Returns
    -------
    {'train': [...], 'val': [...], 'test': [...]}

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Assert sum(ratios) ≈ 1.0                           ║
    ║  2. rng = np.random.RandomState(seed)                  ║
    ║  3. shuffled = rng.permutation(indices)                ║
    ║  4. Compute split points from ratios                   ║
    ║  5. Slice into train/val/test                          ║
    ║  6. Return dict                                        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement make_split()")


def make_stratified_split(
    labels: List[int],
    seed: int,
    ratios: Tuple[float, ...] = (0.8, 0.1, 0.1),
) -> Dict[str, List[int]]:
    """
    Stratified split: each split has same class distribution.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Group indices by label                             ║
    ║  2. For each label group:                              ║
    ║       make_split(group_indices, seed, ratios)          ║
    ║  3. Concatenate splits across labels                   ║
    ║  4. Return dict with combined splits                   ║
    ║                                                       ║
    ║  This ensures each class is proportionally represented ║
    ║  in train, val, AND test.                              ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement make_stratified_split()")


# ─────────────────────────────────────────────────────
# Leakage Detection
# ─────────────────────────────────────────────────────

def leakage_check_exact_duplicates(
    train_items: List,
    val_items: List,
    test_items: List,
    key_fn=None,
) -> Dict:
    """
    Check for exact duplicates across splits.

    Parameters
    ----------
    train_items, val_items, test_items : lists of items (hashes, indices, etc.)
    key_fn : optional function to extract a key from each item
             (e.g., lambda item: item['hash'])

    Returns
    -------
    dict with:
      'train_val_overlap': list of duplicated keys
      'train_test_overlap': list
      'val_test_overlap': list
      'total_leaks': int
      'passed': bool (True if no leaks)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. If key_fn, apply it to all items                   ║
    ║  2. Convert each split to a set                        ║
    ║  3. Compute intersections                              ║
    ║  4. Return report                                      ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement leakage_check_exact_duplicates()")


def leakage_check_near_duplicates(
    embeddings_train: 'np.ndarray',
    embeddings_val: 'np.ndarray',
    threshold: float = 0.95,
) -> Dict:
    """
    Check for near-duplicate samples using embedding cosine similarity.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. L2-normalize both embedding matrices               ║
    ║  2. Compute cosine similarity matrix:                  ║
    ║     sim = train_normed @ val_normed.T                  ║
    ║  3. Find pairs where sim > threshold                   ║
    ║  4. Return:                                            ║
    ║     'near_duplicate_pairs': [(train_idx, val_idx, sim)]║
    ║     'num_suspects': int                                ║
    ║     'max_similarity': float                            ║
    ║     'passed': bool                                     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement leakage_check_near_duplicates()")


# ─────────────────────────────────────────────────────
# Utilities (provided)
# ─────────────────────────────────────────────────────

def print_split_stats(splits: Dict[str, List[int]], labels: List[int]):
    """Print per-split class distribution. Provided."""
    for split_name, indices in splits.items():
        split_labels = [labels[i] for i in indices]
        counts = Counter(split_labels)
        total = len(indices)
        print(f"\n  {split_name}: {total} samples")
        for cls in sorted(counts.keys()):
            print(f"    class {cls}: {counts[cls]:>5} ({counts[cls]/total:.1%})")


def print_leakage_report(report: Dict):
    """Print formatted leakage report. Provided."""
    status = "✅ PASS — no leakage" if report['passed'] else "❌ FAIL — leakage detected"
    print(f"\n{'─'*50}")
    print(f"Leakage Report: {status}")
    print(f"{'─'*50}")
    for key in ['train_val_overlap', 'train_test_overlap', 'val_test_overlap']:
        if key in report:
            n = len(report[key])
            print(f"  {key}: {n} duplicates")
    print(f"  Total leaks: {report.get('total_leaks', 0)}")
    print(f"{'─'*50}")
