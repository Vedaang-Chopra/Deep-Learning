"""Lab 08 starter — Reward Hacking & KL Regularization.

You are inducing Goodhart failure on purpose. This module holds the four
components of the lab; ALL implementation work is yours:

1. ``build_proxy_reward``            — the deliberate misalignment (big format bonus)
2. ``run_kl_sweep``                  — rerun your Lab 07 trainer once per beta
3. ``prepare_divergence_plot_data``  — data prep for the proxy-up / true-down figure
4. ``select_earliest_warning_metric``— which monitored series warned first

NO-SOLUTIONS RULE (plan §9.4): below are signatures, contracts, TODOs and
``NotImplementedError`` only. In particular there is intentionally **no
divergence/Goodhart-detection arithmetic anywhere in this file** — figuring out
what "divergence" means numerically is part of the lab.

This file is torch-free at module level and CPU-safe (numpy fixtures in tests).
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "load_config",
    "build_proxy_reward",
    "run_kl_sweep",
    "prepare_divergence_plot_data",
    "select_earliest_warning_metric",
]

# Relative to this file: configs/08_reward_hacking_kl.yaml
CONFIG_PATH = "configs/08_reward_hacking_kl.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the shared YAML config for this lab.

    Contract:
      - Returns a dict containing at least the required sweep keys
        ``proxy_reward_weight`` and ``kl_coef_grid``.
      - Pure-CPU; no network, no torch.

    Args:
        path: explicit config path; defaults to ``CONFIG_PATH`` resolved
            relative to this file's directory.

    Returns:
        Parsed configuration dict.
    """
    # TODO(08): implement YAML loading. Use `yaml.safe_load` from a config
    # loader helper if your Lab 07 setup has one; otherwise read directly.
    raise NotImplementedError(
        "Lab 08: implement config loading (see CONFIG_PATH docstring)."
    )


# ---------------------------------------------------------------------------
# 1) The hacked proxy reward
# ---------------------------------------------------------------------------
def build_proxy_reward(
    correctness: np.ndarray,
    format_ok: np.ndarray,
    proxy_reward_weight: float,
    correctness_weight: float = 1.0,
) -> np.ndarray:
    """Compose the deliberately misaligned proxy reward.

    Shape contract (per rollout sample):
        total reward = correctness_weight * verifier_correctness
                     + proxy_reward_weight * LARGE_format_bonus

    where LARGE_format_bonus is a fixed constant given ``format_ok``. The
    misalignment is the point: with the shipped config values, a format-perfect
    WRONG answer must outscore an untidy CORRECT one. Before training anything,
    verify this property by hand on numpy inputs.

    Args:
        correctness: array of verifier scores (0 or 1), shape ``[batch]``.
        format_ok:   array of format-compliance flags (0 or 1), shape ``[batch]``
            — same shape/type family as ``correctness`` so tests can compare.
        proxy_reward_weight: fraction of the reward budget spent on the format
            bonus; matches the required config key ``proxy_reward_weight``.
            Must be finite and >= 0. Use 0.0 to recover the baseline arm where
            proxy == truth.
        correctness_weight: weight on true verifier correctness.

    Returns:
        Array of composite rewards, shape ``[batch]``, dtype float64, elementwise
        aligned with the inputs.

    Raises:
        ValueError: if shapes mismatch, weights are negative/non-finite, or
            inputs hold values outside {0, 1} (fail loudly before a run).
    """
    # TODO(08): shape/dtype validation, then compose the two terms.
    # HINT-FREE BY DESIGN: do not hardcode "10" — pull the bonus magnitude from
    # the config's proxy_reward section so arms differ only by config.
    raise NotImplementedError(
        "Lab 08: implement the hacked proxy-reward constructor."
    )


# ---------------------------------------------------------------------------
# 2) KL-coefficient sweep runner
# ---------------------------------------------------------------------------
def run_kl_sweep(config: Dict[str, Any]) -> Dict[str, Any]:
    """Rerun the Lab 07 GRPO/RLVR run once per beta in ``kl_coef_grid``.

    Contract:
      - One job per coefficient in ``config["sweep"]["kl_coef_grid"]``; each job
        uses the SAME engine/checkpoint paths referenced under
        ``reused_lab07_assets`` and ONLY swaps the reward construction +
        beta. Do not fork the Lab 07 trainer.
      - Every job appends JSONL metrics rows that contain AT MINIMUM the keys
        listed in ``config["metrics"]["required"]``: step, proxy_reward,
        true_accuracy, format_rate, approx_kl.
      - Cluster discipline: each launch must be tmux-wrapped with pinned
        CUDA_VISIBLE_DEVICES after an occupancy check (plan §7) — this function
        only ORCHESTRATES launches and collects artifact paths; it never starts
        unattended jobs itself.

    Args:
        config: parsed config dict (see ``load_config``).

    Returns:
        Dict mapping each KL coefficient value -> metrics JSONL artifact path
        (one entry per completed arm).
    """
    # TODO(08): loop over kl_coef_grid, invoke the reused Lab 07 trainer per
    # arm, verify each produced metrics file contains all `required` keys.
    # The true metric (`true_accuracy`) must be tracked separately from
    # `proxy_reward` in every row — add its missing assertion before launching.
    raise NotImplementedError("Lab 08: implement the KL-coefficient sweep runner.")


# ---------------------------------------------------------------------------
# 3) Divergence-plot data preparation (proxy-up / true-down)
# ---------------------------------------------------------------------------
def prepare_divergence_plot_data(
    runs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Prep step-series data for the proxy-up / true-down divergence plot.

    Each input run dict carries the step-series from one sweep arm, e.g.::

        {
          "label": "beta=0.0",
          "steps":  np.ndarray shape [T],
          "proxy_reward":    np.ndarray shape [T],   # rising...
          "true_accuracy":   np.ndarray shape [T],   # ...or falling later
        }

    Contract:
      - Validate alignment (all series within a run have equal length T) BEFORE
        any plotting math.
      - Emit a plain dict the notebook can plot directly: one x-series plus
        normalized y-series per run/metric pair. Series MUST remain comparable
        across runs despite different absolute scales between proxy_reward
        (~O(bonus)) and true_accuracy (0..1) — choose ONE normalization rule,
        document it here, and apply it uniformly.
      - NO divergence computation lives here: no Goodhart onset detection, no
        peak-finding logic of any kind. Output is series data only;
        interpretation is yours in the notebook.

    Returns:
        Plot-ready dict: {"normalization": "<name>", "series": [...]}.
    """
    # TODO(08): validate T-alignment per run, normalize, package series list.
    # Keep numbers unannotated: this stays pure data prep.
    raise NotImplementedError(
        "Lab 08: implement divergence-plot data preparation."
    )


# ---------------------------------------------------------------------------
# 4) Earliest-warning-metric selector
# ---------------------------------------------------------------------------
def select_earliest_warning_metric(
    runs: Sequence[Dict[str, Any]],
    candidates: Sequence[str] = (
        "format_rate", "response_length", "approx_kl", "group_reward_variance"
    ),
    reference_metric: str = "true_accuracy",
) -> str:
    """Pick the candidate metric whose excursion precedes the truth drop earliest.

    A metric "warns" when it moves off its own pre-degradation baseline BEFORE
    ``reference_metric`` begins its sustained decline. How you quantify "moves
    off baseline" and "sustained decline" is the analytical core of this lab —
    define it explicitly and defend it in the notebook.

    Args:
        runs: run dicts as in :func:`prepare_divergence_plot_data`, extended
            with one array per name in ``candidates`` (each length T).
        candidates: names of warning-metric candidates present in every run.
        reference_metric: the late signal to compare against.

    Returns:
        Name of the single candidate (same string as passed in) with the
        largest lead time, aggregated over runs.

    Raises:
        KeyError: if a requested metric is missing from any run (never silently
            drop candidates).
    """
    # TODO(08): define + compute lead times per candidate per run; aggregate;
    # return the winner. Reminder: a warning signal that fires AFTER the truth
    # drop is useless — handle that outcome explicitly rather than taking a max
    # over negative leads silently.
    raise NotImplementedError(
        "Lab 08: implement the earliest-warning-metric selector."
    )


if __name__ == "__main__":
    print(
        "Lab 08 scaffold — nothing implemented yet.\n"
        f"Config expected at {CONFIG_PATH}. Start with build_proxy_reward,\n"
        "then sanity-check misalignment on numpy inputs before any GPU run."
    )
