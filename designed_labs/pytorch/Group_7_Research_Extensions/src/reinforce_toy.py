"""
reinforce_toy.py — REINFORCE on a Toy Bandit
===============================================

Student implements:
  - KArmedBandit: simple multi-armed bandit environment
  - PolicyNetwork: softmax policy over actions
  - reinforce_update(): REINFORCE with optional baseline
  - train_bandit(): full training loop

The log-probability trick and variance reduction via baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List


# ─────────────────────────────────────────────────────
# K-Armed Bandit Environment
# ─────────────────────────────────────────────────────

class KArmedBandit:
    """
    K-armed bandit with Gaussian rewards.

    Each arm k has a true mean reward μ_k.
    Pulling arm k returns: reward ~ N(μ_k, 1.0)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and step.                   ║
    ║                                                       ║
    ║  __init__(k, seed=42):                                ║
    ║    1. Set self.k = k                                   ║
    ║    2. Generate true means:                             ║
    ║       rng = np.random.RandomState(seed)                ║
    ║       self.true_means = rng.randn(k)                   ║
    ║    3. Identify optimal arm:                            ║
    ║       self.optimal_arm = np.argmax(self.true_means)    ║
    ║                                                       ║
    ║  step(action: int) -> float:                           ║
    ║    1. Sample: reward = N(true_means[action], 1.0)      ║
    ║    2. Return reward                                    ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, k: int = 10, seed: int = 42):
        raise NotImplementedError("TODO: implement KArmedBandit.__init__()")

    def step(self, action: int) -> float:
        raise NotImplementedError("TODO: implement KArmedBandit.step()")


# ─────────────────────────────────────────────────────
# Policy Network
# ─────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    A simple policy network for the bandit.

    Since the bandit is stateless, the "input" is a dummy context
    (or no input at all — just learnable logits).

    Architecture: just learnable logits → softmax → action distribution

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__, forward, select_action.    ║
    ║                                                       ║
    ║  __init__(k):                                         ║
    ║    self.logits = nn.Parameter(torch.zeros(k))          ║
    ║                                                       ║
    ║  forward():                                           ║
    ║    return F.softmax(self.logits, dim=-1)                ║
    ║                                                       ║
    ║  select_action():                                     ║
    ║    1. probs = self.forward()                           ║
    ║    2. dist = torch.distributions.Categorical(probs)    ║
    ║    3. action = dist.sample()                           ║
    ║    4. log_prob = dist.log_prob(action)                  ║
    ║    5. return action.item(), log_prob                    ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, k: int):
        super().__init__()
        raise NotImplementedError("TODO: implement PolicyNetwork.__init__()")

    def forward(self) -> torch.Tensor:
        raise NotImplementedError("TODO: implement PolicyNetwork.forward()")

    def select_action(self) -> Tuple[int, torch.Tensor]:
        raise NotImplementedError("TODO: implement PolicyNetwork.select_action()")


# ─────────────────────────────────────────────────────
# REINFORCE Update
# ─────────────────────────────────────────────────────

def reinforce_update(
    log_prob: torch.Tensor,
    reward: float,
    baseline: float,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Perform one REINFORCE update.

    The REINFORCE gradient estimator:
      ∇J ≈ (R - b) · ∇log π(a)

    Where:
      R = reward
      b = baseline (reduces variance)
      π(a) = probability of chosen action

    Parameters
    ----------
    log_prob : scalar tensor — log π(a)
    reward : float — observed reward
    baseline : float — baseline value to subtract
    optimizer : optimizer

    Returns
    -------
    float : the loss value

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. advantage = reward - baseline                      ║
    ║  2. loss = -log_prob * advantage                       ║
    ║     ↑ Negative because we MAXIMIZE reward              ║
    ║     ↑ but optimizers MINIMIZE loss                     ║
    ║  3. optimizer.zero_grad()                              ║
    ║  4. loss.backward()                                    ║
    ║  5. optimizer.step()                                   ║
    ║  6. return loss.item()                                 ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement reinforce_update()")


# ─────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────

def train_bandit(
    env: KArmedBandit,
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    num_steps: int = 2000,
    use_baseline: bool = True,
    baseline_decay: float = 0.99,
) -> Dict[str, List[float]]:
    """
    Train a policy on the bandit using REINFORCE.

    Parameters
    ----------
    env : KArmedBandit
    policy : PolicyNetwork
    optimizer : optimizer
    num_steps : int
    use_baseline : bool — whether to use a moving-average baseline
    baseline_decay : float — exponential decay for baseline

    Returns
    -------
    dict with 'rewards', 'losses', 'baselines', 'optimal_pct'

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  baseline = 0.0                                        ║
    ║  For each step:                                        ║
    ║    1. action, log_prob = policy.select_action()         ║
    ║    2. reward = env.step(action)                         ║
    ║    3. b = baseline if use_baseline else 0.0             ║
    ║    4. loss = reinforce_update(log_prob, reward, b, opt) ║
    ║    5. Update baseline:                                  ║
    ║       baseline = decay * baseline + (1-decay) * reward  ║
    ║    6. Record: reward, loss, baseline,                   ║
    ║       is_optimal (action == env.optimal_arm)            ║
    ║  Return histories                                      ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement train_bandit()")


# ─────────────────────────────────────────────────────
# Visualization (provided)
# ─────────────────────────────────────────────────────

def plot_bandit_results(
    results_no_baseline: Dict,
    results_with_baseline: Dict,
    window: int = 50,
):
    """Plot comparison of REINFORCE with and without baseline. Provided."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Smoothed rewards
    def smooth(vals, w):
        return np.convolve(vals, np.ones(w)/w, mode='valid')

    ax = axes[0]
    ax.plot(smooth(results_no_baseline['rewards'], window),
            label="No baseline", alpha=0.8)
    ax.plot(smooth(results_with_baseline['rewards'], window),
            label="With baseline", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Average Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # Optimal action percentage
    ax = axes[1]
    ax.plot(smooth(results_no_baseline['optimal_pct'], window),
            label="No baseline", alpha=0.8)
    ax.plot(smooth(results_with_baseline['optimal_pct'], window),
            label="With baseline", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("% Optimal")
    ax.set_title("Optimal Action Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    # Reward variance
    ax = axes[2]
    var_no = [np.var(results_no_baseline['rewards'][max(0,i-window):i+1])
              for i in range(len(results_no_baseline['rewards']))]
    var_with = [np.var(results_with_baseline['rewards'][max(0,i-window):i+1])
                for i in range(len(results_with_baseline['rewards']))]
    ax.plot(var_no, label="No baseline", alpha=0.8)
    ax.plot(var_with, label="With baseline", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward Variance")
    ax.set_title(f"Reward Variance (window={window})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
