"""Lab 05 - Policy Gradients on a Toy MDP (scaffold).

Char-level policy network (<1M params, CPU) on a hand-verifiable toy task:
emit an arithmetic expression that evaluates to a target N. This lab is the
RL on-ramp the upstream repo skips -- it jumps straight to LLM rollouts, so
here you meet trajectories, log-probs, rewards, advantages, and variance
reduction on something you can fully inspect before touching a tokenizer.

Prerequisites: Lecture 3 (https://rlhfbook.com/course/) + Chapter 6 early
sections (https://rlhfbook.com/c/06-policy-gradients.html).

NO-SOLUTIONS SCAFFOLD: contracts only. Every mechanism (env reward/episode
logic, softmax/log-softmax, REINFORCE objective, both baselines, variance
statistics) is TODO + NotImplementedError. Implemented helpers are pure
plumbing: config loading, trajectory container bookkeeping, validators. Do
not open policy_gradients/loss.py::ReinforceLoss until YOUR version works.

Framework-neutral by design: signatures use numpy-friendly types; when you
implement you may keep numpy or switch to torch tensors inside -- tests only
rely on numpy plumbing below.

Runs torch-free at import time (numpy + PyYAML + stdlib).
"""

from __future__ import annotations

import math  # noqa: F401  (you may need it in your implementations)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Task definition & constants (given)
# ---------------------------------------------------------------------------

#: The action space over characters. Includes digits, operators, '=' padding,
#: and the end-of-episode marker.
CHAR_ACTIONS: Tuple[str, ...] = tuple("0123456789+= ")
#: Fixed max episode length (chars) so trajectories are finite by construction.
MAX_EPISODE_CHARS = 12

#: Group sizes for the leave-one-out baseline experiment (plan spec K in {2,4,16}).
K_GRID: Tuple[int, ...] = (2, 4, 16)


def expression_value(expr: str) -> Optional[int]:
    """Evaluate 'a+b'-style expressions; None if malformed (reference oracle).

    Deliberately implemented as PLUMBING: it is the environment's ground-truth
    checker you can read now to understand correctness, not the mechanism you
    learn by writing (that is the policy-gradient machinery below). Treat it
    as the verifier spec: supports single '+' operator, nonnegative ints.
    """
    expr = expr.strip()
    if "+" not in expr or "=" in expr or " " in expr:
        return None
    left, right = expr.split("+", 1)
    if not (left.isdigit() and right.isdigit()):
        return None
    return int(left) + int(right)


# ---------------------------------------------------------------------------
# Config plumbing (given)
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configs/05_policy_gradients_toy_mdp.yaml into a plain dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class ToyMDPConfig:
    """Mirror of configs/05_policy_gradients_toy_mdp.yaml (keep in sync)."""

    target_range_min: int          # inclusive low for sampled targets N
    target_range_max: int          # inclusive high
    hidden_size: int               # tiny MLP width (<1M params total)
    learning_rate: float
    episodes_per_update: int       # trajectories per gradient step
    group_size_k: int              # K samples/prompt for LOO baseline arm
    baseline: str                  # "none" | "moving_avg" | "loo"
    moving_avg_alpha: float        # EMA coefficient for the moving-average baseline
    entropy_coef: float            # optional entropy bonus (0.0 = none)
    seed: int

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ToyMDPConfig":
        """Flatten nested YAML sections into the flat dataclass."""
        env = cfg["env"]
        pol = cfg["policy"]
        pg = cfg["reinforce"]
        return cls(
            target_range_min=env["target_range_min"],
            target_range_max=env["target_range_max"],
            hidden_size=pol["hidden_size"],
            learning_rate=pg["learning_rate"],
            episodes_per_update=pg["episodes_per_update"],
            group_size_k=pg["group_size_k"],
            baseline=pg["baseline"],
            moving_avg_alpha=pg["moving_avg_alpha"],
            entropy_coef=pg["entropy_coef"],
            seed=cfg["seed"],
        )


def validate_config(cfg: Dict[str, Any]) -> None:
    """Raise ValueError on structural violations (no algorithm content).

    Checks:
    * sections present: env / policy / reinforce; scalar 'seed'
    * baseline is one of BASELINES
    * target range sane (min <= max); group_size_k in :data:`K_GRID`
    """
    baselines = ("none", "moving_avg", "loo")
    for section in ("env", "policy", "reinforce"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"config missing section: {section!r}")
    if not isinstance(cfg.get("seed"), int):
        raise ValueError("config['seed'] must be int")
    if cfg["reinforce"].get("baseline") not in baselines:
        raise ValueError(f"baseline must be one of {baselines}")
    env = cfg["env"]
    if env["target_range_min"] > env["target_range_max"]:
        raise ValueError("target_range_min > target_range_max")
    k = cfg["reinforce"].get("group_size_k")
    if k not in K_GRID:
        raise ValueError(f"group_size_k must be one of {K_GRID}")


# ---------------------------------------------------------------------------
# Trajectory container bookkeeping (given)
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """One episode's record. Lists stay parallel via index i.

    Fields:
      target_n     - the goal value N this episode conditioned on
      chars        - emitted characters (actions), len == MAX_EPISODE_CHARS at most
      log_probs    - log pi(char_i | prefix_i) per emitted char (floats)
      reward       - terminal scalar reward; None until filled by env.step
                    logic (reward assignment is YOUR implementation)
    """

    target_n: int
    chars: List[str] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    reward: Optional[float] = None

    @property
    def text(self) -> str:
        """Concatenated emitted characters (the raw policy output string)."""
        return "".join(self.chars)

    def validate(self) -> None:
        """Structural invariants (given): parallel lists, finite logs, bounds."""
        if len(self.chars) != len(self.log_probs):
            raise ValueError(
                f"parallel-list violation: {len(self.chars)} chars vs "
                f"{len(self.log_probs)} log_probs"
            )
        if len(self.chars) > MAX_EPISODE_CHARS:
            raise ValueError(f"trajectory longer than MAX_EPISODE_CHARS={MAX_EPISODE_CHARS}")
        if any(not np.isfinite(lp) for lp in self.log_probs):
            raise ValueError("non-finite log_prob recorded")


# ---------------------------------------------------------------------------
# Environment -- student work
# ---------------------------------------------------------------------------


@dataclass
class ToyArithmeticEnv:
    """Emit-characters-until-'=' environment for targets N in [lo, hi].

    Episode contract (what YOU implement):
      * reset(target_n) -> initial observation (you define its shape)
      * step(char) advances the episode; terminates when '=' is emitted or
        MAX_EPISODE_CHARS chars written
      * reward assigned at termination ONLY: +1 iff the emitted string parses
        via :func:`expression_value` and equals target_n, else 0
        (or design a shaped variant -- but document it as a deviation)
    """

    lo: int
    hi: int

    def sample_target(self, rng: np.random.Generator) -> int:
        """Uniform target draw in [lo, hi] using ``rng`` (plumbing given)."""
        return int(rng.integers(self.lo, self.hi + 1))

    def reset(self, target_n: int) -> Dict[str, Any]:
        """Start a new episode conditioning on ``target_n``.

        TODO(student): implement (define/reset observation state).
        """
        raise NotImplementedError("Lab 05: env.reset")

    def step(self, char: str) -> Tuple[Dict[str, Any], float, bool]:
        """Advance by one character emission.

        Returns (observation, reward, done). Reward is nonzero only at the
        terminal step per the class docstring contract.

        TODO(student): implement transition + terminal reward.
        """
        raise NotImplementedError("Lab 05: env.step")


# ---------------------------------------------------------------------------
# Policy & distributions -- student work (hand-written, no autograd softmax)
# ---------------------------------------------------------------------------


@dataclass
class CharPolicyNet:
    """<1M param char-level MLP mapping (target embedding, prefix encoding) -> logits.

    Parameter count guard: hidden_size^2 + vocab projections must stay under
    ~1e6 -- assert this inside forward initialization once you build it.
    """

    vocab_size: int = len(CHAR_ACTIONS)
    hidden_size: int = 128

    def init_params(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Allocate parameter arrays with sensible small-init scale.

        Contract: returns dict of named arrays (your naming choice); sizes must
        keep total params < 1_000_000 (assert it).

        TODO(student): implement initialization.
        """
        raise NotImplementedError("Lab 05: CharPolicyNet.init_params")

    def logits_for_prefix(
        self, params: Dict[str, np.ndarray], target_n: int, prefix_chars: Sequence[str]
    ) -> np.ndarray:
        """Forward pass: condition on (target, emitted-so-far) -> next-char logits.

        Contract:
          - returns shape [vocab_size] float array
          - prefix encoding scheme is yours (bag-of-chars, positional, etc.)
            BUT document parameter-count implications

        TODO(student): implement forward pass.
        """
        raise NotImplementedError("Lab 05: CharPolicyNet.logits_for_prefix")


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D logits array -> probabilities [V].

    Contract: shift-by-max stability argument documented by YOU when
    implementing; must sum to 1 within 1e-9 for the test fixtures' ranges.

    TODO(student): implement by hand (no scipy/torch).
    """
    raise NotImplementedError("Lab 05: softmax")


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Stable log-softmax -> log-probabilities [V].

    Contract: log_softmax(x) == x - logsumexp(x); consistency check
    np.allclose(softmax + log_softmax identities) holds for all fixtures.

    TODO(student): implement by hand.
    """
    raise NotImplementedError("Lab 05: log_softmax")


# ---------------------------------------------------------------------------
# Trajectory sampling -- student work
# ---------------------------------------------------------------------------


def sample_trajectory(
    params: Dict[str, np.ndarray],
    net: CharPolicyNet,
    env: ToyArithmeticEnv,
    rng: np.random.Generator,
) -> Trajectory:
    """Roll out ONE episode recording per-char log-probs alongside actions.

    Contract:
      - uses log_softmax outputs, gathering the log-prob of each EMITTED char
        (action_logprob = lp[argmax-choice index]) into trj.log_probs
      - fills trj.reward via the env's terminal step
      - calls trj.validate() before returning

    TODO(student): implement sampling loop.
    """
    raise NotImplementedError("Lab 05: sample_trajectory")


# ---------------------------------------------------------------------------
# REINFORCE machinery -- THE core student work
# ---------------------------------------------------------------------------


def trajectory_log_prob(trj: Trajectory) -> float:
    """log pi(tau) = sum of stored per-step log-probs (sum-rule derivation)."""
    return float(np.sum(trj.log_probs))


def reinforce_loss(trajs: List[Trajectory], advantages: np.ndarray) -> float:
    """The policy-gradient surrogate: mean_t( -log pi(a_t|s_t) * A(tau) ).

    Contract:
      - advantages aligned index-wise with ``trajs`` (one scalar per trajectory)
      - THIS IS the from-scratch checkpoint: write out the derivation of
        grad E[R] ~= E[ sum_t grad log pi(a_t|s_t) * A ] in the notebook first;
        the loss value itself is just the negative-log-prob-weighted sum
      - sanity identity: identical rewards across ALL trajectories with the
        constant advantage still yield nonzero loss (variance ≠ zero gradient --
        you will explain why in the debugging exercise)

    TODO(student): implement.
    """
    raise NotImplementedError("Lab 05: reinforce_loss")


def compute_advantages(
    trajs: List[Trajectory],
    mode: str,
    alpha: float = 0.9,
    running_baseline_state: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Return A(tau) per trajectory under the chosen variance-reduction scheme.

    Contract:
      - mode == 'none': A = raw reward (identical to returns here)
      - mode == 'moving_avg': A = r - b where b is an EMA of past mean rewards
        (alpha = decay); ``running_baseline_state`` dict carries {'b': float}
        across calls -- mutate it in place so plots stay continuous
      - mode == 'loo': per GROUP of K trajectories sharing a prompt/target,
        A_i = r_i - mean(r_-i) -- the leave-one-out baseline
      - shapes: output [len(trajs)] float

    TODO(student): implement all three modes (+ document bias/variance notes).
    """
    raise NotImplementedError("Lab 05: compute_advantages")


def entropy_of_policy(policy_probs: np.ndarray) -> float:
    """Shannon entropy H(p) of a categorical distribution (nats).

    Contract: input [V] probs (np.float64); expected value has natural-log units.

    TODO(student): implement.
    """
    raise NotImplementedError("Lab 05: entropy_of_policy")


# ---------------------------------------------------------------------------
# Variance statistics & experiment harness hooks -- student work
# ---------------------------------------------------------------------------


def gradient_variance_estimate(
    grads: Sequence[np.ndarray],
) -> float:
    """Mean across-parameter-dimension variance of stacked gradient vectors.

    Contract:
      - input: list/array of equal-shape gradient snapshots [G, D]
      - output: float mean of per-dimension variances across G

    TODO(student): implement (np.var axis semantics + scalar reduction).
    """
    raise NotImplementedError("Lab 05: gradient_variance_estimate")


def run_variance_experiment(
    cfg: ToyMDPConfig,
    n_trials: int = 20,
) -> Dict[str, np.ndarray]:
    """Baseline-on/off + K-in-{2,4,16} variance comparison driver.

    Contract:
      - trains briefly under each baseline mode collecting per-mode arrays of
        :func:`gradient_variance_estimate` snapshots and final success rates
      - returns {'modes': [...], 'grad_var': [n_modes arrays], 'success': [...]};
        plotting happens in the notebook from these artifacts

    TODO(student): implement experiment loop reusing the pieces above.
    """
    raise NotImplementedError("Lab 05: run_variance_experiment")


def observe_entropy_collapse(params_history: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Detect collapse-to-repeated-action from a history of policy snapshots.

    Contract:
      - takes successive policy probability matrices [T, V]
      - flags: max prob -> 1 while distinct-action rate -> 0
      - returns {'collapsed': bool, 'entropy_trace': [T], 'max_prob_trace': [T]}
        -- your two countermeasures go in the README discussion, not code

    TODO(student): implement diagnostic.
    """
    raise NotImplementedError("Lab 05: observe_entropy_collapse")
