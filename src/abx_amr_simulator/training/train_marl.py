"""Multi-agent HRL PPO training loop (Option B: custom rollout, SB3 policy objects).

`MARLTrainer` drives `MARLOptionsWrapper` directly with one SB3 `PPO` object per agent.
It never calls `PPO.learn()`. Instead it owns the complete rollout-collect-update cycle:

  1. Call `wrapper.step(pending_selections)` to advance the MARL environment.
  2. For each completing agent, store its transition in its own `RolloutBuffer`.
  3. When a buffer fills, compute GAE and call `agent.train()`.
  4. Handle episode boundaries, evaluation, and checkpointing.

Both HRL PPO and HRL RecurrentPPO (HRL_RPPO) are supported. See
ADDING_MARL_SUPPORT_TO_ABX_AMR_SIMULATOR.md Section 3.5 for the design rationale and
Section 3.5a for the training loop strategy analysis.

Usage:

    wrapper = MARLOptionsWrapper(base_env=..., option_libraries=..., gamma=0.99)
    agents = {
        aid: make_ppo_for_agent(wrapper, aid, n_steps=256, ...)
        for aid in wrapper.base_env.possible_agents
    }
    trainer = MARLTrainer(
        wrapper=wrapper,
        agents=agents,
        n_steps=256,
        total_primitive_steps=500_000,
        checkpoint_dir=Path("results/run_1/checkpoints"),
    )
    trainer.train()
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib import RecurrentPPO

from abx_amr_simulator.callbacks.marl_callbacks import run_marl_eval_episodes
from abx_amr_simulator.hrl.marl_wrapper import MARLOptionsWrapper
from abx_amr_simulator.hrl.rl_algorithms.recurrent_ppo_masked import RecurrentPPO_Masked


_RUN_TIMESTAMP_SUFFIX_PATTERN = re.compile(pattern=r"_\d{8}_\d{6}$")


def _ensure_timestamped_run_name(*, run_name: str) -> str:
    """Return run_name with a YYYYMMDD_HHMMSS suffix.

    If ``run_name`` already ends with ``_YYYYMMDD_HHMMSS``, it is returned
    unchanged. Otherwise, the current timestamp is appended.
    """
    if _RUN_TIMESTAMP_SUFFIX_PATTERN.search(string=run_name):
        return run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{run_name}_{timestamp}"


def _find_existing_timestamped_run_dir(
    *,
    results_dir: "str | Path",
    run_name_prefix: str,
    agent_ids: List[str],
) -> Optional[Path]:
    """Find a timestamped run dir whose final models are all present.

    Returns the first matching run directory (sorted newest-first) that
    contains all ``final_model_{aid}.zip`` files.
    """
    results_dir_path = Path(results_dir)
    pattern = f"{run_name_prefix}_????????_??????"
    for candidate in sorted(results_dir_path.glob(pattern), reverse=True):
        checkpoint_dir = candidate / "checkpoints"
        all_exist = all(
            (checkpoint_dir / f"final_model_{aid}.zip").exists()
            for aid in agent_ids
        )
        if all_exist:
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Config override helpers
# --------------------------------------------------------------------------- #

def _coerce_value(value_str: str, existing: Any) -> Any:
    """Coerce value_str to the same type as existing.

    bool is checked before int because bool is a subtype of int in Python.
    If existing is neither bool, int, nor float, the string is returned as-is.

    Args:
        value_str: String representation of the new value.
        existing: Current value at the target path (used for type inference).

    Returns:
        value_str coerced to the type of existing, or value_str unchanged.

    Raises:
        ValueError: If existing is bool but value_str cannot be interpreted as one.
    """
    if isinstance(existing, bool):
        lower = value_str.lower()
        if lower in ("true", "1", "yes"):
            return True
        elif lower in ("false", "0", "no"):
            return False
        else:
            raise ValueError(
                f"Cannot coerce {value_str!r} to bool. "
                "Use 'true'/'false', '1'/'0', or 'yes'/'no'."
            )
    elif isinstance(existing, int):
        return int(value_str)
    elif isinstance(existing, float):
        return float(value_str)
    else:
        return value_str


def _apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply dot-path key=value overrides to a nested config dict (in-place).

    Each override has the form ``"key.path=value"``. Path segments are split on
    dots; integer-valued segments (e.g. ``agents.0``) index into lists. The
    value string is coerced to the same Python type as the existing value at
    that path (int, float, or bool); otherwise it is left as a string.

    Args:
        config: Nested config dict. Mutated in-place.
        overrides: List of ``"key.path=value"`` strings.

    Returns:
        The same config dict after all overrides are applied.

    Raises:
        ValueError: If an override string has no ``=`` separator.
        KeyError: If a path segment does not exist in the config.
        IndexError: If an integer segment is out of range for a list.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Override must have the form 'key.path=value', got: {override!r}"
            )
        key_path, value_str = override.split("=", 1)
        segments = key_path.split(".")

        # Walk to the parent of the target node.
        node: Any = config
        for seg in segments[:-1]:
            if isinstance(node, list):
                try:
                    node = node[int(seg)]
                except ValueError:
                    raise KeyError(
                        f"Expected integer list index at segment {seg!r} "
                        f"in path {key_path!r}"
                    )
            elif isinstance(node, dict):
                if seg not in node:
                    raise KeyError(
                        f"Key {seg!r} not found in config at path {key_path!r}"
                    )
                node = node[seg]
            else:
                raise KeyError(
                    f"Cannot navigate into {type(node).__name__} "
                    f"at segment {seg!r} in path {key_path!r}"
                )

        # Set the final segment.
        last = segments[-1]
        if isinstance(node, list):
            try:
                idx = int(last)
            except ValueError:
                raise KeyError(
                    f"Expected integer list index for final segment {last!r} "
                    f"in path {key_path!r}"
                )
            existing = node[idx]
            node[idx] = _coerce_value(value_str, existing)
        elif isinstance(node, dict):
            if last not in node:
                raise KeyError(
                    f"Key {last!r} not found in config at path {key_path!r}"
                )
            existing = node[last]
            node[last] = _coerce_value(value_str, existing)
        else:
            raise KeyError(
                f"Cannot set value on {type(node).__name__} "
                f"at final segment {last!r} in path {key_path!r}"
            )

    return config


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def make_dummy_gym_env(
    obs_space: spaces.Box,
    action_space: spaces.Discrete,
) -> gym.Env:
    """Return a minimal Gymnasium env with the given spaces.

    Used only so SB3's `PPO(env=dummy_env)` can initialise the policy network
    and rollout buffer with the correct observation/action dimensions. The env
    is never stepped during MARL training.

    Args:
        obs_space: Manager-level observation space for one agent.
        action_space: Manager-level action space for one agent.

    Returns:
        A lightweight gym.Env whose spaces match the arguments.
    """

    class _DummyEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = obs_space
            self.action_space = action_space

        def reset(self, seed=None, options=None):
            return np.zeros(obs_space.shape, dtype=np.float32), {}

        def step(self, action):
            return (
                np.zeros(obs_space.shape, dtype=np.float32),
                0.0,
                False,
                False,
                {},
            )

    return _DummyEnv()


def make_ppo_for_agent(
    wrapper: MARLOptionsWrapper,
    agent_id: str,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.02,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: Optional[int] = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 0,
) -> PPO:
    """Instantiate a PPO agent for one agent in a MARLOptionsWrapper.

    The PPO object is initialised with a dummy Gymnasium env of the correct
    observation/action space so SB3 builds the policy network and rollout
    buffer with the right dimensions. `PPO.learn()` is never called on this
    object — the training loop in `MARLTrainer` drives everything directly.

    Args:
        wrapper: The MARLOptionsWrapper whose spaces define the agent's inputs.
        agent_id: Which agent's spaces to use.
        n_steps: Rollout buffer capacity in manager-level steps.
        batch_size: Mini-batch size for PPO gradient updates.
        n_epochs: Number of gradient epochs per buffer flush.
        learning_rate: PPO learning rate.
        gamma: Discount factor (should match wrapper.gamma for consistency).
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        ent_coef: Entropy coefficient.
        vf_coef: Value function coefficient.
        max_grad_norm: Gradient clipping norm.
        seed: Optional random seed for the PPO policy.
        tensorboard_log: Optional path for TensorBoard logging.
        verbose: SB3 verbosity (0 = silent).

    Returns:
        Initialised PPO object ready for use with MARLTrainer.
    """
    dummy_env = make_dummy_gym_env(
        obs_space=wrapper.observation_spaces[agent_id],
        action_space=wrapper.action_spaces[agent_id],
    )
    return PPO(
        policy="MlpPolicy",
        env=dummy_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )


def make_recurrent_ppo_for_agent(
    wrapper: MARLOptionsWrapper,
    agent_id: str,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.02,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    lstm_hidden_size: int = 64,
    n_lstm_layers: int = 1,
    enable_critic_lstm: bool = True,
    seed: Optional[int] = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 0,
) -> RecurrentPPO_Masked:
    """Instantiate a RecurrentPPO agent for one agent in a MARLOptionsWrapper.

    Creates a ``RecurrentPPO_Masked`` with ``MlpLstmPolicy`` and the specified
    LSTM configuration. Like ``make_ppo_for_agent``, the object is initialised
    with a dummy Gymnasium env so SB3 builds the policy network and
    ``RecurrentRolloutBuffer`` with the correct dimensions. ``learn()`` is
    never called — ``MARLTrainer`` drives the rollout loop directly.

    ``RecurrentPPO_Masked`` is used unconditionally (rather than plain
    ``RecurrentPPO``) because the MARL trainer already filters non-trainable
    transitions in ``_store_transition()``, so the masking logic in
    ``RecurrentPPO_Masked.train()`` simply sees all-ones masks. Using the
    masked variant keeps the agent type consistent with the single-agent
    HRL_RPPO path.

    Args:
        wrapper: The MARLOptionsWrapper whose spaces define the agent's inputs.
        agent_id: Which agent's spaces to use.
        n_steps: Rollout buffer capacity in manager-level steps.
        batch_size: Mini-batch size for PPO gradient updates.
        n_epochs: Number of gradient epochs per buffer flush.
        learning_rate: PPO learning rate.
        gamma: Discount factor (should match wrapper.gamma for consistency).
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        ent_coef: Entropy coefficient.
        vf_coef: Value function coefficient.
        max_grad_norm: Gradient clipping norm.
        lstm_hidden_size: Number of units in each LSTM layer.
        n_lstm_layers: Number of stacked LSTM layers.
        enable_critic_lstm: Whether the critic network also uses an LSTM
            (True) or a feedforward network (False).
        seed: Optional random seed for the policy.
        tensorboard_log: Optional path for TensorBoard logging.
        verbose: SB3 verbosity (0 = silent).

    Returns:
        Initialised RecurrentPPO_Masked object ready for use with MARLTrainer.
    """
    dummy_env = make_dummy_gym_env(
        obs_space=wrapper.observation_spaces[agent_id],
        action_space=wrapper.action_spaces[agent_id],
    )
    policy_kwargs = {
        "lstm_hidden_size": lstm_hidden_size,
        "n_lstm_layers": n_lstm_layers,
        "enable_critic_lstm": enable_critic_lstm,
    }
    return RecurrentPPO_Masked(
        policy="MlpLstmPolicy",
        env=dummy_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )


# --------------------------------------------------------------------------- #
# MARLTrainer
# --------------------------------------------------------------------------- #

class MARLTrainer:
    """Custom rollout trainer for multi-agent HRL PPO over MARLOptionsWrapper.

    Drives `MARLOptionsWrapper` with per-agent `RolloutBuffer` instances that
    fill independently at each agent's natural option-completion rate. When a
    buffer fills, GAE is computed and `agent.train()` is called for that agent
    immediately, independent of the other agents' buffer states.

    Attributes:
        wrapper: The MARLOptionsWrapper being trained against.
        agents: Per-agent PPO objects (policy + optimizer).
        n_steps: Rollout buffer capacity per agent (manager-level steps).
        total_primitive_steps: Training budget in primitive env steps.
        checkpoint_dir: Directory where model checkpoints are written.
        eval_freq_episodes: Evaluate every N completed episodes.
        save_freq_episodes: Save periodic checkpoints every N completed episodes.
        n_eval_episodes: Number of deterministic eval episodes per evaluation.
        verbose: 0 = silent, 1 = progress summaries.
    """

    def __init__(
        self,
        wrapper: MARLOptionsWrapper,
        agents: Dict[str, Any],
        n_steps: int | Dict[str, int],
        total_primitive_steps: int,
        checkpoint_dir: Path,
        eval_freq_episodes: int = 10,
        save_freq_episodes: Optional[int] = None,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ) -> None:
        """Initialise MARLTrainer.

        Args:
            wrapper: Pre-instantiated MARLOptionsWrapper (env + option libraries).
            agents: Dict mapping agent_id → PPO or RecurrentPPO object. Must
                contain one entry per agent in wrapper.base_env.possible_agents.
                Each agent must have been constructed with the matching
                observation/action space (e.g. via make_ppo_for_agent() or
                make_recurrent_ppo_for_agent()).
            n_steps: Rollout buffer capacity per agent in manager steps. May
                be a single shared integer for all agents or a dict mapping
                each agent_id to its own rollout size. Values should match the
                n_steps used when constructing each PPO object.
            total_primitive_steps: Training budget measured in primitive env
                steps (not manager steps). Training stops when this is reached.
            checkpoint_dir: Directory for saving model checkpoints. Created if
                it does not exist.
            eval_freq_episodes: Run evaluation every this many completed
                training episodes.
            save_freq_episodes: Save periodic checkpoints every this many
                completed training episodes. If None, defaults to
                eval_freq_episodes (backward-compatible behavior).
            n_eval_episodes: Number of deterministic episodes per evaluation.
            verbose: 0 = silent; 1 = print episode/eval summaries.

        Raises:
            ValueError: If any agent in wrapper is missing from agents, or if
                an expected rollout size does not match that agent's PPO
                rollout buffer size.
        """
        self.wrapper = wrapper
        self.agents = agents
        self.total_primitive_steps = total_primitive_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.eval_freq_episodes = eval_freq_episodes
        self.save_freq_episodes = (
            eval_freq_episodes if save_freq_episodes is None else save_freq_episodes
        )
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose

        if self.eval_freq_episodes <= 0:
            raise ValueError(
                f"eval_freq_episodes must be > 0, got {self.eval_freq_episodes}"
            )
        if self.save_freq_episodes <= 0:
            raise ValueError(
                f"save_freq_episodes must be > 0, got {self.save_freq_episodes}"
            )

        self._agent_ids: List[str] = list(wrapper.base_env.possible_agents)

        # Validate that every agent has a PPO entry
        for aid in self._agent_ids:
            if aid not in agents:
                raise ValueError(
                    f"No PPO agent provided for '{aid}'. "
                    f"agents keys: {sorted(agents.keys())}"
                )

        if isinstance(n_steps, int):
            self.n_steps = {aid: n_steps for aid in self._agent_ids}
        else:
            self.n_steps = {}
            for aid in self._agent_ids:
                if aid not in n_steps:
                    raise ValueError(
                        f"Missing n_steps entry for '{aid}'. "
                        f"Provided keys: {sorted(n_steps.keys())}"
                    )
                self.n_steps[aid] = int(n_steps[aid])

        # Detect which agents are recurrent (RecurrentPPO or subclass).
        self._is_recurrent: Dict[str, bool] = {
            aid: isinstance(agents[aid], RecurrentPPO)
            for aid in self._agent_ids
        }

        # Validate and extract rollout buffers from the PPO/RecurrentPPO objects.
        # The agent objects already own their buffers — we use them directly.
        self._buffers: Dict[str, Union[RolloutBuffer, RecurrentRolloutBuffer]] = {}
        for aid in self._agent_ids:
            buf = agents[aid].rollout_buffer
            expected_n_steps = self.n_steps[aid]
            if buf.buffer_size != expected_n_steps:
                raise ValueError(
                    f"Agent '{aid}': rollout buffer size {buf.buffer_size} "
                    f"does not match n_steps={expected_n_steps}. Construct the "
                    f"agent with n_steps={expected_n_steps}."
                )
            self._buffers[aid] = buf

        # Per-agent LSTM state management for recurrent agents.
        # _lstm_states: current LSTM hidden/cell states, carried across steps.
        #   - For recurrent agents: initialized to zeros (matching _last_lstm_states).
        #   - For non-recurrent agents: None.
        # _lstm_states_at_action: snapshot of LSTM states at the time of action
        #   selection (before the forward pass updates them). Needed for buffer
        #   storage. Initialized to None; populated during action selection.
        self._lstm_states: Dict[str, Optional[RNNStates]] = {}
        self._lstm_states_at_action: Dict[str, Optional[RNNStates]] = {}
        for aid in self._agent_ids:
            if self._is_recurrent[aid]:
                self._lstm_states[aid] = self._make_zero_lstm_states(aid)
            else:
                self._lstm_states[aid] = None
            self._lstm_states_at_action[aid] = None

        # Per-agent best eval reward for best-model checkpointing
        self._best_mean_reward: Dict[str, float] = {
            aid: -math.inf for aid in self._agent_ids
        }

        # Initialise SB3 logger and internal state on each PPO object so that
        # agent.train() works without calling agent.learn() first.
        for aid in self._agent_ids:
            agents[aid]._setup_learn(total_timesteps=total_primitive_steps)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Public entry point
    # ---------------------------------------------------------------------- #

    def train(self) -> None:
        """Run the full MARL training loop.

        Advances `wrapper` primitive-step by primitive-step until
        `total_primitive_steps` have been executed. Each completing agent's
        manager transition is stored in its own rollout buffer; when a buffer
        fills, GAE is computed and the agent's policy is updated immediately.
        Evaluation and checkpointing happen at episode boundaries.
        """
        primitive_steps_elapsed = 0
        episodes_completed = 0

        # Reset environment and initialise per-agent state
        obs_dict, _ = self.wrapper.reset()
        last_obs: Dict[str, np.ndarray] = dict(obs_dict)
        last_episode_start: Dict[str, bool] = {
            aid: True for aid in self._agent_ids
        }

        # All agents need their first option selection after reset
        pending_selections = self._predict_all(last_obs, last_episode_start)

        for aid in self._agent_ids:
            self._buffers[aid].reset()  # start with a clean buffer

        while primitive_steps_elapsed < self.total_primitive_steps:
            m_obs, m_rew, m_term, m_trunc, m_info = self.wrapper.step(
                pending_selections
            )

            # Count actual primitive steps executed this wrapper step
            prim_steps = self._count_primitive_steps(m_info)
            primitive_steps_elapsed += prim_steps

            # Store transitions for completing agents
            for aid, next_obs in m_obs.items():
                self._store_transition(
                    aid=aid,
                    obs=last_obs[aid],
                    action=pending_selections[aid],
                    reward=m_rew[aid],
                    episode_start=last_episode_start[aid],
                    m_info_entry=m_info[aid],
                )
                last_episode_start[aid] = False

                # Flush buffer and update policy if full
                self._maybe_update(
                    aid=aid,
                    next_obs=next_obs,
                    done=m_term[aid] or m_trunc[aid],
                )

                last_obs[aid] = next_obs

            # Handle episode boundary
            episode_done = any(m_term.values()) or any(m_trunc.values())
            if episode_done:
                episodes_completed += 1

                if self.verbose >= 1:
                    print(
                        f"[primitive step {primitive_steps_elapsed:,}] "
                        f"episode {episodes_completed} done"
                    )

                self._maybe_eval_and_checkpoint(
                    episodes_completed=episodes_completed,
                    primitive_steps_elapsed=primitive_steps_elapsed,
                )

                # Reset environment.  Do NOT reset the rollout buffers here —
                # buffers persist across episode boundaries and are only reset
                # inside _maybe_update() after a gradient update.  The
                # episode_start flag passed to buffer.add() marks the boundary
                # so GAE is computed correctly across episodes.
                obs_dict, _ = self.wrapper.reset()
                last_obs = dict(obs_dict)
                last_episode_start = {aid: True for aid in self._agent_ids}

                # Reset LSTM states to zeros for recurrent agents.
                #
                # Strictly speaking, this explicit reset is redundant: the
                # _predict_all() call below passes episode_start=True for every
                # agent, and _predict_single() forwards that flag to
                # policy.forward(), which internally calls _process_sequence().
                # _process_sequence() zeros the LSTM hidden/cell states
                # whenever episode_starts is True, so the forward pass would
                # produce the same output regardless of the pre-call state.
                #
                # We reset explicitly anyway for two reasons:
                # 1. Clarity — a reader of train() can see that LSTM states are
                #    cleaned up at episode boundaries without needing to
                #    understand the internals of _process_sequence().
                # 2. Robustness — if _predict_single() is ever called with
                #    episode_start=False at a boundary (e.g. due to a future
                #    refactor), stale LSTM states from the previous episode
                #    would silently corrupt the new episode's predictions.
                for aid in self._agent_ids:
                    if self._is_recurrent[aid]:
                        self._lstm_states[aid] = self._make_zero_lstm_states(
                            aid
                        )

                pending_selections = self._predict_all(
                    last_obs, last_episode_start
                )
            else:
                # Only completing agents need new option selections
                for aid, next_obs in m_obs.items():
                    pending_selections[aid] = self._predict_single(
                        aid, next_obs, episode_start=False,
                    )

        # Save final models
        for aid in self._agent_ids:
            path = self.checkpoint_dir / f"final_model_{aid}"
            self.agents[aid].save(str(path))
            if self.verbose >= 1:
                print(f"Final model saved: {path}.zip")

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _make_zero_lstm_states(self, aid: str) -> RNNStates:
        """Create zero-initialized LSTM states for a recurrent agent.

        Returns an ``RNNStates`` named tuple with ``.pi`` and ``.vf`` fields,
        each containing a ``(hidden, cell)`` tuple of zero tensors with shape
        ``(n_lstm_layers, 1, lstm_hidden_size)``.  The ``n_envs=1`` dimension
        reflects the MARL convention of one dummy env per agent.

        Args:
            aid: Agent ID.  Must correspond to a recurrent agent
                (``self._is_recurrent[aid]`` is True).

        Returns:
            RNNStates with all-zeros hidden and cell states.
        """
        lstm = self.agents[aid].policy.lstm_actor
        shape = (lstm.num_layers, 1, lstm.hidden_size)
        device = self.agents[aid].device
        return RNNStates(
            pi=(torch.zeros(shape, device=device),
                torch.zeros(shape, device=device)),
            vf=(torch.zeros(shape, device=device),
                torch.zeros(shape, device=device)),
        )

    def _predict_single(
        self,
        aid: str,
        obs: np.ndarray,
        episode_start: bool,
    ) -> int:
        """Select an option for a single agent, tracking LSTM states if recurrent.

        For recurrent agents, uses ``policy.forward()`` to thread LSTM states
        through the prediction and saves a pre-forward snapshot in
        ``_lstm_states_at_action`` for later buffer storage.

        For non-recurrent agents, uses ``policy.predict()`` (unchanged from
        the original code path).

        Args:
            aid: Agent ID.
            obs: Manager observation (1-D numpy array, no batch dim).
            episode_start: Whether this is the first prediction of a new
                episode for this agent.

        Returns:
            Integer option selection.
        """
        obs_batch = obs[np.newaxis, :]  # add batch dim

        if self._is_recurrent[aid]:
            obs_tensor, _ = self.agents[aid].policy.obs_to_tensor(obs_batch)
            ep_start_tensor = torch.tensor(
                [episode_start], dtype=torch.float32,
                device=self.agents[aid].device,
            )
            # Save pre-forward LSTM states for buffer storage
            self._lstm_states_at_action[aid] = deepcopy(self._lstm_states[aid])
            with torch.no_grad():
                action, _value, _log_prob, new_lstm = (
                    self.agents[aid].policy.forward(
                        obs_tensor, self._lstm_states[aid], ep_start_tensor,
                        deterministic=False,
                    )
                )
            self._lstm_states[aid] = new_lstm
            return int(action.item())
        else:
            action, _ = self.agents[aid].policy.predict(
                obs_batch, deterministic=False
            )
            return int(action[0])

    def _predict_all(
        self,
        obs_dict: Dict[str, np.ndarray],
        episode_starts: Dict[str, bool],
    ) -> Dict[str, int]:
        """Select options for all agents and return action dict.

        Args:
            obs_dict: Per-agent current manager observations.
            episode_starts: Per-agent episode-start flags.

        Returns:
            Dict mapping agent_id → integer option selection.
        """
        return {
            aid: self._predict_single(aid, obs_dict[aid], episode_starts[aid])
            for aid in self._agent_ids
        }

    @staticmethod
    def _count_primitive_steps(m_info: Dict[str, Dict]) -> int:
        """Count primitive steps executed in the last wrapper.step() call.

        Each returning agent's info dict has `option_duration` = number of
        primitive steps its option ran for. Because all agents execute the
        same primitive steps simultaneously, we take the max across returning
        agents rather than summing.

        Args:
            m_info: Info dicts returned by MARLOptionsWrapper.step().

        Returns:
            Number of primitive steps that elapsed.
        """
        if not m_info:
            return 0
        return max(info["option_duration"] for info in m_info.values())

    def _store_transition(
        self,
        aid: str,
        obs: np.ndarray,
        action: int,
        reward: float,
        episode_start: bool,
        m_info_entry: Dict,
    ) -> None:
        """Store one manager-level transition in an agent's rollout buffer.

        Skips the transition if `manager_transition_trainable` is False (i.e.
        the option was clipped at an episode boundary).

        Args:
            aid: Agent ID.
            obs: Manager observation at the time the option was selected.
            action: Option ID that was selected.
            reward: Accumulated discounted reward over the option.
            episode_start: True if this transition starts a new episode.
            m_info_entry: Info dict for this agent from wrapper.step().
        """
        if not m_info_entry.get("manager_transition_trainable", True):
            return

        policy = self.agents[aid].policy
        obs_tensor, _ = policy.obs_to_tensor(obs)
        action_array = np.array([action])
        action_tensor = torch.tensor(
            action_array, dtype=torch.long, device=policy.device
        )

        if self._is_recurrent[aid]:
            ep_start_tensor = torch.tensor(
                [episode_start], dtype=torch.float32, device=policy.device
            )
            lstm_at_action = self._lstm_states_at_action[aid]

            with torch.no_grad():
                value = policy.predict_values(
                    obs_tensor, lstm_at_action.vf, ep_start_tensor
                )
                _, log_prob, _ = policy.evaluate_actions(
                    obs_tensor, action_tensor, lstm_at_action, ep_start_tensor
                )

            self._buffers[aid].add(
                obs=obs,
                action=action_array,
                reward=np.array([reward]),
                episode_start=np.array([episode_start]),
                value=value,
                log_prob=log_prob,
                lstm_states=lstm_at_action,
            )
        else:
            with torch.no_grad():
                value = policy.predict_values(obs_tensor)
                _, log_prob, _ = policy.evaluate_actions(
                    obs_tensor, action_tensor
                )

            self._buffers[aid].add(
                obs=obs,
                action=action_array,
                reward=np.array([reward]),
                episode_start=np.array([episode_start]),
                value=value,
                log_prob=log_prob,
            )

    def _maybe_update(
        self,
        aid: str,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Flush the rollout buffer and run a PPO update if the buffer is full.

        Args:
            aid: Agent ID.
            next_obs: The manager observation after the completing option
                (used as the GAE bootstrap value).
            done: True if the episode ended on this step.
        """
        buf = self._buffers[aid]
        if not buf.full:
            return

        policy = self.agents[aid].policy
        next_obs_tensor, _ = policy.obs_to_tensor(next_obs)

        if self._is_recurrent[aid]:
            # episode_starts is False because next_obs belongs to the current
            # episode — the LSTM states in self._lstm_states[aid] carry the
            # correct history.  When done=True the bootstrap value is zeroed
            # out by compute_returns_and_advantage() via the dones array, so
            # the exact value does not affect training in that case.
            lstm_states = self._lstm_states[aid]
            assert lstm_states is not None  # guaranteed by _is_recurrent
            ep_start_tensor = torch.tensor(
                [False], dtype=torch.float32, device=policy.device
            )
            with torch.no_grad():
                last_value = policy.predict_values(
                    next_obs_tensor,
                    lstm_states.vf,
                    ep_start_tensor,
                )
        else:
            with torch.no_grad():
                last_value = policy.predict_values(next_obs_tensor)

        buf.compute_returns_and_advantage(
            last_values=last_value,
            dones=np.array([done]),
        )
        self.agents[aid].train()
        buf.reset()

    def _run_eval_episodes(self) -> Dict[str, float]:
        """Run n_eval_episodes deterministic episodes and return per-agent mean reward.

        All agents act deterministically (greedy option selection). Does not
        modify rollout buffers or training state.

        Returns:
            Dict mapping agent_id → mean cumulative manager-level reward.
        """
        return run_marl_eval_episodes(
            wrapper=self.wrapper,
            agents=self.agents,
            n_episodes=self.n_eval_episodes,
        )

    def _maybe_eval_and_checkpoint(
        self,
        episodes_completed: int,
        primitive_steps_elapsed: int,
    ) -> None:
        """Run evaluation and save checkpoints if the eval frequency is reached.

        Runs evaluation every `eval_freq_episodes` and saves
        `best_model_{aid}.zip` if any agent's mean eval reward improves.
        Saves periodic checkpoints every `save_freq_episodes` as
        `{aid}_checkpoint_{steps}.zip`.

        Args:
            episodes_completed: Total completed training episodes so far.
            primitive_steps_elapsed: Total primitive steps elapsed so far.
        """
        should_eval = (episodes_completed % self.eval_freq_episodes == 0)
        should_save_ckpt = (episodes_completed % self.save_freq_episodes == 0)

        if should_eval:
            mean_rewards = self._run_eval_episodes()
            for aid, mean_r in mean_rewards.items():
                if mean_r > self._best_mean_reward[aid]:
                    self._best_mean_reward[aid] = mean_r
                    best_path = self.checkpoint_dir / f"best_model_{aid}"
                    self.agents[aid].save(str(best_path))
                    tag = " (new best)"
                else:
                    tag = ""

                if self.verbose >= 1:
                    print(
                        f"  eval {aid}: mean_reward={mean_r:.3f} "
                        f"(best={self._best_mean_reward[aid]:.3f}){tag}"
                    )

        if should_save_ckpt:
            for aid in self._agent_ids:
                ckpt_path = (
                    self.checkpoint_dir
                    / f"{aid}_checkpoint_{primitive_steps_elapsed}"
                )
                self.agents[aid].save(str(ckpt_path))


# --------------------------------------------------------------------------- #
# Public training entry point (callable from runner or CLI)
# --------------------------------------------------------------------------- #

def run_marl_training(
    marl_config_path: "str | Path",
    results_dir: "str | Path",
    run_name: str,
    seed: int = 0,
    overrides: Optional[List[str]] = None,
    agent_init_params_path: Optional["str | Path"] = None,
    skip_if_exists: bool = False,
) -> None:
    """Load a MARL config, build all components, and run MARLTrainer.

    This is the canonical entry point for a single training run. The CLI
    ``_main()`` and the experiment runner ``run_marl_experiment_set.py`` both
    call this function — the CLI via argparse, the runner via direct import.

    The resolved config (with absolute option_library paths, seed injected,
    and all overrides applied) is written to
    ``<results_dir>/<run_name>_<timestamp>/marl_full_agents_env_config.yaml``
    before training begins so that the eval script can reconstruct the
    environment without the original template YAML.

    Args:
        marl_config_path: Path to the MARL template YAML file.
        results_dir: Base directory where the run folder is created.
        run_name: Base name for the run folder. A timestamp suffix is appended
            unless one is already present.
        seed: Random seed injected into ``config["training"]["seed"]``.
        overrides: Optional list of dot-path ``"key.path=value"`` overrides
            applied to the config before training. Coercion follows the same
            rules as ``_apply_overrides``.
        agent_init_params_path: Optional path to ``agent_init_params.json``.
            Maps agent_id → ``{best_params_path, source_run_name}``. Agents
            whose ``best_params_path`` is non-empty are initialised with the
            referenced tuning results; others use training-config defaults.
        skip_if_exists: If True and all ``final_model_{aid}.zip`` files
            already exist for an existing run with this base name (or exact
            timestamped name), return without re-running training.
    """
    import json

    import yaml

    from abx_amr_simulator.utils.marl_factories import (
        build_marl_env_from_config,
        build_marl_managers_from_config,
        build_marl_wrapper_from_config,
        load_marl_config,
    )

    # 1. Load config.
    config = load_marl_config(marl_config_path)

    # 2. Apply dot-path overrides.
    if overrides:
        _apply_overrides(config, overrides)

    # 3. Inject seed.
    config["training"]["seed"] = seed

    # 4. Resolve run paths.
    timestamped_run_name = _ensure_timestamped_run_name(run_name=run_name)
    run_dir = Path(results_dir) / timestamped_run_name
    checkpoint_dir = run_dir / "checkpoints"

    # 5. Skip check (needs config to know agent IDs).
    if skip_if_exists:
        agent_ids = [
            str(e["agent_id"]) for e in config["environment"]["agents"]
        ]

        if timestamped_run_name == run_name:
            all_exist = all(
                (checkpoint_dir / f"final_model_{aid}.zip").exists()
                for aid in agent_ids
            )
            if all_exist:
                print(
                    f"[skip] All final models already exist in {checkpoint_dir}. "
                    "Skipping training."
                )
                return
        else:
            existing_run_dir = _find_existing_timestamped_run_dir(
                results_dir=results_dir,
                run_name_prefix=run_name,
                agent_ids=agent_ids,
            )
            if existing_run_dir is not None:
                print(
                    f"[skip] All final models already exist in "
                    f"{existing_run_dir / 'checkpoints'}. Skipping training."
                )
                return

    if timestamped_run_name != run_name:
        print(f"[run] Resolved MARL run folder: {timestamped_run_name}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 6. Resolve relative paths to absolute so the saved config is self-contained.
    # The saved config is loaded from the run folder, so any relative path that
    # was valid relative to the original marl_configs dir must be made absolute
    # before writing.
    config_dir = Path(config["_config_dir"])

    def _resolve_inline_plugin_loader_modules(obj: object) -> None:
        """Recursively resolve relative plugin.loader_module paths.

        Any inline config block may carry ``plugin.loader_module``. Since the
        resolved config is saved then reloaded from a run directory, relative
        plugin paths must be canonicalized to absolute paths first.
        """
        if isinstance(obj, dict):
            plugin_cfg = obj.get("plugin")
            if isinstance(plugin_cfg, dict):
                loader_module = plugin_cfg.get("loader_module")
                if isinstance(loader_module, str) and loader_module:
                    loader_path = Path(loader_module)
                    if not loader_path.is_absolute():
                        plugin_cfg["loader_module"] = str((config_dir / loader_path).resolve())

            for value in obj.values():
                _resolve_inline_plugin_loader_modules(value)
            return

        if isinstance(obj, list):
            for value in obj:
                _resolve_inline_plugin_loader_modules(value)

    # Resolve plugin loader paths across the whole environment block.
    _resolve_inline_plugin_loader_modules(config.get("environment", {}))

    for entry in config["environment"]["agents"]:
        # Resolve option_library path.
        lib_value = entry.get("option_library")
        if lib_value is not None and not Path(lib_value).is_absolute():
            entry["option_library"] = str((config_dir / lib_value).resolve())

        # Resolve patient_generator path when it is a filename reference (string).
        pg_value = entry.get("patient_generator")
        if isinstance(pg_value, str) and not Path(pg_value).is_absolute():
            entry["patient_generator"] = str((config_dir / pg_value).resolve())

        rc_value = entry.get("reward_calculator")
        if isinstance(rc_value, str) and not Path(rc_value).is_absolute():
            entry["reward_calculator"] = str((config_dir / rc_value).resolve())

    # 7. Write resolved config (strip internal keys that start with '_').
    config_save_path = run_dir / "marl_full_agents_env_config.yaml"
    config_to_save = {k: v for k, v in config.items() if not k.startswith("_")}
    with open(config_save_path, "w") as f:
        yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)

    # 8. Load per-agent init params if provided.
    agent_hyperparams: Optional[Dict[str, Dict]] = None
    if agent_init_params_path:
        with open(agent_init_params_path) as f:
            agent_init_data = json.load(f)
        agent_hyperparams = {}
        for aid, entry in agent_init_data.items():
            best_params_path = entry.get("best_params_path") or ""
            if best_params_path:
                with open(best_params_path) as f:
                    agent_hyperparams[aid] = json.load(f)

    # 9. Build and run training.
    # Load from the saved YAML so _config_dir is the run folder and all
    # option_library paths are absolute and self-consistent.
    saved_config = load_marl_config(config_save_path)
    env = build_marl_env_from_config(saved_config)
    wrapper = build_marl_wrapper_from_config(saved_config, env)
    agents = build_marl_managers_from_config(
        saved_config, wrapper, agent_hyperparams=agent_hyperparams
    )

    # Write resolved config that includes the actual per-agent hyperparameters.
    # The config saved above (step 7) was written before tuned hyperparameters
    # were loaded, so it reflects defaults rather than the values used during
    # training.  This second file records the full provenance.
    resolved_config_path = run_dir / "marl_resolved_config.yaml"
    resolved_config = dict(saved_config)
    if agent_hyperparams:
        resolved_config["_resolved_agent_hyperparams"] = agent_hyperparams
    # Also record the actual PPO kwargs each agent was constructed with.
    resolved_ppo_kwargs: Dict[str, Dict] = {}
    for aid, agent in agents.items():
        resolved_ppo_kwargs[aid] = {
            "learning_rate": float(agent.learning_rate),
            "n_steps": int(agent.n_steps),
            "batch_size": int(agent.batch_size),
            "n_epochs": int(agent.n_epochs),
            "gamma": float(agent.gamma),
            "gae_lambda": float(agent.gae_lambda),
            "clip_range": float(agent.clip_range(1.0))
            if callable(agent.clip_range)
            else float(agent.clip_range),
            "ent_coef": float(agent.ent_coef),
            "vf_coef": float(agent.vf_coef),
            "max_grad_norm": float(agent.max_grad_norm),
        }
    resolved_config["_resolved_ppo_kwargs"] = resolved_ppo_kwargs
    with open(resolved_config_path, "w") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

    training_cfg = saved_config.get("training", {})
    trainer_n_steps = {
        aid: int(agent.rollout_buffer.buffer_size)
        for aid, agent in agents.items()
    }
    trainer = MARLTrainer(
        wrapper=wrapper,
        agents=agents,
        n_steps=trainer_n_steps,
        total_primitive_steps=int(training_cfg["total_primitive_steps"]),
        checkpoint_dir=checkpoint_dir,
        eval_freq_episodes=int(training_cfg.get("eval_freq_episodes", 10)),
        save_freq_episodes=int(
            training_cfg.get(
                "save_freq_episodes",
                training_cfg.get("eval_freq_episodes", 10),
            )
        ),
        n_eval_episodes=int(training_cfg.get("n_eval_episodes", 5)),
        verbose=1,
    )
    trainer.train()


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _main() -> None:
    """CLI entry point for ``python -m abx_amr_simulator.training.train_marl``.

    Thin wrapper around ``run_marl_training`` that parses CLI arguments and
    delegates. See ``run_marl_training`` for full parameter documentation.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a MARL HRL PPO experiment from a config YAML."
    )
    parser.add_argument(
        "--marl-config",
        required=True,
        help="Path to the MARL template YAML file.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Base directory where the run folder is created.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the run folder (created under --results-dir).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed injected into config['training']['seed'].",
    )
    parser.add_argument(
        "-p",
        action="append",
        dest="overrides",
        default=[],
        metavar="KEY.PATH=VALUE",
        help=(
            "Dot-path config override. May be repeated. "
            "Example: -p environment.agents.0.patient_generator.personalized_auroc=0.7"
        ),
    )
    parser.add_argument(
        "--agent-init-params",
        default=None,
        help=(
            "Optional path to agent_init_params.json. Maps agent_id → "
            "{best_params_path, source_run_name}. Agents whose "
            "best_params_path is non-empty are initialised with the "
            "referenced tuning results; others use training-config defaults."
        ),
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        default=False,
        help=(
            "Skip training if all final_model_{aid}.zip files already exist "
            "in the checkpoints directory."
        ),
    )

    args = parser.parse_args()
    run_marl_training(
        marl_config_path=args.marl_config,
        results_dir=args.results_dir,
        run_name=args.run_name,
        seed=args.seed,
        overrides=args.overrides or None,
        agent_init_params_path=args.agent_init_params,
        skip_if_exists=args.skip_if_exists,
    )


if __name__ == "__main__":
    _main()
