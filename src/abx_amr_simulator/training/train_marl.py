"""Multi-agent HRL PPO training loop (Option B: custom rollout, SB3 policy objects).

`MARLTrainer` drives `MARLOptionsWrapper` directly with one SB3 `PPO` object per agent.
It never calls `PPO.learn()`. Instead it owns the complete rollout-collect-update cycle:

  1. Call `wrapper.step(pending_selections)` to advance the MARL environment.
  2. For each completing agent, store its transition in its own `RolloutBuffer`.
  3. When a buffer fills, compute GAE and call `agent.train()`.
  4. Handle episode boundaries, evaluation, and checkpointing.

Only HRL PPO (not RecurrentPPO) is supported. See
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
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer

from abx_amr_simulator.callbacks.marl_callbacks import run_marl_eval_episodes
from abx_amr_simulator.hrl.marl_wrapper import MARLOptionsWrapper


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
        n_eval_episodes: Number of deterministic eval episodes per evaluation.
        verbose: 0 = silent, 1 = progress summaries.
    """

    def __init__(
        self,
        wrapper: MARLOptionsWrapper,
        agents: Dict[str, PPO],
        n_steps: int,
        total_primitive_steps: int,
        checkpoint_dir: Path,
        eval_freq_episodes: int = 10,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ) -> None:
        """Initialise MARLTrainer.

        Args:
            wrapper: Pre-instantiated MARLOptionsWrapper (env + option libraries).
            agents: Dict mapping agent_id → PPO object. Must contain one entry
                per agent in wrapper.base_env.possible_agents. Each PPO object
                must have been constructed with the matching observation/action
                space (e.g. via make_ppo_for_agent()).
            n_steps: Rollout buffer capacity per agent in manager steps. Should
                match the n_steps used when constructing each PPO object.
            total_primitive_steps: Training budget measured in primitive env
                steps (not manager steps). Training stops when this is reached.
            checkpoint_dir: Directory for saving model checkpoints. Created if
                it does not exist.
            eval_freq_episodes: Run evaluation every this many completed
                training episodes.
            n_eval_episodes: Number of deterministic episodes per evaluation.
            verbose: 0 = silent; 1 = print episode/eval summaries.

        Raises:
            ValueError: If any agent in wrapper is missing from agents, or if
                n_steps does not match an agent's rollout buffer size.
        """
        self.wrapper = wrapper
        self.agents = agents
        self.n_steps = n_steps
        self.total_primitive_steps = total_primitive_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.eval_freq_episodes = eval_freq_episodes
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose

        self._agent_ids: List[str] = list(wrapper.base_env.possible_agents)

        # Validate that every agent has a PPO entry
        for aid in self._agent_ids:
            if aid not in agents:
                raise ValueError(
                    f"No PPO agent provided for '{aid}'. "
                    f"agents keys: {sorted(agents.keys())}"
                )

        # Validate and extract rollout buffers from the PPO objects.
        # The PPO objects already own RolloutBuffers — we use them directly.
        self._buffers: Dict[str, RolloutBuffer] = {}
        for aid in self._agent_ids:
            buf = agents[aid].rollout_buffer
            if buf.buffer_size != n_steps:
                raise ValueError(
                    f"Agent '{aid}': PPO rollout buffer size {buf.buffer_size} "
                    f"does not match n_steps={n_steps}. Construct PPO with "
                    f"n_steps={n_steps}."
                )
            self._buffers[aid] = buf

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

                pending_selections = self._predict_all(
                    last_obs, last_episode_start
                )
            else:
                # Only completing agents need new option selections
                for aid, next_obs in m_obs.items():
                    pending_selections[aid] = int(
                        self.agents[aid].policy.predict(
                            next_obs[np.newaxis, :], deterministic=False
                        )[0][0]
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

    def _predict_all(
        self,
        obs_dict: Dict[str, np.ndarray],
        episode_starts: Dict[str, bool],
    ) -> Dict[str, int]:
        """Call policy.predict for every agent and return action dict.

        Args:
            obs_dict: Per-agent current manager observations.
            episode_starts: Per-agent episode-start flags (unused for PPO but
                kept for interface consistency).

        Returns:
            Dict mapping agent_id → integer option selection.
        """
        selections: Dict[str, int] = {}
        for aid in self._agent_ids:
            obs = obs_dict[aid][np.newaxis, :]   # add batch dim
            action, _ = self.agents[aid].policy.predict(obs, deterministic=False)
            selections[aid] = int(action[0])
        return selections

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

        with torch.no_grad():
            value = policy.predict_values(obs_tensor)
            _, log_prob, _ = policy.evaluate_actions(obs_tensor, action_tensor)

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

        Saves `best_model_{aid}.zip` if any agent's mean eval reward improves.
        Saves periodic checkpoints as `{aid}_checkpoint_{steps}.zip`.

        Args:
            episodes_completed: Total completed training episodes so far.
            primitive_steps_elapsed: Total primitive steps elapsed so far.
        """
        if episodes_completed % self.eval_freq_episodes != 0:
            return

        mean_rewards = self._run_eval_episodes()

        for aid, mean_r in mean_rewards.items():
            # Save best model
            if mean_r > self._best_mean_reward[aid]:
                self._best_mean_reward[aid] = mean_r
                best_path = self.checkpoint_dir / f"best_model_{aid}"
                self.agents[aid].save(str(best_path))
                tag = " (new best)"
            else:
                tag = ""

            # Save periodic checkpoint
            ckpt_path = (
                self.checkpoint_dir
                / f"{aid}_checkpoint_{primitive_steps_elapsed}"
            )
            self.agents[aid].save(str(ckpt_path))

            if self.verbose >= 1:
                print(
                    f"  eval {aid}: mean_reward={mean_r:.3f} "
                    f"(best={self._best_mean_reward[aid]:.3f}){tag}"
                )
