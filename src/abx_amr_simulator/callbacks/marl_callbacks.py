"""Evaluation utilities for multi-agent HRL PPO training.

Contains `run_marl_eval_episodes`, a standalone function used by `MARLTrainer`
to periodically evaluate trained policies. Extracted here so it can be imported
and tested independently of the full trainer.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO

from abx_amr_simulator.hrl.marl_wrapper import MARLOptionsWrapper


def run_marl_eval_episodes(
    wrapper: MARLOptionsWrapper,
    agents: Dict[str, PPO],
    n_episodes: int,
) -> Dict[str, float]:
    """Run deterministic evaluation episodes and return per-agent mean reward.

    All agents act greedily (deterministic=True). Rollout buffers and training
    state are not modified. The wrapper is reset at the start of each episode.

    Args:
        wrapper: The MARLOptionsWrapper to evaluate in.
        agents: Dict mapping agent_id → PPO (policy used for prediction only).
        n_episodes: Number of complete episodes to run.

    Returns:
        Dict mapping agent_id → mean cumulative manager-level reward across
        all evaluation episodes.
    """
    agent_ids: List[str] = list(wrapper.base_env.possible_agents)

    cumulative_rewards: Dict[str, List[float]] = {
        aid: [] for aid in agent_ids
    }

    for _ in range(n_episodes):
        episode_rewards: Dict[str, float] = {aid: 0.0 for aid in agent_ids}

        obs_dict, _ = wrapper.reset()
        last_obs = dict(obs_dict)

        # Select first option for every agent
        pending: Dict[str, int] = {}
        for aid in agent_ids:
            obs = last_obs[aid][np.newaxis, :]
            action, _ = agents[aid].policy.predict(obs, deterministic=True)
            pending[aid] = int(action[0])

        episode_done = False
        while not episode_done:
            m_obs, m_rew, m_term, m_trunc, _ = wrapper.step(pending)

            for aid, rew in m_rew.items():
                episode_rewards[aid] += rew
                last_obs[aid] = m_obs[aid]

            episode_done = any(m_term.values()) or any(m_trunc.values())

            if not episode_done:
                # Only completing agents need a new option selection
                pending = {}
                for aid in m_obs:
                    obs = m_obs[aid][np.newaxis, :]
                    action, _ = agents[aid].policy.predict(obs, deterministic=True)
                    pending[aid] = int(action[0])

        for aid in agent_ids:
            cumulative_rewards[aid].append(episode_rewards[aid])

    return {
        aid: float(np.mean(cumulative_rewards[aid]))
        for aid in agent_ids
    }
