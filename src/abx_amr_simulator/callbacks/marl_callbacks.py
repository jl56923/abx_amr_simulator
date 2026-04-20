"""Evaluation utilities for multi-agent HRL training.

Contains `run_marl_eval_episodes`, a standalone function used by `MARLTrainer`
to periodically evaluate trained policies. Extracted here so it can be imported
and tested independently of the full trainer.

Both HRL PPO and HRL RecurrentPPO (HRL_RPPO) agents are supported.  For
recurrent agents, LSTM states are threaded across steps within each eval
episode and reset to None at the start of each new episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from abx_amr_simulator.hrl.marl_wrapper import MARLOptionsWrapper


def run_marl_eval_episodes(
    wrapper: MARLOptionsWrapper,
    agents: Dict[str, Any],
    n_episodes: int,
) -> Dict[str, float]:
    """Run deterministic evaluation episodes and return per-agent mean reward.

    All agents act greedily (deterministic=True). Rollout buffers and training
    state are not modified. The wrapper is reset at the start of each episode.

    For recurrent agents (RecurrentPPO / RecurrentPPO_Masked), LSTM hidden
    states are tracked across steps within each episode via the ``state``
    and ``episode_start`` kwargs of ``policy.predict()``.  For non-recurrent
    agents, ``predict()`` accepts and ignores these kwargs (returning
    ``state=None``), so the same code path handles both agent types.

    Args:
        wrapper: The MARLOptionsWrapper to evaluate in.
        agents: Dict mapping agent_id → PPO or RecurrentPPO agent (policy
            used for prediction only).
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

        # Per-agent LSTM states (None for non-recurrent agents, and also
        # None at episode start for recurrent agents — predict() will
        # auto-initialize to zeros).
        lstm_states: Dict[str, Any] = {aid: None for aid in agent_ids}

        # Select first option for every agent
        pending: Dict[str, int] = {}
        for aid in agent_ids:
            obs = last_obs[aid][np.newaxis, :]
            action, lstm_states[aid] = agents[aid].policy.predict(
                obs, state=lstm_states[aid],
                episode_start=np.array([True]),
                deterministic=True,
            )
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
                    action, lstm_states[aid] = agents[aid].policy.predict(
                        obs, state=lstm_states[aid],
                        episode_start=np.array([False]),
                        deterministic=True,
                    )
                    pending[aid] = int(action[0])

        for aid in agent_ids:
            cumulative_rewards[aid].append(episode_rewards[aid])

    return {
        aid: float(np.mean(cumulative_rewards[aid]))
        for aid in agent_ids
    }
