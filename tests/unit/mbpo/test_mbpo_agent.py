"""Unit tests for MBPOAgent."""

from typing import Any, Dict, Tuple
from unittest.mock import MagicMock

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete

from abx_amr_simulator.mbpo.mbpo_agent import MBPOAgent


class DummyMBPOEnv(gym.Env):
    """Simple deterministic environment for MBPOAgent tests."""

    metadata = {"render_modes": []}

    def __init__(self, episode_length: int = 3):
        super().__init__()
        self.observation_space = Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        self.action_space = MultiDiscrete(nvec=[3])
        self.episode_length = int(episode_length)
        self.current_step = 0
        self.state = np.zeros(shape=4, dtype=np.float32)

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.zeros(shape=4, dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        del action
        self.current_step += 1
        self.state = self.state + 1.0
        reward = 1.0
        terminated = self.current_step >= self.episode_length
        truncated = False
        return self.state.copy(), reward, terminated, truncated, {}


def _make_config() -> Dict[str, Any]:
    return {
        "ppo": {
            "learning_rate": 1.0e-3,
            "n_steps": 2,
            "batch_size": 2,
            "n_epochs": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {},
            "verbose": 0,
        },
        "mbpo": {
            "rollout_length": 3,
            "num_synthetic_per_real": 2,
            "model_train_freq": 1,
            "model_warmup_episodes": 0,
            "model_train_epochs": 2,
            "model_batch_size": 4,
            "eval_freq": 0,
        },
        "dynamics_model": {
            "hidden_dims": [16, 16],
            "learning_rate": 1.0e-3,
            "device": "cpu",
        },
    }


class TestMBPOAgentCore:
    """Test MBPOAgent core behaviors."""

    def test_collect_real_episode_and_add_data(self):
        env = DummyMBPOEnv(episode_length=3)
        agent = MBPOAgent(env=env, config=_make_config())

        trajectory = agent._collect_real_episode()
        assert len(trajectory["observations"]) == 4
        assert len(trajectory["actions"]) == 3
        assert len(trajectory["rewards"]) == 3
        assert len(trajectory["dones"]) == 3

        agent._add_to_real_data(trajectory=trajectory)
        assert len(agent.real_data) == 3
        assert "obs" in agent.real_data[0]
        assert "action" in agent.real_data[0]
        assert "reward" in agent.real_data[0]
        assert "next_obs" in agent.real_data[0]
        assert "done" in agent.real_data[0]

    def test_train_dynamics_model_sets_flag(self):
        env = DummyMBPOEnv(episode_length=2)
        agent = MBPOAgent(env=env, config=_make_config())

        trajectory = agent._collect_real_episode()
        agent._add_to_real_data(trajectory=trajectory)

        metrics = agent._train_dynamics_model()
        assert agent.model_is_trained
        assert "total_loss" in metrics
        assert "obs_loss" in metrics
        assert "reward_loss" in metrics

    def test_generate_synthetic_rollouts(self):
        env = DummyMBPOEnv(episode_length=2)
        agent = MBPOAgent(env=env, config=_make_config())

        trajectory = agent._collect_real_episode()
        agent._add_to_real_data(trajectory=trajectory)
        agent._train_dynamics_model()

        synthetic = agent._generate_synthetic_rollouts(episode_idx=0)
        assert len(synthetic) == 2
        for traj in synthetic:
            assert len(traj["observations"]) == 4  # rollout_length + 1
            assert len(traj["actions"]) == 3
            assert len(traj["rewards"]) == 3

    def test_train_policy_on_synthetic_calls_learn(self):
        env = DummyMBPOEnv(episode_length=2)
        agent = MBPOAgent(env=env, config=_make_config())

        synthetic_trajectories = [
            {
                "observations": [np.zeros(shape=4), np.ones(shape=4)],
                "actions": [np.array(object=[0])],
                "rewards": [1.0],
            }
        ]

        agent.policy.learn = MagicMock()
        agent.policy.set_env = MagicMock()
        agent.policy.get_env = MagicMock(return_value=env)

        agent._train_policy_on_synthetic(synthetic_trajectories=synthetic_trajectories)

        agent.policy.learn.assert_called_once()
        agent.policy.set_env.assert_called()

    def test_train_policy_on_real_calls_learn(self):
        env = DummyMBPOEnv(episode_length=2)
        agent = MBPOAgent(env=env, config=_make_config())

        real_trajectories = [
            {
                "observations": [np.zeros(shape=4), np.ones(shape=4)],
                "actions": [np.array(object=[0])],
                "rewards": [1.0],
            }
        ]

        agent.policy.learn = MagicMock()
        agent.policy.set_env = MagicMock()
        agent.policy.get_env = MagicMock(return_value=env)

        agent._train_policy_on_real(
            real_trajectories=real_trajectories,
            updates_multiplier=1,
        )

        agent.policy.learn.assert_called_once()
        agent.policy.set_env.assert_called()

    def test_rollout_and_synthetic_schedules(self):
        env = DummyMBPOEnv(episode_length=2)
        config = _make_config()
        config["mbpo"]["rollout_length_start"] = 1
        config["mbpo"]["rollout_length_max"] = 5
        config["mbpo"]["rollout_length_schedule_episodes"] = 10
        config["mbpo"]["num_synthetic_per_real_start"] = 2
        config["mbpo"]["num_synthetic_per_real_max"] = 6
        config["mbpo"]["num_synthetic_schedule_episodes"] = 10

        agent = MBPOAgent(env=env, config=config)

        assert agent._get_scheduled_rollout_length(episode_idx=0) == 1
        assert agent._get_scheduled_rollout_length(episode_idx=10) == 5
        assert agent._get_scheduled_num_synthetic(episode_idx=0) == 2
        assert agent._get_scheduled_num_synthetic(episode_idx=10) == 6

    def test_check_sufficient_data_blocks_below_min_good_return(self):
        env = DummyMBPOEnv(episode_length=1)
        config = _make_config()
        config["mbpo"]["min_good_transitions_for_model"] = 1
        config["mbpo"]["min_good_episodes_for_model"] = 1
        config["mbpo"]["min_good_return_for_model"] = -200

        agent = MBPOAgent(env=env, config=config)
        agent.episode_returns = [-300.0, -250.0]
        agent.episode_start_indices = [0, 1]
        agent.real_data = [
            {
                "obs": np.zeros(shape=4),
                "action": np.array(object=[0]),
                "reward": -1.0,
                "next_obs": np.zeros(shape=4),
                "done": True,
            },
            {
                "obs": np.zeros(shape=4),
                "action": np.array(object=[0]),
                "reward": -1.0,
                "next_obs": np.zeros(shape=4),
                "done": True,
            },
        ]

        sufficient, reason = agent._check_sufficient_data_for_model(
            filtered_data=agent.real_data,
            num_episodes_kept=2,
        )
        assert not sufficient
        assert "min_good_return" in reason

    def test_check_sufficient_data_allows_above_min_good_return(self):
        env = DummyMBPOEnv(episode_length=1)
        config = _make_config()
        config["mbpo"]["min_good_transitions_for_model"] = 1
        config["mbpo"]["min_good_episodes_for_model"] = 1
        config["mbpo"]["min_good_return_for_model"] = -200

        agent = MBPOAgent(env=env, config=config)
        agent.episode_returns = [-150.0]
        agent.episode_start_indices = [0]
        agent.real_data = [
            {
                "obs": np.zeros(shape=4),
                "action": np.array(object=[0]),
                "reward": -1.0,
                "next_obs": np.zeros(shape=4),
                "done": True,
            }
        ]

        sufficient, reason = agent._check_sufficient_data_for_model(
            filtered_data=agent.real_data,
            num_episodes_kept=1,
        )
        assert sufficient
        assert reason == ""
