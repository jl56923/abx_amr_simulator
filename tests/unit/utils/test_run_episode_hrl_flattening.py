"""Integration tests for run_episode_and_get_trajectory with HRL support."""

import numpy as np
from unittest.mock import Mock, patch
from gymnasium import spaces

from abx_amr_simulator.utils.metrics import run_episode_and_get_trajectory


class _SimpleDeterministicModel:
    """Minimal SB3-like model for testing."""
    def __init__(self, action):
        self._action = action

    def predict(self, obs, deterministic=True):
        return self._action, None


def _create_nohrl_mock_env():
    """Create a mock non-HRL environment."""
    env = Mock()
    env.unwrapped = Mock()
    env.unwrapped.max_time_steps = 5
    env.action_space = spaces.Discrete(2)
    
    # Non-HRL trajectory: 5 steps
    env.reset = Mock(return_value=(np.array([1.0]), {}))
    
    step_sequence = [
        (np.array([2.0]), 1.0, False, False, {"step": 0}),
        (np.array([3.0]), 1.0, False, False, {"step": 1}),
        (np.array([4.0]), 1.0, False, False, {"step": 2}),
        (np.array([5.0]), 1.0, False, False, {"step": 3}),
        (np.array([6.0]), 1.0, True, False, {"step": 4}),
    ]
    env.step = Mock(side_effect=step_sequence)
    
    return env


def _create_hrl_mock_env():
    """Create a mock HRL environment (OptionsWrapper)."""
    env = Mock()
    env.unwrapped = Mock()
    env.unwrapped.max_time_steps = 500
    env.action_space = spaces.Discrete(4)  # 4 options
    
    # Initial reset
    env.reset = Mock(return_value=(np.array([1.0]), {}))
    
    # HRL trajectory: 2 manager steps
    # Step 1: Option 0 executes for 3 primitive steps, then terminates
    step_sequence = [
        # Manager step 1 - Option 0 (3 primitive steps)
        (
            np.array([2.0]),  # obs after option
            10.0,  # aggregated reward for option
            True,  # terminated
            False,  # truncated
            {
                "macro_action": 0,
                "macro_action_duration": 3,
                "primitive_actions": [1, 0, 1],
                "primitive_infos": [
                    {"step": 0, "reward": 3.0},
                    {"step": 1, "reward": 3.0},
                    {"step": 2, "reward": 4.0},
                ],
            }
        ),
    ]
    env.step = Mock(side_effect=step_sequence)
    
    return env


def test_run_episode_non_hrl_trajectory():
    """Verify run_episode_and_get_trajectory works with non-HRL trajectories."""
    env = _create_nohrl_mock_env()
    model = _SimpleDeterministicModel(action=0)
    
    trajectory = run_episode_and_get_trajectory(model=model, env=env, deterministic=True)
    
    # Non-HRL trajectory should NOT be flattened
    assert len(trajectory["actions"]) == 5
    assert len(trajectory["rewards"]) == 5
    assert len(trajectory["infos"]) == 6  # 5 steps + initial
    assert len(trajectory["obs"]) == 6     # 5 steps + initial
    
    # Should NOT have macro_action_boundaries for non-HRL
    assert "macro_action_boundaries" not in trajectory


def test_run_episode_hrl_trajectory_flattening():
    """Verify run_episode_and_get_trajectory flattens HRL trajectories."""
    env = _create_hrl_mock_env()
    model = _SimpleDeterministicModel(action=0)
    
    trajectory = run_episode_and_get_trajectory(model=model, env=env, deterministic=True)
    
    # HRL trajectory should be flattened to 3 primitive steps
    assert len(trajectory["actions"]) == 3, f"Expected 3 actions, got {len(trajectory['actions'])}"
    assert len(trajectory["rewards"]) == 3, f"Expected 3 rewards, got {len(trajectory['rewards'])}"
    assert len(trajectory["infos"]) == 3, f"Expected 3 infos, got {len(trajectory['infos'])}"
    
    # Observations should be: initial + 3 primitive steps = 4
    assert len(trajectory["obs"]) == 4, f"Expected 4 obs (initial + 3 steps), got {len(trajectory['obs'])}"
    
    # Should have macro_action_boundaries for HRL
    assert "macro_action_boundaries" in trajectory
    assert trajectory["macro_action_boundaries"] == [0, 3]
    
    # Verify actions are flattened correctly
    assert trajectory["actions"] == [1, 0, 1]
    
    # Verify rewards are distributed correctly (10.0 / 3 steps)
    expected_reward = 10.0 / 3.0
    for reward in trajectory["rewards"]:
        assert abs(reward - expected_reward) < 1e-6


def test_run_episode_hrl_primitive_infos_preserved():
    """Verify primitive infos are preserved after HRL flattening."""
    env = _create_hrl_mock_env()
    model = _SimpleDeterministicModel(action=0)
    
    trajectory = run_episode_and_get_trajectory(model=model, env=env, deterministic=True)
    
    # Check that primitive infos are preserved
    assert len(trajectory["infos"]) == 3
    
    assert trajectory["infos"][0] == {"step": 0, "reward": 3.0}
    assert trajectory["infos"][1] == {"step": 1, "reward": 3.0}
    assert trajectory["infos"][2] == {"step": 2, "reward": 4.0}


def test_run_episode_hrl_no_flattening():
    """Verify HRL trajectories can be kept at manager-level (for HRL diagnostics)."""
    env = _create_hrl_mock_env()
    model = _SimpleDeterministicModel(action=0)
    
    # Request NO flattening (manager-level trajectory)
    trajectory = run_episode_and_get_trajectory(model=model, env=env, deterministic=True, flatten_hrl=False)
    
    # Manager-level trajectory should have 1 step (one option execution)
    assert len(trajectory["actions"]) == 1, f"Expected 1 manager action, got {len(trajectory['actions'])}"
    assert len(trajectory["rewards"]) == 1, f"Expected 1 manager reward, got {len(trajectory['rewards'])}"
    assert len(trajectory["infos"]) == 2, f"Expected 2 infos (initial + 1 step), got {len(trajectory['infos'])}"
    
    # Observations should be: initial + 1 manager step = 2
    assert len(trajectory["obs"]) == 2, f"Expected 2 obs (initial + 1 manager step), got {len(trajectory['obs'])}"
    
    # Should NOT have macro_action_boundaries when not flattened
    assert "macro_action_boundaries" not in trajectory
    
    # Manager info should contain primitive data
    manager_info = trajectory["infos"][1]
    assert "macro_action" in manager_info
    assert "primitive_actions" in manager_info
    assert "primitive_infos" in manager_info
    assert manager_info["primitive_actions"] == [1, 0, 1]
    assert len(manager_info["primitive_infos"]) == 3
