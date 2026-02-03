"""Unit tests for TrajectoryReplayEnv."""

import pytest
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from abx_amr_simulator.mbpo.trajectory_replay_env import TrajectoryReplayEnv


class TestTrajectoryReplayEnvInit:
    """Test TrajectoryReplayEnv initialization."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        obs_space = Box(low=-1, high=1, shape=(5,))
        action_space = Discrete(n=3)
        
        trajectories = [
            {
                'observations': [np.zeros(5), np.ones(5)],
                'actions': [0],
                'rewards': [1.0]
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        assert env.num_trajectories == 1
        assert env.current_traj_idx == 0
        assert env.current_step == 0
    
    def test_init_empty_trajectories_raises_error(self):
        """Test that empty trajectories list raises ValueError."""
        obs_space = Box(low=-1, high=1, shape=(5,))
        action_space = Discrete(n=3)
        
        with pytest.raises(ValueError, match="Must provide at least one trajectory"):
            TrajectoryReplayEnv(
                trajectories=[],
                action_space=action_space,
                observation_space=obs_space
            )
    
    def test_init_validates_trajectory_keys(self):
        """Test that missing required keys raises ValueError."""
        obs_space = Box(low=-1, high=1, shape=(5,))
        action_space = Discrete(n=3)
        
        # Missing 'rewards' key
        trajectories = [
            {
                'observations': [np.zeros(5), np.ones(5)],
                'actions': [0]
            }
        ]
        
        with pytest.raises(ValueError, match="missing required key"):
            TrajectoryReplayEnv(
                trajectories=trajectories,
                action_space=action_space,
                observation_space=obs_space
            )
    
    def test_init_validates_lengths(self):
        """Test that inconsistent lengths raise ValueError."""
        obs_space = Box(low=-1, high=1, shape=(5,))
        action_space = Discrete(n=3)
        
        # Observations length should be len(actions) + 1
        trajectories = [
            {
                'observations': [np.zeros(5)],  # Should be length 2
                'actions': [0],
                'rewards': [1.0]
            }
        ]
        
        with pytest.raises(ValueError, match="observations length should be"):
            TrajectoryReplayEnv(
                trajectories=trajectories,
                action_space=action_space,
                observation_space=obs_space
            )


class TestResetBehavior:
    """Test reset() functionality."""
    
    def test_reset_returns_first_observation(self):
        """Test that reset returns the first observation of a trajectory."""
        obs_space = Box(low=-1, high=1, shape=(3,))
        action_space = Discrete(n=2)
        
        first_obs = np.array([1.0, 2.0, 3.0])
        trajectories = [
            {
                'observations': [first_obs, np.zeros(3)],
                'actions': [0],
                'rewards': [1.0]
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        obs, info = env.reset()
        
        assert np.array_equal(obs, first_obs)
        assert isinstance(info, dict)
    
    def test_reset_advances_to_next_trajectory(self):
        """Test that reset advances to next trajectory after episode ends."""
        obs_space = Box(low=-1, high=1, shape=(2,))
        action_space = Discrete(n=2)
        
        traj1_first_obs = np.array([1.0, 0.0])
        traj2_first_obs = np.array([0.0, 1.0])
        
        trajectories = [
            {
                'observations': [traj1_first_obs, np.zeros(2)],
                'actions': [0],
                'rewards': [1.0]
            },
            {
                'observations': [traj2_first_obs, np.ones(2)],
                'actions': [1],
                'rewards': [2.0]
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        # First reset - should get first trajectory
        obs1, _ = env.reset()
        assert np.array_equal(obs1, traj1_first_obs)
        
        # Step through first trajectory
        env.step(action=0)
        
        # Second reset - should get second trajectory
        obs2, _ = env.reset()
        assert np.array_equal(obs2, traj2_first_obs)
    
    def test_reset_loops_back_to_start(self):
        """Test that reset loops back to first trajectory after reaching end."""
        obs_space = Box(low=-1, high=1, shape=(2,))
        action_space = Discrete(n=2)
        
        first_obs = np.array([1.0, 0.0])
        
        trajectories = [
            {
                'observations': [first_obs, np.zeros(2)],
                'actions': [0],
                'rewards': [1.0]
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        # Reset and complete first episode
        env.reset()
        env.step(action=0)
        
        # Reset again - should loop back to first trajectory
        obs, _ = env.reset()
        assert np.array_equal(obs, first_obs)


class TestStepBehavior:
    """Test step() functionality."""
    
    def test_step_returns_correct_transition(self):
        """Test that step returns the correct next observation and reward."""
        obs_space = Box(low=-1, high=1, shape=(2,))
        action_space = Discrete(n=2)
        
        obs0 = np.array([0.0, 0.0])
        obs1 = np.array([1.0, 1.0])
        reward = 5.0
        
        trajectories = [
            {
                'observations': [obs0, obs1],
                'actions': [1],
                'rewards': [reward]
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        env.reset()
        next_obs, rew, terminated, truncated, info = env.step(action=999)  # Action ignored
        
        assert np.array_equal(next_obs, obs1)
        assert rew == reward
        assert terminated  # Single-step trajectory
        assert not truncated
        assert isinstance(info, dict)
    
    def test_step_ignores_provided_action(self):
        """Test that step ignores the action argument."""
        obs_space = Box(low=-1, high=1, shape=(2,))
        action_space = Discrete(n=2)
        
        obs0 = np.array([0.0, 0.0])
        obs1 = np.array([1.0, 1.0])
        
        trajectories = [
            {
                'observations': [obs0, obs1],
                'actions': [0],  # Recorded action is 0
                'rewards': [1.0]
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        env.reset()
        
        # Provide different action - should still get same result
        next_obs1, rew1, _, _, _ = env.step(action=0)
        
        env.reset()
        next_obs2, rew2, _, _, _ = env.step(action=1)
        
        assert np.array_equal(next_obs1, next_obs2)
        assert rew1 == rew2
    
    def test_step_multi_step_trajectory(self):
        """Test stepping through a multi-step trajectory."""
        obs_space = Box(low=-1, high=1, shape=(2,))
        action_space = Discrete(n=2)
        
        obs_list = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([3.0, 0.0])
        ]
        actions = [0, 1, 0]
        rewards = [1.0, 2.0, 3.0]
        
        trajectories = [
            {
                'observations': obs_list,
                'actions': actions,
                'rewards': rewards
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        env.reset()
        
        # Step 1
        next_obs, rew, terminated, truncated, _ = env.step(action=0)
        assert np.array_equal(next_obs, obs_list[1])
        assert rew == rewards[0]
        assert not terminated
        
        # Step 2
        next_obs, rew, terminated, truncated, _ = env.step(action=0)
        assert np.array_equal(next_obs, obs_list[2])
        assert rew == rewards[1]
        assert not terminated
        
        # Step 3 (final)
        next_obs, rew, terminated, truncated, _ = env.step(action=0)
        assert np.array_equal(next_obs, obs_list[3])
        assert rew == rewards[2]
        assert terminated
    
    def test_step_sets_done_at_trajectory_end(self):
        """Test that terminated flag is set correctly at end of trajectory."""
        obs_space = Box(low=-1, high=1, shape=(2,))
        action_space = Discrete(n=2)
        
        trajectories = [
            {
                'observations': [np.zeros(2), np.ones(2), np.array([2.0, 2.0])],
                'actions': [0, 1],
                'rewards': [1.0, 2.0]
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        env.reset()
        
        # First step - not done
        _, _, terminated1, _, _ = env.step(action=0)
        assert not terminated1
        
        # Second step - done
        _, _, terminated2, _, _ = env.step(action=0)
        assert terminated2


class TestMultipleTrajectories:
    """Test behavior with multiple trajectories."""
    
    def test_cycles_through_multiple_trajectories(self):
        """Test that environment cycles through all trajectories."""
        obs_space = Box(low=-1, high=1, shape=(1,))
        action_space = Discrete(n=2)
        
        trajectories = [
            {
                'observations': [np.array([1.0]), np.array([1.1])],
                'actions': [0],
                'rewards': [1.0]
            },
            {
                'observations': [np.array([2.0]), np.array([2.1])],
                'actions': [0],
                'rewards': [2.0]
            },
            {
                'observations': [np.array([3.0]), np.array([3.1])],
                'actions': [0],
                'rewards': [3.0]
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        # Cycle through all 3 trajectories
        for i in range(3):
            obs, _ = env.reset()
            expected_obs = np.array([float(i + 1)])
            assert np.allclose(obs, expected_obs)
            
            _, rew, _, _, _ = env.step(action=0)
            assert rew == float(i + 1)
        
        # Should loop back to first trajectory
        obs, _ = env.reset()
        assert np.allclose(obs, np.array([1.0]))


class TestMultiDiscreteActions:
    """Test with MultiDiscrete action space."""
    
    def test_works_with_multidiscrete_actions(self):
        """Test that environment works with MultiDiscrete action space."""
        obs_space = Box(low=-1, high=1, shape=(3,))
        action_space = MultiDiscrete(nvec=[3, 3])
        
        trajectories = [
            {
                'observations': [np.zeros(3), np.ones(3)],
                'actions': [np.array([0, 1])],
                'rewards': [1.0]
            }
        ]
        
        env = TrajectoryReplayEnv(
            trajectories=trajectories,
            action_space=action_space,
            observation_space=obs_space
        )
        
        obs, _ = env.reset()
        next_obs, rew, terminated, truncated, _ = env.step(action=np.array([2, 2]))
        
        assert np.array_equal(next_obs, np.ones(3))
        assert rew == 1.0
        assert terminated
