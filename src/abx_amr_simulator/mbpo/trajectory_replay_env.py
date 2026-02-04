"""
Trajectory Replay Environment for MBPO.

A Gymnasium-compatible environment wrapper that replays pre-generated trajectories.
This allows PPO to train on synthetic data via its standard .learn() interface.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Space
import csv
from pathlib import Path


class TrajectoryReplayEnv(gym.Env):
    """
    Environment that replays pre-collected trajectories.
    
    PPO thinks it's interacting with a real environment, but we're feeding it
    pre-recorded transitions. The action argument to step() is ignored.
    """
    
    def __init__(
        self,
        trajectories: List[Dict[str, List]],
        action_space: Space,
        observation_space: Space,
        log_file: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the trajectory replay environment.
        
        Args:
            trajectories: List of trajectory dicts with keys:
                - 'observations': List[np.ndarray] (length T+1, includes final obs)
                - 'actions': List[np.ndarray or int] (length T)
                - 'rewards': List[float] (length T)
            action_space: Gymnasium action space (must match trajectories)
            observation_space: Gymnasium observation space (must match trajectories)
            log_file: Optional path to CSV log file for detailed debugging
            verbose: If True, print detailed step information
        """
        super().__init__()
        
        if len(trajectories) == 0:
            raise ValueError("Must provide at least one trajectory")
        
        self.trajectories = trajectories
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Validate trajectories
        self._validate_trajectories()
        
        # State tracking
        self.current_traj_idx = 0
        self.current_step = 0
        self.num_trajectories = len(trajectories)
        self.total_steps = 0  # Total steps across all resets
        self.total_resets = 0  # Total number of resets
        self.episode_count = 0  # Episodes completed
        
        # Logging setup
        self.log_file = log_file
        self.verbose = verbose
        self.log_writer = None
        self.log_handle = None
        
        if log_file:
            # Initialize CSV log
            self.log_handle = open(log_file, 'w', newline='')
            self.log_writer = csv.writer(self.log_handle)
            self.log_writer.writerow([
                'total_steps',
                'episode_count',
                'traj_idx',
                'traj_step',
                'event',
                'obs_sum',  # Sum of observation for quick sanity check
                'reward',
                'done',
                'truncated',
            ])
            self.log_handle.flush()
    
    def _validate_trajectories(self):
        """Validate that all trajectories have correct structure."""
        for i, traj in enumerate(self.trajectories):
            # Check required keys
            required_keys = ['observations', 'actions', 'rewards']
            for key in required_keys:
                if key not in traj:
                    raise ValueError(f"Trajectory {i} missing required key: {key}")
            
            # Check lengths are consistent
            num_transitions = len(traj['actions'])
            if len(traj['rewards']) != num_transitions:
                raise ValueError(
                    f"Trajectory {i}: actions and rewards have inconsistent lengths "
                    f"({num_transitions} vs {len(traj['rewards'])})"
                )
            if len(traj['observations']) != num_transitions + 1:
                raise ValueError(
                    f"Trajectory {i}: observations length should be {num_transitions + 1}, "
                    f"got {len(traj['observations'])}"
                )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Start a new trajectory from the list.
        
        Args:
            seed: Random seed (unused, for compatibility)
            options: Additional options (unused, for compatibility)
        
        Returns:
            observation: First observation of the trajectory
            info: Empty dict
        """
        super().reset(seed=seed)
        
        self.total_resets += 1
        
        # Move to next trajectory (loop back to start if at end)
        if self.current_traj_idx >= self.num_trajectories:
            self.current_traj_idx = 0
            if self.verbose:
                print(f"[TrajectoryReplayEnv] Cycling back to trajectory 0 (completed all {self.num_trajectories} trajectories)")
        
        # Reset step counter
        self.current_step = 0
        
        # Return first observation
        traj = self.trajectories[self.current_traj_idx]
        obs = traj['observations'][0]
        
        # Log reset
        if self.log_writer:
            self.log_writer.writerow([
                self.total_steps,
                self.episode_count,
                self.current_traj_idx,
                self.current_step,
                'RESET',
                float(np.sum(obs)),
                '',
                '',
                '',
            ])
            self.log_handle.flush()
        
        if self.verbose:
            print(f"[TrajectoryReplayEnv] RESET: traj_idx={self.current_traj_idx}, obs_sum={np.sum(obs):.4f}, total_resets={self.total_resets}")
        
        return obs, {}
    
    def step(
        self,
        action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Return next transition from current trajectory.
        
        Args:
            action: Action (IGNORED - we replay pre-recorded transitions)
        
        Returns:
            observation: Next observation
            reward: Reward from this transition
            terminated: Whether episode terminated
            truncated: Whether episode was truncated (always False)
            info: Empty dict
        """
        traj = self.trajectories[self.current_traj_idx]
        
        # Get current transition
        reward = float(traj['rewards'][self.current_step])
        
        # Store step index before advancing
        step_before = self.current_step
        
        # Advance step
        self.current_step += 1
        self.total_steps += 1
        
        # Check if trajectory is done
        if self.current_step >= len(traj['actions']):
            done = True
            next_obs = traj['observations'][-1]  # Final observation
            self.episode_count += 1
            
            traj_idx_before = self.current_traj_idx
            self.current_traj_idx += 1  # Prepare for next trajectory
            
            if self.verbose:
                print(f"[TrajectoryReplayEnv] EPISODE DONE: traj_idx={traj_idx_before}, step={step_before}, reward={reward:.2f}, episode_count={self.episode_count}")
        else:
            done = False
            next_obs = traj['observations'][self.current_step]
        
        # Log step
        if self.log_writer:
            self.log_writer.writerow([
                self.total_steps,
                self.episode_count,
                self.current_traj_idx if not done else self.current_traj_idx - 1,
                step_before,
                'STEP' if not done else 'DONE',
                float(np.sum(next_obs)),
                reward,
                done,
                False,
            ])
            self.log_handle.flush()
        
        if self.verbose and not done:
            print(f"[TrajectoryReplayEnv] STEP: traj_idx={self.current_traj_idx}, step={self.current_step}, reward={reward:.2f}, obs_sum={np.sum(next_obs):.4f}")
        
        return next_obs, reward, done, False, {}
    
    def render(self):
        """Render is not supported for replay environments."""
        pass
    
    def close(self):
        """Clean up resources."""
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
            self.log_writer = None
            if self.verbose:
                print(f"[TrajectoryReplayEnv] Closed log file: {self.log_file}")
