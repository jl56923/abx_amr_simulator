"""
MBPO Agent implementation.

Main orchestrator for Model-Based Policy Optimization training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from abx_amr_simulator.mbpo.dynamics_model import DynamicsModel
from abx_amr_simulator.mbpo.trajectory_replay_env import TrajectoryReplayEnv


@dataclass
class MBPOTrainingStats:
    """Container for training statistics returned by MBPOAgent.train()."""

    episode_returns: List[float]
    eval_returns: List[float]
    model_losses: List[Dict[str, float]]


class MBPOAgent:
    """MBPO Agent orchestrator."""

    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        """
        Initialize MBPO agent.

        Args:
            env: Real environment (ABXAMREnv instance)
            config: Full config dict with keys:
                - 'ppo': PPO hyperparameters
                - 'mbpo': MBPO-specific parameters
                - 'dynamics_model': DynamicsModel hyperparameters
        """
        if env is None:
            raise ValueError("env must not be None")
        if config is None:
            raise ValueError("config must not be None")

        self.env = env
        self.config = config
        
        # Optional logging directory for debugging
        self.log_dir = config.get('mbpo', {}).get('log_dir', None)

        obs_space = env.observation_space
        if not hasattr(obs_space, "shape") or obs_space.shape is None:
            raise ValueError("Observation space must define a shape")
        if len(obs_space.shape) != 1:
            raise ValueError("MBPOAgent expects a 1D flattened observation space")

        obs_dim = int(obs_space.shape[0])
        dynamics_config = config.get("dynamics_model", {})
        self.dynamics_model = DynamicsModel(
            obs_dim=obs_dim,
            action_space=env.action_space,
            config=dynamics_config,
        )

        ppo_config = config.get("ppo", {})
        policy_kwargs = ppo_config.get("policy_kwargs", {})

        self.policy = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=ppo_config.get("learning_rate", 3.0e-4),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            ent_coef=ppo_config.get("ent_coef", 0.0),
            vf_coef=ppo_config.get("vf_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            verbose=ppo_config.get("verbose", 0),
            tensorboard_log=ppo_config.get("tensorboard_log", None),
            seed=ppo_config.get("seed", None),
        )

        mbpo_config = config.get("mbpo", {})
        self.rollout_length = int(mbpo_config.get("rollout_length", 5))
        self.rollout_length_start = int(mbpo_config.get("rollout_length_start", self.rollout_length))
        self.rollout_length_max = int(mbpo_config.get("rollout_length_max", self.rollout_length))
        self.rollout_length_schedule_episodes = int(
            mbpo_config.get("rollout_length_schedule_episodes", 0)
        )

        self.num_synthetic_per_real = int(mbpo_config.get("num_synthetic_per_real", 20))
        self.num_synthetic_per_real_start = int(
            mbpo_config.get("num_synthetic_per_real_start", self.num_synthetic_per_real)
        )
        self.num_synthetic_per_real_max = int(
            mbpo_config.get("num_synthetic_per_real_max", self.num_synthetic_per_real)
        )
        self.num_synthetic_schedule_episodes = int(
            mbpo_config.get("num_synthetic_schedule_episodes", 0)
        )

        self.model_train_freq = int(mbpo_config.get("model_train_freq", 1))
        self.model_warmup_episodes = int(mbpo_config.get("model_warmup_episodes", 20))
        self.model_train_epochs = int(mbpo_config.get("model_train_epochs", 50))
        self.model_batch_size = int(mbpo_config.get("model_batch_size", 256))
        self.eval_freq = int(mbpo_config.get("eval_freq", 10))

        self.real_policy_train_freq = int(mbpo_config.get("real_policy_train_freq", 0))
        self.real_policy_train_start_episode = int(
            mbpo_config.get("real_policy_train_start_episode", 0)
        )
        self.real_policy_updates_multiplier = int(
            mbpo_config.get("real_policy_updates_multiplier", 1)
        )

        # Experience replay filtering for dynamics model training
        self.dynamics_training_filter_mode = mbpo_config.get(
            "dynamics_training_filter_mode", "none"  # "none", "threshold", "adaptive_threshold", "percentile"
        )
        self.dynamics_training_min_return_initial = float(
            mbpo_config.get("dynamics_training_min_return_initial", -500)
        )
        self.dynamics_training_min_return_final = float(
            mbpo_config.get("dynamics_training_min_return_final", -100)
        )
        self.dynamics_training_filter_schedule_episodes = int(
            mbpo_config.get("dynamics_training_filter_schedule_episodes", 50)
        )
        
        # Percentile-based filtering
        self.dynamics_training_keep_top_fraction = float(
            mbpo_config.get("dynamics_training_keep_top_fraction", 0.7)
        )
        
        # Minimum data requirements for hybrid mode
        self.min_good_transitions_for_model = int(
            mbpo_config.get("min_good_transitions_for_model", 1000)
        )
        self.min_good_episodes_for_model = int(
            mbpo_config.get("min_good_episodes_for_model", 5)
        )
        
        # Logging
        self.verbose_filtering = bool(
            mbpo_config.get("verbose_filtering", False)
        )

        self.real_data: List[Dict[str, Any]] = []
        self.episode_returns: List[float] = []  # Track return of each episode
        self.episode_start_indices: List[int] = []  # Track where each episode starts in real_data
        self.model_is_trained = False

    def _collect_real_episode(self) -> Dict[str, List[Any]]:
        """Collect one episode from real environment using current policy."""
        trajectory: Dict[str, List[Any]] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        obs, _ = self.env.reset(seed=None)
        done = False
        episode_return = 0.0

        while not done:
            action, _ = self.policy.predict(
                observation=obs,
                deterministic=False,
            )
            next_obs, reward, terminated, truncated, _ = self.env.step(action=action)
            done = bool(terminated or truncated)

            trajectory["observations"].append(obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)

            episode_return += float(reward)
            obs = next_obs

        trajectory["observations"].append(obs)
        trajectory["episode_return"] = episode_return
        return trajectory

    def _reconstruct_trajectories_from_real_data(self) -> List[Dict[str, List[Any]]]:
        """
        Reconstruct full trajectories from individual transitions in real_data.
        
        Since real_data contains individual transitions, we need to group them
        back into trajectories. A trajectory ends when done=True.
        
        Returns:
            List of trajectory dicts with keys 'observations', 'actions', 'rewards'
        """
        trajectories: List[Dict[str, List[Any]]] = []
        
        if not self.real_data:
            return trajectories
        
        current_trajectory: Dict[str, List[Any]] = {
            "observations": [],
            "actions": [],
            "rewards": [],
        }
        
        for transition in self.real_data:
            obs = transition["obs"]
            action = transition["action"]
            reward = transition["reward"]
            next_obs = transition["next_obs"]
            done = transition["done"]
            
            # Add to current trajectory
            current_trajectory["observations"].append(obs)
            current_trajectory["actions"].append(action)
            current_trajectory["rewards"].append(reward)
            
            # If episode ends, finalize trajectory and start new one
            if done:
                current_trajectory["observations"].append(next_obs)
                if (len(current_trajectory["actions"]) > 0 and
                    len(current_trajectory["rewards"]) > 0):
                    trajectories.append(current_trajectory)
                
                current_trajectory = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                }
        
        # Handle final incomplete trajectory (if no done signal)
        if (current_trajectory["actions"] and current_trajectory["observations"]):
            # Add final observation if available
            if self.real_data:
                final_transition = self.real_data[-1]
                current_trajectory["observations"].append(final_transition["next_obs"])
            trajectories.append(current_trajectory)
        
        return trajectories

    def _get_adaptive_threshold(self, episode_idx: int) -> float:
        """
        Compute adaptive return threshold for filtering dynamics training data.
        
        Linearly interpolates from initial to final threshold based on episode progress.
        """
        if self.dynamics_training_filter_schedule_episodes <= 0:
            return self.dynamics_training_min_return_final
        
        progress = min(float(episode_idx), float(self.dynamics_training_filter_schedule_episodes))
        fraction = progress / float(self.dynamics_training_filter_schedule_episodes)
        
        threshold = (self.dynamics_training_min_return_initial +
                    fraction * (self.dynamics_training_min_return_final -
                               self.dynamics_training_min_return_initial))
        return float(threshold)

    def _get_percentile_filtered_data(self) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter real data using percentile-based threshold.
        
        Returns:
            filtered_data: List of transitions from top episodes
            stats: Dict with keys ['threshold', 'num_episodes_kept', 'num_transitions_kept']
        """
        if not self.episode_returns or not self.episode_start_indices:
            return self.real_data, {
                'threshold': float('nan'),
                'num_episodes_kept': 0,
                'num_transitions_kept': len(self.real_data),
            }
        
        # Compute percentile threshold
        keep_fraction = self.dynamics_training_keep_top_fraction
        percentile_rank = (1.0 - keep_fraction) * 100.0
        threshold = float(np.percentile(a=self.episode_returns, q=percentile_rank))
        
        # Collect indices of episodes that meet threshold
        filtered_indices = set()
        num_episodes_kept = 0
        for ep_idx, ep_return in enumerate(self.episode_returns):
            if float(ep_return) >= threshold:
                num_episodes_kept += 1
                # Add all transitions from this episode
                start_idx = self.episode_start_indices[ep_idx]
                if ep_idx + 1 < len(self.episode_start_indices):
                    end_idx = self.episode_start_indices[ep_idx + 1]
                else:
                    end_idx = len(self.real_data)
                
                for trans_idx in range(start_idx, end_idx):
                    filtered_indices.add(trans_idx)
        
        # Build filtered data in original order
        filtered_data = [self.real_data[i] for i in sorted(filtered_indices)]
        
        stats = {
            'threshold': threshold,
            'num_episodes_kept': num_episodes_kept,
            'num_transitions_kept': len(filtered_data),
        }
        
        return filtered_data, stats
    
    def _get_filtered_real_data(self) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter real data based on episode returns using configured filter mode.
        
        Returns:
            filtered_data: Filtered list of transitions
            stats: Dict with diagnostic information (threshold, episodes kept, transitions kept)
        """
        if self.dynamics_training_filter_mode == "none":
            return self.real_data, {
                'threshold': float('nan'),
                'num_episodes_kept': len(self.episode_returns),
                'num_transitions_kept': len(self.real_data),
            }
        
        if self.dynamics_training_filter_mode == "percentile":
            return self._get_percentile_filtered_data()
        
        # Legacy modes: threshold and adaptive_threshold
        if not self.episode_returns or not self.episode_start_indices:
            return self.real_data, {
                'threshold': float('nan'),
                'num_episodes_kept': 0,
                'num_transitions_kept': len(self.real_data),
            }
        
        # For adaptive threshold, we need current episode index
        # Use len(episode_returns) - 1 as proxy for current episode
        current_episode_idx = len(self.episode_returns) - 1
        
        if self.dynamics_training_filter_mode == "adaptive_threshold":
            threshold = self._get_adaptive_threshold(episode_idx=current_episode_idx)
        elif self.dynamics_training_filter_mode == "threshold":
            threshold = self.dynamics_training_min_return_final
        else:
            return self.real_data, {
                'threshold': float('nan'),
                'num_episodes_kept': len(self.episode_returns),
                'num_transitions_kept': len(self.real_data),
            }
        
        # Collect indices of episodes that meet threshold
        filtered_indices = set()
        num_episodes_kept = 0
        for ep_idx, ep_return in enumerate(self.episode_returns):
            if float(ep_return) >= threshold:
                num_episodes_kept += 1
                # Add all transitions from this episode
                start_idx = self.episode_start_indices[ep_idx]
                if ep_idx + 1 < len(self.episode_start_indices):
                    end_idx = self.episode_start_indices[ep_idx + 1]
                else:
                    end_idx = len(self.real_data)
                
                for trans_idx in range(start_idx, end_idx):
                    filtered_indices.add(trans_idx)
        
        # Build filtered data in original order
        filtered_data = [self.real_data[i] for i in sorted(filtered_indices)]
        
        stats = {
            'threshold': threshold,
            'num_episodes_kept': num_episodes_kept,
            'num_transitions_kept': len(filtered_data),
        }
        
        return filtered_data, stats

    def _check_sufficient_data_for_model(self, filtered_data: List[Dict[str, Any]], num_episodes_kept: int) -> tuple[bool, str]:
        """
        Check if filtered data meets minimum requirements for model training.
        
        Args:
            filtered_data: List of filtered transitions
            num_episodes_kept: Number of episodes in filtered data
        
        Returns:
            sufficient: True if requirements met
            reason: Explanation if insufficient
        """
        num_transitions = len(filtered_data)
        
        if num_transitions < self.min_good_transitions_for_model:
            return False, (
                f"Insufficient transitions: {num_transitions}/{self.min_good_transitions_for_model} "
                f"(from {num_episodes_kept} episodes)"
            )
        
        if num_episodes_kept < self.min_good_episodes_for_model:
            return False, (
                f"Insufficient episodes: {num_episodes_kept}/{self.min_good_episodes_for_model} "
                f"(with {num_transitions} transitions)"
            )
        
        return True, ""
    
    def _add_to_real_data(self, trajectory: Dict[str, List[Any]]) -> None:
        """Add trajectory transitions to real data buffer, tracking episode metadata."""
        observations = trajectory["observations"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]
        episode_return = trajectory.get("episode_return", 0.0)

        # Track start index of this episode
        self.episode_start_indices.append(len(self.real_data))
        self.episode_returns.append(float(episode_return))

        for t in range(len(actions)):
            next_obs = observations[t + 1]
            self.real_data.append(
                {
                    "obs": observations[t],
                    "action": actions[t],
                    "reward": rewards[t],
                    "next_obs": next_obs,
                    "done": dones[t],
                }
            )

    def _train_dynamics_model(self, data: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """Train dynamics model on accumulated real data with optional filtering.
        
        Args:
            data: Optional pre-filtered data to train on. If None, uses self.real_data.
        """
        if data is not None:
            data_for_training = data
        elif self.dynamics_training_filter_mode == "none":
            data_for_training = self.real_data
        else:
            data_for_training, _ = self._get_filtered_real_data()
        
        if len(data_for_training) == 0:
            # No data passes filter, train on all data as fallback
            data_for_training = self.real_data
        
        metrics = self.dynamics_model.train_on_data(
            data=data_for_training,
            epochs=self.model_train_epochs,
            batch_size=self.model_batch_size,
            verbose=False,
        )
        self.model_is_trained = True
        return metrics

    def _get_scheduled_rollout_length(self, episode_idx: int) -> int:
        """Return rollout length scheduled by episode index."""
        if self.rollout_length_schedule_episodes <= 0:
            return int(self.rollout_length_max)
        progress = min(episode_idx, self.rollout_length_schedule_episodes)
        fraction = float(progress) / float(self.rollout_length_schedule_episodes)
        scheduled = self.rollout_length_start + fraction * (
            self.rollout_length_max - self.rollout_length_start
        )
        return max(1, int(round(scheduled)))

    def _get_scheduled_num_synthetic(self, episode_idx: int) -> int:
        """Return synthetic rollout count scheduled by episode index."""
        if self.num_synthetic_schedule_episodes <= 0:
            return int(self.num_synthetic_per_real_max)
        progress = min(episode_idx, self.num_synthetic_schedule_episodes)
        fraction = float(progress) / float(self.num_synthetic_schedule_episodes)
        scheduled = self.num_synthetic_per_real_start + fraction * (
            self.num_synthetic_per_real_max - self.num_synthetic_per_real_start
        )
        return max(1, int(round(scheduled)))

    def _generate_synthetic_rollouts(self, episode_idx: int) -> List[Dict[str, List[Any]]]:
        """Generate synthetic k-step rollouts using the dynamics model."""
        if len(self.real_data) == 0:
            return []
        if not self.model_is_trained:
            return []

        synthetic_trajectories: List[Dict[str, List[Any]]] = []
        num_samples = len(self.real_data)
        rollout_length = self._get_scheduled_rollout_length(episode_idx=episode_idx)
        num_synthetic = self._get_scheduled_num_synthetic(episode_idx=episode_idx)

        for _ in range(num_synthetic):
            start_idx = int(np.random.randint(low=0, high=num_samples))
            start_obs = self.real_data[start_idx]["obs"]

            trajectory: Dict[str, List[Any]] = {
                "observations": [],
                "actions": [],
                "rewards": [],
            }

            obs = start_obs
            for _ in range(rollout_length):
                action, _ = self.policy.predict(
                    observation=obs,
                    deterministic=False,
                )
                next_obs, reward = self.dynamics_model.predict(
                    obs=obs,
                    action=action,
                )

                trajectory["observations"].append(obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)

                obs = next_obs

            trajectory["observations"].append(obs)
            synthetic_trajectories.append(trajectory)

        return synthetic_trajectories

    def _train_policy_on_real(
        self,
        real_trajectories: List[Dict[str, List[Any]]],
        updates_multiplier: int,
    ) -> None:
        """Train PPO on real trajectories using replay environment."""
        if len(real_trajectories) == 0:
            return
        
        # Set up logging if log_dir is configured
        log_file = None
        verbose = False
        if self.log_dir:
            from pathlib import Path
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = str(log_path / 'trajectory_replay_real.csv')
            verbose = self.config.get('mbpo', {}).get('replay_env_verbose', False)

        replay_env = TrajectoryReplayEnv(
            trajectories=real_trajectories,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            log_file=log_file,
            verbose=verbose,
        )

        total_timesteps = int(
            sum(len(traj["actions"]) for traj in real_trajectories)
        )
        total_timesteps = int(total_timesteps * max(1, updates_multiplier))
        
        # CRITICAL FIX: PPO's rollout buffer size is fixed at initialization
        # The buffer must be completely filled before training can occur
        # Changing n_steps after init doesn't resize the buffer
        # So we must ensure total_timesteps >= buffer_size
        buffer_size = self.policy.rollout_buffer.buffer_size
        if total_timesteps < buffer_size:
            # Not enough real data to fill the buffer
            # We need to cycle through trajectories multiple times
            total_timesteps = buffer_size

        original_env = self.policy.get_env()
        self.policy.set_env(env=replay_env)
        self.policy.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
        )
        self.policy.set_env(env=original_env)

    def _train_policy_on_synthetic(self, synthetic_trajectories: List[Dict[str, List[Any]]]) -> None:
        """Train PPO on synthetic trajectories using replay environment."""
        if len(synthetic_trajectories) == 0:
            return
        
        # Set up logging if log_dir is configured
        log_file = None
        verbose = False
        if self.log_dir:
            from pathlib import Path
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = str(log_path / 'trajectory_replay_synthetic.csv')
            verbose = self.config.get('mbpo', {}).get('replay_env_verbose', False)

        replay_env = TrajectoryReplayEnv(
            trajectories=synthetic_trajectories,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            log_file=log_file,
            verbose=verbose,
        )

        total_timesteps = int(
            sum(len(traj["actions"]) for traj in synthetic_trajectories)
        )

        # CRITICAL FIX: PPO's rollout buffer size is fixed at initialization
        # The buffer must be completely filled before training can occur
        # Changing n_steps after init doesn't resize the buffer
        # So we must ensure total_timesteps >= buffer_size
        buffer_size = self.policy.rollout_buffer.buffer_size
        if total_timesteps < buffer_size:
            # Not enough synthetic data to fill the buffer
            # We need to cycle through trajectories multiple times
            total_timesteps = buffer_size

        original_env = self.policy.get_env()
        self.policy.set_env(env=replay_env)
        self.policy.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
        )
        self.policy.set_env(env=original_env)

    def train(self, total_episodes: int) -> MBPOTrainingStats:
        """
        Main MBPO training loop.

        Args:
            total_episodes: Number of real episodes to collect

        Returns:
            MBPOTrainingStats with episode returns, eval returns, and model losses
        """
        episode_returns: List[float] = []
        eval_returns: List[float] = []
        model_losses: List[Dict[str, float]] = []

        for episode_idx in range(total_episodes):
            trajectory = self._collect_real_episode()
            self._add_to_real_data(trajectory=trajectory)
            episode_returns.append(float(trajectory.get("episode_return", 0.0)))

            if self.real_policy_train_freq > 0:
                if episode_idx >= self.real_policy_train_start_episode:
                    if episode_idx % self.real_policy_train_freq == 0:
                        # Reconstruct full trajectories from accumulated real data
                        trajectories_for_training = self._reconstruct_trajectories_from_real_data()
                        
                        if trajectories_for_training:
                            self._train_policy_on_real(
                                real_trajectories=trajectories_for_training,
                                updates_multiplier=self.real_policy_updates_multiplier,
                            )

            if episode_idx >= self.model_warmup_episodes:
                if episode_idx % self.model_train_freq == 0:
                    metrics = self._train_dynamics_model()
                    model_losses.append(metrics)

                synthetic_trajectories = self._generate_synthetic_rollouts(
                    episode_idx=episode_idx,
                )
                self._train_policy_on_synthetic(
                    synthetic_trajectories=synthetic_trajectories,
                )

            if self.eval_freq > 0 and (episode_idx + 1) % self.eval_freq == 0:
                eval_stats = self.evaluate(num_episodes=5)
                eval_returns.append(float(eval_stats["mean_return"]))

        return MBPOTrainingStats(
            episode_returns=episode_returns,
            eval_returns=eval_returns,
            model_losses=model_losses,
        )

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate current policy on real environment."""
        returns: List[float] = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset(seed=None)
            done = False
            episode_return = 0.0

            while not done:
                action, _ = self.policy.predict(
                    observation=obs,
                    deterministic=True,
                )
                obs, reward, terminated, truncated, _ = self.env.step(action=action)
                episode_return += float(reward)
                done = bool(terminated or truncated)

            returns.append(episode_return)

        mean_return = float(np.mean(a=returns)) if len(returns) > 0 else 0.0
        return {
            "mean_return": mean_return,
            "num_episodes": float(num_episodes),
        }
