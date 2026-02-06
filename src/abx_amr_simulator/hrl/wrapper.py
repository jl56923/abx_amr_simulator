"""
OptionsWrapper: Gymnasium-compatible wrapper that adds HRL temporal abstraction.

Wraps ABXAMREnv to present a macro-action interface to the manager agent.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Any

from .options import Option, OptionLibrary
from .manager_obs import ManagerObsBuilder


class OptionsWrapper(gym.Wrapper):
    """
    Wraps ABXAMREnv to enable hierarchical RL via macro-actions.
    
    The wrapper:
    1. Receives macro-action selection from manager (option_id)
    2. Executes the option in the base env for k steps
    3. Aggregates reward (discounted sum)
    4. Constructs manager observation
    5. Returns aggregated state and reward to manager
    """
    
    def __init__(
        self,
        env: gym.Env,
        option_library: OptionLibrary,
        gamma: float = 0.99,
        include_prospective_cohort_stats: bool = False,
        prospective_attributes: Optional[list] = None,
    ):
        """
        Initialize OptionsWrapper.
        
        Args:
            env: Base ABXAMREnv environment
            option_library: OptionLibrary with deterministic macro-actions
            gamma: Discount factor for reward aggregation
            include_prospective_cohort_stats: Whether to add cohort stats to manager obs
            prospective_attributes: List of patient attributes for cohort stats
        """
        super().__init__(env)
        
        self.option_library = option_library
        self.gamma = gamma
        
        self._num_abx = getattr(env, "num_abx", 2)
        self._num_patients = getattr(env, "num_patients_per_time_step", 1)

        # Manager observation builder
        self.obs_builder = ManagerObsBuilder(
            num_antibiotics=self._num_abx,
            include_prospective_cohort_stats=include_prospective_cohort_stats,
            prospective_attributes=prospective_attributes,
        )
        
        # Observation space: manager operates on aggregated observations
        manager_obs_dim = self.obs_builder.compute_observation_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(manager_obs_dim,),
            dtype=np.float32,
        )
        
        # Action space: manager selects option id
        self.action_space = gym.spaces.Discrete(len(self.option_library))
        
        # Internal state tracking
        self._base_env_step_count = 0
        self._amr_start_of_option = None
        self._amr_end_of_option = None
        self._last_manager_obs = None
        self._last_base_obs = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        base_obs, base_info = self.env.reset(seed=seed, options=options)
        
        self._base_env_step_count = 0
        self.obs_builder.reset()
        
        self._last_base_obs = base_obs

        # Extract initial AMR from base observation
        # Base obs structure: [patient_features..., amr_levels]
        amr_start = base_obs[-self._num_abx:].copy()
        self._amr_start_of_option = amr_start
        self._amr_end_of_option = amr_start.copy()
        
        # Build initial manager observation
        manager_obs = self.obs_builder.build_observation(
            amr_start=amr_start,
            amr_end=amr_start,
            current_option_id=0,
            steps_in_episode=0,
            total_episode_steps=getattr(self.env, "max_time_steps", self.env.spec.max_episode_steps if self.env.spec else 500),
        )
        
        self._last_manager_obs = manager_obs
        
        return manager_obs, base_info
    
    def step(self, manager_action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute manager action (select option), run it in base env, return aggregated result.
        
        Args:
            manager_action: Option ID (int in [0, num_options))
        
        Returns:
            (manager_obs, aggregated_reward, terminated, truncated, info)
        """
        # Validate action
        if manager_action < 0 or manager_action >= len(self.option_library):
            raise ValueError(
                f"Manager action {manager_action} out of range [0, {len(self.option_library)})"
            )
        
        option = self.option_library.get_option(manager_action)
        
        # Get remaining steps in episode
        max_steps = getattr(self.env, "max_time_steps", self.env.spec.max_episode_steps if self.env.spec else 500)
        remaining_steps = max_steps - self._base_env_step_count
        
        # Cap option duration to remaining steps
        actual_k = min(option.duration, remaining_steps)
        
        if actual_k <= 0:
            raise RuntimeError("No remaining steps in episode before option execution")
        
        # Record AMR at start of option
        # Extract from last base obs
        if self._last_manager_obs is not None:
            # First 2 elements of manager obs are amr_start, next 2 are amr_end
            amr_start = self._amr_end_of_option.copy()  # Use end from previous option
        else:
            amr_start = np.array([0.0, 0.0])
        
        # Execute option in base env
        cumulative_reward = 0.0
        base_obs = None
        terminated = False
        truncated = False
        
        for step_in_option in range(actual_k):
            # Get action from option
            base_action = option.get_action(step_in_option)
            
            # Convert to base env action format
            if hasattr(self.env.action_space, 'nvec'):
                # MultiDiscrete: one choice per patient
                env_action = np.full(self._num_patients, base_action, dtype=np.int32)
            else:
                env_action = base_action

            if base_action > 0:
                self.obs_builder.update_steps_since_drug(drug_id=base_action - 1)
            
            # Step base env
            base_obs, base_reward, terminated, truncated, base_info = self.env.step(env_action)
            self._last_base_obs = base_obs
            
            # Accumulate reward with discount
            cumulative_reward += (self.gamma ** step_in_option) * base_reward
            self._base_env_step_count += 1
            
            if terminated or truncated:
                break
        
        # Record AMR at end of option
        if base_obs is None and self._last_base_obs is not None:
            base_obs = self._last_base_obs

        amr_end = base_obs[-self._num_abx:].copy() if base_obs is not None else np.zeros(self._num_abx, dtype=np.float32)
        self._amr_end_of_option = amr_end
        
        # Build manager observation
        manager_obs = self.obs_builder.build_observation(
            amr_start=amr_start,
            amr_end=amr_end,
            current_option_id=manager_action,
            steps_in_episode=self._base_env_step_count,
            total_episode_steps=max_steps,
        )
        
        self._last_manager_obs = manager_obs
        
        # Compute macro-discount for next manager step
        gamma_macro = self.gamma ** actual_k
        
        # Return info for analysis
        info = base_info if base_info is not None else {}
        info['actual_option_duration'] = actual_k
        info['gamma_macro'] = gamma_macro
        info['base_env_step_count'] = self._base_env_step_count
        
        return manager_obs, cumulative_reward, terminated, truncated, info
