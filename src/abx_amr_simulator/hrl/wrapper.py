"""Gymnasium wrapper for hierarchical RL with options.

The OptionsWrapper converts a flat-action environment into a hierarchical one:
- Manager selects options (macro-actions) at each step
- Options execute deterministically for k steps, collecting discounted rewards
- Wrapper handles env_state construction, action validation, and reward aggregation
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from abx_amr_simulator.hrl.base_option import OptionBase
from abx_amr_simulator.hrl.options import OptionLibrary


class OptionsWrapper(gym.Wrapper):
    """Gymnasium wrapper that converts flat-action env to hierarchical HRL env.
    
    The wrapper sits between the manager policy and the base environment:
    - Manager selects an option ID at each macro-timestep
    - Wrapper executes that option for up to k steps
    - Wrapper collects and aggregates rewards with discounting
    - Wrapper returns manager-level observations (AMR dynamics, progress, etc.)
    
    Attributes:
        env: Base environment (ABXAMREnv).
        option_library: OptionLibrary with all available options.
        gamma: Discount factor for reward aggregation across option steps.
        antibiotic_names: Ordered list of antibiotic names.
    """

    def __init__(
        self,
        env: gym.Env,
        option_library: OptionLibrary,
        gamma: float = 0.99,
    ):
        """Initialize OptionsWrapper.
        
        Args:
            env: Base Gymnasium environment (expected to be ABXAMREnv).
            option_library: OptionLibrary with instantiated options.
            gamma: Discount factor for reward aggregation (default 0.99).
        
        Raises:
            ValueError: If environment or option library invalid.
            RuntimeError: If validation fails.
        """
        super().__init__(env)

        self.option_library = option_library
        self.gamma = gamma

        # Antibiotic names are accessed via option_library.abx_name_to_index (no duplication)

        # Extract patient generator
        try:
            self.patient_generator = env.unwrapped.patient_generator
        except AttributeError as e:
            raise ValueError(
                f"Environment must have patient_generator. Error: {e}"
            )

        # Validate option library compatibility with environment
        try:
            option_library.validate_environment_compatibility(env, self.patient_generator)
        except ValueError as e:
            raise ValueError(
                f"Option library '{option_library.name}' incompatible with environment: {e}"
            )

        # Set up manager action/observation spaces
        # Manager action space: select option ID
        self.action_space = spaces.Discrete(len(option_library))

        # Manager observation space: Will be set after first env reset
        # For now, create a placeholder; will be updated in reset()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),  # Placeholder
            dtype=np.float32,
        )

        # State tracking
        self.current_env_obs = None
        self.current_step = 0
        self.max_steps = None

    def reset(self, seed=None, options=None):
        """Reset environment and option state.
        
        Args:
            seed: Optional seed for reproducibility.
            options: Optional dict of options for reset.
        
        Returns:
            Tuple of (manager_obs, info).
        """
        # Reset base environment
        base_obs, info = self.env.reset(seed=seed, options=options)
        self.current_env_obs = base_obs

        # Extract max_steps from environment
        try:
            self.max_steps = self.env.unwrapped.max_steps
        except AttributeError:
            self.max_steps = 1000  # Default fallback

        self.current_step = 0

        # Reset all options
        for option in self.option_library.options.values():
            option.reset()

        # Build initial manager observation
        manager_obs = self._build_manager_observation()

        return manager_obs, info

    def step(self, manager_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one macro-step (execute selected option for up to k steps).
        
        Args:
            manager_action: Integer in [0, len(option_library)) selecting which option to execute.
        
        Returns:
            Tuple of (manager_obs, aggregated_reward, terminated, truncated, info).
        
        Raises:
            ValueError: If manager_action invalid or option.decide() returns invalid actions.
        """
        # Validate manager action
        if not isinstance(manager_action, (int, np.integer)):
            raise TypeError(f"manager_action must be int, got {type(manager_action).__name__}")
        if not (0 <= manager_action < len(self.option_library)):
            raise ValueError(
                f"manager_action {manager_action} out of range [0, {len(self.option_library)-1}]. "
                f"Library has {len(self.option_library)} options."
            )

        # Get selected option
        option = self.option_library.get_option(manager_action)

        # Execute option for up to k steps
        macro_reward = 0.0
        discount = 1.0
        episode_terminated = False
        episode_truncated = False

        for substep in range(int(option.k) if option.k != float('inf') else self.max_steps):
            # Build env_state
            env_state = self._build_env_state(self.current_env_obs)

            # Get action from option (option accesses abx_name_to_index via env_state['option_library'])
            try:
                actions = option.decide(env_state)
            except Exception as e:
                raise RuntimeError(
                    f"Option '{option.name}' decide() failed: {e}\n"
                    f"env_state keys: {env_state.keys()}"
                )

            # Validate actions (Layer 3 validation)
            self._validate_actions(actions, option.name)

            # Step base environment
            base_obs, base_reward, terminated, truncated, info = self.env.step(actions)
            self.current_env_obs = base_obs
            self.current_step += 1

            # Accumulate discounted reward
            macro_reward += discount * base_reward
            discount *= self.gamma

            # Check for early termination by option
            if option.should_terminate(env_state):
                break

            # Check for episode end
            if terminated or truncated:
                episode_terminated = terminated
                episode_truncated = truncated
                break

        # Build manager observation
        manager_obs = self._build_manager_observation()

        # Build info dict
        step_info = {
            'option_name': option.name,
            'option_duration': substep + 1,  # Actual duration (may be less than k)
        }
        if 'info' in locals() and isinstance(info, dict):
            step_info.update(info)

        return manager_obs, macro_reward, episode_terminated, episode_truncated, step_info

    def _build_env_state(self, observation: np.ndarray) -> Dict[str, Any]:
        """Build decoded env_state dict from current observation.
        
        Args:
            observation: Raw observation from base environment.
        
        Returns:
            Dict with keys:
                - 'patients': List of patient dicts
                - 'num_patients': Number of patients
                - 'current_amr_levels': Dict mapping antibiotic -> resistance
                - 'current_step': Current timestep
                - 'max_steps': Episode length limit
        """
        # Extract patient data using patient generator
        num_patients = self.env.unwrapped.num_patients_per_time_step
        patients = self._extract_patients_from_obs(observation, num_patients)

        # Extract AMR levels
        current_amr_levels = self._get_current_amr_levels()

        env_state = {
            'patients': patients,
            'num_patients': num_patients,
            'current_amr_levels': current_amr_levels,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'option_library': self.option_library,  # Reference for options to access abx_name_to_index
        }

        return env_state

    def _extract_patients_from_obs(
        self, observation: np.ndarray, num_patients: int
    ) -> List[Dict[str, float]]:
        """Extract patient dicts from raw observation.
        
        Uses PatientGenerator to decode observed attributes.
        
        Args:
            observation: Raw observation array.
            num_patients: Number of patients per timestep.
        
        Returns:
            List of dicts, one per patient, with observed attributes.
        """
        patients = []
        try:
            # Get the patient objects from the environment
            env_patients = self.env.unwrapped.current_patients
            for patient in env_patients:
                patient_dict = {}
                for attr in self.patient_generator.visible_patient_attributes:
                    # Get observed value (with noise/bias) if available
                    obs_attr = f"{attr}_obs"
                    if hasattr(patient, obs_attr):
                        patient_dict[attr] = getattr(patient, obs_attr)
                    else:
                        # Fall back to true value
                        patient_dict[attr] = getattr(patient, attr, 0.0)
                patients.append(patient_dict)
        except (AttributeError, IndexError):
            # Fallback: create dummy patient dicts if extraction fails
            patients = [
                {attr: 0.5 for attr in self.patient_generator.visible_patient_attributes}
                for _ in range(num_patients)
            ]

        return patients

    def _get_current_amr_levels(self) -> Dict[str, float]:
        """Get current AMR levels for all antibiotics.
        
        Returns:
            Dict mapping antibiotic name -> resistance level (float in [0, 1]).
        """
        try:
            leaky_balloons = self.env.unwrapped.leaky_balloons
            current_amr = {
                abx_name: balloon.get_current_level()
                for abx_name, balloon in leaky_balloons.items()
            }
        except (AttributeError, TypeError):
            # Fallback: return zeros if not available
            current_amr = {abx_name: 0.0 for abx_name in self.antibiotic_names}

        return current_amr

    def _build_manager_observation(self) -> np.ndarray:
        """Build manager-level observation.
        
        For MVP, returns minimal observation:
        - Current AMR levels for all antibiotics
        - Current step / max_steps (progress)
        
        Returns:
            1D numpy array of floats.
        """
        # Get current AMR levels
        amr_levels = self._get_current_amr_levels()
        antibiotic_names = list(self.option_library.abx_name_to_index.keys())
        amr_obs = np.array(
            [amr_levels.get(abx, 0.0) for abx in antibiotic_names],
            dtype=np.float32,
        )

        # Progress: normalized current step
        progress = np.array([self.current_step / self.max_steps], dtype=np.float32)

        # Concatenate
        manager_obs = np.concatenate([amr_obs, progress])

        return manager_obs

    def _validate_actions(self, actions: np.ndarray, option_name: str) -> None:
        """Validate actions returned by option.decide() (Layer 3 validation).
        
        Args:
            actions: Action array returned by option.
            option_name: Name of option that returned actions (for error messages).
        
        Raises:
            TypeError: If actions not np.ndarray.
            ValueError: If shape wrong or indices out of range.
        """
        # Check type
        if not isinstance(actions, np.ndarray):
            raise TypeError(
                f"Option '{option_name}': decide() returned {type(actions).__name__}, "
                f"expected np.ndarray"
            )

        # Check shape
        num_patients = self.env.unwrapped.num_patients_per_time_step
        if actions.shape != (num_patients,):
            raise ValueError(
                f"Option '{option_name}': Expected action shape ({num_patients},), "
                f"got {actions.shape}"
            )

        # Check dtype (should be integer)
        if not np.issubdtype(actions.dtype, np.integer):
            raise TypeError(
                f"Option '{option_name}': Expected integer dtype, got {actions.dtype}"
            )

        # Check action range
        num_antibiotics = len(self.option_library.abx_name_to_index)
        max_action = num_antibiotics  # 0 to num_antibiotics-1 = prescribe; num_antibiotics = NO_RX
        if np.any((actions < 0) | (actions > max_action)):
            invalid_actions = actions[(actions < 0) | (actions > max_action)]
            antibiotic_names = list(self.option_library.abx_name_to_index.keys())
            raise ValueError(
                f"Option '{option_name}': Invalid action indices {set(invalid_actions)}. "
                f"Valid range: [0, {max_action}] for {num_antibiotics} antibiotics "
                f"{antibiotic_names}"
            )
