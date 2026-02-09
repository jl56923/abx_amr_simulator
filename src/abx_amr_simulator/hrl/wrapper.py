"""Gymnasium wrapper for hierarchical RL with options.

The OptionsWrapper converts a flat-action environment into a hierarchical one:
- Manager selects options (macro-actions) at each step
- Options execute deterministically for k steps, collecting discounted rewards
- Wrapper handles env_state construction, action validation, and reward aggregation
"""

from typing import Dict, Any, List, Tuple, Optional, cast
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from abx_amr_simulator.hrl.base_option import OptionBase
from abx_amr_simulator.hrl.options import OptionLibrary
from abx_amr_simulator.core import ABXAMREnv


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
        env: ABXAMREnv,
        option_library: OptionLibrary,
        gamma: float = 0.99,
        front_edge_use_full_vector: bool = False,
    ):
        """Initialize OptionsWrapper.
        
        Args:
            env: Base Gymnasium environment (expected to be ABXAMREnv).
            option_library: OptionLibrary with instantiated options.
            gamma: Discount factor for reward aggregation (default 0.99).
            front_edge_use_full_vector: If True, append the full boundary cohort
                feature vector (num_patients * num_visible_attrs). If False,
                append summary stats (mean + std) for each visible attribute.
                For large cohorts, summary stats are recommended.
        
        Raises:
            ValueError: If environment or option library invalid.
            RuntimeError: If validation fails.
        """
        super().__init__(env)

        self._base_env = cast(ABXAMREnv, env.unwrapped)

        self.option_library = option_library
        self.gamma = gamma
        self.front_edge_use_full_vector = front_edge_use_full_vector

        self.antibiotic_names = list(self.option_library.abx_name_to_index.keys())
        self._action_index_to_abx = {
            idx: abx_name for abx_name, idx in self.option_library.abx_name_to_index.items()
        }

        # Antibiotic names are accessed via option_library.abx_name_to_index (no duplication)

        # Extract patient generator
        try:
            self.patient_generator = self._base_env.patient_generator
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

        # Manager observation space: compute dimension analytically
        obs_dim = self._compute_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # State tracking
        self.current_env_obs: Optional[np.ndarray] = None
        self.current_step = 0
        try:
            self.max_steps = self._base_env.max_time_steps
        except AttributeError:
            self.max_steps = 1000  # Default fallback
        self._previous_option_id = -1
        self._consecutive_same_option_count = 0
        self._steps_since_prescribed = {abx: 0 for abx in self.antibiotic_names}
        self._last_aggregate_stats = None
        self._last_amr_start = None
        self._last_amr_end = None

    def _compute_observation_dimension(self) -> int:
        """Compute the manager observation space dimension analytically.
        
        Manager observation components:
        1. Aggregate patient stats: len(visible_attrs) * 4 (mean, std, max, min)
        2. AMR trajectory: 2 * num_antibiotics (start + end for each)
        3. Option history: 2 + num_antibiotics (previous_option_id, consecutive_count, steps_since_prescribed for each)
        4. Episode progress: 1 (current_step / max_steps)
        5. Front-edge cohort features:
           - If front_edge_use_full_vector: num_patients * len(visible_attrs)
           - Else: 2 * len(visible_attrs) (mean + std for each)
        
        Returns:
            Total observation dimension.
        """
        num_visible_attrs = len(self.patient_generator.visible_patient_attributes)
        num_antibiotics = len(self.antibiotic_names)
        num_patients = self._base_env.num_patients_per_time_step
        
        # Component dimensions
        aggregate_stats_dim = num_visible_attrs * 4
        amr_obs_dim = 2 * num_antibiotics
        option_history_dim = 2 + num_antibiotics
        progress_dim = 1
        
        if self.front_edge_use_full_vector:
            front_edge_dim = num_patients * num_visible_attrs
        else:
            front_edge_dim = 2 * num_visible_attrs
        
        total_dim = aggregate_stats_dim + amr_obs_dim + option_history_dim + progress_dim + front_edge_dim
        return total_dim

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
        self.current_env_obs = cast(np.ndarray, base_obs)

        self.current_step = 0
        self._previous_option_id = -1
        self._consecutive_same_option_count = 0
        self._steps_since_prescribed = {abx: 0 for abx in self.antibiotic_names}

        current_amr = self._get_current_amr_levels()
        self._last_amr_start = current_amr
        self._last_amr_end = current_amr
        self._last_aggregate_stats = self._initialize_empty_aggregate_stats()

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

        amr_start = self._get_current_amr_levels()
        tracked_patients = []

        if self.current_env_obs is None:
            raise RuntimeError("Environment observation not initialized. Call reset() first.")

        current_obs = cast(np.ndarray, self.current_env_obs)

        for substep in range(int(option.k) if option.k != float('inf') else self.max_steps):
            # Build env_state
            env_state = self._build_env_state(current_obs)

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
            self.current_env_obs = cast(np.ndarray, base_obs)
            current_obs = self.current_env_obs
            self.current_step += 1

            tracked_patients.extend(env_state["patients"])
            self._update_steps_since_prescribed(actions=actions)

            # Accumulate discounted reward
            base_reward_value = float(base_reward)
            macro_reward += discount * base_reward_value
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
        self._last_aggregate_stats = self._compute_aggregate_stats(tracked_patients=tracked_patients)
        self._last_amr_start = amr_start
        self._last_amr_end = self._get_current_amr_levels()
        self._update_option_history(option_id=manager_action)
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
                - 'current_step': Current episode timestep (from env.current_time_step)
                - 'max_steps': Episode length limit
                - 'option_library': Reference to OptionLibrary for accessing abx_name_to_index
        """
        # Extract patient data using patient generator
        num_patients = self._base_env.num_patients_per_time_step
        patients = self._extract_patients_from_obs(observation, num_patients)

        # Extract AMR levels
        current_amr_levels = self._get_current_amr_levels()

        # Get current step from environment (should be exposed as current_time_step)
        try:
            current_step = self._base_env.current_time_step
        except AttributeError:
            current_step = 0

        env_state = {
            'patients': patients,
            'num_patients': num_patients,
            'current_amr_levels': current_amr_levels,
            'current_step': current_step,
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
            env_patients = self._base_env.current_patients
            if env_patients is None:
                raise ValueError("Environment current_patients not initialized.")
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
        except (AttributeError, IndexError, ValueError):
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
            amr_balloons = self._base_env.amr_balloon_models
            current_amr = {
                abx_name: balloon.get_volume()
                for abx_name, balloon in amr_balloons.items()
            }
        except (AttributeError, TypeError):
            # Fallback: return zeros if not available
            current_amr = {abx_name: 0.0 for abx_name in self.antibiotic_names}

        return current_amr

    def _build_manager_observation(self) -> np.ndarray:
        """Build manager-level observation.
        
        Manager sees:
        - Aggregate patient statistics over the previous macro-action
        - AMR trajectory at macro-action start/end
        - Option history and steps since last prescribed
        - Episode progress
        - Front-edge cohort features (summary stats or full vector)
        
        Returns:
            1D numpy array of floats.
        """
        if self._last_aggregate_stats is None:
            aggregate_stats = self._initialize_empty_aggregate_stats()
        else:
            aggregate_stats = self._last_aggregate_stats

        amr_start = self._last_amr_start
        if amr_start is None:
            amr_start = self._get_current_amr_levels()

        amr_end = self._last_amr_end
        if amr_end is None:
            amr_end = self._get_current_amr_levels()

        amr_obs = np.array(
            [amr_start.get(abx, 0.0) for abx in self.antibiotic_names]
            + [amr_end.get(abx, 0.0) for abx in self.antibiotic_names],
            dtype=np.float32,
        )

        option_history = np.array(
            [
                float(self._previous_option_id),
                float(self._consecutive_same_option_count),
            ]
            + [float(self._steps_since_prescribed[abx]) for abx in self.antibiotic_names],
            dtype=np.float32,
        )

        progress = np.array([self.current_step / self.max_steps], dtype=np.float32)
        front_edge = self._build_front_edge_features()

        return np.concatenate([aggregate_stats, amr_obs, option_history, progress, front_edge])

    def _build_front_edge_features(self) -> np.ndarray:
        """Build front-edge patient cohort features at the boundary timestep.

        Returns:
            np.ndarray: Either full cohort vector or summary stats (mean + std)
                for each visible attribute.
        """
        num_patients = self._base_env.num_patients_per_time_step
        visible_attrs = list(self.patient_generator.visible_patient_attributes)
        if self.current_env_obs is None:
            raise RuntimeError("Environment observation not initialized. Call reset() first.")
        current_obs = cast(np.ndarray, self.current_env_obs)
        patients = self._extract_patients_from_obs(current_obs, num_patients)

        if not visible_attrs:
            return np.zeros(0, dtype=np.float32)
        if not patients:
            if self.front_edge_use_full_vector:
                length = num_patients * len(visible_attrs)
            else:
                length = 2 * len(visible_attrs)
            return np.zeros(length, dtype=np.float32)

        if self.front_edge_use_full_vector:
            values = []
            for patient in patients:
                for attr in visible_attrs:
                    values.append(float(patient.get(attr, 0.0)))
            return np.array(values, dtype=np.float32)

        cohort_matrix = np.array(
            [
                [float(patient.get(attr, 0.0)) for attr in visible_attrs]
                for patient in patients
            ],
            dtype=np.float32,
        )

        means = np.mean(cohort_matrix, axis=0)
        stds = np.std(cohort_matrix, axis=0)
        stats = []
        for idx in range(len(visible_attrs)):
            stats.extend([means[idx], stds[idx]])
        return np.array(stats, dtype=np.float32)

    def _initialize_empty_aggregate_stats(self) -> np.ndarray:
        visible_attrs = list(self.patient_generator.visible_patient_attributes)
        if not visible_attrs:
            return np.zeros(0, dtype=np.float32)
        return np.zeros(len(visible_attrs) * 4, dtype=np.float32)

    def _compute_aggregate_stats(self, tracked_patients: List[Dict[str, float]]) -> np.ndarray:
        visible_attrs = list(self.patient_generator.visible_patient_attributes)
        if not visible_attrs:
            return np.zeros(0, dtype=np.float32)
        if not tracked_patients:
            return np.zeros(len(visible_attrs) * 4, dtype=np.float32)

        matrix = np.array(
            [
                [float(patient.get(attr, 0.0)) for attr in visible_attrs]
                for patient in tracked_patients
            ],
            dtype=np.float32,
        )
        means = np.mean(matrix, axis=0)
        stds = np.std(matrix, axis=0)
        maxs = np.max(matrix, axis=0)
        mins = np.min(matrix, axis=0)
        stats = []
        for idx in range(len(visible_attrs)):
            stats.extend([means[idx], stds[idx], maxs[idx], mins[idx]])
        return np.array(stats, dtype=np.float32)

    def _update_option_history(self, option_id: int) -> None:
        if option_id == self._previous_option_id:
            self._consecutive_same_option_count += 1
        else:
            self._consecutive_same_option_count = 1
        self._previous_option_id = option_id

    def _update_steps_since_prescribed(self, actions: np.ndarray) -> None:
        prescribed = set()
        for action in actions:
            abx_name = self._action_index_to_abx.get(int(action))
            if abx_name is None or abx_name == "no_treatment":
                continue
            prescribed.add(abx_name)

        for abx in self.antibiotic_names:
            if abx in prescribed:
                self._steps_since_prescribed[abx] = 0
            else:
                self._steps_since_prescribed[abx] += 1

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
        num_patients = self._base_env.num_patients_per_time_step
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
