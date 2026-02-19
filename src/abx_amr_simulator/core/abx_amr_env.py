# This is a class definition file for the abx_amr_env environment. The purpose of this environment is to allow a RL agent to choose to prescribe particular antibiotics, with the goal of maximizing overall benefit, where benefit is defined as the sum of the individual clinical benefit over all patients, the sum of the individual adverse effects over all patients, and the overall AMR cost incurred by the prescriptions.

# %% Import libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from collections import deque

import pdb # for debugging

# Import leaky_balloon.py from the same directory
import os 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
sys.path.append(parent_dir)

from .leaky_balloon import AMR_LeakyBalloon
from .base_amr_dynamics import AMRDynamicsBase
from .base_patient_generator import PatientGeneratorBase
from .base_reward_calculator import RewardCalculatorBase
from .types import Patient

# %% Define the ABX AMR Environment
class ABXAMREnv(gym.Env):
    """
    Single-agent antibiotic prescribing environment with AMR dynamics.
    
    Custom Environment for AMR capacitors that 'charge up' in response to antibiotic 
    prescribing rates. This environment follows gymnasium interface.
    
    OWNERSHIP (for future multi-agent refactor):
    - Owns: episode timestep, action/observation spaces, step orchestration
    - Delegates: patient sampling (PatientGenerator), 
                 reward calculation (RewardCalculator),
                 AMR dynamics (AMRDynamicsBase instances via AMR_LeakyBalloon)
    
    Future refactor notes:
    - PatientGenerator will become per-Locale
    - AMR balloons will move into Locale abstraction
    - Step logic will move to World orchestrator
    - This class will become thin adapter for PettingZoo ParallelEnv
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_calculator, patient_generator, antibiotics_AMR_dict: dict = None, crossresistance_matrix: dict = None,num_patients_per_time_step: int = 10, update_visible_AMR_levels_every_n_timesteps: int = 1, add_noise_to_visible_AMR_levels: float = 0.0, add_bias_to_visible_AMR_levels: float = 0.0, max_time_steps: int = 1000, include_steps_since_amr_update_in_obs: bool = False, enable_temporal_features: bool = False, temporal_windows: List[int] = None):
        """
        Initializes the environment with antibiotic specifications and heterogeneous patient population.
        
        Args:
            reward_calculator: Instantiated RewardCalculator instance (pre-initialized externally).
            patient_generator: Instantiated PatientGenerator instance (or PatientGeneratorMixer).
                Must have .sample(n_patients, true_amr_levels, rng) method that returns List[Patient].
                Must implement .observe(patients) and .obs_dim(num_patients) methods.
                PatientGenerator owns visibility configuration internally.
                Pre-initialized externally like RewardCalculator for composability.
            antibiotics_AMR_dict: Dictionary mapping antibiotic names to parameter dicts for AMRDynamicsBase (default: AMR_LeakyBalloon).
                Each parameter dict must contain:
                {'leak': float, 'flatness_parameter': float, 'permanent_residual_volume': float, 'initial_amr_level': float}
            num_patients_per_time_step: Number of patients per time step
            update_visible_AMR_levels_every_n_timesteps: Frequency of visible AMR level updates
            add_noise_to_visible_AMR_levels: Standard deviation of Gaussian noise added to visible AMR levels
            add_bias_to_visible_AMR_levels: Constant bias added to visible AMR levels (range: [-1.0, 1.0]).
                Negative values underestimate true AMR (agent sees lower AMR than true value).
                Positive values overestimate true AMR (agent sees higher AMR than true value).
                Resulting visible levels are clipped to [0.0, 1.0] range.
            max_time_steps: Maximum number of steps per episode
            crossresistance_matrix: Optional dict of dicts defining off-diagonal crossresistance ratios. Only non-self entries.
                Diagonal auto-set to 1.0. Example: {"A": {"B": 0.5}, "B": {"A": 0.01}}
            include_steps_since_amr_update_in_obs: If True, includes the number of steps since the last AMR update as an additional observation feature.
            enable_temporal_features: If True, append prescription counts and AMR deltas to observations.
            temporal_windows: List of window sizes for prescription counting (e.g., [10, 50]). Only used if enable_temporal_features=True.
        """
        
        super().__init__()
        
        # Store flag for observation augmentation
        self.include_steps_since_amr_update_in_obs: bool = include_steps_since_amr_update_in_obs
        
        # Temporal features config
        self.enable_temporal_features: bool = enable_temporal_features
        self.temporal_windows: List[int] = temporal_windows if temporal_windows is not None else [10, 50]
        
        # Do validation checks:
        if not len(antibiotics_AMR_dict) > 0:
            raise ValueError("antibiotics_AMR_dict must contain at least one antibiotic definition.")
        
        if not hasattr(patient_generator, 'sample'):
            raise ValueError("patient_generator must have a .sample(n_patients, true_amr_levels, rng) method.")
        
        if not hasattr(patient_generator, 'observe'):
            raise ValueError(
                "patient_generator must implement observe(patients) method. "
                "Ensure your PatientGenerator inherits from PatientGeneratorBase."
            )
        
        if not hasattr(patient_generator, 'obs_dim'):
            raise ValueError(
                "patient_generator must implement obs_dim(num_patients) method. "
                "Ensure your PatientGenerator inherits from PatientGeneratorBase."
            )
        
        if not len(reward_calculator.abx_clinical_reward_penalties_info_dict) > 0:
            raise ValueError("reward_calculator must contain at least one antibiotic definition in abx_clinical_reward_penalties_info_dict.")
        
        if num_patients_per_time_step <= 0:
            raise ValueError("num_patients_per_time_step must be a positive integer.")
        
        if update_visible_AMR_levels_every_n_timesteps < 1:
            raise ValueError("update_visible_AMR_levels_every_n_timesteps must be an integer greater than or equal to 1.")
        
        # add_noise_to_visible_AMR_levels must be greater than 0
        if add_noise_to_visible_AMR_levels < 0.0:
            raise ValueError("add_noise_to_visible_AMR_levels must be greater than or equal to 0.0.")
        
        if not (-1.0 <= add_bias_to_visible_AMR_levels <= 1.0):
            raise ValueError("add_bias_to_visible_AMR_levels must be between -1.0 and 1.0 (inclusive). "
                           "Negative values underestimate AMR, positive values overestimate. "
                           "Resulting visible AMR levels are clipped to [0.0, 1.0].")

        # Check that the keys for antibiotics_AMR_dict and reward_calculator match:
        if set(antibiotics_AMR_dict.keys()) != set(reward_calculator.antibiotic_names):
            raise ValueError(f"The antibiotic names in antibiotics_AMR_dict and reward_calculator.abx_clinical_reward_penalties_info_dict['abx_adverse_effects_info'] must match. The antibiotic names in antibiotics_AMR_dict are {list(antibiotics_AMR_dict.keys())}, while the antibiotic names in reward_calculator are {reward_calculator.antibiotic_names}.")
        
        # Check that each entry in antibiotics_AMR_dict has the required keys
        required_keys_antibiotics_AMR_dict = {'leak', 'flatness_parameter', 'permanent_residual_volume', 'initial_amr_level'}
        for abx_name, params in antibiotics_AMR_dict.items():
            if not required_keys_antibiotics_AMR_dict.issubset(params.keys()):
                missing = required_keys_antibiotics_AMR_dict - set(params.keys())
                raise ValueError(f"Antibiotic '{abx_name}' is missing required parameters: {missing} in antibiotics_AMR_dict.")
        
        self.antibiotics_AMR_dict = antibiotics_AMR_dict
        self.antibiotic_names: List[str] = list(antibiotics_AMR_dict.keys())
        self.num_abx: int = len(antibiotics_AMR_dict)
        
        # Build and validate crossresistance matrix
        self.crossresistance_matrix: Dict[str, Dict[str, float]] = self._build_and_validate_crossresistance_matrix(crossresistance_matrix, self.antibiotic_names)
        
        # Store patient generator with type hint
        self.patient_generator: PatientGeneratorBase = patient_generator
        
        # Seed synchronization: prefer RewardCalculator seed; warn if PatientGenerator differs
        if hasattr(reward_calculator, 'seed') and hasattr(patient_generator, 'seed'):
            if reward_calculator.seed != patient_generator.seed:
                import warnings
                warnings.warn(
                    f"PatientGenerator seed ({patient_generator.seed}) differs from "
                    f"RewardCalculator seed ({reward_calculator.seed}). "
                    f"Using RewardCalculator seed for shared RNG to preserve reproducibility.",
                    UserWarning
                )
                patient_generator.seed = reward_calculator.seed
                if hasattr(patient_generator, '_sync_child_seeds'):
                    patient_generator._sync_child_seeds(reward_calculator.seed)
        
        self.num_patients_per_time_step: int = num_patients_per_time_step
        self.max_time_steps: int = max_time_steps
        self.update_visible_AMR_levels_every_n_timesteps: int = update_visible_AMR_levels_every_n_timesteps
        # Keep a single canonical name to avoid typos in helpers.
        self.amr_update_frequency: int = update_visible_AMR_levels_every_n_timesteps
        self.add_noise_to_visible_AMR_levels: float = add_noise_to_visible_AMR_levels
        self.add_bias_to_visible_AMR_levels: float = add_bias_to_visible_AMR_levels
        
        # Set up reward calculator
        self.reward_calculator: RewardCalculatorBase = reward_calculator
        
        # Validate that both components have required class constants (fail loud if missing)
        if not hasattr(self.reward_calculator, 'REQUIRED_PATIENT_ATTRS'):
            raise ValueError(
                "RewardCalculator must define REQUIRED_PATIENT_ATTRS class constant. "
                "Ensure your RewardCalculator inherits from RewardCalculatorBase."
            )
        if not hasattr(self.patient_generator, 'PROVIDES_ATTRIBUTES'):
            raise ValueError(
                "PatientGenerator must define PROVIDES_ATTRIBUTES class constant. "
                "Ensure your PatientGenerator inherits from PatientGeneratorBase."
            )
        
        # Validate compatibility between PatientGenerator and RewardCalculator
        required = set(self.reward_calculator.REQUIRED_PATIENT_ATTRS)
        provided = set(self.patient_generator.PROVIDES_ATTRIBUTES)
        missing = required - provided
        if missing:
            raise ValueError(
                f"RewardCalculator requires Patient attributes {sorted(missing)} "
                f"but PatientGenerator doesn't provide them. "
                f"PatientGenerator provides: {sorted(provided)}, "
                f"RewardCalculator requires: {sorted(required)}"
            )
        
        # RNG: environment owns a single Generator and threads it through RewardCalculator and PatientGenerator
        self._initial_seed: Optional[int] = getattr(self.reward_calculator, 'seed', None)
        self.np_random: np.random.Generator = np.random.default_rng(self._initial_seed)
        if hasattr(self.reward_calculator, 'rng'):
            self.reward_calculator.rng = self.np_random
        if hasattr(self.reward_calculator, 'seed'):
            self.reward_calculator.seed = self._initial_seed
        if hasattr(self.patient_generator, 'rng'):
            self.patient_generator.rng = self.np_random
        if hasattr(self.patient_generator, 'seed'):
            self.patient_generator.seed = self._initial_seed
        
        # Transfer ownership: mark components as environment-owned
        if hasattr(self.reward_calculator, '_set_environment_owned'):
            self.reward_calculator._set_environment_owned()
        if hasattr(self.patient_generator, '_set_environment_owned'):
            self.patient_generator._set_environment_owned()

        # Action space: MultiDiscrete with mutually exclusive prescribing
        # Each patient gets one action from: 0 (no treatment), 1 (Antibiotic_A), 2 (Antibiotic_B), ...
        # Total actions per patient = num_antibiotics + 1
        num_actions_per_patient = self.num_abx + 1
        self.action_space = spaces.MultiDiscrete([num_actions_per_patient] * self.num_patients_per_time_step)
        
        # Initialize temporal feature tracking (will be properly initialized in reset())
        from collections import deque
        self.prescription_history: Dict[str, List[deque]] = {}
        self.previous_visible_amr_levels: Dict[str, float] = {}
        if self.enable_temporal_features:
            for abx_name in self.antibiotic_names:
                # One deque per window size per antibiotic
                self.prescription_history[abx_name] = [deque(maxlen=w) for w in self.temporal_windows]
                self.previous_visible_amr_levels[abx_name] = 0.0
        
        # Observation space consists of:
        # - Patient observed attributes provided by PatientGenerator.observe (num_patients * num_visible_attrs)
        # - Community AMR levels for each antibiotic (num_abx values, each in [0, 1])
        # - (Optional) Temporal features: prescription counts per window + AMR deltas
        # Shape: (patient_generator.obs_dim(num_patients) + num_abx + temporal_dim,)
        # Note: Actions are mutually exclusive - agent picks one antibiotic OR no treatment per patient
        obs_dim_patients = int(self.patient_generator.obs_dim(num_patients_per_time_step))
        obs_dim = obs_dim_patients + self.num_abx
        if include_steps_since_amr_update_in_obs:
            # Add an extra dimension for the steps since AMR update (scalar, repeated for each patient)
            obs_dim += 1
        if self.enable_temporal_features:
            # Add temporal features: (num_abx * num_windows) prescription counts + (num_abx) AMR deltas
            num_windows = len(self.temporal_windows)
            temporal_dim = self.num_abx * num_windows + self.num_abx
            obs_dim += temporal_dim
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.state = None
        self.current_time_step: int = 0
        self.current_patients: Optional[List[Patient]] = None  # Will be populated by reset() and step()
        
        # AMR levels visible to the agent (updated periodically); intialized to zero, but this will be updated on reset()
        # Dictionary mapping antibiotic names to visible AMR levels
        self.visible_amr_levels: Dict[str, float] = {name: 0.0 for name in self.antibiotic_names}
        
        # Flag and counter for updating visible AMR levels
        self.steps_since_amr_update: int = 0
        
        # Flag to control whether to log full patient attributes in info dict
        # Set to True during eval episodes to collect detailed patient data
        self.log_full_patient_attributes: bool = False
        
        # Default parameters for AMRDynamicsBase (AMR_LeakyBalloon)
        default_params = {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0
        }
        
        # Initialize leaky balloon models for each antibiotic using provided parameters with fallback defaults
        self.amr_balloon_models: Dict[str, AMRDynamicsBase] = {}
        for abx_name in self.antibiotic_names:
            abx_params = antibiotics_AMR_dict[abx_name]
            # Extract parameters with defaults
            leak = abx_params.get('leak', default_params['leak'])
            flatness_parameter = abx_params.get('flatness_parameter', default_params['flatness_parameter'])
            permanent_residual_volume = abx_params.get('permanent_residual_volume', default_params['permanent_residual_volume'])
            initial_amr_level = abx_params.get('initial_amr_level', default_params['initial_amr_level'])
            
            self.amr_balloon_models[abx_name] = AMR_LeakyBalloon(
                leak=leak,
                flatness_parameter=flatness_parameter,
                permanent_residual_volume=permanent_residual_volume,
                initial_amr_level=initial_amr_level
            )

        # At the end of the initializer, reset the environment to set initial state
        self.reset(seed=self._initial_seed)

    def _build_and_validate_crossresistance_matrix(self, crossresistance_dict, antibiotic_names):
        """
        Builds and validates a crossresistance matrix from user-provided dict.
        
        Args:
            crossresistance_dict: Optional dict of dicts with off-diagonal crossresistance ratios.
                           Only non-self entries should be specified.
            antibiotic_names: List of antibiotic names.
        
        Returns:
            dict: Fully populated crossresistance matrix with all entries (diagonal set to 1.0).
        
        Raises:
            ValueError: If matrix has invalid structure or values.
        """
        # Initialize matrix: all zeros
        matrix = {from_abx: {to_abx: 0.0 for to_abx in antibiotic_names} for from_abx in antibiotic_names}
        
        # If no crossresistance provided, return identity matrix
        if crossresistance_dict is None:
            for abx in antibiotic_names:
                matrix[abx][abx] = 1.0
            return matrix
        
        # Populate with user-provided off-diagonal values
        for from_abx, targets in crossresistance_dict.items():
            if from_abx not in antibiotic_names:
                raise ValueError(f"Antibiotic '{from_abx}' in crossresistance_matrix not found in antibiotic_names.")
            
            for to_abx, ratio in targets.items():
                if to_abx not in antibiotic_names:
                    raise ValueError(f"Antibiotic '{to_abx}' in crossresistance_matrix not found in antibiotic_names.")
                
                # Disallow specifying self-entries in user input
                if from_abx == to_abx:
                    raise ValueError(f"Crossresistance matrix should not include self-entries (e.g., '{from_abx}' -> '{to_abx}'). Diagonal is auto-set to 1.0.")
                
                # Validate ratio is in [0, 1]
                if not isinstance(ratio, (int, float)) or not (0.0 <= ratio <= 1.0):
                    raise ValueError(f"Crossresistance ratio '{from_abx}' -> '{to_abx}' must be a number in [0, 1], got {ratio}.")
                
                matrix[from_abx][to_abx] = ratio
        
        # Force all diagonal entries to 1.0
        for abx in antibiotic_names:
            matrix[abx][abx] = 1.0
        
        return matrix

    # Define small internal helper function to get actual AMR levels:
    def _get_actual_amr_levels(self) -> Dict[str, float]:
        """Returns the actual AMR levels from the balloon models as a dictionary."""
        return {abx_name: self.amr_balloon_models[abx_name].get_volume() for abx_name in self.antibiotic_names}

    @property
    def no_treatment_action(self) -> int:
        """Action index for no treatment."""
        return self.reward_calculator.abx_name_to_index['no_treatment']

    def reset(self, *, seed=None, options: Optional[dict] = None):
        """Reset the environment to initial state.
        
        Resets AMR balloon dynamics to initial pressure levels, samples a new patient cohort,
        and constructs the initial observation. Synchronizes RNG across all components
        (environment, reward calculator, patient generator) when seed is provided.

        Args:
            seed (int, optional): Random seed for reproducibility. If provided, reseeds
                the shared RNG used by environment, reward calculator, and patient generator
                to ensure synchronized stochastic draws across components.
            options (dict, optional): Additional reset options (Gymnasium API requirement,
                currently unused).

        Returns:
            tuple: (observation, info) where:
                - observation (np.ndarray): Initial state vector of shape (obs_dim,) containing
                  patient observed attributes + community AMR levels (+ optional steps_since_amr_update).
                - info (dict): Empty dict (no extra information at reset).

        Notes:
            - RNG is shared with RewardCalculator and PatientGenerator. When a seed is provided,
              we reseed the shared Generator in-place to preserve object identity while ensuring
              all components draw from the same deterministic stream.
            - AMR balloons are reset to `initial_amr_level` as specified in `antibiotics_AMR_dict`.
            - Visible AMR levels are immediately updated (no delay on reset).
            - Patient cohort is sampled fresh using PatientGenerator.
        """
        super().reset(seed=seed)

        # Re-seed the shared RNG in-place when a seed is supplied so env+reward+patient_generator
        # draw from the same deterministic stream and remain synchronized.
        if seed is not None:
            tmp = np.random.default_rng(seed)
            # Preserve object identity by copying state into the shared generator
            self.np_random.bit_generator.state = tmp.bit_generator.state
            if hasattr(self.reward_calculator, 'rng'):
                self.reward_calculator.rng = self.np_random
            if hasattr(self.reward_calculator, 'seed'):
                self.reward_calculator.seed = seed
            if hasattr(self.patient_generator, 'rng'):
                self.patient_generator.rng = self.np_random
            if hasattr(self.patient_generator, 'seed'):
                self.patient_generator.seed = seed

        # Reset time step counter
        self.current_time_step = 0
        
        # Reset all AMR balloon models
        for abx_name in self.antibiotic_names:
            initial_pressure = self.antibiotics_AMR_dict.get(abx_name, {}).get('initial_amr_level', 0.0)
            self.amr_balloon_models[abx_name].reset(initial_amr_level=initial_pressure)
        
        # Update visible AMR levels immediately; this will automatically take care of setting the state variable self.visible_amr_levels, and it will also set self.steps_since_amr_update to 0.
        self.update_visible_amr_levels(force_update_for_reset=True)
        
        # Reset temporal feature tracking
        if self.enable_temporal_features:
            from collections import deque
            for abx_name in self.antibiotic_names:
                # Clear all deques and reinitialize with zeros
                self.prescription_history[abx_name] = [deque([0] * w, maxlen=w) for w in self.temporal_windows]
                self.previous_visible_amr_levels[abx_name] = self.visible_amr_levels[abx_name]
        
        # Sample initial patient cohort using PatientGenerator
        true_amr_levels = {
            abx_name: self.amr_balloon_models[abx_name].get_volume()
            for abx_name in self.antibiotic_names
        }
        self.current_patients = self.patient_generator.sample(
            n_patients=self.num_patients_per_time_step,
            true_amr_levels=true_amr_levels,
            rng=self.np_random,
        )
        
        # Construct initial observation from patients and AMR levels
        obs = self._construct_observation_from_patients(self.current_patients)
        if self.include_steps_since_amr_update_in_obs:
            obs = np.append(obs, self.steps_since_amr_update)
        self.state = obs

        # Return the initial AMR levels in the info dict for transparency
        info = {"visible_amr_levels": self.visible_amr_levels}
        
        return self.state, info

    # Actually, let's write a helper function that takes care of updating the visible AMR levels, because it will get reused across multiple functions, so I don't have to keep rewriting it.
    def update_visible_amr_levels(self, called_from_step: bool = False, force_update_for_reset: bool = False):
        """Update the visible AMR levels based on current balloon volumes; also account for the noise/bias settings, and also uses self.steps_since_amr_update to determine whether to update or not. The boolean 'called_from_step' indicates whether this function is being called from the step() function, in which case we need to increment the steps_since_amr_update counter. The boolean 'force_update_for_reset' indicates whether this function is being called from reset(), in which case we want to force an update regardless of the counter."""
        
        if called_from_step:
            # Update the counter, but do not change the visible AMR levels yet.
            self.steps_since_amr_update += 1
            
        if self.steps_since_amr_update >= self.update_visible_AMR_levels_every_n_timesteps or force_update_for_reset:
            # Otherwise, if it's time to update the visible AMR levels, iterate through each antibiotic and update the visible AMR levels, and add noise/bias if specified.
            visible_amr_levels = {}
            for abx_name in self.antibiotic_names:
                visible_amr_levels[abx_name] = self.amr_balloon_models[abx_name].get_volume()
                # Apply noise/bias if specified
                if self.add_noise_to_visible_AMR_levels > 0:
                    noise = self.np_random.normal(0, self.add_noise_to_visible_AMR_levels)
                    visible_amr_levels[abx_name] += noise
                if self.add_bias_to_visible_AMR_levels != 0:
                    visible_amr_levels[abx_name] += self.add_bias_to_visible_AMR_levels
                    
                # Set a floor and ceiling of 0 and 1 for visible AMR levels
                visible_amr_levels[abx_name] = min(max(visible_amr_levels[abx_name], 0.0), 1.0)
                
            self.visible_amr_levels = visible_amr_levels
            self.steps_since_amr_update = 0
    
    def _get_patient_attribute_value(self, patient, attr_name: str) -> float:
        """
        Extract a single attribute value from a Patient object.
        
        For multipliers, uses the observed (_obs) version which already has bias/noise applied.
        For prob_infected, uses the value directly (no _obs version exists).
        
        Args:
            patient: Patient object from PatientGenerator
            attr_name: Name of the attribute (e.g., 'prob_infected', 'benefit_value_multiplier')
        
        Returns:
            float: The attribute value
        
        Raises:
            ValueError: If attr_name is not a valid patient attribute
        """
        # Map attribute names to Patient object attributes
        # For multipliers, use _obs versions (bias/noise already applied by PatientGenerator)
        attr_map = {
            'prob_infected': 'prob_infected_obs',
            'benefit_value_multiplier': 'benefit_value_multiplier_obs',
            'failure_value_multiplier': 'failure_value_multiplier_obs',
            'benefit_probability_multiplier': 'benefit_probability_multiplier_obs',
            'failure_probability_multiplier': 'failure_probability_multiplier_obs',
            'recovery_without_treatment_prob': 'recovery_without_treatment_prob_obs',
        }
        
        if attr_name not in attr_map:
            valid_attrs = list(attr_map.keys())
            raise ValueError(f"Invalid attribute name '{attr_name}'. Valid attributes: {valid_attrs}")
        
        patient_attr = attr_map[attr_name]
        return getattr(patient, patient_attr)
    
    def _construct_observation_from_patients(self, patients) -> np.ndarray:
        """
        Construct observation vector from a list of Patient objects and current AMR levels.
        
        Observation structure:
            [patient_0_attr_0, patient_0_attr_1, ..., patient_0_attr_M,
             patient_1_attr_0, patient_1_attr_1, ..., patient_1_attr_M,
             ...
             patient_N_attr_0, patient_N_attr_1, ..., patient_N_attr_M,
             amr_level_0, amr_level_1, ..., amr_level_K]
        
        Args:
            patients: List of Patient objects from PatientGenerator.sample()
        
        Returns:
            np.ndarray: Observation vector with shape determined by
                (num_patients * len(visible_attributes) + num_abx,)
        """
        # Extract observed attributes from PatientGenerator in configured order
        if not hasattr(self.patient_generator, 'observe'):
            raise ValueError(
                "PatientGenerator must implement observe(patients) method. "
                "Ensure your PatientGenerator inherits from PatientGeneratorBase."
            )
        patient_features_array = self.patient_generator.observe(patients)
        
        # Get visible AMR levels in order of antibiotic_names
        amr_features = [self.visible_amr_levels[abx_name] for abx_name in self.antibiotic_names]
        
        # Build base observation
        obs_components = [
            np.array(patient_features_array, dtype=np.float32),
            np.array(amr_features, dtype=np.float32)
        ]
        
        # Append temporal features if enabled
        if self.enable_temporal_features:
            temporal_features = self._get_temporal_features()
            obs_components.append(temporal_features)
        
        # Concatenate all observation components
        observation = np.concatenate(obs_components)
        
        return observation
    
    def _get_temporal_features(self) -> np.ndarray:
        """
        Extract temporal features: prescription counts and AMR deltas.
        
        Returns:
            np.ndarray: Temporal feature vector with shape (num_abx * num_windows + num_abx,)
                Structure: [prescriptions_A_window1, prescriptions_A_window2, ...,
                            prescriptions_B_window1, prescriptions_B_window2, ...,
                            delta_AMR_A, delta_AMR_B, ...]
                Prescription counts are normalized by window size to be in [0, 1].
        """
        features = []
        
        # Prescription counts per antibiotic per window (normalized)
        for abx_name in self.antibiotic_names:
            for window_idx, window_size in enumerate(self.temporal_windows):
                count = sum(self.prescription_history[abx_name][window_idx])
                normalized_count = count / window_size  # Normalize to [0, 1]
                features.append(normalized_count)
        
        # AMR deltas per antibiotic
        for abx_name in self.antibiotic_names:
            delta = self.visible_amr_levels[abx_name] - self.previous_visible_amr_levels[abx_name]
            features.append(delta)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_patient_stats(self, patients) -> dict:
        """
        Compute aggregate statistics for patient attributes (true and observed values).
        
        Used for lightweight logging during training to track distribution shifts and
        observation error without storing full patient trajectories.
        
        Args:
            patients: List of Patient objects
            
        Returns:
            dict: Statistics including mean, std, min, max for each attribute (true and observed)
                  plus observation errors (mean absolute difference between true and observed)
        """
        if not patients:
            return {}
        
        # Extract all patient attributes
        attrs_true = {
            'prob_infected': np.array([p.prob_infected for p in patients]),
            'benefit_value_multiplier': np.array([p.benefit_value_multiplier for p in patients]),
            'failure_value_multiplier': np.array([p.failure_value_multiplier for p in patients]),
            'benefit_probability_multiplier': np.array([p.benefit_probability_multiplier for p in patients]),
            'failure_probability_multiplier': np.array([p.failure_probability_multiplier for p in patients]),
            'recovery_without_treatment_prob': np.array([p.recovery_without_treatment_prob for p in patients]),
        }
        
        attrs_obs = {
            'prob_infected': np.array([p.prob_infected_obs for p in patients]),
            'benefit_value_multiplier': np.array([p.benefit_value_multiplier_obs for p in patients]),
            'failure_value_multiplier': np.array([p.failure_value_multiplier_obs for p in patients]),
            'benefit_probability_multiplier': np.array([p.benefit_probability_multiplier_obs for p in patients]),
            'failure_probability_multiplier': np.array([p.failure_probability_multiplier_obs for p in patients]),
            'recovery_without_treatment_prob': np.array([p.recovery_without_treatment_prob_obs for p in patients]),
        }
        
        stats = {}
        
        # Compute stats for each attribute
        for attr_name in attrs_true.keys():
            true_vals = attrs_true[attr_name]
            obs_vals = attrs_obs[attr_name]
            
            # True value stats
            stats[f'{attr_name}_true_mean'] = float(np.mean(true_vals))
            stats[f'{attr_name}_true_std'] = float(np.std(true_vals))
            stats[f'{attr_name}_true_min'] = float(np.min(true_vals))
            stats[f'{attr_name}_true_max'] = float(np.max(true_vals))
            
            # Observed value stats
            stats[f'{attr_name}_obs_mean'] = float(np.mean(obs_vals))
            stats[f'{attr_name}_obs_std'] = float(np.std(obs_vals))
            stats[f'{attr_name}_obs_min'] = float(np.min(obs_vals))
            stats[f'{attr_name}_obs_max'] = float(np.max(obs_vals))
            
            # Observation error (absolute difference)
            abs_error = np.abs(true_vals - obs_vals)
            stats[f'{attr_name}_obs_error_mean'] = float(np.mean(abs_error))
            stats[f'{attr_name}_obs_error_max'] = float(np.max(abs_error))
        
        return stats
    
    def _extract_full_patient_attributes(self, patients) -> dict:
        """
        Extract full patient attribute arrays for detailed logging during eval episodes.
        
        Returns both true and observed values for all patient attributes.
        Used when self.log_full_patient_attributes is True.
        
        Args:
            patients: List of Patient objects
            
        Returns:
            dict: Contains 'true' and 'observed' sub-dicts, each mapping attribute names
                  to lists of values for all patients in this timestep
        """
        if not patients:
            return {'true': {}, 'observed': {}}
        
        true_attrs = {
            'prob_infected': [float(p.prob_infected) for p in patients],
            'benefit_value_multiplier': [float(p.benefit_value_multiplier) for p in patients],
            'failure_value_multiplier': [float(p.failure_value_multiplier) for p in patients],
            'benefit_probability_multiplier': [float(p.benefit_probability_multiplier) for p in patients],
            'failure_probability_multiplier': [float(p.failure_probability_multiplier) for p in patients],
            'recovery_without_treatment_prob': [float(p.recovery_without_treatment_prob) for p in patients],
        }
        
        obs_attrs = {
            'prob_infected': [float(p.prob_infected_obs) for p in patients],
            'benefit_value_multiplier': [float(p.benefit_value_multiplier_obs) for p in patients],
            'failure_value_multiplier': [float(p.failure_value_multiplier_obs) for p in patients],
            'benefit_probability_multiplier': [float(p.benefit_probability_multiplier_obs) for p in patients],
            'failure_probability_multiplier': [float(p.failure_probability_multiplier_obs) for p in patients],
            'recovery_without_treatment_prob': [float(p.recovery_without_treatment_prob_obs) for p in patients],
        }
        
        return {
            'true': true_attrs,
            'observed': obs_attrs,
        }
    
    def step(self, action):
        """Execute one timestep in the environment.
        
        Applies antibiotic prescriptions to current patient cohort, computes effective
        doses (including crossresistance effects), updates AMR leaky balloons, calculates
        rewards, and returns next observation. Terminates when max_time_steps reached.
        
        Execution flow:
        1. Process per-patient prescriptions (mutually exclusive: one antibiotic OR no treatment)
        2. Count prescriptions per antibiotic
        3. Apply crossresistance matrix to compute effective doses
        4. Step each antibiotic's leaky balloon with effective puff
        5. Calculate reward (individual + community components)
        6. Sample new patient cohort for next timestep
        7. Construct observation (patient features + AMR levels)
        
        Args:
            action (np.ndarray): Action array, shape (num_patients_per_time_step,).
                Each value is an integer representing which antibiotic to prescribe:
                - 0: No treatment
                - 1: First antibiotic (e.g., Antibiotic_A)
                - 2: Second antibiotic (e.g., Antibiotic_B)
                - ...
                Note: Actions are mutually exclusive - only one antibiotic per patient.
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info) where:
                - observation (np.ndarray): Next state vector, shape = obs_space.shape
                - reward (float): Composite reward for this step
                - terminated (bool): True if episode naturally ended (always False here)
                - truncated (bool): True if max_time_steps reached
                - info (dict): Contains keys:\n                    'prescriptions_per_abx', 'effective_doses', 'crossresistance_applied',\n                    'actual_amr_levels', 'visible_amr_levels', 'delta_amr_per_antibiotic',\n                    'patient_stats', 'patient_outcomes', 'pts_w_clinical_benefits',\n                    'pts_w_clinical_failures', 'pts_w_adverse_events', 'total_reward',\n                    'individual_reward', 'community_reward', 'mean_individual_reward'
        
        Example:
            >>> obs, reward, terminated, truncated, info = env.step(action)
            >>> print(f\"AMR levels: {info['actual_amr_levels']}\")
            >>> print(f\"Reward: {reward:.3f} (individual: {info['individual_reward']:.3f})\")
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Action is already per-patient MultiDiscrete with mutually exclusive choices
        # No decoding needed - action[i] directly maps to antibiotic index for patient i
        multi_action = np.array(action, dtype=int)

        # Extract current patients that the action applies to
        current_patients_for_reward = self.current_patients
        
        # Count prescriptions per antibiotic (for updating AMR balloons)
        # action values 0 to (num_abx-1) represent prescribing that antibiotic (by index in antibiotic_names)
        # action value self.no_treatment_action (num_abx) represents no prescription
        antibiotic_prescription_counts = {name: 0 for name in self.antibiotic_names}
        for abx_prescription_action in multi_action:
            abx_name = self.reward_calculator.index_to_abx_name[abx_prescription_action]
            # Only count if it's not 'no_treatment'
            if abx_name != 'no_treatment':
                antibiotic_prescription_counts[abx_name] += 1
        
        # Compute effective doses for each antibiotic considering crossresistance
        effective_doses = {}
        crossresistance_applied = {}
        for target_abx in self.antibiotic_names:
            total_doses = 0.0
            crossresistance_applied[target_abx] = {}
            for prescriber_abx in self.antibiotic_names:
                prescriber_count = antibiotic_prescription_counts[prescriber_abx]
                crossresistance_ratio = self.crossresistance_matrix[prescriber_abx][target_abx]
                contribution = prescriber_count * crossresistance_ratio
                total_doses += contribution
                if contribution > 0:
                    crossresistance_applied[target_abx][prescriber_abx] = contribution
            effective_doses[target_abx] = total_doses

        # Update AMR balloon models based on effective doses from crossresistance
        for abx_name in self.antibiotic_names:
            self.amr_balloon_models[abx_name].step(effective_doses[abx_name])
        
        # Store previous visible AMR levels BEFORE updating (for temporal delta calculation)
        if self.enable_temporal_features:
            for abx_name in self.antibiotic_names:
                self.previous_visible_amr_levels[abx_name] = self.visible_amr_levels[abx_name]
        
        # Update visible AMR levels; call self.update_visible_amr_levels with 'called_from_step=True' to increment the counter
        self.update_visible_amr_levels(called_from_step=True)
        
        # Update temporal feature tracking if enabled (AFTER visible AMR update)
        if self.enable_temporal_features:
            # Update prescription history (1 if prescribed this step, 0 otherwise)
            for abx_name in self.antibiotic_names:
                prescribed_this_step = 1 if antibiotic_prescription_counts[abx_name] > 0 else 0
                for deque_window in self.prescription_history[abx_name]:
                    deque_window.append(prescribed_this_step)
        
        # Pre-compute marginal AMR contribution per antibiotic using effective doses
        delta_visible_amr_per_antibiotic = {}
        for abx_name in self.antibiotic_names:
            model = self.amr_balloon_models[abx_name]
            
            # Have to make a copy of the model state, then set the volume to the current visible AMR level, then compute delta volume for counterfactual doses
            model_copy = model.copy()
            model_copy.reset(initial_amr_level=self.visible_amr_levels[abx_name])
            
            # Compute volume with actual number of doses, then with one less dose.
            delta_visible_amr = model_copy.get_delta_volume_for_counterfactual_num_doses_vs_one_less(effective_doses[abx_name])
            delta_visible_amr_per_antibiotic[abx_name] = delta_visible_amr
        
        # Increment time step
        self.current_time_step += 1
        
        # Sample new patient cohort for next time step
        true_amr_levels = {
            abx_name: self.amr_balloon_models[abx_name].get_volume()
            for abx_name in self.antibiotic_names
        }
        self.current_patients = self.patient_generator.sample(
            n_patients=self.num_patients_per_time_step,
            true_amr_levels=true_amr_levels,
            rng=self.np_random,
        )
        
        # Construct next observation from new patients and updated AMR levels
        obs = self._construct_observation_from_patients(self.current_patients)
        if self.include_steps_since_amr_update_in_obs:
            obs = np.append(obs, self.steps_since_amr_update)
        self.state = obs
        
        # Compute reward via RewardCalculator
        reward, reward_info = self.reward_calculator.calculate_reward(
            patients=current_patients_for_reward,
            actions=multi_action,
            antibiotic_names=self.antibiotic_names,
            visible_amr_levels=self.visible_amr_levels,
            delta_visible_amr_per_antibiotic=delta_visible_amr_per_antibiotic,
            rng=self.np_random,
        )
        
        # Check termination conditions; the condition for termination and truncation is the same, it's if we reached max_time_steps. I might want to change this in the future to have different conditions for termination vs truncation.
        terminated = self.current_time_step >= self.max_time_steps
        truncated = self.current_time_step >= self.max_time_steps
        
        # Collect patient data for logging
        # Always compute aggregate stats (lightweight, ~36 scalars per step)
        patient_stats = self._compute_patient_stats(current_patients_for_reward)
        
        # Optionally collect full patient attributes (used during eval episodes)
        patient_full_data = None
        if self.log_full_patient_attributes:
            patient_full_data = self._extract_full_patient_attributes(current_patients_for_reward)
        
        # Info dictionary with useful debugging information
        info = {
            'current_time_step': self.current_time_step,
            'actual_amr_levels': {abx_name: self.amr_balloon_models[abx_name].get_volume() for abx_name in self.antibiotic_names},
            'visible_amr_levels': self.visible_amr_levels.copy(),
            'prescriptions_per_abx': antibiotic_prescription_counts,
            'delta_visible_amr_per_antibiotic': delta_visible_amr_per_antibiotic,
            'effective_doses': effective_doses,
            'crossresistance_applied': crossresistance_applied,
            'patient_stats': patient_stats,
        }
        
        # Add full patient data if logging is enabled
        if patient_full_data is not None:
            info['patient_full_data'] = patient_full_data
        
        info.update(reward_info)
        
        return self.state, reward, terminated, truncated, info

    def render(self):
        """Render environment state to console for debugging.
        
        Prints current timestep, antibiotic names, true AMR levels (from balloon models),
        visible AMR levels (observed by agent, possibly delayed/noisy), current observation
        shape, and patient infection probabilities.
        
        This is primarily useful for debugging and manual inspection of environment state.
        For automated monitoring during training, use TensorBoard logs or callbacks.
        
        Example output:
            Time step: 42/1000
            Antibiotic names: ['penicillin', 'amoxicillin']
            AMR levels: {penicillin: 0.235, amoxicillin: 0.189}
            Visible AMR levels: {penicillin: 0.220, amoxicillin: 0.180}
            Current state shape: (26,)
            Patient infection probs: [0.45 0.62 0.33 ...]
        """
        print(f"Time step: {self.current_time_step}/{self.max_time_steps}")
        print(f"Antibiotic names: {self.antibiotic_names}")
        amr_str = ', '.join([f'{name}: {self.amr_balloon_models[name].get_volume():.3f}' for name in self.antibiotic_names])
        print(f"AMR levels: {{{amr_str}}}")
        visible_str = ', '.join([f'{name}: {self.visible_amr_levels[name]:.3f}' for name in self.antibiotic_names])
        print(f"Visible AMR levels: {{{visible_str}}}")
        if self.state is not None:
            print(f"Current state shape: {self.state.shape}")
            print(f"Patient infection probs: {self.state[:self.num_patients_per_time_step]}")
    
    def get_action_to_antibiotic_mapping(self) -> dict:
        """Returns mapping of action indices to antibiotic names.
        
        Returns:
            dict: Maps action indices to antibiotic names. Includes 'no_treatment' entry
                  for the no-treatment action.
                  E.g., {0: 'Antibiotic_0', 1: 'Antibiotic_1', 2: 'no_treatment'}
        """
        # Just return the existing mapping from self.reward_calculator:
        return self.reward_calculator.index_to_abx_name
    
    def get_antibiotic_to_action_mapping(self) -> dict:
        """Returns mapping of antibiotic names to action indices.
        
        Returns:
            dict: Maps antibiotic names to action indices. Includes 'no_treatment' entry
                  for the no-treatment action.
                  E.g., {'Antibiotic_0': 0, 'Antibiotic_1': 1, 'no_treatment': 2}
        """
        # Reverse the existing mapping from self.reward_calculator:
        return self.reward_calculator.abx_name_to_index
    
    def _is_valid_action_index(self, action_idx: int) -> bool:
        """Check if action index is valid (real antibiotic or no-treatment).
        
        Args:
            action_idx: Action index to validate
            
        Returns:
            bool: True if action_idx is valid, False otherwise
        """
        return action_idx in self.reward_calculator.index_to_abx_name
