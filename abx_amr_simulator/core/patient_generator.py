"""
PatientGenerator for sampling heterogeneous patient populations.

This module provides functionality to generate diverse patient populations
with configurable distributions for infection probability, treatment benefit,
failure risk, and recovery probability. Supports applying observation-level
noise and bias to patient attributes for robustness experiments.

Nested Configuration Format:
Each of 6 patient attributes (prob_infected, benefit_value_multiplier, etc.) is
configured independently with:
  - prob_dist: Distribution specification (type: 'constant' or 'gaussian')
  - obs_bias_multiplier: Multiplicative bias (1.0 = unbiased)
  - obs_noise_one_std_dev: Reference magnitude for noise (user-specified)
  - obs_noise_std_dev_fraction: Fraction of reference to use as std dev (unitless)
  - clipping_bounds: [lower, upper] or [lower, None] for post-noise clipping

Observation Model (applied in sequence):
1. **Bias** (multiplicative, unitless): Scales the true value by a constant factor.
   Example: `obs_bias_multiplier=1.2` → obs_value = true_value × 1.2
   
2. **Noise** (additive, reference-magnitude based): Adds Gaussian noise AFTER bias.
   - `effective_std = obs_noise_one_std_dev × obs_noise_std_dev_fraction`
   - Example: `obs_noise_one_std_dev=0.2, obs_noise_std_dev_fraction=0.5` → effective_std=0.1
   - Example: `obs_value = 0.77 + N(0, 0.1)` (noise added to biased value)
   - Noise is skipped if obs_noise_std_dev_fraction ≤ 0 or obs_noise_one_std_dev ≤ 0

3. **Clipping** (post-noise): Clips to user-specified clipping_bounds
   - Example: `clipping_bounds=[0.0, 1.0]` ensures probability stays in [0,1]
   - None in either position means unbounded (e.g., [0.0, None] for multipliers)

Design Rationale:
- **Modularity**: Bias and noise are independent. Users can model systematic error
  (bias) separately from measurement uncertainty (noise).
- **Interpretability**: obs_noise_one_std_dev is set once per attribute (user-specified
  reference); obs_noise_std_dev_fraction is a portable unitless multiplier.
- **Flexibility**: User-specified clipping_bounds replace hardcoded ranges.
"""

import numpy as np
from typing import List, Dict, Optional, Any, ClassVar
from .types import Patient
from .base_patient_generator import PatientGeneratorBase


class PatientGenerator(PatientGeneratorBase):
    """
    Generates heterogeneous patient populations with configurable distributions.
    
    Uses nested per-attribute configuration format where each of 6 patient attributes
    (prob_infected, benefit_value_multiplier, etc.) specifies its own distribution,
    observation bias, noise, and clipping. Supports sampling from 'constant' and
    'gaussian' distributions. Applies observation-level multiplicative bias and
    reference-magnitude additive noise to simulate imperfect patient risk assessment.
    
    Configuration Format (nested per-attribute):
    {
        'prob_infected': {
            'prob_dist': {'type': 'constant', 'value': 0.8},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.2,      # Reference magnitude
            'obs_noise_std_dev_fraction': 0.5, # Unitless fraction
            'clipping_bounds': [0.0, 1.0],
        },
        'visible_patient_attributes': ['prob_infected', ...]
    }
    
    For homogeneous populations, use 'constant' distribution with appropriate value parameter.
    For heterogeneous populations, use 'gaussian' with mu and sigma parameters.
    
    OWNERSHIP (for future multi-agent refactor):
    - Owns: patient attribute distributions, visibility configuration, observation extraction
    - Returns: List[Patient] with true + observed attributes
    - Provides: observe(patients) -> np.ndarray, obs_dim(num_patients) -> int
    
    Future multi-agent notes:
    - Will be instantiated per-Locale (each locale can have different population characteristics)
    - Patient objects will include patient_id, origin_locale fields for tracking
    - Travel mechanics will reassign patients across locales while preserving origin AMR profile
    
    Attributes:
        config: Dictionary with nested per-attribute configuration
        visible_patient_attributes: List of attributes to include in observations
        attribute_configs: Dict mapping attribute name -> per-attribute config
    
    Class Methods:
        default_config(): Get a template configuration with sensible defaults
    """
    
    # Declare all attributes that generated Patient objects will have
    PROVIDES_ATTRIBUTES: ClassVar[List[str]] = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob',
        'infection_status',
        'abx_sensitivity_dict',
        'prob_infected_obs',
        'benefit_value_multiplier_obs',
        'failure_value_multiplier_obs',
        'benefit_probability_multiplier_obs',
        'failure_probability_multiplier_obs',
        'recovery_without_treatment_prob_obs',
    ]
    
    # Known attribute types for validation and inference
    # Maps attribute name → semantic type (probability or multiplier)
    KNOWN_ATTRIBUTE_TYPES: ClassVar[Dict[str, str]] = {
        'prob_infected': 'probability',
        'benefit_value_multiplier': 'multiplier',
        'failure_value_multiplier': 'multiplier',
        'benefit_probability_multiplier': 'multiplier',
        'failure_probability_multiplier': 'multiplier',
        'recovery_without_treatment_prob': 'probability',
    }
    
    # Validation rules and defaults per attribute type
    ATTRIBUTE_TYPE_VALIDATION: ClassVar[Dict[str, Dict[str, Any]]] = {
        'probability': {
            'default_bounds': [0.0, 1.0],
            'description': 'Probability attribute (value ∈ [0,1])',
        },
        'multiplier': {
            'default_bounds': [0.0, None],
            'description': 'Multiplier attribute (typically ≥ 0)',
        },
    }
    
    # Required configuration keys for initialization (nested per-attribute format)
    REQUIRED_CONFIG_KEYS: ClassVar[List[str]] = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob',
        'visible_patient_attributes',
    ]
    
    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """
        Return a template configuration with sensible defaults.
        
        Uses the new nested per-attribute configuration format.
        
        Returns:
            Dictionary with all required attributes configured with sensible defaults.
        
        Example:
            config = PatientGenerator.default_config()
            config['prob_infected']['prob_dist']['value'] = 0.9  # Modify as needed
            gen = PatientGenerator(config)
        """
        return {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.8, 'mu': None, 'sigma': None},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0, 'mu': None, 'sigma': None},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0, 'mu': None, 'sigma': None},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0, 'mu': None, 'sigma': None},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0, 'mu': None, 'sigma': None},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.0, 'mu': None, 'sigma': None},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """
        Initialize PatientGenerator from nested per-attribute config dictionary.
        
        Args:
            config: Dictionary with nested per-attribute configuration. New format:
                {
                    'prob_infected': {
                        'prob_dist': {'type': 'constant', 'value': 0.8, 'mu': None, 'sigma': None},
                        'obs_bias_multiplier': 1.0,
                        'obs_noise_one_std_dev': 0.2,
                        'obs_noise_std_dev_fraction': 0.5,
                        'clipping_bounds': [0.0, 1.0],
                    },
                    ...
                    'visible_patient_attributes': ['prob_infected', ...]
                }
        
        Raises:
            ValueError: If required config keys are missing or invalid
        
        Example:
            config = PatientGenerator.default_config()
            config['prob_infected']['prob_dist']['value'] = 0.9
            gen = PatientGenerator(config)
        """
        self.config = config
        
        # Validate and extract visible_patient_attributes (REQUIRED)
        if 'visible_patient_attributes' not in config:
            raise ValueError(
                "Missing required config key: 'visible_patient_attributes'. "
                "Use default_config() as a template."
            )
        self.visible_patient_attributes: List[str] = config['visible_patient_attributes']
        
        # Validate each visible attribute exists and is valid
        for attr in self.visible_patient_attributes:
            if attr not in config:
                raise ValueError(
                    f"visible_patient_attributes lists '{attr}', but no config exists for it. "
                    f"Add '{attr}' as a top-level key in config."
                )
            if attr not in self.KNOWN_ATTRIBUTE_TYPES:
                raise ValueError(
                    f"Unknown attribute '{attr}'. Known attributes: "
                    f"{sorted(self.KNOWN_ATTRIBUTE_TYPES.keys())}"
                )
        
        # Parse and validate each attribute's configuration
        self.attribute_configs: Dict[str, Dict[str, Any]] = {}
        for attr_name in self.KNOWN_ATTRIBUTE_TYPES:
            if attr_name not in config:
                continue  # Attribute not configured (OK if not in visible_patient_attributes)
            
            attr_config = config[attr_name]
            self._validate_attribute_config(attr_name, attr_config)
            self.attribute_configs[attr_name] = attr_config
        
        # Optional seed/rng for external synchronization
        self.seed: Optional[int] = None
        self.rng: Optional[np.random.Generator] = None
        
        # Ownership lifecycle: start in standalone mode
        self._standalone: bool = True
    
    def _set_environment_owned(self):
        """Mark this component as environment-owned (one-way transition).
        
        After calling this method, the component will require explicit rng argument
        in sample() and will fail loudly if not provided.
        This enforces that environment-owned components receive RNG from environment.
        
        Called by ABXAMREnv.__init__ after accepting this component as a parameter.
        """
        self._standalone = False
    
    def _validate_attribute_config(self, attr_name: str, attr_config: Dict[str, Any]) -> None:
        """
        Validate configuration for a single attribute.
        
        Args:
            attr_name: Name of the attribute
            attr_config: Configuration dict for this attribute
        
        Raises:
            ValueError: If config is invalid
        """
        # Check required keys
        required_keys = {'prob_dist', 'obs_bias_multiplier', 'obs_noise_one_std_dev', 
                        'obs_noise_std_dev_fraction', 'clipping_bounds'}
        missing_keys = required_keys - set(attr_config.keys())
        if missing_keys:
            raise ValueError(
                f"Attribute '{attr_name}' config missing required keys: {sorted(missing_keys)}"
            )
        
        # Validate prob_dist
        prob_dist = attr_config.get('prob_dist', {})
        if 'type' not in prob_dist:
            raise ValueError(f"Missing 'type' in {attr_name}.prob_dist")
        
        dist_type = prob_dist['type']
        if dist_type not in ['constant', 'gaussian']:
            raise ValueError(
                f"Unknown distribution type '{dist_type}' in {attr_name}.prob_dist. "
                f"Must be 'constant' or 'gaussian'."
            )
        
        # Validate distribution parameters
        if dist_type == 'constant':
            if 'value' not in prob_dist or prob_dist['value'] is None:
                raise ValueError(f"{attr_name}.prob_dist type is 'constant' but 'value' is missing")
        elif dist_type == 'gaussian':
            if 'mu' not in prob_dist or prob_dist['mu'] is None:
                raise ValueError(f"{attr_name}.prob_dist type is 'gaussian' but 'mu' is missing")
            if 'sigma' not in prob_dist or prob_dist['sigma'] is None:
                raise ValueError(f"{attr_name}.prob_dist type is 'gaussian' but 'sigma' is missing")
            if prob_dist['sigma'] < 0:
                raise ValueError(f"{attr_name}.prob_dist sigma must be non-negative")
        
        # Validate observation parameters
        obs_bias = attr_config.get('obs_bias_multiplier', 1.0)
        if obs_bias <= 0:
            raise ValueError(f"{attr_name}.obs_bias_multiplier must be > 0")
        
        obs_noise_std_dev = attr_config.get('obs_noise_one_std_dev', 0.0)
        if obs_noise_std_dev < 0:
            raise ValueError(f"{attr_name}.obs_noise_one_std_dev must be >= 0")
        
        obs_noise_frac = attr_config.get('obs_noise_std_dev_fraction', 0.0)
        if obs_noise_frac < 0:
            raise ValueError(f"{attr_name}.obs_noise_std_dev_fraction must be >= 0")
        
        # Validate clipping_bounds
        bounds = attr_config.get('clipping_bounds', [0.0, 1.0])
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"{attr_name}.clipping_bounds must be a 2-element list/tuple")
        
        lower, upper = bounds
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"{attr_name}.clipping_bounds: lower ({lower}) must be <= upper ({upper})"
            )
    
    def _obs_attr_name(self, attr: str) -> str:
        """Map true attribute name to its observed counterpart (*_obs)."""
        mapping = {
            'prob_infected': 'prob_infected_obs',
            'benefit_value_multiplier': 'benefit_value_multiplier_obs',
            'failure_value_multiplier': 'failure_value_multiplier_obs',
            'benefit_probability_multiplier': 'benefit_probability_multiplier_obs',
            'failure_probability_multiplier': 'failure_probability_multiplier_obs',
            'recovery_without_treatment_prob': 'recovery_without_treatment_prob_obs',
        }
        return mapping[attr]

    def sample(
        self,
        n_patients: int,
        true_amr_levels: Dict[str, float],
        rng: Optional[np.random.Generator] = None,
    ) -> List[Patient]:
        """Sample n_patients from configured distributions with observation noise/bias.
        
        Generates heterogeneous patient population by sampling true attribute values from
        configured distributions (constant or gaussian), then applies observation bias and
        noise to simulate imperfect risk assessment. Each Patient dataclass contains both
        true values (used internally for reward calculation) and observed values (used for
        agent observation).
        
        Args:
            n_patients (int): Number of patients to sample. Typically equals
                environment's num_patients_per_time_step.
            true_amr_levels (Dict[str, float]): Ground-truth AMR levels per antibiotic,
                used to sample infection sensitivity per patient.
            rng (np.random.Generator): NumPy random generator for reproducible sampling.
                Should be the shared RNG from environment/reward_calculator to maintain
                synchronized stochastic draws across components.
        
        Returns:
            List[Patient]: List of Patient dataclass instances, each containing:
                - True attributes (prob_infected, benefit_value_multiplier, etc.)
                - Observed attributes (*_obs versions) with bias/noise applied
                - Optional tracking fields (patient_id, etc., unused in single-agent mode)
        
        Raises:
            ValueError: If n_patients <= 0 or true_amr_levels is missing/empty.
        
        Example:
            >>> from abx_amr_simulator.core import PatientGenerator
            >>> import numpy as np
            >>> config = PatientGenerator.default_config()
            >>> config['prob_infected_dist'] = {'type': 'gaussian', 'mu': 0.5, 'sigma': 0.1}
            >>> gen = PatientGenerator(config)
            >>> rng = np.random.default_rng(42)
            >>> true_amr_levels = {'A': 0.2, 'B': 0.4}
            >>> patients = gen.sample(n_patients=10, true_amr_levels=true_amr_levels, rng=rng)
            >>> print(patients[0].prob_infected, patients[0].prob_infected_obs)
            0.523 0.540  # Observed value has bias/noise applied
        """
        if n_patients <= 0:
            raise ValueError(f"n_patients must be > 0, got {n_patients}")
        if not true_amr_levels:
            raise ValueError("true_amr_levels must be provided and non-empty")

        # RNG resolution: enforce explicit RNG when environment-owned
        if not self._standalone and rng is None:
            raise ValueError(
                "PatientGenerator is environment-owned and requires explicit rng argument. "
                "Pass rng=env.np_random when calling from outside environment."
            )
        rng = rng if rng is not None else self.rng
        if rng is None:
            raise ValueError("RNG must be provided either via argument or initialized on PatientGenerator")
        
        patients = []
        for _ in range(n_patients):
            patient_kwargs = {}
            
            # Iterate through all known attributes
            for attr_name in self.KNOWN_ATTRIBUTE_TYPES:
                if attr_name not in self.attribute_configs:
                    continue  # Skip unconfigured attributes
                
                attr_cfg = self.attribute_configs[attr_name]
                
                # Get sampling bounds based on attribute type
                attr_type = self.KNOWN_ATTRIBUTE_TYPES[attr_name]
                sampling_bounds = self.ATTRIBUTE_TYPE_VALIDATION[attr_type]['default_bounds']
                
                # Sample true value from the distribution
                true_value = self._sample_from_dist(
                    rng=rng,
                    dist_config=attr_cfg['prob_dist'],
                    bounds=tuple(sampling_bounds) if sampling_bounds else None,
                )
                
                # Apply observation transformation (bias + noise + clipping)
                obs_value = self._apply_multip_bias_and_additive_noise(
                    true_value=true_value,
                    obs_bias_multiplier=attr_cfg['obs_bias_multiplier'],
                    obs_noise_one_std_dev=attr_cfg['obs_noise_one_std_dev'],
                    obs_noise_std_dev_fraction=attr_cfg['obs_noise_std_dev_fraction'],
                    clipping_bounds=tuple(attr_cfg['clipping_bounds']) if attr_cfg['clipping_bounds'] else None,
                    rng=rng,
                )
                
                # Store both true and observed values
                patient_kwargs[attr_name] = true_value
                patient_kwargs[f'{attr_name}_obs'] = obs_value
            
            infection_prob = float(patient_kwargs['prob_infected'])
            patient_kwargs['infection_status'] = bool(rng.random() < infection_prob)

            abx_sensitivity_dict: Dict[str, bool] = {}
            for abx_name, amr_level in true_amr_levels.items():
                amr_value = float(amr_level)
                if not 0.0 <= amr_value <= 1.0:
                    raise ValueError(
                        f"true_amr_levels['{abx_name}'] must be in [0, 1], got {amr_value}"
                    )
                sensitivity_probability = 1.0 - amr_value
                abx_sensitivity_dict[abx_name] = bool(rng.random() < sensitivity_probability)
            patient_kwargs['abx_sensitivity_dict'] = abx_sensitivity_dict

            # Create Patient object dynamically with all sampled attributes
            patient = Patient(**patient_kwargs)
            patients.append(patient)
        
        return patients

    def observe(self, patients: List[Patient]) -> np.ndarray:
        """Extract observed patient attributes for agent observation.
        
        Constructs the patient observation component by extracting *_obs (observed)
        versions of attributes specified in visible_patient_attributes, flattened into
        a single 1D array. This array is concatenated with AMR levels in the environment
        to form the complete agent observation.
        
        Args:
            patients (List[Patient]): List of Patient objects (typically from sample()).
        
        Returns:
            np.ndarray: Flat array of shape (num_patients * len(visible_patient_attributes),)
                containing observed attribute values in order:
                [patient_0_attr_0, patient_0_attr_1, ..., patient_1_attr_0, ...]\n                where attr order matches visible_patient_attributes list.
        
        Example:
            >>> config = PatientGenerator.default_config()
            >>> config['visible_patient_attributes'] = ['prob_infected', 'benefit_value_multiplier']
            >>> gen = PatientGenerator(config)
            >>> rng = np.random.default_rng(42)
            >>> patients = gen.sample(n_patients=3, rng=rng)
            >>> obs = gen.observe(patients)
            >>> print(obs.shape)
            (6,)  # 3 patients * 2 visible attributes
            >>> print(obs)
            [0.8 1.0 0.8 1.0 0.8 1.0]  # [p0_infected, p0_benefit, p1_infected, p1_benefit, ...]
        """
        if patients is None:
            return np.array([], dtype=np.float32)
        num_patients = len(patients)
        if num_patients == 0:
            return np.array([], dtype=np.float32)

        obs_values = []
        for p in patients:
            for attr in self.visible_patient_attributes:
                obs_name = self._obs_attr_name(attr)
                obs_values.append(float(getattr(p, obs_name)))
        return np.array(obs_values, dtype=np.float32)

    def obs_dim(self, num_patients: int) -> int:
        """Compute patient observation dimension for environment observation space sizing.
        
        Calculates the dimension of the patient component of the environment observation
        (before AMR levels and optional steps_since_amr_update are appended). Used by
        ABXAMREnv during __init__ to construct the observation space.
        
        Args:
            num_patients (int): Number of patients per timestep (environment's
                num_patients_per_time_step parameter).
        
        Returns:
            int: Dimension of patient observation array returned by observe().
                Equal to num_patients * len(visible_patient_attributes).
        
        Example:
            >>> config = PatientGenerator.default_config()
            >>> config['visible_patient_attributes'] = ['prob_infected', 'benefit_value_multiplier']
            >>> gen = PatientGenerator(config)
            >>> print(gen.obs_dim(num_patients=10))
            20  # 10 patients * 2 visible attributes
        """
        num_attrs = len(self.visible_patient_attributes)
        return int(num_patients * num_attrs)
    
    def _sample_from_dist(
        self,
        rng: np.random.Generator,
        dist_config: Dict[str, Any],
        bounds: tuple = None,
    ) -> float:
        """
        Sample a single value from a specified distribution.
        
        Args:
            rng: NumPy random generator
            dist_config: Dictionary with 'type' and distribution parameters
            bounds: Optional (min, max) tuple to clip result. None means no bound.
            
        Returns:
            Sampled float value
            
        Raises:
            ValueError: If distribution type or parameters are invalid
        """
        dist_type = dist_config['type']
        
        if dist_type == 'constant':
            if 'value' not in dist_config:
                raise ValueError(
                    f"Constant distribution requires 'value' parameter"
                )
            value = dist_config['value']
        
        elif dist_type == 'gaussian':
            if 'mu' not in dist_config or 'sigma' not in dist_config:
                raise ValueError(
                    f"Gaussian distribution requires 'mu' and 'sigma' parameters"
                )
            mu = dist_config['mu']
            sigma = dist_config['sigma']
            if sigma < 0:
                raise ValueError(f"Gaussian sigma must be non-negative, got {sigma}")
            value = rng.normal(mu, sigma)
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
        # Apply bounds if specified
        if bounds is not None:
            lower, upper = bounds
            if lower is not None:
                value = max(value, lower)
            if upper is not None:
                value = min(value, upper)
        
        return float(value)
    
    def _apply_multip_bias_and_additive_noise(
        self,
        true_value: float,
        obs_bias_multiplier: float,
        obs_noise_one_std_dev: float,
        obs_noise_std_dev_fraction: float,
        clipping_bounds: tuple,
        rng: np.random.Generator,
    ) -> float:
        """
        Apply multiplicative bias and additive Gaussian noise to a true value.
        
        **Two-stage observation model:**
        1. Apply multiplicative bias: `obs_value = true_value × obs_bias_multiplier`
        2. Add Gaussian noise: `obs_value += N(0, obs_noise_std_dev_fraction × obs_noise_one_std_dev)`
        3. Clip to clipping_bounds: `obs_value = clip(obs_value, lower, upper)`
        
        **Bias** (multiplicative, unitless):
        - Example: `obs_bias_multiplier=1.2` means observations are systematically 20% higher
        - Represents systematic measurement error or assessment bias
        
        **Noise** (additive, with reference magnitude):
        - `obs_noise_one_std_dev`: Reference standard deviation magnitude
          - Example: "I know patient values range ~0-5, so 1 std = 1.0"
          - Defined once at config time; decoupled from sampling distribution
        - `obs_noise_std_dev_fraction`: Unitless multiplier applied to reference magnitude
          - Example: 0.5 means "use 50% of the reference"
          - `effective_std = obs_noise_std_dev_fraction × obs_noise_one_std_dev`
        
        **Why separate reference magnitude and fraction?**
        - **Modularity**: Bias and noise are independent. Users can model:
          - Systematic error only: `obs_noise_one_std_dev=0, obs_noise_std_dev_fraction=0`
          - Unbiased noisy obs: `obs_bias_multiplier=1.0, obs_noise_std_dev_fraction>0`
        - **Interpretability**: `obs_noise_std_dev_fraction` is unitless and portable
        - **Flexibility**: Reference magnitude can be set once per attribute type
        
        **Clipping**:
        - User-specified bounds applied after noise
        - `None` means unbounded in that direction
        - Example: probabilities get [0.0, 1.0], multipliers typically [0.0, None]
        
        Args:
            true_value: Ground-truth attribute value
            obs_bias_multiplier: Multiplicative bias (1.0 = unbiased, must be > 0)
            obs_noise_one_std_dev: Reference magnitude for noise std dev (>= 0)
            obs_noise_std_dev_fraction: Fraction of reference to use as std dev (>= 0)
            clipping_bounds: Tuple (lower, upper) for clipping; None means unbounded
            rng: NumPy random generator
            
        Returns:
            Observed value with bias, noise, and clipping applied
        """
        # Step 1: Apply multiplicative bias
        obs_value = true_value * obs_bias_multiplier
        
        # Step 2: Add Gaussian noise (if requested)
        if obs_noise_std_dev_fraction > 0 and obs_noise_one_std_dev > 0:
            effective_std = obs_noise_std_dev_fraction * obs_noise_one_std_dev
            obs_value += rng.normal(0, effective_std)
        
        # Step 3: Apply clipping bounds
        if clipping_bounds is not None:
            lower, upper = clipping_bounds
            if lower is not None:
                obs_value = max(obs_value, lower)
            if upper is not None:
                obs_value = min(obs_value, upper)
        
        return float(obs_value)


class PatientGeneratorMixer(PatientGenerator):
    """
    Mixer that combines multiple PatientGenerator instances according to specified proportions.
    
    This class is a subclass of PatientGenerator and can be used interchangeably anywhere
    a PatientGenerator is expected. It creates mixed patient populations (e.g., 80% low-risk + 20% high-risk)
    by combining multiple PatientGenerator instances.
    
    Heterogeneous Visibility Support:
    The mixer automatically detects whether subordinate generators have matching or different
    visibility configurations:
    - If all generators have identical visible_patient_attributes → uses uniform visibility
    - If any generator differs → automatically computes union and uses padding
    
    For heterogeneous visibility, the mixer:
    1. Computes the union of all visible attributes across all subordinate generators
    2. Pads observation vectors from each generator to match the union length
    3. Uses PADDING_VALUE (-1.0) for attributes not visible in a particular generator
    
    This ensures all patient vectors have consistent dimensionality for the RL agent,
    while allowing flexibility in individual generator configurations.
    
    Configuration Format (YAML):
        patient_generator:
          type: mixer
          generators:
            - config_file: path/to/low_risk_config.yaml
              proportion: 0.8
            - config_file: path/to/high_risk_config.yaml
              proportion: 0.2
          # No need to specify visible_patient_attributes - automatically inherited/unified
    
    Example:
        >>> low_risk_gen = PatientGenerator(config=low_risk_config)
        >>> high_risk_gen = PatientGenerator(config=high_risk_config)
        >>> mixer = PatientGeneratorMixer(
        ...     config={
        ...         'generators': [low_risk_gen, high_risk_gen],
        ...         'proportions': [0.8, 0.2]
        ...     }
        ... )
        >>> rng = np.random.default_rng(42)
        >>> patients = mixer.sample(100, rng)  # 80 low-risk, 20 high-risk (shuffled)
    """
    
    # Padding value for heterogeneous visibility: marks non-visible attributes
    PADDING_VALUE: float = -1.0
    
    # Inherit PROVIDES_ATTRIBUTES and other class variables from PatientGenerator
    
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """
        Initialize PatientGeneratorMixer from config dictionary.
        
        Args:
            config: Dictionary with the following structure:
                {
                    'generators': List[PatientGenerator] - Pre-instantiated PatientGenerator instances
                    'proportions': List[float] - Proportions for each generator (must sum to 1.0)
                }
            
        Raises:
            ValueError: If generators/proportions lengths don't match, proportions don't sum to 1.0,
                       any proportion is negative, or generators list is empty
        
        Note:
            The mixer inherits from PatientGenerator but does NOT need the distribution config
            keys (prob_infected_dist, etc.) since it delegates sampling to child generators.
            Visibility configuration is automatically inherited from subordinate generators.
            If all generators have matching visibility, uses uniform visibility.
            If any differ, automatically computes union and uses padding.
        """
        # Extract mixer-specific config
        generators = config.get('generators', [])
        proportions = config.get('proportions', [])
        
        if len(generators) != len(proportions):
            raise ValueError(
                f"Number of generators ({len(generators)}) must match "
                f"number of proportions ({len(proportions)})"
            )
        
        if len(generators) == 0:
            raise ValueError("PatientGeneratorMixer must provide at least one generator")
        
        # Validate proportions
        proportions_array = np.array(proportions, dtype=float)
        if np.any(proportions_array < 0):
            raise ValueError(f"All proportions must be non-negative, got {proportions_array}")
        
        if not np.isclose(proportions_array.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"Proportions must sum to 1.0, got {proportions_array.sum()}"
            )
        
        # Store mixer-specific attributes
        self.generators: List[PatientGenerator] = generators
        self.proportions: np.ndarray = proportions_array
        
        # Initialize the RNG for this mixer (not the base class __init__)
        # We don't call super().__init__() because we don't have distribution configs
        
        # Visibility configuration: automatically detect if generators have matching visibility
        # Check if all generators have the same visible_patient_attributes
        first_gen_attrs = self.generators[0].visible_patient_attributes
        all_match = all(
            gen.visible_patient_attributes == first_gen_attrs
            for gen in self.generators
        )
        
        if all_match:
            # Case 1: Uniform visibility (all generators have same config)
            self.visible_patient_attributes: List[str] = first_gen_attrs
            self._uses_heterogeneous_visibility: bool = False
            self._visible_attrs_union: Optional[List[str]] = None
            self._generator_visibility_map: Optional[Dict[int, set]] = None
        else:
            # Case 2: Heterogeneous visibility (generators have different configs)
            # Compute union of all visible attributes across generators
            all_attrs = set()
            visibility_map: Dict[int, set] = {}
            for i, gen in enumerate(self.generators):
                gen_attrs = set(gen.visible_patient_attributes)
                all_attrs.update(gen_attrs)
                visibility_map[i] = gen_attrs
            
            # Sort for consistent ordering
            self.visible_patient_attributes: List[str] = sorted(all_attrs)
            self._uses_heterogeneous_visibility: bool = True
            self._visible_attrs_union: Optional[List[str]] = self.visible_patient_attributes
            self._generator_visibility_map: Optional[Dict[int, set]] = visibility_map
        
        # Store seed and initialize RNG
        self.seed: Optional[int] = config.get('seed', None)
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        
        # Store config for compatibility with parent class interface
        self.config: Dict[str, Any] = config
        
        # Ownership lifecycle: start in standalone mode (will cascade to children if owned)
        self._standalone: bool = True
        
        # Synchronize child generator seeds/rng for reproducibility
        self._sync_child_seeds(self.seed)
    
    def _set_environment_owned(self):
        """Mark this mixer and all subordinate generators as environment-owned.
        
        Cascades ownership transfer to all child generators recursively.
        This ensures the entire generator hierarchy uses the environment's shared RNG.
        """
        self._standalone = False
        # Cascade to all subordinate generators
        for gen in self.generators:
            if hasattr(gen, '_set_environment_owned'):
                gen._set_environment_owned()

    def _sync_child_seeds(self, seed: Optional[int]):
        """Propagate seed/rng to all child generators if they support it."""
        if seed is None:
            return
        for gen in self.generators:
            try:
                gen.seed = seed
                if hasattr(gen, 'rng'):
                    gen.rng = np.random.default_rng(seed)
            except Exception:
                # Best-effort; if a custom generator rejects seed assignment we continue
                pass
    
    def _compute_visible_attributes_union(self) -> List[str]:
        """
        Compute the union of visible attributes across all subordinate generators.
        
        For heterogeneous visibility configs, this ensures we know the full set of
        attributes that need to be present in the padded observation vectors.
        
        Returns:
            List[str]: Sorted list of all unique visible attributes across generators
        """
        if not self._uses_heterogeneous_visibility:
            return self.visible_patient_attributes
        
        all_attrs = set()
        for gen in self.generators:
            all_attrs.update(gen.visible_patient_attributes)
        return sorted(all_attrs)
    
    def _pad_observation_vector(
        self,
        obs_vector: np.ndarray,
        generator_index: int,
    ) -> np.ndarray:
        """
        Pad an observation vector from a subordinate generator to match the union length.
        
        For heterogeneous visibility, subordinate generators may only output features
        for their visible attributes. This method pads missing attributes with PADDING_VALUE (-1.0).
        
        Args:
            obs_vector: Observation array from a subordinate generator (flat 1D array)
            generator_index: Index of the generator that produced this observation
            
        Returns:
            np.ndarray: Padded observation vector matching union length
        """
        if not self._uses_heterogeneous_visibility:
            # No padding needed for uniform visibility
            return obs_vector
        
        # Get the attributes visible in this generator
        gen_visible_attrs = self.generators[generator_index].visible_patient_attributes
        gen_visible_attrs_set = set(gen_visible_attrs)
        
        # Number of patients in this observation vector
        num_patients = len(obs_vector) // len(gen_visible_attrs)
        
        # Build padded vector with placeholders for missing attributes
        union_attrs = self._visible_attrs_union
        padded = []
        
        for p_idx in range(num_patients):
            for union_attr in union_attrs:
                if union_attr in gen_visible_attrs_set:
                    # Find position of this attribute in the generator's output
                    gen_attr_idx = gen_visible_attrs.index(union_attr)
                    value = obs_vector[p_idx * len(gen_visible_attrs) + gen_attr_idx]
                    padded.append(value)
                else:
                    # Attribute not visible in this generator: use padding value
                    padded.append(self.PADDING_VALUE)
        
        return np.array(padded, dtype=np.float32)
    
    def sample(
        self,
        n_patients: int,
        true_amr_levels: Dict[str, float],
        rng: np.random.Generator = None,
    ) -> List[Patient]:
        """
        Sample a mixed cohort of patients from all generators according to proportions.
        
        Args:
            n_patients: Total number of patients to sample
            true_amr_levels: Ground-truth AMR levels per antibiotic (used for sensitivity sampling)
            rng: NumPy random generator for reproducibility (optional, will use self.rng if not provided)
            
        Returns:
            List of Patient instances, shuffled to avoid ordering bias
            
        Raises:
            ValueError: If n_patients <= 0
        """
        if n_patients <= 0:
            raise ValueError(f"n_patients must be > 0, got {n_patients}")
        if not true_amr_levels:
            raise ValueError("true_amr_levels must be provided and non-empty")
        
        # RNG resolution: enforce explicit RNG when environment-owned
        if not self._standalone and rng is None:
            raise ValueError(
                "PatientGeneratorMixer is environment-owned and requires explicit rng argument. "
                "Pass rng=env.np_random when calling from outside environment."
            )
        # Use provided rng or fall back to self.rng (for standalone mode)
        if rng is None:
            rng = self.rng
        
        # Calculate number of patients to sample from each generator
        # Use proportions to allocate, ensuring total equals n_patients
        n_per_generator = (self.proportions * n_patients).astype(int)
        
        # Handle rounding: add remaining patients to generators with largest fractional parts
        remaining = n_patients - n_per_generator.sum()
        if remaining > 0:
            fractional_parts = (self.proportions * n_patients) - n_per_generator
            # Add one patient to generators with largest fractional parts
            indices_to_increment = np.argsort(fractional_parts)[-remaining:]
            n_per_generator[indices_to_increment] += 1
        
        # Sample from each generator
        mixed_patients = []
        for i, (gen, n) in enumerate(zip(self.generators, n_per_generator)):
            if n > 0:
                # Create a unique rng for each generator based on mixer's rng state
                # This ensures reproducibility while maintaining independence
                gen_seed = rng.integers(0, 2**31)
                gen_rng = np.random.default_rng(gen_seed)
                patients = gen.sample(
                    n_patients=int(n),
                    true_amr_levels=true_amr_levels,
                    rng=gen_rng,
                )
                # Tag patients with source generator index for visibility padding
                for p in patients:
                    try:
                        p.source_generator_index = i
                    except Exception:
                        # Best-effort; continue even if attribute assignment fails
                        pass
                mixed_patients.extend(patients)
        
        # Shuffle to avoid ordering bias (low-risk always first, etc.)
        rng.shuffle(mixed_patients)
        
        return mixed_patients
    
    def observe(self, patients: List[Patient]) -> np.ndarray:
        """
        Extract observed patient attributes with padding for heterogeneous visibility.
        
        Overrides PatientGenerator.observe() to handle mixers where subordinate generators
        have different sets of visible patient attributes. Each patient's observation vector
        is padded to match the union of all visible attributes across generators, using
        PADDING_VALUE (-1.0) for attributes not visible in that patient's source generator.
        
        Args:
            patients: List of Patient objects (typically from sample())
            
        Returns:
            np.ndarray: Flat array of shape (num_patients * len(union_visible_attributes),)
                containing observed attribute values, with PADDING_VALUE for non-visible attributes
        
        Example (heterogeneous visibility):
            >>> gen_a = PatientGenerator(config={'visible_patient_attributes': ['prob_infected', 'benefit_value_multiplier']})
            >>> gen_b = PatientGenerator(config={'visible_patient_attributes': ['prob_infected']})
            >>> mixer = PatientGeneratorMixer(config={'generators': [gen_a, gen_b], 'proportions': [0.5, 0.5]})
            >>> patients = mixer.sample(2, rng)  # 1 from gen_a, 1 from gen_b
            >>> obs = mixer.observe(patients)
            >>> # For patient from gen_a: [prob_infected, benefit_value_multiplier]
            >>> # For patient from gen_b: [prob_infected, PADDING_VALUE]
        """
        if patients is None or len(patients) == 0:
            return np.array([], dtype=np.float32)
        
        # If not using heterogeneous visibility, use parent class implementation
        if not self._uses_heterogeneous_visibility:
            return super().observe(patients)
        
        # For heterogeneous visibility: pad each patient's observation to union length
        obs_values = []
        union_attrs = self._visible_attrs_union
        for p in patients:
            # Determine the originating generator's visibility set
            gen_idx = getattr(p, 'source_generator_index', None)
            gen_visible_attrs = self._generator_visibility_map.get(gen_idx, set())
            for attr in union_attrs:
                if attr in gen_visible_attrs:
                    obs_name = self._obs_attr_name(attr)
                    obs_values.append(float(getattr(p, obs_name)))
                else:
                    obs_values.append(self.PADDING_VALUE)
        return np.array(obs_values, dtype=np.float32)
    
    def obs_dim(self, num_patients: int) -> int:
        """
        Compute patient observation dimension based on union of visible attributes.
        
        Overrides PatientGenerator.obs_dim() to return dimension based on the union
        of all visible attributes across subordinate generators (when using heterogeneous
        visibility), rather than a single generator's visibility config.
        
        Args:
            num_patients: Number of patients per timestep
            
        Returns:
            int: Dimension of padded observation array from observe()
        
        Example:
            >>> gen_a = PatientGenerator(config={'visible_patient_attributes': ['prob_infected', 'benefit_value_multiplier']})
            >>> gen_b = PatientGenerator(config={'visible_patient_attributes': ['prob_infected']})
            >>> mixer = PatientGeneratorMixer(config={'generators': [gen_a, gen_b], 'proportions': [0.5, 0.5]})
            >>> print(mixer.obs_dim(num_patients=10))
            20  # 10 patients * 2 attributes (union)
        """
        num_attrs = len(self.visible_patient_attributes)
        return int(num_patients * num_attrs)

