"""Heuristic policy worker option for hierarchical RL.

HeuristicWorker implements deterministic prescribing logic based on:
1. Expected reward computed from patient attributes + clinical parameters
2. Uncertainty score (missing/padded attributes)
3. Configurable thresholds for each action

Design:
- Implements full OptionBase protocol (drop-in replacement for Block/Alternation)
- Accesses clinical parameters via env_state['reward_calculator']
- Supports both relative (count -1s) and absolute (missing from total) uncertainty metrics
- Manager learns temporal strategy (when to switch between heuristic workers with different thresholds)
"""

from typing import Any, Dict, List, ClassVar
import numpy as np

from abx_amr_simulator.hrl import OptionBase


class HeuristicWorker(OptionBase):
    """Heuristic policy worker implementing deterministic prescribing rules.
    
    Prescribes antibiotics based on expected reward exceeding action thresholds,
    with uncertainty-based conservativeness (refuse to prescribe if too much missing data).
    
    Attributes:
        name: Unique identifier (e.g., 'HEURISTIC_aggressive_10')
        k: Duration in steps (manager learns to switch based on AMR state)
        action_thresholds: Dict mapping action keys to reward thresholds
            Example: {'prescribe_A': 0.7, 'prescribe_B': 0.5, 'no_treatment': 0.0}
        uncertainty_threshold: Threshold for uncertainty score (interpretation varies by logic)
        use_relative_uncertainty: If True, count -1s; if False, count missing from total observable
    """
    
    # Required by OptionBase protocol: declare minimal dependencies
    # Only prob_infected is truly required; other attributes use defaults if missing
    REQUIRES_OBSERVATION_ATTRIBUTES: ClassVar[List[str]] = ['prob_infected']
    REQUIRES_AMR_LEVELS: ClassVar[bool] = True  # Needed for accurate expected reward calculation
    REQUIRES_STEP_NUMBER: ClassVar[bool] = False  # Not used for MVP
    PROVIDES_TERMINATION_CONDITION: ClassVar[bool] = False  # Fixed duration
    
    def __init__(
        self,
        name: str,
        duration: int,
        action_thresholds: Dict[str, float],
        uncertainty_threshold: float = 2.0,
        default_recovery_without_treatment_prob: float = 0.1,
    ):
        """Initialize heuristic worker.
        
        Args:
            name: Unique identifier (e.g., 'HEURISTIC_aggressive_10')
            duration: Fixed duration in steps (k)
            action_thresholds: Dict mapping action keys to reward thresholds
                Keys: 'prescribe_{abx}' for each antibiotic, 'no_treatment'
                Values: Minimum expected reward to select that action
                Example: {'prescribe_A': 0.7, 'prescribe_B': 0.5, 'no_treatment': 0.0}
            uncertainty_threshold: Threshold for refusing to prescribe due to missing data
                Interpretation depends on logic (e.g., refuse if uncertainty > threshold)
            default_recovery_without_treatment_prob: Default value for spontaneous recovery probability
                when recovery_without_treatment_prob is not visible in patient observation (default: 0.1)
        """
        super().__init__(name=name, k=duration)
        self.action_thresholds = action_thresholds
        self.uncertainty_threshold = uncertainty_threshold
        self.default_recovery_prob = default_recovery_without_treatment_prob
        # Will be set by OptionsWrapper during initialization
        self._observable_patient_attributes: List[str] = []
    
    def _estimate_unobserved_attribute_values_from_observed(
        self,
        patient: Dict[str, float],
    ) -> Dict[str, float]:
        """Extension point: estimate missing attributes from observed values.
        
        Override this method in a subclass to implement custom clinical reasoning
        for estimating unobserved patient attributes based on observed ones.
        
        This enables modeling realistic clinical decision-making where clinicians
        use available information to form educated guesses about missing data.
        
        Called at the start of compute_expected_reward() BEFORE extracting attribute values.
        Default implementation: returns patient unchanged (uses fallback defaults).
        
        Example override in custom subclass:
            class CustomHeuristicWorker(HeuristicWorker):
                def _estimate_unobserved_attribute_values_from_observed(
                    self, patient: Dict[str, float]
                ) -> Dict[str, float]:
                    # Make a copy to avoid mutating original
                    patient = patient.copy()
                    
                    # Estimate recovery_prob from infection risk
                    # (sicker patients less likely to recover spontaneously)
                    if ('recovery_without_treatment_prob' not in patient 
                        or patient['recovery_without_treatment_prob'] == -1):
                        pI = patient.get('prob_infected', 0.5)
                        # Linear model: high infection risk → low recovery probability
                        patient['recovery_without_treatment_prob'] = max(0.05, 0.2 - 0.15 * pI)
                    
                    # Estimate benefit multiplier from age/frailty proxy
                    if ('benefit_value_multiplier' not in patient
                        or patient['benefit_value_multiplier'] == -1):
                        # Assume healthier patients benefit more from treatment
                        patient['benefit_value_multiplier'] = 1.0 + (1.0 - pI) * 0.3
                    
                    return patient
        
        Args:
            patient: Dict with observed attributes (may have -1 for missing/padded attributes)
        
        Returns:
            Dict with estimates filled in. Should NOT overwrite non-missing observed values.
            Default: returns input unchanged.
        
        Note:
            - Only fill in attributes that are missing (not in dict OR value == -1)
            - Don't overwrite valid observed values
            - Always return a dict (copy if modified, original if unchanged)
            - See docs/tutorials/custom_heuristic_attribute_estimation.md for full guide
        """
        return patient
    
    def compute_expected_reward(
        self,
        patient: Dict[str, float],
        antibiotic_names: List[str],
        current_amr_levels: Dict[str, float],
        reward_calculator: Any,
    ) -> Dict[str, float]:
        """Compute expected reward for each action using clinical reasoning.
        
        This is the core clinical decision logic: given observed patient attributes and
        current AMR levels, calculate the expected utility of each prescribing decision.
        
        Expected reward formula (prescribe antibiotic):
            E[reward] = pI×pS×pB×RB×vB + pI×(1−pS)×pF×RF×vF + pAE×AE
        
        Expected reward formula (no treatment):
            E[reward] = pI×r×RB×vB + pI×(1−r)×pF×RF×vF
        
        Where:
            - pI = prob_infected (observed)
            - pS = 1 - amr_level (sensitivity probability)
            - pB = clinical_benefit_probability × benefit_probability_multiplier (clamped)
            - pF = clinical_failure_probability × failure_probability_multiplier (clamped)
            - RB = normalized_clinical_benefit_reward
            - RF = normalized_clinical_failure_penalty
            - vB = benefit_value_multiplier (observed)
            - vF = failure_value_multiplier (observed)
            - pAE = adverse_effect_probability
            - AE = normalized_adverse_effect_penalty
            - r = recovery_without_treatment_prob (observed)
        
        NOTE: Does NOT include epsilon penalty to avoid leaking counterfactual AMR dynamics.
        
        Args:
            patient: Dict with observed patient attributes
            antibiotic_names: List of antibiotic names in environment
            current_amr_levels: Dict mapping antibiotic name → current resistance level ∈ [0, 1]
            reward_calculator: RewardCalculator instance (used only to access clinical_params)
        
        Returns:
            Dict mapping action key → expected reward
            Example: {'prescribe_A': 0.85, 'prescribe_B': 0.42, 'no_treatment': 0.0}
        
        Raises:
            ValueError: If prob_infected is not present in patient observation
        """
        # Apply custom attribute estimation if overridden in subclass
        patient = self._estimate_unobserved_attribute_values_from_observed(patient=patient)
        
        # Extract REQUIRED attribute (fail loudly if missing)
        if 'prob_infected' not in patient:
            raise ValueError(
                f"HeuristicWorker '{self.name}' requires 'prob_infected' in patient observation. "
                "This attribute must be in visible_patient_attributes. "
                f"Got patient keys: {list(patient.keys())}"
            )
        pI = float(patient['prob_infected'])
        
        # Extract optional attributes with defaults
        vB = float(patient.get('benefit_value_multiplier', 1.0))
        vF = float(patient.get('failure_value_multiplier', 1.0))
        benefit_prob_mult = float(patient.get('benefit_probability_multiplier', 1.0))
        failure_prob_mult = float(patient.get('failure_probability_multiplier', 1.0))
        r_spont = float(patient.get('recovery_without_treatment_prob', self.default_recovery_prob))
        
        # Get clinical parameters from reward_calculator
        clinical_params = reward_calculator.abx_clinical_reward_penalties_info_dict
        
        # Base clinical probabilities
        base_benefit_prob = clinical_params['clinical_benefit_probability']
        base_failure_prob = clinical_params['clinical_failure_probability']
        
        # Clamp-scaled probabilities
        pB = min(1.0, max(0.0, base_benefit_prob * benefit_prob_mult))
        pF = min(1.0, max(0.0, base_failure_prob * failure_prob_mult))
        
        # Normalized rewards
        RB = clinical_params['normalized_clinical_benefit_reward']
        RF = clinical_params['normalized_clinical_failure_penalty']
        
        expected_rewards = {}
        
        # Compute expected reward for each antibiotic
        for abx_name in antibiotic_names:
            visible_amr = current_amr_levels.get(abx_name, 0.0)
            pS = 1.0 - float(visible_amr)
            
            # Expected benefit and failure terms
            expected_reward = pI * pS * pB * RB * vB
            expected_reward += pI * (1.0 - pS) * pF * RF * vF
            
            # Adverse effects expectation
            ae_info = clinical_params['abx_adverse_effects_info'][abx_name]
            ae_penalty = ae_info['normalized_adverse_effect_penalty']
            pAE = ae_info['adverse_effect_probability']
            expected_reward += float(pAE) * float(ae_penalty)
            
            expected_rewards[f'prescribe_{abx_name}'] = float(expected_reward)
        
        # No treatment expected reward
        expected_reward_no_treatment = pI * r_spont * RB * vB
        expected_reward_no_treatment += pI * (1.0 - r_spont) * pF * RF * vF
        expected_rewards['no_treatment'] = float(expected_reward_no_treatment)
        
        return expected_rewards
    
    def set_observable_attributes(self, attributes: List[str]) -> None:
        """Set the list of observable patient attributes (called by OptionsWrapper).
        
        This method is called by OptionsWrapper during initialization to inject
        the actual list of patient attributes that PatientGenerator exposes.
        Used for uncertainty scoring to check all potentially observable attributes.
        
        Args:
            attributes: List of patient attribute names from PatientGenerator
        """
        self._observable_patient_attributes = attributes
    
    def compute_relative_uncertainty_score(
        self,
        patient: Dict[str, float],
    ) -> int:
        """Count number of padded (missing) patient attributes relative to what's observed.
        
        This is uncertainty "relative to what we can see": if we observe some attributes,
        how many are padded (-1)? This captures missing subset of the patient profile.
        
        Args:
            patient: Dict with observed patient attributes
        
        Returns:
            Count of attributes with value -1 (indicates padding/missing data)
        
        Example:
            - Patient sees {'prob_infected': 0.7, 'benefit_multiplier': -1, 'failure_multiplier': -1, ...} → score = 4
            - Patient sees {'prob_infected': 0.7} (only 1 attribute visible, no padding) → score = 0
        """
        uncertainty = 0
        # Use injected observable attributes list, fall back to minimal requirements if not set
        check_attrs = self._observable_patient_attributes or self.REQUIRES_OBSERVATION_ATTRIBUTES
        for attr in check_attrs:
            if attr in patient and patient[attr] == -1.0:
                uncertainty += 1
        return uncertainty
    
    def compute_absolute_uncertainty_score(
        self,
        patient: Dict[str, float],
        total_observable_attrs: int,
    ) -> int:
        """Count how many of the total observable patient attributes are missing (not observed).
        
        This is uncertainty "absolute": out of all N possible patient attributes,
        how many are we NOT seeing for this patient? Captures overall information poverty.
        
        Args:
            patient: Dict with observed patient attributes
            total_observable_attrs: Total number of observable attributes (from PatientGenerator)
        
        Returns:
            Number of attributes not observed for this patient (out of total observable)
        
        Example (assuming 6 total observable attributes):
            - Patient sees {'prob_infected': 0.7, 'benefit_multiplier': 1.2} → missing 4 of 6 → score = 4
            - Patient sees {'prob_infected': 0.7} → missing 5 of 6 → score = 5
            - Patient sees all 6 attributes → score = 0
        """
        num_observed = 0
        # Use injected observable attributes list, fall back to minimal requirements if not set
        check_attrs = self._observable_patient_attributes or self.REQUIRES_OBSERVATION_ATTRIBUTES
        for attr in check_attrs:
            if attr in patient and patient[attr] != -1.0:
                num_observed += 1
        return total_observable_attrs - num_observed
    
    def decide(self, env_state: Dict[str, Any]) -> np.ndarray:
        """Implement OptionBase protocol's decide() method.
        
        Deterministic action selection for all patients based on expected rewards and uncertainty.
        
        Args:
            env_state: Dict with keys:
                - 'patients': List of patient dicts
                - 'num_patients': Number of patients this timestep
                - 'current_amr_levels': Dict of current AMR levels (REQUIRED)
                - 'reward_calculator': RewardCalculator instance (REQUIRED for heuristic workers)
                - 'patient_generator': PatientGenerator instance (REQUIRED for absolute uncertainty)
                - 'use_relative_uncertainty': Boolean flag (from option library config)
                - 'option_library': OptionLibrary reference (for abx_name_to_index)
                - 'current_step': Current episode step (unused)
                - 'max_steps': Episode length limit (unused)
        
        Returns:
            np.ndarray of shape (num_patients,) with action indices
            - 0 to len(antibiotic_names)-1: prescribe that antibiotic
            - len(antibiotic_names): no treatment
        """
        patients = env_state['patients']
        current_amr_levels = env_state['current_amr_levels']
        reward_calculator = env_state['reward_calculator']
        patient_generator = env_state['patient_generator']
        use_relative_uncertainty = env_state['use_relative_uncertainty']
        option_library = env_state['option_library']
        
        # Get antibiotic names and total observable attributes
        antibiotic_names = list(option_library.abx_name_to_index.keys())
        total_observable_attrs = len(patient_generator.KNOWN_ATTRIBUTE_TYPES)
        
        actions = []
        for patient in patients:
            action = self._select_action_for_patient(
                patient=patient,
                antibiotic_names=antibiotic_names,
                current_amr_levels=current_amr_levels,
                reward_calculator=reward_calculator,
                use_relative_uncertainty=use_relative_uncertainty,
                total_observable_attrs=total_observable_attrs,
                option_library=option_library,
            )
            actions.append(action)
        
        return np.array(actions, dtype=np.int32)
    
    def _select_action_for_patient(
        self,
        patient: Dict[str, float],
        antibiotic_names: List[str],
        current_amr_levels: Dict[str, float],
        reward_calculator: Any,
        use_relative_uncertainty: bool,
        total_observable_attrs: int,
        option_library: Any,
    ) -> int:
        """Deterministic action selection for a single patient.
        
        Logic:
        1. Compute expected reward for each action
        2. Compute uncertainty score
        3. If uncertainty > threshold, default to no_treatment
        4. Otherwise, select best action exceeding its threshold
        
        Args:
            patient: Single patient dict
            antibiotic_names: Ordered list of antibiotic names
            current_amr_levels: Dict mapping antibiotic name → current resistance level
            reward_calculator: RewardCalculator instance from environment
            use_relative_uncertainty: If True, use relative; if False, use absolute
            total_observable_attrs: Total observable attributes (for absolute uncertainty)
            option_library: OptionLibrary reference (for abx_name_to_index)
        
        Returns:
            Action index ∈ {0, ..., len(antibiotic_names)} where len(antibiotic_names) = no treatment
        """
        # Compute expected rewards
        expected_rewards = self.compute_expected_reward(
            patient=patient,
            antibiotic_names=antibiotic_names,
            current_amr_levels=current_amr_levels,
            reward_calculator=reward_calculator,
        )
        
        # Compute uncertainty score
        if use_relative_uncertainty:
            uncertainty = self.compute_relative_uncertainty_score(patient=patient)
        else:
            uncertainty = self.compute_absolute_uncertainty_score(
                patient=patient,
                total_observable_attrs=total_observable_attrs,
            )
        
        # Check uncertainty threshold (refuse to prescribe if too uncertain)
        if uncertainty > self.uncertainty_threshold:
            # Too much missing data—default to no treatment
            return len(antibiotic_names)  # no treatment action index
        
        # Find best action exceeding its threshold
        best_action = len(antibiotic_names)  # default to no_treatment
        best_value = expected_rewards.get('no_treatment', -np.inf)
        
        for abx_idx, abx in enumerate(antibiotic_names):
            action_key = f'prescribe_{abx}'
            if action_key not in expected_rewards:
                continue  # Skip if this antibiotic not in clinical params
            
            action_value = expected_rewards[action_key]
            threshold = self.action_thresholds.get(action_key, -np.inf)
            
            # Only consider actions exceeding their threshold
            if action_value >= threshold and action_value > best_value:
                best_action = abx_idx
                best_value = action_value
        
        return best_action
    
    def get_referenced_antibiotics(self) -> List[str]:
        """Return list of antibiotics this worker might prescribe (extracted from thresholds).
        
        Returns:
            List of antibiotic names referenced in action_thresholds
        """
        antibiotics = []
        for action_key in self.action_thresholds.keys():
            if action_key.startswith('prescribe_'):
                abx_name = action_key.replace('prescribe_', '')
                antibiotics.append(abx_name)
        return antibiotics


def load_heuristic_option(name: str, config: Dict[str, Any]) -> OptionBase:
    """Instantiate a HeuristicWorker from config.
    
    Expected config keys:
        - duration (int): Fixed duration in steps (required)
        - action_thresholds (dict): Mapping action keys → reward thresholds (required)
            Example: {'prescribe_A': 0.7, 'prescribe_B': 0.5, 'no_treatment': 0.0}
        - uncertainty_threshold (float): Threshold for refusing to prescribe (default 2.0)
        - default_recovery_without_treatment_prob (float): Default spontaneous recovery probability (default 0.1)
    
    Args:
        name: Unique identifier for this option instance
        config: Merged configuration dict (default config + overrides)
    
    Returns:
        HeuristicWorker instance implementing OptionBase protocol
    
    Raises:
        TypeError: If config is not a dict
        ValueError: If required keys missing or invalid
    """
    if not isinstance(config, dict):
        raise TypeError(f"HeuristicWorker config must be a dict, got {type(config).__name__}")
    
    # Validate required keys
    if 'duration' not in config:
        raise ValueError(f"Heuristic Worker '{name}' config missing required key 'duration'")
    if 'action_thresholds' not in config:
        raise ValueError(f"Heuristic Worker '{name}' config missing required key 'action_thresholds'")
    
    duration = config['duration']
    action_thresholds = config['action_thresholds']
    uncertainty_threshold = config.get('uncertainty_threshold', 2.0)
    default_recovery_prob = config.get('default_recovery_without_treatment_prob', 0.1)
    
    # Validate duration
    if not isinstance(duration, int) or duration < 1:
        raise ValueError(
            f"HeuristicWorker '{name}': 'duration' must be an int >= 1, got {duration} ({type(duration).__name__})"
        )
    
    # Validate action_thresholds
    if not isinstance(action_thresholds, dict):
        raise ValueError(
            f"HeuristicWorker '{name}': 'action_thresholds' must be a dict, got {type(action_thresholds).__name__}"
        )
    if not action_thresholds:
        raise ValueError(
            f"HeuristicWorker '{name}': 'action_thresholds' dict cannot be empty"
        )
    for action_key, threshold_value in action_thresholds.items():
        if not isinstance(action_key, str):
            raise ValueError(
                f"HeuristicWorker '{name}': action_thresholds keys must be strings, got {type(action_key).__name__}"
            )
        if not isinstance(threshold_value, (int, float)):
            raise ValueError(
                f"HeuristicWorker '{name}': action_thresholds['{action_key}'] must be numeric, "
                f"got {type(threshold_value).__name__}"
            )
    
    # Validate uncertainty_threshold
    if not isinstance(uncertainty_threshold, (int, float)):
        raise ValueError(
            f"HeuristicWorker '{name}': 'uncertainty_threshold' must be numeric, "
            f"got {type(uncertainty_threshold).__name__}"
        )
    
    # Validate default_recovery_prob
    if not isinstance(default_recovery_prob, (int, float)):
        raise ValueError(
            f"HeuristicWorker '{name}': 'default_recovery_without_treatment_prob' must be numeric, "
            f"got {type(default_recovery_prob).__name__}"
        )
    if not (0.0 <= default_recovery_prob <= 1.0):
        raise ValueError(
            f"HeuristicWorker '{name}': 'default_recovery_without_treatment_prob' must be in [0, 1], "
            f"got {default_recovery_prob}"
        )
    
    # Instantiate HeuristicWorker
    return HeuristicWorker(
        name=name,
        duration=duration,
        action_thresholds=action_thresholds,
        uncertainty_threshold=uncertainty_threshold,
        default_recovery_without_treatment_prob=default_recovery_prob,
    )
