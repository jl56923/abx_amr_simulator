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
    
    # Required by OptionBase protocol: declare dependencies
    REQUIRES_OBSERVATION_ATTRIBUTES: ClassVar[List[str]] = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob'
    ]
    REQUIRES_AMR_LEVELS: ClassVar[bool] = True  # Needed for accurate expected reward calculation
    REQUIRES_STEP_NUMBER: ClassVar[bool] = False  # Not used for MVP
    PROVIDES_TERMINATION_CONDITION: ClassVar[bool] = False  # Fixed duration
    
    def __init__(
        self,
        name: str,
        duration: int,
        action_thresholds: Dict[str, float],
        uncertainty_threshold: float = 2.0,
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
        """
        super().__init__(name=name, k=duration)
        self.action_thresholds = action_thresholds
        self.uncertainty_threshold = uncertainty_threshold
    
    def compute_expected_reward(
        self,
        patient: Dict[str, float],
        current_amr_levels: Dict[str, float],
        clinical_params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute expected reward for each action given patient attributes and clinical params.
        
        Expected reward formula (simplified):
            reward(prescribe_A) = (prob_infected × clinical_benefit × benefit_multiplier × (1 - amr_A))
                                  - adverse_effect_penalty
            reward(no_treatment) = baseline (typically 0.0)
        
        Args:
            patient: Dict with observed patient attributes
            current_amr_levels: Dict mapping antibiotic name → current resistance level ∈ [0, 1]
            clinical_params: Dict from reward_calculator.abx_clinical_reward_penalties_info_dict
        
        Returns:
            Dict mapping action key → expected reward
            Example: {'prescribe_A': 0.85, 'prescribe_B': 0.42, 'no_treatment': 0.0}
        """
        expected_rewards = {}
        
        # Extract patient attributes (use .get() with defaults for robustness)
        prob_infected = patient.get('prob_infected', 0.0)
        benefit_multiplier = patient.get('benefit_value_multiplier', 1.0)
        
        # Compute expected reward for each antibiotic
        for abx_name, abx_params in clinical_params.items():
            if abx_name == 'no_treatment':
                continue  # Skip no_treatment in this loop
            
            # Extract clinical parameters
            clinical_benefit = abx_params.get('clinical_benefit_reward', 0.0)
            adverse_effect_penalty = abx_params.get('adverse_effect_penalty', 0.0)
            
            # Get current resistance to this antibiotic
            current_amr = current_amr_levels.get(abx_name, 0.0)
            
            # Compute expected reward: effectiveness × benefit - cost
            # Key insight: effectiveness is (1 - amr), so high resistance → low expected benefit
            success_reward = prob_infected * clinical_benefit * benefit_multiplier * (1 - current_amr)
            expected_reward = success_reward - adverse_effect_penalty
            
            expected_rewards[f'prescribe_{abx_name}'] = expected_reward
        
        # No-treatment reward (typically neutral/small negative)
        expected_rewards['no_treatment'] = self.action_thresholds.get('no_treatment', 0.0)
        
        return expected_rewards
    
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
        for attr in self.REQUIRES_OBSERVATION_ATTRIBUTES:
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
        for attr in self.REQUIRES_OBSERVATION_ATTRIBUTES:
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
        
        # Extract clinical parameters from reward calculator
        clinical_params = reward_calculator.abx_clinical_reward_penalties_info_dict
        antibiotic_names = list(option_library.abx_name_to_index.keys())
        
        # Get total observable attributes count for absolute uncertainty
        total_observable_attrs = len(patient_generator.KNOWN_ATTRIBUTE_TYPES)
        
        actions = []
        for patient in patients:
            action = self._select_action_for_patient(
                patient=patient,
                antibiotic_names=antibiotic_names,
                current_amr_levels=current_amr_levels,
                clinical_params=clinical_params,
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
        clinical_params: Dict[str, Any],
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
            clinical_params: Clinical parameters from reward calculator
            use_relative_uncertainty: If True, use relative; if False, use absolute
            total_observable_attrs: Total observable attributes (for absolute uncertainty)
            option_library: OptionLibrary reference (for abx_name_to_index)
        
        Returns:
            Action index ∈ {0, ..., len(antibiotic_names)} where len(antibiotic_names) = no treatment
        """
        # Compute expected rewards
        expected_rewards = self.compute_expected_reward(
            patient=patient,
            current_amr_levels=current_amr_levels,
            clinical_params=clinical_params,
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
    
    # Instantiate HeuristicWorker
    return HeuristicWorker(
        name=name,
        duration=duration,
        action_thresholds=action_thresholds,
        uncertainty_threshold=uncertainty_threshold,
    )
