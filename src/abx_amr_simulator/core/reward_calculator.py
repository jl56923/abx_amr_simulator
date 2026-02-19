"""
RewardCalculator class for calculating rewards in the ABXAMREnv environment.

The reward function combines individual patient outcomes with community AMR burden,
weighted by a lambda parameter.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, ClassVar
from .base_reward_calculator import RewardCalculatorBase
from .types import Patient


class RewardCalculator(RewardCalculatorBase):
    """
    Handles reward calculation logic for antibiotic prescribing decisions.
    
    The reward has two components:
    1. Individual patient rewards (clinical benefit, clinical failure, adverse effects, marginal AMR)
    2. Community AMR burden (sum of resistance levels across antibiotics)
    
    These are weighted by lambda:
        reward = lambda * community_reward + (1 - lambda) * sum(individual_rewards)
    
    OWNERSHIP (for future multi-agent refactor):
    - Owns: reward formula, lambda weighting, per-antibiotic reward parameters
    - Receives: List[Patient] with outcomes, current AMR levels, prescriptions
    - Returns: scalar reward (single-agent) or dict of per-agent rewards (multi-agent)
    
    Future multi-agent notes:
    - Will be instantiated per-agent (each agent can have different lambda)
    - Community reward scope configurable: locale-only, neighboring, global
    - Individual reward attribution clear (only patients treated by this agent)
    """
    
    # Declare required Patient attributes for compatibility validation
    REQUIRED_PATIENT_ATTRS: ClassVar[List[str]] = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob',
        'infection_status',
        'abx_sensitivity_dict',
    ]

    @classmethod
    def default_config(cls) -> Dict:
        """
        Return a template configuration with sensible defaults.
        
        Use this method to get a starting point for configuration. Modify the
        returned dictionary as needed for your use case.
        
        Returns:
            Dictionary with all required keys set to reasonable defaults.
        
        Example:
            >>> config = RewardCalculator.default_config()
            >>> config['lambda_weight'] = 0.3  # Modify as needed
            >>> config['abx_clinical_reward_penalties_info_dict']['clinical_benefit_reward'] = 15.0
            >>> rc = RewardCalculator(config=config)
        """
        return {
            'abx_clinical_reward_penalties_info_dict': {
                'clinical_benefit_reward': 10.0,
                'clinical_benefit_probability': 0.9,
                'clinical_failure_penalty': -5.0,
                'clinical_failure_probability': 0.1,
                'abx_adverse_effects_info': {
                    'A': {
                        'adverse_effect_penalty': -2.0,
                        'adverse_effect_probability': 0.05,
                    },
                },
            },
            'lambda_weight': 0.5,
            'epsilon': 0.05,
            'seed': None,
        }

    def __init__(self, config: Dict):
        """Initialize reward calculator from configuration dictionary.
        
        Configures the composite reward function:
            reward = lambda * community_amr_penalty + (1-lambda) * sum(individual_patient_rewards)
        
        where individual rewards include clinical benefit, clinical failure, adverse effects,
        and marginal AMR contribution (weighted by epsilon).
        
        Args:
            config (Dict): Configuration dictionary with required keys:
                - 'abx_clinical_reward_penalties_info_dict' (dict): Structure:
                    {
                        'clinical_benefit_reward': float (> 0),
                        'clinical_benefit_probability': float in [0, 1],
                        'clinical_failure_penalty': float (< 0),
                        'clinical_failure_probability': float in [0, 1],
                        'abx_adverse_effects_info': {
                            'antibiotic_name': {
                                'adverse_effect_penalty': float (< 0),
                                'adverse_effect_probability': float in [0, 1]
                            },
                            ...
                        }
                    }
                - 'lambda_weight' (float, default 0.5): Trade-off between community visible AMR
                    burden and individual outcomes. Must be in [0, 1].
                    lambda=0 → pure individual; lambda=1 → pure community.
                - 'epsilon' (float, default 0.05): Relative weight for marginal visible AMR contribution,
                    interpreted as percentage of max reward/penalty magnitude. For example,
                    epsilon=0.05 means a 5% mini-penalty relative to the clinical reward scale.
                    Must be in (0, 0.1].
                - 'seed' (int, optional): Random seed for reproducibility.
        
        Raises:
            ValueError: If required config keys missing or parameters outside valid ranges.
        
        Example:
            >>> config = RewardCalculator.default_config()
            >>> config['lambda_weight'] = 0.3
            >>> config['seed'] = 42
            >>> rc = RewardCalculator(config=config)
        """
        # Extract parameters from config dict with defaults
        if 'abx_clinical_reward_penalties_info_dict' not in config:
            raise ValueError("Missing required key 'abx_clinical_reward_penalties_info_dict' in config")
        
        abx_clinical_reward_penalties_info_dict = config['abx_clinical_reward_penalties_info_dict']
        lambda_weight = config.get('lambda_weight', 0.5)
        epsilon = config.get('epsilon', 0.05)
        seed = config.get('seed', None)
        
        # Do value checks for: lambda_weight, epsilon
        if not (0.0 <= lambda_weight <= 1.0):
            raise ValueError(f"lambda_weight must be in [0, 1], got {lambda_weight}")
        # epsilon has to be positive and small:
        if not (0.0 <= epsilon <= 0.1):
            raise ValueError(f"epsilon must be in [0, 0.1], got {epsilon}")
        
        # Now, iterate through the abx_clinical_reward_penalties_info_dict to extract and validate the clinical benefit reward, clinical benefit probability, adverse effect penalty, and adverse effect probability for each antibiotic. Do validation checks for all four values.
        
        # Check that the abx_clinical_reward_penalties_info_dict has all of the necessary parameters:
        required_keys_for_abx_clinical_reward_penalties_info_dict = ['clinical_benefit_reward', 'clinical_benefit_probability', 'clinical_failure_penalty', 'clinical_failure_probability', 'abx_adverse_effects_info']
        
        for key in required_keys_for_abx_clinical_reward_penalties_info_dict:
            if key not in abx_clinical_reward_penalties_info_dict:
                raise ValueError(f"Missing key '{key}' in abx_clinical_reward_penalties_info_dict")
            
        # Now, do value checks for each of the four values:
        clinical_benefit_reward = abx_clinical_reward_penalties_info_dict['clinical_benefit_reward']
        clinical_benefit_probability = abx_clinical_reward_penalties_info_dict['clinical_benefit_probability']
        clinical_failure_penalty = abx_clinical_reward_penalties_info_dict['clinical_failure_penalty']
        clinical_failure_probability = abx_clinical_reward_penalties_info_dict['clinical_failure_probability']
        
        max_abs_value_of_any_reward_or_penalty = max(abs(clinical_benefit_reward), abs(clinical_failure_penalty)) # This is going to be used for normalization
        
        # The probabilities must be in [0, 1]:
        if not (0.0 <= clinical_benefit_probability <= 1.0):
            raise ValueError(f"clinical_benefit_probability for '{abx}' must be in [0, 1], got {clinical_benefit_probability}")
        if not (0.0 <= clinical_failure_probability <= 1.0):
            raise ValueError(f"clinical_failure_probability for '{abx}' must be in [0, 1], got {clinical_failure_probability}")
        
        if clinical_benefit_reward <= 0.0:
            raise ValueError(f"clinical_benefit_reward for '{abx}' must be positive, got {clinical_benefit_reward}")
        if clinical_failure_penalty >= 0.0:
            raise ValueError(f"clinical_failure_penalty for '{abx}' must be negative, got {clinical_failure_penalty}")
        
        # Now, iterate through 
        for abx, abx_info in abx_clinical_reward_penalties_info_dict['abx_adverse_effects_info'].items():
            # Make sure that the following four keys are present in abx_info:
            required_keys_adverse_effects = ['adverse_effect_penalty', 'adverse_effect_probability']
            for key in required_keys_adverse_effects:
                if key not in abx_info:
                    raise ValueError(f"Missing key '{key}' in antibiotic info for '{abx}'")

            adverse_effect_penalty = abx_info['adverse_effect_penalty']
            adverse_effect_probability = abx_info['adverse_effect_probability']
            
            if not (0.0 <= adverse_effect_probability <= 1.0):
                raise ValueError(f"adverse_effect_probability for '{abx}' must be in [0, 1], got {adverse_effect_probability}")
            # adverse_effect_penalty has to be negative:
            if adverse_effect_penalty >= 0.0:
                raise ValueError(f"adverse_effect_penalty for '{abx}' must be negative, got {adverse_effect_penalty}")
            
            # Check if the max of the absolute value of clinical_benefit_reward, clinical_failure_penalty, or adverse_effect_penalty is greater than the current max_abs_value_of_any_reward_or_penalty:
            if abs(adverse_effect_penalty) > max_abs_value_of_any_reward_or_penalty:
                max_abs_value_of_any_reward_or_penalty = abs(adverse_effect_penalty)
                
        # Now that the whole dictionary has been iterated through, max_abs_value_of_any_reward_or_penalty is guaranteed to be the maximum value of any reward or penalty in the entire dictionary. Let's normalize the dictionary now:
        abx_clinical_reward_penalties_info_dict['normalized_clinical_benefit_reward'] = clinical_benefit_reward/max_abs_value_of_any_reward_or_penalty
        abx_clinical_reward_penalties_info_dict['normalized_clinical_failure_penalty'] = clinical_failure_penalty/max_abs_value_of_any_reward_or_penalty
        
        for abx, abx_info in abx_clinical_reward_penalties_info_dict['abx_adverse_effects_info'].items():
            normalized_info = {
                'normalized_adverse_effect_penalty': abx_info['adverse_effect_penalty'] / max_abs_value_of_any_reward_or_penalty,
            }
            abx_clinical_reward_penalties_info_dict['abx_adverse_effects_info'][abx].update(normalized_info)
            
        self.abx_clinical_reward_penalties_info_dict = abx_clinical_reward_penalties_info_dict
        self.lambda_weight = lambda_weight
        self.epsilon = epsilon
        
        # Store normalization factor (used to precompute normalized clinical rewards)
        self.max_abs_value_of_any_reward_or_penalty = max_abs_value_of_any_reward_or_penalty
        
        # Store the action-to-antibiotic mapping for explicit conversion.
        abx_names = list(abx_clinical_reward_penalties_info_dict['abx_adverse_effects_info'].keys())
        self.antibiotic_names: List[str] = abx_names
        self.index_to_abx_name: Dict[int, str] = {idx: name for idx, name in enumerate(abx_names)}
        self.abx_name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(abx_names)}
        
        # Add 'no_treatment' action mapping to both dictionaries:
        no_treatment_index = len(abx_names)
        self.index_to_abx_name[no_treatment_index] = 'no_treatment'
        self.abx_name_to_index['no_treatment'] = no_treatment_index
        
        # Initialize RNG for stochastic draws (seeded if provided)
        self.seed: Optional[int] = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        
        # Ownership lifecycle: start in standalone mode
        self._standalone: bool = True
        
        return
    
    def _set_environment_owned(self):
        """Mark this component as environment-owned (one-way transition).
        
        After calling this method, the component will require explicit rng argument
        in all stochastic methods and will fail loudly if not provided.
        This enforces that environment-owned components receive RNG from environment.
        
        Called by ABXAMREnv.__init__ after accepting this component as a parameter.
        """
        self._standalone = False
    
    def calculate_individual_reward(
        self,
        patient_infected: bool,
        antibiotic_name: str,
        infection_sensitive_to_prescribed_abx: bool = None,
        delta_visible_amr: float = None,
        return_clinical_benefit_adverse_event_occurrence: bool = False,
        benefit_value_multiplier: float = 1.0,
        failure_value_multiplier: float = 1.0,
        benefit_probability_multiplier: float = 1.0,
        failure_probability_multiplier: float = 1.0,
        recovery_without_treatment_prob: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, Tuple[float, bool]]:
        """
        Calculate reward for a single patient with heterogeneous patient-specific multipliers.
        
        All patient attributes (multipliers, probabilities) provided as arguments represent
        TRUE values (ground truth), not observed/noisy versions. The reward calculation uses
        ground truth to determine actual clinical outcomes, while the agent observes potentially
        biased or noisy attribute estimates.
        
        Args:
            patient_infected (bool): TRUE infection status (ground truth, not observed value)
            antibiotic_name (str): Name of the antibiotic prescribed (or 'no_treatment' for no treatment)
            infection_sensitive_to_prescribed_abx (bool): TRUE sensitivity status (ground truth, not observed)
            delta_visible_amr (float): Marginal visible AMR contribution from this prescription
            return_clinical_benefit_adverse_event_occurrence (bool): If True, return (reward, benefit, failure, adverse)
            benefit_value_multiplier (float): TRUE value - scales clinical benefit reward value (default 1.0)
            failure_value_multiplier (float): TRUE value - scales clinical failure penalty value (default 1.0)
            benefit_probability_multiplier (float): TRUE value - scales clinical benefit probability (default 1.0)
            failure_probability_multiplier (float): TRUE value - scales clinical failure probability (default 1.0)
            recovery_without_treatment_prob (float): TRUE value - probability of spontaneous recovery without treatment (default 0.0)
        
        Returns:
            Reward value for this patient, or (reward, benefit_occurred, failure_occurred, adverse_occurred) if requested
        """
        # RNG resolution: enforce explicit RNG when environment-owned
        if not self._standalone and rng is None:
            raise ValueError(
                "RewardCalculator is environment-owned and requires explicit rng argument. "
                "Pass rng=env.np_random when calling from outside environment."
            )
        rng = rng if rng is not None else self.rng
        if rng is None:
            raise ValueError("RNG must be provided either via argument or initialized on RewardCalculator")

        reward = 0.0
        adverse_effects_occurred = False
        clinical_failure_occurred = False
        clinical_benefit_occurred = False
        
        antibiotic_is_prescribed = antibiotic_name != 'no_treatment'
        
        # Do a validation check that if antibiotic is prescribed, infection_sensitivity and delta_amr are provided:
        if antibiotic_is_prescribed:
            if infection_sensitive_to_prescribed_abx is None:
                raise ValueError("infection_sensitive_to_prescribed_abx must be provided if an antibiotic is prescribed")
            if delta_visible_amr is None:
                raise ValueError("delta_visible_amr must be provided if an antibiotic is prescribed")
        
        # Scale probabilities by patient-specific multipliers (clamped to [0, 1])
        base_benefit_prob = self.abx_clinical_reward_penalties_info_dict['clinical_benefit_probability']
        base_failure_prob = self.abx_clinical_reward_penalties_info_dict['clinical_failure_probability']
        scaled_benefit_prob = min(1.0, max(0.0, base_benefit_prob * benefit_probability_multiplier))
        scaled_failure_prob = min(1.0, max(0.0, base_failure_prob * failure_probability_multiplier))
        
        # Get the clinical benefit and adverse effect parameters for the prescribed antibiotic
        
        # Set reward to 0 initially:
        reward = 0
        
        # Handle what happens if the an antibiotic is prescribed vs not prescribed:
        if antibiotic_is_prescribed:
            # If the antibiotic that is prescribed is not in the dictionary, raise an error.
            if antibiotic_name not in self.antibiotic_names:
                    raise ValueError(f"Antibiotic '{antibiotic_name}' not found in abx_clinical_reward_penalties_info_dict.")
                
            # If an antibiotic is prescribed there are three possible scenarios:
            # 1. Patient is infected, infection is sensitive to antibiotic
            # 2. Patient is infected, infection is resistant to antibiotic
            # 3. Patient is not infected
            # For all three scenarios, there is the chance of adverse effects occurring.
            
            # First scenario: patient is infected, infection is sensitive to antibiotic.
            if patient_infected and infection_sensitive_to_prescribed_abx:
                # Use the scaled probability of clinical benefit to determine if this patient gets a clinical benefit:
                clinical_benefit_occurred = rng.random() < scaled_benefit_prob
                
                if clinical_benefit_occurred:
                    reward += self.abx_clinical_reward_penalties_info_dict['normalized_clinical_benefit_reward'] * benefit_value_multiplier
            else:
                # Second scenario: patient is infected, infection is resistant to antibiotic.
                if patient_infected and not infection_sensitive_to_prescribed_abx:
                    # Use the scaled probability of clinical failure to determine if this patient gets a clinical failure penalty:
                    clinical_failure_occurred = rng.random() < scaled_failure_prob
                    
                    if clinical_failure_occurred:
                        reward += self.abx_clinical_reward_penalties_info_dict['normalized_clinical_failure_penalty'] * failure_value_multiplier
                # Third scenario: patient is not infected -> no clinical benefit or failure, just move on to adverse effects.
                
            # Adverse effects penalty; for adverse effects, it does matter which antibiotic was prescirbed.
            adverse_effects_occurred = rng.random() < self.abx_clinical_reward_penalties_info_dict['abx_adverse_effects_info'][antibiotic_name]['adverse_effect_probability']
            if adverse_effects_occurred:
                reward += self.abx_clinical_reward_penalties_info_dict['abx_adverse_effects_info'][antibiotic_name]['normalized_adverse_effect_penalty']
            
        else:
            # If an antibiotic is not prescribed, there are 3 possible scenarios:
            # 1. Patient is infected + spontaneous recovery occurs -> clinical benefit reward
            # 2. Patient is infected + no spontaneous recovery -> check for clinical failure penalty
            # 3. Patient is not infected -> this is the desired behavior, no immediate reward or penalty, but this gives the AMR balloons some time to deflate, also avoids the chance of incurring adverse effects.
            
            if patient_infected:
                # First check for spontaneous recovery (patient-specific probability)
                spontaneous_recovery_occurred = rng.random() < recovery_without_treatment_prob
                
                if spontaneous_recovery_occurred:
                    # Patient recovers without treatment -> grant clinical benefit
                    clinical_benefit_occurred = True
                    reward += self.abx_clinical_reward_penalties_info_dict['normalized_clinical_benefit_reward'] * benefit_value_multiplier
                else:
                    # No spontaneous recovery -> check for clinical failure with scaled probability
                    clinical_failure_occurred = rng.random() < scaled_failure_prob
                    
                    if clinical_failure_occurred:
                        reward += self.abx_clinical_reward_penalties_info_dict['normalized_clinical_failure_penalty'] * failure_value_multiplier
        
        # Marginal AMR contribution (small penalty for prescribing)
        if antibiotic_is_prescribed:
            reward -= self.epsilon * delta_visible_amr
        
        if return_clinical_benefit_adverse_event_occurrence:
            return reward, clinical_benefit_occurred, clinical_failure_occurred, adverse_effects_occurred
        return reward
    
    def calculate_expected_individual_reward(
        self,
        patient: Patient,
        antibiotic_name: str,
        visible_amr_level: float,
        delta_visible_amr: float
    ) -> float:
        """
        Calculate expected (deterministic) reward for a single patient-action pair.
        
        This method computes the mathematical expectation of the reward without
        performing any random sampling. It is designed for value iteration, analytical
        equilibrium analysis, and MDP solving on homogeneous patient populations.
        
        The Patient object must contain TRUE attribute values (not observed/noisy versions).
        These ground-truth values are used to compute expected clinical outcomes and rewards.
        
        For a given patient and action, the expected reward is the sum of:
        - Expected clinical benefit (if infected & sensitive & treatment works)
        - Expected clinical failure penalty (if infected & resistant & failure occurs)
        - Expected adverse effects (if prescribing)
        - Expected AMR penalty (if prescribing)
        
        Args:
            patient (Patient): Patient object with TRUE attributes (not observed):
                - prob_infected: TRUE probability patient is infected ∈ [0,1]
                - benefit_value_multiplier: TRUE patient-specific benefit scaling
                - failure_value_multiplier: TRUE patient-specific failure scaling
                - benefit_probability_multiplier: TRUE scales base benefit probability
                - failure_probability_multiplier: TRUE scales base failure probability
                - recovery_without_treatment_prob: TRUE spontaneous recovery probability
            antibiotic_name (str): Name of antibiotic to prescribe, or 'no_treatment'
                for no prescription. Must be in configured antibiotic set or 'no_treatment'.
            visible_amr_level (float): Current AMR level for this antibiotic ∈ [0,1].
                Used to compute sensitivity probability pS = 1 - visible_amr_level.
                Ignored if antibiotic_name == 'no_treatment'.
            delta_visible_amr (float): Marginal AMR contribution (dose) if prescribing.
                Used in epsilon penalty term. Ignored if antibiotic_name == 'no_treatment'.
        
        Returns:
            float: Expected reward value. No randomness involved.
        
        Raises:
            ValueError: If antibiotic_name is not in configured set and not 'no_treatment'.
        
        Expected Reward Formulas:
        
        **Prescribe antibiotic (antibiotic_name != 'no_treatment'):**
            E[reward] = pI×pS×pB×RB×vB + pI×(1−pS)×pF×RF×vF + pAE×AE − ε×δ
            
            Where:
            - pI = patient.prob_infected
            - pS = 1 - amr_level (sensitivity probability)
            - pB = clamp(base_benefit_prob × benefit_probability_multiplier, 0, 1)
            - pF = clamp(base_failure_prob × failure_probability_multiplier, 0, 1)
            - RB = normalized clinical_benefit_reward
            - RF = normalized clinical_failure_penalty
            - vB = patient.benefit_value_multiplier
            - vF = patient.failure_value_multiplier
            - pAE = adverse_effect_probability[antibiotic_name]
            - AE = normalized adverse_effect_penalty[antibiotic_name]
            - ε = epsilon (AMR penalty weight)
            - δ = delta_amr (marginal AMR contribution)
        
        **No treatment (antibiotic_name == 'no_treatment'):**
            E[reward] = pI×r×RB×vB + pI×(1−r)×pF×RF×vF
            
            Where:
            - r = patient.recovery_without_treatment_prob
            - No adverse effects term (no prescription)
            - No epsilon penalty (no AMR contribution)
        
        Example:
            >>> # Homogeneous patient population for value iteration
            >>> patient = Patient(
            ...     prob_infected=0.5,
            ...     benefit_value_multiplier=1.0,
            ...     failure_value_multiplier=1.0,
            ...     benefit_probability_multiplier=1.0,
            ...     failure_probability_multiplier=1.0,
            ...     recovery_without_treatment_prob=0.1,
            ...     # ... _obs versions ...
            ... )
            >>> expected_r = rc.calculate_expected_individual_reward(
            ...     patient=patient,
            ...     antibiotic_name='A',
            ...     amr_level=0.3,  # 30% resistance
            ...     delta_amr=0.02,  # Dose contribution
            ... )
            >>> # For validation: MC mean should match expected with large samples
            >>> mc_samples = [rc.calculate_individual_reward(...) for _ in range(50000)]
            >>> assert abs(np.mean(mc_samples) - expected_r) < 1e-2
        
        See Also:
            calculate_individual_reward: Stochastic version with RNG sampling
            calculate_expected_reward: Batch version for multiple patients
        """
        # Determine whether an antibiotic is prescribed
        antibiotic_is_prescribed = antibiotic_name != 'no_treatment'

        # Clamp-scaled probabilities to [0,1] consistent with stochastic method
        base_benefit_prob = self.abx_clinical_reward_penalties_info_dict['clinical_benefit_probability']
        base_failure_prob = self.abx_clinical_reward_penalties_info_dict['clinical_failure_probability']
        pB = min(1.0, max(0.0, base_benefit_prob * float(patient.benefit_probability_multiplier)))
        pF = min(1.0, max(0.0, base_failure_prob * float(patient.failure_probability_multiplier)))

        # Normalized clinical rewards are always used
        RB = self.abx_clinical_reward_penalties_info_dict['normalized_clinical_benefit_reward']
        RF = self.abx_clinical_reward_penalties_info_dict['normalized_clinical_failure_penalty']

        pI = float(patient.prob_infected)
        vB = float(patient.benefit_value_multiplier)
        vF = float(patient.failure_value_multiplier)

        expected_reward = 0.0

        if antibiotic_is_prescribed:
            if antibiotic_name not in self.antibiotic_names:
                raise ValueError(f"Antibiotic '{antibiotic_name}' not found in abx_clinical_reward_penalties_info_dict.")

            # Sensitivity probability derived from AMR level for this antibiotic
            pS = 1.0 - float(visible_amr_level)

            # Benefit and failure expected values
            expected_reward += pI * pS * pB * RB * vB
            expected_reward += pI * (1.0 - pS) * pF * RF * vF

            # Adverse effects expectation (depends on specific antibiotic)
            ae_penalty = self.abx_clinical_reward_penalties_info_dict['abx_adverse_effects_info'][antibiotic_name]['normalized_adverse_effect_penalty']
                
            pAE = self.abx_clinical_reward_penalties_info_dict['abx_adverse_effects_info'][antibiotic_name]['adverse_effect_probability']
            
            expected_reward += float(pAE) * float(ae_penalty)

            # Marginal AMR penalty expectation (deterministic given delta)
            expected_reward -= self.epsilon * float(delta_visible_amr)
        else:
            # No treatment branch: spontaneous recovery benefit and potential failure penalty
            r_spont = float(patient.recovery_without_treatment_prob)
            expected_reward += pI * r_spont * RB * vB
            expected_reward += pI * (1.0 - r_spont) * pF * RF * vF

        return float(expected_reward)

    def calculate_community_reward(self, visible_amr_levels: Dict[str, float]) -> float:
        """
        Calculate community-level reward based on visible (observed) AMR burden.
        
        Uses the visible AMR levels (which may be delayed or have observation noise)
        rather than ground-truth actual levels. This ensures agents only receive feedback
        based on what they can observe, creating realistic partial observability.
        
        Args:
            visible_amr_levels: Dictionary mapping antibiotic names to visible AMR levels [0, 1].
                These are the observed/visible resistance estimates, not ground truth.
        Returns:
            Community reward (negative sum of visible AMR levels, higher visible AMR = lower reward)
        """
        total_amr = sum(visible_amr_levels.values())
        return -total_amr
    
    # I actually don't need a 'calculate_expected_community_reward' method, because the community reward is deterministic given the visible_amr_levels.
    
    def calculate_reward(self, patients: List[Patient], actions: np.ndarray, antibiotic_names: List[str], visible_amr_levels: Dict[str, float], delta_visible_amr_per_antibiotic: Dict[str, float], rng: Optional[np.random.Generator] = None) -> tuple[float, Dict]:
        """Calculate total composite reward for a timestep with multiple patients.
        
        Computes lambda-weighted combination of individual patient rewards (clinical
        benefit minus adverse events) and community AMR penalty. For each patient,
        samples infection status and antibiotic sensitivity stochastically, then
        applies patient-specific multipliers (benefit_value, failure_value, etc.).
        
        IMPORTANT: The Patient objects passed to this method must contain TRUE attribute
        values (not observed/noisy versions). These ground-truth values are used to compute
        actual clinical outcomes and reward. The agent observes potentially biased or noisy
        estimates, but rewards are calculated on ground truth.
        
        Reward decomposition:
            total_reward = (1-λ) × sum(individual_rewards) + λ × community_reward
        
        Where:
            individual_reward = clinical_benefit - adverse_effect + ε × marginal_amr
            community_reward = -sum(visible AMR levels across all antibiotics)
        
        Args:
            patients (List[Patient]): List of Patient objects with TRUE attributes
                (prob_infected, benefit_value_multiplier, etc.), NOT observed versions.
            actions (np.ndarray): Action array, shape (num_patients,) or
                (num_patients, num_antibiotics) depending on action mode. Each element
                is antibiotic index or no_treatment_action (typically num_antibiotics).
            antibiotic_names (List[str]): Ordered list of antibiotic names for indexing.
            visible_amr_levels (Dict[str, float]): Current visible AMR levels per antibiotic, ∈ [0,1].
                These are the observed/potentially-delayed resistance estimates. Used for:
                1. Computing community reward penalty (based on visible burden, not ground truth)
            delta_visible_amr_per_antibiotic (Dict[str, float]): Marginal AMR contribution
                (dose) per antibiotic for this step. Used in epsilon penalty term.
        
        Returns:
            tuple[float, Dict]: (total_reward, info_dict) where:
                - total_reward: Scalar composite reward value
                - info_dict: Detailed breakdown with keys: 'total_reward',
                  'individual_reward', 'community_reward', 'patient_outcomes',
                  'pts_w_clinical_benefits', 'pts_w_clinical_failures',
                  'pts_w_adverse_events', 'mean_individual_reward'
        
        Example:
            >>> rc = RewardCalculator(config)
            >>> reward, info = rc.calculate_reward(patients, actions, antibiotic_names,
            ...                                      visible_amr_levels, delta_visible_amr_per_antibiotic)
            >>> print(f"Total: {reward:.2f}, Individual: {info['individual_reward']:.2f}")
        """
        # RNG resolution: enforce explicit RNG when environment-owned
        if not self._standalone and rng is None:
            raise ValueError(
                "RewardCalculator is environment-owned and requires explicit rng argument. "
                "Pass rng=env.np_random when calling from outside environment."
            )
        rng = rng if rng is not None else self.rng
        if rng is None:
            raise ValueError("RNG must be provided either via argument or initialized on RewardCalculator")

        num_patients = len(patients)
        num_antibiotics = len(antibiotic_names)
        
        # Extract patient attributes into arrays
        benefit_value_multipliers = np.array([p.benefit_value_multiplier for p in patients], dtype=np.float32)
        failure_value_multipliers = np.array([p.failure_value_multiplier for p in patients], dtype=np.float32)
        benefit_probability_multipliers = np.array([p.benefit_probability_multiplier for p in patients], dtype=np.float32)
        failure_probability_multipliers = np.array([p.failure_probability_multiplier for p in patients], dtype=np.float32)
        recovery_without_treatment_probs = np.array([p.recovery_without_treatment_prob for p in patients], dtype=np.float32)

        patients_actually_infected = np.array(
            [bool(p.infection_status) for p in patients],
            dtype=bool,
        )

        # Map actions to names once for reuse
        antibiotic_actions_str = [self.index_to_abx_name[int(action)] for action in actions]

        abx_sensitivity_boolean_matrix = np.zeros((num_patients, num_antibiotics), dtype=bool)
        for i, patient in enumerate(patients):
            if not hasattr(patient, 'abx_sensitivity_dict'):
                raise ValueError("Patient missing required attribute 'abx_sensitivity_dict'")
            abx_sensitivity_dict = patient.abx_sensitivity_dict
            missing_abx = [abx for abx in antibiotic_names if abx not in abx_sensitivity_dict]
            if missing_abx:
                raise ValueError(
                    "Patient abx_sensitivity_dict missing antibiotics: "
                    f"{missing_abx}. Expected: {antibiotic_names}"
                )
            for abx_name in antibiotic_names:
                abx_index = self.abx_name_to_index[abx_name]
                abx_sensitivity_boolean_matrix[i, abx_index] = bool(abx_sensitivity_dict[abx_name])
        
        # First, categorize outcomes
        outcomes_dict = self.categorize_outcomes(
            patient_infection_status_boolean=patients_actually_infected,
            actions=actions,
            abx_sensitivity_boolean_matrix=abx_sensitivity_boolean_matrix
        )
        
        individual_rewards = []
        pts_w_clinical_benefits = 0
        pts_w_clinical_failures = 0
        pts_w_adverse_events = 0
        
        # Now, iterate through each patient to calculate individual rewards.
        
        for i in range(num_patients):
            infection_sensitive_to_prescribed_abx = None
            if antibiotic_actions_str[i] != 'no_treatment':
                infection_sensitive_to_prescribed_abx = bool(
                    patients[i].abx_sensitivity_dict[antibiotic_actions_str[i]]
                )
            individual_reward, clinical_benefit_occurred, clinical_failure_occurred, adverse_effects_occurred = self.calculate_individual_reward(
                patient_infected=patients_actually_infected[i],
                antibiotic_name=antibiotic_actions_str[i],
                infection_sensitive_to_prescribed_abx=infection_sensitive_to_prescribed_abx,
                delta_visible_amr=delta_visible_amr_per_antibiotic[antibiotic_actions_str[i]] if antibiotic_actions_str[i] != 'no_treatment' else None,
                return_clinical_benefit_adverse_event_occurrence=True,
                benefit_value_multiplier=float(benefit_value_multipliers[i]),
                failure_value_multiplier=float(failure_value_multipliers[i]),
                benefit_probability_multiplier=float(benefit_probability_multipliers[i]),
                failure_probability_multiplier=float(failure_probability_multipliers[i]),
                recovery_without_treatment_prob=float(recovery_without_treatment_probs[i]),
                rng=rng,
            )
            
            individual_rewards.append(individual_reward)
            pts_w_clinical_benefits += int(clinical_benefit_occurred)
            pts_w_clinical_failures += int(clinical_failure_occurred)
            pts_w_adverse_events += int(adverse_effects_occurred)

        total_individual_reward = sum(individual_rewards)
        normalized_individual_reward = total_individual_reward / num_patients if num_patients > 0 else 0.0

        community_reward = self.calculate_community_reward(visible_amr_levels)
        normalized_community_reward = community_reward / num_antibiotics if num_antibiotics > 0 else 0.0

        total_reward = (
            self.lambda_weight * normalized_community_reward +
            (1.0 - self.lambda_weight) * normalized_individual_reward
        )

        info = {
            'total_reward': total_reward,
            'overall_individual_reward_component': total_individual_reward,
            'normalized_individual_reward': normalized_individual_reward,
            'overall_community_reward_component': community_reward,
            'normalized_community_reward': normalized_community_reward,
            'individual_rewards': individual_rewards,
            'count_clinical_benefits': pts_w_clinical_benefits,
            'count_clinical_failures': pts_w_clinical_failures,
            'count_adverse_events': pts_w_adverse_events,
            'patients_actually_infected': patients_actually_infected.tolist(),
            'outcomes_breakdown': outcomes_dict,
        }

        return total_reward, info

    def calculate_expected_reward(
        self,
        patients: List[Patient],
        actions: np.ndarray,
        antibiotic_names: List[str],
        visible_amr_levels: Dict[str, float],
        delta_visible_amr_per_antibiotic: Dict[str, float]
    ) -> float:
        """
        Calculate expected (deterministic) composite reward for multiple patients.
        
        This method computes the mathematical expectation of the total reward without
        performing any random sampling. It mirrors the normalization and lambda-weighting
        logic of calculate_reward() exactly, but uses deterministic expected individual
        rewards instead of stochastic sampling.
        
        Designed for value iteration and analytical equilibrium analysis on homogeneous
        populations, where we need exact expected rewards for dynamic programming.
        
        The Patient objects must contain TRUE attribute values (not observed versions).
        These ground-truth values are used in expectation calculations.
        
        The composite reward formula is:
            total_expected = λ × normalized_community + (1−λ) × normalized_individual
        
        Where:
            normalized_individual = sum(E[individual_reward_i]) / num_patients
            normalized_community = -sum(visible_amr_levels) / num_antibiotics (visible levels only)
        
        Args:
            patients (List[Patient]): List of Patient objects with TRUE attributes
                (prob_infected, benefit_value_multiplier, etc.), NOT observed versions.
                For homogeneous populations, all patients have identical attributes.
            actions (np.ndarray): Action array, shape (num_patients,). Each element
                is an integer index mapping to an antibiotic or no_treatment.
            antibiotic_names (List[str]): Ordered list of antibiotic names for
                mapping action indices to names.
            visible_amr_levels (Dict[str, float]): Visible AMR levels per antibiotic ∈ [0,1].
                These are the observed/potentially-delayed resistance estimates.
                Used for:
                1. Computing sensitivity probabilities (pS = 1 - visible_amr_level)
                2. Computing community reward component (based on visible burden)
            delta_visible_amr_per_antibiotic (Dict[str, float]): Marginal AMR contributions
                (dose) per antibiotic for this step. Used in epsilon penalty term.
        
        Returns:
            float: Expected composite reward (deterministic, no randomness).
        
        Example:
            >>> # Homogeneous population (all patients identical)
            >>> patients = [create_homogeneous_patient() for _ in range(10)]
            >>> actions = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # Mix of actions
            >>> expected_total = rc.calculate_expected_reward(
            ...     patients=patients,
            ...     actions=actions,
            ...     antibiotic_names=['A', 'B'],
            ...     visible_amr_levels={'A': 0.3, 'B': 0.4},
            ...     delta_visible_amr_per_antibiotic={'A': 0.02, 'B': 0.03},
            ... )
            >>> # For validation: MC mean should match expected with large samples
            >>> mc_rewards = []
            >>> rng = np.random.default_rng(42)
            >>> for _ in range(50000):
            ...     r, _ = rc.calculate_reward(patients, actions, antibiotic_names,
            ...                                 visible_amr_levels, delta_visible_amr_per_antibiotic, rng=rng)
            ...     mc_rewards.append(r)
            >>> assert abs(np.mean(mc_rewards) - expected_total) < 1e-2
        
        Notes:
            - No RNG parameter: expected reward is deterministic
            - Normalization and lambda-weighting match calculate_reward() exactly
            - Works for heterogeneous populations but designed for homogeneous use
            - Both community and individual components use visible_amr_levels only
            - This ensures partial observability: agents only get feedback from observations
        
        See Also:
            calculate_reward: Stochastic version with RNG sampling
            calculate_expected_individual_reward: Per-patient expected reward
        """
        # Expected composite reward with deterministic aggregation mirroring calculate_reward normalization
        num_patients = len(patients)
        num_antibiotics = len(antibiotic_names)

        # Map actions to antibiotic names
        antibiotic_actions_str = [self.index_to_abx_name[int(action)] for action in actions]

        # For expected reward we assume homogeneous population usage by caller, but we compute generically
        individual_expected_rewards: List[float] = []

        for i, patient in enumerate(patients):
            abx_name = antibiotic_actions_str[i]
            if abx_name == 'no_treatment':
                amr_level = 0.0  # ignored in no-treatment branch
                delta = 0.0
            else:
                amr_level = float(visible_amr_levels[abx_name])
                delta = float(delta_visible_amr_per_antibiotic[abx_name])
                
            r = self.calculate_expected_individual_reward(
                patient=patient,
                antibiotic_name=abx_name,
                visible_amr_level=amr_level,
                delta_visible_amr=delta,
            )
            individual_expected_rewards.append(r)

        total_individual_expected = float(sum(individual_expected_rewards))
        normalized_individual_expected = total_individual_expected / num_patients if num_patients > 0 else 0.0

        # Community term is deterministic, computed using visible AMR levels (not ground truth)
        community_reward = self.calculate_community_reward(visible_amr_levels)
        normalized_community_reward = community_reward / num_antibiotics if num_antibiotics > 0 else 0.0

        total_expected = (
            self.lambda_weight * normalized_community_reward +
            (1.0 - self.lambda_weight) * normalized_individual_expected
        )

        return float(total_expected)

    def categorize_outcomes(self, patient_infection_status_boolean: np.ndarray, actions: np.ndarray, abx_sensitivity_boolean_matrix: np.ndarray) -> Dict:
        """Categorize patient outcomes by TRUE infection status and treatment effectiveness.
        
        Helper function for calculate_reward that categorizes patients into outcome buckets
        based on their TRUE (ground-truth) infection status, antibiotic prescriptions, and
        actual antibiotic sensitivity. Used for reward computation and diagnostic logging.
        
        Args:
            patient_infection_status_boolean (np.ndarray): Boolean array of TRUE infection
                status for each patient (True = infected, False = not infected). This is
                ground truth, not the observed/noisy version.
            actions (np.ndarray): Array of actions taken, shape (num_patients,). Each
                element is antibiotic index or no_treatment_action.
            abx_sensitivity_boolean_matrix (np.ndarray): Boolean matrix, shape
                (num_patients, num_antibiotics), indicating TRUE sensitivity of each patient
                infection to each antibiotic (True = sensitive, False = resistant). This is
                ground truth, not the observed version.
        
        Returns:
            Dict: Outcome counts with keys:
                - 'not_infected_no_treatment': Count of uninfected patients untreated
                - 'not_infected_treated': Count of uninfected patients treated
                - 'infected_no_treatment': Count of infected patients untreated
                - 'infected_treated': Dict mapping antibiotic names to dicts with keys:
                    - 'sensitive_infection_treated': Successfully treated (sensitive)
                    - 'resistant_infection_treated': Unsuccessfully treated (resistant)
        """
        
        # First, translate the actions into antibiotic names using the index_to_abx_name mapping:
        antibiotic_actions = [self.index_to_abx_name[int(action)] for action in actions]
        
        outcomes_dict = {
            'not_infected_no_treatment': 0,
            'not_infected_treated': 0,
            'infected_no_treatment': 0,
            'infected_treated': {},
        }
        
        for antibiotic in self.antibiotic_names:
            outcomes_dict['infected_treated'][antibiotic] = {
                'sensitive_infection_treated': 0,
                'resistant_infection_treated': 0,
            }
        
        # Second, iterate through each patient and categorize them accordingly:
        for i, patient_is_infected in enumerate(patient_infection_status_boolean):
            
            # Get the action for this patient:
            abx_name = antibiotic_actions[i]
            
            # If patient is not infected:
            if not patient_is_infected:
                if abx_name == 'no_treatment':
                    outcomes_dict['not_infected_no_treatment'] += 1
                else:
                    outcomes_dict['not_infected_treated'] += 1
            else:
                # Patient is infected
                if abx_name == 'no_treatment':
                    outcomes_dict['infected_no_treatment'] += 1
                else:
                    # Patient is infected and treated; check sensitivity
                    abx_index = self.abx_name_to_index[abx_name]
                    infection_sensitive_to_prescribed_abx = abx_sensitivity_boolean_matrix[i, abx_index]
                    
                    if infection_sensitive_to_prescribed_abx:
                        outcomes_dict['infected_treated'][abx_name]['sensitive_infection_treated'] += 1
                    else:
                        outcomes_dict['infected_treated'][abx_name]['resistant_infection_treated'] += 1
        
        return outcomes_dict
    
    def _export_init_kwargs(self, include_lambda: bool = True) -> Dict:
        """Collect configuration dict for cloning/conversion helpers (new API)."""
        config = {
            'abx_clinical_reward_penalties_info_dict': self.abx_clinical_reward_penalties_info_dict,
            'epsilon': self.epsilon,
            'seed': getattr(self, 'seed', None),
        }
        if include_lambda:
            config['lambda_weight'] = self.lambda_weight
        return config

    def clone_with_lambda(self, lambda_weight: float) -> 'RewardCalculator':
        """Create a new RewardCalculator with the same parameters but a different lambda_weight.

        The RNG state is copied so stochastic draws continue consistently from the source model.
        """
        config = self._export_init_kwargs(include_lambda=False)
        config['lambda_weight'] = lambda_weight
        clone = RewardCalculator(config=config)
        clone.rng = np.random.default_rng()
        clone.rng.bit_generator.state = self.rng.bit_generator.state
        return clone
    
class IndividualOnlyReward(RewardCalculator):
    """Reward model that only considers individual patient outcomes (lambda=0)."""

    @classmethod
    def from_existing(cls, model: RewardCalculator) -> 'IndividualOnlyReward':
        config = model._export_init_kwargs(include_lambda=False)
        return cls(config=config)

    def __init__(self, config: Dict):
        config = dict(config)
        config['lambda_weight'] = 0.0
        super().__init__(config=config)


class CommunityOnlyReward(RewardCalculator):
    """Reward model that only considers community AMR burden (lambda=1)."""

    @classmethod
    def from_existing(cls, model: RewardCalculator) -> 'CommunityOnlyReward':
        config = model._export_init_kwargs(include_lambda=False)
        return cls(config=config)

    def __init__(self, config: Dict):
        config = dict(config)
        config['lambda_weight'] = 1.0
        super().__init__(config=config)


class BalancedReward(RewardCalculator):
    """Reward model with equal weighting of individual and community (lambda=0.5)."""

    @classmethod
    def from_existing(cls, model: RewardCalculator) -> 'BalancedReward':
        config = model._export_init_kwargs(include_lambda=False)
        return cls(config=config)

    def __init__(self, config: Dict):
        config = dict(config)
        config['lambda_weight'] = 0.5
        super().__init__(config=config)
