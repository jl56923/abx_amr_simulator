"""Fixed prescribing rule policies (heuristic baselines) for ABX-AMR envs.

Factored out of workspace/scripts/pipeline/train_w_fixed_prescribing_rules.py so both
the single-agent FP driver and the MARL parallel-env FP evaluator can share one
definition. Each policy exposes an SB3-style ``predict(obs) -> (action_names, None)``
returning antibiotic STRING names; callers translate names to action indices.
"""

import numpy as np

from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker,
)


def _parse_observation_for_single_patient(
    obs: np.ndarray,
    num_antibiotics: int,
    antibiotic_names: list,
    visible_patient_attributes: list,
) -> tuple:
    """
    Parse flattened observation array to extract patient attributes and AMR levels.
    
    Observation structure (single patient):
        [patient_attr_0, patient_attr_1, ..., patient_attr_M,
         amr_level_0, amr_level_1, ..., amr_level_K,
         (optional) steps_since_amr_update,
         (optional) temporal_features]
    
    Args:
        obs: Flattened observation array
        num_antibiotics: Number of antibiotics in environment
        antibiotic_names: List of antibiotic names (e.g., ['A', 'B'])
        visible_patient_attributes: List of patient attribute names in observation order
    
    Returns:
        tuple: (patient_dict, amr_dict)
            - patient_dict: Dict mapping attribute name → observed value
            - amr_dict: Dict mapping antibiotic name → AMR level
    
    Example:
        >>> obs = np.array([0.8, 1.2, 0.3, 0.5])  # [prob_infected, benefit_mult, amr_A, amr_B]
        >>> patient_dict, amr_dict = _parse_observation_for_single_patient(
        ...     obs=obs, num_antibiotics=2, antibiotic_names=['A', 'B'],
        ...     visible_patient_attributes=['prob_infected', 'benefit_value_multiplier'])
        >>> patient_dict
        {'prob_infected': 0.8, 'benefit_value_multiplier': 1.2}
        >>> amr_dict
        {'A': 0.3, 'B': 0.5}
    """
    # Number of patient attributes
    n_patient_attrs = len(visible_patient_attributes)
    
    # Extract patient attributes (first n_patient_attrs values)
    patient_values = obs[:n_patient_attrs]
    patient_dict = {attr: float(val) for attr, val in zip(visible_patient_attributes, patient_values)}
    
    # Extract AMR levels (next num_antibiotics values)
    amr_start = n_patient_attrs
    amr_end = amr_start + num_antibiotics
    amr_values = obs[amr_start:amr_end]
    amr_dict = {abx_name: float(amr_val) for abx_name, amr_val in zip(antibiotic_names, amr_values)}
    
    return patient_dict, amr_dict


# ============================================================================
# FIXED PRESCRIBING RULES CLASSES
# ============================================================================

class FixedPrescribingRules:
    """Base class for fixed prescribing rules.
    
    IMPORTANT: These policies return antibiotic STRING NAMES (e.g., 'A', 'B', 'no_treatment'),
    not action indices. The main script translates names to indices using
    reward_calculator.abx_name_to_index mapping.
    """
    
    def __init__(self, config, reward_calculator, num_patients_per_time_step=None, visible_patient_attributes=None, antibiotic_names=None):
        self.config = config
        self.reward_calculator = reward_calculator
        
        # REQUIRED: antibiotic_names must be provided explicitly (no defaults)
        if antibiotic_names is None:
            raise ValueError(
                "antibiotic_names is REQUIRED for FixedPrescribingRules. "
                "Must be passed explicitly from reward_calculator.antibiotic_names"
            )
        self.antibiotic_names = antibiotic_names
        self.num_antibiotics = len(antibiotic_names)
        
        # REQUIRED: num_patients_per_time_step must be provided explicitly (no defaults)
        if num_patients_per_time_step is None:
            raise ValueError(
                "num_patients_per_time_step is REQUIRED for FixedPrescribingRules. "
                "Must be passed from environment"
            )
        self.num_patients_per_time_step = num_patients_per_time_step
        
        # REQUIRED: visible_patient_attributes must be provided explicitly (no defaults)
        if visible_patient_attributes is None:
            raise ValueError(
                "visible_patient_attributes is REQUIRED for FixedPrescribingRules. "
                "Must be passed from patient_generator"
            )
        self.visible_patient_attributes = visible_patient_attributes
    
    def predict(self, obs, deterministic=True):
        """
        Predict antibiotic STRING NAMES given observation(s).
        
        NOTE: Unlike many RL policies, FixedPrescribingRules return antibiotic STRING NAMES
        (e.g., 'A', 'B', 'no_treatment'), not action indices. The calling code translates
        names to indices using reward_calculator.abx_name_to_index mapping.
        
        Args:
            obs: Observation (1D array or dict depending on environment)
            deterministic: Ignored (fixed prescribing rules are deterministic by definition)
            
        Returns:
            tuple: (action_names, None) where action_names is numpy array of antibiotic strings
                   (e.g., np.array(['A', 'B', 'no_treatment', ...])), None is state placeholder
        """
        raise NotImplementedError
    
    def _parse_observation_all_patients(self, obs):
        """Parse observation to extract per-patient attributes and AMR levels.
        
        Returns:
            tuple: (patient_obs_list, amr_levels) where:
                - patient_obs_list: list of dicts with attributes for each patient
                - amr_levels: dict of {abx_name: amr_level} for the current timestep
        """
        obs_arr = np.array(obs) if not isinstance(obs, np.ndarray) else obs
        n_attrs = len(self.visible_patient_attributes)
        
        # Extract per-patient observations
        patient_obs_list = []
        for patient_idx in range(self.num_patients_per_time_step):
            start_idx = patient_idx * n_attrs
            end_idx = start_idx + n_attrs
            patient_attrs = obs_arr[start_idx:end_idx]
            patient_dict = {
                attr: float(val) 
                for attr, val in zip(self.visible_patient_attributes, patient_attrs)
            }
            patient_obs_list.append(patient_dict)
        
        # Extract AMR levels (after all patient attributes)
        amr_start = self.num_patients_per_time_step * n_attrs
        amr_end = amr_start + self.num_antibiotics
        amr_values = obs_arr[amr_start:amr_end]
        amr_levels = {
            abx_name: float(amr_val)
            for abx_name, amr_val in zip(self.antibiotic_names, amr_values)
        }
        
        return patient_obs_list, amr_levels
    
    def reset(self):
        """Reset any internal state (e.g., for escalation tracking)."""
        pass


class ThresholdPolicy(FixedPrescribingRules):
    """Prescribe if any visible AMR level is below threshold."""
    
    def __init__(self, config, reward_calculator, threshold=0.5, num_patients_per_time_step=None, **kwargs):
        super().__init__(config, reward_calculator, num_patients_per_time_step=num_patients_per_time_step, **kwargs)
        self.threshold = threshold
        self.policy_name = "threshold"
        self.policy_params = {"threshold": threshold}
    
    def predict(self, obs, deterministic=True):
        """Choose lowest-AMR antibiotic if any AMR < threshold, else no_treatment (by NAME)."""
        patient_obs_list, amr_levels = self._parse_observation_all_patients(obs)
        
        # Determine action for each patient (return antibiotic NAMES)
        actions = []
        amr_array = np.array([amr_levels[abx] for abx in self.antibiotic_names])
        
        for patient_dict in patient_obs_list:
            if np.any(amr_array < self.threshold):
                # Prescribe the antibiotic with lowest AMR (by name)
                best_abx_idx = np.argmin(amr_array)
                action = self.antibiotic_names[best_abx_idx]
            else:
                # No prescription
                action = 'no_treatment'
            actions.append(action)
        
        return np.array(actions), None


class RiskStratifiedPolicy(FixedPrescribingRules):
    """Only prescribe for high-risk (high infection probability) patients."""
    
    def __init__(self, config, reward_calculator, risk_threshold=0.6, num_patients_per_time_step=None, **kwargs):
        super().__init__(config, reward_calculator, num_patients_per_time_step=num_patients_per_time_step, **kwargs)
        self.risk_threshold = risk_threshold
        self.policy_name = "risk_stratified"
        self.policy_params = {"risk_threshold": risk_threshold}
    
    def predict(self, obs, deterministic=True):
        """Prescribe lowest-AMR antibiotic only if prob_infected > risk_threshold (by NAME)."""
        patient_obs_list, amr_levels = self._parse_observation_all_patients(obs)
        
        # Determine action for each patient independently (return antibiotic NAMES)
        actions = []
        amr_array = np.array([amr_levels[abx] for abx in self.antibiotic_names])
        
        for patient_dict in patient_obs_list:
            # Get this patient's infection probability
            if 'prob_infected' not in patient_dict:
                raise ValueError("prob_infected not found in patient observation. Check visible_patient_attributes.")
            prob_infected = patient_dict['prob_infected']
            
            if prob_infected > self.risk_threshold:
                # Patient is high-risk; prescribe lowest-AMR antibiotic (by name)
                best_abx_idx = np.argmin(amr_array)
                action = self.antibiotic_names[best_abx_idx]
            else:
                # Low-risk patient; no prescription
                action = 'no_treatment'
            
            actions.append(action)
        
        return np.array(actions), None


class FixedCyclingPolicy(FixedPrescribingRules):
    """Cycle through all treatment options (antibiotics + no treatment) on fixed schedule.
    
    For 1 antibiotic: cycles through [A, no_treatment]
    For 2 antibiotics: cycles through [A, B, no_treatment]
    For N antibiotics: cycles through [A, B, ..., no_treatment]
    """
    
    def __init__(self, config, reward_calculator, cycle_period=30, num_patients_per_time_step=None, **kwargs):
        super().__init__(config, reward_calculator, num_patients_per_time_step=num_patients_per_time_step, **kwargs)
        self.cycle_period = cycle_period
        self.policy_name = "fixed_cycling"
        self.policy_params = {"cycle_period": cycle_period}
        self.current_step = 0
        # Cycle through: [A, B, ..., no_treatment] by NAME
        self.cycle_options = list(self.antibiotic_names) + ['no_treatment']
    
    def predict(self, obs, deterministic=True):
        """Select action based on fixed cycling schedule for all patients (by NAME)."""
        # Increment step counter first
        self.current_step += 1
        
        # Determine which option we're currently in based on current step
        option_index = (self.current_step // self.cycle_period) % len(self.cycle_options)
        action_name = self.cycle_options[option_index]
        
        # Return same action name for all patients (as numpy array of strings)
        return np.array([action_name] * self.num_patients_per_time_step), None
    
    def reset(self):
        """Reset cycling counter at episode start."""
        self.current_step = 0


class RandomPolicy(FixedPrescribingRules):
    """Uniformly random treatment decisions (lower-bound baseline)."""
    
    def __init__(self, config, reward_calculator, num_patients_per_time_step=None, **kwargs):
        super().__init__(config, reward_calculator, num_patients_per_time_step=num_patients_per_time_step, **kwargs)
        self.policy_name = "random"
        self.policy_params = {}
        # All possible actions by NAME: each antibiotic name + 'no_treatment'
        self.all_action_names = list(self.antibiotic_names) + ['no_treatment']
    
    def predict(self, obs, deterministic=True):
        """Randomly select action from all available options by NAME for each patient."""
        # Randomly select from all action names for each patient
        # Note: This uses numpy's default RNG; if env provides seeded RNG, 
        # it will be captured in the environment's seed setting
        actions = [np.random.choice(self.all_action_names) for _ in range(self.num_patients_per_time_step)]
        return np.array(actions), None


class AlwaysPrescribePolicy(FixedPrescribingRules):
    """Always prescribe the lowest-AMR antibiotic (baseline upper bound on clinical benefit)."""
    
    def __init__(self, config, reward_calculator, num_patients_per_time_step=None, **kwargs):
        super().__init__(config, reward_calculator, num_patients_per_time_step=num_patients_per_time_step, **kwargs)
        self.policy_name = "always_prescribe"
        self.policy_params = {}
    
    def predict(self, obs, deterministic=True):
        """Always prescribe lowest-AMR antibiotic for each patient (by NAME, not index)."""
        patient_obs_list, amr_levels = self._parse_observation_all_patients(obs)
        
        # Same decision for all patients: prescribe lowest-AMR antibiotic (by name)
        amr_array = np.array([amr_levels[abx] for abx in self.antibiotic_names])
        best_abx_idx = np.argmin(amr_array)
        action_name = self.antibiotic_names[best_abx_idx]
        
        # Return same action name for all patients (as numpy array of strings)
        return np.array([action_name] * self.num_patients_per_time_step), None


class NeverPrescribePolicy(FixedPrescribingRules):
    """Never prescribe any antibiotic (baseline lower bound on clinical benefit)."""
    
    def __init__(self, config, reward_calculator, num_patients_per_time_step=None, **kwargs):
        super().__init__(config, reward_calculator, num_patients_per_time_step=num_patients_per_time_step, **kwargs)
        self.policy_name = "never_prescribe"
        self.policy_params = {}
    
    def predict(self, obs, deterministic=True):
        """Never prescribe; always return 'no_treatment' by NAME for all patients."""
        # Always prescribe 'no_treatment' (as string, not index)
        return np.array(['no_treatment'] * self.num_patients_per_time_step), None


class ExpectedRewardLowestAMRPolicy(FixedPrescribingRules):
    """Prescribe antibiotic with lowest AMR among those with E[reward] > 0 per patient.
    
    This policy uses clinical reasoning to filter actions that are expected to be
    beneficial (E[reward] > 0), then selects the one with lowest current AMR to
    minimize resistance burden. This tests whether RL adds value beyond greedy
    application of the reward model combined with AMR-awareness.
    """
    
    def __init__(
        self,
        config,
        reward_calculator,
        visible_patient_attributes=None,
        antibiotic_names=None,
        num_patients_per_time_step=None,
        heuristic_worker=None,
    ):
        super().__init__(config=config, reward_calculator=reward_calculator, num_patients_per_time_step=num_patients_per_time_step, visible_patient_attributes=visible_patient_attributes, antibiotic_names=antibiotic_names)
        self.policy_name = "expected_reward_lowest_amr"
        self.policy_params = {}
        
        # Create internal HeuristicWorker for compute_expected_reward
        # (We only use its compute_expected_reward method, not the full option logic)
        # HeuristicWorker signature: __init__(name, duration, action_thresholds, ...)
        # We only need the compute_expected_reward method, so we use dummy values
        if heuristic_worker is not None:
            self.heuristic_worker = heuristic_worker
        else:
            self.heuristic_worker = HeuristicWorker(
                name='internal_expected_reward_helper',
                duration=1,  # Dummy value, not used
                action_thresholds={},  # Dummy value, not used
                # Only used as a fallback when a patient's recovery_without_treatment_prob
                # is missing/-1. The ABX-AMR patient generators always supply this attribute,
                # so this value is effectively unused; kept explicit because the arg is required.
                default_recovery_without_treatment_prob=0.05,
            )

    def predict(self, obs, deterministic=True):
        """Select lowest-AMR antibiotic among those with positive expected reward per patient (by NAME)."""
        # Parse observation to extract all patients and AMR levels
        patient_obs_list, amr_dict = self._parse_observation_all_patients(obs)
        
        # Compute action for each patient independently
        actions = []
        
        for patient_dict in patient_obs_list:
            # Compute expected reward for each action using HeuristicWorker
            expected_rewards = self.heuristic_worker.compute_expected_reward(
                patient=patient_dict,
                antibiotic_names=self.antibiotic_names,
                current_amr_levels=amr_dict,
                reward_calculator=self.reward_calculator,
            )
            
            # Filter antibiotics with E[reward] > 0 (exclude 'no_treatment')
            positive_expected_reward_actions = []
            for abx_name in self.antibiotic_names:
                action_key = f'prescribe_{abx_name}'
                if action_key in expected_rewards and expected_rewards[action_key] > 0:
                    positive_expected_reward_actions.append(abx_name)
            
            # If no antibiotic has positive expected reward, choose no treatment
            if not positive_expected_reward_actions:
                action = 'no_treatment'
            else:
                # Among positive E[reward] antibiotics, choose one with lowest AMR (by NAME)
                best_abx = min(positive_expected_reward_actions, key=lambda abx: amr_dict[abx])
                action = best_abx
            
            actions.append(action)
        
        return np.array(actions), None


class ExpectedRewardGreedyPolicy(FixedPrescribingRules):
    """Prescribe argmax antibiotic only when at least one antibiotic has E[reward] > 0.
    
    Comparator semantics are intentionally locked:
    - choose argmax antibiotic iff any antibiotic has expected reward > 0
    - otherwise choose no_treatment

    The no_treatment expected reward is NOT part of the comparator among
    antibiotics; it is only used as the fallback when no antibiotic passes
    the positive-expected-reward gate.
    """
    
    def __init__(
        self,
        config,
        reward_calculator,
        visible_patient_attributes=None,
        antibiotic_names=None,
        num_patients_per_time_step=None,
        heuristic_worker=None,
    ):
        super().__init__(config=config, reward_calculator=reward_calculator, num_patients_per_time_step=num_patients_per_time_step, visible_patient_attributes=visible_patient_attributes, antibiotic_names=antibiotic_names)
        self.policy_name = "expected_reward_greedy"
        self.policy_params = {}

        if heuristic_worker is not None:
            self.heuristic_worker = heuristic_worker
        else:
            self.heuristic_worker = HeuristicWorker(
                name='internal_expected_reward_helper',
                duration=1,  # Dummy value, not used
                action_thresholds={},  # Dummy value, not used
                # Only used as a fallback when a patient's recovery_without_treatment_prob
                # is missing/-1. The ABX-AMR patient generators always supply this attribute,
                # so this value is effectively unused; kept explicit because the arg is required.
                default_recovery_without_treatment_prob=0.05,
            )
    
    def predict(self, obs, deterministic=True):
        """Select argmax antibiotic if any antibiotic has E[R] > 0, else no_treatment (by NAME)."""
        # Parse observation to extract all patients and AMR levels
        patient_obs_list, amr_dict = self._parse_observation_all_patients(obs)
        
        # Compute action for each patient independently
        actions = []
        
        for patient_dict in patient_obs_list:
            # Compute expected reward for each action using HeuristicWorker
            expected_rewards = self.heuristic_worker.compute_expected_reward(
                patient=patient_dict,
                antibiotic_names=self.antibiotic_names,
                current_amr_levels=amr_dict,
                reward_calculator=self.reward_calculator,
            )
            
            # Locked semantics:
            # 1) Compare ONLY antibiotics (exclude 'no_treatment')
            # 2) Choose argmax antibiotic iff best antibiotic E[R] > 0
            # 3) Otherwise fall back to 'no_treatment'
            best_action_name = 'no_treatment'
            best_expected_reward = float('-inf')

            for abx_name in self.antibiotic_names:
                action_key = f'prescribe_{abx_name}'
                if action_key in expected_rewards:
                    e_reward = expected_rewards[action_key]
                    if e_reward > best_expected_reward:
                        best_expected_reward = e_reward
                        best_action_name = abx_name

            if best_expected_reward <= 0:
                best_action_name = 'no_treatment'
            
            actions.append(best_action_name)
        
        return np.array(actions), None


POLICY_REGISTRY = {
    'threshold': ThresholdPolicy,
    'risk_stratified': RiskStratifiedPolicy,
    'fixed_cycling': FixedCyclingPolicy,
    'random': RandomPolicy,
    'always_prescribe': AlwaysPrescribePolicy,
    'never_prescribe': NeverPrescribePolicy,
    'expected_reward_lowest_amr': ExpectedRewardLowestAMRPolicy,
    'expected_reward_greedy': ExpectedRewardGreedyPolicy,
}
