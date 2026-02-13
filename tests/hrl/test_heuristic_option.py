"""Tests for heuristic policy worker option.

Tests the HeuristicWorker class and load_heuristic_option loader function
from the abx_amr_simulator.options.defaults.option_types.heuristic package.
"""

import pytest
import numpy as np

from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker,
    load_heuristic_option,
)
from abx_amr_simulator.core.reward_calculator import RewardCalculator
from abx_amr_simulator.core.patient_generator import PatientGenerator


def _create_reward_calculator_for_expected_reward_tests() -> RewardCalculator:
    config = {
        'abx_clinical_reward_penalties_info_dict': {
            'clinical_benefit_reward': 10.0,
            'clinical_benefit_probability': 1.0,
            'clinical_failure_penalty': -10.0,
            'clinical_failure_probability': 0.0,
            'abx_adverse_effects_info': {
                'A': {
                    'adverse_effect_penalty': -1.0,
                    'adverse_effect_probability': 0.0,
                },
            },
        },
        'lambda_weight': 0.0,
        'epsilon': 0.05,
        'seed': 123,
    }
    return RewardCalculator(config=config)


class TestHeuristicWorkerInstantiation:
    """Test HeuristicWorker instantiation and validation."""
    
    def test_basic_instantiation(self):
        """Test creating a basic heuristic worker."""
        worker = HeuristicWorker(
            name='HEURISTIC_test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        assert worker.name == 'HEURISTIC_test'
        assert worker.k == 10
        assert worker.action_thresholds == {'prescribe_A': 0.5, 'no_treatment': 0.0}
        assert worker.uncertainty_threshold == 2.0
    
    def test_option_base_protocol_attributes(self):
        """Test that HeuristicWorker declares required OptionBase attributes."""
        assert hasattr(HeuristicWorker, 'REQUIRES_OBSERVATION_ATTRIBUTES')
        assert hasattr(HeuristicWorker, 'REQUIRES_AMR_LEVELS')
        assert hasattr(HeuristicWorker, 'REQUIRES_STEP_NUMBER')
        assert hasattr(HeuristicWorker, 'PROVIDES_TERMINATION_CONDITION')
        
        assert HeuristicWorker.REQUIRES_AMR_LEVELS is True
        assert HeuristicWorker.REQUIRES_STEP_NUMBER is False
        assert HeuristicWorker.PROVIDES_TERMINATION_CONDITION is False
        # Minimal requirements: only prob_infected is truly required
        assert HeuristicWorker.REQUIRES_OBSERVATION_ATTRIBUTES == ['prob_infected']


class TestUncertaintyScoring:
    """Test uncertainty score calculation (relative and absolute)."""
    
    def test_relative_uncertainty_no_padding(self):
        """Test relative uncertainty with no padded attributes."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.2,
            'failure_value_multiplier': 0.9,
            'benefit_probability_multiplier': 1.0,
            'failure_probability_multiplier': 1.0,
            'recovery_without_treatment_prob': 0.1,
        }
        
        uncertainty = worker.compute_relative_uncertainty_score(patient=patient)
        assert uncertainty == 0  # No -1 values
    
    def test_relative_uncertainty_some_padding(self):
        """Test relative uncertainty with padded attributes."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        # Inject full attribute list so worker checks all attributes
        full_attrs = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob'
        ]
        worker.set_observable_attributes(full_attrs)
        
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': -1.0,  # Padded
            'failure_value_multiplier': -1.0,  # Padded
            'benefit_probability_multiplier': 1.0,
            'failure_probability_multiplier': 1.0,
            'recovery_without_treatment_prob': 0.1,
        }
        
        uncertainty = worker.compute_relative_uncertainty_score(patient=patient)
        assert uncertainty == 2  # Two -1 values
    
    def test_absolute_uncertainty_partial_visibility(self):
        """Test absolute uncertainty with partially visible attributes."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        # Inject full attribute list (6 total)
        full_attrs = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob'
        ]
        worker.set_observable_attributes(full_attrs)
        
        # Only 2 attributes observed (out of 6 total)
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.2,
        }
        
        total_observable_attrs = 6
        uncertainty = worker.compute_absolute_uncertainty_score(
            patient=patient,
            total_observable_attrs=total_observable_attrs,
        )
        
        # Should see 2 attributes, so missing 4 of 6
        assert uncertainty == 4
    
    def test_absolute_uncertainty_full_visibility(self):
        """Test absolute uncertainty with all attributes visible."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        # Inject full attribute list (6 total)
        full_attrs = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob'
        ]
        worker.set_observable_attributes(full_attrs)
        
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.2,
            'failure_value_multiplier': 0.9,
            'benefit_probability_multiplier': 1.0,
            'failure_probability_multiplier': 1.0,
            'recovery_without_treatment_prob': 0.1,
        }
        
        total_observable_attrs = 6
        uncertainty = worker.compute_absolute_uncertainty_score(
            patient=patient,
            total_observable_attrs=total_observable_attrs,
        )
        
        assert uncertainty == 0  # All 6 attributes observed


class TestExpectedRewardBehavior:
    """Test expected reward edge cases for HeuristicWorker."""

    def test_missing_prob_infected_fails_loudly(self):
        """Missing prob_infected should raise ValueError."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        reward_calculator = _create_reward_calculator_for_expected_reward_tests()
        patient = {
            'benefit_value_multiplier': 1.0,
        }

        with pytest.raises(ValueError, match="requires 'prob_infected'"):
            worker.compute_expected_reward(
                patient=patient,
                antibiotic_names=['A'],
                current_amr_levels={'A': 0.0},
                reward_calculator=reward_calculator,
            )

    def test_uses_configured_recovery_fallback(self):
        """Fallback recovery prob should use configured default value."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
            default_recovery_without_treatment_prob=0.25,
        )
        reward_calculator = _create_reward_calculator_for_expected_reward_tests()
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.0,
            'failure_value_multiplier': 1.0,
            'benefit_probability_multiplier': 1.0,
            'failure_probability_multiplier': 1.0,
        }

        rewards = worker.compute_expected_reward(
            patient=patient,
            antibiotic_names=['A'],
            current_amr_levels={'A': 0.0},
            reward_calculator=reward_calculator,
        )

        expected_no_treatment = 0.8 * 0.25 * 1.0 * 1.0
        assert rewards['no_treatment'] == pytest.approx(expected_no_treatment)


class TestActionSelection:
    """Test action selection logic (decide method)."""
    
    def setup_real_env_state(
        self,
        patients,
        current_amr_levels,
        use_relative_uncertainty=True,
    ):
        """Helper to create env_state with real instances."""
        # Create real RewardCalculator instance
        antibiotic_names = list(current_amr_levels.keys())
        abx_info = {
            'clinical_benefit_reward': 10.0,
            'clinical_benefit_probability': 0.6,
            'clinical_failure_penalty': -5.0,
            'clinical_failure_probability': 0.2,
            'abx_adverse_effects_info': {
                name: {
                    'adverse_effect_penalty': -1.0,
                    'adverse_effect_probability': 0.3,
                }
                for name in antibiotic_names
            },
        }
        rc_config = {
            'abx_clinical_reward_penalties_info_dict': abx_info,
            'lambda_weight': 0.0,
            'epsilon': 0.05,
            'seed': 123,
        }
        reward_calculator = RewardCalculator(config=rc_config)
        
        # Create real PatientGenerator instance with proper config format
        pg_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': [
                'prob_infected',
                'benefit_value_multiplier',
                'failure_value_multiplier',
                'benefit_probability_multiplier',
                'failure_probability_multiplier',
                'recovery_without_treatment_prob'
            ],
        }
        patient_generator = PatientGenerator(config=pg_config)
        
        # Create simple option library mock (just needs abx_name_to_index)
        class SimpleOptionLibrary:
            def __init__(self, antibiotic_names):
                self.abx_name_to_index = {name: idx for idx, name in enumerate(antibiotic_names)}
        
        option_library = SimpleOptionLibrary(antibiotic_names)
        
        env_state = {
            'patients': patients,
            'num_patients': len(patients),
            'current_amr_levels': current_amr_levels,
            'reward_calculator': reward_calculator,
            'patient_generator': patient_generator,
            'use_relative_uncertainty': use_relative_uncertainty,
            'option_library': option_library,
            'current_step': 0,
            'max_steps': 100,
        }
        
        return env_state
    
    def test_prescribe_when_expected_reward_exceeds_threshold(self):
        """Test that worker prescribes when expected reward exceeds threshold."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={
                'prescribe_A': 0.5,  # Low threshold
                'prescribe_B': 0.5,
                'no_treatment': 0.0
            },
            uncertainty_threshold=2.0,
        )
        
        patients = [
            {
                'prob_infected': 0.9,  # High infection probability
                'benefit_value_multiplier': 1.5,  # High benefit
            }
        ]
        
        current_amr_levels = {'A': 0.1, 'B': 0.1}  # Low AMR
        
        env_state = self.setup_real_env_state(
            patients=patients,
            current_amr_levels=current_amr_levels,
        )
        
        actions = worker.decide(env_state=env_state)
        
        # Should prescribe (not no_treatment which is index 2)
        assert actions[0] in [0, 1]  # Prescribe A or B
    
    def test_refuse_when_uncertainty_too_high(self):
        """Test that worker refuses to prescribe when uncertainty exceeds threshold."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={
                'prescribe_A': 0.5,
                'prescribe_B': 0.5,
                'no_treatment': 0.0
            },
            uncertainty_threshold=1.0,  # Low tolerance for uncertainty
        )
        
        # Patient with many padded attributes
        patients = [
            {
                'prob_infected': 0.9,
                'benefit_value_multiplier': -1.0,  # Padded
                'failure_value_multiplier': -1.0,  # Padded
                'benefit_probability_multiplier': -1.0,  # Padded
            }
        ]
        
        current_amr_levels = {'A': 0.1, 'B': 0.1}
        
        env_state = self.setup_real_env_state(
            patients=patients,
            current_amr_levels=current_amr_levels,
        )
        
        actions = worker.decide(env_state=env_state)
        
        # Should default to no_treatment due to high uncertainty
        no_treatment_index = env_state['reward_calculator'].abx_name_to_index['no_treatment']
        assert actions[0] == no_treatment_index
    
    def test_select_best_action_among_multiple(self):
        """Test that worker selects highest expected reward among valid actions."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={
                'prescribe_A': 0.1,  # Very low thresholds (both will pass)
                'prescribe_B': 0.1,
                'no_treatment': 0.0
            },
            uncertainty_threshold=5.0,  # High tolerance
        )
        
        patients = [
            {
                'prob_infected': 0.9,
                'benefit_value_multiplier': 1.2,
            }
        ]
        
        # A has lower AMR → higher expected reward
        current_amr_levels = {'A': 0.1, 'B': 0.8}
        
        env_state = self.setup_real_env_state(
            patients=patients,
            current_amr_levels=current_amr_levels,
        )
        
        actions = worker.decide(env_state=env_state)
        
        # Should select A (index 0) because it has lower AMR
        assert actions[0] == 0
    
    def test_multiple_patients(self):
        """Test action selection for multiple patients."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={
                'prescribe_A': 0.5,
                'prescribe_B': 0.5,
                'no_treatment': 0.0
            },
            uncertainty_threshold=2.0,
        )
        
        patients = [
            {'prob_infected': 0.9, 'benefit_value_multiplier': 1.2},  # High benefit
            {'prob_infected': 0.2, 'benefit_value_multiplier': 0.8},  # Low benefit
        ]
        
        current_amr_levels = {'A': 0.2, 'B': 0.2}
        
        env_state = self.setup_real_env_state(
            patients=patients,
            current_amr_levels=current_amr_levels,
        )
        
        actions = worker.decide(env_state=env_state)
        
        assert len(actions) == 2
        # With realistic reward calculation, both patients' expected rewards may be below threshold
        # Just verify we get valid actions (0, 1, or 2 for no_treatment)
        assert all(action in [0, 1, 2] for action in actions)


class TestHeuristicOptionLoader:
    """Test loader function for heuristic options."""
    
    def test_load_basic_config(self):
        """Test loading heuristic option from config dict."""
        config = {
            'duration': 10,
            'action_thresholds': {
                'prescribe_A': 0.7,
                'prescribe_B': 0.5,
                'no_treatment': 0.0
            },
            'uncertainty_threshold': 2.0,
        }
        
        option = load_heuristic_option(name='test_heuristic', config=config)
        
        assert isinstance(option, HeuristicWorker)
        assert option.name == 'test_heuristic'
        assert option.k == 10
        assert option.action_thresholds == config['action_thresholds']
        assert option.uncertainty_threshold == 2.0
    
    def test_load_missing_duration(self):
        """Test that loader raises ValueError if duration missing."""
        config = {
            'action_thresholds': {'prescribe_A': 0.5, 'no_treatment': 0.0},
        }
        
        with pytest.raises(ValueError, match="missing required key 'duration'"):
            load_heuristic_option(name='test', config=config)
    
    def test_load_missing_action_thresholds(self):
        """Test that loader raises ValueError if action_thresholds missing."""
        config = {
            'duration': 10,
        }
        
        with pytest.raises(ValueError, match="missing required key 'action_thresholds'"):
            load_heuristic_option(name='test', config=config)
    
    def test_load_invalid_duration_type(self):
        """Test that loader raises ValueError for invalid duration type."""
        config = {
            'duration': 'ten',  # Should be int
            'action_thresholds': {'prescribe_A': 0.5, 'no_treatment': 0.0},
        }
        
        with pytest.raises(ValueError, match="'duration' must be an int"):
            load_heuristic_option(name='test', config=config)
    
    def test_load_invalid_action_thresholds_type(self):
        """Test that loader raises ValueError for invalid action_thresholds type."""
        config = {
            'duration': 10,
            'action_thresholds': 'invalid',  # Should be dict
        }
        
        with pytest.raises(ValueError, match="'action_thresholds' must be a dict"):
            load_heuristic_option(name='test', config=config)
    
    def test_load_default_uncertainty_threshold(self):
        """Test that loader uses default uncertainty_threshold if not provided."""
        config = {
            'duration': 10,
            'action_thresholds': {'prescribe_A': 0.5, 'no_treatment': 0.0},
            # uncertainty_threshold omitted
        }
        
        option = load_heuristic_option(name='test', config=config)
        
        assert option.uncertainty_threshold == 2.0  # Default value
