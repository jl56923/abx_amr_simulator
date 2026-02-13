"""Tests for observable attribute injection pattern in HeuristicWorker."""

import pytest

from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker,
)
from abx_amr_simulator.core.reward_calculator import RewardCalculator
from abx_amr_simulator.core.patient_generator import PatientGenerator


class TestObservableAttributeInjection:
    """Test that HeuristicWorker correctly receives and uses injected observable attributes."""
    
    def test_initial_state(self):
        """Test that _observable_patient_attributes starts empty."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        assert worker._observable_patient_attributes == []
    
    def test_set_observable_attributes(self):
        """Test that set_observable_attributes properly stores the attribute list."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        attrs = ['prob_infected', 'benefit_value_multiplier', 'failure_value_multiplier']
        worker.set_observable_attributes(attrs)
        
        assert worker._observable_patient_attributes == attrs
    
    def test_relative_uncertainty_uses_minimal_when_not_injected(self):
        """Test that relative uncertainty falls back to minimal requirements if not injected."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        # No injection - should use REQUIRES_OBSERVATION_ATTRIBUTES (just prob_infected)
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': -1.0,  # Padded (won't be checked)
        }
        
        uncertainty = worker.compute_relative_uncertainty_score(patient=patient)
        # Only checks prob_infected (not -1), so uncertainty = 0
        assert uncertainty == 0
    
    def test_relative_uncertainty_uses_injected_attributes(self):
        """Test that relative uncertainty uses injected attributes when set."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        # Inject full attribute list
        attrs = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob'
        ]
        worker.set_observable_attributes(attrs)
        
        # Patient with some padded attributes
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': -1.0,  # Padded
            'failure_value_multiplier': -1.0,  # Padded
            'benefit_probability_multiplier': 1.0,
            'failure_probability_multiplier': 1.0,
            'recovery_without_treatment_prob': 0.1,
        }
        
        uncertainty = worker.compute_relative_uncertainty_score(patient=patient)
        # Should find 2 padded attributes
        assert uncertainty == 2
    
    def test_absolute_uncertainty_uses_minimal_when_not_injected(self):
        """Test that absolute uncertainty falls back to minimal requirements if not injected."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        # No injection - should use REQUIRES_OBSERVATION_ATTRIBUTES
        patient = {
            'prob_infected': 0.8,
        }
        
        total_observable = 6
        uncertainty = worker.compute_absolute_uncertainty_score(
            patient=patient,
            total_observable_attrs=total_observable,
        )
        # Sees 1 attribute (prob_infected), missing 5 of 6 (but only checks minimal requirement)
        # Actually checks if prob_infected is observed → 1 observed, so 6 - 1 = 5
        assert uncertainty == 5
    
    def test_absolute_uncertainty_uses_injected_attributes(self):
        """Test that absolute uncertainty uses injected attributes when set."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        # Inject full attribute list
        attrs = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob'
        ]
        worker.set_observable_attributes(attrs)
        
        # Patient with partial visibility
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.2,
        }
        
        total_observable = 6
        uncertainty = worker.compute_absolute_uncertainty_score(
            patient=patient,
            total_observable_attrs=total_observable,
        )
        # Observes 2 attributes, missing 4 of 6
        assert uncertainty == 4
    
    def test_injection_enables_proper_uncertainty_scoring(self):
        """Integration test: verify injection enables correct uncertainty-based decisions."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={
                'prescribe_A': 0.1,  # Low threshold
                'no_treatment': 0.0
            },
            uncertainty_threshold=1.0,  # Strict: refuse if >1 attributes padded
        )
        
        # Inject full attribute list
        attrs = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
        ]
        worker.set_observable_attributes(attrs)
        
        # Patient with high uncertainty (2 padded)
        patient_uncertain = {
            'prob_infected': 0.9,
            'benefit_value_multiplier': -1.0,  # Padded
            'failure_value_multiplier': -1.0,  # Padded
        }
        
        # Patient with low uncertainty (0 padded)
        patient_certain = {
            'prob_infected': 0.9,
            'benefit_value_multiplier': 1.2,
            'failure_value_multiplier': 0.9,
        }
        
        # Create real env_state with real RewardCalculator and PatientGenerator
        abx_info = {
            'clinical_benefit_reward': 10.0,
            'clinical_benefit_probability': 0.6,
            'clinical_failure_penalty': -5.0,
            'clinical_failure_probability': 0.2,
            'abx_adverse_effects_info': {
                'A': {
                    'adverse_effect_penalty': -1.0,
                    'adverse_effect_probability': 0.3,
                }
            },
        }
        rc_config = {
            'abx_clinical_reward_penalties_info_dict': abx_info,
            'lambda_weight': 0.0,
            'epsilon': 0.05,
            'seed': 123,
        }
        reward_calculator = RewardCalculator(config=rc_config)
        
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
            'visible_patient_attributes': attrs,
        }
        patient_generator = PatientGenerator(config=pg_config)
        
        class SimpleOptionLibrary:
            def __init__(self):
                self.abx_name_to_index = {'A': 0}
        
        option_library = SimpleOptionLibrary()
        
        env_state = {
            'patients': [patient_uncertain],
            'num_patients': 1,
            'current_amr_levels': {'A': 0.1},
            'reward_calculator': reward_calculator,
            'patient_generator': patient_generator,
            'use_relative_uncertainty': True,
            'option_library': option_library,
            'current_step': 0,
            'max_steps': 100,
        }
        
        # Uncertain patient: should refuse (return no_treatment action index)
        actions_uncertain = worker.decide(env_state=env_state)
        no_treatment_index = env_state['reward_calculator'].abx_name_to_index['no_treatment']
        assert actions_uncertain[0] == no_treatment_index
        
        # Certain patient: should prescribe (action 0 = prescribe A)
        env_state['patients'] = [patient_certain]
        actions_certain = worker.decide(env_state=env_state)
        assert actions_certain[0] == 0  # prescribe A
