"""Tests for heuristic policy worker option.

Tests the HeuristicWorker class and load_heuristic_option loader function
from the abx_amr_simulator.options.defaults.option_types.heuristic package.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker,
    load_heuristic_option,
)


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
        assert len(HeuristicWorker.REQUIRES_OBSERVATION_ATTRIBUTES) == 6


class TestExpectedRewardComputation:
    """Test expected reward calculation logic."""
    
    def test_expected_reward_single_antibiotic(self):
        """Test expected reward computation for a single antibiotic."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.2,
        }
        
        current_amr_levels = {'A': 0.2}
        
        clinical_params = {
            'A': {
                'clinical_benefit_reward': 10.0,
                'adverse_effect_penalty': 1.0,
            }
        }
        
        expected_rewards = worker.compute_expected_reward(
            patient=patient,
            current_amr_levels=current_amr_levels,
            clinical_params=clinical_params,
        )
        
        # Expected: 0.8 * 10.0 * 1.2 * (1 - 0.2) - 1.0 = 9.6 * 0.8 - 1.0 = 7.68 - 1.0 = 6.68
        assert 'prescribe_A' in expected_rewards
        expected_value = 0.8 * 10.0 * 1.2 * (1 - 0.2) - 1.0
        assert abs(expected_rewards['prescribe_A'] - expected_value) < 0.01
    
    def test_expected_reward_high_amr(self):
        """Test that high AMR reduces expected reward."""
        worker = HeuristicWorker(
            name='test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.0,
        }
        
        clinical_params = {
            'A': {
                'clinical_benefit_reward': 10.0,
                'adverse_effect_penalty': 1.0,
            }
        }
        
        # Low AMR case
        low_amr = {'A': 0.1}
        rewards_low = worker.compute_expected_reward(
            patient=patient,
            current_amr_levels=low_amr,
            clinical_params=clinical_params,
        )
        
        # High AMR case
        high_amr = {'A': 0.9}
        rewards_high = worker.compute_expected_reward(
            patient=patient,
            current_amr_levels=high_amr,
            clinical_params=clinical_params,
        )
        
        # High AMR should give lower expected reward
        assert rewards_low['prescribe_A'] > rewards_high['prescribe_A']
    
    def test_expected_reward_multiple_antibiotics(self):
        """Test expected reward with multiple antibiotics."""
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
        
        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.0,
        }
        
        current_amr_levels = {'A': 0.2, 'B': 0.5}
        
        clinical_params = {
            'A': {
                'clinical_benefit_reward': 10.0,
                'adverse_effect_penalty': 1.0,
            },
            'B': {
                'clinical_benefit_reward': 8.0,
                'adverse_effect_penalty': 0.5,
            }
        }
        
        expected_rewards = worker.compute_expected_reward(
            patient=patient,
            current_amr_levels=current_amr_levels,
            clinical_params=clinical_params,
        )
        
        assert 'prescribe_A' in expected_rewards
        assert 'prescribe_B' in expected_rewards
        assert 'no_treatment' in expected_rewards
        
        # A should have higher expected reward (lower AMR + higher benefit)
        assert expected_rewards['prescribe_A'] > expected_rewards['prescribe_B']


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


class TestActionSelection:
    """Test action selection logic (decide method)."""
    
    def setup_mock_env_state(
        self,
        patients,
        current_amr_levels,
        use_relative_uncertainty=True,
    ):
        """Helper to create mock env_state."""
        # Mock reward calculator
        mock_rc = Mock()
        mock_rc.abx_clinical_reward_penalties_info_dict = {
            'A': {
                'clinical_benefit_reward': 10.0,
                'adverse_effect_penalty': 1.0,
            },
            'B': {
                'clinical_benefit_reward': 8.0,
                'adverse_effect_penalty': 0.5,
            }
        }
        
        # Mock patient generator
        mock_pg = Mock()
        mock_pg.KNOWN_ATTRIBUTE_TYPES = {
            'prob_infected': None,
            'benefit_value_multiplier': None,
            'failure_value_multiplier': None,
            'benefit_probability_multiplier': None,
            'failure_probability_multiplier': None,
            'recovery_without_treatment_prob': None,
        }
        
        # Mock option library
        mock_ol = Mock()
        mock_ol.abx_name_to_index = {'A': 0, 'B': 1}  # no_treatment is implicit at index 2
        
        env_state = {
            'patients': patients,
            'num_patients': len(patients),
            'current_amr_levels': current_amr_levels,
            'reward_calculator': mock_rc,
            'patient_generator': mock_pg,
            'use_relative_uncertainty': use_relative_uncertainty,
            'option_library': mock_ol,
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
        
        env_state = self.setup_mock_env_state(
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
        
        env_state = self.setup_mock_env_state(
            patients=patients,
            current_amr_levels=current_amr_levels,
        )
        
        actions = worker.decide(env_state=env_state)
        
        # Should default to no_treatment (index 2) due to high uncertainty
        assert actions[0] == 2
    
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
        
        env_state = self.setup_mock_env_state(
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
        
        env_state = self.setup_mock_env_state(
            patients=patients,
            current_amr_levels=current_amr_levels,
        )
        
        actions = worker.decide(env_state=env_state)
        
        assert len(actions) == 2
        assert actions[0] in [0, 1]  # First patient likely prescribed
        # Second patient might be no_treatment due to low expected reward


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
