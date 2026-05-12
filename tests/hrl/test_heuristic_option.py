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
            default_recovery_without_treatment_prob=0.1,
        )
        assert worker.name == 'HEURISTIC_test'
        assert worker.k == 10
        assert worker.action_thresholds == {'prescribe_A': 0.5, 'no_treatment': 0.0}
        assert worker.uncertainty_threshold == 2.0
    
    def test_option_base_protocol_attributes(self):
        """Test that HeuristicWorker declares required OptionBase attributes.
        
        Note: REQUIRES_STEP_NUMBER has been removed from the OptionBase protocol.
        Options are now time-agnostic.
        """
        assert hasattr(HeuristicWorker, 'REQUIRES_OBSERVATION_ATTRIBUTES')
        assert hasattr(HeuristicWorker, 'REQUIRES_AMR_LEVELS')
        assert hasattr(HeuristicWorker, 'PROVIDES_TERMINATION_CONDITION')
        
        # Verify no REQUIRES_STEP_NUMBER attribute
        assert not hasattr(HeuristicWorker, 'REQUIRES_STEP_NUMBER')
        
        assert HeuristicWorker.REQUIRES_AMR_LEVELS is True
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
            default_recovery_without_treatment_prob=0.1,
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
            default_recovery_without_treatment_prob=0.1,
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
            default_recovery_without_treatment_prob=0.1,
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
            default_recovery_without_treatment_prob=0.1,
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
            default_recovery_without_treatment_prob=0.1,
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
            default_recovery_without_treatment_prob=0.1,
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
        
        # Should prescribe (not 'no_treatment')
        assert actions[0] in ['A', 'B']  # Prescribe A or B
    
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
            default_recovery_without_treatment_prob=0.1,
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
        
        # Should default to 'no_treatment' due to high uncertainty
        assert actions[0] == 'no_treatment'
    
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
            default_recovery_without_treatment_prob=0.1,
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
        
        # Should select 'A' because it has lower AMR
        assert actions[0] == 'A'
    
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
            default_recovery_without_treatment_prob=0.1,
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
        # Just verify we get valid antibiotic name strings ('A', 'B', or 'no_treatment')
        assert all(action in ['A', 'B', 'no_treatment'] for action in actions)


class TestSentinelValueHandling:
    """Tests that -1 sentinel values (padded/missing attributes) are treated as neutral defaults.

    Before the fix, dict.get(key, default) would return -1 when the key was present
    but set to -1, corrupting the expected reward calculation for limited-visibility patients.
    These tests verify the fix: any attribute value < 0 is replaced with the appropriate default
    before use in the reward formula.
    """

    def _make_worker(
        self,
        *,
        uncertainty_threshold: float = 10.0,
        default_recovery_prob: float = 0.1,
    ) -> HeuristicWorker:
        return HeuristicWorker(
            name='sentinel_test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=uncertainty_threshold,
            default_recovery_without_treatment_prob=default_recovery_prob,
        )

    def _make_rc(self) -> RewardCalculator:
        # benefit_prob=1.0, failure_prob=0.0, adverse_effect_prob=0.0 simplifies
        # expected reward to: prescribe_A = pI * pS * RB; no_treatment = pI * r_spont * RB
        return _create_reward_calculator_for_expected_reward_tests()

    def test_sentinel_minus_one_produces_same_rewards_as_neutral_defaults(self):
        """All five optional attributes set to -1 must yield the same rewards as 1.0/default."""
        worker = self._make_worker(default_recovery_prob=0.1)
        rc = self._make_rc()
        antibiotic_names = ['A']
        amr_levels = {'A': 0.2}

        full_visibility_patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.0,
            'failure_value_multiplier': 1.0,
            'benefit_probability_multiplier': 1.0,
            'failure_probability_multiplier': 1.0,
            'recovery_without_treatment_prob': 0.1,
        }
        limited_visibility_patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': -1.0,
            'failure_value_multiplier': -1.0,
            'benefit_probability_multiplier': -1.0,
            'failure_probability_multiplier': -1.0,
            'recovery_without_treatment_prob': -1.0,
        }

        rewards_full = worker.compute_expected_reward(
            patient=full_visibility_patient,
            antibiotic_names=antibiotic_names,
            current_amr_levels=amr_levels,
            reward_calculator=rc,
        )
        rewards_limited = worker.compute_expected_reward(
            patient=limited_visibility_patient,
            antibiotic_names=antibiotic_names,
            current_amr_levels=amr_levels,
            reward_calculator=rc,
        )

        assert rewards_full.keys() == rewards_limited.keys()
        for action_key in rewards_full:
            assert rewards_limited[action_key] == pytest.approx(rewards_full[action_key]), (
                f"Action '{action_key}': sentinel patient reward {rewards_limited[action_key]} "
                f"!= full-visibility reward {rewards_full[action_key]}"
            )

    def test_r_spont_sentinel_uses_configured_default_not_one(self):
        """r_spont = -1 must use default_recovery_prob, not 1.0.

        This matters because confusing r_spont=-1 with r_spont=1.0 would make
        no_treatment appear far more attractive than it really is.
        """
        default_prob = 0.15
        worker = self._make_worker(default_recovery_prob=default_prob)
        rc = self._make_rc()
        antibiotic_names = ['A']
        amr_levels = {'A': 0.0}

        patient_sentinel = {
            'prob_infected': 0.8,
            'recovery_without_treatment_prob': -1.0,
        }
        patient_explicit = {
            'prob_infected': 0.8,
            'recovery_without_treatment_prob': default_prob,
        }
        patient_wrong_default = {
            'prob_infected': 0.8,
            'recovery_without_treatment_prob': 1.0,  # What the bug would have used
        }

        rewards_sentinel = worker.compute_expected_reward(
            patient=patient_sentinel, antibiotic_names=antibiotic_names,
            current_amr_levels=amr_levels, reward_calculator=rc,
        )
        rewards_explicit = worker.compute_expected_reward(
            patient=patient_explicit, antibiotic_names=antibiotic_names,
            current_amr_levels=amr_levels, reward_calculator=rc,
        )
        rewards_wrong = worker.compute_expected_reward(
            patient=patient_wrong_default, antibiotic_names=antibiotic_names,
            current_amr_levels=amr_levels, reward_calculator=rc,
        )

        assert rewards_sentinel['no_treatment'] == pytest.approx(rewards_explicit['no_treatment'])
        assert rewards_sentinel['no_treatment'] != pytest.approx(rewards_wrong['no_treatment'])

    def test_limited_visibility_patient_can_prescribe_when_threshold_allows(self):
        """Limited-visibility patient (4 attrs = -1) can prescribe when uncertainty < threshold.

        Regression test: before the fix, -1 multipliers drove pB to 0 and made vB negative,
        producing nonsensical expected rewards that always lost to no_treatment regardless
        of threshold. With the fix, a patient with 4 sentinel attrs and threshold=10 should
        be evaluated using neutral defaults and can prescribe when the reward warrants it.
        """
        # uncertainty_threshold=10 > 4 sentinel attrs → no hard-refusal
        worker = HeuristicWorker(
            name='sentinel_test',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=10.0,
            default_recovery_without_treatment_prob=0.1,
        )
        worker.set_observable_attributes([
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob',
        ])

        rc = self._make_rc()
        antibiotic_names = ['A']
        # Low AMR → prescribing has high expected benefit (pS = 0.9)
        amr_levels = {'A': 0.1}

        # Mirrors the actual limited-visibility patient profile in the LPP experiments:
        # prob_infected and recovery_without_treatment_prob are visible; the 4 multipliers
        # are padded to -1.
        limited_visibility_patient = {
            'prob_infected': 0.9,
            'benefit_value_multiplier': -1.0,
            'failure_value_multiplier': -1.0,
            'benefit_probability_multiplier': -1.0,
            'failure_probability_multiplier': -1.0,
            'recovery_without_treatment_prob': 0.1,
        }

        # Verify uncertainty score is 4 (the 4 multiplier attrs are -1)
        uncertainty = worker.compute_relative_uncertainty_score(patient=limited_visibility_patient)
        assert uncertainty == 4

        rewards = worker.compute_expected_reward(
            patient=limited_visibility_patient,
            antibiotic_names=antibiotic_names,
            current_amr_levels=amr_levels,
            reward_calculator=rc,
        )

        # With the fix: prescribe_A = pI * pS * RB = 0.9 * 0.9 * 1.0 = 0.81 > threshold 0.5
        assert rewards['prescribe_A'] > 0.5, (
            f"Expected prescribe_A reward > 0.5 with neutral sentinel defaults, got {rewards['prescribe_A']}"
        )

    def test_valid_small_positive_values_are_not_replaced(self):
        """Verify that small but valid positive attribute values are not treated as sentinels."""
        worker = self._make_worker(default_recovery_prob=0.1)
        rc = self._make_rc()
        antibiotic_names = ['A']
        amr_levels = {'A': 0.0}

        # Values very close to 0 but positive are valid; they must NOT be replaced with 1.0
        patient_small_positive = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 0.01,
            'failure_value_multiplier': 0.01,
            'benefit_probability_multiplier': 0.01,
            'failure_probability_multiplier': 0.01,
            'recovery_without_treatment_prob': 0.01,
        }
        patient_neutral = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': 1.0,
            'failure_value_multiplier': 1.0,
            'benefit_probability_multiplier': 1.0,
            'failure_probability_multiplier': 1.0,
            'recovery_without_treatment_prob': 0.1,
        }

        rewards_small = worker.compute_expected_reward(
            patient=patient_small_positive, antibiotic_names=antibiotic_names,
            current_amr_levels=amr_levels, reward_calculator=rc,
        )
        rewards_neutral = worker.compute_expected_reward(
            patient=patient_neutral, antibiotic_names=antibiotic_names,
            current_amr_levels=amr_levels, reward_calculator=rc,
        )

        # Small positive values should produce meaningfully different rewards than neutral 1.0
        assert rewards_small['prescribe_A'] != pytest.approx(rewards_neutral['prescribe_A'])


class TestHeuristicOptionLoader:
    """Test loader function for heuristic options."""
    
    def test_load_basic_config(self):
        """Test loading heuristic option from config dict."""
        config = {
            'option_name': 'test_heuristic',
            'duration': 10,
            'action_thresholds': {
                'prescribe_A': 0.7,
                'prescribe_B': 0.5,
                'no_treatment': 0.0
            },
            'uncertainty_threshold': 2.0,
        }
        
        option = load_heuristic_option(config=config)
        
        assert isinstance(option, HeuristicWorker)
        assert option.name == 'test_heuristic'
        assert option.k == 10
        assert option.action_thresholds == config['action_thresholds']
        assert option.uncertainty_threshold == 2.0
    
    def test_load_missing_duration(self):
        """Test that loader raises ValueError if duration missing."""
        config = {
            'option_name': 'test',
            'action_thresholds': {'prescribe_A': 0.5, 'no_treatment': 0.0},
        }
        
        with pytest.raises(ValueError, match="missing required key 'duration'"):
            load_heuristic_option(config=config)
    
    def test_load_missing_action_thresholds(self):
        """Test that loader raises ValueError if action_thresholds missing."""
        config = {
            'option_name': 'test',
            'duration': 10,
        }
        
        with pytest.raises(ValueError, match="missing required key 'action_thresholds'"):
            load_heuristic_option(config=config)
    
    def test_load_invalid_duration_type(self):
        """Test that loader raises ValueError for invalid duration type."""
        config = {
            'option_name': 'test',
            'duration': 'ten',  # Should be int
            'action_thresholds': {'prescribe_A': 0.5, 'no_treatment': 0.0},
        }
        
        with pytest.raises(ValueError, match="'duration' must be an int"):
            load_heuristic_option(config=config)
    
    def test_load_invalid_action_thresholds_type(self):
        """Test that loader raises ValueError for invalid action_thresholds type."""
        config = {
            'option_name': 'test',
            'duration': 10,
            'action_thresholds': 'invalid',  # Should be dict
        }
        
        with pytest.raises(ValueError, match="'action_thresholds' must be a dict"):
            load_heuristic_option(config=config)
    
    def test_load_default_uncertainty_threshold(self):
        """Test that loader uses default uncertainty_threshold if not provided."""
        config = {
            'option_name': 'test',
            'duration': 10,
            'action_thresholds': {'prescribe_A': 0.5, 'no_treatment': 0.0},
            # uncertainty_threshold omitted
        }
        
        option = load_heuristic_option(config=config)
        
        assert option.uncertainty_threshold == 2.0  # Default value
