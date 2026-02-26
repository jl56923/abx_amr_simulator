"""Integration tests for HeuristicWorker attribute estimation extension point."""

from typing import Dict

import pytest

from abx_amr_simulator.core.reward_calculator import RewardCalculator
from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker,
)


class EstimatingHeuristicWorker(HeuristicWorker):
    """HeuristicWorker subclass with custom attribute estimation."""

    def _estimate_unobserved_attribute_values_from_observed(
        self,
        patient: Dict[str, float],
    ) -> Dict[str, float]:
        patient = patient.copy()
        if (
            'recovery_without_treatment_prob' not in patient
            or patient['recovery_without_treatment_prob'] == -1.0
        ):
            patient['recovery_without_treatment_prob'] = 0.2
        return patient


def _create_reward_calculator() -> RewardCalculator:
    config = {
        'abx_clinical_reward_penalties_info_dict': {
            'clinical_benefit_reward': 10.0,
            'clinical_benefit_probability': 0.6,
            'clinical_failure_penalty': -10.0,
            'clinical_failure_probability': 0.0,
            'abx_adverse_effects_info': {
                'A': {
                    'adverse_effect_penalty': -1.0,
                    'adverse_effect_probability': 0.0,
                }
            },
        },
        'lambda_weight': 0.0,
        'epsilon': 0.05,
        'seed': 123,
    }
    return RewardCalculator(config=config)


def test_custom_estimation_changes_no_treatment_reward() -> None:
    """Custom estimator should change no-treatment expected reward."""
    reward_calculator = _create_reward_calculator()
    base_worker = HeuristicWorker(
        name='base_worker',
        duration=5,
        action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
        uncertainty_threshold=2.0,
        default_recovery_without_treatment_prob=0.1,
    )
    estimating_worker = EstimatingHeuristicWorker(
        name='estimating_worker',
        duration=5,
        action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
        uncertainty_threshold=2.0,
        default_recovery_without_treatment_prob=0.1,
    )

    patient = {
        'prob_infected': 0.8,
        'benefit_value_multiplier': 1.0,
        'failure_value_multiplier': 1.0,
        'benefit_probability_multiplier': 1.0,
        'failure_probability_multiplier': 1.0,
    }
    antibiotic_names = ['A']
    current_amr_levels = {'A': 0.0}

    base_rewards = base_worker.compute_expected_reward(
        patient=patient,
        antibiotic_names=antibiotic_names,
        current_amr_levels=current_amr_levels,
        reward_calculator=reward_calculator,
    )
    estimated_rewards = estimating_worker.compute_expected_reward(
        patient=patient,
        antibiotic_names=antibiotic_names,
        current_amr_levels=current_amr_levels,
        reward_calculator=reward_calculator,
    )

    rb = reward_calculator.abx_clinical_reward_penalties_info_dict[
        'normalized_clinical_benefit_reward'
    ]
    expected_base = 0.8 * 0.1 * rb * 1.0
    expected_estimated = 0.8 * 0.2 * rb * 1.0

    assert base_rewards['no_treatment'] == pytest.approx(expected_base)
    assert estimated_rewards['no_treatment'] == pytest.approx(expected_estimated)
    assert estimated_rewards['no_treatment'] > base_rewards['no_treatment']


def test_custom_estimation_does_not_override_observed_value() -> None:
    """Estimator should not override observed recovery probability."""
    reward_calculator = _create_reward_calculator()
    estimating_worker = EstimatingHeuristicWorker(
        name='estimating_worker',
        duration=5,
        action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
        uncertainty_threshold=2.0,
        default_recovery_without_treatment_prob=0.1,
    )

    patient = {
        'prob_infected': 0.8,
        'recovery_without_treatment_prob': 0.05,
        'benefit_value_multiplier': 1.0,
        'failure_value_multiplier': 1.0,
        'benefit_probability_multiplier': 1.0,
        'failure_probability_multiplier': 1.0,
    }
    antibiotic_names = ['A']
    current_amr_levels = {'A': 0.0}

    rewards = estimating_worker.compute_expected_reward(
        patient=patient,
        antibiotic_names=antibiotic_names,
        current_amr_levels=current_amr_levels,
        reward_calculator=reward_calculator,
    )

    rb = reward_calculator.abx_clinical_reward_penalties_info_dict[
        'normalized_clinical_benefit_reward'
    ]
    expected_reward = 0.8 * 0.05 * rb * 1.0

    assert rewards['no_treatment'] == pytest.approx(expected_reward)
