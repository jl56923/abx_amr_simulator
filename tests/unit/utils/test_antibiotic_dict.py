"""Tests for antibiotic dict configuration and step behavior in ABXAMREnv."""
import pathlib
import sys
from typing import Optional

import numpy as np

from tests.unit.utils.test_reference_helpers import create_mock_environment
from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.core import RewardCalculator
from abx_amr_simulator.core import PatientGenerator


def create_test_reward_calculator(
    antibiotic_names=None,
    clinical_benefit_reward=10.0,
    clinical_benefit_probability=1.0,
    adverse_effect_penalty=-2.0,
    adverse_effect_probability=0.0,
    lambda_weight=0.5,
    epsilon=0.05,
    clinical_failure_penalty=-1.0,
    clinical_failure_probability=0.0,
):
    """Helper to create a RewardCalculator for testing (updated API)."""
    if antibiotic_names is None:
        antibiotic_names = ["Antibiotic_0", "Antibiotic_1"]

    abx_clinical_reward_penalties_info_dict = {
        'clinical_benefit_reward': clinical_benefit_reward,
        'clinical_benefit_probability': clinical_benefit_probability,
        'clinical_failure_penalty': clinical_failure_penalty,
        'clinical_failure_probability': clinical_failure_probability,
        'abx_adverse_effects_info': {
            name: {
                'adverse_effect_penalty': adverse_effect_penalty,
                'adverse_effect_probability': adverse_effect_probability,
            }
            for name in antibiotic_names
        },
    }

    config = {
        'abx_clinical_reward_penalties_info_dict': abx_clinical_reward_penalties_info_dict,
        'lambda_weight': lambda_weight,
        'epsilon': epsilon,
    }
    return RewardCalculator(config=config)


def create_test_antibiotics_dict(antibiotic_names=None):
    """Helper to create a default antibiotics_AMR_dict for testing."""
    if antibiotic_names is None:
        antibiotic_names = ["Antibiotic_0", "Antibiotic_1"]
    
    return {
        name: {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0
        }
        for name in antibiotic_names
    }


def test_default_antibiotics_configuration():
    """Test default antibiotics configuration."""
    antibiotic_names = ["Antibiotic_0", "Antibiotic_1"]
    # Minimal PatientGenerator (uniform infection probability)
    pg_config = {
        'visible_patient_attributes': ['prob_infected'],
        'prob_infected': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0]
        },
        'benefit_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'failure_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'benefit_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'failure_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'recovery_without_treatment_prob': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0]
        }
    }
    patient_generator = PatientGenerator(config=pg_config)
    env = create_mock_environment(
        reward_calculator=create_test_reward_calculator(antibiotic_names=antibiotic_names),
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=5,
        max_time_steps=10,
        include_steps_since_amr_update_in_obs=False
    )
    obs, _ = env.reset(seed=42)

    assert env.antibiotic_names == antibiotic_names
    assert obs.shape == (env.num_patients_per_time_step + env.num_abx,)
    assert set(env.visible_amr_levels.keys()) == set(env.antibiotic_names)


def test_custom_antibiotics_defaults():
    """Test custom antibiotic names with default parameters."""
    antibiotics_2 = {
        "Penicillin": {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0
        },
        "Cephalosporin": {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0
        },
        "Fluoroquinolone": {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0
        },
    }
    pg_config = {
        'visible_patient_attributes': ['prob_infected'],
        'prob_infected': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0]
        },
        'benefit_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'failure_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'benefit_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'failure_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'recovery_without_treatment_prob': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0]
        }
    }
    patient_generator = PatientGenerator(config=pg_config)
    env = create_mock_environment(
        reward_calculator=create_test_reward_calculator(antibiotic_names=list(antibiotics_2.keys())),
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=antibiotics_2,
        num_patients_per_time_step=5,
        max_time_steps=10,
        include_steps_since_amr_update_in_obs=False
    )
    obs, _ = env.reset(seed=42)

    assert env.antibiotic_names == ["Penicillin", "Cephalosporin", "Fluoroquinolone"]
    assert env.no_treatment_action == env.num_abx == 3
    assert obs.shape == (env.num_patients_per_time_step + env.num_abx,)
    assert set(env.visible_amr_levels.keys()) == set(antibiotics_2.keys())


def test_custom_antibiotics_step_counts_and_reward():
    """Test that prescriptions are counted correctly per antibiotic."""
    antibiotics_3 = {
        "Amoxicillin": {
            "leak": 0.1,
            "flatness_parameter": 1.0,
            "permanent_residual_volume": 0.0,
            "initial_amr_level": 0.0,
        },
        "Azithromycin": {
            "leak": 0.08,
            "flatness_parameter": 1.0,
            "permanent_residual_volume": 0.0,
            "initial_amr_level": 0.1,
        },
    }
    pg_config = {
        'visible_patient_attributes': ['prob_infected'],
        'prob_infected': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0]
        },
        'benefit_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'failure_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'benefit_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'failure_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None]
        },
        'recovery_without_treatment_prob': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0]
        }
    }
    patient_generator = PatientGenerator(config=pg_config)
    env = create_mock_environment(
        reward_calculator=create_test_reward_calculator(antibiotic_names=list(antibiotics_3.keys())),
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=antibiotics_3,
        num_patients_per_time_step=10,
        max_time_steps=20
    )
    env.reset(seed=42)

    action = np.array([0, 0, 1, 0, 0, 1, env.no_treatment_action, 1, env.no_treatment_action, 1])
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    # Count: Amoxicillin (0) appears 4 times, Azithromycin (1) appears 4 times
    assert info["prescriptions_per_abx"] == {"Amoxicillin": 4, "Azithromycin": 4}
    assert set(info["actual_amr_levels"].keys()) == set(antibiotics_3.keys())
    assert set(info["visible_amr_levels"].keys()) == set(antibiotics_3.keys())
    assert isinstance(reward, float)

    # AMR levels should increase after prescribing compared to initial visible levels
    assert info["actual_amr_levels"]["Amoxicillin"] >= env.visible_amr_levels["Amoxicillin"]
    assert info["actual_amr_levels"]["Azithromycin"] >= env.visible_amr_levels["Azithromycin"]
