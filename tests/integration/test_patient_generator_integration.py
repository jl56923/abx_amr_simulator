"""Integration tests for PatientGenerator with ABXAMREnv."""
import pathlib
import sys
from typing import Optional

import numpy as np
import pytest
from unit.utils.test_reference_helpers import create_mock_environment
from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.core import RewardCalculator
from abx_amr_simulator.core import PatientGenerator
from abx_amr_simulator.core import Patient


def create_test_reward_calculator(antibiotic_names=None):
    """Helper to create a RewardCalculator for testing."""
    if antibiotic_names is None:
        antibiotic_names = ["A"]
    
    config = {
        'abx_clinical_reward_penalties_info_dict': {
            'clinical_benefit_reward': 10.0,
            'clinical_benefit_probability': 0.9,
            'clinical_failure_penalty': -5.0,
            'clinical_failure_probability': 0.5,
            'abx_adverse_effects_info': {
                name: {
                    'adverse_effect_penalty': -2.0,
                    'adverse_effect_probability': 0.1,
                }
                for name in antibiotic_names
            },
        },
        'lambda_weight': 0.5,
        'epsilon': 0.05,
        'seed': None,
    }
    
    return RewardCalculator(config=config)


def create_test_patient_generator(antibiotic_names=None, high_variance=False):
    """Helper to create a PatientGenerator for testing."""
    sigma = 0.3 if high_variance else 0.05
    config = {
        'prob_infected': {
            'prob_dist': {'type': 'gaussian', 'mu': 0.5, 'sigma': 0.1},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'benefit_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': sigma},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': sigma},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'benefit_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': sigma},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': sigma},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'recovery_without_treatment_prob': {
            'prob_dist': {'type': 'constant', 'value': 0.01},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'visible_patient_attributes': ['prob_infected'],
    }
    return PatientGenerator(config=config)


def create_test_antibiotics_dict(antibiotic_names=None):
    """Helper to create antibiotics_AMR_dict."""
    if antibiotic_names is None:
        antibiotic_names = ["A"]
    
    return {
        name: {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0
        }
        for name in antibiotic_names
    }


def test_new_patients_sampled_every_step():
    """Test that new patients are sampled on every step."""
    antibiotic_names = ["A"]
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    patient_generator = create_test_patient_generator(high_variance=True)
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=3,
        max_time_steps=10,
        include_steps_since_amr_update_in_obs=False
    )
    
    obs, _ = env.reset(seed=42)
    
    # Store first cohort
    first_cohort_probs = [p.prob_infected for p in env.current_patients]
    
    # Take a step
    action = np.array([0, 0, 0], dtype=int)  # Prescribe A to all patients
    obs, _, _, _, _ = env.step(action)
    
    # Store second cohort
    second_cohort_probs = [p.prob_infected for p in env.current_patients]
    
    # Cohorts should be different (different patient instances)
    assert first_cohort_probs != second_cohort_probs, "Patients should be resampled each step"
    
    # Take another step
    obs, _, _, _, _ = env.step(action)
    third_cohort_probs = [p.prob_infected for p in env.current_patients]
    
    # Third cohort should differ from second
    assert second_cohort_probs != third_cohort_probs, "Patients should be resampled each step"


def test_observation_shape_with_single_attribute():
    """Test observation shape with only prob_infected visible."""
    antibiotic_names = ["A", "B"]
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    patient_generator = create_test_patient_generator()
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=5,
        include_steps_since_amr_update_in_obs=False
    )
    
    obs, _ = env.reset(seed=42)
    
    # Shape should be: (num_patients * 1 attribute + num_abx,)
    expected_shape = (5 * 1 + 2,)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    
    # First 5 values should be infection probabilities (patient features)
    # Last 2 values should be AMR levels
    patient_features = obs[:5]
    amr_features = obs[5:]
    
    assert len(patient_features) == 5
    assert len(amr_features) == 2
    assert np.all((patient_features >= 0.0) & (patient_features <= 1.0))
    assert np.all((amr_features >= 0.0) & (amr_features <= 1.0))


def test_observation_shape_with_multiple_attributes():
    """Test observation shape with multiple patient attributes visible."""
    antibiotic_names = ["A"]
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    patient_generator = create_test_patient_generator()
    
    visible_attrs = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier'
    ]
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=visible_attrs,
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=3,
        include_steps_since_amr_update_in_obs=False
    )
    
    obs, _ = env.reset(seed=42)
    
    # Shape should be: (num_patients * 3 attributes + num_abx,)
    expected_shape = (3 * 3 + 1,)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    
    # First 9 values are patient features (3 patients × 3 attributes)
    # Last 1 value is AMR level
    patient_features = obs[:9]
    amr_features = obs[9:]
    
    assert len(patient_features) == 9
    assert len(amr_features) == 1


def test_observation_shape_all_attributes():
    """Test observation shape with all 6 patient attributes visible."""
    antibiotic_names = ["A"]
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    patient_generator = create_test_patient_generator()
    
    all_attrs = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob'
    ]
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=all_attrs,
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=2,
        include_steps_since_amr_update_in_obs=False
    )
    
    obs, _ = env.reset(seed=42)
    
    # Shape should be: (num_patients * 6 attributes + num_abx,)
    expected_shape = (2 * 6 + 1,)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"


def test_amr_accumulation_across_steps():
    """Test that AMR levels accumulate across steps with prescriptions."""
    antibiotic_names = ["A"]
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    patient_generator = create_test_patient_generator()
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=5,
        include_steps_since_amr_update_in_obs=False
    )
    
    obs, _ = env.reset(seed=42)
    initial_amr = env.amr_balloon_models["A"].get_volume()
    
    # Prescribe A to all 5 patients
    action = np.zeros(5, dtype=int)
    obs, _, _, _, info = env.step(action)
    amr_after_step1 = info["actual_amr_levels"]["A"]
    
    # AMR should increase
    assert amr_after_step1 > initial_amr, "AMR should increase after prescriptions"
    
    # Take another step with prescriptions
    obs, _, _, _, info = env.step(action)
    amr_after_step2 = info["actual_amr_levels"]["A"]
    
    # AMR should continue increasing
    assert amr_after_step2 > amr_after_step1, "AMR should continue increasing"
    
    # Take a step with no prescriptions (use no_treatment_action)
    no_treatment = np.full(5, env.no_treatment_action, dtype=int)
    obs, _, _, _, info = env.step(no_treatment)
    amr_after_rest = info["actual_amr_levels"]["A"]
    
    # AMR should decrease (leak) when no treatment
    assert amr_after_rest < amr_after_step2, "AMR should leak when no treatment given"


def test_amr_reset_on_env_reset():
    """Test that AMR balloons reset when env.reset() is called."""
    antibiotic_names = ["A", "B"]
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    patient_generator = create_test_patient_generator()
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=3,
        include_steps_since_amr_update_in_obs=False
    )
    
    obs, _ = env.reset(seed=42)
    initial_amr_a = env.amr_balloon_models["A"].get_volume()
    initial_amr_b = env.amr_balloon_models["B"].get_volume()
    
    # Prescribe antibiotics to build up AMR
    for _ in range(5):
        action = np.array([0, 1, 0], dtype=int)  # Prescribe A, B, A
        env.step(action)
    
    amr_a_before_reset = env.amr_balloon_models["A"].get_volume()
    amr_b_before_reset = env.amr_balloon_models["B"].get_volume()
    
    # AMR should have increased
    assert amr_a_before_reset > initial_amr_a
    assert amr_b_before_reset > initial_amr_b
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    amr_a_after_reset = env.amr_balloon_models["A"].get_volume()
    amr_b_after_reset = env.amr_balloon_models["B"].get_volume()
    
    # AMR should be back to initial levels
    assert np.isclose(amr_a_after_reset, initial_amr_a), "AMR for A should reset"
    assert np.isclose(amr_b_after_reset, initial_amr_b), "AMR for B should reset"


def test_reproducibility_with_seed():
    """Test that same seed produces reproducible results."""
    antibiotic_names = ["A"]
    
    # Create two independent environments with same seed
    reward_calculator1 = create_test_reward_calculator(antibiotic_names)
    patient_generator1 = create_test_patient_generator()
    
    env1 = create_mock_environment(
        reward_calculator=reward_calculator1,
        patient_generator=patient_generator1,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=5,
        include_steps_since_amr_update_in_obs=False
    )
    
    reward_calculator2 = create_test_reward_calculator(antibiotic_names)
    patient_generator2 = create_test_patient_generator()
    
    env2 = create_mock_environment(
        reward_calculator=reward_calculator2,
        patient_generator=patient_generator2,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=5,
        include_steps_since_amr_update_in_obs=False
    )
    
    # Reset both with same seed
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    # Initial observations should be identical
    assert np.allclose(obs1, obs2), "Initial observations should match with same seed"
    
    # Take identical actions
    action = np.array([0, 0, 0, 1, 1], dtype=int)
    obs1, r1, _, _, info1 = env1.step(action)
    obs2, r2, _, _, info2 = env2.step(action)
    
    # Results should be identical
    assert np.allclose(obs1, obs2), "Observations should match with same seed"
    assert np.isclose(r1, r2), "Rewards should match with same seed"
    assert info1["actual_amr_levels"]["A"] == info2["actual_amr_levels"]["A"]


def test_multipliers_affect_rewards():
    """Test that patient multipliers affect reward calculation."""
    antibiotic_names = ["A"]
    
    # Create generator with high variance to get diverse multipliers
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    patient_generator = create_test_patient_generator(high_variance=True)
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=10,
        include_steps_since_amr_update_in_obs=False
    )
    
    # Run multiple episodes and collect rewards
    rewards = []
    for seed in range(5):
        env.reset(seed=seed)
        action = np.zeros(10, dtype=int)  # Prescribe A to all
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    
    # Rewards should vary because patient multipliers vary
    reward_std = np.std(rewards)
    assert reward_std > 0.0, "Rewards should vary due to heterogeneous patient multipliers"


def test_observed_vs_true_multipliers():
    """Test that agent sees observed multipliers, not true multipliers."""
    antibiotic_names = ["A"]
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    
    # Create generator with observation noise
    config = {
        'prob_infected': {
            'prob_dist': {'type': 'gaussian', 'mu': 0.5, 'sigma': 0.05},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'benefit_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.1,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'benefit_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'recovery_without_treatment_prob': {
            'prob_dist': {'type': 'constant', 'value': 0.01},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'visible_patient_attributes': ['benefit_value_multiplier'],
    }
    patient_generator = PatientGenerator(config=config)
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=['benefit_value_multiplier'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=5,
        include_steps_since_amr_update_in_obs=False
    )
    
    obs, _ = env.reset(seed=42)
    
    # Observation should use _obs versions (with noise)
    # Extract benefit_value_multiplier from observation (first 5 values)
    observed_multipliers = obs[:5]
    
    # Compare with true multipliers
    true_multipliers = [p.benefit_value_multiplier for p in env.current_patients]
    observed_from_patients = [p.benefit_value_multiplier_obs for p in env.current_patients]
    
    # Observation should match _obs versions
    assert np.allclose(observed_multipliers, observed_from_patients), \
        "Observation should use observed (_obs) multipliers"
    
    # With noise, true and observed should generally differ
    # (though they might occasionally be close due to randomness)


def test_patient_cohort_consistency_within_step():
    """Test that the same patient cohort is used for entire step."""
    antibiotic_names = ["A"]
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    patient_generator = create_test_patient_generator()
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=3,
        include_steps_since_amr_update_in_obs=False
    )
    
    obs, _ = env.reset(seed=42)
    
    # Store current patients
    patients_before_step = env.current_patients
    
    # Take a step
    action = np.array([0, 0, 0], dtype=int)
    obs, reward, _, _, info = env.step(action)
    
    # After step, current_patients should be a NEW cohort
    patients_after_step = env.current_patients
    
    # Patient instances should be different (new cohort)
    assert patients_before_step is not patients_after_step
    assert patients_before_step != patients_after_step


def test_all_patients_have_all_attributes():
    """Test that all sampled patients have all required attributes."""
    antibiotic_names = ["A"]
    reward_calculator = create_test_reward_calculator(antibiotic_names)
    patient_generator = create_test_patient_generator()
    
    env = create_mock_environment(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        visible_patient_attributes=['prob_infected'],
        antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
        num_patients_per_time_step=5,
        include_steps_since_amr_update_in_obs=False
    )
    
    obs, _ = env.reset(seed=42)
    
    # Check that all patients have all required attributes
    required_attrs = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob',
        'benefit_value_multiplier_obs',
        'failure_value_multiplier_obs',
        'benefit_probability_multiplier_obs',
        'failure_probability_multiplier_obs',
        'recovery_without_treatment_prob_obs',
    ]
    
    for patient in env.current_patients:
        assert isinstance(patient, Patient)
        for attr in required_attrs:
            assert hasattr(patient, attr), f"Patient missing attribute: {attr}"
            value = getattr(patient, attr)
            assert isinstance(value, (int, float)), f"Attribute {attr} should be numeric"
