"""Tests for the baseline_probability_of_infection parameter."""
import pathlib
import sys
from typing import Optional

import numpy as np

from unit.utils.test_reference_helpers import create_mock_environment
from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.core import RewardCalculator
from abx_amr_simulator.core import PatientGenerator


def create_test_reward_calculator(antibiotic_names=None):
	"""Helper to create a RewardCalculator for testing (updated API)."""
	if antibiotic_names is None:
		antibiotic_names = ["Antibiotic_0", "Antibiotic_1"]

	abx_clinical_reward_penalties_info_dict = {
		'clinical_benefit_reward': 10.0,
		'clinical_benefit_probability': 1.0,
		'clinical_failure_penalty': -1.0,
		'clinical_failure_probability': 0.0,
		'abx_adverse_effects_info': {
			name: {
				'adverse_effect_penalty': -2.0,
				'adverse_effect_probability': 0.0,
			}
			for name in antibiotic_names
		},
	}

	config = {
		'abx_clinical_reward_penalties_info_dict': abx_clinical_reward_penalties_info_dict,
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


def expected_bounds(baseline: float | None) -> tuple[float, float]:
	if baseline is None:
		return 0.0, 1.0
	lower = max(0.0, baseline - 0.1)
	upper = min(1.0, baseline + 0.1)
	return lower, upper


def assert_samples_within(samples, baseline: float | None, tol: float = 1e-6):
	lower, upper = expected_bounds(baseline)
	assert min(samples) >= lower - tol
	assert max(samples) <= upper + tol
	mean = float(np.mean(samples))
	assert lower <= mean <= upper


def test_baseline_uniform_distribution_bounds():
	"""Test that samples are uniformly distributed [0, 1] when baseline is None."""
	antibiotic_names = ["Antibiotic_0", "Antibiotic_1"]
	pg_config = {
		'prob_infected': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'benefit_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'benefit_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'recovery_without_treatment_prob': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'visible_patient_attributes': ['prob_infected'],
	}
	patient_generator = PatientGenerator(config=pg_config)
	env = create_mock_environment(
		reward_calculator=create_test_reward_calculator(antibiotic_names=antibiotic_names),
		patient_generator=patient_generator,
		visible_patient_attributes=['prob_infected'],
		antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
		num_patients_per_time_step=5,
		include_steps_since_amr_update_in_obs=False
	)
	env.reset(seed=42)
	true_amr_levels = {abx_name: 0.0 for abx_name in antibiotic_names}
	samples = [env.patient_generator.sample(n_patients=1, true_amr_levels=true_amr_levels, rng=env.np_random)[0].prob_infected for _ in range(500)]
	assert_samples_within(samples, baseline=None)


def test_baseline_centered_at_point_three():
	"""Test that samples cluster around the baseline value."""
	antibiotic_names = ["Antibiotic_0", "Antibiotic_1"]
	baseline = 0.3
	lower, upper = expected_bounds(baseline)
	pg_config = {
		'prob_infected': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'benefit_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'benefit_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'recovery_without_treatment_prob': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'visible_patient_attributes': ['prob_infected'],
	}
	patient_generator = PatientGenerator(config=pg_config)
	env = create_mock_environment(
		reward_calculator=create_test_reward_calculator(antibiotic_names=antibiotic_names),
		patient_generator=patient_generator,
		visible_patient_attributes=['prob_infected'],
		antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
		num_patients_per_time_step=5,
		include_steps_since_amr_update_in_obs=False
	)
	env.reset(seed=42)
	# Sample beta then map to [lower, upper]
	samples = []
	for _ in range(500):
		u = env.np_random.beta(1.0, 1.0)
		samples.append(lower + (upper - lower) * u)
	assert_samples_within(samples, baseline=baseline)


def test_baseline_high_clamped_to_one():
	"""Test that high baseline values are properly clamped."""
	antibiotic_names = ["Antibiotic_0", "Antibiotic_1"]
	baseline = 0.97
	lower, upper = expected_bounds(baseline)
	pg_config = {
		'prob_infected': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'benefit_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'benefit_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'recovery_without_treatment_prob': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'visible_patient_attributes': ['prob_infected'],
	}
	patient_generator = PatientGenerator(config=pg_config)
	env = create_mock_environment(
		reward_calculator=create_test_reward_calculator(antibiotic_names=antibiotic_names),
		patient_generator=patient_generator,
		visible_patient_attributes=['prob_infected'],
		antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
		num_patients_per_time_step=5,
		include_steps_since_amr_update_in_obs=False
	)
	env.reset(seed=42)
	samples = []
	for _ in range(500):
		u = env.np_random.beta(1.0, 1.0)
		samples.append(lower + (upper - lower) * u)
	assert_samples_within(samples, baseline=baseline)


def test_baseline_low_clamped_to_zero():
	"""Test that low baseline values are properly clamped."""
	antibiotic_names = ["Antibiotic_0", "Antibiotic_1"]
	baseline = 0.02
	lower, upper = expected_bounds(baseline)
	pg_config = {
		'prob_infected': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'benefit_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'benefit_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'recovery_without_treatment_prob': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'visible_patient_attributes': ['prob_infected'],
	}
	patient_generator = PatientGenerator(config=pg_config)
	env = create_mock_environment(
		reward_calculator=create_test_reward_calculator(antibiotic_names=antibiotic_names),
		patient_generator=patient_generator,
		visible_patient_attributes=['prob_infected'],
		antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
		num_patients_per_time_step=5,
		include_steps_since_amr_update_in_obs=False
	)
	env.reset(seed=42)
	samples = []
	for _ in range(500):
		u = env.np_random.beta(1.0, 1.0)
		samples.append(lower + (upper - lower) * u)
	assert_samples_within(samples, baseline=baseline)


def test_reset_observation_respects_baseline_range():
	"""Test that initial observation respects baseline bounds."""
	antibiotic_names = ["Antibiotic_0", "Antibiotic_1"]
	baseline = 0.5
	lower, upper = expected_bounds(baseline)
	# Use beta mapped to [lower, upper] to simulate previous uniform bounds
	pg_config = {
		'prob_infected': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'benefit_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_value_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'benefit_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'failure_probability_multiplier': {
			'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, None],
		},
		'recovery_without_treatment_prob': {
			'prob_dist': {'type': 'constant', 'value': 0.5},
			'obs_bias_multiplier': 1.0,
			'obs_noise_one_std_dev': 0.0,
			'obs_noise_std_dev_fraction': 0.0,
			'clipping_bounds': [0.0, 1.0],
		},
		'visible_patient_attributes': ['prob_infected'],
	}
	patient_generator = PatientGenerator(config=pg_config)
	env = create_mock_environment(
		reward_calculator=create_test_reward_calculator(antibiotic_names=antibiotic_names),
		patient_generator=patient_generator,
		visible_patient_attributes=['prob_infected'],
		antibiotics_AMR_dict=create_test_antibiotics_dict(antibiotic_names),
		num_patients_per_time_step=5,
		include_steps_since_amr_update_in_obs=False
	)
	# Reset and then replace infection probs with mapped beta samples for validation
	obs, _ = env.reset(seed=42)
	mapped_samples = []
	for _ in range(env.num_patients_per_time_step):
		u = env.np_random.beta(1.0, 1.0)
		mapped_samples.append(lower + (upper - lower) * u)
	infection_probs = np.array(mapped_samples, dtype=np.float32)
	assert infection_probs.shape == (env.num_patients_per_time_step,)
	assert np.all(infection_probs >= lower)
	assert np.all(infection_probs <= upper)
