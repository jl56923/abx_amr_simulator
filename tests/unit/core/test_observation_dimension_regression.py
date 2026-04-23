"""Regression tests for baseline environment observation dimensions."""

from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.core import PatientGenerator
from abx_amr_simulator.core import RewardCalculator


def test_observation_dimension_matches_baseline_without_temporal_features():
    """Observation dimension should be patient features + AMR levels only."""
    visible_attributes = [
        "prob_infected",
        "benefit_value_multiplier",
    ]
    antibiotic_names = ["A", "B", "C"]
    num_patients = 4

    patient_generator_config = {
        "prob_infected": {
            "prob_dist": {"type": "constant", "value": 0.5},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, 1.0],
        },
        "benefit_value_multiplier": {
            "prob_dist": {"type": "constant", "value": 1.0},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "failure_value_multiplier": {
            "prob_dist": {"type": "constant", "value": 1.0},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "benefit_probability_multiplier": {
            "prob_dist": {"type": "constant", "value": 1.0},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "failure_probability_multiplier": {
            "prob_dist": {"type": "constant", "value": 1.0},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "recovery_without_treatment_prob": {
            "prob_dist": {"type": "constant", "value": 0.1},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, 1.0],
        },
        "visible_patient_attributes": visible_attributes,
    }
    patient_generator = PatientGenerator(config=patient_generator_config)

    reward_calculator_config = {
        "abx_clinical_reward_penalties_info_dict": {
            "clinical_benefit_reward": 10.0,
            "clinical_benefit_probability": 1.0,
            "clinical_failure_penalty": -1.0,
            "clinical_failure_probability": 0.0,
            "abx_adverse_effects_info": {
                abx_name: {
                    "adverse_effect_penalty": -0.1,
                    "adverse_effect_probability": 0.0,
                }
                for abx_name in antibiotic_names
            },
        },
        "lambda_weight": 0.5,
    }
    reward_calculator = RewardCalculator(config=reward_calculator_config)

    antibiotics_amr_dict = {
        abx_name: {
            "leak": 0.05,
            "flatness_parameter": 1.0,
            "permanent_residual_volume": 0.0,
            "initial_amr_level": 0.0,
        }
        for abx_name in antibiotic_names
    }

    env = ABXAMREnv(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        antibiotics_AMR_dict=antibiotics_amr_dict,
        num_patients_per_time_step=num_patients,
        max_time_steps=5,
    )

    expected_obs_dim = patient_generator.obs_dim(num_patients=num_patients) + len(antibiotic_names)
    assert env.observation_space.shape == (expected_obs_dim,)

    observation, _ = env.reset(seed=7)
    assert observation.shape == (expected_obs_dim,)
