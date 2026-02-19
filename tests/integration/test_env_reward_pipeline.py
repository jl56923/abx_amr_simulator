"""Integration tests for reward pipeline, crossresistance, and AMR update observations."""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from abx_amr_simulator.core import ABXAMREnv, PatientGenerator, RewardCalculator


def _create_reward_calculator(
    antibiotic_names: List[str],
    lambda_weight: float = 0.0,
    epsilon: float = 0.0,
) -> RewardCalculator:
    config = {
        "abx_clinical_reward_penalties_info_dict": {
            "clinical_benefit_reward": 10.0,
            "clinical_benefit_probability": 1.0,
            "clinical_failure_penalty": -5.0,
            "clinical_failure_probability": 0.0,
            "abx_adverse_effects_info": {
                name: {
                    "adverse_effect_penalty": -1.0,
                    "adverse_effect_probability": 0.0,
                }
                for name in antibiotic_names
            },
        },
        "lambda_weight": lambda_weight,
        "epsilon": epsilon,
        "seed": 123,
    }
    return RewardCalculator(config=config)


def _create_constant_patient_generator(
    visible_patient_attributes: List[str],
) -> PatientGenerator:
    config = {
        "prob_infected": {
            "prob_dist": {"type": "constant", "value": 1.0},
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
            "prob_dist": {"type": "constant", "value": 0.0},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, 1.0],
        },
        "visible_patient_attributes": visible_patient_attributes,
    }
    return PatientGenerator(config=config)


def _create_antibiotics_amr_dict(antibiotic_names: List[str]) -> Dict[str, Dict[str, float]]:
    return {
        name: {
            "leak": 0.05,
            "flatness_parameter": 1.0,
            "permanent_residual_volume": 0.0,
            "initial_amr_level": 0.0,
        }
        for name in antibiotic_names
    }


def test_reward_pipeline_matches_expected_deterministic() -> None:
    antibiotic_names = ["A"]
    reward_calculator = _create_reward_calculator(
        antibiotic_names=antibiotic_names,
        lambda_weight=0.0,
        epsilon=0.0,
    )
    patient_generator = _create_constant_patient_generator(
        visible_patient_attributes=["prob_infected"],
    )

    env = ABXAMREnv(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        antibiotics_AMR_dict=_create_antibiotics_amr_dict(antibiotic_names),
        num_patients_per_time_step=2,
        update_visible_AMR_levels_every_n_timesteps=1,
        add_noise_to_visible_AMR_levels=0.0,
        add_bias_to_visible_AMR_levels=0.0,
        max_time_steps=5,
        include_steps_since_amr_update_in_obs=False,
        crossresistance_matrix=None,
    )

    env.reset(seed=42)
    patients_before = list(env.current_patients)
    actions = np.zeros(shape=(2,), dtype=int)
    rng_state = copy.deepcopy(env.np_random.bit_generator.state)

    _, reward, _, _, info = env.step(action=actions)

    rng_copy = np.random.default_rng()
    rng_copy.bit_generator.state = rng_state
    expected, _ = reward_calculator.calculate_reward(
        patients=patients_before,
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=info["visible_amr_levels"],
        delta_visible_amr_per_antibiotic=info["delta_visible_amr_per_antibiotic"],
        rng=rng_copy,
    )

    assert np.isclose(reward, expected, atol=1e-6)


def test_crossresistance_end_to_end_coupling() -> None:
    antibiotic_names = ["A", "B"]
    reward_calculator = _create_reward_calculator(antibiotic_names=antibiotic_names)
    patient_generator = _create_constant_patient_generator(
        visible_patient_attributes=["prob_infected"],
    )
    crossresistance_matrix = {"A": {"B": 0.5}}

    env = ABXAMREnv(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        antibiotics_AMR_dict=_create_antibiotics_amr_dict(antibiotic_names),
        num_patients_per_time_step=4,
        update_visible_AMR_levels_every_n_timesteps=1,
        add_noise_to_visible_AMR_levels=0.0,
        add_bias_to_visible_AMR_levels=0.0,
        max_time_steps=3,
        include_steps_since_amr_update_in_obs=False,
        crossresistance_matrix=crossresistance_matrix,
    )

    env.reset(seed=1)
    initial_b = env.amr_balloon_models["B"].get_volume()
    actions = np.zeros(shape=(4,), dtype=int)

    _, _, _, _, info = env.step(action=actions)

    expected_contribution = 4 * 0.5
    assert info["effective_doses"]["B"] == expected_contribution
    assert info["crossresistance_applied"]["B"]["A"] == expected_contribution
    assert info["actual_amr_levels"]["B"] > initial_b


def test_observation_includes_steps_since_amr_update() -> None:
    antibiotic_names = ["A"]
    reward_calculator = _create_reward_calculator(antibiotic_names=antibiotic_names)
    patient_generator = _create_constant_patient_generator(
        visible_patient_attributes=["prob_infected"],
    )

    env = ABXAMREnv(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        antibiotics_AMR_dict=_create_antibiotics_amr_dict(antibiotic_names),
        num_patients_per_time_step=2,
        update_visible_AMR_levels_every_n_timesteps=2,
        add_noise_to_visible_AMR_levels=0.0,
        add_bias_to_visible_AMR_levels=0.0,
        max_time_steps=4,
        include_steps_since_amr_update_in_obs=True,
        crossresistance_matrix=None,
    )

    obs, _ = env.reset(seed=7)
    expected_length = patient_generator.obs_dim(num_patients=2) + len(antibiotic_names) + 1
    assert obs.shape == (expected_length,)
    assert int(obs[-1]) == 0

    no_treatment_action = env.no_treatment_action
    actions = np.full(shape=(2,), fill_value=no_treatment_action, dtype=int)

    obs_step1, _, _, _, _ = env.step(action=actions)
    assert int(obs_step1[-1]) == 1

    obs_step2, _, _, _, _ = env.step(action=actions)
    assert int(obs_step2[-1]) == 0


def test_multi_patient_action_mapping_counts_prescriptions() -> None:
    antibiotic_names = ["A", "B"]
    reward_calculator = _create_reward_calculator(antibiotic_names=antibiotic_names)
    patient_generator = _create_constant_patient_generator(
        visible_patient_attributes=["prob_infected"],
    )

    env = ABXAMREnv(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        antibiotics_AMR_dict=_create_antibiotics_amr_dict(antibiotic_names),
        num_patients_per_time_step=3,
        update_visible_AMR_levels_every_n_timesteps=1,
        add_noise_to_visible_AMR_levels=0.0,
        add_bias_to_visible_AMR_levels=0.0,
        max_time_steps=3,
        include_steps_since_amr_update_in_obs=False,
        crossresistance_matrix=None,
    )

    env.reset(seed=11)
    no_treatment_action = env.no_treatment_action
    actions = np.array([0, 1, no_treatment_action], dtype=int)

    _, _, _, _, info = env.step(action=actions)

    assert info["prescriptions_per_abx"]["A"] == 1
    assert info["prescriptions_per_abx"]["B"] == 1
    assert info["crossresistance_applied"]["A"] == {"A": 1}
    assert info["crossresistance_applied"]["B"] == {"B": 1}
