"""
Real-instance helpers for creating PatientGenerator, RewardCalculator, and ABXAMREnv
in tests.

All helpers construct genuine objects — no stubs or MagicMocks.  If you need to
change how test environments are built, update these helpers and every test that
imports them will pick up the change automatically.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from abx_amr_simulator.core import ABXAMREnv, RewardCalculator, PatientGenerator


def make_pg(
    baseline_probability_of_infection: float = 0.5,
    std_dev_probability_of_infection: float = 0.1,
) -> PatientGenerator:
    """Return a real PatientGenerator with lightweight constant distributions."""
    config = {
        'prob_infected': {
            'prob_dist': {
                'type': 'gaussian',
                'mu': baseline_probability_of_infection,
                'sigma': std_dev_probability_of_infection,
            },
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
            'prob_dist': {'type': 'constant', 'value': 0.01},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'visible_patient_attributes': ['prob_infected'],
    }
    return PatientGenerator(config=config)


def make_rc(antibiotic_names=None) -> RewardCalculator:
    """Return a real RewardCalculator for the given antibiotic names."""
    if antibiotic_names is None:
        antibiotic_names = ["A"]
    config = {
        'abx_clinical_reward_penalties_info_dict': {
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
        },
        'lambda_weight': 0.5,
        'seed': None,
    }
    return RewardCalculator(config=config)


def make_antibiotics_dict(antibiotic_names=None) -> dict:
    """Return a minimal antibiotics AMR dict for the given antibiotic names."""
    if antibiotic_names is None:
        antibiotic_names = ["A"]
    return {
        name: {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0,
        }
        for name in antibiotic_names
    }


def make_env(
    antibiotic_names=None,
    num_patients_per_time_step: int = 5,
    max_time_steps: int = 10,
    update_visible_AMR_levels_every_n_timesteps: int = 1,
    add_noise_to_visible_AMR_levels: float = 0.0,
    add_bias_to_visible_AMR_levels: float = 0.0,
    crossresistance_matrix=None,
    visible_patient_attributes=None,
    include_steps_since_amr_update_in_obs: bool = False,
    patient_generator=None,
    reward_calculator=None,
    antibiotics_AMR_dict=None,
) -> ABXAMREnv:
    """Return a real lightweight ABXAMREnv instance.

    All components are real objects.  Override individual components by passing
    pre-built patient_generator, reward_calculator, or antibiotics_AMR_dict.
    """
    if antibiotic_names is None:
        antibiotic_names = ["A"]
    if patient_generator is None:
        patient_generator = make_pg()
    if reward_calculator is None:
        reward_calculator = make_rc(antibiotic_names)
    if antibiotics_AMR_dict is None:
        antibiotics_AMR_dict = make_antibiotics_dict(antibiotic_names)
    if visible_patient_attributes is None:
        visible_patient_attributes = ['prob_infected']

    patient_generator.visible_patient_attributes = visible_patient_attributes

    return ABXAMREnv(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        antibiotics_AMR_dict=antibiotics_AMR_dict,
        num_patients_per_time_step=num_patients_per_time_step,
        update_visible_AMR_levels_every_n_timesteps=update_visible_AMR_levels_every_n_timesteps,
        add_noise_to_visible_AMR_levels=add_noise_to_visible_AMR_levels,
        add_bias_to_visible_AMR_levels=add_bias_to_visible_AMR_levels,
        max_time_steps=max_time_steps,
        crossresistance_matrix=crossresistance_matrix,
        include_steps_since_amr_update_in_obs=include_steps_since_amr_update_in_obs,
    )


# ---------------------------------------------------------------------------
# Backward-compat aliases — kept so that any remaining old import sites
# continue to work during migration.  Remove once all callers are updated.
# ---------------------------------------------------------------------------
create_mock_patient_generator = make_pg
create_mock_reward_calculator = make_rc
create_mock_antibiotics_dict = make_antibiotics_dict
create_mock_environment = make_env
