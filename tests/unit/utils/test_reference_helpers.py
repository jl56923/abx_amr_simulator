import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Reference helpers for creating mock PatientGenerator, RewardCalculator, and ABXAMREnv for tests.
Update these helpers to change mock environment construction globally.
"""
import numpy as np
from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.core import RewardCalculator
from abx_amr_simulator.core import PatientGenerator

def create_mock_patient_generator( baseline_probability_of_infection=0.5, std_dev_probability_of_infection=0.1):
    config = {
        'prob_infected': {
            'prob_dist': {'type': 'gaussian', 'mu': baseline_probability_of_infection, 'sigma': std_dev_probability_of_infection},
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

def create_mock_reward_calculator(antibiotic_names=None):
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
                } for name in antibiotic_names
            },
        },
        'lambda_weight': 0.5,
        'epsilon': 0.05,
        'seed': None,
    }
    return RewardCalculator(config=config)

def create_mock_antibiotics_dict(antibiotic_names=None):
    if antibiotic_names is None:
        antibiotic_names = ["A"]
    return {
        name: {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0
        } for name in antibiotic_names
    }

def create_mock_environment(
    antibiotic_names=None,
    num_patients_per_time_step=5,
    max_time_steps=10,
    update_visible_AMR_levels_every_n_timesteps=1,
    add_noise_to_visible_AMR_levels=0.0,
    add_bias_to_visible_AMR_levels=0.0,
    crossresistance_matrix=None,
    visible_patient_attributes=None,
    include_steps_since_amr_update_in_obs=False,
    patient_generator=None,
    reward_calculator=None,
    antibiotics_AMR_dict=None,
):
    if antibiotic_names is None:
        antibiotic_names = ["A"]
    if patient_generator is None:
        patient_generator = create_mock_patient_generator()
    if reward_calculator is None:
        reward_calculator = create_mock_reward_calculator(antibiotic_names)
    if antibiotics_AMR_dict is None:
        antibiotics_AMR_dict = create_mock_antibiotics_dict(antibiotic_names)
    if visible_patient_attributes is None:
        visible_patient_attributes = ['prob_infected']
    
    # Set visibility in patient_generator
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
