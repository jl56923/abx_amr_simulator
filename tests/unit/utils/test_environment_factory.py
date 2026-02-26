"""
Tests for create_environment() factory function.

Tests environment instantiation with pre-created RewardCalculator and PatientGenerator,
validation of architectural constraints, and proper configuration of environment parameters.
"""

import tempfile
import yaml
from pathlib import Path
import pytest

from abx_amr_simulator.utils import (
    create_environment,
    create_reward_calculator,
    create_patient_generator,
    load_config,
)
from abx_amr_simulator.core import ABXAMREnv


class TestEnvironmentFactory:
    """Tests for create_environment() factory function."""
    
    def test_creates_environment_with_injected_components(self):
        """Test that environment accepts pre-created RC and PG (enforced instantiation order)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create component configs
            rc_config = {
                'abx_clinical_reward_penalties_info_dict': {
                    'clinical_benefit_reward': 10.0,
                    'clinical_benefit_probability': 1.0,
                    'clinical_failure_penalty': -5.0,
                    'clinical_failure_probability': 1.0,
                    'abx_adverse_effects_info': {
                        'A': {
                            'adverse_effect_penalty': -1.0,
                            'adverse_effect_probability': 0.1,
                        }
                    }
                },
                'lambda_weight': 0.5,
                'epsilon': 0.05,
            }
            rc_path = tmpdir_path / "rc.yaml"
            with open(rc_path, 'w') as f:
                yaml.dump(rc_config, f)
            
            pg_config = {
                'prob_infected': {
                    'prob_dist': {'type': 'constant', 'value': 0.8},
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
                'visible_patient_attributes': ['prob_infected'],
            }
            pg_path = tmpdir_path / "pg.yaml"
            with open(pg_path, 'w') as f:
                yaml.dump(pg_config, f)
            
            env_config = {
                'num_patients_per_time_step': 10,
                'max_time_steps': 50,
                'antibiotics_AMR_dict': {
                    'A': {
                        'leak': 0.95,
                        'flatness_parameter': 5.0,
                        'permanent_residual_volume': 0.1,
                        'initial_amr_level': 0.3,
                    }
                }
            }
            env_path = tmpdir_path / "env.yaml"
            with open(env_path, 'w') as f:
                yaml.dump(env_config, f)
            
            # Create umbrella config
            umbrella = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc.yaml',
                'patient_generator': 'pg.yaml',
                'environment': 'env.yaml',
                'training': {'seed': 42}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella, f)
            
            # Load config and create components in correct order
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            pg = create_patient_generator(config=config)
            env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
            
            assert isinstance(env, ABXAMREnv)
            assert env.num_patients_per_time_step == 10
            assert env.max_time_steps == 50
    
    def test_validates_no_patient_generator_in_env_config(self):
        """Test that environment config cannot contain patient_generator (fails loudly)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create minimal component configs
            rc_config = {
                'abx_clinical_reward_penalties_info_dict': {
                    'clinical_benefit_reward': 10.0,
                    'clinical_benefit_probability': 1.0,
                    'clinical_failure_penalty': -5.0,
                    'clinical_failure_probability': 1.0,
                    'abx_adverse_effects_info': {
                        'A': {'adverse_effect_penalty': -1.0, 'adverse_effect_probability': 0.1}
                    }
                },
                'lambda_weight': 0.5,
                'epsilon': 0.05,
            }
            rc_path = tmpdir_path / "rc.yaml"
            with open(rc_path, 'w') as f:
                yaml.dump(rc_config, f)
            
            pg_config = {
                'prob_infected': {
                    'prob_dist': {'type': 'constant', 'value': 0.8},
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
                'visible_patient_attributes': ['prob_infected'],
            }
            pg_path = tmpdir_path / "pg.yaml"
            with open(pg_path, 'w') as f:
                yaml.dump(pg_config, f)
            
            # Environment config with illegal patient_generator key (should fail loudly)
            env_config = {
                'num_patients_per_time_step': 10,
                'max_time_steps': 50,
                'antibiotics_AMR_dict': {
                    'A': {
                        'leak': 0.95,
                        'flatness_parameter': 5.0,
                        'permanent_residual_volume': 0.1,
                        'initial_amr_level': 0.3,
                    }
                },
                'patient_generator': {'some': 'config'},  # ILLEGAL!
            }
            env_path = tmpdir_path / "env.yaml"
            with open(env_path, 'w') as f:
                yaml.dump(env_config, f)
            
            umbrella = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc.yaml',
                'patient_generator': 'pg.yaml',
                'environment': 'env.yaml',
                'training': {'seed': 42}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella, f)
            
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            pg = create_patient_generator(config=config)
            
            # Should raise ValueError about prohibited key
            with pytest.raises(ValueError, match="patient_generator.*environment config"):
                create_environment(config=config, reward_calculator=rc, patient_generator=pg)
    
    def test_leaky_balloon_parameters_passed_through(self):
        """Test that leaky balloon parameters are correctly configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create minimal configs
            rc_config = {
                'abx_clinical_reward_penalties_info_dict': {
                    'clinical_benefit_reward': 10.0,
                    'clinical_benefit_probability': 1.0,
                    'clinical_failure_penalty': -5.0,
                    'clinical_failure_probability': 1.0,
                    'abx_adverse_effects_info': {
                        'A': {'adverse_effect_penalty': -1.0, 'adverse_effect_probability': 0.1}
                    }
                },
                'lambda_weight': 0.5,
                'epsilon': 0.05,
            }
            rc_path = tmpdir_path / "rc.yaml"
            with open(rc_path, 'w') as f:
                yaml.dump(rc_config, f)
            
            pg_config = {
                'prob_infected': {
                    'prob_dist': {'type': 'constant', 'value': 0.8},
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
                'visible_patient_attributes': ['prob_infected'],
            }
            pg_path = tmpdir_path / "pg.yaml"
            with open(pg_path, 'w') as f:
                yaml.dump(pg_config, f)
            
            # Environment config with custom leaky balloon params
            env_config = {
                'num_patients_per_time_step': 10,
                'max_time_steps': 50,
                'antibiotics_AMR_dict': {
                    'A': {
                        'leak': 0.90,
                        'flatness_parameter': 10.0,
                        'permanent_residual_volume': 0.15,
                        'initial_amr_level': 0.25,
                    }
                }
            }
            env_path = tmpdir_path / "env.yaml"
            with open(env_path, 'w') as f:
                yaml.dump(env_config, f)
            
            umbrella = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc.yaml',
                'patient_generator': 'pg.yaml',
                'environment': 'env.yaml',
                'training': {'seed': 42}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella, f)
            
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            pg = create_patient_generator(config=config)
            env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
            
            # Verify leaky balloon parameters
            balloon = env.amr_balloon_models['A']
            assert balloon.leak == 0.90
            assert balloon.flatness_parameter == 10.0
            assert balloon.permanent_residual_volume == 0.15
    
    def test_observation_space_dimension_matches_pg_and_amr(self):
        """Test that observation space dimension = pg.obs_dim(num_patients) + num_abx."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create minimal configs
            rc_config = {
                'abx_clinical_reward_penalties_info_dict': {
                    'clinical_benefit_reward': 10.0,
                    'clinical_benefit_probability': 1.0,
                    'clinical_failure_penalty': -5.0,
                    'clinical_failure_probability': 1.0,
                    'abx_adverse_effects_info': {
                        'A': {'adverse_effect_penalty': -1.0, 'adverse_effect_probability': 0.1},
                        'B': {'adverse_effect_penalty': -1.5, 'adverse_effect_probability': 0.15},
                    }
                },
                'lambda_weight': 0.5,
                'epsilon': 0.05,
            }
            rc_path = tmpdir_path / "rc.yaml"
            with open(rc_path, 'w') as f:
                yaml.dump(rc_config, f)
            
            pg_config = {
                'prob_infected': {
                    'prob_dist': {'type': 'constant', 'value': 0.8},
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
                'visible_patient_attributes': ['prob_infected', 'benefit_value_multiplier'],  # 2 attrs
            }
            pg_path = tmpdir_path / "pg.yaml"
            with open(pg_path, 'w') as f:
                yaml.dump(pg_config, f)
            
            env_config = {
                'num_patients_per_time_step': 5,
                'max_time_steps': 50,
                'antibiotics_AMR_dict': {
                    'A': {
                        'leak': 0.95,
                        'flatness_parameter': 5.0,
                        'permanent_residual_volume': 0.1,
                        'initial_amr_level': 0.3,
                    },
                    'B': {
                        'leak': 0.95,
                        'flatness_parameter': 5.0,
                        'permanent_residual_volume': 0.1,
                        'initial_amr_level': 0.3,
                    }
                }
            }
            env_path = tmpdir_path / "env.yaml"
            with open(env_path, 'w') as f:
                yaml.dump(env_config, f)
            
            umbrella = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc.yaml',
                'patient_generator': 'pg.yaml',
                'environment': 'env.yaml',
                'training': {'seed': 42}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella, f)
            
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            pg = create_patient_generator(config=config)
            env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
            
            # Expected obs dim: (5 patients × 2 attrs) + 2 AMR levels = 12
            expected_obs_dim = 5 * 2 + 2
            assert env.observation_space.shape[0] == expected_obs_dim
