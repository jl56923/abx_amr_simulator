"""
Tests for create_agent() factory function.

Tests agent instantiation for all supported algorithms: PPO, RecurrentPPO, A2C,
HRL_PPO, and HRL_RPPO.
"""

import tempfile
import yaml
from pathlib import Path
import pytest

from abx_amr_simulator.utils import (
    create_agent,
    create_environment,
    create_reward_calculator,
    create_patient_generator,
    load_config,
)


class TestAgentFactory:
    """Tests for create_agent() factory function."""
    
    def _create_test_env(self, tmpdir_path):
        """Helper to create a minimal test environment."""
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
        
        env_config = {
            'num_patients_per_time_step': 5,
            'max_time_steps': 10,
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
        
        return rc_path, pg_path, env_path
    
    def test_creates_ppo_agent(self):
        """Test PPO agent instantiation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rc_path, pg_path, env_path = self._create_test_env(tmpdir_path)
            
            # Create agent config for PPO
            agent_config = {
                'algorithm': 'PPO',
                'policy': 'MlpPolicy',
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
            }
            agent_path = tmpdir_path / "agent.yaml"
            with open(agent_path, 'w') as f:
                yaml.dump(agent_config, f)
            
            # Create umbrella config
            umbrella = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc.yaml',
                'patient_generator': 'pg.yaml',
                'environment': 'env.yaml',
                'agent_algorithm': 'agent.yaml',
                'training': {'seed': 42}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella, f)
            
            # Load and create
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            pg = create_patient_generator(config=config)
            env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
            agent = create_agent(config=config, env=env)
            
            # Verify it's a PPO instance
            assert agent.__class__.__name__ == 'PPO'
            assert agent.learning_rate == 0.0003
    
    def test_creates_recurrent_ppo_agent(self):
        """Test RecurrentPPO agent instantiation (distinct from PPO)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rc_path, pg_path, env_path = self._create_test_env(tmpdir_path)
            
            # Create agent config for RecurrentPPO
            agent_config = {
                'algorithm': 'RecurrentPPO',
                'policy': 'MlpLstmPolicy',
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
            }
            agent_path = tmpdir_path / "agent.yaml"
            with open(agent_path, 'w') as f:
                yaml.dump(agent_config, f)
            
            umbrella = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc.yaml',
                'patient_generator': 'pg.yaml',
                'environment': 'env.yaml',
                'agent_algorithm': 'agent.yaml',
                'training': {'seed': 42}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella, f)
            
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            pg = create_patient_generator(config=config)
            env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
            agent = create_agent(config=config, env=env)
            
            # Verify it's RecurrentPPO (not regular PPO)
            assert agent.__class__.__name__ == 'RecurrentPPO'
    
    
    def test_creates_a2c_agent(self):
        """Test A2C agent instantiation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rc_path, pg_path, env_path = self._create_test_env(tmpdir_path)
            
            agent_config = {
                'algorithm': 'A2C',
                'policy': 'MlpPolicy',
                'learning_rate': 0.0007,
                'n_steps': 5,
                'gamma': 0.99,
            }
            agent_path = tmpdir_path / "agent.yaml"
            with open(agent_path, 'w') as f:
                yaml.dump(agent_config, f)
            
            umbrella = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc.yaml',
                'patient_generator': 'pg.yaml',
                'environment': 'env.yaml',
                'agent_algorithm': 'agent.yaml',
                'training': {'seed': 42}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella, f)
            
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            pg = create_patient_generator(config=config)
            env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
            agent = create_agent(config=config, env=env)
            
            assert agent.__class__.__name__ == 'A2C'
    
    def test_validates_unsupported_algorithm(self):
        """Test that factory raises error for unsupported algorithms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rc_path, pg_path, env_path = self._create_test_env(tmpdir_path)
            
            # Use an unsupported algorithm
            agent_config = {
                'algorithm': 'UNSUPPORTED_ALG',
                'policy': 'MlpPolicy',
            }
            agent_path = tmpdir_path / "agent.yaml"
            with open(agent_path, 'w') as f:
                yaml.dump(agent_config, f)
            
            umbrella = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc.yaml',
                'patient_generator': 'pg.yaml',
                'environment': 'env.yaml',
                'agent_algorithm': 'agent.yaml',
                'training': {'seed': 42}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella, f)
            
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            pg = create_patient_generator(config=config)
            env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
            
            # Should raise ValueError about unsupported algorithm
            with pytest.raises(ValueError, match="Unknown algorithm"):
                create_agent(config=config, env=env)
