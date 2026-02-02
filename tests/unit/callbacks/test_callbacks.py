"""
Tests for custom Stable-Baselines3 callbacks.

Verifies:
- PatientStatsLoggingCallback extracts and logs patient stats correctly
- DetailedEvalCallback enables/disables log_full_patient_attributes flag
- DetailedEvalCallback collects and saves trajectory data
- Trajectory files have correct structure and content
"""

import copy
import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.core import PatientGenerator
from abx_amr_simulator.core import RewardCalculator
from abx_amr_simulator.callbacks import PatientStatsLoggingCallback, DetailedEvalCallback


@pytest.fixture
def simple_env_config():
    """Basic configuration for test environments."""
    return {
        'antibiotics_AMR_dict': {
            'Antibiotic_A': {
                'leak': 0.5,
                'flatness_parameter': 30,
                'permanent_residual_volume': 0.0,
                'initial_amr_level': 0.0
            }
        },
        'patient_generator_config': {
            'visible_patient_attributes': ['prob_infected'],
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.8},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0]
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None]
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None]
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None]
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None]
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0]
            }
        },
        'reward_calculator_config': {
            'lambda_weight': 0.5,
            'epsilon': 0.05,
            'abx_clinical_reward_penalties_info_dict': {
                'clinical_benefit_reward': 10.0,
                'clinical_benefit_probability': 0.9,
                'clinical_failure_penalty': -10.0,
                'clinical_failure_probability': 0.1,
                'abx_adverse_effects_info': {
                    'Antibiotic_A': {
                        'adverse_effect_penalty': -1.0,
                        'adverse_effect_probability': 0.1
                    }
                }
            }
        }
    }


@pytest.fixture
def test_env(simple_env_config):
    """Create a test environment."""
    pg_config = copy.deepcopy(simple_env_config['patient_generator_config'])
    # Config already has visible_patient_attributes set
    
    patient_gen = PatientGenerator(config=pg_config)
    reward_calc = RewardCalculator(config=simple_env_config['reward_calculator_config'])
    
    env = ABXAMREnv(
        reward_calculator=reward_calc,
        patient_generator=patient_gen,
        antibiotics_AMR_dict=simple_env_config['antibiotics_AMR_dict'],
        num_patients_per_time_step=3,
        max_time_steps=10,
    )
    return env


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)


class TestPatientStatsLoggingCallback:
    """Tests for PatientStatsLoggingCallback."""
    
    def test_callback_initialization(self):
        """Test that callback initializes correctly."""
        callback = PatientStatsLoggingCallback(verbose=1)
        assert callback.verbose == 1
        assert callback.patient_stats_buffer == []
    
    def test_extracts_patient_stats_from_info(self, test_env):
        """Test that callback extracts and processes patient_stats from info dict."""
        vec_env = DummyVecEnv([lambda: test_env])
        model = PPO('MlpPolicy', vec_env, verbose=0)
        
        # Set up logger (normally done during learn())
        from stable_baselines3.common.logger import Logger
        logger = Logger(folder=None, output_formats=[])
        model.set_logger(logger)
        
        callback = PatientStatsLoggingCallback()
        
        # Initialize callback with model
        callback.init_callback(model)
        
        # Create mock info with patient_stats
        test_env.reset(seed=42)
        obs, reward, terminated, truncated, info = test_env.step(np.array([0, 0, 0]))
        
        # Simulate what SB3 does: wrap info in a list
        callback.locals = {'infos': [info]}
        
        # Call _on_step (should not crash and should return True)
        result = callback._on_step()
        
        # Should return True (continue training)
        assert result is True
        
        # Verify that patient_stats was present in info
        assert 'patient_stats' in info
        assert isinstance(info['patient_stats'], dict)
        assert 'prob_infected_true_mean' in info['patient_stats']
    
    def test_handles_missing_patient_stats(self):
        """Test that callback handles missing patient_stats gracefully."""
        callback = PatientStatsLoggingCallback()
        
        # Info without patient_stats
        callback.locals = {'infos': [{'some_other_key': 123}]}
        
        # Should not crash and return True
        result = callback._on_step()
        assert result is True
    
    def test_aggregates_stats_across_multiple_envs(self):
        """Test that callback aggregates stats from multiple environments."""
        callback = PatientStatsLoggingCallback()
        
        # Create multiple stat dicts simulating different envs
        stats_list = [
            {'prob_infected_true_mean': 0.7, 'prob_infected_obs_mean': 0.72},
            {'prob_infected_true_mean': 0.9, 'prob_infected_obs_mean': 0.88},
        ]
        
        aggregated = callback._aggregate_stats(stats_list)
        
        # Should take mean across envs
        assert np.isclose(aggregated['prob_infected_true_mean'], 0.8)  # (0.7 + 0.9) / 2
        assert np.isclose(aggregated['prob_infected_obs_mean'], 0.8)  # (0.72 + 0.88) / 2


class TestDetailedEvalCallback:
    """Tests for DetailedEvalCallback."""
    
    def test_callback_initialization(self, test_env, temp_log_dir):
        """Test that callback initializes correctly."""
        vec_env = DummyVecEnv([lambda: test_env])
        
        callback = DetailedEvalCallback(
            eval_env=vec_env,
            n_eval_episodes=2,
            eval_freq=100,
            log_path=temp_log_dir,
            save_patient_trajectories=True,
            verbose=1
        )
        
        assert callback.n_eval_episodes == 2
        assert callback.eval_freq == 100
        assert callback.save_patient_trajectories is True
        assert callback.eval_count == 0
        
        # Check that eval_logs directory was created
        eval_logs_dir = Path(temp_log_dir) / 'eval_logs'
        assert eval_logs_dir.exists()
    
    def test_sets_logging_flag_on_eval_env(self, test_env, temp_log_dir):
        """Test that callback enables/disables log_full_patient_attributes flag."""
        vec_env = DummyVecEnv([lambda: test_env])
        
        callback = DetailedEvalCallback(
            eval_env=vec_env,
            n_eval_episodes=1,
            eval_freq=100,
            log_path=temp_log_dir,
            save_patient_trajectories=True
        )
        
        # Initially should be False
        assert test_env.log_full_patient_attributes is False
        
        # Enable logging
        callback._set_eval_env_logging_flag(True)
        assert test_env.log_full_patient_attributes is True
        
        # Disable logging
        callback._set_eval_env_logging_flag(False)
        assert test_env.log_full_patient_attributes is False
    
    def test_saves_trajectory_files(self, test_env, temp_log_dir):
        """Test that callback saves trajectory files with correct structure."""
        vec_env = DummyVecEnv([lambda: test_env])
        
        # Create a simple PPO model
        model = PPO('MlpPolicy', vec_env, verbose=0)
        
        callback = DetailedEvalCallback(
            eval_env=vec_env,
            n_eval_episodes=2,
            eval_freq=10,  # Evaluate after 10 steps
            log_path=temp_log_dir,
            save_patient_trajectories=True,
            deterministic=True,
            verbose=0
        )
        
        # Train for a few steps to trigger evaluation
        callback.init_callback(model)
        
        # Manually trigger evaluation
        callback._set_eval_env_logging_flag(True)
        
        # Simulate a short episode
        trajectories = []
        episode_rewards = []
        episode_lengths = []
        
        obs = vec_env.reset()
        episode_reward = 0
        episode_data = {
            'patient_full_data': [],
            'patient_stats': [],
            'actions': [],
            'rewards': [],
            'actual_amr_levels': [],
            'visible_amr_levels': [],
        }
        
        for _ in range(3):  # Short episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            info = info[0]  # Unwrap from list
            
            if 'patient_full_data' in info:
                episode_data['patient_full_data'].append(info['patient_full_data'])
                episode_data['patient_stats'].append(info['patient_stats'])
                episode_data['actions'].append(action)
                episode_data['rewards'].append(float(reward))
                episode_data['actual_amr_levels'].append(info.get('actual_amr_levels', {}))
                episode_data['visible_amr_levels'].append(info.get('visible_amr_levels', {}))
            
            episode_reward += reward
            
            if done[0]:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(len(episode_data['actions']))
        trajectories.append(episode_data)
        
        callback._set_eval_env_logging_flag(False)
        
        # Save trajectories
        callback._save_trajectories(trajectories, episode_rewards, episode_lengths)
        
        # Check that file was created
        eval_logs_dir = Path(temp_log_dir) / 'eval_logs'
        saved_files = list(eval_logs_dir.glob('*.npz'))
        assert len(saved_files) == 1
        
        # Load and verify structure
        data = np.load(saved_files[0], allow_pickle=True)
        
        # Check top-level keys
        assert 'episode_rewards' in data
        assert 'episode_lengths' in data
        assert 'num_episodes' in data
        
        # Check episode data exists
        assert 'episode_0/patient_true' in data or 'episode_0/actions' in data
    
    def test_trajectory_data_structure(self, test_env, temp_log_dir):
        """Test that saved trajectory data has correct shape and content."""
        vec_env = DummyVecEnv([lambda: test_env])
        model = PPO('MlpPolicy', vec_env, verbose=0)
        
        callback = DetailedEvalCallback(
            eval_env=vec_env,
            n_eval_episodes=1,
            eval_freq=10,
            log_path=temp_log_dir,
            save_patient_trajectories=True,
            deterministic=True,
            verbose=0
        )
        
        callback.init_callback(model)
        callback._set_eval_env_logging_flag(True)
        
        # Run one short episode
        trajectories = []
        obs = vec_env.reset()
        episode_data = {
            'patient_full_data': [],
            'patient_stats': [],
            'actions': [],
            'rewards': [],
            'actual_amr_levels': [],
            'visible_amr_levels': [],
        }
        
        num_steps = 3
        for _ in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            info = info[0]
            episode_data['patient_full_data'].append(info['patient_full_data'])
            episode_data['patient_stats'].append(info['patient_stats'])
            episode_data['actions'].append(action)
            episode_data['rewards'].append(float(reward))
            episode_data['actual_amr_levels'].append(info.get('actual_amr_levels', {}))
            episode_data['visible_amr_levels'].append(info.get('visible_amr_levels', {}))
        
        trajectories.append(episode_data)
        callback._save_trajectories(trajectories, [0.0], [num_steps])
        callback._set_eval_env_logging_flag(False)
        
        # Load saved data
        eval_logs_dir = Path(temp_log_dir) / 'eval_logs'
        saved_file = list(eval_logs_dir.glob('*.npz'))[0]
        data = np.load(saved_file, allow_pickle=True)
        
        # Verify shapes
        if 'episode_0/patient_true' in data:
            patient_true = data['episode_0/patient_true']
            # Should be (num_steps, num_patients, num_attrs)
            assert patient_true.shape[0] == num_steps
            assert patient_true.shape[1] == 3  # num_patients_per_time_step
            assert patient_true.shape[2] == 6  # 6 patient attributes
            
            # Check observed data matches
            patient_obs = data['episode_0/patient_observed']
            assert patient_obs.shape == patient_true.shape
        
        # Check actions
        if 'episode_0/actions' in data:
            actions = data['episode_0/actions']
            assert len(actions) == num_steps
    
    def test_disables_flag_after_evaluation(self, test_env, temp_log_dir):
        """Test that flag is disabled after evaluation completes."""
        vec_env = DummyVecEnv([lambda: test_env])
        
        callback = DetailedEvalCallback(
            eval_env=vec_env,
            n_eval_episodes=1,
            eval_freq=100,
            log_path=temp_log_dir,
            save_patient_trajectories=True
        )
        
        # Flag should start False
        assert test_env.log_full_patient_attributes is False
        
        # Enable it
        callback._set_eval_env_logging_flag(True)
        assert test_env.log_full_patient_attributes is True
        
        # Disable it (simulating end of eval)
        callback._set_eval_env_logging_flag(False)
        assert test_env.log_full_patient_attributes is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
