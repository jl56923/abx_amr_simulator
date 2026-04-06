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
            'manager_clipped': [],
            'steps_clipped': [],
            'manager_transition_trainable': [],
            'option_id': [],
            'primitive_actions': [],
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
                episode_data['manager_clipped'].append(info.get('manager_clipped', False))
                episode_data['steps_clipped'].append(info.get('steps_clipped', 0))
                episode_data['manager_transition_trainable'].append(info.get('manager_transition_trainable', True))
                episode_data['option_id'].append(info.get('option_id', None))
                episode_data['primitive_actions'].append(info.get('primitive_actions', None))

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
            'manager_clipped': [],
            'steps_clipped': [],
            'manager_transition_trainable': [],
            'option_id': [],
            'primitive_actions': [],
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
            episode_data['manager_clipped'].append(info.get('manager_clipped', False))
            episode_data['steps_clipped'].append(info.get('steps_clipped', 0))
            episode_data['manager_transition_trainable'].append(info.get('manager_transition_trainable', True))
            episode_data['option_id'].append(info.get('option_id', None))
            episode_data['primitive_actions'].append(info.get('primitive_actions', None))
        
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

class TestDetailedEvalCallbackHRLLogging:
    """Tests for Phase B clipping metadata and HRL primitive-action logging."""

    def _make_traj(self, num_steps, option_ids=None, primitive_actions=None):
        """Build a minimal trajectory dict that _save_trajectories accepts."""
        return {
            'patient_full_data': [],   # empty — skip patient array saving branch
            'patient_stats': [],
            'actions': list(range(num_steps)),
            'rewards': [0.0] * num_steps,
            'patients_actually_infected': None,
            'individual_rewards': None,
            'actual_amr_levels': [],
            'visible_amr_levels': [],
            'antibiotic_names': [],
            'manager_clipped': [False] * num_steps,
            'steps_clipped': [0] * num_steps,
            'manager_transition_trainable': [True] * num_steps,
            'option_id': option_ids if option_ids is not None else [None] * num_steps,
            'primitive_actions': primitive_actions if primitive_actions is not None else [None] * num_steps,
        }

    def test_clipping_metadata_saved_for_non_hrl(self, test_env, temp_log_dir):
        """Phase B clipping fields are always written, even for non-HRL runs."""
        vec_env = DummyVecEnv([lambda: test_env])
        model = PPO('MlpPolicy', vec_env, verbose=0)
        callback = DetailedEvalCallback(
            eval_env=vec_env, n_eval_episodes=1, eval_freq=10,
            log_path=temp_log_dir, save_patient_trajectories=True, verbose=0
        )
        callback.init_callback(model)

        num_steps = 4
        traj = self._make_traj(num_steps)
        callback._save_trajectories([traj], [1.0], [num_steps])

        npz_path = list((Path(temp_log_dir) / 'eval_logs').glob('*.npz'))[0]
        data = np.load(npz_path, allow_pickle=True)

        assert 'episode_0/manager_clipped' in data
        assert 'episode_0/steps_clipped' in data
        assert 'episode_0/manager_transition_trainable' in data
        assert len(data['episode_0/manager_clipped']) == num_steps
        assert len(data['episode_0/steps_clipped']) == num_steps
        assert len(data['episode_0/manager_transition_trainable']) == num_steps

    def test_option_id_saved_when_present(self, test_env, temp_log_dir):
        """option_id is written to npz when non-None values are present."""
        vec_env = DummyVecEnv([lambda: test_env])
        model = PPO('MlpPolicy', vec_env, verbose=0)
        callback = DetailedEvalCallback(
            eval_env=vec_env, n_eval_episodes=1, eval_freq=10,
            log_path=temp_log_dir, save_patient_trajectories=True, verbose=0
        )
        callback.init_callback(model)

        option_ids = [3, 3, 5, 3]
        traj = self._make_traj(len(option_ids), option_ids=option_ids)
        callback._save_trajectories([traj], [1.0], [len(option_ids)])

        npz_path = list((Path(temp_log_dir) / 'eval_logs').glob('*.npz'))[0]
        data = np.load(npz_path, allow_pickle=True)

        assert 'episode_0/option_id' in data
        np.testing.assert_array_equal(data['episode_0/option_id'], option_ids)

    def test_option_id_omitted_when_all_none(self, test_env, temp_log_dir):
        """option_id key is absent from npz when all values are None (non-HRL run)."""
        vec_env = DummyVecEnv([lambda: test_env])
        model = PPO('MlpPolicy', vec_env, verbose=0)
        callback = DetailedEvalCallback(
            eval_env=vec_env, n_eval_episodes=1, eval_freq=10,
            log_path=temp_log_dir, save_patient_trajectories=True, verbose=0
        )
        callback.init_callback(model)

        traj = self._make_traj(3)  # option_id all None
        callback._save_trajectories([traj], [0.0], [3])

        npz_path = list((Path(temp_log_dir) / 'eval_logs').glob('*.npz'))[0]
        data = np.load(npz_path, allow_pickle=True)

        assert 'episode_0/option_id' not in data

    def test_primitive_actions_saved_when_present(self, test_env, temp_log_dir):
        """primitive_actions (list of lists) is written to npz correctly."""
        vec_env = DummyVecEnv([lambda: test_env])
        model = PPO('MlpPolicy', vec_env, verbose=0)
        callback = DetailedEvalCallback(
            eval_env=vec_env, n_eval_episodes=1, eval_freq=10,
            log_path=temp_log_dir, save_patient_trajectories=True, verbose=0
        )
        callback.init_callback(model)

        # Simulate 3 macro-steps with k=2 primitive steps each
        prim = [[0, 1], [0, 0], [1, 0]]
        traj = self._make_traj(3, option_ids=[2, 2, 2], primitive_actions=prim)
        callback._save_trajectories([traj], [1.0], [3])

        npz_path = list((Path(temp_log_dir) / 'eval_logs').glob('*.npz'))[0]
        data = np.load(npz_path, allow_pickle=True)

        assert 'episode_0/primitive_actions' in data
        loaded = data['episode_0/primitive_actions']
        assert len(loaded) == 3
        assert list(loaded[0]) == [0, 1]
        assert list(loaded[1]) == [0, 0]
        assert list(loaded[2]) == [1, 0]

    def test_primitive_actions_omitted_when_all_none(self, test_env, temp_log_dir):
        """primitive_actions key is absent from npz when all values are None."""
        vec_env = DummyVecEnv([lambda: test_env])
        model = PPO('MlpPolicy', vec_env, verbose=0)
        callback = DetailedEvalCallback(
            eval_env=vec_env, n_eval_episodes=1, eval_freq=10,
            log_path=temp_log_dir, save_patient_trajectories=True, verbose=0
        )
        callback.init_callback(model)

        traj = self._make_traj(3)  # primitive_actions all None
        callback._save_trajectories([traj], [0.0], [3])

        npz_path = list((Path(temp_log_dir) / 'eval_logs').glob('*.npz'))[0]
        data = np.load(npz_path, allow_pickle=True)

        assert 'episode_0/primitive_actions' not in data

    def test_clipping_values_round_trip(self, test_env, temp_log_dir):
        """Non-default clipping values are preserved exactly through save/load."""
        vec_env = DummyVecEnv([lambda: test_env])
        model = PPO('MlpPolicy', vec_env, verbose=0)
        callback = DetailedEvalCallback(
            eval_env=vec_env, n_eval_episodes=1, eval_freq=10,
            log_path=temp_log_dir, save_patient_trajectories=True, verbose=0
        )
        callback.init_callback(model)

        traj = self._make_traj(4)
        traj['manager_clipped'] = [False, True, False, True]
        traj['steps_clipped'] = [0, 3, 0, 7]
        traj['manager_transition_trainable'] = [True, False, True, False]

        callback._save_trajectories([traj], [2.0], [4])

        npz_path = list((Path(temp_log_dir) / 'eval_logs').glob('*.npz'))[0]
        data = np.load(npz_path, allow_pickle=True)

        np.testing.assert_array_equal(data['episode_0/manager_clipped'], [False, True, False, True])
        np.testing.assert_array_equal(data['episode_0/steps_clipped'], [0, 3, 0, 7])
        np.testing.assert_array_equal(data['episode_0/manager_transition_trainable'], [True, False, True, False])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
