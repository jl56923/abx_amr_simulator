"""
Comprehensive tests for factory utility functions.

Tests for:
- create_run_directory() - timestamped run directory creation
- save_training_config() - saves config to YAML file
- save_training_summary() - saves JSON summary of training run
"""

import tempfile
import json
import yaml
from pathlib import Path
import re
from datetime import datetime
import pytest
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from abx_amr_simulator.utils import (
    create_run_directory,
    save_training_config,
    save_training_summary,
    setup_callbacks,
)
from abx_amr_simulator.callbacks import EpisodeFrequencyTriggerCallback, EpisodeProgressBarCallback


class TestCreateRunDirectory:
    """Tests for create_run_directory() function."""
    
    def test_creates_timestamped_directory(self):
        """Test that run directory is created with timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'run_name': 'test_exp',
                },
                'output_dir': 'results',
            }
            run_dir, timestamp = create_run_directory(config=config, project_root=tmpdir)
            
            run_path = Path(run_dir)
            assert run_path.exists()
            assert run_path.is_dir()
    
    def test_directory_name_format(self):
        """Test that directory name follows expected pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'run_name': 'test_exp',
                },
                'output_dir': 'results',
            }
            run_dir, timestamp = create_run_directory(config=config, project_root=tmpdir)
            
            dir_name = Path(run_dir).name
            
            # Should contain experiment name
            assert "test_exp" in dir_name
            
            # Should contain timestamp (YYYYMMDD_HHMMSS)
            timestamp_pattern = r'\d{8}_\d{6}'
            assert re.search(pattern=timestamp_pattern, string=dir_name)
    
    def test_directory_structure_created(self):
        """Test that subdirectories for logs, checkpoints, and eval_logs are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'run_name': 'test_exp',
                },
                'output_dir': 'results',
            }
            run_dir, timestamp = create_run_directory(config=config, project_root=tmpdir)
            
            run_path = Path(run_dir)
            
            # These subdirectories should be created
            assert (run_path / "logs").exists()
            assert (run_path / "checkpoints").exists()
            assert (run_path / "eval_logs").exists()
    
    def test_returns_string_path(self):
        """Test that function returns a tuple with string path and timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'run_name': 'test_exp',
                },
                'output_dir': 'results',
            }
            run_dir, timestamp = create_run_directory(config=config, project_root=tmpdir)
            
            assert isinstance(run_dir, str)
            assert len(run_dir) > 0
            assert isinstance(timestamp, str)
            assert len(timestamp) > 0
    
    def test_unique_directories_for_multiple_calls(self):
        """Test that multiple calls create different directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'run_name': 'test_exp',
                },
                'output_dir': 'results',
            }
            
            run_dir1, ts1 = create_run_directory(config=config, project_root=tmpdir)
            
            # Sleep for 1 second to ensure different timestamps (strftime uses seconds)
            import time
            time.sleep(1.1)
            
            run_dir2, ts2 = create_run_directory(config=config, project_root=tmpdir)
            
            assert run_dir1 != run_dir2
            assert ts1 != ts2
            assert Path(run_dir1).exists()
            assert Path(run_dir2).exists()


class TestSaveTrainingConfig:
    """Tests for save_training_config() function."""
    
    def test_saves_config_to_yaml(self):
        """Test that config is saved as YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "environment": {
                    "max_time_steps": 1000,
                    "num_patients": 10,
                },
                "reward_calculator": {
                    "lambda_weight": 0.5,
                },
                "training": {
                    "total_episodes": 100,
                    "seed": 42,
                },
            }
            
            run_dir = Path(tmpdir)
            save_training_config(config=config, run_dir=str(run_dir))
            
            # Should create full_agent_env_config.yaml
            config_file = run_dir / "full_agent_env_config.yaml"
            assert config_file.exists()
    
    def test_saved_config_is_valid_yaml(self):
        """Test that saved config can be loaded as valid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "environment": {"max_time_steps": 1000},
                "reward_calculator": {"lambda_weight": 0.5},
            }
            
            run_dir = Path(tmpdir)
            save_training_config(config=config, run_dir=str(run_dir))
            
            config_file = run_dir / "full_agent_env_config.yaml"
            with open(file=config_file, mode='r') as f:
                loaded_config = yaml.safe_load(stream=f)
            
            assert loaded_config == config
    
    def test_config_content_preserved(self):
        """Test that all config content is preserved when saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "run_name": "test_experiment",
                "environment": {
                    "max_time_steps": 1000,
                    "num_patients": 10,
                    "num_antibiotics": 3,
                },
                "reward_calculator": {
                    "lambda_weight": 0.5,
                    "clinical_benefit_reward": 1.0,
                },
                "training": {
                    "total_episodes": 100,
                    "seed": 42,
                    "learning_rate": 0.001,
                },
            }
            
            run_dir = Path(tmpdir)
            save_training_config(config=config, run_dir=str(run_dir))
            
            config_file = run_dir / "full_agent_env_config.yaml"
            with open(file=config_file, mode='r') as f:
                loaded_config = yaml.safe_load(stream=f)
            
            # All keys should be preserved
            assert loaded_config["run_name"] == "test_experiment"
            assert loaded_config["environment"]["max_time_steps"] == 1000
            assert loaded_config["reward_calculator"]["lambda_weight"] == 0.5
            assert loaded_config["training"]["seed"] == 42
    
    def test_handles_nested_structures(self):
        """Test that deeply nested structures are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "level1": {
                    "level2": {
                        "level3": {
                            "value": 42,
                        }
                    }
                }
            }
            
            run_dir = Path(tmpdir)
            save_training_config(config=config, run_dir=str(run_dir))
            
            config_file = run_dir / "full_agent_env_config.yaml"
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(stream=f)
            
            assert loaded_config["level1"]["level2"]["level3"]["value"] == 42


class TestSaveTrainingSummary:
    """Tests for save_training_summary() function."""
    
    def test_saves_summary_to_json(self):
        """Test that summary is saved as JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "run_name": "test_exp",
                "algorithm": "PPO",
            }
            
            run_dir = Path(tmpdir)
            save_training_summary(
                config=config,
                run_dir=str(run_dir),
                total_timesteps=10000,
                total_episodes=100
            )
            
            summary_file = run_dir / "summary.json"
            assert summary_file.exists()
    
    def test_saved_summary_is_valid_json(self):
        """Test that saved summary can be loaded as valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "run_name": "test_exp",
                "algorithm": "PPO",
            }
            
            run_dir = Path(tmpdir)
            save_training_summary(
                config=config,
                run_dir=str(run_dir),
                total_timesteps=10000,
                total_episodes=100
            )
            
            summary_file = run_dir / "summary.json"
            with open(summary_file, 'r') as f:
                loaded_summary = json.load(fp=f)
            
            assert loaded_summary is not None
            assert isinstance(loaded_summary, dict)
    
    def test_summary_contains_required_fields(self):
        """Test that summary contains all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "run_name": "test_experiment",
                "algorithm": "PPO",
            }
            
            run_dir = Path(tmpdir)
            save_training_summary(
                config=config,
                run_dir=str(run_dir),
                total_timesteps=10000,
                total_episodes=100
            )
            
            summary_file = run_dir / "summary.json"
            with open(summary_file, 'r') as f:
                loaded_summary = json.load(fp=f)
            
            # Should have required fields
            assert "run_name" in loaded_summary
            assert "algorithm" in loaded_summary
            assert "total_timesteps" in loaded_summary
            assert "total_episodes" in loaded_summary
            assert "timestamp" in loaded_summary
    
    def test_summary_preserves_config_values(self):
        """Test that config values are preserved in summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "run_name": "test_experiment",
                "algorithm": "A2C",
                "action_mode": "multidiscrete",
            }
            
            run_dir = Path(tmpdir)
            save_training_summary(
                config=config,
                run_dir=str(run_dir),
                total_timesteps=5000,
                total_episodes=50
            )
            
            summary_file = run_dir / "summary.json"
            with open(summary_file, 'r') as f:
                loaded_summary = json.load(fp=f)
            
            assert loaded_summary["run_name"] == "test_experiment"
            assert loaded_summary["algorithm"] == "A2C"
            assert loaded_summary["total_timesteps"] == 5000
            assert loaded_summary["total_episodes"] == 50

class TestSetupCallbacksPersonalizedLogging:
    """Tests for personalized callback decoupling in canonical setup_callbacks."""

    def test_defaults_toggle_off_when_not_provided(self):
        """Omitted toggle should default to OFF and callback setup should succeed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'log_patient_trajectories': True,
                },
                'patient_generator': {},
            }

            callbacks = setup_callbacks(
                config=config,
                run_dir=tmpdir,
                eval_env=None,
            )
            assert isinstance(callbacks, list)
            assert len(callbacks) >= 2

    def test_toggle_on_with_sentinel_fails_loudly(self):
        """Deprecated canonical personalized toggle should fail loudly even with sentinel config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'log_patient_trajectories': True,
                    'log_personalized_patient_attributes': True,
                },
                'patient_generator': {
                    'personalized_missing_prediction_fill_value': -1.0,
                },
            }

            with pytest.raises(
                expected_exception=ValueError,
                match='no longer supported in canonical abx_amr_simulator callbacks',
            ):
                setup_callbacks(
                    config=config,
                    run_dir=tmpdir,
                    eval_env=None,
                )


class TestSetupCallbacksCustomDetailedEvalCallback:
    """Tests for custom detailed eval callback override selection and kwargs wiring."""

    def test_custom_detailed_eval_callback_override_from_filesystem_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = Path(tmpdir) / 'custom_eval_callback.py'
            plugin_path.write_text(
                (
                    "from abx_amr_simulator.callbacks import DetailedEvalCallback\n"
                    "\n"
                    "class TestDetailedEvalCallback(DetailedEvalCallback):\n"
                    "    def __init__(self, *args, custom_note='unset', **kwargs):\n"
                    "        super().__init__(*args, **kwargs)\n"
                    "        self.custom_note = custom_note\n"
                ),
                encoding='utf-8',
            )

            config = {
                'training': {
                    'log_patient_trajectories': True,
                    'detailed_eval_callback_module': str(plugin_path),
                    'detailed_eval_callback_class': 'TestDetailedEvalCallback',
                    'detailed_eval_callback_kwargs': {
                        'custom_note': 'from_test',
                    },
                },
            }

            eval_env = gym.make(id='CartPole-v1')
            callbacks = setup_callbacks(
                config=config,
                run_dir=tmpdir,
                eval_env=eval_env,
            )
            eval_env.close()

            matched = [cb for cb in callbacks if cb.__class__.__name__ == 'TestDetailedEvalCallback']
            assert len(matched) == 1
            assert matched[0].custom_note == 'from_test'

    def test_custom_detailed_eval_callback_requires_module_and_class_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'log_patient_trajectories': True,
                    'detailed_eval_callback_module': 'some.module.path',
                },
            }

            eval_env = gym.make(id='CartPole-v1')
            with pytest.raises(
                expected_exception=ValueError,
                match='Both training.detailed_eval_callback_module and training.detailed_eval_callback_class',
            ):
                setup_callbacks(
                    config=config,
                    run_dir=tmpdir,
                    eval_env=eval_env,
                )
            eval_env.close()

    def test_toggle_on_without_sentinel_fails_loudly(self):
        """Deprecated canonical personalized toggle should fail loudly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'log_patient_trajectories': True,
                    'log_personalized_patient_attributes': True,
                },
                'patient_generator': {},
            }

            with pytest.raises(
                expected_exception=ValueError,
                match='no longer supported in canonical abx_amr_simulator callbacks',
            ):
                setup_callbacks(
                    config=config,
                    run_dir=tmpdir,
                    eval_env=None,
                )


class TestSetupCallbacksHRLEpisodeFrequency:
    """Tests for HRL episode-based eval/save callback scheduling."""

    def test_hrl_uses_episode_frequency_scheduler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'algorithm': 'HRL_PPO',
                'training': {
                    'log_patient_trajectories': False,
                    'eval_freq_every_n_episodes': 3,
                    'save_freq_every_n_episodes': 4,
                    '_converted_eval_freq': 99999,
                    '_converted_save_freq': 99999,
                },
            }

            eval_env = gym.make(id='CartPole-v1')
            callbacks = setup_callbacks(
                config=config,
                run_dir=tmpdir,
                eval_env=eval_env,
                stop_after_n_episodes=12,
            )
            eval_env.close()

            matched = [cb for cb in callbacks if isinstance(cb, EpisodeFrequencyTriggerCallback)]
            assert len(matched) == 1

            scheduler = matched[0]
            assert scheduler.eval_freq_episodes == 3
            assert scheduler.save_freq_episodes == 4
            assert isinstance(scheduler.eval_callback, EvalCallback)
            assert scheduler.eval_callback.eval_freq == 1
            assert isinstance(scheduler.checkpoint_callback, CheckpointCallback)
            assert scheduler.checkpoint_callback.save_freq == 1

            episode_progress = [cb for cb in callbacks if isinstance(cb, EpisodeProgressBarCallback)]
            assert len(episode_progress) == 1
            assert episode_progress[0].total_episodes == 12

    def test_non_hrl_keeps_timestep_callbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'algorithm': 'PPO',
                'training': {
                    'log_patient_trajectories': False,
                    '_converted_eval_freq': 123,
                    '_converted_save_freq': 456,
                    'eval_freq_every_n_episodes': 3,
                    'save_freq_every_n_episodes': 4,
                },
            }

            eval_env = gym.make(id='CartPole-v1')
            callbacks = setup_callbacks(
                config=config,
                run_dir=tmpdir,
                eval_env=eval_env,
                stop_after_n_episodes=12,
            )
            eval_env.close()

            scheduler = [cb for cb in callbacks if isinstance(cb, EpisodeFrequencyTriggerCallback)]
            assert len(scheduler) == 0

            eval_callbacks = [cb for cb in callbacks if isinstance(cb, EvalCallback)]
            checkpoint_callbacks = [cb for cb in callbacks if isinstance(cb, CheckpointCallback)]
            assert len(eval_callbacks) == 1
            assert len(checkpoint_callbacks) == 1
            assert eval_callbacks[0].eval_freq == 123
            assert checkpoint_callbacks[0].save_freq == 456

    def test_hrl_can_disable_episode_progress_bar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'algorithm': 'HRL_PPO',
                'training': {
                    'log_patient_trajectories': False,
                    'show_episode_progress_bar': False,
                    'eval_freq_every_n_episodes': 3,
                    'save_freq_every_n_episodes': 4,
                },
            }

            eval_env = gym.make(id='CartPole-v1')
            callbacks = setup_callbacks(
                config=config,
                run_dir=tmpdir,
                eval_env=eval_env,
                stop_after_n_episodes=8,
            )
            eval_env.close()

            episode_progress = [cb for cb in callbacks if isinstance(cb, EpisodeProgressBarCallback)]
            assert len(episode_progress) == 0


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v"])
