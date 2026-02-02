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

from abx_amr_simulator.utils import (
    create_run_directory,
    save_training_config,
    save_training_summary,
)


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
            
            # Should create config.yaml
            config_file = run_dir / "config.yaml"
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
            
            config_file = run_dir / "config.yaml"
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
            
            config_file = run_dir / "config.yaml"
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
            
            config_file = run_dir / "config.yaml"
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


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v"])
