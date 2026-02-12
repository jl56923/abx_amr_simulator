"""
Comprehensive tests for config utility functions.

Tests for:
- load_config() - YAML config loading with legacy/nested format support
- apply_subconfig_overrides() - subconfig swapping from command line
- apply_param_overrides() - dot-notation parameter overrides
- setup_config_folders_with_defaults() - default config folder creation
"""

import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import pytest

from abx_amr_simulator.utils import (
    load_config,
    apply_subconfig_overrides,
    apply_param_overrides,
    setup_config_folders_with_defaults,
)


class TestLoadConfig:
    """Tests for load_config() function."""
    
    def test_load_nested_config_legacy(self):
        """Test loading a legacy nested YAML config with relative path references."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            configs_dir = tmpdir / "configs"
            configs_dir.mkdir()
            
            # Create subconfig files
            env_config = {"max_time_steps": 1000, "num_patients": 10}
            (configs_dir / "environment.yaml").write_text(yaml.dump(data=env_config))
            
            reward_config = {"lambda_weight": 0.5}
            (configs_dir / "reward_calculator.yaml").write_text(yaml.dump(data=reward_config))
            
            # Create umbrella config with legacy relative path format
            umbrella_config = {
                "environment": "environment.yaml",
                "reward_calculator": "reward_calculator.yaml",
                "run_name": "test_exp",
            }
            config_path = configs_dir / "config.yaml"
            config_path.write_text(yaml.dump(data=umbrella_config))
            
            # Load it
            config = load_config(config_path=str(config_path))
            
            # Should load umbrella config and merge subconfigs
            assert config["run_name"] == "test_exp"
            assert config["environment"]["max_time_steps"] == 1000
            assert config["reward_calculator"]["lambda_weight"] == 0.5
    
    def test_load_nested_config_modern(self):
        """Test loading a modern nested YAML config with explicit config_folder_location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            configs_dir = tmpdir / "configs"
            umbrella_dir = configs_dir / "umbrella_configs"
            component_dir = configs_dir / "components"
            
            umbrella_dir.mkdir(parents=True)
            component_dir.mkdir(parents=True)
            
            # Create subconfig files
            env_config = {"max_time_steps": 1000, "num_patients": 10}
            (component_dir / "environment.yaml").write_text(yaml.dump(data=env_config))
            
            reward_config = {"lambda_weight": 0.5}
            (component_dir / "reward_calculator.yaml").write_text(yaml.dump(data=reward_config))
            
            # Create umbrella config with modern format
            umbrella_config = {
                "config_folder_location": "../components",  # Relative to umbrella config
                "environment": "environment.yaml",
                "reward_calculator": "reward_calculator.yaml",
                "run_name": "test_exp_modern",
            }
            config_path = umbrella_dir / "config.yaml"
            config_path.write_text(yaml.dump(data=umbrella_config))
            
            # Load it
            config = load_config(config_path=str(config_path))
            
            # Should load umbrella config and merge subconfigs
            assert config["run_name"] == "test_exp_modern"
            assert config["environment"]["max_time_steps"] == 1000
            assert config["reward_calculator"]["lambda_weight"] == 0.5
            assert config["config_folder_location"] == "../components"

    
    def test_load_flat_config(self):
        """Test loading a flat YAML config without subconfig references."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config_data = {
                "environment": {"max_time_steps": 1000},
                "reward_calculator": {"lambda_weight": 0.5},
                "run_name": "flat_test",
            }
            config_path = tmpdir / "config.yaml"
            config_path.write_text(yaml.dump(data=config_data))
            
            config = load_config(config_path=str(config_path))
            
            assert config["environment"]["max_time_steps"] == 1000
            assert config["reward_calculator"]["lambda_weight"] == 0.5
            assert config["run_name"] == "flat_test"
    
    def test_load_config_file_not_found(self):
        """Test load_config raises error for missing file."""
        with pytest.raises(expected_exception=FileNotFoundError):
            load_config(config_path="/nonexistent/path/config.yaml")
    
    def test_load_config_invalid_yaml(self):
        """Test load_config raises error for invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config_path = tmpdir / "bad.yaml"
            config_path.write_text("{ invalid: yaml: syntax:")
            
            with pytest.raises(expected_exception=yaml.YAMLError):
                load_config(config_path=str(config_path))


class TestApplySubconfigOverrides:
    """Tests for apply_subconfig_overrides() function."""
    
    def test_override_environment_subconfig(self):
        """Test overriding environment subconfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            configs_dir = tmpdir / "configs"
            configs_dir.mkdir()
            
            # Create original and override environment configs
            orig_env = {"max_time_steps": 1000, "num_patients": 10}
            override_env = {"max_time_steps": 500, "num_patients": 20}
            
            (configs_dir / "environment_orig.yaml").write_text(yaml.dump(data=orig_env))
            (configs_dir / "environment_override.yaml").write_text(yaml.dump(data=override_env))
            
            orig_config = {
                "environment": orig_env,
                "run_name": "test",
            }
            
            overrides = {
                "environment_subconfig": "environment_override.yaml",
            }
            
            result = apply_subconfig_overrides(
                configs_dir=str(configs_dir),
                orig_config=orig_config,
                overrides=overrides
            )
            
            assert result["environment"]["max_time_steps"] == 500
            assert result["environment"]["num_patients"] == 20
            assert result["run_name"] == "test"  # Other keys preserved
    
    def test_override_reward_subconfig(self):
        """Test overriding reward_calculator subconfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            configs_dir = tmpdir / "configs"
            configs_dir.mkdir()
            
            reward_1 = {"lambda_weight": 0.5}
            reward_2 = {"lambda_weight": 0.8}
            
            (configs_dir / "reward_1.yaml").write_text(yaml.dump(data=reward_1))
            (configs_dir / "reward_2.yaml").write_text(yaml.dump(data=reward_2))
            
            orig_config = {
                "reward_calculator": reward_1,
                "run_name": "test",
            }
            
            overrides = {
                "reward_calculator_subconfig": "reward_2.yaml",
            }
            
            result = apply_subconfig_overrides(
                configs_dir=str(configs_dir),
                orig_config=orig_config,
                overrides=overrides
            )
            
            assert result["reward_calculator"]["lambda_weight"] == 0.8
    
    def test_multiple_subconfig_overrides(self):
        """Test overriding multiple subconfigs at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            configs_dir = tmpdir / "configs"
            configs_dir.mkdir()
            
            env_cfg = {"max_time_steps": 500}
            reward_cfg = {"lambda_weight": 0.7}
            
            (configs_dir / "env_new.yaml").write_text(yaml.dump(data=env_cfg))
            (configs_dir / "reward_new.yaml").write_text(yaml.dump(data=reward_cfg))
            
            orig_config = {
                "environment": {"max_time_steps": 1000},
                "reward_calculator": {"lambda_weight": 0.5},
            }
            
            overrides = {
                "environment_subconfig": "env_new.yaml",
                "reward_calculator_subconfig": "reward_new.yaml",
            }
            
            result = apply_subconfig_overrides(
                configs_dir=str(configs_dir),
                orig_config=orig_config,
                overrides=overrides
            )
            
            assert result["environment"]["max_time_steps"] == 500
            assert result["reward_calculator"]["lambda_weight"] == 0.7
    
    def test_no_subconfig_overrides(self):
        """Test that non-subconfig overrides are ignored."""
        configs_dir = "/tmp"
        orig_config = {
            "environment": {"max_time_steps": 1000},
            "run_name": "test",
        }
        
        overrides = {
            "training.seed": 42,  # Not a subconfig override
            "reward_calculator.lambda": 0.5,  # Not a subconfig override
        }
        
        result = apply_subconfig_overrides(
            configs_dir=configs_dir,
            orig_config=orig_config,
            overrides=overrides
        )
        
        # Config should be unchanged since no subconfig overrides
        assert result == orig_config
    
    def test_subconfig_file_not_found(self):
        """Test error when subconfig file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            configs_dir = tmpdir / "configs"
            configs_dir.mkdir()
            
            orig_config = {"environment": {"max_time_steps": 1000}}
            
            overrides = {
                "environment_subconfig": "nonexistent.yaml",
            }
            
            with pytest.raises(expected_exception=FileNotFoundError):
                apply_subconfig_overrides(
                    configs_dir=str(configs_dir),
                    orig_config=orig_config,
                    overrides=overrides
                )


class TestApplyParamOverrides:
    """Tests for apply_param_overrides() function."""
    
    def test_override_single_nested_param(self):
        """Test overriding a single nested parameter."""
        config = {
            "environment": {
                "max_time_steps": 1000,
                "num_patients": 10,
            },
            "training": {
                "seed": 42,
            },
        }
        
        overrides = {
            "environment.max_time_steps": 500,
        }
        
        result = apply_param_overrides(config=config, overrides=overrides)
        
        assert result["environment"]["max_time_steps"] == 500
        assert result["environment"]["num_patients"] == 10
        assert result["training"]["seed"] == 42
    
    def test_override_multiple_params(self):
        """Test overriding multiple parameters."""
        config = {
            "environment": {
                "max_time_steps": 1000,
                "num_patients": 10,
            },
            "training": {
                "seed": 42,
                "total_episodes": 100,
            },
        }
        
        overrides = {
            "environment.max_time_steps": 500,
            "environment.num_patients": 20,
            "training.seed": 123,
        }
        
        result = apply_param_overrides(config=config, overrides=overrides)
        
        assert result["environment"]["max_time_steps"] == 500
        assert result["environment"]["num_patients"] == 20
        assert result["training"]["seed"] == 123
        assert result["training"]["total_episodes"] == 100
    
    def test_override_top_level_param(self):
        """Test overriding a top-level parameter."""
        config = {
            "run_name": "original",
            "algorithm": "PPO",
        }
        
        overrides = {
            "run_name": "modified",
        }
        
        result = apply_param_overrides(config=config, overrides=overrides)
        
        assert result["run_name"] == "modified"
        assert result["algorithm"] == "PPO"
    
    def test_override_creates_nested_keys(self):
        """Test that overriding creates nested keys if they don't exist."""
        config = {"environment": {"max_time_steps": 1000}}
        
        overrides = {
            "environment.new_key": 42,
        }
        
        result = apply_param_overrides(config=config, overrides=overrides)
        
        assert result["environment"]["new_key"] == 42
        assert result["environment"]["max_time_steps"] == 1000
    
    def test_override_numeric_and_boolean_values(self):
        """Test overriding numeric and boolean values."""
        config = {
            "environment": {
                "max_time_steps": 1000,
                "flag": False,
            },
        }
        
        overrides = {
            "environment.max_time_steps": 500,
            "environment.flag": True,
        }
        
        result = apply_param_overrides(config=config, overrides=overrides)
        
        assert result["environment"]["max_time_steps"] == 500
        assert result["environment"]["flag"] is True
    
    def test_override_filters_subconfig_keys(self):
        """Test that override ignores keys containing 'subconfig'."""
        config = {
            "environment": {"max_time_steps": 1000},
        }
        
        overrides = {
            "environment.max_time_steps": 500,
            "environment_subconfig": "override.yaml",  # Should be ignored
        }
        
        result = apply_param_overrides(config=config, overrides=overrides)
        
        assert result["environment"]["max_time_steps"] == 500
        # subconfig key should not be added
        assert "environment_subconfig" not in result["environment"]


class TestSetupConfigFoldersWithDefaults:
    """Tests for setup_config_folders_with_defaults() function."""
    
    def test_creates_config_folder_structure(self):
        """Test that default config folders are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            setup_config_folders_with_defaults(str(tmpdir))
            
            # Check that all expected subdirectories were created
            assert (tmpdir / "configs" / "umbrella_configs").exists()
            assert (tmpdir / "configs" / "environment").exists()
            assert (tmpdir / "configs" / "reward_calculator").exists()
            assert (tmpdir / "configs" / "patient_generator").exists()
            assert (tmpdir / "configs" / "agent_algorithm").exists()
    
    def test_creates_default_config_files(self):
        """Test that default config files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            setup_config_folders_with_defaults(str(tmpdir))
            
            # Check that default configs were created
            assert (tmpdir / "configs" / "environment" / "default.yaml").exists()
            assert (tmpdir / "configs" / "reward_calculator" / "default.yaml").exists()
            assert (tmpdir / "configs" / "patient_generator" / "default.yaml").exists()
            assert (tmpdir / "configs" / "agent_algorithm" / "default.yaml").exists()
            assert (tmpdir / "configs" / "agent_algorithm" / "hrl_rppo.yaml").exists()
            assert (tmpdir / "configs" / "umbrella_configs" / "base_experiment.yaml").exists()
    
    def test_default_configs_are_valid_yaml(self):
        """Test that created config files contain valid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            setup_config_folders_with_defaults(str(tmpdir))
            
            config_files = [
                tmpdir / "configs" / "environment" / "default.yaml",
                tmpdir / "configs" / "reward_calculator" / "default.yaml",
                tmpdir / "configs" / "patient_generator" / "default.yaml",
                tmpdir / "configs" / "agent_algorithm" / "default.yaml",
                tmpdir / "configs" / "agent_algorithm" / "hrl_rppo.yaml",
                tmpdir / "configs" / "umbrella_configs" / "base_experiment.yaml",
            ]
            
            for config_file in config_files:
                # Should not raise
                with open(file=config_file, mode='r') as f:
                    cfg = yaml.safe_load(stream=f)
                    assert cfg is not None  # Config should not be empty
    
    def test_umbrella_config_references_subconfigs(self):
        """Test that umbrella config properly references subconfigs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            setup_config_folders_with_defaults(str(tmpdir))
            
            umbrella_path = tmpdir / "configs" / "umbrella_configs" / "base_experiment.yaml"
            with open(file=umbrella_path, mode='r') as f:
                umbrella = yaml.safe_load(stream=f)
            
            # Should have subconfig references
            assert "environment_subconfig" in umbrella or "environment" in umbrella
    
    def test_idempotent_operation(self):
        """Test that running setup twice doesn't cause issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Run twice
            setup_config_folders_with_defaults(str(tmpdir))
            setup_config_folders_with_defaults(target_path=str(tmpdir))  # Should not raise
            
            # Verify files still exist
            assert (tmpdir / "configs" / "umbrella_configs" / "base_experiment.yaml").exists()
    
    def test_copies_default_mixer_yaml(self):
        """Test that default_mixer.yaml is copied into patient_generator folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            setup_config_folders_with_defaults(str(tmpdir))
            
            # Check that default_mixer.yaml was copied
            mixer_file = tmpdir / "configs" / "patient_generator" / "default_mixer.yaml"
            assert mixer_file.exists(), f"Expected default_mixer.yaml at {mixer_file}"
            
            # Verify it contains valid YAML
            with open(file=mixer_file, mode='r') as f:
                mixer_config = yaml.safe_load(stream=f)
                assert mixer_config is not None
    
    def test_recursively_copies_all_default_yaml_files(self):
        """Test that all YAML files with 'default' in filename are recursively copied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            setup_config_folders_with_defaults(str(tmpdir))
            
            configs_dir = tmpdir / "configs"
            
            # Walk through all created config files and verify 'default' files are present
            # At minimum: default.yaml in each component folder
            component_folders = ["environment", "reward_calculator", "patient_generator", "agent_algorithm"]
            
            for component in component_folders:
                default_file = configs_dir / component / "default.yaml"
                assert default_file.exists(), f"Expected default.yaml in {component}"
            
            # Check patient_generator has both default.yaml and default_mixer.yaml
            pg_dir = configs_dir / "patient_generator"
            assert (pg_dir / "default.yaml").exists()
            assert (pg_dir / "default_mixer.yaml").exists()


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v"])
