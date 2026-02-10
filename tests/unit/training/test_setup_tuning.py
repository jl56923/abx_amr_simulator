"""Unit tests for tuning scaffolding setup utilities."""

import pytest
import tempfile
from pathlib import Path

from abx_amr_simulator.training import setup_optimization_folders_with_defaults


class TestSetupOptimizationFoldersWithDefaults:
    """Tests for setup_optimization_folders_with_defaults function."""
    
    def test_creates_tuning_configs_directory(self):
        """Test that the function creates tuning_configs/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = setup_optimization_folders_with_defaults(target_path=tmpdir)
            
            assert result_path.exists()
            assert result_path.is_dir()
            assert result_path.name == "tuning_configs"
    
    def test_returns_correct_path(self):
        """Test that the function returns the Path object for tuning_configs/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = setup_optimization_folders_with_defaults(target_path=tmpdir)
            
            expected_path = Path(tmpdir) / "tuning_configs"
            assert result_path == expected_path
    
    def test_creates_ppo_tuning_default_config(self):
        """Test that ppo_tuning_default.yaml is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = setup_optimization_folders_with_defaults(target_path=tmpdir)
            
            ppo_tuning_path = result_path / "ppo_tuning_default.yaml"
            assert ppo_tuning_path.exists()
            assert ppo_tuning_path.is_file()
    
    def test_creates_hrl_ppo_tuning_default_config(self):
        """Test that hrl_ppo_tuning_default.yaml is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = setup_optimization_folders_with_defaults(target_path=tmpdir)
            
            hrl_ppo_tuning_path = result_path / "hrl_ppo_tuning_default.yaml"
            assert hrl_ppo_tuning_path.exists()
            assert hrl_ppo_tuning_path.is_file()
    
    def test_ppo_tuning_config_has_required_structure(self):
        """Test that ppo_tuning_default.yaml has required YAML structure."""
        import yaml
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = setup_optimization_folders_with_defaults(target_path=tmpdir)
            
            ppo_tuning_path = result_path / "ppo_tuning_default.yaml"
            with open(ppo_tuning_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check top-level keys
            assert 'optimization' in config
            assert 'search_space' in config
            
            # Check optimization section
            assert 'n_trials' in config['optimization']
            assert 'n_seeds_per_trial' in config['optimization']
            assert 'truncated_episodes' in config['optimization']
            assert 'direction' in config['optimization']
            assert 'sampler' in config['optimization']
            
            # Check search space has at least some parameters
            assert len(config['search_space']) > 0
            
            # Check a few expected hyperparameters
            assert 'learning_rate' in config['search_space']
            assert 'n_steps' in config['search_space']
            assert 'gamma' in config['search_space']
    
    def test_hrl_ppo_tuning_config_has_required_structure(self):
        """Test that hrl_ppo_tuning_default.yaml has required YAML structure."""
        import yaml
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = setup_optimization_folders_with_defaults(target_path=tmpdir)
            
            hrl_ppo_tuning_path = result_path / "hrl_ppo_tuning_default.yaml"
            with open(hrl_ppo_tuning_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check top-level keys
            assert 'optimization' in config
            assert 'search_space' in config
            
            # Check optimization section
            assert 'n_trials' in config['optimization']
            assert 'n_seeds_per_trial' in config['optimization']
            assert 'truncated_episodes' in config['optimization']
            assert 'direction' in config['optimization']
            assert 'sampler' in config['optimization']
            
            # Check search space has at least some parameters
            assert len(config['search_space']) > 0
            
            # Check HRL-specific hyperparameters
            assert 'learning_rate' in config['search_space']
            assert 'option_gamma' in config['search_space']  # HRL-specific
    
    def test_accepts_string_path(self):
        """Test that the function accepts string path argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = setup_optimization_folders_with_defaults(target_path=tmpdir)
            
            assert result_path.exists()
            assert isinstance(result_path, Path)
    
    def test_accepts_path_object(self):
        """Test that the function accepts Path object argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            result_path = setup_optimization_folders_with_defaults(target_path=tmpdir_path)
            
            assert result_path.exists()
            assert isinstance(result_path, Path)
    
    def test_raises_error_if_target_path_is_none(self):
        """Test that ValueError is raised if target_path is None."""
        with pytest.raises(ValueError, match="target_path must be provided"):
            setup_optimization_folders_with_defaults(target_path=None)
    
    def test_creates_nested_path_if_not_exists(self):
        """Test that function creates nested directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "experiments" / "workspace"
            result_path = setup_optimization_folders_with_defaults(target_path=nested_path)
            
            assert result_path.exists()
            assert result_path == nested_path / "tuning_configs"
    
    def test_idempotent_safe_to_run_multiple_times(self):
        """Test that running function multiple times doesn't break existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run twice
            result_path_1 = setup_optimization_folders_with_defaults(target_path=tmpdir)
            result_path_2 = setup_optimization_folders_with_defaults(target_path=tmpdir)
            
            # Both should succeed and return same path
            assert result_path_1 == result_path_2
            assert result_path_1.exists()
            
            # Files should still exist
            assert (result_path_2 / "ppo_tuning_default.yaml").exists()
            assert (result_path_2 / "hrl_ppo_tuning_default.yaml").exists()
