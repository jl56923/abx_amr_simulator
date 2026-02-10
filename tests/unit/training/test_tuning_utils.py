"""Unit tests for hyperparameter tuning utilities."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
import optuna

from abx_amr_simulator.training.tune import suggest_hyperparameters
from abx_amr_simulator.training.train import load_best_params_from_optimization
from abx_amr_simulator.utils.registry import load_registry, update_registry


class TestSuggestHyperparameters:
    """Tests for suggest_hyperparameters function."""
    
    def test_suggest_float_parameter(self):
        """Test suggesting a float parameter."""
        search_space = {
            'learning_rate': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-3,
                'log': True
            }
        }
        
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.0001
        
        result = suggest_hyperparameters(trial=trial, search_space=search_space)
        
        assert 'learning_rate' in result
        assert result['learning_rate'] == 0.0001
        trial.suggest_float.assert_called_once_with(
            name='learning_rate',
            low=1e-5,
            high=1e-3,
            log=True
        )
    
    def test_suggest_float_parameter_without_log(self):
        """Test suggesting a float parameter without log scale."""
        search_space = {
            'gamma': {
                'type': 'float',
                'low': 0.9,
                'high': 0.999
            }
        }
        
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.95
        
        result = suggest_hyperparameters(trial=trial, search_space=search_space)
        
        assert 'gamma' in result
        assert result['gamma'] == 0.95
        trial.suggest_float.assert_called_once_with(
            name='gamma',
            low=0.9,
            high=0.999,
            log=False
        )
    
    def test_suggest_int_parameter(self):
        """Test suggesting an integer parameter."""
        search_space = {
            'n_steps': {
                'type': 'int',
                'low': 128,
                'high': 2048,
                'step': 128
            }
        }
        
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 512
        
        result = suggest_hyperparameters(trial=trial, search_space=search_space)
        
        assert 'n_steps' in result
        assert result['n_steps'] == 512
        trial.suggest_int.assert_called_once_with(
            name='n_steps',
            low=128,
            high=2048,
            step=128
        )
    
    def test_suggest_int_parameter_default_step(self):
        """Test suggesting an integer parameter with default step=1."""
        search_space = {
            'n_epochs': {
                'type': 'int',
                'low': 3,
                'high': 10
            }
        }
        
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 5
        
        result = suggest_hyperparameters(trial=trial, search_space=search_space)
        
        assert 'n_epochs' in result
        assert result['n_epochs'] == 5
        trial.suggest_int.assert_called_once_with(
            name='n_epochs',
            low=3,
            high=10,
            step=1
        )
    
    def test_suggest_categorical_parameter(self):
        """Test suggesting a categorical parameter."""
        search_space = {
            'ent_coef': {
                'type': 'categorical',
                'choices': [0.0, 0.01, 0.1]
            }
        }
        
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_categorical.return_value = 0.01
        
        result = suggest_hyperparameters(trial=trial, search_space=search_space)
        
        assert 'ent_coef' in result
        assert result['ent_coef'] == 0.01
        trial.suggest_categorical.assert_called_once_with(
            name='ent_coef',
            choices=[0.0, 0.01, 0.1]
        )
    
    def test_suggest_multiple_parameters(self):
        """Test suggesting multiple parameters of different types."""
        search_space = {
            'learning_rate': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-3,
                'log': True
            },
            'n_steps': {
                'type': 'int',
                'low': 128,
                'high': 2048,
                'step': 128
            },
            'ent_coef': {
                'type': 'categorical',
                'choices': [0.0, 0.01, 0.1]
            }
        }
        
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.0001
        trial.suggest_int.return_value = 512
        trial.suggest_categorical.return_value = 0.01
        
        result = suggest_hyperparameters(trial=trial, search_space=search_space)
        
        assert len(result) == 3
        assert result['learning_rate'] == 0.0001
        assert result['n_steps'] == 512
        assert result['ent_coef'] == 0.01
    
    def test_unknown_parameter_type_raises_error(self):
        """Test that unknown parameter type raises ValueError."""
        search_space = {
            'unknown_param': {
                'type': 'invalid_type',
                'value': 42
            }
        }
        
        trial = MagicMock(spec=optuna.Trial)
        
        with pytest.raises(ValueError, match="Unknown parameter type 'invalid_type'"):
            suggest_hyperparameters(trial=trial, search_space=search_space)
    
    def test_empty_search_space(self):
        """Test that empty search space returns empty dict."""
        search_space = {}
        trial = MagicMock(spec=optuna.Trial)
        
        result = suggest_hyperparameters(trial=trial, search_space=search_space)
        
        assert result == {}


class TestLoadBestParamsFromOptimization:
    """Tests for load_best_params_from_optimization function."""
    
    def test_loads_best_params_from_most_recent_run(self):
        """Test loading best params from most recent optimization run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimization_dir = Path(tmpdir)
            
            # Create two optimization runs (different timestamps)
            run1_dir = optimization_dir / "exp1_tuning_20260101_120000"
            run2_dir = optimization_dir / "exp1_tuning_20260102_120000"  # More recent
            
            run1_dir.mkdir(parents=True)
            run2_dir.mkdir(parents=True)
            
            # Create best_params.json in both
            best_params_1 = {'learning_rate': 0.0001, 'gamma': 0.95}
            best_params_2 = {'learning_rate': 0.0003, 'gamma': 0.99}  # Expected winner
            
            with open(run1_dir / "best_params.json", 'w') as f:
                json.dump(best_params_1, f)
            
            with open(run2_dir / "best_params.json", 'w') as f:
                json.dump(best_params_2, f)
            
            # Load best params (should get most recent: run2)
            params, opt_dir = load_best_params_from_optimization(
                experiment_name="exp1_tuning",
                optimization_base_dir=str(optimization_dir)
            )
            
            assert params == best_params_2
            assert opt_dir == str(run2_dir)
    
    def test_returns_none_if_no_matching_runs(self):
        """Test that function returns None if no matching runs found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimization_dir = Path(tmpdir)
            optimization_dir.mkdir(parents=True, exist_ok=True)
            
            params, opt_dir = load_best_params_from_optimization(
                experiment_name="nonexistent_exp",
                optimization_base_dir=str(optimization_dir)
            )
            
            assert params is None
            assert opt_dir is None
    
    def test_returns_none_if_best_params_json_missing(self):
        """Test that function returns None if best_params.json missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimization_dir = Path(tmpdir)
            run_dir = optimization_dir / "exp1_tuning_20260101_120000"
            run_dir.mkdir(parents=True)
            
            # Don't create best_params.json
            
            params, opt_dir = load_best_params_from_optimization(
                experiment_name="exp1_tuning",
                optimization_base_dir=str(optimization_dir)
            )
            
            assert params is None
            assert opt_dir is None
    
    def test_handles_malformed_json(self):
        """Test that function handles malformed JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimization_dir = Path(tmpdir)
            run_dir = optimization_dir / "exp1_tuning_20260101_120000"
            run_dir.mkdir(parents=True)
            
            # Create invalid JSON
            with open(run_dir / "best_params.json", 'w') as f:
                f.write("{ invalid json }")
            
            params, opt_dir = load_best_params_from_optimization(
                experiment_name="exp1_tuning",
                optimization_base_dir=str(optimization_dir)
            )
            
            assert params is None
            assert opt_dir is None
    
    def test_sorts_by_timestamp_correctly(self):
        """Test that runs are sorted by timestamp (not alphabetically)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimization_dir = Path(tmpdir)
            
            # Create runs with different timestamps (not in chronological order)
            run1_dir = optimization_dir / "exp1_tuning_20260103_120000"  # Most recent
            run2_dir = optimization_dir / "exp1_tuning_20260101_120000"  # Oldest
            run3_dir = optimization_dir / "exp1_tuning_20260102_120000"  # Middle
            
            for run_dir in [run1_dir, run2_dir, run3_dir]:
                run_dir.mkdir(parents=True)
                best_params = {'learning_rate': float(run_dir.name.split('_')[-2])}
                with open(run_dir / "best_params.json", 'w') as f:
                    json.dump(best_params, f)
            
            params, opt_dir = load_best_params_from_optimization(
                experiment_name="exp1_tuning",
                optimization_base_dir=str(optimization_dir)
            )
            
            # Should load from most recent (run1)
            assert opt_dir == str(run1_dir)


class TestRegistryFunctions:
    """Tests for registry utility functions."""
    
    def test_update_and_load_registry(self):
        """Test updating and loading registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, ".test_registry.txt")
            
            # Update registry with entries
            update_registry(
                registry_path=registry_path,
                run_name="exp1",
                timestamp="20260101_120000"
            )
            update_registry(
                registry_path=registry_path,
                run_name="exp2",
                timestamp="20260102_130000"
            )
            
            # Load registry
            completed_runs = load_registry(registry_path)
            
            assert "exp1" in completed_runs
            assert "exp2" in completed_runs
            assert len(completed_runs) == 2
    
    def test_load_registry_returns_empty_set_if_not_exists(self):
        """Test that load_registry returns empty set if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, ".nonexistent_registry.txt")
            
            completed_runs = load_registry(registry_path)
            
            assert completed_runs == set()
    
    def test_registry_deduplication(self):
        """Test that registry handles duplicate entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, ".test_registry.txt")
            
            # Add same experiment twice
            update_registry(
                registry_path=registry_path,
                run_name="exp1",
                timestamp="20260101_120000"
            )
            update_registry(
                registry_path=registry_path,
                run_name="exp1",
                timestamp="20260101_120000"
            )
            
            # Load registry (should deduplicate)
            completed_runs = load_registry(registry_path)
            
            # Set should naturally deduplicate
            assert len(completed_runs) == 1
            assert "exp1" in completed_runs
