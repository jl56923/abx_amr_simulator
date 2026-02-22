"""Unit tests for config validation in tune.py."""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest
import yaml

# Import the functions we're testing
from abx_amr_simulator.training.tune import (
    _normalize_config,
    _remove_derived_fields,
    validate_configs_match_existing,
)


class TestNormalizeConfig:
    """Tests for _normalize_config function."""
    
    def test_normalize_empty_dict(self):
        """Test normalizing an empty dictionary."""
        config = {}
        result = _normalize_config(config)
        assert result == {}
    
    def test_normalize_dict_with_single_key(self):
        """Test normalizing a dict with a single key."""
        config = {'key': 'value'}
        result = _normalize_config(config)
        assert result == {'key': 'value'}
    
    def test_normalize_dict_with_alphabetical_sorting(self):
        """Test that dict keys are sorted alphabetically."""
        config = {'z': 1, 'a': 2, 'm': 3}
        result = _normalize_config(config)
        # Check that keys are sorted
        assert list(result.keys()) == ['a', 'm', 'z']
        assert result['a'] == 2
        assert result['m'] == 3
        assert result['z'] == 1
    
    def test_normalize_nested_dict(self):
        """Test normalizing nested dictionaries."""
        config = {
            'z_section': {'z_key': 1, 'a_key': 2},
            'a_section': {'b_key': 3}
        }
        result = _normalize_config(config)
        # Top level should be sorted
        assert list(result.keys()) == ['a_section', 'z_section']
        # Nested level should also be sorted
        assert list(result['z_section'].keys()) == ['a_key', 'z_key']
    
    def test_normalize_dict_with_list_values(self):
        """Test normalizing dict containing lists."""
        config = {
            'items': [
                {'z': 1, 'a': 2},
                {'b': 3, 'c': 4}
            ]
        }
        result = _normalize_config(config)
        # Lists themselves should be preserved, but nested dicts in them should be sorted
        assert len(result['items']) == 2
        assert list(result['items'][0].keys()) == ['a', 'z']
        assert list(result['items'][1].keys()) == ['b', 'c']
    
    def test_normalize_with_scalar_values(self):
        """Test normalizing dict with various scalar values."""
        config = {
            'string': 'value',
            'number': 42,
            'float': 3.14,
            'none': None,
            'bool': True
        }
        result = _normalize_config(config)
        # Scalars should be unchanged, just in sorted order
        assert result['bool'] is True
        assert result['float'] == 3.14
        assert result['none'] is None
        assert result['number'] == 42
        assert result['string'] == 'value'
    
    def test_normalize_deeply_nested_structure(self):
        """Test normalizing deeply nested structures."""
        config = {
            'level1': {
                'level2': {
                    'z_key': 1,
                    'a_key': 2,
                    'level3': {
                        'deep': 'value'
                    }
                }
            }
        }
        result = _normalize_config(config)
        # Verify nested structure is preserved and sorted
        assert 'level1' in result
        assert 'level2' in result['level1']
        assert 'level3' in result['level1']['level2']
        # Check sorting at level2
        assert list(result['level1']['level2'].keys()) == ['a_key', 'level3', 'z_key']


class TestRemoveDerivedFields:
    """Tests for _remove_derived_fields function."""
    
    def test_remove_option_library_path(self):
        """Test that option_library_path is removed."""
        config = {
            'algorithm': 'PPO',
            'option_library_path': '/some/absolute/path',
            'other_field': 'value'
        }
        result = _remove_derived_fields(config)
        assert 'option_library_path' not in result
        assert result['algorithm'] == 'PPO'
        assert result['other_field'] == 'value'
    
    def test_remove_umbrella_config_dir(self):
        """Test that _umbrella_config_dir is removed."""
        config = {
            'algorithm': 'PPO',
            '_umbrella_config_dir': '/Users/local/path/configs',
            'other_field': 'value'
        }
        result = _remove_derived_fields(config)
        assert '_umbrella_config_dir' not in result
        assert result['algorithm'] == 'PPO'
        assert result['other_field'] == 'value'
    
    def test_remove_multiple_derived_fields(self):
        """Test that all derived fields are removed."""
        config = {
            'algorithm': 'HRL_RPPO',
            '_umbrella_config_dir': '/home/remote/path/configs',
            'option_library_path': '/some/path/to/library.yaml',
            'normal_field': 'keep_me'
        }
        result = _remove_derived_fields(config)
        assert '_umbrella_config_dir' not in result
        assert 'option_library_path' not in result
        assert result['algorithm'] == 'HRL_RPPO'
        assert result['normal_field'] == 'keep_me'
    
    def test_remove_nested_derived_fields(self):
        """Test removal of derived fields in nested structures."""
        config = {
            'section1': {
                'option_library_path': '/path/to/library',
                'normal_field': 'keep_me'
            },
            'section2': {
                'another_field': 'value'
            }
        }
        result = _remove_derived_fields(config)
        assert 'option_library_path' not in result['section1']
        assert result['section1']['normal_field'] == 'keep_me'
        assert result['section2']['another_field'] == 'value'
    
    def test_keep_non_derived_fields(self):
        """Test that non-derived fields are preserved."""
        config = {
            'algorithm': 'HRL_RPPO',
            'learning_rate': 0.0003,
            'environment': {
                'max_time_steps': 500,
                'num_patients': 10
            }
        }
        result = _remove_derived_fields(config)
        # All fields should be present
        assert result['algorithm'] == 'HRL_RPPO'
        assert result['learning_rate'] == 0.0003
        assert result['environment']['max_time_steps'] == 500
        assert result['environment']['num_patients'] == 10
    
    def test_empty_dict(self):
        """Test removing fields from empty dict."""
        result = _remove_derived_fields({})
        assert result == {}


class TestConfigValidation:
    """Tests for validate_configs_match_existing function."""
    
    def test_validation_passes_for_identical_configs(self):
        """Test that identical configs pass validation."""
        config = {
            'algorithm': 'PPO',
            'learning_rate': 0.0003,
            'environment': {
                'max_steps': 500
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the configs
            with open(os.path.join(tmpdir, 'full_agent_env_config.yaml'), 'w') as f:
                yaml.dump(config, f)
            with open(os.path.join(tmpdir, 'tuning_config.yaml'), 'w') as f:
                yaml.dump({'n_trials': 100}, f)
            
            # Should not raise
            validate_configs_match_existing(
                current_umbrella_config=config,
                current_tuning_config={'n_trials': 100},
                current_option_library_config=None,
                optimization_dir=tmpdir,
                run_name='test_run'
            )
    
    def test_validation_passes_with_different_key_ordering(self):
        """Test that configs with different key ordering pass validation."""
        # Current config with one key order
        current_config = {
            'z_section': 'value1',
            'a_section': 'value2',
            'environment': {
                'z_param': 1,
                'a_param': 2
            }
        }
        
        # Saved config with different key order (simulates YAML ordering differences)
        saved_config = {
            'a_section': 'value2',
            'environment': {
                'a_param': 2,
                'z_param': 1
            },
            'z_section': 'value1'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the configs
            with open(os.path.join(tmpdir, 'full_agent_env_config.yaml'), 'w') as f:
                yaml.dump(saved_config, f)
            with open(os.path.join(tmpdir, 'tuning_config.yaml'), 'w') as f:
                yaml.dump({'n_trials': 100}, f)
            
            # Should not raise even though key ordering differs
            validate_configs_match_existing(
                current_umbrella_config=current_config,
                current_tuning_config={'n_trials': 100},
                current_option_library_config=None,
                optimization_dir=tmpdir,
                run_name='test_run'
            )
    
    def test_validation_passes_ignoring_option_library_path(self):
        """Test that option_library_path differences are ignored."""
        current_config = {
            'algorithm': 'PPO',
            'option_library_path': '/local/path/to/library.yaml'
        }
        
        saved_config = {
            'algorithm': 'PPO',
            # Note: no option_library_path in saved
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'full_agent_env_config.yaml'), 'w') as f:
                yaml.dump(saved_config, f)
            with open(os.path.join(tmpdir, 'tuning_config.yaml'), 'w') as f:
                yaml.dump({'n_trials': 100}, f)
            
            # Should pass because option_library_path is ignored
            validate_configs_match_existing(
                current_umbrella_config=current_config,
                current_tuning_config={'n_trials': 100},
                current_option_library_config=None,
                optimization_dir=tmpdir,
                run_name='test_run'
            )
    
    def test_validation_passes_ignoring_umbrella_config_dir(self):
        """Test that _umbrella_config_dir differences are ignored (local vs remote paths)."""
        # Simulates local macOS path
        current_config = {
            'algorithm': 'RecurrentPPO',
            '_umbrella_config_dir': '/Users/joycelee/Work/Code/project/configs',
            'learning_rate': 0.0003
        }
        
        # Simulates remote Linux path
        saved_config = {
            'algorithm': 'RecurrentPPO',
            '_umbrella_config_dir': '/home/jl56923/Work/project/configs',
            'learning_rate': 0.0003
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'full_agent_env_config.yaml'), 'w') as f:
                yaml.dump(saved_config, f)
            with open(os.path.join(tmpdir, 'tuning_config.yaml'), 'w') as f:
                yaml.dump({'n_trials': 100}, f)
            
            # Should pass despite different absolute paths (_umbrella_config_dir is ignored)
            validate_configs_match_existing(
                current_umbrella_config=current_config,
                current_tuning_config={'n_trials': 100},
                current_option_library_config=None,
                optimization_dir=tmpdir,
                run_name='test_run'
            )
    
    def test_validation_passes_ignoring_all_derived_fields(self):
        """Test that all derived fields are ignored during validation."""
        current_config = {
            'algorithm': 'HRL_RPPO',
            '_umbrella_config_dir': '/Users/local/configs',
            'option_library_path': '/local/path/library.yaml',
            'learning_rate': 0.0001
        }
        
        saved_config = {
            'algorithm': 'HRL_RPPO',
            '_umbrella_config_dir': '/home/remote/configs',
            'option_library_path': '/remote/path/library.yaml',
            'learning_rate': 0.0001
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'full_agent_env_config.yaml'), 'w') as f:
                yaml.dump(saved_config, f)
            with open(os.path.join(tmpdir, 'tuning_config.yaml'), 'w') as f:
                yaml.dump({'n_trials': 100}, f)
            
            # Should pass because all derived fields are ignored
            validate_configs_match_existing(
                current_umbrella_config=current_config,
                current_tuning_config={'n_trials': 100},
                current_option_library_config=None,
                optimization_dir=tmpdir,
                run_name='test_run'
            )
    
    def test_validation_fails_for_different_configs(self):
        """Test that validation fails when configs actually differ."""
        current_config = {
            'algorithm': 'PPO',
            'learning_rate': 0.0003
        }
        
        saved_config = {
            'algorithm': 'PPO',
            'learning_rate': 0.0001  # Different!
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'full_agent_env_config.yaml'), 'w') as f:
                yaml.dump(saved_config, f)
            with open(os.path.join(tmpdir, 'tuning_config.yaml'), 'w') as f:
                yaml.dump({'n_trials': 100}, f)
            
            # Should raise SystemExit
            with pytest.raises(SystemExit):
                validate_configs_match_existing(
                    current_umbrella_config=current_config,
                    current_tuning_config={'n_trials': 100},
                    current_option_library_config=None,
                    optimization_dir=tmpdir,
                    run_name='test_run'
                )
    
    def test_validation_fails_for_different_tuning_configs(self):
        """Test that validation fails when tuning configs differ."""
        umbrella_config = {'algorithm': 'PPO'}
        
        current_tuning = {'n_trials': 100}
        saved_tuning = {'n_trials': 50}  # Different!
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'full_agent_env_config.yaml'), 'w') as f:
                yaml.dump(umbrella_config, f)
            with open(os.path.join(tmpdir, 'tuning_config.yaml'), 'w') as f:
                yaml.dump(saved_tuning, f)
            
            with pytest.raises(SystemExit):
                validate_configs_match_existing(
                    current_umbrella_config=umbrella_config,
                    current_tuning_config=current_tuning,
                    current_option_library_config=None,
                    optimization_dir=tmpdir,
                    run_name='test_run'
                )
    
    def test_validation_with_missing_saved_configs(self):
        """Test handling when saved configs don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create any saved configs
            
            # Should return without raising
            validate_configs_match_existing(
                current_umbrella_config={'algorithm': 'PPO'},
                current_tuning_config={'n_trials': 100},
                current_option_library_config=None,
                optimization_dir=tmpdir,
                run_name='test_run'
            )
    
    def test_validation_with_option_library_configs(self):
        """Test validation with option library configs."""
        umbrella_config = {'algorithm': 'HRL_RPPO'}
        tuning_config = {'n_trials': 100}
        option_library_config = {
            'options': [
                {'name': 'option_a', 'params': [1, 2]},
                {'name': 'option_b', 'params': [3, 4]}
            ]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'full_agent_env_config.yaml'), 'w') as f:
                yaml.dump(umbrella_config, f)
            with open(os.path.join(tmpdir, 'tuning_config.yaml'), 'w') as f:
                yaml.dump(tuning_config, f)
            with open(os.path.join(tmpdir, 'option_library_config.yaml'), 'w') as f:
                yaml.dump(option_library_config, f)
            
            # Should pass when all configs match
            validate_configs_match_existing(
                current_umbrella_config=umbrella_config,
                current_tuning_config=tuning_config,
                current_option_library_config=option_library_config,
                optimization_dir=tmpdir,
                run_name='test_run'
            )
    
    def test_validation_fails_when_option_library_differs(self):
        """Test validation fails when option library configs differ."""
        umbrella_config = {'algorithm': 'HRL_RPPO'}
        tuning_config = {'n_trials': 100}
        
        current_option_library = {
            'options': [
                {'name': 'option_a', 'params': [1, 2]}
            ]
        }
        
        saved_option_library = {
            'options': [
                {'name': 'option_b', 'params': [3, 4]}
            ]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'full_agent_env_config.yaml'), 'w') as f:
                yaml.dump(umbrella_config, f)
            with open(os.path.join(tmpdir, 'tuning_config.yaml'), 'w') as f:
                yaml.dump(tuning_config, f)
            with open(os.path.join(tmpdir, 'option_library_config.yaml'), 'w') as f:
                yaml.dump(saved_option_library, f)
            
            with pytest.raises(SystemExit):
                validate_configs_match_existing(
                    current_umbrella_config=umbrella_config,
                    current_tuning_config=tuning_config,
                    current_option_library_config=current_option_library,
                    optimization_dir=tmpdir,
                    run_name='test_run'
                )
