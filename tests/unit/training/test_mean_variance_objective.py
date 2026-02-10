"""
Unit tests for mean-variance penalty objective function in tune.py.

Tests verify:
- Objective computation: mean(rewards) - λ × std(rewards)
- User attribute logging (mean_reward, std_reward, stability_penalty)
- Behavior with different lambda values
- Handling of failed trials (all -inf)
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from abx_amr_simulator.training.tune import create_objective_function


class TestMeanVarianceObjective:
    """Test suite for mean-variance penalty objective function."""
    
    def test_objective_computes_mean_minus_lambda_std(self):
        """Verify objective = mean(rewards) - λ × std(rewards)."""
        # Mock tuning config
        tuning_config = {
            'optimization': {
                'n_seeds_per_trial': 3,
                'truncated_episodes': 10,
                'stability_penalty_weight': 0.2,
            },
            'search_space': {}
        }
        
        # Create objective function
        with tempfile.TemporaryDirectory() as tmpdir:
            objective = create_objective_function(
                umbrella_config_path='dummy.yaml',
                tuning_config=tuning_config,
                subconfig_overrides={},
                base_param_overrides={},
                results_dir=tmpdir,
                base_seed=42
            )
            
            # Mock trial and run_training_trial
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.set_user_attr = Mock()
            
            # Test rewards: [10, 20, 30]
            # mean = 20, std = 8.165 (approx), lambda*std = 0.2 * 8.165 = 1.633
            # expected objective = 20 - 1.633 = 18.367
            test_rewards = [10.0, 20.0, 30.0]
            expected_mean = 20.0
            expected_std = np.std(test_rewards)
            expected_objective = expected_mean - 0.2 * expected_std
            
            with patch('abx_amr_simulator.training.tune.run_training_trial', return_value=test_rewards):
                with patch('abx_amr_simulator.training.tune.suggest_hyperparameters', return_value={}):
                    result = objective(mock_trial)
            
            # Verify objective value
            assert np.isclose(result, expected_objective, atol=1e-6)
            
            # Verify user attributes were logged
            assert mock_trial.set_user_attr.call_count == 3
            calls = {call[0][0]: call[0][1] for call in mock_trial.set_user_attr.call_args_list}
            assert 'mean_reward' in calls
            assert 'std_reward' in calls
            assert 'stability_penalty' in calls
            assert np.isclose(calls['mean_reward'], expected_mean, atol=1e-6)
            assert np.isclose(calls['std_reward'], expected_std, atol=1e-6)
            assert np.isclose(calls['stability_penalty'], 0.2 * expected_std, atol=1e-6)
    
    def test_objective_with_zero_lambda(self):
        """Zero lambda should reduce to pure mean optimization."""
        tuning_config = {
            'optimization': {
                'n_seeds_per_trial': 3,
                'truncated_episodes': 10,
                'stability_penalty_weight': 0.0,  # Zero lambda
            },
            'search_space': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            objective = create_objective_function(
                umbrella_config_path='dummy.yaml',
                tuning_config=tuning_config,
                subconfig_overrides={},
                base_param_overrides={},
                results_dir=tmpdir,
                base_seed=42
            )
            
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.set_user_attr = Mock()
            
            test_rewards = [10.0, 20.0, 30.0]
            expected_mean = 20.0
            
            with patch('abx_amr_simulator.training.tune.run_training_trial', return_value=test_rewards):
                with patch('abx_amr_simulator.training.tune.suggest_hyperparameters', return_value={}):
                    result = objective(mock_trial)
            
            # With lambda=0, objective should equal mean
            assert np.isclose(result, expected_mean, atol=1e-6)
    
    def test_objective_with_high_lambda(self):
        """High lambda should heavily penalize variance."""
        tuning_config = {
            'optimization': {
                'n_seeds_per_trial': 3,
                'truncated_episodes': 10,
                'stability_penalty_weight': 0.5,  # High lambda
            },
            'search_space': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            objective = create_objective_function(
                umbrella_config_path='dummy.yaml',
                tuning_config=tuning_config,
                subconfig_overrides={},
                base_param_overrides={},
                results_dir=tmpdir,
                base_seed=42
            )
            
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.set_user_attr = Mock()
            
            # High variance rewards
            test_rewards = [0.0, 10.0, 100.0]
            expected_mean = np.mean(test_rewards)
            expected_std = np.std(test_rewards)
            expected_objective = expected_mean - 0.5 * expected_std
            
            with patch('abx_amr_simulator.training.tune.run_training_trial', return_value=test_rewards):
                with patch('abx_amr_simulator.training.tune.suggest_hyperparameters', return_value={}):
                    result = objective(mock_trial)
            
            # Objective should be significantly lower than mean due to high lambda
            assert np.isclose(result, expected_objective, atol=1e-6)
            assert result < expected_mean
            assert (expected_mean - result) > 0.4 * expected_std  # Significant penalty
    
    def test_objective_prefers_stable_over_high_variance(self):
        """Verify that lower variance is preferred when means are similar."""
        tuning_config = {
            'optimization': {
                'n_seeds_per_trial': 3,
                'truncated_episodes': 10,
                'stability_penalty_weight': 0.3,
            },
            'search_space': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            objective = create_objective_function(
                umbrella_config_path='dummy.yaml',
                tuning_config=tuning_config,
                subconfig_overrides={},
                base_param_overrides={},
                results_dir=tmpdir,
                base_seed=42
            )
            
            # Scenario 1: Low variance, slightly lower mean
            mock_trial_1 = Mock()
            mock_trial_1.number = 0
            mock_trial_1.set_user_attr = Mock()
            
            stable_rewards = [19.0, 20.0, 21.0]  # mean=20, std≈0.816
            
            with patch('abx_amr_simulator.training.tune.run_training_trial', return_value=stable_rewards):
                with patch('abx_amr_simulator.training.tune.suggest_hyperparameters', return_value={}):
                    stable_objective = objective(mock_trial_1)
            
            # Scenario 2: High variance, slightly higher mean
            mock_trial_2 = Mock()
            mock_trial_2.number = 1
            mock_trial_2.set_user_attr = Mock()
            
            unstable_rewards = [10.0, 20.0, 33.0]  # mean=21, std≈9.54
            
            with patch('abx_amr_simulator.training.tune.run_training_trial', return_value=unstable_rewards):
                with patch('abx_amr_simulator.training.tune.suggest_hyperparameters', return_value={}):
                    unstable_objective = objective(mock_trial_2)
            
            # Stable should win despite slightly lower mean
            # stable: 20 - 0.3*0.816 = 19.755
            # unstable: 21 - 0.3*9.54 = 18.138
            assert stable_objective > unstable_objective
    
    def test_objective_filters_failed_trials(self):
        """Verify that -inf rewards are filtered before computing objective."""
        tuning_config = {
            'optimization': {
                'n_seeds_per_trial': 5,
                'truncated_episodes': 10,
                'stability_penalty_weight': 0.2,
            },
            'search_space': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            objective = create_objective_function(
                umbrella_config_path='dummy.yaml',
                tuning_config=tuning_config,
                subconfig_overrides={},
                base_param_overrides={},
                results_dir=tmpdir,
                base_seed=42
            )
            
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.set_user_attr = Mock()
            
            # Mix of successful and failed trials
            mixed_rewards = [10.0, float('-inf'), 20.0, float('-inf'), 30.0]
            expected_valid = [10.0, 20.0, 30.0]
            expected_mean = np.mean(expected_valid)
            expected_std = np.std(expected_valid)
            expected_objective = expected_mean - 0.2 * expected_std
            
            with patch('abx_amr_simulator.training.tune.run_training_trial', return_value=mixed_rewards):
                with patch('abx_amr_simulator.training.tune.suggest_hyperparameters', return_value={}):
                    result = objective(mock_trial)
            
            # Should only use valid rewards
            assert np.isclose(result, expected_objective, atol=1e-6)
    
    def test_objective_all_failed_trials_returns_negative_inf(self):
        """When all seeds fail, objective should return -inf."""
        tuning_config = {
            'optimization': {
                'n_seeds_per_trial': 3,
                'truncated_episodes': 10,
                'stability_penalty_weight': 0.2,
            },
            'search_space': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            objective = create_objective_function(
                umbrella_config_path='dummy.yaml',
                tuning_config=tuning_config,
                subconfig_overrides={},
                base_param_overrides={},
                results_dir=tmpdir,
                base_seed=42
            )
            
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.set_user_attr = Mock()
            
            all_failed = [float('-inf'), float('-inf'), float('-inf')]
            
            with patch('abx_amr_simulator.training.tune.run_training_trial', return_value=all_failed):
                with patch('abx_amr_simulator.training.tune.suggest_hyperparameters', return_value={}):
                    result = objective(mock_trial)
            
            # Should return -inf
            assert result == float('-inf')
            
            # Should not set user attributes when all failed
            assert mock_trial.set_user_attr.call_count == 0
    
    def test_objective_defaults_lambda_to_0_1_if_not_specified(self):
        """If stability_penalty_weight not in config, should default to 0.1."""
        tuning_config = {
            'optimization': {
                'n_seeds_per_trial': 3,
                'truncated_episodes': 10,
                # stability_penalty_weight not specified
            },
            'search_space': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            objective = create_objective_function(
                umbrella_config_path='dummy.yaml',
                tuning_config=tuning_config,
                subconfig_overrides={},
                base_param_overrides={},
                results_dir=tmpdir,
                base_seed=42
            )
            
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.set_user_attr = Mock()
            
            test_rewards = [10.0, 20.0, 30.0]
            expected_mean = 20.0
            expected_std = np.std(test_rewards)
            expected_objective = expected_mean - 0.1 * expected_std  # Default lambda=0.1
            
            with patch('abx_amr_simulator.training.tune.run_training_trial', return_value=test_rewards):
                with patch('abx_amr_simulator.training.tune.suggest_hyperparameters', return_value={}):
                    result = objective(mock_trial)
            
            assert np.isclose(result, expected_objective, atol=1e-6)
