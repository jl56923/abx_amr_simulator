"""
Tests for patient attribute logging functionality in ABXAMREnv.

Verifies that:
1. _compute_patient_stats() returns correct structure and valid statistics
2. _extract_full_patient_attributes() returns correct structure
3. step() always includes patient_stats in info dict
4. step() conditionally includes patient_full_data based on log_full_patient_attributes flag
5. Flag defaults to False and can be toggled
"""

import numpy as np
import pytest

from unit.utils.test_reference_helpers import create_mock_environment
from abx_amr_simulator.core import ABXAMREnv


def test_log_full_patient_attributes_flag_defaults_to_false():
    """Test that log_full_patient_attributes flag defaults to False."""
    env = create_mock_environment()
    
    assert hasattr(env, 'log_full_patient_attributes')
    assert env.log_full_patient_attributes is False


def test_compute_patient_stats_returns_correct_structure():
    """Test that _compute_patient_stats returns the expected structure."""
    env = create_mock_environment()
    
    # Sample some patients
    true_amr_levels = {abx_name: env.amr_balloon_models[abx_name].get_volume() for abx_name in env.antibiotic_names}
    patients = env.patient_generator.sample(5, true_amr_levels=true_amr_levels, rng=env.np_random)
    
    # Compute stats
    stats = env._compute_patient_stats(patients)
    
    # Check that it's a dict
    assert isinstance(stats, dict)
    
    # Expected attributes
    expected_attrs = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob'
    ]
    
    # Check that all expected keys exist
    for attr in expected_attrs:
        assert f'{attr}_true_mean' in stats
        assert f'{attr}_true_std' in stats
        assert f'{attr}_true_min' in stats
        assert f'{attr}_true_max' in stats
        assert f'{attr}_obs_mean' in stats
        assert f'{attr}_obs_std' in stats
        assert f'{attr}_obs_min' in stats
        assert f'{attr}_obs_max' in stats
        assert f'{attr}_obs_error_mean' in stats
        assert f'{attr}_obs_error_max' in stats
    
    # Check that values are floats (not numpy types)
    for key, value in stats.items():
        assert isinstance(value, float), f"Key {key} has type {type(value)}, expected float"


def test_compute_patient_stats_values_are_reasonable():
    """Test that computed statistics have reasonable values."""
    env = create_mock_environment()
    
    # Sample patients
    true_amr_levels = {abx_name: env.amr_balloon_models[abx_name].get_volume() for abx_name in env.antibiotic_names}
    patients = env.patient_generator.sample(10, true_amr_levels=true_amr_levels, rng=env.np_random)
    
    # Compute stats
    stats = env._compute_patient_stats(patients)
    
    # Check prob_infected (should be around 0.8 since config uses constant 0.8)
    assert 0.0 <= stats['prob_infected_true_mean'] <= 1.0
    assert stats['prob_infected_true_min'] >= 0.0
    assert stats['prob_infected_true_max'] <= 1.0
    
    # Check that observation bias/noise creates difference
    # prob_infected has bias 1.1 and noise 0.05
    assert stats['prob_infected_obs_error_mean'] >= 0.0
    
    # Check benefit_value_multiplier (gaussian with mu=1.0, sigma=0.2)
    # Should have non-zero std
    assert stats['benefit_value_multiplier_true_std'] >= 0.0
    
    # Check that observation error exists for benefit_value_multiplier
    # (bias 0.9 and noise 0.1)
    assert stats['benefit_value_multiplier_obs_error_mean'] >= 0.0


def test_extract_full_patient_attributes_returns_correct_structure():
    """Test that _extract_full_patient_attributes returns expected structure."""
    env = create_mock_environment()
    
    # Sample patients
    true_amr_levels = {abx_name: env.amr_balloon_models[abx_name].get_volume() for abx_name in env.antibiotic_names}
    patients = env.patient_generator.sample(5, true_amr_levels=true_amr_levels, rng=env.np_random)
    
    # Extract full attributes
    full_data = env._extract_full_patient_attributes(patients)
    
    # Check top-level structure
    assert isinstance(full_data, dict)
    assert 'true' in full_data
    assert 'observed' in full_data
    
    # Expected attributes
    expected_attrs = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob'
    ]
    
    # Check 'true' sub-dict
    for attr in expected_attrs:
        assert attr in full_data['true']
        assert isinstance(full_data['true'][attr], list)
        assert len(full_data['true'][attr]) == 5
        # Check that values are floats
        for val in full_data['true'][attr]:
            assert isinstance(val, float)
    
    # Check 'observed' sub-dict
    for attr in expected_attrs:
        assert attr in full_data['observed']
        assert isinstance(full_data['observed'][attr], list)
        assert len(full_data['observed'][attr]) == 5
        # Check that values are floats
        for val in full_data['observed'][attr]:
            assert isinstance(val, float)


def test_step_always_includes_patient_stats():
    """Test that step() always includes patient_stats in info dict."""
    env = create_mock_environment()
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    # Take a step
    action = np.array([0, 1, 0, 1, 0])  # Some arbitrary action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check that patient_stats is in info
    assert 'patient_stats' in info
    assert isinstance(info['patient_stats'], dict)
    
    # Check that it has expected keys
    assert 'prob_infected_true_mean' in info['patient_stats']
    assert 'prob_infected_obs_mean' in info['patient_stats']


def test_step_excludes_patient_full_data_when_flag_false():
    """Test that step() excludes patient_full_data when flag is False."""
    env = create_mock_environment()
    
    # Ensure flag is False
    env.log_full_patient_attributes = False
    
    # Reset and step
    obs, info = env.reset(seed=42)
    action = np.array([0, 1, 0, 1, 0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    # patient_full_data should NOT be in info
    assert 'patient_full_data' not in info


def test_step_includes_patient_full_data_when_flag_true():
    """Test that step() includes patient_full_data when flag is True."""
    env = create_mock_environment()
    
    # Set flag to True
    env.log_full_patient_attributes = True
    
    # Reset and step
    obs, info = env.reset(seed=42)
    action = np.array([0, 1, 0, 1, 0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    # patient_full_data SHOULD be in info
    assert 'patient_full_data' in info
    assert isinstance(info['patient_full_data'], dict)
    assert 'true' in info['patient_full_data']
    assert 'observed' in info['patient_full_data']


def test_flag_can_be_toggled():
    """Test that the flag can be toggled on and off."""
    env = create_mock_environment()
    
    # Start with False
    assert env.log_full_patient_attributes is False
    
    # Toggle to True
    env.log_full_patient_attributes = True
    assert env.log_full_patient_attributes is True
    
    # Reset and step - should include full data
    obs, info = env.reset(seed=42)
    action = np.array([0, 1, 0, 1, 0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert 'patient_full_data' in info
    
    # Toggle to False
    env.log_full_patient_attributes = False
    assert env.log_full_patient_attributes is False
    
    # Step again - should NOT include full data
    action = np.array([1, 0, 1, 0, 1])
    obs, reward, terminated, truncated, info = env.step(action)
    assert 'patient_full_data' not in info
    assert 'patient_stats' in info  # But stats should still be there


def test_patient_stats_consistent_across_steps():
    """Test that patient_stats structure is consistent across multiple steps."""
    env = create_mock_environment()
    
    obs, info = env.reset(seed=42)
    
    # Take multiple steps and collect patient_stats keys
    keys_per_step = []
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        keys_per_step.append(set(info['patient_stats'].keys()))
    
    # All steps should have the same keys
    first_keys = keys_per_step[0]
    for keys in keys_per_step[1:]:
        assert keys == first_keys, "patient_stats keys should be consistent across steps"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
