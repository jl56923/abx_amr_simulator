"""Unit tests for temporal features in ABXAMREnv."""
import pytest
import numpy as np

from abx_amr_simulator.core import ABXAMREnv, RewardCalculator, PatientGenerator


def create_test_reward_calculator(antibiotic_names=None):
    """Helper to create a minimal RewardCalculator for testing."""
    if antibiotic_names is None:
        antibiotic_names = ["A", "B"]

    abx_clinical_reward_penalties_info_dict = {
        'clinical_benefit_reward': 10.0,
        'clinical_benefit_probability': 1.0,
        'clinical_failure_penalty': -1.0,
        'clinical_failure_probability': 0.0,
        'abx_adverse_effects_info': {
            name: {
                'adverse_effect_penalty': -2.0,
                'adverse_effect_probability': 0.0,
            }
            for name in antibiotic_names
        },
    }

    config = {
        'abx_clinical_reward_penalties_info_dict': abx_clinical_reward_penalties_info_dict,
        'lambda_weight': 0.5,
        'epsilon': 0.05,
    }
    return RewardCalculator(config=config)


def create_test_patient_generator(num_patients=5):
    """Helper to create a minimal PatientGenerator for testing."""
    config = PatientGenerator.default_config()
    # Override to only show prob_infected for simpler testing
    config['visible_patient_attributes'] = ['prob_infected']
    return PatientGenerator(config=config)


def create_test_env(enable_temporal_features=False, temporal_windows=None, num_patients=5):
    """Helper to create a test environment with optional temporal features."""
    antibiotic_names = ["A", "B"]
    rc = create_test_reward_calculator(antibiotic_names=antibiotic_names)
    pg = create_test_patient_generator(num_patients=num_patients)
    
    antibiotics_AMR_dict = {
        "A": {
            'leak': 0.2,
            'flatness_parameter': 50,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0,
        },
        "B": {
            'leak': 0.1,
            'flatness_parameter': 25,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0,
        },
    }
    
    if temporal_windows is None:
        temporal_windows = [10, 50]
    
    env = ABXAMREnv(
        reward_calculator=rc,
        patient_generator=pg,
        antibiotics_AMR_dict=antibiotics_AMR_dict,
        num_patients_per_time_step=num_patients,
        max_time_steps=100,
        enable_temporal_features=enable_temporal_features,
        temporal_windows=temporal_windows,
    )
    return env


class TestTemporalFeaturesObservationSpace:
    """Tests for observation space dimensionality with temporal features."""
    
    def test_observation_space_without_temporal_features(self):
        """Observation space should match baseline when temporal features disabled."""
        num_patients = 5
        env = create_test_env(enable_temporal_features=False, num_patients=num_patients)
        
        # Expected: (num_patients * num_visible_attrs) + num_abx
        # num_visible_attrs = 1 (prob_infected), num_abx = 2
        expected_dim = (num_patients * 1) + 2
        assert env.observation_space.shape[0] == expected_dim
        
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] == expected_dim
    
    def test_observation_space_with_temporal_features(self):
        """Observation space should expand correctly when temporal features enabled."""
        num_patients = 5
        temporal_windows = [10, 50]
        env = create_test_env(
            enable_temporal_features=True,
            temporal_windows=temporal_windows,
            num_patients=num_patients
        )
        
        # Expected: (num_patients * num_visible_attrs) + num_abx + temporal_dim
        # temporal_dim = (num_abx * num_windows) + num_abx
        # = (2 * 2) + 2 = 6
        base_dim = (num_patients * 1) + 2  # 7
        temporal_dim = (2 * 2) + 2  # 6
        expected_dim = base_dim + temporal_dim  # 13
        
        assert env.observation_space.shape[0] == expected_dim
        
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] == expected_dim
    
    def test_observation_space_with_custom_windows(self):
        """Observation space should adjust to custom window sizes."""
        num_patients = 3
        temporal_windows = [5, 20, 100]  # 3 windows
        env = create_test_env(
            enable_temporal_features=True,
            temporal_windows=temporal_windows,
            num_patients=num_patients
        )
        
        # temporal_dim = (num_abx * num_windows) + num_abx
        # = (2 * 3) + 2 = 8
        base_dim = (num_patients * 1) + 2  # 5
        temporal_dim = (2 * 3) + 2  # 8
        expected_dim = base_dim + temporal_dim  # 13
        
        assert env.observation_space.shape[0] == expected_dim
        
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] == expected_dim


class TestTemporalFeaturesTracking:
    """Tests for prescription history and AMR delta tracking."""
    
    def test_prescription_history_initialized_on_reset(self):
        """Prescription history should be initialized with zeros on reset."""
        env = create_test_env(enable_temporal_features=True, temporal_windows=[10, 50])
        env.reset(seed=42)
        
        # Check that history deques exist for both antibiotics
        assert "A" in env.prescription_history
        assert "B" in env.prescription_history
        
        # Check that each antibiotic has correct number of deques (one per window)
        assert len(env.prescription_history["A"]) == 2
        assert len(env.prescription_history["B"]) == 2
        
        # Check that deques are initialized with zeros
        for abx_name in ["A", "B"]:
            for deque_window in env.prescription_history[abx_name]:
                assert all(val == 0 for val in deque_window)
    
    def test_prescription_history_updates_correctly(self):
        """Prescription history should update with 1 when antibiotic prescribed."""
        env = create_test_env(enable_temporal_features=True, temporal_windows=[3, 5], num_patients=2)
        env.reset(seed=42)
        action_mapping = env.get_antibiotic_to_action_mapping()
        
        # Prescribe A for patient 0, B for patient 1 (actions: [1, 2])
        action = np.array([
            action_mapping["A"],
            action_mapping["B"],
        ])
        env.step(action)
        
        # Check that A has 1 in history, B has 1 in history
        assert sum(env.prescription_history["A"][0]) == 1  # Window size 3
        assert sum(env.prescription_history["A"][1]) == 1  # Window size 5
        assert sum(env.prescription_history["B"][0]) == 1
        assert sum(env.prescription_history["B"][1]) == 1
        
        # Prescribe neither (actions: [0, 0])
        action = np.array([
            action_mapping["no_treatment"],
            action_mapping["no_treatment"],
        ])
        env.step(action)
        
        # Counts should still be 1 (one prescription in history)
        assert sum(env.prescription_history["A"][0]) == 1
        assert sum(env.prescription_history["B"][0]) == 1
        
        # Prescribe A again for both patients (actions: [1, 1])
        action = np.array([
            action_mapping["A"],
            action_mapping["A"],
        ])
        env.step(action)
        
        # A should now have 2 prescriptions in history, B still 1
        assert sum(env.prescription_history["A"][0]) == 2
        assert sum(env.prescription_history["B"][0]) == 1
    
    def test_prescription_history_rolling_window(self):
        """Prescription history should maintain correct rolling window size."""
        window_size = 3
        env = create_test_env(
            enable_temporal_features=True,
            temporal_windows=[window_size],
            num_patients=1
        )
        env.reset(seed=42)
        action_mapping = env.get_antibiotic_to_action_mapping()
        
        # Prescribe A for 5 steps
        for _ in range(5):
            action = np.array([action_mapping["A"]])
            env.step(action)
        
        # Window size is 3, so only last 3 should count
        assert len(env.prescription_history["A"][0]) == window_size
        assert sum(env.prescription_history["A"][0]) == 3
    
    def test_amr_delta_calculation(self):
        """AMR deltas should be calculated correctly."""
        env = create_test_env(enable_temporal_features=True, temporal_windows=[10])
        obs, _ = env.reset(seed=42)
        action_mapping = env.get_antibiotic_to_action_mapping()
        
        # Initial AMR should be 0, so deltas should be 0
        initial_amr_A = env.visible_amr_levels["A"]
        initial_amr_B = env.visible_amr_levels["B"]
        assert initial_amr_A == 0.0
        assert initial_amr_B == 0.0
        
        # Take a step that prescribes A
        action = np.array([action_mapping["A"]] * env.num_patients_per_time_step)
        obs, _, _, _, _ = env.step(action)
        
        # AMR should have increased for A
        new_amr_A = env.visible_amr_levels["A"]
        assert new_amr_A > initial_amr_A
        
        # Previous AMR should be stored
        assert env.previous_visible_amr_levels["A"] == initial_amr_A


class TestTemporalFeaturesObservationContent:
    """Tests for temporal feature values in observations."""
    
    def test_temporal_features_normalized_prescription_counts(self):
        """Prescription counts should be normalized by window size."""
        window_sizes = [10, 20]
        env = create_test_env(
            enable_temporal_features=True,
            temporal_windows=window_sizes,
            num_patients=2
        )
        env.reset(seed=42)
        action_mapping = env.get_antibiotic_to_action_mapping()
        
        # Prescribe A 5 times
        for _ in range(5):
            action = np.array([
                action_mapping["A"],
                action_mapping["A"],
            ])
            obs, _, _, _, _ = env.step(action)
        
        # Extract temporal features from observation
        # Structure: [patient features, AMR levels, temporal features]
        # Patient features: 2 patients * 1 attr = 2
        # AMR levels: 2
        # Temporal features start at index 4
        temporal_start_idx = 2 + 2
        
        # Temporal features structure:
        # [prescriptions_A_win1, prescriptions_A_win2, prescriptions_B_win1, prescriptions_B_win2, delta_A, delta_B]
        prescriptions_A_win1 = obs[temporal_start_idx]
        prescriptions_A_win2 = obs[temporal_start_idx + 1]
        
        # 5 prescriptions in window of 10 → 5/10 = 0.5
        assert np.isclose(prescriptions_A_win1, 5.0 / 10.0, atol=1e-6)
        # 5 prescriptions in window of 20 → 5/20 = 0.25
        assert np.isclose(prescriptions_A_win2, 5.0 / 20.0, atol=1e-6)
        
        # B should have 0 prescriptions
        prescriptions_B_win1 = obs[temporal_start_idx + 2]
        prescriptions_B_win2 = obs[temporal_start_idx + 3]
        assert np.isclose(prescriptions_B_win1, 0.0, atol=1e-6)
        assert np.isclose(prescriptions_B_win2, 0.0, atol=1e-6)
    
    def test_temporal_features_amr_deltas(self):
        """AMR deltas should appear in observation."""
        env = create_test_env(
            enable_temporal_features=True,
            temporal_windows=[10],
            num_patients=5
        )
        obs1, _ = env.reset(seed=42)
        action_mapping = env.get_antibiotic_to_action_mapping()
        
        # First observation should have zero deltas (initial state)
        # Temporal features structure: [prescriptions_A_win1, prescriptions_B_win1, delta_A, delta_B]
        temporal_start_idx = 5 + 2  # 5 patients * 1 attr + 2 AMR levels
        delta_A_idx = temporal_start_idx + 2  # After 2 prescription counts
        delta_B_idx = temporal_start_idx + 3
        
        # Initial deltas should be 0
        assert np.isclose(obs1[delta_A_idx], 0.0, atol=1e-6)
        assert np.isclose(obs1[delta_B_idx], 0.0, atol=1e-6)
        
        # Prescribe A heavily to increase AMR
        action = np.array([action_mapping["A"]] * 5)
        obs2, _, _, _, _ = env.step(action)
        
        # Delta for A should be positive (AMR increased)
        delta_A = obs2[delta_A_idx]
        assert delta_A > 0.0
        
        # Delta for B should be zero or near-zero (no prescriptions)
        delta_B = obs2[delta_B_idx]
        assert np.isclose(delta_B, 0.0, atol=1e-6)
    
    def test_temporal_features_full_observation_structure(self):
        """Full observation should have correct structure with temporal features."""
        num_patients = 3
        temporal_windows = [5, 10]
        env = create_test_env(
            enable_temporal_features=True,
            temporal_windows=temporal_windows,
            num_patients=num_patients
        )
        obs, _ = env.reset(seed=42)
        
        # Expected structure:
        # [patient_0_prob_infected, patient_1_prob_infected, patient_2_prob_infected,
        #  AMR_A, AMR_B,
        #  prescriptions_A_win5, prescriptions_A_win10,
        #  prescriptions_B_win5, prescriptions_B_win10,
        #  delta_A, delta_B]
        
        expected_indices = {
            'patient_features': (0, 3),
            'amr_levels': (3, 5),
            'temporal_features': (5, 11),
        }
        
        # Verify patient features are in [0, 1] (prob_infected)
        for i in range(expected_indices['patient_features'][0], expected_indices['patient_features'][1]):
            assert 0.0 <= obs[i] <= 1.0
        
        # Verify AMR levels are in [0, 1]
        for i in range(expected_indices['amr_levels'][0], expected_indices['amr_levels'][1]):
            assert 0.0 <= obs[i] <= 1.0
        
        # Verify temporal features exist
        temporal_start, temporal_end = expected_indices['temporal_features']
        assert obs.shape[0] >= temporal_end


class TestTemporalFeaturesDisabled:
    """Tests to ensure temporal features don't interfere when disabled."""
    
    def test_disabled_temporal_features_no_tracking(self):
        """When disabled, temporal tracking structures should exist but not be used in obs."""
        env = create_test_env(enable_temporal_features=False)
        obs, _ = env.reset(seed=42)
        
        # Take some actions
        action = np.array([1] * env.num_patients_per_time_step)
        obs, _, _, _, _ = env.step(action)
        
        # Observation should not include temporal features
        # Expected: (5 patients * 1 attr) + 2 AMR = 7
        assert obs.shape[0] == 7
    
    def test_disabled_then_enabled_requires_new_env(self):
        """Changing temporal features requires creating a new environment."""
        # This is just to document the expected behavior: temporal features
        # are set at init time and cannot be toggled during an episode
        env1 = create_test_env(enable_temporal_features=False)
        obs1, _ = env1.reset(seed=42)
        
        env2 = create_test_env(enable_temporal_features=True)
        obs2, _ = env2.reset(seed=42)
        
        # Different observation sizes
        assert obs1.shape[0] < obs2.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
