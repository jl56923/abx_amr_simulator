"""
Unit tests for HRL module (options, wrapper, obs builder).
"""

import pytest
import numpy as np
from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.core import RewardCalculator
from abx_amr_simulator.core import PatientGenerator
from abx_amr_simulator.hrl.options import Option, OptionLibrary, get_default_option_library
from abx_amr_simulator.hrl.manager_obs import ManagerObsBuilder
from abx_amr_simulator.hrl.wrapper import OptionsWrapper


def create_test_env(
    num_patients_per_time_step: int = 1,
    max_time_steps: int = 50,
) -> ABXAMREnv:
    """Create a minimal ABXAMREnv instance for HRL tests."""
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

    reward_calculator = RewardCalculator(config={
        'abx_clinical_reward_penalties_info_dict': abx_clinical_reward_penalties_info_dict,
        'lambda_weight': 0.5,
        'epsilon': 0.05,
    })

    antibiotics_AMR_dict = {
        name: {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0,
        }
        for name in antibiotic_names
    }

    pg_config = {
        'prob_infected': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'benefit_value_multiplier': {
            'prob_dist': {'type': 'constant', 'value': 1.0},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_value_multiplier': {
            'prob_dist': {'type': 'constant', 'value': 1.0},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'benefit_probability_multiplier': {
            'prob_dist': {'type': 'constant', 'value': 1.0},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_probability_multiplier': {
            'prob_dist': {'type': 'constant', 'value': 1.0},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'recovery_without_treatment_prob': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'visible_patient_attributes': ['prob_infected'],
    }

    patient_generator = PatientGenerator(config=pg_config)

    return ABXAMREnv(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        antibiotics_AMR_dict=antibiotics_AMR_dict,
        num_patients_per_time_step=num_patients_per_time_step,
        max_time_steps=max_time_steps,
        update_visible_AMR_levels_every_n_timesteps=1,
        add_noise_to_visible_AMR_levels=0.0,
        add_bias_to_visible_AMR_levels=0.0,
        crossresistance_matrix=None,
        include_steps_since_amr_update_in_obs=False,
    )


class TestOption:
    """Tests for Option class."""
    
    def test_option_creation(self):
        """Test basic option creation."""
        opt = Option(
            option_id=0,
            name="A×5",
            description="Prescribe A for 5 steps",
            action_sequence=[1, 1, 1, 1, 1],
            duration=5,
        )
        assert opt.option_id == 0
        assert opt.duration == 5
        assert opt.get_action(0) == 1
        assert opt.get_action(4) == 1
    
    def test_option_validation_length_mismatch(self):
        """Test that mismatched action sequence and duration raises error."""
        with pytest.raises(ValueError, match="action_sequence length"):
            Option(
                option_id=0,
                name="bad",
                description="",
                action_sequence=[1, 1, 1],
                duration=5,  # Mismatch
            )
    
    def test_option_validation_invalid_actions(self):
        """Test that invalid action values raise error."""
        with pytest.raises(ValueError, match="actions must be in"):
            Option(
                option_id=0,
                name="bad",
                description="",
                action_sequence=[1, 1, 5],  # 5 is invalid
                duration=3,
            )
    
    def test_option_get_action_bounds(self):
        """Test that out-of-bounds step raises error."""
        opt = Option(
            option_id=0,
            name="A×3",
            description="",
            action_sequence=[1, 1, 1],
            duration=3,
        )
        with pytest.raises(IndexError):
            opt.get_action(5)


class TestOptionLibrary:
    """Tests for OptionLibrary class."""
    
    def test_library_creation(self):
        """Test basic library creation."""
        opts = [
            Option(0, "A×5", "", [1]*5, 5),
            Option(1, "B×5", "", [2]*5, 5),
        ]
        lib = OptionLibrary(opts)
        assert len(lib) == 2
        assert lib.get_option(0).name == "A×5"
    
    def test_library_get_nonexistent(self):
        """Test that retrieving nonexistent option raises error."""
        lib = OptionLibrary([Option(0, "A×5", "", [1]*5, 5)])
        with pytest.raises(KeyError):
            lib.get_option(99)
    
    def test_library_duplicate_ids(self):
        """Test that duplicate IDs raise error."""
        opts = [
            Option(0, "A×5", "", [1]*5, 5),
            Option(0, "B×5", "", [2]*5, 5),  # Duplicate ID
        ]
        with pytest.raises(ValueError, match="unique"):
            OptionLibrary(opts)
    
    def test_default_library(self):
        """Test default option library creation."""
        lib = get_default_option_library()
        assert len(lib) == 12
        assert lib.get_option(0).name == "NO_RX×5"
        assert lib.get_option(11).name == "A,NO_RX,B,A,NO_RX"


class TestManagerObsBuilder:
    """Tests for ManagerObsBuilder class."""
    
    def test_builder_creation(self):
        """Test basic builder creation."""
        builder = ManagerObsBuilder(num_antibiotics=2)
        assert builder.compute_observation_dim() == 9  # 2*2 + 2 + 2 + 1
    
    def test_builder_observation_shape(self):
        """Test that built observation has correct shape."""
        builder = ManagerObsBuilder(num_antibiotics=2)
        amr_start = np.array([0.1, 0.2], dtype=np.float32)
        amr_end = np.array([0.15, 0.25], dtype=np.float32)
        
        obs = builder.build_observation(
            amr_start=amr_start,
            amr_end=amr_end,
            current_option_id=0,
            steps_in_episode=10,
            total_episode_steps=500,
        )
        
        expected_dim = builder.compute_observation_dim()
        assert obs.shape == (expected_dim,)
        assert obs.dtype == np.float32
    
    def test_builder_reset(self):
        """Test that reset clears state."""
        builder = ManagerObsBuilder(num_antibiotics=2)
        builder.prev_option_id = 5
        builder.consecutive_same_option = 10
        
        builder.reset()
        
        assert builder.prev_option_id is None
        assert builder.consecutive_same_option == 0
    
    def test_builder_consecutive_tracking(self):
        """Test tracking of consecutive option repeats."""
        builder = ManagerObsBuilder(num_antibiotics=2)
        amr = np.array([0.1, 0.2], dtype=np.float32)
        
        # First option
        obs1 = builder.build_observation(
            amr_start=amr, amr_end=amr, current_option_id=0,
            steps_in_episode=0, total_episode_steps=500,
        )
        assert builder.consecutive_same_option == 1
        
        # Same option again
        obs2 = builder.build_observation(
            amr_start=amr, amr_end=amr, current_option_id=0,
            steps_in_episode=5, total_episode_steps=500,
        )
        assert builder.consecutive_same_option == 2
        
        # Different option
        obs3 = builder.build_observation(
            amr_start=amr, amr_end=amr, current_option_id=1,
            steps_in_episode=10, total_episode_steps=500,
        )
        assert builder.consecutive_same_option == 1
    
    def test_builder_steps_since_drug(self):
        """Test steps_since_drug tracking."""
        builder = ManagerObsBuilder(num_antibiotics=2)
        amr = np.array([0.1, 0.2], dtype=np.float32)
        
        # Use drug 0
        builder.update_steps_since_drug(0)
        assert builder.steps_since_last_drug[0] == 0
        
        # Advance (no drugs used)
        obs = builder.build_observation(
            amr_start=amr, amr_end=amr, current_option_id=0,
            steps_in_episode=0, total_episode_steps=500,
        )
        assert builder.steps_since_last_drug[0] == 1


class TestOptionsWrapper:
    """Tests for OptionsWrapper class."""
    
    def test_wrapper_creation(self):
        """Test wrapper creation."""
        env = create_test_env(num_patients_per_time_step=1, max_time_steps=50)
        lib = get_default_option_library()
        wrapper = OptionsWrapper(env, lib)
        
        assert wrapper.action_space.n == 12  # 12 options
        assert wrapper.observation_space.shape[0] == 9  # Manager obs dim
    
    def test_wrapper_reset(self):
        """Test wrapper reset."""
        env = create_test_env(num_patients_per_time_step=1, max_time_steps=50)
        lib = get_default_option_library()
        wrapper = OptionsWrapper(env, lib)
        
        obs, info = wrapper.reset()
        
        assert obs.shape == (9,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
    
    def test_wrapper_step(self):
        """Test wrapper step."""
        env = create_test_env(num_patients_per_time_step=1, max_time_steps=50)
        lib = get_default_option_library()
        wrapper = OptionsWrapper(env, lib)
        
        obs, _ = wrapper.reset()
        
        # Execute option 0 (NO_RX×5)
        manager_obs, reward, terminated, truncated, info = wrapper.step(0)
        
        assert manager_obs.shape == (9,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'actual_option_duration' in info
        assert 'gamma_macro' in info
    
    def test_wrapper_episode_truncation(self):
        """Test that option duration is capped at episode end."""
        env = create_test_env(num_patients_per_time_step=1, max_time_steps=10)
        lib = OptionLibrary([Option(0, "A×100", "", [1]*100, 100)])  # Option longer than episode
        wrapper = OptionsWrapper(env, lib, gamma=0.99)
        
        obs, _ = wrapper.reset()
        manager_obs, reward, terminated, truncated, info = wrapper.step(0)
        
        # Should be capped to max_time_steps
        assert info['actual_option_duration'] <= 10
        assert info['gamma_macro'] == pytest.approx(0.99 ** info['actual_option_duration'], rel=1e-5)
    
    def test_wrapper_invalid_action(self):
        """Test that invalid action raises error."""
        env = create_test_env(num_patients_per_time_step=1, max_time_steps=50)
        lib = get_default_option_library()
        wrapper = OptionsWrapper(env, lib)
        
        wrapper.reset()
        
        with pytest.raises(ValueError, match="out of range"):
            wrapper.step(999)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
