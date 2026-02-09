"""End-to-end test for HRL training with real components.

Tests that:
1. Option library setup workflow works
2. OptionLibrary instantiates correctly with real default options
3. OptionsWrapper wraps ABXAMREnv successfully
4. HRL_PPO agent trains without errors

This test uses real instances (not mocks) of:
- ABXAMREnv
- RewardCalculator
- PatientGenerator
- OptionLibrary (loaded from real default configs)
- OptionsWrapper
- HRL_PPO (via stable-baselines3)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import numpy as np
from stable_baselines3 import PPO

from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.hrl import (
    OptionLibrary,
    OptionLibraryLoader,
    OptionsWrapper,
    setup_options_folders_with_defaults,
)
from abx_amr_simulator.utils import create_agent

# Import test helpers from centralized location to avoid duplication
# (sys.path is configured in tests/conftest.py)
from test_reference_helpers import (  # type: ignore[import-not-found]
    create_mock_environment,
    create_mock_patient_generator,
    create_mock_reward_calculator,
)


@pytest.fixture
def temp_options_dir():
    """Create a temporary options directory with real default scaffolding.
    
    Uses setup_options_folders_with_defaults() to create the standard options structure
    with all necessary YAML configs and Python loader scripts bundled with the package.
    This tests the actual workflow users will follow.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Call the real setup function - it creates tmpdir/options/ with all bundled defaults
        # including YAML configs and Python loaders for block and alternation options,
        # plus the default_deterministic.yaml library config
        setup_options_folders_with_defaults(target_path=tmpdir)
        
        # Return the created options directory
        yield Path(tmpdir) / "options"


def test_option_library_loads_from_real_configs(temp_options_dir):
    """Test that OptionLibrary loads from real default option library."""
    env = create_mock_environment(antibiotic_names=["A", "B"], num_patients_per_time_step=1)
    
    # Use the real default_deterministic library created by setup_options_folders_with_defaults()
    library_config_path = temp_options_dir / "option_libraries" / "default_deterministic.yaml"
    
    # Load library
    library, resolved_config = OptionLibraryLoader.load_library(
        library_config_path=str(library_config_path),
        env=env,
    )
    
    # Verify library loaded (default library has 12 options: 9 blocks + 3 alternations)
    assert len(library) >= 4  # At least some options load successfully given env
    assert "A_5" in library.list_options()
    assert "B_5" in library.list_options()
    assert "no_treatment_5" in library.list_options()
    
    # Verify resolved config
    assert resolved_config is not None
    assert "options" in resolved_config
    assert resolved_config["num_options"] >= 4


def test_options_wrapper_wraps_environment(temp_options_dir):
    """Test that OptionsWrapper correctly wraps ABXAMREnv with real default options."""
    env = create_mock_environment(antibiotic_names=["A", "B"], num_patients_per_time_step=1, max_time_steps=50)
    
    # Load the real default_deterministic library
    library_config_path = temp_options_dir / "option_libraries" / "default_deterministic.yaml"
    library, _ = OptionLibraryLoader.load_library(
        library_config_path=str(library_config_path),
        env=env,
    )
    
    # Wrap environment
    wrapped_env = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    
    # Verify wrapper properties
    assert wrapped_env is not None
    assert wrapped_env.observation_space is not None
    assert wrapped_env.action_space is not None
    # Default library has multiple options for antibiotics A, B, and no_treatment
    assert wrapped_env.action_space.n >= 4  # type: ignore[attr-defined]
    
    # Test reset
    obs, info = wrapped_env.reset(seed=42)
    assert obs is not None
    assert isinstance(obs, np.ndarray)


def test_hrl_ppo_agent_trains_without_error(temp_options_dir):
    """Test that HRL_PPO agent can train on wrapped environment with real default options."""
    env = create_mock_environment(antibiotic_names=["A", "B"], num_patients_per_time_step=1, max_time_steps=50)
    
    # Load the real default_deterministic library
    library_config_path = temp_options_dir / "option_libraries" / "default_deterministic.yaml"
    library, _ = OptionLibraryLoader.load_library(
        library_config_path=str(library_config_path),
        env=env,
    )
    
    # Wrap environment
    wrapped_env = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    
    # CRITICAL: Call reset() before passing to PPO to ensure observation_space is properly sized
    wrapped_env.reset(seed=42)
    
    # Create agent
    agent = PPO(
        policy="MlpPolicy",
        env=wrapped_env,
        learning_rate=3.0e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        verbose=0,
        seed=42,
    )
    
    # Verify agent created
    assert agent is not None
    
    # Train for a short period
    agent.learn(total_timesteps=500)
    
    # Verify training completed
    assert agent.num_timesteps > 0


def test_end_to_end_with_real_factory_functions(temp_options_dir):
    """Test full end-to-end with config dict, factory functions, and real default library."""
    from abx_amr_simulator.utils import (
        create_reward_calculator,
        create_patient_generator,
        create_environment,
        wrap_environment_for_hrl,
    )
    
    # Use the real default_deterministic library from setup_options_folders_with_defaults()
    library_config_path = temp_options_dir / "option_libraries" / "default_deterministic.yaml"
    
    # Build config dict for factory functions
    config = {
        'algorithm': 'HRL_PPO',
        'training': {'seed': 42},
        'reward_calculator': {
            'abx_clinical_reward_penalties_info_dict': {
                'clinical_benefit_reward': 10.0,
                'clinical_benefit_probability': 1.0,
                'clinical_failure_penalty': -1.0,
                'clinical_failure_probability': 0.0,
                'abx_adverse_effects_info': {
                    'A': {
                        'adverse_effect_penalty': -2.0,
                        'adverse_effect_probability': 0.0,
                    },
                    'B': {
                        'adverse_effect_penalty': -2.0,
                        'adverse_effect_probability': 0.0,
                    }
                }
            },
            'lambda_weight': 0.5,
            'epsilon': 0.05,
        },
        'patient_generator': {
            'prob_infected': {
                'prob_dist': {'type': 'gaussian', 'mu': 0.5, 'sigma': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.01},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        },
        'environment': {
            'antibiotics_AMR_dict': {
                'A': {
                    'leak': 0.05,
                    'flatness_parameter': 1.0,
                    'permanent_residual_volume': 0.0,
                    'initial_amr_level': 0.0
                },
                'B': {
                    'leak': 0.05,
                    'flatness_parameter': 1.0,
                    'permanent_residual_volume': 0.0,
                    'initial_amr_level': 0.0
                }
            },
            'num_patients_per_time_step': 1,
            'max_time_steps': 50,
        },
        'hrl': {
            'option_library': str(library_config_path),
            'option_gamma': 0.99,
        },
        'ppo': {
            'learning_rate': 3.0e-4,
            'n_steps': 128,
            'batch_size': 32,
            'n_epochs': 5,
            'gamma': 0.99,
            'verbose': 0,
        }
    }
    
    # Create components using factory functions
    rc = create_reward_calculator(config)
    assert rc is not None
    
    pg = create_patient_generator(config)
    assert pg is not None
    
    env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
    assert env is not None
    
    # Wrap with HRL
    wrapped_env = wrap_environment_for_hrl(env=env, config=config)  # type: ignore[arg-type]
    assert wrapped_env is not None
    assert wrapped_env.observation_space is not None
    assert wrapped_env.action_space is not None
    
    # CRITICAL: Call reset() before passing to create_agent to ensure observation_space is properly sized
    wrapped_env.reset(seed=42)
    
    # Create agent using factory
    agent = create_agent(config=config, env=wrapped_env, verbose=0)
    assert isinstance(agent, PPO)
    
    # Train briefly
    agent.learn(total_timesteps=500)
    assert agent.num_timesteps > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
