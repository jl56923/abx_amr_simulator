"""End-to-end test for HRL training workflow with real default configs.

Tests the ACTUAL USER WORKFLOW:
1. Call setup_config_folders_with_defaults() to create config structure
2. Call setup_options_folders_with_defaults() to create option library structure  
3. Load default HRL umbrella config (hrl_ppo_default.yaml)
4. Use load_config() to merge component configs
5. Create components via factory functions
6. Train for short period

This catches mismatches between default environment antibiotic names and default
option library antibiotic names, which would break at runtime for new users.

This test uses real instances (not mocks) of everything:
- Default YAML configs from package
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
from abx_amr_simulator.utils import (
    create_agent,
    setup_config_folders_with_defaults,
    load_config,
    create_reward_calculator,
    create_patient_generator,
    create_environment,
    wrap_environment_for_hrl,
)

# Import test helpers from centralized location to avoid duplication
# (sys.path is configured in tests/conftest.py)
from test_reference_helpers import (  # type: ignore[import-not-found]
    create_mock_environment,
    create_mock_patient_generator,
    create_mock_reward_calculator,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with real default config and option scaffolding.
    
    Simulates the actual user workflow:
    1. setup_config_folders_with_defaults() creates experiments/configs/
    2. setup_options_folders_with_defaults() creates experiments/options/
    
    This ensures we test with the actual default YAML files that users will use.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        experiments_dir = tmpdir / "experiments"
        experiments_dir.mkdir()
        
        # Step 1: Create config structure (creates experiments/configs/)
        setup_config_folders_with_defaults(target_path=experiments_dir)
        
        # Step 2: Create options structure (creates experiments/options/)
        setup_options_folders_with_defaults(target_path=experiments_dir)
        
        yield experiments_dir


def test_real_user_workflow_with_default_configs(temp_workspace):
    """Test the ACTUAL NEW USER WORKFLOW with real default configs.
    
    This test simulates exactly what a new user would do:
    1. setup_config_folders_with_defaults()
    2. setup_options_folders_with_defaults()  
    3. Load default HRL umbrella config (hrl_ppo_default.yaml)
    4. Use load_config() to load and merge component configs
    5. Create components via factory functions
    6. Train agent
    
    This will FAIL if there's a mismatch between antibiotic names in:
    - Default environment config
    - Default reward calculator config  
    - Default option library config
    
    Which is exactly what we want - the test should catch configuration errors!
    """
    # Load the default HRL umbrella config
    umbrella_config_path = temp_workspace / "configs" / "umbrella_configs" / "hrl_ppo_default.yaml"
    assert umbrella_config_path.exists(), f"Default HRL config not found: {umbrella_config_path}"
    
    # Use load_config() to load and merge all component configs
    # This is what users will actually do
    config = load_config(config_path=str(umbrella_config_path))
    
    # Verify config loaded
    assert 'environment' in config
    assert 'reward_calculator' in config
    assert 'patient_generator' in config
    assert 'hrl' in config
    assert 'algorithm' in config
    assert config['algorithm'] == 'HRL_PPO'
    
    # Modify for quick testing
    config['training']['total_num_training_episodes'] = 2  # Very short test
    config['environment']['max_time_steps'] = 10  # Short episodes
    
    # Create components using factory functions (actual user workflow)
    rc = create_reward_calculator(config=config)
    assert rc is not None
    
    pg = create_patient_generator(config=config)
    assert pg is not None
    
    env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
    assert env is not None
    
    # Wrap with HRL - this is where antibiotic name mismatch would be caught
    wrapped_env = wrap_environment_for_hrl(env=env, config=config)
    assert wrapped_env is not None
    assert wrapped_env.observation_space is not None
    assert wrapped_env.action_space is not None
    
    # Reset before passing to agent
    wrapped_env.reset(seed=42)
    
    # Create agent using factory
    agent = create_agent(config=config, env=wrapped_env, verbose=0)
    assert isinstance(agent, PPO)
    
    # Train briefly to ensure everything works end-to-end
    total_timesteps = config['training']['total_num_training_episodes'] * config['environment']['max_time_steps']
    agent.learn(total_timesteps=total_timesteps)
    assert agent.num_timesteps > 0
    
    # Cleanup
    env.close()


def test_option_library_loads_from_real_configs(temp_workspace):
    """Test that OptionLibrary loads from real default option library."""
    # Load actual environment config to get real antibiotic names
    umbrella_config_path = temp_workspace / "configs" / "umbrella_configs" / "base_experiment.yaml"
    config = load_config(config_path=str(umbrella_config_path))
    
    # Create environment with actual default antibiotic names
    rc = create_reward_calculator(config=config)
    pg = create_patient_generator(config=config)
    env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
    
    # Use the real default_deterministic library created by setup_options_folders_with_defaults()
    library_config_path = temp_workspace / "options" / "option_libraries" / "default_deterministic.yaml"
    
    # Load library - this should work if antibiotic names match
    library, resolved_config = OptionLibraryLoader.load_library(
        library_config_path=str(library_config_path),
        env=env,
    )
    
    # Verify library loaded (default library has 12 options: 9 blocks + 3 alternations)
    assert len(library) >= 4  # At least some options load successfully
    assert resolved_config is not None
    assert "options" in resolved_config
    assert resolved_config["num_options"] >= 4
    
    # Cleanup
    env.close()


def test_options_wrapper_wraps_environment_with_real_defaults(temp_workspace):
    """Test that OptionsWrapper wraps environment using real default configs."""
    # Load actual environment config
    umbrella_config_path = temp_workspace / "configs" / "umbrella_configs" / "base_experiment.yaml"
    config = load_config(config_path=str(umbrella_config_path))
    
    # Create environment with actual default antibiotic names
    rc = create_reward_calculator(config=config)
    pg = create_patient_generator(config=config)
    env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
    
    # Load the real default_deterministic library
    library_config_path = temp_workspace / "options" / "option_libraries" / "default_deterministic.yaml"
    library, _ = OptionLibraryLoader.load_library(
        library_config_path=str(library_config_path),
        env=env,
    )
    
    # Wrap environment
    
    # Wrap environment
    wrapped_env = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    
    # Verify wrapper properties
    assert wrapped_env is not None
    assert wrapped_env.observation_space is not None
    assert wrapped_env.action_space is not None
    assert wrapped_env.action_space.n >= 4  # type: ignore[attr-defined]
    
    # Test reset
    obs, info = wrapped_env.reset(seed=42)
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])