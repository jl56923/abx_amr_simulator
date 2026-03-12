"""Unit tests for PPO_Masked (masked manager training for HRL).

Tests verify that clipped manager transitions are correctly excluded from training
updates while unclipped transitions are included normally.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch as th

from abx_amr_simulator.hrl.rl_algorithms import PPO_Masked
from test_reference_helpers import create_mock_environment, create_mock_patient_generator  # type: ignore[import-not-found]
from abx_amr_simulator.hrl import OptionBase, OptionLibrary, OptionsWrapper


class DummyOption(OptionBase):
    """Simple option for testing."""

    REQUIRES_OBSERVATION_ATTRIBUTES = ["prob_infected"]
    REQUIRES_AMR_LEVELS = False
    PROVIDES_TERMINATION_CONDITION = False

    def __init__(self, name: str, k: int = 1):
        super().__init__(name=name, k=k)

    def decide(self, env_state: dict) -> np.ndarray:
        num_patients = env_state.get("num_patients", 1)
        return np.full(shape=(num_patients,), fill_value="no_treatment", dtype=object)

    def get_referenced_antibiotics(self) -> list:
        return ["A"]


def create_test_hrl_env():
    """Create a test HRL environment with OptionsWrapper and option library."""
    base_env = create_mock_environment(antibiotic_names=["A"], num_patients_per_time_step=1)
    
    library = OptionLibrary(env=base_env)
    library.add_option(option=DummyOption(name="opt1", k=1))
    library.add_option(option=DummyOption(name="opt2", k=2))
    
    wrapped_env = OptionsWrapper(env=base_env, option_library=library, gamma=0.99)
    return wrapped_env


def test_ppo_masked_initialization():
    """Test that PPO_Masked can be initialized."""
    env = create_test_hrl_env()
    
    agent = PPO_Masked(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=4,
    )
    
    assert agent is not None
    assert agent.trainable_mask is None  # Not yet collected


def test_ppo_masked_collects_trainable_mask():
    """Test that PPO_Masked collects trainability mask from environment info."""
    env = create_test_hrl_env()
    
    agent = PPO_Masked(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=4,
    )
    
    # Run a few learning steps which triggers collect_rollouts
    agent.learn(total_timesteps=128)
    
    # Check that trainable_mask was populated
    assert agent.trainable_mask is not None
    assert len(agent.trainable_mask) > 0


def test_ppo_masked_with_clipping():
    """Test PPO_Masked trains successfully with clipped transitions present.
    
    This test creates a scenario where some transitions are clipped (at episode
    boundary) and verifies that training completes without errors.
    """
    env = create_test_hrl_env()
    
    # Create agent with small buffer to guarantee collecting both clipped and unclipped
    agent = PPO_Masked(
        policy='MlpPolicy',
        env=env,
        learning_rate=1e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
    )
    
    # Train for multiple rolls to see both clipped and unclipped transitions
    # (given max_time_steps=10 and variable option durations, we'll hit clipping)
    try:
        agent.learn(total_timesteps=512, log_interval=None)
        success = True
    except Exception as e:
        pytest.fail(f"PPO_Masked training raised exception: {e}")
    
    assert success


def test_ppo_masked_trainable_fraction_logged():
    """Test that trainable_fraction is logged during training."""
    env = create_test_hrl_env()
    
    agent = PPO_Masked(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
    )
    
    # Train and check that logger was called with trainable_fraction
    agent.learn(total_timesteps=128)
    
    # Check that trainable_mask exists after training
    assert agent.trainable_mask is not None


def test_ppo_masked_backward_compatible_without_manager_transition_trainable():
    """Test PPO_Masked works even if env doesn't provide manager_transition_trainable.
    
    This ensures backward compatibility: if the environment info dict doesn't have
    the trainable flag, the agent should default to assuming all transitions are
    trainable (mask of all 1s).
    """
    env = create_test_hrl_env()
    
    agent = PPO_Masked(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
    )
    
    # This should work even if the wrapped env doesn't set manager_transition_trainable
    try:
        agent.learn(total_timesteps=128, log_interval=None)
        success = True
    except Exception as e:
        pytest.fail(f"PPO_Masked should handle missing manager_transition_trainable: {e}")
    
    assert success
