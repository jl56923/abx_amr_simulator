"""Unit tests for OptionsWrapper class."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../unit/utils')))

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, MagicMock, patch
from gymnasium import spaces

from abx_amr_simulator.hrl import OptionBase, OptionLibrary, OptionsWrapper
from test_reference_helpers import create_mock_environment


class SimpleOption(OptionBase):
    """Simple option that returns fixed actions."""
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False

    def __init__(self, name: str, action_value: int, k: int = 5):
        super().__init__(name=name, k=k)
        self.action_value = action_value

    def decide(self, env_state):
        num_patients = env_state['num_patients']
        return np.full(num_patients, self.action_value, dtype=np.int32)


def create_mock_env(num_patients=2, num_abx=2, max_steps=100):
    """Helper to create a mock environment."""
    env = Mock(spec=gym.Env)
    unwrapped = Mock()
    
    # Set up basic attributes
    unwrapped.num_patients_per_time_step = num_patients
    unwrapped.max_steps = max_steps
    unwrapped.current_patients = [Mock() for _ in range(num_patients)]
    unwrapped.current_step = 0
    
    # Set up reward calculator
    abx_names = [f'ABX_{i}' for i in range(num_abx)]
    reward_calculator = Mock()
    reward_calculator.abx_name_to_index = {abx: i for i, abx in enumerate(abx_names)}
    unwrapped.reward_calculator = reward_calculator
    
    # Set up leaky balloons for AMR tracking
    unwrapped.leaky_balloons = {
        abx: Mock(get_current_level=Mock(return_value=0.5))
        for abx in abx_names
    }
    
    # Set up patient generator
    patient_generator = Mock()
    patient_generator.visible_patient_attributes = []
    unwrapped.patient_generator = patient_generator
    
    env.unwrapped = unwrapped
    
    # Mock reset and step methods
    def mock_reset(seed=None, options=None):
        obs = np.zeros(num_patients * 2 + num_abx, dtype=np.float32)
        return obs, {}
    
    def mock_step(action):
        obs = np.zeros(num_patients * 2 + num_abx, dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
    
    env.reset = mock_reset
    env.step = mock_step
    env.observation_space = spaces.Box(0, 1, shape=(10,))
    env.action_space = spaces.MultiDiscrete([num_abx + 1] * num_patients)
    
    return env


class TestOptionsWrapperInit:
    """Test OptionsWrapper initialization."""

    def test_init_with_valid_env_and_library(self):
        """Test initialization with valid environment and library."""
        env = create_mock_environment(antibiotic_names=['ABX_0', 'ABX_1'], num_patients_per_time_step=2)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=5))
        
        wrapper = OptionsWrapper(env=env, option_library=lib, gamma=0.99)
        
        assert wrapper.env is env
        assert wrapper.option_library is lib
        assert wrapper.gamma == 0.99

    def test_init_with_invalid_env_no_reward_calculator(self):
        """Test initialization fails if env missing reward_calculator."""
        # Create a valid env for OptionLibrary initialization
        valid_env = create_mock_env()
        lib = OptionLibrary(env=valid_env)
        lib.add_option(SimpleOption(name='opt1', action_value=0))
        
        # Create invalid env without reward_calculator for OptionsWrapper
        invalid_env = Mock()
        invalid_env.unwrapped = Mock(spec=[])  # No reward_calculator or patient_generator
        
        with pytest.raises(ValueError) as exc_info:
            OptionsWrapper(env=invalid_env, option_library=lib)
        
        # Should fail on one of the required attributes (reward_calculator or patient_generator)
        error_msg = str(exc_info.value)
        assert 'reward_calculator' in error_msg or 'patient_generator' in error_msg

    def test_init_validates_option_library_compatibility(self):
        """Test that init calls validation."""
        env = create_mock_env()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0))
        
        # Mock validate to raise an error
        with patch.object(lib, 'validate_environment_compatibility') as mock_validate:
            mock_validate.side_effect = ValueError("Incompatible!")
            
            with pytest.raises(ValueError) as exc_info:
                OptionsWrapper(env=env, option_library=lib)
            
            assert 'Incompatible' in str(exc_info.value)
            mock_validate.assert_called_once()


class TestOptionsWrapperReset:
    """Test reset() method."""

    def test_reset_returns_manager_obs(self):
        """Test that reset returns manager observation."""
        env = create_mock_env(num_patients=2, num_abx=2)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        manager_obs, info = wrapper.reset()
        
        assert isinstance(manager_obs, np.ndarray)
        assert isinstance(info, dict)
        # Manager obs should have: 2 AMR levels + 1 progress = 3
        assert manager_obs.shape == (3,)

    def test_reset_calls_option_reset(self):
        """Test that reset calls reset on all options."""
        env = create_mock_env()
        lib = OptionLibrary(env=env)
        
        opt1 = SimpleOption(name='opt1', action_value=0)
        opt2 = SimpleOption(name='opt2', action_value=1)
        lib.add_option(opt1)
        lib.add_option(opt2)
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        
        # Mock option reset methods
        opt1.reset = Mock()
        opt2.reset = Mock()
        
        wrapper.reset()
        
        opt1.reset.assert_called_once()
        opt2.reset.assert_called_once()


class TestOptionsWrapperStep:
    """Test step() method."""

    def test_step_with_valid_manager_action(self):
        """Test step with valid manager action."""
        env = create_mock_env()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=2))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        manager_obs, reward, terminated, truncated, info = wrapper.step(0)
        
        assert isinstance(manager_obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert info['option_name'] == 'opt1'

    def test_step_executes_option_for_k_steps(self):
        """Test that step executes option for k substeps."""
        env = create_mock_env()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=3))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        # Mock env.step to track calls
        step_calls = []
        original_step = env.step
        
        def tracked_step(action):
            step_calls.append(action)
            return original_step(action)
        
        env.step = tracked_step
        
        wrapper.step(0)
        
        # Should have called env.step 3 times (once for each substep)
        assert len(step_calls) == 3

    def test_step_accumulates_discounted_reward(self):
        """Test that step accumulates rewards with discounting."""
        env = create_mock_env()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=3))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        # Mock env.step to return reward=1.0
        def mock_step_reward_1(action):
            obs = np.zeros(10, dtype=np.float32)
            reward = 1.0
            terminated = False
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info
        
        env.step = mock_step_reward_1
        
        # Execute option
        manager_obs, reward, terminated, truncated, info = wrapper.step(0)
        
        # Expected: 1.0 + 0.99*1.0 + 0.99^2*1.0 ≈ 2.9701
        expected_reward = 1.0 + 0.99 + 0.99**2
        assert np.isclose(reward, expected_reward)

    def test_step_with_invalid_manager_action(self):
        """Test step with invalid manager action."""
        env = create_mock_env()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        with pytest.raises(ValueError):
            wrapper.step(99)  # Out of range

    def test_step_early_termination_by_episode(self):
        """Test that step stops if episode terminates."""
        env = create_mock_env()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=10))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        # Mock env.step to return terminated=True on 2nd call
        call_count = [0]
        def mock_step_terminate(action):
            call_count[0] += 1
            obs = np.zeros(10, dtype=np.float32)
            reward = 1.0
            terminated = call_count[0] >= 2  # Terminate on 2nd step
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info
        
        env.step = mock_step_terminate
        
        manager_obs, reward, terminated, truncated, info = wrapper.step(0)
        
        # Should have stopped after 2 substeps even though k=10
        assert info['option_duration'] == 2
        assert terminated is True


class TestOptionsWrapperActionValidation:
    """Test action validation (Layer 3)."""

    def test_validate_actions_wrong_type(self):
        """Test validation fails if actions wrong type."""
        env = create_mock_env()
        lib = OptionLibrary(env=env)
        
        class BadOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = False
            
            def decide(self, env_state):
                return [0, 1]  # List instead of ndarray
        
        lib.add_option(BadOption(name='bad', k=1))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        with pytest.raises(TypeError):
            wrapper.step(0)

    def test_validate_actions_wrong_shape(self):
        """Test validation fails if shape wrong."""
        env = create_mock_env(num_patients=2)
        lib = OptionLibrary(env=env)
        
        class BadOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = False
            
            def decide(self, env_state):
                return np.array([0])  # Wrong shape (1,) instead of (2,)
        
        lib.add_option(BadOption(name='bad', k=1))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        with pytest.raises(ValueError):
            wrapper.step(0)

    def test_validate_actions_out_of_range(self):
        """Test validation fails if action indices out of range."""
        env = create_mock_env(num_patients=2, num_abx=2)
        lib = OptionLibrary(env=env)
        
        class BadOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = False
            
            def decide(self, env_state):
                return np.array([0, 99], dtype=np.int32)  # 99 out of range
        
        lib.add_option(BadOption(name='bad', k=1))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        with pytest.raises(ValueError):
            wrapper.step(0)


class TestOptionsWrapperBuildsEnvState:
    """Test that wrapper correctly builds env_state."""

    def test_build_env_state_structure(self):
        """Test that env_state has correct structure."""
        env = create_mock_env(num_patients=2, num_abx=2)
        lib = OptionLibrary(env=env)
        
        class StateCheckOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = False
            
            def decide(self, env_state):
                # Check env_state structure
                assert 'patients' in env_state
                assert 'num_patients' in env_state
                assert 'current_amr_levels' in env_state
                assert 'current_step' in env_state
                assert 'max_steps' in env_state
                
                assert env_state['num_patients'] == 2
                assert len(env_state['patients']) == 2
                
                return np.zeros(env_state['num_patients'], dtype=np.int32)
        
        lib.add_option(StateCheckOption(name='check', k=1))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        # Step should pass the assertion checks in decide()
        wrapper.step(0)
