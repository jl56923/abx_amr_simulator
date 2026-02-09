"""Unit tests for OptionsWrapper class."""

import pytest
import numpy as np

from abx_amr_simulator.hrl import OptionBase, OptionLibrary, OptionsWrapper
# Import test helpers (sys.path configured in tests/conftest.py)
from test_reference_helpers import create_mock_environment  # type: ignore[import-not-found]


class SimpleOption(OptionBase):
    """Simple option that returns fixed actions."""
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False

    def __init__(self, name: str, action_value: int, k: int = 5):
        super().__init__(name=name, k=k)
        self.action_value = action_value

    def decide(self, env_state):
        num_patients = env_state['num_patients']
        return np.full(shape=num_patients, fill_value=self.action_value, dtype=np.int32)
    
    def get_referenced_antibiotics(self):
        """Test option returns no specific antibiotics."""
        return []


def create_test_environment(
    num_patients: int = 2,
    num_abx: int = 2,
    max_steps: int = 100,
    visible_attrs: list[str] | None = None,
):
    """Helper to create a real environment with small deterministic settings."""
    if visible_attrs is None:
        visible_attrs = ["prob_infected"]
    antibiotic_names = [f"ABX_{i}" for i in range(num_abx)]
    return create_mock_environment(
        antibiotic_names=antibiotic_names,
        num_patients_per_time_step=num_patients,
        max_time_steps=max_steps,
        visible_patient_attributes=visible_attrs,
    )


def extract_patient_matrix(patients, visible_attrs):
    """Extract observed attribute matrix from patient objects."""
    values = []
    for patient in patients:
        row = []
        for attr in visible_attrs:
            obs_attr = f"{attr}_obs"
            row.append(float(getattr(patient, obs_attr, getattr(patient, attr, 0.0))))
        values.append(row)
    return np.array(object=values, dtype=np.float32)


class TestOptionsWrapperInit:
    """Test OptionsWrapper initialization."""

    def test_init_with_valid_env_and_library(self):
        """Test initialization with valid environment and library."""
        env = create_test_environment(num_patients=2, num_abx=2)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=5))
        
        wrapper = OptionsWrapper(env=env, option_library=lib, gamma=0.99)
        
        assert wrapper.env is env
        assert wrapper.option_library is lib
        assert wrapper.gamma == 0.99

    def test_init_with_invalid_env_no_reward_calculator(self):
        """Test initialization fails if env missing reward_calculator."""
        # Create a valid env for OptionLibrary initialization
        valid_env = create_test_environment()
        lib = OptionLibrary(env=valid_env)
        lib.add_option(SimpleOption(name='opt1', action_value=0))
        
        # Create invalid env with patient_generator set to None for OptionsWrapper
        invalid_env = create_test_environment()
        invalid_env.patient_generator = None
        
        with pytest.raises(ValueError) as exc_info:
            OptionsWrapper(env=invalid_env, option_library=lib)
        
        # Should fail on missing PatientGenerator attributes
        error_msg = str(exc_info.value)
        assert 'PatientGenerator' in error_msg or 'visible_patient_attributes' in error_msg

    def test_init_validates_option_library_compatibility(self):
        """Test that init calls validation."""
        env = create_test_environment(num_abx=1)
        lib = OptionLibrary(env=env)
        
        class BadOption(SimpleOption):
            def get_referenced_antibiotics(self):
                return ["MISSING_ABX"]

        lib.add_option(BadOption(name='opt1', action_value=0))

        with pytest.raises(ValueError) as exc_info:
            OptionsWrapper(env=env, option_library=lib)
        
        assert 'MISSING_ABX' in str(exc_info.value)


class TestOptionsWrapperReset:
    """Test reset() method."""

    def test_reset_returns_manager_obs(self):
        """Test that reset returns manager observation."""
        env = create_test_environment(num_patients=2, num_abx=2)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        manager_obs, info = wrapper.reset()
        
        assert isinstance(manager_obs, np.ndarray)
        assert isinstance(info, dict)
        visible_attrs = list(env.patient_generator.visible_patient_attributes)
        num_abx = len(env.reward_calculator.abx_name_to_index)
        expected_len = (
            4 * len(visible_attrs)
            + 2 * num_abx
            + (2 + num_abx)
            + 1
            + 2 * len(visible_attrs)
        )
        assert manager_obs.shape == (expected_len,)

    def test_reset_includes_front_edge_summary_stats(self):
        """Test that reset appends mean + std for front-edge attributes."""
        visible_attrs = ["prob_infected", "benefit_value_multiplier"]
        env = create_test_environment(
            num_patients=3,
            num_abx=2,
            visible_attrs=visible_attrs,
        )
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0))

        wrapper = OptionsWrapper(env=env, option_library=lib, front_edge_use_full_vector=False)
        manager_obs, _ = wrapper.reset()

        front_edge = manager_obs[-4:]
        cohort_matrix = extract_patient_matrix(env.current_patients, visible_attrs)
        means = np.mean(a=cohort_matrix, axis=0)
        stds = np.std(a=cohort_matrix, axis=0)
        expected = np.array(
            object=[means[0], stds[0], means[1], stds[1]],
            dtype=np.float32,
        )
        assert np.allclose(a=front_edge, b=expected, atol=1e-6)

    def test_reset_includes_front_edge_full_vector(self):
        """Test that reset appends full cohort vector when enabled."""
        visible_attrs = ["prob_infected", "benefit_value_multiplier"]
        env = create_test_environment(
            num_patients=3,
            num_abx=2,
            visible_attrs=visible_attrs,
        )
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0))

        wrapper = OptionsWrapper(env=env, option_library=lib, front_edge_use_full_vector=True)
        manager_obs, _ = wrapper.reset()

        front_edge = manager_obs[-6:]
        cohort_matrix = extract_patient_matrix(env.current_patients, visible_attrs)
        expected = cohort_matrix.flatten()
        assert np.allclose(a=front_edge, b=expected, atol=1e-6)

    def test_reset_calls_option_reset(self):
        """Test that reset calls reset on all options."""
        env = create_test_environment()
        lib = OptionLibrary(env=env)

        class ResetTrackingOption(SimpleOption):
            def __init__(self, name: str, action_value: int, k: int = 5):
                super().__init__(name=name, action_value=action_value, k=k)
                self.reset_count = 0

            def reset(self) -> None:
                self.reset_count += 1

        opt1 = ResetTrackingOption(name='opt1', action_value=0)
        opt2 = ResetTrackingOption(name='opt2', action_value=1)
        lib.add_option(opt1)
        lib.add_option(opt2)
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        assert opt1.reset_count == 1
        assert opt2.reset_count == 1


class TestOptionsWrapperStep:
    """Test step() method."""

    def test_step_with_valid_manager_action(self):
        """Test step with valid manager action."""
        env = create_test_environment()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=2))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        manager_obs, reward, terminated, truncated, info = wrapper.step(manager_action=0)
        
        assert isinstance(manager_obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert info['option_name'] == 'opt1'

    def test_step_executes_option_for_k_steps(self):
        """Test that step executes option for k substeps."""
        env = create_test_environment()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=3))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        # Mock env.step to track calls
        step_calls = []
        original_step = env.step
        
        def tracked_step(action):
            step_calls.append(action)
            return original_step(action=action)
        
        env.step = tracked_step
        
        wrapper.step(manager_action=0)
        
        # Should have called env.step 3 times (once for each substep)
        assert len(step_calls) == 3

    def test_step_accumulates_discounted_reward(self):
        """Test that step accumulates rewards with discounting."""
        env = create_test_environment()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=3))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        # Mock env.step to return reward=1.0
        def mock_step_reward_1(action):
            obs = np.zeros(shape=10, dtype=np.float32)
            reward = 1.0
            terminated = False
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info
        
        env.step = mock_step_reward_1
        
        # Execute option
        manager_obs, reward, terminated, truncated, info = wrapper.step(manager_action=0)
        
        # Expected: 1.0 + 0.99*1.0 + 0.99^2*1.0 ≈ 2.9701
        expected_reward = 1.0 + 0.99 + 0.99**2
        assert np.isclose(a=reward, b=expected_reward)

    def test_step_with_invalid_manager_action(self):
        """Test step with invalid manager action."""
        env = create_test_environment()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        with pytest.raises(ValueError):
            wrapper.step(manager_action=99)  # Out of range

    def test_step_early_termination_by_episode(self):
        """Test that step stops if episode terminates."""
        env = create_test_environment()
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', action_value=0, k=10))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        # Mock env.step to return terminated=True on 2nd call
        call_count = [0]
        def mock_step_terminate(action):
            call_count[0] += 1
            obs = np.zeros(shape=10, dtype=np.float32)
            reward = 1.0
            terminated = call_count[0] >= 2  # Terminate on 2nd step
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info
        
        env.step = mock_step_terminate
        
        manager_obs, reward, terminated, truncated, info = wrapper.step(manager_action=0)
        
        # Should have stopped after 2 substeps even though k=10
        assert info['option_duration'] == 2
        assert terminated is True


class TestOptionsWrapperActionValidation:
    """Test action validation (Layer 3)."""

    def test_validate_actions_wrong_type(self):
        """Test validation fails if actions wrong type."""
        env = create_test_environment()
        lib = OptionLibrary(env=env)
        
        class BadOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = False
            
            def decide(self, env_state):
                return [0, 1]  # List instead of ndarray
            
            def get_referenced_antibiotics(self):
                return []
        
        lib.add_option(BadOption(name='bad', k=1))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        with pytest.raises(TypeError):
            wrapper.step(manager_action=0)

    def test_validate_actions_wrong_shape(self):
        """Test validation fails if shape wrong."""
        env = create_test_environment(num_patients=2)
        lib = OptionLibrary(env=env)
        
        class BadOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = False
            
            def decide(self, env_state):
                return np.array(object=[0])  # Wrong shape (1,) instead of (2,)
            
            def get_referenced_antibiotics(self):
                return []
        
        lib.add_option(BadOption(name='bad', k=1))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        with pytest.raises(ValueError):
            wrapper.step(manager_action=0)

    def test_validate_actions_out_of_range(self):
        """Test validation fails if action indices out of range."""
        env = create_test_environment(num_patients=2, num_abx=2)
        lib = OptionLibrary(env=env)
        
        class BadOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = False
            
            def decide(self, env_state):
                return np.array(object=[0, 99], dtype=np.int32)  # 99 out of range
            
            def get_referenced_antibiotics(self):
                return []
        
        lib.add_option(BadOption(name='bad', k=1))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        with pytest.raises(ValueError):
            wrapper.step(manager_action=0)


class TestOptionsWrapperBuildsEnvState:
    """Test that wrapper correctly builds env_state."""

    def test_build_env_state_structure(self):
        """Test that env_state has correct structure."""
        env = create_test_environment(num_patients=2, num_abx=2)
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

                return np.zeros(shape=env_state['num_patients'], dtype=np.int32)
            
            def get_referenced_antibiotics(self):
                return []
        
        lib.add_option(StateCheckOption(name='check', k=1))
        
        wrapper = OptionsWrapper(env=env, option_library=lib)
        wrapper.reset()
        
        # Step should pass the assertion checks in decide()
        wrapper.step(manager_action=0)
