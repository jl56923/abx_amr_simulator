"""Unit tests for OptionsWrapper behavior and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from abx_amr_simulator.hrl import OptionBase, OptionLibrary, OptionsWrapper

# Import test helpers (sys.path configured in tests/conftest.py)
from test_reference_helpers import create_mock_environment, create_mock_patient_generator  # type: ignore[import-not-found]


class ConstantOption(OptionBase):
    """Option that always prescribes the same action index."""

    REQUIRES_OBSERVATION_ATTRIBUTES = ["prob_infected"]
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = False

    def __init__(self, name: str, action_index: int, k: int = 1):
        super().__init__(name=name, k=k)
        self._action_index = action_index

    def decide(self, env_state: dict) -> np.ndarray:
        num_patients = env_state.get("num_patients", 1)
        return np.full(shape=(num_patients,), fill_value=self._action_index, dtype=np.int32)

    def get_referenced_antibiotics(self) -> list:
        return ["A"]


def _build_library(env, action_index: int = 0) -> OptionLibrary:
    library = OptionLibrary(env=env)
    library.add_option(option=ConstantOption(name="opt", action_index=action_index, k=1))
    return library


def test_wrapper_init_requires_patient_generator() -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    library = _build_library(env=env)

    delattr(env.unwrapped, "patient_generator")

    with pytest.raises(ValueError, match="patient_generator"):
        OptionsWrapper(env=env, option_library=library, gamma=0.99)


def test_wrapper_init_invalid_option_library_raises() -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    library = OptionLibrary(env=env)

    with pytest.raises(ValueError, match="incompatible"):
        OptionsWrapper(env=env, option_library=library, gamma=0.99)


def test_compute_observation_dimension_summary_vs_full_vector() -> None:
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=3,
    )
    library = _build_library(env=env)

    wrapper_summary = OptionsWrapper(
        env=env,
        option_library=library,
        gamma=0.99,
        front_edge_use_full_vector=False,
    )
    wrapper_full = OptionsWrapper(
        env=env,
        option_library=library,
        gamma=0.99,
        front_edge_use_full_vector=True,
    )

    summary_dim = wrapper_summary._compute_observation_dimension()
    full_dim = wrapper_full._compute_observation_dimension()

    assert full_dim > summary_dim


def test_reset_initializes_manager_observation() -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    library = _build_library(env=env)
    wrapper = OptionsWrapper(env=env, option_library=library, gamma=0.99)

    obs, info = wrapper.reset(seed=42)

    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


def test_step_invalid_manager_action_type_raises() -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    library = _build_library(env=env)
    wrapper = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    wrapper.reset(seed=42)

    with pytest.raises(TypeError, match="manager_action must be int"):
        wrapper.step(manager_action="bad")  # type: ignore[arg-type]


def test_step_out_of_range_manager_action_raises() -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    library = _build_library(env=env)
    wrapper = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    wrapper.reset(seed=42)

    with pytest.raises(ValueError, match="out of range"):
        wrapper.step(manager_action=5)


def test_validate_actions_type_shape_and_range() -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    library = _build_library(env=env)
    wrapper = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    wrapper.reset(seed=42)
    num_patients = env.unwrapped.num_patients_per_time_step

    with pytest.raises(TypeError, match="expected np.ndarray"):
        wrapper._validate_actions(actions=[1, 2], option_name="opt")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected action shape"):
        wrapper._validate_actions(
            actions=np.array([0, 1], dtype=np.int32),
            option_name="opt",
        )

    with pytest.raises(TypeError, match="Expected integer dtype"):
        wrapper._validate_actions(
            actions=np.full(shape=(num_patients,), fill_value=0.5, dtype=np.float32),
            option_name="opt",
        )

    with pytest.raises(ValueError, match="Invalid action indices"):
        wrapper._validate_actions(
            actions=np.full(shape=(num_patients,), fill_value=999, dtype=np.int32),
            option_name="opt",
        )


def test_build_env_state_fallbacks() -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    library = _build_library(env=env)
    wrapper = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    obs, _ = wrapper.reset(seed=42)

    if hasattr(env.unwrapped, "current_time_step"):
        delattr(env.unwrapped, "current_time_step")

    env_state = wrapper._build_env_state(observation=obs)

    assert env_state["current_step"] == 0
    assert env_state["num_patients"] == env.unwrapped.num_patients_per_time_step


def test_extract_patients_fallback_when_missing_current_patients() -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    library = _build_library(env=env)
    wrapper = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    obs, _ = wrapper.reset(seed=42)

    env.unwrapped.current_patients = None

    patients = wrapper._extract_patients_from_obs(
        observation=obs,
        num_patients=env.unwrapped.num_patients_per_time_step,
    )

    assert len(patients) == env.unwrapped.num_patients_per_time_step
    assert "prob_infected" in patients[0]


def test_get_current_amr_levels_fallback_when_missing_amr() -> None:
    env = create_mock_environment(antibiotic_names=["A", "B"])
    library = _build_library(env=env)
    wrapper = OptionsWrapper(env=env, option_library=library, gamma=0.99)

    if hasattr(env.unwrapped, "amr_balloon_models"):
        delattr(env.unwrapped, "amr_balloon_models")

    current_amr = wrapper._get_current_amr_levels()

    assert current_amr["A"] == 0.0
    assert current_amr["B"] == 0.0


def test_build_front_edge_features_with_no_patients_summary() -> None:
    env = create_mock_environment(antibiotic_names=["A"], num_patients_per_time_step=2)
    library = _build_library(env=env)
    wrapper = OptionsWrapper(
        env=env,
        option_library=library,
        gamma=0.99,
        front_edge_use_full_vector=False,
    )
    obs, _ = wrapper.reset(seed=42)

    env.unwrapped.current_patients = []
    wrapper.current_env_obs = obs

    features = wrapper._build_front_edge_features()

    assert features.shape == (2 * len(wrapper.patient_generator.visible_patient_attributes),)


def test_build_front_edge_features_with_no_patients_full_vector() -> None:
    env = create_mock_environment(antibiotic_names=["A"], num_patients_per_time_step=3)
    library = _build_library(env=env)
    wrapper = OptionsWrapper(
        env=env,
        option_library=library,
        gamma=0.99,
        front_edge_use_full_vector=True,
    )
    obs, _ = wrapper.reset(seed=42)

    env.unwrapped.current_patients = []
    wrapper.current_env_obs = obs

    features = wrapper._build_front_edge_features()

    expected_length = env.unwrapped.num_patients_per_time_step * len(
        wrapper.patient_generator.visible_patient_attributes
    )
    assert features.shape == (expected_length,)


def test_update_steps_since_prescribed_tracks_actions() -> None:
    env = create_mock_environment(antibiotic_names=["A", "B"])
    library = _build_library(env=env)
    wrapper = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    wrapper.reset(seed=42)

    actions = np.array([0, 2], dtype=np.int32)
    wrapper._update_steps_since_prescribed(actions=actions)

    assert wrapper._steps_since_prescribed["A"] == 0
    assert wrapper._steps_since_prescribed["B"] == 1


def test_step_runs_single_macro_action() -> None:
    env = create_mock_environment(antibiotic_names=["A"], num_patients_per_time_step=1)
    library = _build_library(env=env)
    wrapper = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    wrapper.reset(seed=42)

    obs, reward, terminated, truncated, info = wrapper.step(manager_action=0)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(info, dict)
    assert terminated in {True, False}
    assert truncated in {True, False}
