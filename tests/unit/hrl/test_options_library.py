"""Tests for OptionLibrary class.

Tests cover library initialization, option management, validation, and error handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from abx_amr_simulator.hrl import OptionBase, OptionLibrary

# Import test helpers
from test_reference_helpers import create_mock_environment, create_mock_patient_generator  # type: ignore[import-not-found]


def create_test_option(
    name: str = "simple",
    requires_observation_attributes: list | None = None,
    requires_amr_levels: bool = False,
    requires_step_number: bool = False,
    provides_termination: bool = False,
    referenced_antibiotics: list | None = None,
    k: int | float | None = None,
) -> OptionBase:
    """Create a simple OptionBase instance with configurable requirements."""

    class TestOption(OptionBase):
        REQUIRES_OBSERVATION_ATTRIBUTES = (
            requires_observation_attributes
            if requires_observation_attributes is not None
            else ["prob_infected"]
        )
        REQUIRES_AMR_LEVELS = requires_amr_levels
        REQUIRES_STEP_NUMBER = requires_step_number
        PROVIDES_TERMINATION_CONDITION = provides_termination

        def __init__(self, name: str = name, k: int | float | None = k):
            super().__init__(name=name, k=k)

        def get_referenced_antibiotics(self) -> list:
            if referenced_antibiotics is None:
                return ["A"]
            return list(referenced_antibiotics)

        def decide(self, env_state: dict) -> np.ndarray:
            num_patients = env_state.get("num_patients", 1)
            return np.zeros(shape=num_patients, dtype=np.int32)

    return TestOption(name=name, k=k)


class ObservableOption(OptionBase):
    """Option that captures observable attributes injection."""

    REQUIRES_OBSERVATION_ATTRIBUTES = ["prob_infected"]

    def __init__(self, name: str = "observable"):
        super().__init__(name=name)
        self.observed_attributes: list | None = None

    def get_referenced_antibiotics(self) -> list:
        return ["A"]

    def decide(self, env_state: dict) -> np.ndarray:
        num_patients = env_state.get("num_patients", 1)
        return np.zeros(shape=num_patients, dtype=np.int32)

    def set_observable_attributes(self, attrs: list) -> None:
        self.observed_attributes = list(attrs)


class TestOptionLibraryInitialization:
    """Tests for OptionLibrary initialization and basic operations."""

    def test_init_with_environment(self):
        """Test OptionLibrary initialization with environment."""
        env = create_mock_environment(antibiotic_names=["A", "B"])
        library = OptionLibrary(env=env, name="test_lib")

        assert library.name == "test_lib"
        assert library.env is not None
        assert len(library) == 0
        assert library.abx_name_to_index == {"A": 0, "B": 1, "no_treatment": 2}

    def test_init_with_default_name(self):
        """Test OptionLibrary initialization with default name."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)

        assert library.name == "default"

    def test_init_extracts_antibiotic_mapping(self):
        """Test that init extracts antibiotic name-to-index mapping from environment."""
        env = create_mock_environment(antibiotic_names=["X", "Y", "Z"])
        library = OptionLibrary(env=env)

        assert "X" in library.abx_name_to_index
        assert "Y" in library.abx_name_to_index
        assert "Z" in library.abx_name_to_index
        assert "no_treatment" in library.abx_name_to_index

    def test_init_missing_reward_calculator_raises(self):
        """Test that init raises ValueError if environment lacks reward_calculator."""
        # Create a mock without unwrapped.reward_calculator
        class BadEnv:
            class Unwrapped:
                pass
            unwrapped = Unwrapped()

        with pytest.raises(ValueError, match="reward_calculator.abx_name_to_index"):
            OptionLibrary(env=BadEnv())


class TestAddOption:
    """Tests for OptionLibrary.add_option()."""

    def test_add_single_option(self):
        """Test adding a single option to library."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)
        option = create_test_option(name="opt1")

        library.add_option(option=option)

        assert len(library) == 1
        assert "opt1" in library.options

    def test_add_multiple_options(self):
        """Test adding multiple options to library."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)

        for i in range(3):
            option = create_test_option(name=f"opt{i}")
            library.add_option(option=option)

        assert len(library) == 3
        for i in range(3):
            assert f"opt{i}" in library.options

    def test_add_duplicate_name_raises(self):
        """Test that adding option with duplicate name raises ValueError."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)
        option1 = create_test_option(name="duplicate")
        option2 = create_test_option(name="duplicate")

        library.add_option(option=option1)

        with pytest.raises(ValueError, match="already exists"):
            library.add_option(option=option2)

    def test_add_non_option_raises(self):
        """Test that adding non-OptionBase instance raises TypeError."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)

        with pytest.raises(TypeError, match="must be OptionBase instance"):
            library.add_option(option="not_an_option")  # type: ignore


class TestOptionRetrieval:
    """Tests for retrieving options from library."""

    def test_getitem_by_name(self):
        """Test retrieving option by name using __getitem__."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)
        option = create_test_option(name="test_opt")
        library.add_option(option=option)

        retrieved = library["test_opt"]

        assert retrieved is option
        assert retrieved.name == "test_opt"

    def test_getitem_invalid_name_raises(self):
        """Test that accessing non-existent option by name raises KeyError."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)

        with pytest.raises(KeyError, match="not in library"):
            _ = library["nonexistent"]

    def test_get_option_by_index(self):
        """Test retrieving option by integer index using get_option()."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)
        options = [create_test_option(name=f"opt{i}") for i in range(3)]

        for opt in options:
            library.add_option(option=opt)

        # Get by index (order determined by insertion)
        retrieved0 = library.get_option(option_id=0)
        retrieved1 = library.get_option(option_id=1)
        retrieved2 = library.get_option(option_id=2)

        assert retrieved0.name == "opt0"
        assert retrieved1.name == "opt1"
        assert retrieved2.name == "opt2"

    def test_get_option_invalid_index_raises(self):
        """Test that get_option with out-of-range index raises IndexError."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)
        option = create_test_option(name="opt")
        library.add_option(option=option)

        with pytest.raises(IndexError, match="out of range"):
            library.get_option(option_id=5)

    def test_get_option_negative_index_raises(self):
        """Test that get_option with negative index raises IndexError."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)
        option = create_test_option(name="opt")
        library.add_option(option=option)

        with pytest.raises(IndexError, match="out of range"):
            library.get_option(option_id=-1)

    def test_len(self):
        """Test library length with __len__."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)

        assert len(library) == 0

        for i in range(5):
            library.add_option(option=create_test_option(name=f"opt{i}"))

        assert len(library) == 5


class TestValidationEnvironmentCompatibility:
    """Tests for OptionLibrary.validate_environment_compatibility()."""

    def test_validate_empty_library_raises(self):
        """Test that validating empty library raises ValueError."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        library = OptionLibrary(env=env)

        with pytest.raises(ValueError, match="is empty"):
            library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_single_compatible_option(self):
        """Test validation passes for compatible option."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        library = OptionLibrary(env=env)
        option = create_test_option(
            name="compatible",
            requires_observation_attributes=["prob_infected"],
        )
        library.add_option(option=option)

        # Should not raise
        library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_missing_patient_attribute_raises(self):
        """Test that missing patient attribute raises ValueError."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        library = OptionLibrary(env=env)
        option = create_test_option(
            name="requires_unknown",
            requires_observation_attributes=["nonexistent_attribute"],
        )
        library.add_option(option=option)

        with pytest.raises(ValueError, match="requires patient attributes"):
            library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_missing_antibiotic_raises(self):
        """Test that missing antibiotic raises ValueError."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        library = OptionLibrary(env=env)
        library.add_option(
            option=create_test_option(
                name="bad",
                referenced_antibiotics=["NONEXISTENT"],
            )
        )

        with pytest.raises(ValueError, match="not in environment's action space"):
            library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_no_treatment_special_case(self):
        """Test validation handles 'no_treatment' antibiotic correctly."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        library = OptionLibrary(env=env)
        library.add_option(
            option=create_test_option(
                name="no_rx",
                referenced_antibiotics=["no_treatment"],
            )
        )

        # Should not raise; no_treatment is always available
        library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_invalid_no_treatment_variant_raises(self):
        """Test that invalid no_treatment variant raises helpful ValueError."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        library = OptionLibrary(env=env)
        library.add_option(
            option=create_test_option(
                name="bad",
                referenced_antibiotics=["NO_RX"],
            )
        )

        with pytest.raises(ValueError, match="NO_RX"):
            library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_option_not_implementing_get_referenced_antibiotics_raises(self):
        """Test that option without get_referenced_antibiotics() raises."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()

        class BadOption(OptionBase):
            def __init__(self):
                super().__init__(name="bad")

            def decide(self, env_state: dict) -> np.ndarray:
                num_patients = env_state.get("num_patients", 1)
                return np.zeros(shape=num_patients, dtype=np.int32)

            def get_referenced_antibiotics(self) -> list:
                raise NotImplementedError("Not implemented for test")

        library = OptionLibrary(env=env)
        library.add_option(option=BadOption())

        with pytest.raises(ValueError, match="does not implement get_referenced_antibiotics"):
            library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_requires_amr_levels_with_amr(self):
        """Test validation passes when option requires AMR and env has it."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        option = create_test_option(
            name="needs_amr",
            requires_amr_levels=True,
        )
        library = OptionLibrary(env=env)
        library.add_option(option=option)

        # Should not raise; mock environment has amr_balloon_models
        library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_missing_patient_generator_attribute_raises(self):
        """Test that missing patient_generator.visible_patient_attributes raises."""
        env = create_mock_environment(antibiotic_names=["A"])

        class BadPG:
            pass

        library = OptionLibrary(env=env)
        library.add_option(option=create_test_option(name="opt"))

        with pytest.raises(ValueError, match="visible_patient_attributes"):
            library.validate_environment_compatibility(env=env, patient_generator=BadPG())

    def test_validate_missing_prob_infected_raises(self):
        """Test that missing prob_infected in visible attributes raises."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        pg.visible_patient_attributes = []

        library = OptionLibrary(env=env)
        library.add_option(option=create_test_option(name="opt"))

        with pytest.raises(ValueError, match="prob_infected"):
            library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_no_antibiotics_in_environment_raises(self):
        """Test that environment with no antibiotics raises ValueError."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()

        # Manually clear the antibiotic mapping
        library = OptionLibrary(env=env)
        library.abx_name_to_index = {}  # Empty mapping

        library.add_option(option=create_test_option(name="opt"))

        with pytest.raises(ValueError, match="no antibiotics configured"):
            library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_requires_step_number_with_step(self):
        """Test validation passes when option requires step and env has it."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        option = create_test_option(
            name="needs_step",
            requires_step_number=True,
        )
        library = OptionLibrary(env=env)
        library.add_option(option=option)

        # Should not raise; mock environment has current_time_step
        library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_requires_amr_levels_missing_amr_raises(self):
        """Test that missing AMR tracking raises ValueError."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        option = create_test_option(
            name="needs_amr",
            requires_amr_levels=True,
        )
        library = OptionLibrary(env=env)
        library.add_option(option=option)

        if hasattr(env.unwrapped, "amr_balloon_models"):
            delattr(env.unwrapped, "amr_balloon_models")

        with pytest.raises(ValueError, match="requires AMR levels"):
            library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_requires_step_number_missing_step_raises(self):
        """Test that missing current_time_step raises ValueError."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        option = create_test_option(
            name="needs_step",
            requires_step_number=True,
        )
        library = OptionLibrary(env=env)
        library.add_option(option=option)

        if hasattr(env.unwrapped, "current_time_step"):
            delattr(env.unwrapped, "current_time_step")

        with pytest.raises(ValueError, match="requires step number"):
            library.validate_environment_compatibility(env=env, patient_generator=pg)

    def test_validate_injects_observable_attributes(self):
        """Test that validation injects observable attributes into options."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        pg.visible_patient_attributes = ["prob_infected", "benefit_value_multiplier"]
        library = OptionLibrary(env=env)
        option = ObservableOption(name="observable")
        library.add_option(option=option)

        library.validate_environment_compatibility(env=env, patient_generator=pg)

        assert option.observed_attributes == ["prob_infected", "benefit_value_multiplier"]

    def test_validate_with_termination_condition_flag(self):
        """Test validation accepts options that provide termination condition."""
        env = create_mock_environment(antibiotic_names=["A"])
        pg = create_mock_patient_generator()
        option = create_test_option(
            name="terminating",
            provides_termination=True,
        )
        library = OptionLibrary(env=env)
        library.add_option(option=option)

        library.validate_environment_compatibility(env=env, patient_generator=pg)


class TestLibraryIntegration:
    """Integration tests for OptionLibrary with multiple operations."""

    def test_full_workflow(self):
        """Test complete workflow: init, add options, validate, retrieve."""
        env = create_mock_environment(antibiotic_names=["A", "B"])
        pg = create_mock_patient_generator()
        library = OptionLibrary(env=env, name="integration_test")

        # Add multiple options
        opts = [
            create_test_option(name="opt1", requires_observation_attributes=["prob_infected"]),
            create_test_option(name="opt2", requires_observation_attributes=["prob_infected"]),
            create_test_option(name="opt3", requires_observation_attributes=["prob_infected"]),
        ]
        for opt in opts:
            library.add_option(option=opt)

        # Validate all compatible
        library.validate_environment_compatibility(env=env, patient_generator=pg)

        # Retrieve and verify
        assert len(library) == 3
        assert library["opt1"].name == "opt1"
        assert library.get_option(1).name == "opt2"

    def test_library_preserves_insertion_order(self):
        """Test that library preserves option insertion order."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)

        names = ["first", "second", "third", "fourth", "fifth"]
        for name in names:
            library.add_option(option=create_test_option(name=name))

        retrieved_names = [library.get_option(i).name for i in range(len(library))]
        assert retrieved_names == names

    def test_list_options_returns_ordered_names(self):
        """Test list_options returns names in insertion order."""
        env = create_mock_environment(antibiotic_names=["A"])
        library = OptionLibrary(env=env)

        names = ["alpha", "beta", "gamma"]
        for name in names:
            library.add_option(option=create_test_option(name=name))

        assert library.list_options() == names

    def test_to_dict_serializes_options(self):
        """Test to_dict returns expected serialized structure."""
        env = create_mock_environment(antibiotic_names=["A", "B"])
        library = OptionLibrary(env=env, name="serialize_test")

        library.add_option(option=create_test_option(name="finite", k=3))
        library.add_option(option=create_test_option(name="infinite", k=float("inf")))

        data = library.to_dict()

        assert data["name"] == "serialize_test"
        assert data["num_options"] == 2
        assert data["antibiotic_names"] == ["A", "B", "no_treatment"]
        assert data["options"][0]["name"] == "finite"
        assert data["options"][0]["k"] == 3
        assert data["options"][1]["name"] == "infinite"
        assert data["options"][1]["k"] == "inf"
