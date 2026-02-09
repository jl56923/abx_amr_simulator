"""Unit tests for OptionLibrary class."""

import pytest
import numpy as np

from abx_amr_simulator.hrl import OptionBase, OptionLibrary
# Import test helpers (sys.path configured in tests/conftest.py)
from test_reference_helpers import create_mock_environment  # type: ignore[import-not-found]


class SimpleOption(OptionBase):
    """Simple concrete option for testing."""
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False

    def decide(self, env_state):
        """Decide method matching new signature (no antibiotic_names param)."""
        num_patients = env_state['num_patients']
        return np.zeros(num_patients, dtype=np.int32)
    
    def get_referenced_antibiotics(self):
        """Test option returns no specific antibiotics."""
        return []


class TestOptionLibraryInit:
    """Test OptionLibrary initialization."""

    def test_init_default_name(self):
        """Test initialization with default name."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        assert lib.name == 'default'
        assert len(lib) == 0

    def test_init_with_custom_name(self):
        """Test initialization with custom name."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env, name='my_library')
        assert lib.name == 'my_library'


class TestOptionLibraryAddOption:
    """Test adding options to library."""

    def test_add_single_option(self):
        """Test adding a single option."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        opt = SimpleOption(name='opt1', k=5)
        lib.add_option(opt)
        assert len(lib) == 1

    def test_add_multiple_options(self):
        """Test adding multiple options."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        for i in range(3):
            opt = SimpleOption(name=f'opt{i}', k=5)
            lib.add_option(opt)
        assert len(lib) == 3

    def test_add_duplicate_option_name(self):
        """Test that duplicate option names raise error."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', k=5))
        
        with pytest.raises(ValueError):
            lib.add_option(SimpleOption(name='opt1', k=10))

    def test_add_non_option_base_raises_error(self):
        """Test that adding non-OptionBase raises TypeError."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        
        with pytest.raises(TypeError):
            lib.add_option({'not': 'an_option'})  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            lib.add_option(None)  # type: ignore[arg-type]


class TestOptionLibraryGetOption:
    """Test retrieving options from library."""

    def test_get_option_by_index(self):
        """Test get_option() by index."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        opt1 = SimpleOption(name='opt1', k=5)
        opt2 = SimpleOption(name='opt2', k=10)
        lib.add_option(opt1)
        lib.add_option(opt2)

        retrieved1 = lib.get_option(0)
        retrieved2 = lib.get_option(1)
        
        assert retrieved1 is opt1
        assert retrieved2 is opt2

    def test_get_option_out_of_range(self):
        """Test get_option() with invalid index."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', k=5))

        with pytest.raises(IndexError):
            lib.get_option(10)

        with pytest.raises(IndexError):
            lib.get_option(-1)

    def test_get_option_empty_library(self):
        """Test get_option() on empty library."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        with pytest.raises(IndexError):
            lib.get_option(0)


class TestOptionLibraryItemAccess:
    """Test dictionary-style access."""

    def test_getitem_by_name(self):
        """Test retrieving option by name using __getitem__."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        opt = SimpleOption(name='my_option', k=5)
        lib.add_option(opt)

        retrieved = lib['my_option']
        assert retrieved is opt

    def test_getitem_nonexistent_name(self):
        """Test __getitem__ with nonexistent name."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        
        with pytest.raises(KeyError):
            lib['nonexistent']


class TestOptionLibraryListOptions:
    """Test list_options() method."""

    def test_list_options_empty(self):
        """Test list_options on empty library."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        assert lib.list_options() == []

    def test_list_options_ordered(self):
        """Test that list_options returns options in order."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        names = ['opt1', 'opt2', 'opt3']
        for name in names:
            lib.add_option(SimpleOption(name=name, k=5))

        result = lib.list_options()
        assert result == names


class TestOptionLibraryToDict:
    """Test to_dict() serialization."""

    def test_to_dict_structure(self):
        """Test that to_dict returns expected structure."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env, name='test_lib')
        opt1 = SimpleOption(name='opt1', k=5)
        opt2 = SimpleOption(name='opt2', k=float('inf'))
        lib.add_option(opt1)
        lib.add_option(opt2)

        result = lib.to_dict()

        assert result['name'] == 'test_lib'
        assert result['num_options'] == 2
        assert len(result['options']) == 2
        assert result['options'][0]['name'] == 'opt1'
        assert result['options'][0]['k'] == 5
        assert result['options'][1]['k'] == 'inf'

    def test_to_dict_with_empty_library(self):
        """Test to_dict on empty library."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        result = lib.to_dict()
        
        assert result['num_options'] == 0
        assert result['options'] == []


class TestOptionLibraryValidation:
    """Test validate_environment_compatibility method."""

    class BlockOption(OptionBase):
        """Test implementation of BlockOption for validation tests."""
        REQUIRES_OBSERVATION_ATTRIBUTES = []
        REQUIRES_AMR_LEVELS = False
        REQUIRES_STEP_NUMBER = False

        def __init__(self, name: str, antibiotic: str, k: int):
            super().__init__(name=name, k=k)
            self.antibiotic = antibiotic

        def decide(self, env_state):
            num_patients = env_state['num_patients']
            option_library = env_state.get('option_library')
            if option_library is None:
                raise ValueError("option_library not found in env_state")
            
            abx_name_to_index = option_library.abx_name_to_index
            antibiotic_names = list(abx_name_to_index.keys())
            
            try:
                action_idx = antibiotic_names.index(self.antibiotic)
            except ValueError:
                raise ValueError(
                    f"Antibiotic '{self.antibiotic}' not in {antibiotic_names}"
                )
            return np.full(num_patients, action_idx, dtype=np.int32)
        
        def get_referenced_antibiotics(self):
            """Return the single antibiotic this option uses."""
            return [self.antibiotic]

    def test_validation_with_compatible_env(self):
        """Test validation passes with compatible environment."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', k=5))

        # Should not raise - validate against the same env used for initialization
        lib.validate_environment_compatibility(env=env, patient_generator=env.unwrapped.patient_generator)
        # Check that abx_name_to_index was cached correctly (includes no_treatment)
        assert 'A' in lib.abx_name_to_index
        assert 'B' in lib.abx_name_to_index
        assert 'no_treatment' in lib.abx_name_to_index

    def test_validation_empty_library_raises(self):
        """Test validation fails on empty library."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)

        with pytest.raises(ValueError):
            lib.validate_environment_compatibility(env=env, patient_generator=env.unwrapped.patient_generator)

    def test_validation_requires_amr_levels(self):
        """Test validation passes when option requires AMR and env has it."""
        class RequiresAMROption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = True

            def decide(self, env_state):
                return np.zeros(1, dtype=np.int32)
            
            def get_referenced_antibiotics(self):
                return []

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(RequiresAMROption(name='opt1', k=5))

        # The real environment HAS leaky_balloons (AMR tracking), so validation should pass
        lib.validate_environment_compatibility(env=env, patient_generator=env.unwrapped.patient_generator)


    def test_validation_missing_patient_attributes(self):
        """Test validation fails when option requires unavailable attributes."""
        class RequiresAttrOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = ['nonexistent_attribute']
            
            REQUIRES_AMR_LEVELS = False

            def get_referenced_antibiotics(self):
                return []

            def decide(self, env_state):
                return np.zeros(1, dtype=np.int32)

        env = create_mock_environment(
            antibiotic_names=['A', 'B'], 
            num_patients_per_time_step=1,
            visible_patient_attributes=['prob_infected']  # Only has prob_infected
        )
        lib = OptionLibrary(env=env)
        lib.add_option(RequiresAttrOption(name='opt1', k=5))

        with pytest.raises(ValueError) as exc_info:
            lib.validate_environment_compatibility(env=env, patient_generator=env.unwrapped.patient_generator)
        
        assert 'nonexistent_attribute' in str(exc_info.value)

    def test_validation_requires_step_number(self):
        """Test validation passes when option requires step number and env has it."""
        class RequiresStepOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = False
            REQUIRES_STEP_NUMBER = True
            
            def get_referenced_antibiotics(self):
                return []

            def decide(self, env_state):
                return np.zeros(1, dtype=np.int32)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(RequiresStepOption(name='opt1', k=5))

        # The real environment HAS current_time_step attribute, so validation should pass
        lib.validate_environment_compatibility(env=env, patient_generator=env.unwrapped.patient_generator)

    def test_validation_invalid_antibiotic_name(self):
        """Test validation fails when option references non-existent antibiotic.
        
        This test ensures that antibiotic name mismatches (like 'Antibiotic_A' vs 'A')
        are caught early with clear error messages, preventing silent failures.
        """
        env = create_mock_environment(
            antibiotic_names=['A', 'B'],  # Environment has A and B
            num_patients_per_time_step=1
        )
        lib = OptionLibrary(env=env)
        # Option tries to use 'Antibiotic_A' which doesn't exist
        lib.add_option(self.BlockOption(name='InvalidAbx', antibiotic='Antibiotic_A', k=5))
        
        with pytest.raises(ValueError) as exc_info:
            lib.validate_environment_compatibility(
                env=env, 
                patient_generator=env.unwrapped.patient_generator
            )
        
        # Verify error message contains all relevant information
        error_msg = str(exc_info.value)
        assert 'Antibiotic_A' in error_msg
        assert 'not in environment' in error_msg or 'not in' in error_msg
        # Should show available antibiotics for debugging
        assert 'A' in error_msg and 'B' in error_msg


