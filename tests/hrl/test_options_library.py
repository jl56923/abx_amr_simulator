"""Unit tests for OptionLibrary class."""

import pytest
import numpy as np

from abx_amr_simulator.hrl import OptionBase, OptionLibrary
# Import test helpers (sys.path configured in tests/conftest.py)
from test_reference_helpers import create_mock_environment  # type: ignore[import-not-found]


class SimpleOption(OptionBase):
    """Simple concrete option for testing that uses correct action mapping."""
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False

    def decide(self, env_state):
        """Decide method that returns correct no_treatment index for all patients.
        
        Updated to return the canonical no_treatment index (last action) instead of 0,
        to work with the enhanced runtime validation that verifies semantic correctness.
        """
        num_patients = env_state['num_patients']
        # Use the canonical mapping to get no_treatment index
        no_treatment_index = env_state['reward_calculator'].abx_name_to_index['no_treatment']
        return np.full(num_patients, no_treatment_index, dtype=np.int32)
    
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

    def test_validation_fails_when_no_treatment_not_last(self):
        """Test validation fails if RewardCalculator maps no_treatment incorrectly."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', k=5))

        # Corrupt mapping to simulate a bad no_treatment index
        env.reward_calculator.abx_name_to_index['no_treatment'] = 0

        with pytest.raises(ValueError, match="no_treatment"):
            lib.validate_environment_compatibility(env=env, patient_generator=env.unwrapped.patient_generator)

    def test_validation_fails_when_mapping_mismatch(self):
        """Test validation fails if OptionLibrary mapping diverges from RewardCalculator."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', k=5))

        # Corrupt option library mapping without touching RewardCalculator
        lib.abx_name_to_index = dict(env.reward_calculator.abx_name_to_index)
        lib.abx_name_to_index['A'] = 1

        with pytest.raises(ValueError, match="action mapping"):
            lib.validate_environment_compatibility(env=env, patient_generator=env.unwrapped.patient_generator)

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
                num_patients = env_state['num_patients']
                no_treatment_index = env_state['reward_calculator'].abx_name_to_index['no_treatment']
                return np.full(num_patients, no_treatment_index, dtype=np.int32)
            
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
                num_patients = env_state['num_patients']
                no_treatment_index = env_state['reward_calculator'].abx_name_to_index['no_treatment']
                return np.full(num_patients, no_treatment_index, dtype=np.int32)

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


class TestRuntimeActionIndexValidation:
    """Test runtime validation of action index mapping correctness.
    
    These tests verify that the enhanced validation catches options that build
    incorrect action_to_index mappings internally, which would cause semantic
    errors (e.g., prescribing antibiotic A when deciding no_treatment).
    """

    class BuggyActionMappingOption(OptionBase):
        """Option that builds incorrect action_to_index mapping (no_treatment at index 0).
        
        This mimics the bug in UncertaintyModulatedHeuristicWorker where the option
        builds its own action mapping with no_treatment first, conflicting with the
        canonical mapping where no_treatment is last.
        """
        REQUIRES_OBSERVATION_ATTRIBUTES = []
        REQUIRES_AMR_LEVELS = False
        
        def decide(self, env_state):
            """Build buggy action mapping and return actions."""
            num_patients = env_state['num_patients']
            
            # Build INCORRECT mapping (no_treatment at index 0)
            antibiotic_names = [
                abx for abx in env_state['option_library'].abx_name_to_index.keys()
                if abx != 'no_treatment'
            ]
            action_keys = ['no_treatment'] + [f'prescribe_{abx}' for abx in antibiotic_names]
            action_to_index = {action: i for i, action in enumerate(action_keys)}
            
            # Always choose no_treatment using buggy mapping (returns 0)
            return np.full(num_patients, action_to_index['no_treatment'], dtype=np.int32)
        
        def get_referenced_antibiotics(self):
            return []  # No specific antibiotics
    
    class CorrectActionMappingOption(OptionBase):
        """Option that uses canonical action mapping correctly."""
        REQUIRES_OBSERVATION_ATTRIBUTES = []
        REQUIRES_AMR_LEVELS = False
        
        def decide(self, env_state):
            """Use correct action mapping from reward_calculator."""
            num_patients = env_state['num_patients']
            # Use canonical mapping (no_treatment is LAST)
            no_treatment_index = env_state['reward_calculator'].abx_name_to_index['no_treatment']
            return np.full(num_patients, no_treatment_index, dtype=np.int32)
        
        def get_referenced_antibiotics(self):
            return []
    
    class OutOfRangeOption(OptionBase):
        """Option that returns invalid action indices."""
        REQUIRES_OBSERVATION_ATTRIBUTES = []
        REQUIRES_AMR_LEVELS = False
        
        def decide(self, env_state):
            """Return out-of-range action indices."""
            num_patients = env_state['num_patients']
            # Return index 999 which is way out of range
            return np.full(num_patients, 999, dtype=np.int32)
        
        def get_referenced_antibiotics(self):
            return []
    
    class WrongShapeOption(OptionBase):
        """Option that returns wrong shape."""
        REQUIRES_OBSERVATION_ATTRIBUTES = []
        REQUIRES_AMR_LEVELS = False
        
        def decide(self, env_state):
            """Return actions with wrong shape."""
            # Return wrong number of actions
            return np.array([0], dtype=np.int32)  # Only 1 action instead of num_patients
        
        def get_referenced_antibiotics(self):
            return []
    
    def test_validation_catches_buggy_action_mapping(self):
        """Test that runtime validation catches options with incorrect action index mapping.
        
        This is the key test - it should catch the UncertaintyModulatedHeuristicWorker bug
        where the option builds its own action_to_index with no_treatment at index 0,
        but the environment expects no_treatment at the last index.
        """
        env = create_mock_environment(
            antibiotic_names=['A'],  # Single antibiotic: A=0, no_treatment=1
            num_patients_per_time_step=3
        )
        lib = OptionLibrary(env=env)
        lib.add_option(self.BuggyActionMappingOption(name='buggy_opt', k=10))
        
        # Validation should FAIL with clear semantic error message
        with pytest.raises(ValueError) as exc_info:
            lib.validate_environment_compatibility(
                env=env,
                patient_generator=env.unwrapped.patient_generator
            )
        
        error_msg = str(exc_info.value)
        assert 'buggy_opt' in error_msg
        assert 'SEMANTIC ERROR' in error_msg
        assert 'no_treatment' in error_msg
        # Should mention the index mismatch
        assert '0' in error_msg  # What option returned
        assert '1' in error_msg  # What it should be for single-abx case
        # Should suggest the fix
        assert 'reward_calculator.abx_name_to_index' in error_msg or 'action_to_index' in error_msg
    
    def test_validation_passes_correct_action_mapping(self):
        """Test that validation passes for options using correct action mapping."""
        env = create_mock_environment(
            antibiotic_names=['A', 'B'],
            num_patients_per_time_step=3
        )
        lib = OptionLibrary(env=env)
        lib.add_option(self.CorrectActionMappingOption(name='correct_opt', k=10))
        
        # Should pass without errors
        lib.validate_environment_compatibility(
            env=env,
            patient_generator=env.unwrapped.patient_generator
        )
    
    def test_validation_catches_out_of_range_indices(self):
        """Test that validation catches action indices outside valid range."""
        env = create_mock_environment(
            antibiotic_names=['A', 'B'],
            num_patients_per_time_step=3
        )
        lib = OptionLibrary(env=env)
        lib.add_option(self.OutOfRangeOption(name='bad_range', k=5))
        
        with pytest.raises(ValueError) as exc_info:
            lib.validate_environment_compatibility(
                env=env,
                patient_generator=env.unwrapped.patient_generator
            )
        
        error_msg = str(exc_info.value)
        assert 'bad_range' in error_msg
        assert 'invalid action indices' in error_msg.lower() or 'out of range' in error_msg.lower()
        assert '999' in error_msg  # The invalid index
    
    def test_validation_catches_wrong_shape(self):
        """Test that validation catches actions with wrong shape."""
        env = create_mock_environment(
            antibiotic_names=['A'],
            num_patients_per_time_step=3
        )
        lib = OptionLibrary(env=env)
        lib.add_option(self.WrongShapeOption(name='wrong_shape', k=5))
        
        with pytest.raises(ValueError) as exc_info:
            lib.validate_environment_compatibility(
                env=env,
                patient_generator=env.unwrapped.patient_generator
            )
        
        error_msg = str(exc_info.value)
        assert 'wrong_shape' in error_msg
        assert 'shape' in error_msg.lower()
    
    def test_validation_two_antibiotics_buggy_mapping(self):
        """Test semantic validation with two antibiotics (more complex case)."""
        env = create_mock_environment(
            antibiotic_names=['A', 'B'],  # A=0, B=1, no_treatment=2
            num_patients_per_time_step=3
        )
        lib = OptionLibrary(env=env)
        lib.add_option(self.BuggyActionMappingOption(name='buggy_two_abx', k=10))
        
        # Should catch that option returns 0 for no_treatment when it should be 2
        with pytest.raises(ValueError) as exc_info:
            lib.validate_environment_compatibility(
                env=env,
                patient_generator=env.unwrapped.patient_generator
            )
        
        error_msg = str(exc_info.value)
        assert 'buggy_two_abx' in error_msg
        assert 'SEMANTIC ERROR' in error_msg
        assert '2' in error_msg  # Correct no_treatment index for 2-abx case

