"""Unit tests for OptionLibrary class."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../unit/utils')))

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from abx_amr_simulator.hrl import OptionBase, OptionLibrary
from test_reference_helpers import create_mock_environment


class SimpleOption(OptionBase):
    """Simple concrete option for testing."""
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False

    def decide(self, env_state):
        """Decide method matching new signature (no antibiotic_names param)."""
        num_patients = env_state['num_patients']
        return np.zeros(num_patients, dtype=np.int32)


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
            lib.add_option({'not': 'an_option'})

        with pytest.raises(TypeError):
            lib.add_option(None)


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

    def create_mock_env(self, abx_names=None, has_amr=True, has_step=True):
        """Helper to create mock environment."""
        if abx_names is None:
            abx_names = ['A', 'B']

        env = Mock()
        unwrapped = Mock(spec=['reward_calculator'])
        
        # Set up reward calculator with antibiotics
        reward_calculator = Mock()
        reward_calculator.abx_name_to_index = {
            abx: i for i, abx in enumerate(abx_names)
        }
        
        unwrapped.reward_calculator = reward_calculator
        
        # Set up AMR tracking if needed
        if has_amr:
            unwrapped.leaky_balloons = {
                abx: Mock() for abx in abx_names
            }
        
        # Set up step tracking if needed
        if has_step:
            unwrapped.current_step = 0
        
        env.unwrapped = unwrapped
        return env

    def create_mock_patient_generator(self, visible_attrs=None):
        """Helper to create mock patient generator."""
        if visible_attrs is None:
            visible_attrs = ['prob_infected']

        pg = Mock()
        pg.visible_patient_attributes = visible_attrs
        pg.observe = Mock(return_value=np.array([]))
        pg.obs_dim = Mock(return_value=1)
        return pg

    def test_validation_with_compatible_env(self):
        """Test validation passes with compatible environment."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(SimpleOption(name='opt1', k=5))

        env = self.create_mock_env(abx_names=['A', 'B'])
        pg = self.create_mock_patient_generator(visible_attrs=[])

        # Should not raise
        lib.validate_environment_compatibility(env, pg)
        # Check that abx_name_to_index was cached correctly (includes no_treatment)
        assert 'A' in lib.abx_name_to_index
        assert 'B' in lib.abx_name_to_index
        # Also assert that no_treatment is included
        assert 'no_treatment' in lib.abx_name_to_index

    def test_validation_empty_library_raises(self):
        """Test validation fails on empty library."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        env = self.create_mock_env()
        pg = self.create_mock_patient_generator()

        with pytest.raises(ValueError):
            lib.validate_environment_compatibility(env, pg)

    def test_validation_missing_patient_attributes(self):
        """Test validation fails when option requires unavailable attributes."""
        class RequiresAttrOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = ['benefit_value_multiplier']
            REQUIRES_AMR_LEVELS = False

            def decide(self, env_state):
                return np.zeros(1, dtype=np.int32)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(RequiresAttrOption(name='opt1', k=5))

        env = self.create_mock_env()
        pg = self.create_mock_patient_generator(visible_attrs=[])  # Empty attrs

        with pytest.raises(ValueError) as exc_info:
            lib.validate_environment_compatibility(env, pg)
        
        assert 'benefit_value_multiplier' in str(exc_info.value)

    def test_validation_requires_amr_levels(self):
        """Test validation passes when option requires AMR levels."""
        class RequiresAMROption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = True

            def decide(self, env_state):
                return np.zeros(1, dtype=np.int32)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(RequiresAMROption(name='opt1', k=5))

        env = self.create_mock_env(has_amr=True)
        pg = self.create_mock_patient_generator()

        lib.validate_environment_compatibility(env, pg)

    def test_validation_missing_amr_levels_raises(self):
        """Test validation fails when option requires AMR but env doesn't provide."""
        class RequiresAMROption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = True

            def decide(self, env_state):
                return np.zeros(1, dtype=np.int32)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(RequiresAMROption(name='opt1', k=5))

        env = self.create_mock_env(has_amr=False)
        pg = self.create_mock_patient_generator()

        with pytest.raises(ValueError) as exc_info:
            lib.validate_environment_compatibility(env, pg)
        
        assert 'AMR' in str(exc_info.value)

    def test_validation_requires_step_number(self):
        """Test validation passes when option requires step number."""
        class RequiresStepOption(OptionBase):
            REQUIRES_OBSERVATION_ATTRIBUTES = []
            REQUIRES_AMR_LEVELS = False
            REQUIRES_STEP_NUMBER = True

            def decide(self, env_state):
                return np.zeros(1, dtype=np.int32)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib = OptionLibrary(env=env)
        lib.add_option(RequiresStepOption(name='opt1', k=5))

        env = self.create_mock_env(has_step=True)
        pg = self.create_mock_patient_generator()

        lib.validate_environment_compatibility(env, pg)
