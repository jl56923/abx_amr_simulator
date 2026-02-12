"""Unit tests for OptionBase abstract class."""

import pytest
import numpy as np
from abx_amr_simulator.hrl import OptionBase


class TestOptionBaseCannotInstantiate:
    """Test that OptionBase is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        """OptionBase should raise TypeError when instantiated directly."""
        with pytest.raises(TypeError):
            OptionBase(name='test', k=5)

    def test_subclass_must_implement_decide(self):
        """Subclass without decide() implementation should fail."""
        with pytest.raises(TypeError):
            class IncompleteOption(OptionBase):
                REQUIRES_OBSERVATION_ATTRIBUTES = []
            
            IncompleteOption(name='test', k=5)


class ConcreteTestOption(OptionBase):
    """Concrete implementation for testing."""
    REQUIRES_OBSERVATION_ATTRIBUTES = ['prob_infected']
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False

    def decide(self, env_state):
        num_patients = env_state['num_patients']
        return np.zeros(num_patients, dtype=np.int32)
    
    def get_referenced_antibiotics(self):
        """Test option returns no specific antibiotics."""
        return []


class TestOptionBaseAttributes:
    """Test OptionBase attributes."""

    def test_init_with_valid_k(self):
        """Test initialization with valid k values."""
        opt1 = ConcreteTestOption(name='test1', k=5)
        assert opt1.name == 'test1'
        assert opt1.k == 5

        opt2 = ConcreteTestOption(name='test2', k=float('inf'))
        assert opt2.k == float('inf')

    def test_init_with_default_k(self):
        """Test initialization with default k=1."""
        opt = ConcreteTestOption(name='test')
        assert opt.k == 1

    def test_init_with_invalid_k(self):
        """Test initialization with invalid k values."""
        with pytest.raises(ValueError):
            ConcreteTestOption(name='test', k=0)

        with pytest.raises(ValueError):
            ConcreteTestOption(name='test', k=-5)

        with pytest.raises(ValueError):
            ConcreteTestOption(name='test', k=1.5)

        with pytest.raises(ValueError):
            ConcreteTestOption(name='test', k='invalid')


class TestOptionBaseClassVariables:
    """Test OptionBase class variables."""

    def test_class_variables_inherited(self):
        """Test that subclass inherits class variables correctly."""
        assert ConcreteTestOption.REQUIRES_OBSERVATION_ATTRIBUTES == ['prob_infected']
        assert ConcreteTestOption.REQUIRES_AMR_LEVELS is False
        assert ConcreteTestOption.REQUIRES_STEP_NUMBER is False
        assert ConcreteTestOption.PROVIDES_TERMINATION_CONDITION is False


class TestOptionBaseDecide:
    """Test decide() method."""

    def test_decide_returns_correct_shape(self):
        """Test that decide() returns correct shape."""
        opt = ConcreteTestOption(name='test', k=5)
        env_state = {
            'patients': [{'prob_infected': 0.8}, {'prob_infected': 0.3}],
            'num_patients': 2,
            'current_amr_levels': {},
        }
        actions = opt.decide(env_state)

        assert isinstance(actions, np.ndarray)
        assert actions.shape == (2,)
        assert actions.dtype == np.int32

    def test_decide_with_different_num_patients(self):
        """Test decide() with different numbers of patients."""
        opt = ConcreteTestOption(name='test')
        
        for n_patients in [1, 2, 5, 10]:
            env_state = {
                'patients': [{'prob_infected': 0.5} for _ in range(n_patients)],
                'num_patients': n_patients,
                'current_amr_levels': {},
            }
            actions = opt.decide(env_state)
            assert actions.shape == (n_patients,)


class TestOptionBaseReset:
    """Test reset() method."""

    def test_reset_does_not_raise(self):
        """Test that reset() can be called without error."""
        opt = ConcreteTestOption(name='test')
        opt.reset()  # Should not raise


class TestOptionBaseTermination:
    """Test should_terminate() method."""

    def test_should_terminate_default_false(self):
        """Test that default should_terminate() returns False."""
        opt = ConcreteTestOption(name='test')
        env_state = {'current_amr_levels': {'A': 0.5}}
        assert opt.should_terminate(env_state) is False

    def test_should_terminate_can_be_overridden(self):
        """Test that should_terminate() can be overridden."""
        class AdaptiveOption(ConcreteTestOption):
            PROVIDES_TERMINATION_CONDITION = True

            def should_terminate(self, env_state):
                amr_a = env_state['current_amr_levels'].get('A', 0.0)
                return amr_a > 0.8

        opt = AdaptiveOption(name='test')
        
        # Below threshold
        assert opt.should_terminate({'current_amr_levels': {'A': 0.5}}) is False
        
        # Above threshold
        assert opt.should_terminate({'current_amr_levels': {'A': 0.9}}) is True


class TestOptionBaseSubclassExample:
    """Test realistic option subclass implementations."""

    class BlockOption(OptionBase):
        """Prescribe same antibiotic to all patients for k steps."""
        REQUIRES_OBSERVATION_ATTRIBUTES = []
        REQUIRES_AMR_LEVELS = False
        REQUIRES_STEP_NUMBER = False

        def __init__(self, name: str, antibiotic: str, k: int):
            super().__init__(name=name, k=k)
            self.antibiotic = antibiotic

        def decide(self, env_state):
            num_patients = env_state['num_patients']
            # Get antibiotic names from option_library in env_state
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

    def test_block_option_prescribe_a(self):
        """Test BlockOption prescribing antibiotic A."""
        # Create mock option_library
        class MockOptionLibrary:
            abx_name_to_index = {'A': 0, 'B': 1, 'C': 2}
        
        opt = self.BlockOption(name='A_5', antibiotic='A', k=5)
        env_state = {
            'num_patients': 3,
            'option_library': MockOptionLibrary()
        }
        actions = opt.decide(env_state)

        assert np.array_equal(actions, [0, 0, 0])

    def test_block_option_prescribe_different_abx(self):
        """Test BlockOption with different antibiotics."""
        class MockOptionLibrary:
            abx_name_to_index = {'A': 0, 'B': 1}
        
        opt_b = self.BlockOption(name='B_10', antibiotic='B', k=10)
        env_state = {
            'num_patients': 2,
            'option_library': MockOptionLibrary()
        }
        actions = opt_b.decide(env_state)

        assert np.array_equal(actions, [1, 1])

    def test_block_option_invalid_antibiotic(self):
        """Test BlockOption with invalid antibiotic."""
        class MockOptionLibrary:
            abx_name_to_index = {'A': 0, 'B': 1}
        
        opt = self.BlockOption(name='X_5', antibiotic='X', k=5)
        env_state = {
            'num_patients': 1,
            'option_library': MockOptionLibrary()
        }
        
        with pytest.raises(ValueError):
            opt.decide(env_state)
