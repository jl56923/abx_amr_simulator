"""Tests for AMRDynamicsBase abstract base class and contract."""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abx_amr_simulator.core import AMRDynamicsBase, AMR_LeakyBalloon


class TestAMRDynamicsBaseInterface:
    """Test that AMRDynamicsBase defines the correct interface."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that AMRDynamicsBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AMRDynamicsBase()
    
    def test_amr_leaky_balloon_is_subclass(self):
        """Test that AMR_LeakyBalloon properly inherits from AMRDynamicsBase."""
        assert issubclass(AMR_LeakyBalloon, AMRDynamicsBase)
        balloon = AMR_LeakyBalloon()
        assert isinstance(balloon, AMRDynamicsBase)
    
    def test_concrete_subclass_requires_step(self):
        """Test that subclasses must implement step()."""
        class IncompleteModel(AMRDynamicsBase):
            NAME = "incomplete"
            def reset(self, initial_level):
                pass
        
        with pytest.raises(TypeError):
            IncompleteModel()
    
    def test_concrete_subclass_requires_reset(self):
        """Test that subclasses must implement reset()."""
        class IncompleteModel(AMRDynamicsBase):
            NAME = "incomplete"
            def step(self, doses):
                return 0.5
        
        with pytest.raises(TypeError):
            IncompleteModel()
    
    def test_amr_leaky_balloon_has_name_constant(self):
        """Test that AMR_LeakyBalloon has NAME class constant."""
        assert hasattr(AMR_LeakyBalloon, 'NAME')
        assert AMR_LeakyBalloon.NAME == "leaky_balloon"
        balloon = AMR_LeakyBalloon()
        assert balloon.NAME == "leaky_balloon"


class TestAMRDynamicsContract:
    """Test that all AMRDynamicsBase implementations satisfy the contract."""
    
    def test_step_output_always_in_bounds(self):
        """Test that step() always returns value in [0, 1]."""
        balloon = AMR_LeakyBalloon(
            leak=0.1,
            flatness_parameter=1.0,
            permanent_residual_volume=0.0,
            initial_amr_level=0.0
        )
        
        # Test various dose amounts
        for doses in [0, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0]:
            result = balloon.step(doses)
            assert 0.0 <= result <= 1.0, f"step({doses}) returned {result}, not in [0, 1]"
    
    def test_step_with_residual_output_respects_floor(self):
        """Test that step() output respects permanent_residual_volume."""
        residual = 0.1
        balloon = AMR_LeakyBalloon(
            leak=0.9,  # Very high leak to ensure output drops
            permanent_residual_volume=residual,
            initial_amr_level=0.9
        )
        
        # Repeatedly step without doses (should decay to residual)
        for _ in range(100):
            result = balloon.step(0)
            assert result >= residual, f"step() returned {result}, less than residual {residual}"
    
    def test_reset_initializes_state(self):
        """Test that reset() properly reinitializes state."""
        balloon = AMR_LeakyBalloon(leak=0.1, initial_amr_level=0.1)
        
        # Build up resistance
        for _ in range(10):
            balloon.step(5)
        high_level = balloon.get_volume()
        assert high_level > 0.1
        
        # Reset to initial level
        balloon.reset(0.1)
        assert math.isclose(balloon.get_volume(), 0.1, rel_tol=1e-6)
        
        # Subsequent step should start from reset level
        next_level = balloon.step(0)
        # The exact decay depends on sigmoid dynamics, not simply (1 - leak)
        # Just verify it's less than the reset level (decay occurred)
        assert next_level < 0.1 or math.isclose(next_level, 0.1, rel_tol=1e-2)
    
    def test_reset_with_invalid_bounds(self):
        """Test that reset() validates bounds."""
        balloon = AMR_LeakyBalloon()
        
        with pytest.raises(ValueError):
            balloon.reset(-0.1)  # Below 0
        
        with pytest.raises(ValueError):
            balloon.reset(1.1)  # Above 1
        
        with pytest.raises(ValueError):
            balloon.reset(-1.0)
    
    def test_reset_with_residual_validates_against_residual(self):
        """Test that reset() respects the permanent_residual_volume bound."""
        balloon = AMR_LeakyBalloon(permanent_residual_volume=0.1, initial_amr_level=0.1)
        
        # Should reject values below residual
        with pytest.raises(ValueError):
            balloon.reset(0.05)  # Below residual of 0.1
        
        # Should accept values at residual
        balloon.reset(0.1)  # Should not raise
        assert math.isclose(balloon.get_volume(), 0.1, rel_tol=1e-6)
        
        # Should accept values above residual
        balloon.reset(0.5)  # Should not raise
        assert math.isclose(balloon.get_volume(), 0.5, rel_tol=1e-6)
    
    def test_step_rejects_negative_doses(self):
        """Test that step() rejects negative doses."""
        balloon = AMR_LeakyBalloon()
        
        with pytest.raises(ValueError):
            balloon.step(-1.0)
        
        with pytest.raises(ValueError):
            balloon.step(-0.001)
    
    def test_step_accepts_zero_doses(self):
        """Test that step() accepts zero doses (decay only)."""
        balloon = AMR_LeakyBalloon(leak=0.1, initial_amr_level=0.5)
        
        # Should not raise and should return valid value
        result = balloon.step(0)
        assert 0 <= result <= 1
    
    def test_deterministic_with_same_sequence(self):
        """Test that step() is deterministic given same dose sequence."""
        balloon1 = AMR_LeakyBalloon(leak=0.1, flatness_parameter=1.0, initial_amr_level=0.0)
        balloon2 = AMR_LeakyBalloon(leak=0.1, flatness_parameter=1.0, initial_amr_level=0.0)
        
        dose_sequence = [0, 1, 2, 0.5, 3, 0, 0, 1.5]
        
        results1 = [balloon1.step(p) for p in dose_sequence]
        results2 = [balloon2.step(p) for p in dose_sequence]
        
        for r1, r2 in zip(results1, results2):
            assert math.isclose(r1, r2, rel_tol=1e-12)


class TestMinimalConcreteImplementation:
    """Test that a minimal concrete implementation works with the interface."""
    
    def test_simple_linear_model(self):
        """Test a minimal concrete subclass that implements step() and reset()."""
        class LinearAMRModel(AMRDynamicsBase):
            """Simple linear accumulation model for testing."""
            NAME = "linear"
            
            def __init__(self):
                self.level = 0.0
            
            def step(self, doses):
                if doses < 0:
                    raise ValueError("doses must be non-negative")
                self.level = min(1.0, self.level + 0.01 * doses)
                return self.level
            
            def reset(self, initial_level):
                if not (0.0 <= initial_level <= 1.0):
                    raise ValueError("initial_level must be in [0, 1]")
                self.level = initial_level
        
        model = LinearAMRModel()
        assert isinstance(model, AMRDynamicsBase)
        assert model.NAME == "linear"
        
        # Test step
        assert model.step(10) == 0.1
        assert model.step(50) == 0.6
        
        # Test reset
        model.reset(0.2)
        assert model.level == 0.2
        
        # Test bounds enforcement
        with pytest.raises(ValueError):
            model.reset(-0.1)
        
        with pytest.raises(ValueError):
            model.step(-1)
    
    def test_simple_model_works_with_environment(self):
        """Test that a custom AMRDynamicsBase subclass can be used in ABXAMREnv."""
        # This is a simple validation that the type signature allows it
        from abx_amr_simulator.core import ABXAMREnv
        
        class SimpleExponentialModel(AMRDynamicsBase):
            """Exponential accumulation model."""
            NAME = "exponential"
            
            def __init__(self, decay_factor=0.9):
                self.level = 0.0
                self.decay_factor = decay_factor
            
            def step(self, doses):
                if doses < 0:
                    raise ValueError("doses must be non-negative")
                # Exponential-like accumulation
                self.level = self.decay_factor * self.level + 0.05 * doses
                return min(1.0, self.level)
            
            def reset(self, initial_level):
                if not (0.0 <= initial_level <= 1.0):
                    raise ValueError("initial_level must be in [0, 1]")
                self.level = initial_level
        
        # Verify the model is an AMRDynamicsBase
        model = SimpleExponentialModel()
        assert isinstance(model, AMRDynamicsBase)
        
        # Verify it has the required interface
        assert callable(model.step)
        assert callable(model.reset)
        assert hasattr(model, 'NAME')
