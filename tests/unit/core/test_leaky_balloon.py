"""Unit tests for AMR_LeakyBalloon dynamics."""
import math
import pathlib
import sys

import pytest
import pdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abx_amr_simulator.core import AMR_LeakyBalloon


def test_leak_and_bounds_no_doses():
    # This test verifies that the balloon's latent pressure decays over time
    # due to the leak rate when no doses are added. It checks that the volume
    # after one step matches the expected volume based on the leak formula:
    # pressure_new = pressure_old - leak, and that the output is bounded [0, 1].
    """Test that balloon leaks when no doses are added."""
    balloon = AMR_LeakyBalloon(
        leak=0.2,
        flatness_parameter=1.0,
        permanent_residual_volume=0.0,
        initial_amr_level=0.5
    )
    initial_volume = balloon.get_volume()
    volume_after_step = balloon.step(doses=0)
    
    # After one step with no doses, latent pressure should decay
    # which reduces volume slightly
    assert volume_after_step < initial_volume
    assert 0.0 <= volume_after_step <= 1.0


def test_monotonic_with_doses_and_bounded():
    # This test ensures that adding doses always increases the balloon's volume
    # monotonically, and that the volume stays within the valid bounds [0, 1].
    # It verifies that more doses lead to higher volumes in sequence.
    """Test that balloon volume increases with doses and stays bounded."""
    balloon = AMR_LeakyBalloon(
        leak=0.1,
        flatness_parameter=1.0,
        permanent_residual_volume=0.0,
        initial_amr_level=0.0
    )
    v0 = balloon.get_volume()
    v1 = balloon.step(doses=1)
    v2 = balloon.step(doses=3)
    
    assert v1 > v0
    assert v2 > v1
    assert 0.0 <= v2 <= 1.0


def test_negative_doses_raises():
    # This test verifies that the step() method rejects invalid input (negative doses)
    # by raising a ValueError. Negative doses don't make physical sense in the model.
    """Test that negative doses raise an error."""
    balloon = AMR_LeakyBalloon()
    with pytest.raises(ValueError):
        balloon.step(-1)


def test_reset_restores_initial_amr():
    # This test confirms that calling reset() on the balloon restores the internal
    # latent pressure to produce the specified AMR level, which then maps to the correct
    # volume via the sigmoid function. This is essential for reinitializing the simulation.
    """Test that reset restores the AMR level."""
    balloon = AMR_LeakyBalloon(
        leak=0.1,
        flatness_parameter=1.0,
        permanent_residual_volume=0.0,
        initial_amr_level=0.0
    )
    balloon.step(doses=5)
    balloon.reset(initial_amr_level=0.3)
    
    expected_volume = 0.3
    assert math.isclose(balloon.get_volume(), expected_volume, rel_tol=1e-6, abs_tol=1e-6)


def test_permanent_residual_volume_shifts_baseline():
    # This test verifies that the permanent_residual_volume parameter acts as a minimum
    # baseline volume floor. A balloon with residual should always have a higher volume
    # than one without residual (all else equal), and should never go below the residual.
    """Test that permanent_residual_volume shifts the baseline volume."""
    # Balloon with no permanent residual
    balloon_no_residual = AMR_LeakyBalloon(
        leak=0.1,
        flatness_parameter=1.0,
        permanent_residual_volume=0.0,
        initial_amr_level=0.0
    )
    vol_no_residual = balloon_no_residual.get_volume()
    
    # Balloon with permanent residual
    balloon_with_residual = AMR_LeakyBalloon(
        leak=0.1,
        flatness_parameter=1.0,
        permanent_residual_volume=0.1,
        initial_amr_level=0.1
    )
    vol_with_residual = balloon_with_residual.get_volume()
    
    # Volume with residual should be higher
    assert vol_with_residual > vol_no_residual
    assert vol_with_residual >= 0.1  # Should be at least the residual


def test_flatness_parameter_affects_sigmoid_slope():
    # This test checks that the flatness_parameter controls the steepness of the sigmoid.
    # A smaller flatness_parameter produces a steeper sigmoid curve, so the same doses
    # should increase the volume more than with a larger flatness_parameter.
    """Test that flatness_parameter affects how steeply the sigmoid rises."""
    # Steeper sigmoid (lower flatness parameter)
    balloon_steep = AMR_LeakyBalloon(
        leak=0.05,
        flatness_parameter=2,
        permanent_residual_volume=0.0,
        initial_amr_level=0.3
    )
    
    # Flatter sigmoid (higher flatness parameter)
    balloon_flat = AMR_LeakyBalloon(
        leak=0.05,
        flatness_parameter=10,
        permanent_residual_volume=0.0,
        initial_amr_level=0.3
    )
    
    # Get the initial volumes for each balloon:
    volume_steep_initial = balloon_steep.get_volume()
    volume_flat_initial = balloon_flat.get_volume()
    
    # Print each of these:
    
    puff_sequence = [1, 1]
    
    # Step both balloons with same doses
    for doses in puff_sequence:
        balloon_steep.step(doses=doses)
        balloon_flat.step(doses=doses)
    
    # Get the final volumes for each balloon:
    final_volume_steep = balloon_steep.get_volume()
    final_volume_flat = balloon_flat.get_volume()
    
    print(f"Initial volume steep: {volume_steep_initial}")
    print(f"Final volume steep: {final_volume_steep}")
    print("-----")

    print(f"Initial volume flat: {volume_flat_initial}")
    print(f"Final volume flat: {final_volume_flat}")
    print("-----")
    
    # Calculate the 'slope' for the steep balloon vs. the flat balloon
    delta_v_steep = final_volume_steep - volume_steep_initial
    delta_v_flat = final_volume_flat - volume_flat_initial
    
    print(f"Delta volume steep: {delta_v_steep}")
    print(f"Delta volume flat: {delta_v_flat}")
    
    # pdb.set_trace()
    
    assert delta_v_steep > delta_v_flat


def test_delta_volume_counterfactual():
    # This test verifies that the get_delta_volume_for_counterfactual_num_doses_vs_one_less()
    # method correctly computes the marginal volume change when increasing doses by one.
    # It checks that the delta is a valid float and non-negative.
    """Test delta volume computation for counterfactual doses."""
    balloon = AMR_LeakyBalloon(
        leak=0.1,
        flatness_parameter=1.0,
        permanent_residual_volume=0.0,
        initial_amr_level=0.0
    )
    
    delta = balloon.get_delta_volume_for_counterfactual_num_doses_vs_one_less(num_counterfactual_doses=5)
    assert isinstance(delta, float)
    assert delta >= 0.0  # Delta should be non-negative


def test_delta_volume_zero_doses():
    # This test is an edge case check: when the counterfactual doses is 0, the delta
    # between 0 doses and -1 doses should be 0 (since -1 would be invalid/clamped).
    """Test that delta volume is 0 when counterfactual doses is 0."""
    balloon = AMR_LeakyBalloon()
    delta = balloon.get_delta_volume_for_counterfactual_num_doses_vs_one_less(0)
    assert delta == 0.0

def test_copy_creates_independent_instance():
    """Test that copy() creates an independent copy with same state but separate dynamics.
    
    Verifies that:
    1. Copy starts with the same volume as original
    2. Stepping one copy doesn't affect the original
    3. Both copies maintain independent internal state
    4. After divergent steps, copies have different volumes
    """
    # Create original balloon
    original = AMR_LeakyBalloon(
        leak=0.1,
        flatness_parameter=1.0,
        permanent_residual_volume=0.05,
        initial_amr_level=0.3
    )
    
    # Record original initial state
    original_initial_volume = original.get_volume()
    original_initial_pressure = original.pressure
    
    # Create a copy
    copy = original.copy()
    
    # Verify copy has same initial volume and configuration
    assert copy.get_volume() == original_initial_volume
    assert copy.pressure == original_initial_pressure
    assert copy.leak == original.leak
    assert copy.flatness_parameter == original.flatness_parameter
    assert copy.permanent_residual_volume == original.permanent_residual_volume
    
    # Step the copy multiple times
    copy_volume_after_step1 = copy.step(doses=2.0)
    copy_volume_after_step2 = copy.step(doses=1.0)
    
    # Verify original is unaffected by copy's steps
    assert original.get_volume() == original_initial_volume
    assert original.pressure == original_initial_pressure
    
    # Now step the original
    original_volume_after_step1 = original.step(doses=0.0)  # Just leak, no doses
    
    # Verify copy is unaffected by original's steps
    assert copy.pressure != original.pressure
    assert copy.get_volume() != original_initial_volume
    
    # Original should have leaked (lower volume than initial)
    assert original_volume_after_step1 < original_initial_volume
    
    # Copy should have higher volume (added doses)
    assert copy_volume_after_step1 > original_initial_volume


def test_copy_with_reset_counterfactual():
    """Test the counterfactual usage pattern: copy, reset, then compute delta.
    
    This mimics the environment usage where we:
    1. Copy the balloon model
    2. Reset it to visible AMR level
    3. Compute delta for a hypothetical number of doses
    4. Do NOT modify the original
    """
    # Create original balloon with some pressure
    original = AMR_LeakyBalloon(
        leak=0.2,
        flatness_parameter=1.0,
        permanent_residual_volume=0.0,
        initial_amr_level=0.5
    )
    original_initial_volume = original.get_volume()
    
    # Simulate usage pattern: copy, reset, compute delta
    balloon_copy = original.copy()
    balloon_copy.reset(initial_amr_level=0.3)  # Reset to visible level
    
    # Compute counterfactual delta (how much volume would change with 3 doses vs 2)
    delta_for_3_vs_2_doses = balloon_copy.get_delta_volume_for_counterfactual_num_doses_vs_one_less(3)
    
    # Verify delta is positive (more doses = more volume)
    assert delta_for_3_vs_2_doses > 0.0
    
    # Verify original is completely unaffected
    assert original.get_volume() == original_initial_volume
    assert original.pressure > 0.0  # Original still has original pressure
    
    # Verify copy's pressure is different from original (it was reset)
    assert balloon_copy.pressure != original.pressure


# ============================================================================
# ADDITIONAL TESTS FOR IMPROVED COVERAGE
# ============================================================================

class TestParameterValidation:
    """Comprehensive parameter validation tests."""
    
    def test_invalid_leak_boundaries(self):
        """Test leak parameter at all invalid boundaries."""
        # Below lower bound
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(leak=-0.001)
        
        # At lower bound
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(leak=0.0)
        
        # At upper bound
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(leak=1.0)
        
        # Above upper bound
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(leak=1.001)
    
    def test_invalid_flatness_parameter_boundaries(self):
        """Test flatness_parameter at boundaries."""
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(flatness_parameter=0.0)
        
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(flatness_parameter=-0.5)
    
    def test_invalid_residual_boundaries(self):
        """Test permanent_residual_volume at boundaries."""
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(permanent_residual_volume=-0.001)
        
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(permanent_residual_volume=1.0)
        
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(permanent_residual_volume=1.001)
    
    def test_invalid_initial_amr_boundaries(self):
        """Test initial_amr_level at boundaries."""
        # With residual=0.2
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(permanent_residual_volume=0.2, initial_amr_level=0.19)
        
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(initial_amr_level=1.001)
        
        with pytest.raises(ValueError):
            AMR_LeakyBalloon(initial_amr_level=-0.001)
    
    def test_valid_boundary_combinations(self):
        """Test valid combinations at parameter boundaries."""
        # Valid: initial_amr_level equals residual
        balloon = AMR_LeakyBalloon(permanent_residual_volume=0.3, initial_amr_level=0.3)
        assert balloon.pressure == pytest.approx(0.0, abs=1e-6)
        
        # Valid: initial_amr_level = 1.0
        balloon = AMR_LeakyBalloon(initial_amr_level=1.0)
        assert balloon.pressure > 0.0
        
        # Valid: leak at upper bound (0.999)
        balloon = AMR_LeakyBalloon(leak=0.999)
        assert balloon.leak == 0.999
        
        # Valid: very small leak
        balloon = AMR_LeakyBalloon(leak=0.001)
        assert balloon.leak == 0.001


class TestVolumeMappingComprehensive:
    """Comprehensive tests for sigmoid pressure-to-volume mapping."""
    
    def test_volume_at_zero_pressure_with_various_residuals(self):
        """Test volume at zero pressure is always equal to residual."""
        residuals = [0.0, 0.05, 0.1, 0.2, 0.5]
        for res in residuals:
            balloon = AMR_LeakyBalloon(
                permanent_residual_volume=res,
                initial_amr_level=res  # Start at residual level
            )
            assert balloon.get_volume(pressure=0.0) == pytest.approx(res, abs=1e-6)
    
    def test_volume_monotonic_increasing_comprehensive(self):
        """Test volume strictly increases with pressure across range."""
        balloon = AMR_LeakyBalloon()
        
        pressures = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
        volumes = [balloon.get_volume(pressure=p) for p in pressures]
        
        # Verify strictly increasing
        for i in range(len(volumes) - 1):
            assert volumes[i] < volumes[i + 1], \
                f"Not strictly increasing: {volumes[i]} >= {volumes[i+1]}"
    
    def test_high_pressure_asymptote_near_one(self):
        """Test that volume asymptotes at 1.0 as pressure increases."""
        residual = 0.05
        balloon = AMR_LeakyBalloon(
            permanent_residual_volume=residual,
            initial_amr_level=residual,
            flatness_parameter=1.0
        )
        
        # With rescaled sigmoid, pressure reaches 1.0 volume relatively quickly
        pressures = [10.0, 20.0, 50.0]
        volumes = [balloon.get_volume(pressure=p) for p in pressures]
        
        # All should be very close to 1.0 and strictly bounded by it
        for v in volumes:
            assert v > 0.95
            assert v <= 1.0
        
        # Verify monotonic increase
        assert volumes[0] < volumes[1] < volumes[2]
        
        # Closer to 1.0 for higher pressures
        assert volumes[-1] > volumes[-2] > volumes[-3]
    
    def test_flatness_parameter_slope_effect(self):
        """Test flatness parameter controls sigmoid steepness."""
        # Three balloons with different flatness parameters
        steep = AMR_LeakyBalloon(flatness_parameter=0.5)
        medium = AMR_LeakyBalloon(flatness_parameter=1.0)
        flat = AMR_LeakyBalloon(flatness_parameter=2.0)
        
        # At same pressure, slopes should differ
        pressure = 1.0
        v_steep = steep.get_volume(pressure=pressure)
        v_medium = medium.get_volume(pressure=pressure)
        v_flat = flat.get_volume(pressure=pressure)
        
        # Steeper should give higher volume (further along sigmoid)
        assert v_steep > v_medium > v_flat


class TestDynamicsComprehensive:
    """Comprehensive tests for puff and leak dynamics."""
    
    def test_puff_accumulation_without_leak(self):
        """Test doses accumulate linearly when leak=0 is simulated with very small leak."""
        # Use very small leak to approximate no leak
        balloon = AMR_LeakyBalloon(leak=0.0001, flatness_parameter=1.0)
        pressures = []
        
        for i in range(5):
            balloon.step(doses=1.0)
            pressures.append(balloon.pressure)
        
        # Pressure should increase but not exactly linearly (leak gradually removes)
        for i in range(len(pressures) - 1):
            assert pressures[i] < pressures[i + 1]
    
    def test_linear_decay_with_no_doses(self):
        """Test pressure decays linearly with leak amount."""
        leak_rate = 0.5
        balloon = AMR_LeakyBalloon(leak=leak_rate)
        
        # Build up pressure
        balloon.step(doses=10.0)
        initial_pressure = balloon.pressure
        
        # Record pressure after each step - should decrease by leak amount each step
        pressures = [initial_pressure]
        for _ in range(10):
            balloon.step(doses=0.0)
            pressures.append(balloon.pressure)
        
        # Check linear decay: p_n = p_{n-1} - leak (until pressure hits zero)
        for i in range(1, len(pressures)):
            if pressures[i] > 0:
                # Should decrease by exactly leak amount
                expected_pressure = pressures[i - 1] - leak_rate
                assert pressures[i] == pytest.approx(max(0.0, expected_pressure), abs=1e-6)
    
    def test_long_term_evolution_to_residual(self):
        """Test that without doses, volume converges to residual."""
        residual = 0.15
        balloon = AMR_LeakyBalloon(
            leak=0.3,
            permanent_residual_volume=residual,
            initial_amr_level=0.9
        )
        
        # Run for many steps
        for _ in range(200):
            balloon.step(doses=0.0)
        
        final_volume = balloon.get_volume()
        assert final_volume == pytest.approx(residual, abs=1e-5)
    
    def test_doses_with_leak_equilibrium(self):
        """Test system behavior with constant doses and leak."""
        leak_rate = 0.1
        balloon = AMR_LeakyBalloon(leak=leak_rate)
        
        # Constant puff rate
        constant_doses = 0.2
        
        # Run until equilibrium
        volumes = []
        for _ in range(100):
            v = balloon.step(doses=constant_doses)
            volumes.append(v)
        
        # Last 10 steps should be nearly constant (equilibrium reached)
        last_volumes = volumes[-10:]
        avg_last = sum(last_volumes) / len(last_volumes)
        
        for v in last_volumes:
            assert abs(v - avg_last) < 0.01  # Within 1% of average
    
    def test_fractional_doses(self):
        """Test that fractional puff values work correctly."""
        balloon = AMR_LeakyBalloon(leak=0.001, initial_amr_level=0.1)
        
        v1 = balloon.step(doses=0.5)
        v2 = balloon.step(doses=0.3)
        v3 = balloon.step(doses=0.2)
        
        # All should be valid volumes bounded in [0, 1]
        assert 0.0 <= v1 <= 1.0
        assert 0.0 <= v2 <= 1.0
        assert 0.0 <= v3 <= 1.0
        
        # With small leak (0.001) and positive doses, volumes should generally trend upward
        # (accumulation > leak for these puff values)
        assert v1 > 0.1  # Should have accumulated from initial state
        """Test that zero doses still applies leak (doesn't maintain pressure)."""
        balloon = AMR_LeakyBalloon(leak=0.2)
        
        # Build pressure
        balloon.step(doses=3.0)
        p_after_puff = balloon.pressure
        
        # Zero doses, leak applies
        balloon.step(doses=0.0)
        p_after_leak = balloon.pressure
        
        # Pressure should decrease by leak amount
        expected = p_after_puff - 0.2
        assert p_after_leak == pytest.approx(max(0.0, expected), abs=1e-6)


class TestEdgeCasesComprehensive:
    """Comprehensive edge case testing."""
    
    def test_very_high_initial_amr(self):
        """Test balloon initialized at very high AMR initial state."""
        balloon = AMR_LeakyBalloon(initial_amr_level=0.99)
        
        # Volume should be at the set level (0.99)
        assert abs(balloon.get_volume() - 0.99) < 0.01
        
        # After one step with small puff, should still be bounded at 1.0
        v = balloon.step(doses=1.0)
        assert v <= 1.0
    
    def test_high_residual_floor(self):
        """Test balloon with high residual floor (80%)."""
        residual = 0.8
        balloon = AMR_LeakyBalloon(
            permanent_residual_volume=residual,
            initial_amr_level=residual
        )
        
        # Even with no doses, should stay above 80%
        for _ in range(100):
            balloon.step(doses=0.0)
        
        final_volume = balloon.get_volume()
        assert final_volume >= residual - 1e-5
    
    def test_very_slow_decay(self):
        """Test balloon with minimal leak (very slow decay)."""
        balloon = AMR_LeakyBalloon(leak=0.001, initial_amr_level=0.5)
        balloon.step(doses=1.0)
        initial_pressure = balloon.pressure
        
        # After 100 steps with no doses, still should have>90% pressure
        for _ in range(100):
            balloon.step(doses=0.0)
        
        remaining_fraction = balloon.pressure / initial_pressure
        assert remaining_fraction > 0.9
    
    def test_very_fast_decay(self):
        """Test balloon with high leak (very fast decay)."""
        balloon = AMR_LeakyBalloon(leak=0.95)
        balloon.step(doses=5.0)
        
        # After a few steps, pressure should be nearly zero
        for _ in range(10):
            balloon.step(doses=0.0)
        
        # Should be very close to residual (which is 0.0 by default)
        assert balloon.get_volume() < 0.01
    
    def test_step_with_very_large_doses(self):
        """Test step with extremely large puff values."""
        balloon = AMR_LeakyBalloon()
        
        # Very large puff
        volume = balloon.step(doses=10000.0)
        
        # Should saturate near 1.0
        assert volume > 0.9999
        assert volume <= 1.0


class TestInverseSigmoidComprehensive:
    """Comprehensive tests for inverse sigmoid calculation."""
    
    def test_inverse_sigmoid_round_trip_multiple_volumes(self):
        """Test inverse sigmoid round-trip for many volumes."""
        balloon = AMR_LeakyBalloon(
            flatness_parameter=1.5,
            permanent_residual_volume=0.1,
            initial_amr_level=0.1
        )
        
        test_volumes = [
            0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]
        
        for target_vol in test_volumes:
            pressure = balloon._inverse_sigmoid(target_vol)
            recovered_vol = balloon.get_volume(pressure=pressure)
            assert recovered_vol == pytest.approx(target_vol, abs=1e-5)
    
    def test_inverse_sigmoid_at_extremes(self):
        """Test inverse sigmoid at extreme volume values."""
        balloon = AMR_LeakyBalloon()
        
        # At residual (should give ~zero pressure)
        pressure_at_residual = balloon._inverse_sigmoid(0.0)
        assert pressure_at_residual >= 0.0
        assert pressure_at_residual < 0.01
        
        # Very close to 1.0 (should give high pressure)
        # With rescaled sigmoid, the required pressure is lower than with original sigmoid
        pressure_high = balloon._inverse_sigmoid(0.9999)
        assert pressure_high > 1.0
        
        # Pressure should strictly increase with volume
        pressure_mid = balloon._inverse_sigmoid(0.5)
        assert pressure_at_residual < pressure_mid < pressure_high


class TestStepNoStateChange:
    """Tests for the _step_no_internal_state_change method."""
    
    def test_no_state_change_preserves_pressure(self):
        """Test that _step_no_internal_state_change doesn't modify pressure."""
        balloon = AMR_LeakyBalloon()
        balloon.step(doses=2.0)
        
        saved_pressure = balloon.pressure
        balloon._step_no_internal_state_change(doses=1.0)
        
        assert balloon.pressure == saved_pressure
    
    def test_no_state_change_returns_correct_volume(self):
        """Test that _step_no_internal_state_change returns correct hypothetical volume."""
        balloon = AMR_LeakyBalloon(leak=0.1)
        balloon.step(doses=2.0)
        
        # Compute hypothetical
        hypothetical_volume = balloon._step_no_internal_state_change(doses=1.0)
        
        # Now actually step
        actual_volume = balloon.step(doses=1.0)
        
        # Should match
        assert hypothetical_volume == pytest.approx(actual_volume, abs=1e-10)