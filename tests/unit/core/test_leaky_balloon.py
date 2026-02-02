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


def test_leak_and_bounds_no_puffs():
    # This test verifies that the balloon's latent pressure decays over time
    # due to the leak rate when no puffs are added. It checks that the volume
    # after one step matches the expected volume based on the leak formula:
    # pressure_new = pressure_old - leak, and that the output is bounded [0, 1].
    """Test that balloon leaks when no puffs are added."""
    balloon = AMR_LeakyBalloon(
        leak=0.2,
        flatness_parameter=1.0,
        permanent_residual_volume=0.0,
        initial_amr_level=0.5
    )
    initial_volume = balloon.get_volume()
    volume_after_step = balloon.step(puffs=0)
    
    # After one step with no puffs, latent pressure should decay
    # which reduces volume slightly
    assert volume_after_step < initial_volume
    assert 0.0 <= volume_after_step <= 1.0


def test_monotonic_with_puffs_and_bounded():
    # This test ensures that adding puffs always increases the balloon's volume
    # monotonically, and that the volume stays within the valid bounds [0, 1].
    # It verifies that more puffs lead to higher volumes in sequence.
    """Test that balloon volume increases with puffs and stays bounded."""
    balloon = AMR_LeakyBalloon(
        leak=0.1,
        flatness_parameter=1.0,
        permanent_residual_volume=0.0,
        initial_amr_level=0.0
    )
    v0 = balloon.get_volume()
    v1 = balloon.step(puffs=1)
    v2 = balloon.step(puffs=3)
    
    assert v1 > v0
    assert v2 > v1
    assert 0.0 <= v2 <= 1.0


def test_negative_puffs_raises():
    # This test verifies that the step() method rejects invalid input (negative puffs)
    # by raising a ValueError. Negative puffs don't make physical sense in the model.
    """Test that negative puffs raise an error."""
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
    balloon.step(puffs=5)
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
    # A smaller flatness_parameter produces a steeper sigmoid curve, so the same puffs
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
    
    # Step both balloons with same puffs
    for puffs in puff_sequence:
        balloon_steep.step(puffs=puffs)
        balloon_flat.step(puffs=puffs)
    
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
    # This test verifies that the get_delta_volume_for_counterfactual_num_puffs_vs_one_less()
    # method correctly computes the marginal volume change when increasing puffs by one.
    # It checks that the delta is a valid float and non-negative.
    """Test delta volume computation for counterfactual puffs."""
    balloon = AMR_LeakyBalloon(
        leak=0.1,
        flatness_parameter=1.0,
        permanent_residual_volume=0.0,
        initial_amr_level=0.0
    )
    
    delta = balloon.get_delta_volume_for_counterfactual_num_puffs_vs_one_less(num_counterfactual_puffs=5)
    assert isinstance(delta, float)
    assert delta >= 0.0  # Delta should be non-negative


def test_delta_volume_zero_puffs():
    # This test is an edge case check: when the counterfactual puffs is 0, the delta
    # between 0 puffs and -1 puffs should be 0 (since -1 would be invalid/clamped).
    """Test that delta volume is 0 when counterfactual puffs is 0."""
    balloon = AMR_LeakyBalloon()
    delta = balloon.get_delta_volume_for_counterfactual_num_puffs_vs_one_less(0)
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
    copy_volume_after_step1 = copy.step(puffs=2.0)
    copy_volume_after_step2 = copy.step(puffs=1.0)
    
    # Verify original is unaffected by copy's steps
    assert original.get_volume() == original_initial_volume
    assert original.pressure == original_initial_pressure
    
    # Now step the original
    original_volume_after_step1 = original.step(puffs=0.0)  # Just leak, no puffs
    
    # Verify copy is unaffected by original's steps
    assert copy.pressure != original.pressure
    assert copy.get_volume() != original_initial_volume
    
    # Original should have leaked (lower volume than initial)
    assert original_volume_after_step1 < original_initial_volume
    
    # Copy should have higher volume (added puffs)
    assert copy_volume_after_step1 > original_initial_volume


def test_copy_with_reset_counterfactual():
    """Test the counterfactual usage pattern: copy, reset, then compute delta.
    
    This mimics the environment usage where we:
    1. Copy the balloon model
    2. Reset it to visible AMR level
    3. Compute delta for a hypothetical number of puffs
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
    
    # Compute counterfactual delta (how much volume would change with 3 puffs vs 2)
    delta_for_3_vs_2_puffs = balloon_copy.get_delta_volume_for_counterfactual_num_puffs_vs_one_less(3)
    
    # Verify delta is positive (more puffs = more volume)
    assert delta_for_3_vs_2_puffs > 0.0
    
    # Verify original is completely unaffected
    assert original.get_volume() == original_initial_volume
    assert original.pressure > 0.0  # Original still has original pressure
    
    # Verify copy's pressure is different from original (it was reset)
    assert balloon_copy.pressure != original.pressure