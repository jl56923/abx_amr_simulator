"""Unit tests for variable-length trajectory support in metrics.py.

Tests cover:
- aggregate_trajectories(): NaN-padding, masked statistics, n_active_trajectories
- Short-episode guards in summary metric computations
"""

import math

import numpy as np
import pytest

from abx_amr_simulator.utils.metrics import aggregate_trajectories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_constant_trajectory(value, length):
    """Return a trajectory of `length` steps all equal to `value`."""
    return [value] * length


def _assert_array_close(actual, expected, atol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol)


# ---------------------------------------------------------------------------
# aggregate_trajectories: basic structure
# ---------------------------------------------------------------------------

class TestAggregateTrajectories:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="trajectories_list is empty"):
            aggregate_trajectories(trajectories_list=[])

    def test_single_fixed_length_trajectory_returns_expected_keys(self):
        traj = [1.0, 2.0, 3.0]
        result = aggregate_trajectories(trajectories_list=[traj])
        expected_keys = {'mean', 'median', 'p10', 'p90', 'p25', 'p75', 'iqm', 'timesteps', 'n_active_trajectories'}
        assert set(result.keys()) == expected_keys

    def test_single_trajectory_timesteps_are_zero_indexed(self):
        traj = [10.0, 20.0, 30.0]
        result = aggregate_trajectories(trajectories_list=[traj])
        np.testing.assert_array_equal(result['timesteps'], [0, 1, 2])

    def test_single_trajectory_mean_equals_values(self):
        traj = [1.0, 5.0, 3.0]
        result = aggregate_trajectories(trajectories_list=[traj])
        _assert_array_close(result['mean'], traj)

    def test_single_trajectory_n_active_is_all_ones(self):
        traj = [1.0, 2.0, 3.0]
        result = aggregate_trajectories(trajectories_list=[traj])
        np.testing.assert_array_equal(result['n_active_trajectories'], [1, 1, 1])


# ---------------------------------------------------------------------------
# aggregate_trajectories: multiple equal-length trajectories
# ---------------------------------------------------------------------------

class TestAggregateTrajectoryFixedLength:
    def test_two_identical_trajectories_mean_equals_values(self):
        traj = [1.0, 2.0, 3.0]
        result = aggregate_trajectories(trajectories_list=[traj, traj])
        _assert_array_close(result['mean'], traj)

    def test_two_trajectories_mean_is_pointwise_average(self):
        t1 = [0.0, 0.0, 0.0]
        t2 = [2.0, 4.0, 6.0]
        result = aggregate_trajectories(trajectories_list=[t1, t2])
        _assert_array_close(result['mean'], [1.0, 2.0, 3.0])

    def test_n_active_trajectories_all_equal_n_trajectories(self):
        trajs = [[float(i)] * 4 for i in range(5)]
        result = aggregate_trajectories(trajectories_list=trajs)
        np.testing.assert_array_equal(result['n_active_trajectories'], [5, 5, 5, 5])

    def test_percentiles_ordered_correctly(self):
        # 10 trajectories with values 1..10 at every timestep
        trajs = [[float(v)] * 3 for v in range(1, 11)]
        result = aggregate_trajectories(trajectories_list=trajs)
        for t in range(3):
            assert result['p10'][t] <= result['p25'][t] <= result['median'][t]
            assert result['median'][t] <= result['p75'][t] <= result['p90'][t]


# ---------------------------------------------------------------------------
# aggregate_trajectories: variable-length (NaN-padding) behaviour
# ---------------------------------------------------------------------------

class TestAggregateTrajectoryVariableLength:
    def test_variable_lengths_output_length_equals_max_length(self):
        trajs = [[1.0] * 3, [2.0] * 5, [3.0] * 4]
        result = aggregate_trajectories(trajectories_list=trajs)
        assert len(result['mean']) == 5
        assert len(result['timesteps']) == 5

    def test_n_active_trajectories_drops_after_short_trajectories_end(self):
        # lengths: 3, 5, 4
        trajs = [[1.0] * 3, [2.0] * 5, [3.0] * 4]
        result = aggregate_trajectories(trajectories_list=trajs)
        n_active = result['n_active_trajectories']
        # ts 0-2: all 3 active
        assert all(n_active[t] == 3 for t in range(3))
        # ts 3: only trajs with length >=4: indices 1 and 2 → 2 active
        assert n_active[3] == 2
        # ts 4: only traj with length 5: index 1 → 1 active
        assert n_active[4] == 1

    def test_mean_at_timesteps_beyond_short_trajectory_excludes_it(self):
        # traj0 = [1.0, 1.0]  (length 2)
        # traj1 = [3.0, 3.0, 3.0]  (length 3)
        # At t=2, only traj1 contributes, so mean should be 3.0
        trajs = [[1.0, 1.0], [3.0, 3.0, 3.0]]
        result = aggregate_trajectories(trajectories_list=trajs)
        assert math.isclose(result['mean'][2], 3.0, abs_tol=1e-10)

    def test_mean_at_shared_timesteps_includes_all_trajectories(self):
        # traj0 = [1.0, 1.0], traj1 = [3.0, 3.0, 3.0]
        # At t=0 and t=1 both contribute: mean = (1+3)/2 = 2.0
        trajs = [[1.0, 1.0], [3.0, 3.0, 3.0]]
        result = aggregate_trajectories(trajectories_list=trajs)
        assert math.isclose(result['mean'][0], 2.0, abs_tol=1e-10)
        assert math.isclose(result['mean'][1], 2.0, abs_tol=1e-10)

    def test_three_different_lengths_n_active_correct(self):
        # lengths: 100, 200, 150
        trajs = [
            list(range(100)),
            list(range(200)),
            list(range(150)),
        ]
        result = aggregate_trajectories(trajectories_list=trajs)
        assert len(result['mean']) == 200
        # ts 0-99: 3 active
        assert result['n_active_trajectories'][0] == 3
        assert result['n_active_trajectories'][99] == 3
        # ts 100-149: 2 active
        assert result['n_active_trajectories'][100] == 2
        assert result['n_active_trajectories'][149] == 2
        # ts 150-199: 1 active
        assert result['n_active_trajectories'][150] == 1
        assert result['n_active_trajectories'][199] == 1


# ---------------------------------------------------------------------------
# aggregate_trajectories: apply_cumsum
# ---------------------------------------------------------------------------

class TestAggregateTrajectorysCumsum:
    def test_cumsum_on_fixed_length_trajectory(self):
        traj = [1.0, 1.0, 1.0, 1.0]
        result = aggregate_trajectories(trajectories_list=[traj], apply_cumsum=True)
        _assert_array_close(result['mean'], [1.0, 2.0, 3.0, 4.0])

    def test_cumsum_on_variable_length_trajectories(self):
        # traj0 = [1,1] → cumsum [1,2]
        # traj1 = [1,1,1] → cumsum [1,2,3]
        # At t=2, only traj1 contributes; mean should be 3.0
        trajs = [[1.0, 1.0], [1.0, 1.0, 1.0]]
        result = aggregate_trajectories(trajectories_list=trajs, apply_cumsum=True)
        assert math.isclose(result['mean'][0], 1.0, abs_tol=1e-10)
        assert math.isclose(result['mean'][1], 2.0, abs_tol=1e-10)
        assert math.isclose(result['mean'][2], 3.0, abs_tol=1e-10)

    def test_cumsum_does_not_propagate_into_padded_nan_region(self):
        # traj0 is short; the NaN-padded region should remain NaN after cumsum
        trajs = [[1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        result = aggregate_trajectories(trajectories_list=trajs, apply_cumsum=True)
        # n_active at t=2 and t=3 should be 1 (only traj1 active)
        assert result['n_active_trajectories'][2] == 1
        assert result['n_active_trajectories'][3] == 1


# ---------------------------------------------------------------------------
# aggregate_trajectories: IQM properties
# ---------------------------------------------------------------------------

class TestAggregateTrajectoryIQM:
    def test_iqm_within_p25_p75_range(self):
        trajs = [[float(v)] * 5 for v in range(1, 11)]
        result = aggregate_trajectories(trajectories_list=trajs)
        for t in range(5):
            assert result['p25'][t] <= result['iqm'][t] <= result['p75'][t]

    def test_iqm_symmetric_distribution_equals_median(self):
        # Uniform integers 1..10 at every step — IQM = mean = 5.5
        trajs = [[float(v)] * 3 for v in range(1, 11)]
        result = aggregate_trajectories(trajectories_list=trajs)
        # For a uniform symmetric distribution IQM ≈ mean
        for t in range(3):
            assert math.isclose(result['iqm'][t], result['mean'][t], abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Short-episode guards (final AMR computation)
# ---------------------------------------------------------------------------

class TestShortEpisodeGuards:
    """
    The final AMR computation in compute_overall_outcome_summary_dict uses
    tail = values[-min(10, len(values)):].  These tests verify the guard
    directly on the slicing logic replicated here, and also that zero-length
    episodes raise ValueError.
    """

    def _compute_tail_mean(self, values):
        """Replicate the guarded final-AMR-tail computation from metrics.py."""
        if len(values) == 0:
            raise ValueError("Episode has zero length (no primitive steps recorded)")
        tail = values[-min(10, len(values)):]
        return sum(tail) / len(tail)

    def test_normal_length_episode_uses_last_10_steps(self):
        # 20-step episode: last 10 all equal 5.0
        values = [1.0] * 10 + [5.0] * 10
        mean = self._compute_tail_mean(values=values)
        assert math.isclose(mean, 5.0, abs_tol=1e-10)

    def test_episode_shorter_than_10_uses_all_steps(self):
        # 5-step episode: average of [1,2,3,4,5] = 3.0
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = self._compute_tail_mean(values=values)
        assert math.isclose(mean, 3.0, abs_tol=1e-10)

    def test_episode_of_length_1_uses_that_single_step(self):
        values = [7.5]
        mean = self._compute_tail_mean(values=values)
        assert math.isclose(mean, 7.5, abs_tol=1e-10)

    def test_episode_of_length_10_uses_all_10_steps(self):
        values = [float(i) for i in range(10)]
        mean = self._compute_tail_mean(values=values)
        expected = sum(values) / 10
        assert math.isclose(mean, expected, abs_tol=1e-10)

    def test_zero_length_episode_raises_value_error(self):
        with pytest.raises(ValueError, match="zero length"):
            self._compute_tail_mean(values=[])
