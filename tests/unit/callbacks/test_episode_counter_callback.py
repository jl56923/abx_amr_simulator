"""Unit tests for EpisodeCounterCallback.

Verifies:
- Episode counting from dones dict (single and vectorised)
- stop_after_n_episodes correctly terminates training
- Counting-only mode (no stop) continues training indefinitely
- Invalid constructor argument raises ValueError
- TensorBoard logging occurs at every step
"""

import numpy as np
import pytest

from stable_baselines3.common.logger import Logger

from abx_amr_simulator.callbacks import EpisodeCounterCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal model stub that provides a real SB3 Logger to the callback."""
    def __init__(self):
        self.logger = Logger(folder=None, output_formats=[])


def _make_callback(stop_after_n_episodes=None, verbose=0):
    """Return an EpisodeCounterCallback wired with a real no-output logger."""
    cb = EpisodeCounterCallback(
        stop_after_n_episodes=stop_after_n_episodes,
        verbose=verbose,
    )
    # Attach a minimal model stub so cb.logger (→ cb.model.logger) resolves.
    cb.model = _FakeModel()
    return cb


def _step(callback, dones):
    """Simulate one training step by setting locals and calling _on_step."""
    callback.locals['dones'] = dones
    return callback._on_step()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestEpisodeCounterCallbackConstruction:
    def test_default_stop_is_none(self):
        cb = _make_callback()
        assert cb.stop_after_n_episodes is None

    def test_explicit_stop_stored(self):
        cb = _make_callback(stop_after_n_episodes=100)
        assert cb.stop_after_n_episodes == 100

    def test_initial_episode_count_is_zero(self):
        cb = _make_callback(stop_after_n_episodes=10)
        assert cb.n_episodes == 0

    def test_zero_stop_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            EpisodeCounterCallback(stop_after_n_episodes=0)

    def test_negative_stop_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            EpisodeCounterCallback(stop_after_n_episodes=-5)


# ---------------------------------------------------------------------------
# Episode counting
# ---------------------------------------------------------------------------

class TestEpisodeCounterCallbackCounting:
    def test_done_true_increments_count(self):
        cb = _make_callback()
        _step(callback=cb, dones=[True])
        assert cb.n_episodes == 1

    def test_done_false_does_not_increment_count(self):
        cb = _make_callback()
        _step(callback=cb, dones=[False])
        assert cb.n_episodes == 0

    def test_multiple_dones_in_vectorised_step(self):
        """All True entries in dones should each increment the counter."""
        cb = _make_callback()
        _step(callback=cb, dones=[True, False, True])
        assert cb.n_episodes == 2

    def test_count_accumulates_across_steps(self):
        cb = _make_callback()
        _step(callback=cb, dones=[True])
        _step(callback=cb, dones=[False])
        _step(callback=cb, dones=[True])
        _step(callback=cb, dones=[True])
        assert cb.n_episodes == 3

    def test_bare_bool_true_treated_as_single_done(self):
        """A plain Python bool (non-vectorised env) must be handled."""
        cb = _make_callback()
        cb.locals['dones'] = True
        cb._on_step()
        assert cb.n_episodes == 1

    def test_bare_bool_false_not_counted(self):
        cb = _make_callback()
        cb.locals['dones'] = False
        cb._on_step()
        assert cb.n_episodes == 0

    def test_numpy_bool_true_counted(self):
        cb = _make_callback()
        cb.locals['dones'] = np.bool_(True)
        cb._on_step()
        assert cb.n_episodes == 1

    def test_fallback_to_done_key_when_dones_absent(self):
        """Some SB3 versions use 'done' instead of 'dones'."""
        cb = _make_callback()
        cb.locals['done'] = True
        cb._on_step()
        assert cb.n_episodes == 1


# ---------------------------------------------------------------------------
# Stop-after-N behaviour
# ---------------------------------------------------------------------------

class TestEpisodeCounterCallbackStopping:
    def test_continues_before_target(self):
        cb = _make_callback(stop_after_n_episodes=3)
        result = _step(callback=cb, dones=[True])   # 1 episode
        assert result is True, "Should continue training before reaching target"

    def test_stops_at_exact_target(self):
        cb = _make_callback(stop_after_n_episodes=2)
        _step(callback=cb, dones=[True])             # episode 1
        result = _step(callback=cb, dones=[True])    # episode 2 → should stop
        assert result is False, "Should stop training when target is reached"

    def test_stops_when_multiple_dones_cross_target(self):
        """If multiple episodes finish in one step and the count crosses the
        target, training must stop."""
        cb = _make_callback(stop_after_n_episodes=3)
        _step(callback=cb, dones=[True])             # episode 1
        result = _step(callback=cb, dones=[True, True, True])  # episodes 2,3,4
        assert result is False

    def test_counting_only_never_stops(self):
        """Without stop_after_n_episodes, _on_step always returns True."""
        cb = _make_callback(stop_after_n_episodes=None)
        for _ in range(1000):
            result = _step(callback=cb, dones=[True])
        assert result is True

    def test_single_episode_target(self):
        cb = _make_callback(stop_after_n_episodes=1)
        result = _step(callback=cb, dones=[True])
        assert result is False
        assert cb.n_episodes == 1

    def test_no_done_step_before_target_does_not_stop(self):
        cb = _make_callback(stop_after_n_episodes=5)
        for _ in range(100):
            result = _step(callback=cb, dones=[False])
        assert result is True
        assert cb.n_episodes == 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestEpisodeCounterCallbackLogging:
    def test_logger_record_called_every_step(self):
        """Verify that the logger receives 'training/completed_episodes' each step.

        We use a real SB3 Logger and read from its name_to_value dict to confirm
        the record was written.
        """
        cb = _make_callback()
        _step(callback=cb, dones=[False])
        # SB3 Logger stores recorded values keyed by name
        assert 'training/completed_episodes' in cb.logger.name_to_value

    def test_logged_value_matches_n_episodes(self):
        cb = _make_callback()
        _step(callback=cb, dones=[True])
        _step(callback=cb, dones=[True])
        _step(callback=cb, dones=[False])
        assert cb.logger.name_to_value['training/completed_episodes'] == cb.n_episodes
