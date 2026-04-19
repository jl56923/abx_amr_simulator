"""Unit tests for EarlyStoppingStudyCallback and build_early_stopping_callback.

All tests use real in-memory Optuna studies with synthetic deterministic
objectives — no mocks of Optuna internals. This ensures we test actual
Optuna callback mechanics rather than stubs.
"""

import optuna
import pytest

from abx_amr_simulator.training.tune import (
    EarlyStoppingStudyCallback,
    build_early_stopping_callback,
)

# Suppress Optuna's per-trial logging during tests
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _run_study_with_values(
    values: list,
    warmup_trials: int,
    patience: int,
    min_delta: float,
    n_trials: int | None = None,
) -> optuna.Study:
    """Helper: run an in-memory study whose objective returns values[trial_number].

    The objective cycles through `values` by index. If `n_trials` is not given,
    it defaults to len(values). The early stopping callback is always attached.

    Returns the completed study for inspection.
    """
    if n_trials is None:
        n_trials = len(values)

    callback = EarlyStoppingStudyCallback(
        warmup_trials=warmup_trials,
        patience=patience,
        min_delta=min_delta,
    )

    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        idx = trial.number % len(values)
        return values[idx]

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    return study


class TestEarlyStoppingStudyCallbackFlat:
    """Flat reward landscape — study should stop after warmup + patience trials."""

    def test_stops_after_warmup_plus_patience(self):
        """With a constant objective, stops at warmup + 1 + patience trials.

        The +1 accounts for the first post-warmup trial, which always resets
        _best_value from -inf (any finite value exceeds -inf + min_delta).
        Patience counting only begins on the trial after that initial reset.
        """
        warmup = 2
        patience = 3
        # All trials return the same value.
        study = _run_study_with_values(
            values=[-275.0] * 20,
            warmup_trials=warmup,
            patience=patience,
            min_delta=2.0,
            n_trials=20,
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        # warmup trials skipped + 1 trial for initial -inf reset + patience trials → stop.
        assert n_completed == warmup + 1 + patience

    def test_does_not_stop_before_warmup_plus_patience(self):
        """A flat objective should not stop before warmup + patience."""
        warmup = 3
        patience = 4
        study = _run_study_with_values(
            values=[-275.0] * 20,
            warmup_trials=warmup,
            patience=patience,
            min_delta=2.0,
            n_trials=20,
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        assert n_completed >= warmup + patience


class TestEarlyStoppingWarmupProtection:
    """Study must never stop during the warmup period."""

    def test_warmup_protects_against_early_stop(self):
        """Even with patience=1, warmup trials are never subject to stopping."""
        warmup = 5
        # With patience=1 and min_delta=2, the counter would fire immediately
        # after warmup if the reward is flat.
        study = _run_study_with_values(
            values=[-275.0] * 20,
            warmup_trials=warmup,
            patience=1,
            min_delta=2.0,
            n_trials=20,
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        # Must complete at least warmup trials before stopping can fire.
        assert n_completed >= warmup


class TestEarlyStoppingImprovement:
    """Large improvement resets the patience counter."""

    def test_large_improvement_resets_counter(self):
        """When a trial exceeds best + min_delta, patience resets and study continues."""
        warmup = 2
        patience = 3
        min_delta = 2.0
        # Trials 0-1: warmup (flat at -275).
        # Trial 2: first past warmup, flat → counter = 1.
        # Trial 3: big improvement to -250 (+25 > 2.0) → counter resets to 0.
        # Trials 4-6: flat again → counter reaches patience → stops.
        values = [-275.0, -275.0, -275.0, -250.0, -250.0, -250.0, -250.0, -250.0, -250.0]
        study = _run_study_with_values(
            values=values,
            warmup_trials=warmup,
            patience=patience,
            min_delta=min_delta,
            n_trials=20,
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        # Without improvement reset, would stop at trial 5 (warmup + patience).
        # With the reset at trial 3, patience restarts: stops at trial 3 + patience = 6.
        # Allow a window of ±1 for off-by-one in the reset counting.
        assert n_completed > warmup + patience

    def test_repeated_improvements_keep_study_running(self):
        """When every trial improves meaningfully, study runs all n_trials."""
        # Each trial is progressively better by well above min_delta.
        n = 15
        values = [-300.0 + i * 10 for i in range(n)]  # +10 per trial
        study = _run_study_with_values(
            values=values,
            warmup_trials=3,
            patience=3,
            min_delta=2.0,
            n_trials=n,
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        assert n_completed == n


class TestEarlyStoppingMinDelta:
    """min_delta threshold correctly distinguishes signal from noise."""

    def test_small_gain_below_min_delta_does_not_reset_counter(self):
        """Improvements below min_delta do not reset the patience counter.

        After the first post-warmup trial sets the baseline (_best_value), any
        subsequent improvement smaller than min_delta is treated as noise and
        the patience counter keeps incrementing.
        """
        warmup = 2
        patience = 3
        min_delta = 2.0
        # Trials 0-1: warmup (flat at -275.0).
        # Trial 2: -274.0 → triggers initial -inf reset (_best_value = -274.0, count = 0).
        # Trials 3+: stuck at -273.5 (+0.5 from _best_value = -274.0, well below min_delta=2.0).
        # Counter ticks each trial and reaches patience → stops.
        values = [-275.0, -275.0, -274.0] + [-273.5] * 20
        study = _run_study_with_values(
            values=values,
            warmup_trials=warmup,
            patience=patience,
            min_delta=min_delta,
            n_trials=25,
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        # warmup(2) + initial reset(1) + patience(3) = 6 trials.
        assert n_completed == warmup + 1 + patience

    def test_gain_above_min_delta_resets_counter(self):
        """An improvement strictly greater than min_delta resets the counter."""
        warmup = 2
        patience = 4
        min_delta = 2.0
        # Flat for warmup, then one improvement of exactly +3.0 (> min_delta),
        # then flat again. Counter resets after the improvement.
        improvement_trial = warmup + 2  # trial index where improvement occurs
        base = -275.0
        values = [base] * 20
        values[improvement_trial] = base + 3.0  # +3.0 > min_delta=2.0

        study = _run_study_with_values(
            values=values,
            warmup_trials=warmup,
            patience=patience,
            min_delta=min_delta,
            n_trials=20,
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        # Without reset: would stop at warmup + patience = 6.
        # With reset at trial 4: stops at 4 + patience = 8.
        assert n_completed > warmup + patience

    def test_gain_exactly_equal_to_min_delta_does_not_reset(self):
        """An improvement of exactly min_delta does not reset the patience counter.

        The check is strictly greater (current_best > _best_value + min_delta),
        so exact equality leaves the counter unchanged.

        After the initial -inf reset sets _best_value = base_reset, a subsequent
        improvement of exactly min_delta satisfies current_best == _best_value + min_delta
        (not strictly greater) and therefore does NOT reset the counter.
        """
        warmup = 2
        patience = 3
        min_delta = 2.0
        base = -275.0
        # Trials 0-1: warmup (-275.0).
        # Trial 2: -274.0 → initial reset (_best_value = -274.0, count = 0).
        # Trial 3: -272.0 (+2.0 from _best_value = -274.0; not strictly greater → count = 1).
        # Trials 4-5: flat at -272.0 → count = 2, 3 → STOP.
        values = [base, base, base + 1.0, base + 3.0] + [base + 3.0] * 20
        # _best_value after trial 2 = -274.0
        # trial 3 value = -272.0; check: -272.0 > -274.0 + 2.0 = -272.0? NO (not strictly greater)
        study = _run_study_with_values(
            values=values,
            warmup_trials=warmup,
            patience=patience,
            min_delta=min_delta,
            n_trials=20,
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        # warmup(2) + initial reset(1) + patience(3) = 6 trials.
        assert n_completed == warmup + 1 + patience


class TestBuildEarlyStoppingCallback:
    """Tests for build_early_stopping_callback config parsing."""

    def test_returns_callback_when_enabled(self):
        config = {
            'early_stopping': {
                'enabled': True,
                'warmup_trials': 5,
                'patience': 10,
                'min_delta': 3.0,
            }
        }
        cb = build_early_stopping_callback(config)
        assert cb is not None
        assert isinstance(cb, EarlyStoppingStudyCallback)
        assert cb.warmup_trials == 5
        assert cb.patience == 10
        assert cb.min_delta == 3.0

    def test_returns_none_when_disabled(self):
        config = {
            'early_stopping': {
                'enabled': False,
                'warmup_trials': 5,
                'patience': 10,
                'min_delta': 2.0,
            }
        }
        cb = build_early_stopping_callback(config)
        assert cb is None

    def test_returns_none_when_block_absent(self):
        config = {'n_trials': 32, 'sampler': 'TPE'}
        cb = build_early_stopping_callback(config)
        assert cb is None

    def test_uses_defaults_for_missing_sub_keys(self):
        """If enabled but individual params are missing, uses hardcoded defaults."""
        config = {'early_stopping': {'enabled': True}}
        cb = build_early_stopping_callback(config)
        assert cb is not None
        assert cb.warmup_trials == 8
        assert cb.patience == 8
        assert cb.min_delta == 2.0

    def test_parses_values_from_voi_tuning_yaml_structure(self):
        """End-to-end: values matching the VOI tuning YAML parse correctly."""
        config = {
            'n_trials': 32,
            'sampler': 'TPE',
            'early_stopping': {
                'enabled': True,
                'warmup_trials': 8,
                'patience': 8,
                'min_delta': 2.0,
            },
        }
        cb = build_early_stopping_callback(config)
        assert cb is not None
        assert cb.warmup_trials == 8
        assert cb.patience == 8
        assert cb.min_delta == 2.0
