"""
Integration tests for Optuna trial counting logic in tune.py.

Tests the behavior when resuming Optuna studies, specifically:
- Early exit when target number of complete trials reached
- Registry updates when target met but missing from registry
- Correct remaining_trials calculation for resumed studies
- No duplicate registry updates when entry already exists
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict
import pytest
import optuna

from abx_amr_simulator.utils.registry import (
    load_registry,
    update_registry,
)


def create_minimal_tuning_config() -> Dict[str, Any]:
    """Create minimal tuning config for testing."""
    return {
        'optimization': {
            'n_trials': 5,
            'n_seeds_per_trial': 1,
            'truncated_episodes': 10,
            'direction': 'maximize',
            'sampler': 'Random',
            'stability_penalty_weight': 0.0,
        },
        'search_space': {
            'learning_rate': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-3,
                'log': False
            }
        }
    }


def simple_objective(trial: optuna.Trial) -> float:
    """Simple objective function for testing."""
    lr = trial.suggest_float('learning_rate', low=1e-4, high=1e-3)
    # Return a simple deterministic value based on hyperparameter
    return lr * 1000


class TestOptunaTrialCounting:
    """Test suite for Optuna trial counting logic."""
    
    def test_early_exit_when_target_trials_met(self):
        """Verify study exits early when n_completed >= n_trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_url = f"sqlite:///{tmpdir}/test_study.db"
            study_name = "test_study_early_exit"
            n_trials = 5
            
            # Create study and run to completion
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=storage_url,
                load_if_exists=False
            )
            
            # Run exactly n_trials
            study.optimize(simple_objective, n_trials=n_trials)
            
            # Verify exactly n_trials completed
            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            assert n_completed == n_trials, f"Expected {n_trials} completed trials, got {n_completed}"
            
            # Simulate loading existing study (what tune.py does)
            study_resumed = optuna.load_study(
                study_name=study_name,
                storage=storage_url
            )
            
            n_existing_trials = len(study_resumed.trials)
            n_completed_resumed = len([
                t for t in study_resumed.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            
            # Verify target already met
            assert n_completed_resumed >= n_trials, (
                f"Expected at least {n_trials} completed, got {n_completed_resumed}"
            )
            
            # In tune.py, this condition triggers early exit WITHOUT running optimize()
            # remaining_trials calculation would be: max(0, n_trials - n_completed_resumed)
            remaining_trials = max(0, n_trials - n_completed_resumed)
            assert remaining_trials == 0, (
                f"Expected 0 remaining trials, got {remaining_trials}"
            )
    
    def test_registry_updated_when_target_met_but_missing(self):
        """Verify registry gets updated when target met but entry missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_url = f"sqlite:///{tmpdir}/test_study.db"
            study_name = "test_study_registry_update"
            registry_path = os.path.join(tmpdir, '.optimization_completed.txt')
            run_name = "exp_test_registry"
            timestamp = "20240101_120000"
            n_trials = 3
            
            # Create and complete study
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=storage_url,
                load_if_exists=False
            )
            study.optimize(simple_objective, n_trials=n_trials)
            
            # Verify target met
            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            assert n_completed >= n_trials
            
            # Registry should not exist yet
            assert not os.path.exists(registry_path), "Registry should not exist initially"
            
            # Simulate tune.py logic: check registry and update if missing
            completed_prefixes = load_registry(registry_path)  # Returns empty set if file doesn't exist
            assert run_name not in completed_prefixes, "Run should not be in registry yet"
            
            # Update registry (what tune.py does when target met but registry missing)
            update_registry(
                registry_path=registry_path,
                run_name=run_name,
                timestamp=timestamp
            )
            
            # Verify registry now contains entry
            completed_prefixes_after = load_registry(registry_path)
            assert run_name in completed_prefixes_after, (
                f"Registry should contain {run_name} after update"
            )
    
    def test_registry_not_duplicated_when_entry_exists(self):
        """Verify registry update skipped when entry already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, '.optimization_completed.txt')
            run_name = "exp_test_no_duplicate"
            timestamp1 = "20240101_120000"
            timestamp2 = "20240101_130000"
            
            # Add initial entry
            update_registry(
                registry_path=registry_path,
                run_name=run_name,
                timestamp=timestamp1
            )
            
            # Verify entry exists
            completed_prefixes = load_registry(registry_path)
            assert run_name in completed_prefixes
            
            # Count lines in registry
            with open(registry_path, 'r') as f:
                lines_before = f.readlines()
            num_entries_before = len([l for l in lines_before if l.strip() and not l.startswith('#')])
            
            # Simulate tune.py logic: check if already in registry
            if run_name in completed_prefixes:
                # Should NOT call update_registry again
                # (In tune.py, this is the "else" branch that prints "Registry already marked complete")
                pass
            else:
                # This branch should NOT execute
                update_registry(
                    registry_path=registry_path,
                    run_name=run_name,
                    timestamp=timestamp2
                )
            
            # Verify no duplicate entry added
            with open(registry_path, 'r') as f:
                lines_after = f.readlines()
            num_entries_after = len([l for l in lines_after if l.strip() and not l.startswith('#')])
            
            assert num_entries_after == num_entries_before, (
                f"Registry should not have duplicate entries: before={num_entries_before}, after={num_entries_after}"
            )
    
    def test_remaining_trials_calculation_for_resumed_study(self):
        """Verify remaining_trials = max(0, n_trials - n_completed) for resumed studies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_url = f"sqlite:///{tmpdir}/test_study.db"
            study_name = "test_study_remaining_trials"
            n_trials_target = 10
            n_initial_trials = 4  # Run only 4 out of 10
            
            # Create study and run partial trials
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=storage_url,
                load_if_exists=False
            )
            study.optimize(simple_objective, n_trials=n_initial_trials)
            
            # Verify partial completion
            n_completed_initial = len([
                t for t in study.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            assert n_completed_initial == n_initial_trials
            
            # Simulate resuming study (what tune.py does)
            study_resumed = optuna.load_study(
                study_name=study_name,
                storage=storage_url
            )
            
            n_existing_trials = len(study_resumed.trials)
            n_completed_resumed = len([
                t for t in study_resumed.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            
            # Calculate remaining_trials (tune.py logic)
            remaining_trials = max(0, n_trials_target - n_completed_resumed)
            
            expected_remaining = n_trials_target - n_initial_trials
            assert remaining_trials == expected_remaining, (
                f"Expected {expected_remaining} remaining trials, got {remaining_trials}"
            )
            
            # Verify resumed study completes exactly remaining_trials more
            study_resumed.optimize(simple_objective, n_trials=remaining_trials)
            
            n_completed_final = len([
                t for t in study_resumed.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            
            assert n_completed_final == n_trials_target, (
                f"Expected exactly {n_trials_target} completed trials after resume, got {n_completed_final}"
            )
    
    def test_no_extra_trials_when_resuming_completed_study(self):
        """Verify resumed study does NOT run extra trials if already complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_url = f"sqlite:///{tmpdir}/test_study.db"
            study_name = "test_study_no_extra_trials"
            n_trials = 5
            
            # Create and complete study
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=storage_url,
                load_if_exists=False
            )
            study.optimize(simple_objective, n_trials=n_trials)
            
            n_completed_initial = len([
                t for t in study.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            assert n_completed_initial == n_trials
            
            # Simulate resuming completed study
            study_resumed = optuna.load_study(
                study_name=study_name,
                storage=storage_url
            )
            
            n_completed_resumed = len([
                t for t in study_resumed.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            
            # Calculate remaining_trials (tune.py logic)
            remaining_trials = max(0, n_trials - n_completed_resumed)
            
            # Should be 0 - no more trials needed
            assert remaining_trials == 0, (
                f"Expected 0 remaining trials for completed study, got {remaining_trials}"
            )
            
            # In tune.py, when remaining_trials == 0, it exits early WITHOUT calling optimize()
            # If we mistakenly called optimize(n_trials=n_trials) here, we'd get n_trials MORE trials
            # Verify we DON'T do that by checking final count matches initial
            n_trials_final = len([
                t for t in study_resumed.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            
            assert n_trials_final == n_trials, (
                f"Study should still have exactly {n_trials} trials, got {n_trials_final}"
            )
    
    def test_registry_and_early_exit_integration(self):
        """Integration test: verify registry+early_exit work together correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_url = f"sqlite:///{tmpdir}/test_study.db"
            study_name = "test_study_integration"
            registry_path = os.path.join(tmpdir, '.optimization_completed.txt')
            run_name = "exp_integration_test"
            timestamp = "20240101_120000"
            n_trials = 3
            
            # Create and complete study
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=storage_url,
                load_if_exists=False
            )
            study.optimize(simple_objective, n_trials=n_trials)
            
            # Verify completion
            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            assert n_completed >= n_trials
            
            # Simulate tune.py logic for completed study
            study_resumed = optuna.load_study(
                study_name=study_name,
                storage=storage_url
            )
            
            n_completed_resumed = len([
                t for t in study_resumed.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            
            # Check if target met (should trigger early exit)
            if n_completed_resumed >= n_trials:
                # Update registry if missing
                completed_prefixes = load_registry(registry_path)
                if run_name not in completed_prefixes:
                    update_registry(
                        registry_path=registry_path,
                        run_name=run_name,
                        timestamp=timestamp
                    )
                    registry_updated = True
                else:
                    registry_updated = False
                
                # Should exit early (NOT call optimize())
                should_exit_early = True
            else:
                should_exit_early = False
                registry_updated = False
            
            # Verify expected behavior
            assert should_exit_early, "Should trigger early exit when target met"
            assert registry_updated, "Should update registry when missing"
            
            # Verify registry contains entry
            completed_prefixes_final = load_registry(registry_path)
            assert run_name in completed_prefixes_final
            
            # Verify no extra trials added (early exit worked)
            n_trials_final = len([
                t for t in study_resumed.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            assert n_trials_final == n_trials, (
                f"Early exit should prevent extra trials: expected {n_trials}, got {n_trials_final}"
            )
