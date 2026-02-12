"""Integration test for hyperparameter tuning workflow.

Tests the end-to-end workflow:
1. Run tune.py with minimal settings (1 trial, 1 seed, very short training)
2. Verify optimization directory and best_params.json are created
3. Run train.py with --load-best-params-by-experiment-name
4. Verify training completes successfully with loaded params

This test ensures that the complete tuning → training pipeline works correctly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from abx_amr_simulator.utils import (
    load_config,
    setup_config_folders_with_defaults,
)
from abx_amr_simulator.training import setup_optimization_folders_with_defaults
from abx_amr_simulator.utils.registry import load_registry


@pytest.fixture
def test_workspace():
    """Create a persistent test workspace for tuning integration test.
    
    Uses a stable directory for inspection: tests/integration/test_outputs/tuning_integration/
    """
    # Use stable directory relative to this test file
    test_dir = Path(__file__).parent / "test_outputs" / "tuning_integration"
    
    # Clean up if exists from previous run
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    test_dir.mkdir(parents=True)
    experiments_dir = test_dir / "experiments"
    experiments_dir.mkdir()
    
    # Create config structure
    setup_config_folders_with_defaults(target_path=experiments_dir)
    
    # Create tuning config structure
    setup_optimization_folders_with_defaults(target_path=experiments_dir)
    
    yield test_dir
    
    # Cleanup after test completes
    # Comment this out for debugging
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_tune_train_integration(test_workspace):
    """Test complete tuning workflow: tune → verify → train with best params.
    
    This test:
    1. Loads default PPO config and creates minimal tuning config
    2. Runs tune.py with 1 trial, 1 seed, 3 episodes (ultra-fast)
    3. Verifies optimization outputs (best_params.json, study_summary.json, registry)
    4. Runs train.py with --load-best-params-by-experiment-name
    5. Verifies training completes with loaded hyperparameters
    """
    experiments_dir = test_workspace / "experiments"
    
    # === STEP 1: Prepare configs ===
    
    # Load default umbrella config
    umbrella_config_path = experiments_dir / "configs" / "umbrella_configs" / "base_experiment.yaml"
    assert umbrella_config_path.exists(), f"Umbrella config not found: {umbrella_config_path}"
    
    config = load_config(config_path=str(umbrella_config_path))
    
    # Modify for ultra-brief testing
    config['training']['total_num_training_episodes'] = 3  # Very short
    config['environment']['max_time_steps'] = 5  # Minimal episode length
    config['training']['eval_freq_every_n_episodes'] = 3  # Eval once at end
    config['training']['save_freq_every_n_episodes'] = 3  # Save once at end
    config['training']['run_name'] = "tune_integration_test"
    
    # Save modified umbrella config
    test_umbrella_path = experiments_dir / "configs" / "umbrella_configs" / "test_tune_integration.yaml"
    with open(test_umbrella_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create minimal tuning config (1 trial, 1 seed, 3 episodes)
    tuning_config = {
        'optimization': {
            'n_trials': 1,  # Single trial
            'n_seeds_per_trial': 1,  # Single seed
            'truncated_episodes': 3,  # Ultra-short training
            'direction': 'maximize',
            'sampler': 'Random'  # Random sampler (faster than TPE for 1 trial)
        },
        'search_space': {
            'learning_rate': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-3,
                'log': True
            },
            'gamma': {
                'type': 'float',
                'low': 0.9,
                'high': 0.99
            }
        }
    }
    
    tuning_config_path = experiments_dir / "tuning_configs" / "test_tune_integration.yaml"
    with open(tuning_config_path, 'w') as f:
        yaml.dump(tuning_config, f)
    
    optimization_dir = test_workspace / "optimization"
    optimization_dir.mkdir()
    
    results_dir = test_workspace / "results"
    results_dir.mkdir()
    
    # === STEP 2: Run tune.py ===
    
    tune_cmd = [
        sys.executable, '-m', 'abx_amr_simulator.training.tune',
        '--umbrella-config', str(test_umbrella_path),
        '--tuning-config', str(tuning_config_path),
        '--run-name', 'tune_integration_test',
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir)
    ]
    
    print(f"\n[TEST] Running tune.py: {' '.join(tune_cmd)}")
    
    result = subprocess.run(
        tune_cmd,
        capture_output=True,
        text=True,
        timeout=120  # 2 minutes max
    )
    
    print(f"[TEST] tune.py stdout:\n{result.stdout}")
    if result.stderr:
        print(f"[TEST] tune.py stderr:\n{result.stderr}")
    
    assert result.returncode == 0, f"tune.py failed with return code {result.returncode}"
    
    # === STEP 3: Verify optimization outputs ===
    
    # Optimization folder should be optimization/{run_name}/ (no timestamp)
    optimization_run_dir = optimization_dir / "tune_integration_test"
    assert optimization_run_dir.exists(), f"Expected optimization folder not found: {optimization_run_dir}"
    
    print(f"[TEST] Found optimization run: {optimization_run_dir}")
    
    # Check best_params.json exists and is valid
    best_params_path = optimization_run_dir / "best_params.json"
    assert best_params_path.exists(), "best_params.json not created"
    
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    assert 'learning_rate' in best_params, "learning_rate not in best_params"
    assert 'gamma' in best_params, "gamma not in best_params"
    print(f"[TEST] Loaded best params: {best_params}")
    
    # Check study_summary.json exists
    study_summary_path = optimization_run_dir / "study_summary.json"
    assert study_summary_path.exists(), "study_summary.json not created"
    
    with open(study_summary_path, 'r') as f:
        study_summary = json.load(f)
    
    assert 'best_value' in study_summary
    assert 'n_trials' in study_summary
    assert study_summary['n_trials'] == 1
    print(f"[TEST] Study summary: best_value={study_summary['best_value']}, n_trials={study_summary['n_trials']}")
    
    # Check registry updated
    registry_path = optimization_dir / ".optimization_completed.txt"
    assert registry_path.exists(), "Optimization registry not created"
    
    completed_optimizations = load_registry(str(registry_path))
    assert "tune_integration_test" in completed_optimizations, "Optimization not recorded in registry"
    
    # === STEP 4: Run train.py with --load-best-params-by-experiment-name ===
    
    train_cmd = [
        sys.executable, '-m', 'abx_amr_simulator.training.train',
        '--umbrella-config', str(test_umbrella_path),
        '--load-best-params-by-experiment-name', 'tune_integration_test',
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir),
        '-p', 'training.total_num_training_episodes=3',  # Override to keep it short
        '-p', 'training.run_name=train_with_best_params'
    ]
    
    print(f"\n[TEST] Running train.py with loaded params: {' '.join(train_cmd)}")
    
    result = subprocess.run(
        train_cmd,
        capture_output=True,
        text=True,
        timeout=120  # 2 minutes max
    )
    
    print(f"[TEST] train.py stdout:\n{result.stdout}")
    if result.stderr:
        print(f"[TEST] train.py stderr:\n{result.stderr}")
    
    assert result.returncode == 0, f"train.py failed with return code {result.returncode}"
    
    # === STEP 5: Verify training outputs ===
    
    # Check that training created results directory
    training_runs = list(results_dir.glob("results/train_with_best_params_*"))
    assert len(training_runs) >= 1, f"Expected at least 1 training run, found {len(training_runs)}"
    
    training_run_dir = training_runs[0]
    print(f"[TEST] Found training run: {training_run_dir}")
    
    # Check that config was saved with loaded_best_params_from reference
    config_path = training_run_dir / "full_agent_env_config.yaml"
    assert config_path.exists(), "Training config not saved"
    
    with open(config_path, 'r') as f:
        saved_config = yaml.safe_load(f)
    
    # Verify that best params were loaded
    assert 'training' in saved_config
    assert 'loaded_best_params_from' in saved_config['training'], "Config doesn't reference optimization run"
    print(f"[TEST] Config references optimization: {saved_config['training']['loaded_best_params_from']}")
    
    # Verify that hyperparameters were actually applied
    assert 'agent_algorithm' in saved_config
    assert 'learning_rate' in saved_config['agent_algorithm']
    assert 'gamma' in saved_config['agent_algorithm']
    
    # Check that saved hyperparameters match best params from optimization
    assert saved_config['agent_algorithm']['learning_rate'] == best_params['learning_rate'], \
        "learning_rate not correctly applied"
    assert saved_config['agent_algorithm']['gamma'] == best_params['gamma'], \
        "gamma not correctly applied"
    
    print(f"[TEST] Verified hyperparameters applied correctly:")
    print(f"  learning_rate: {saved_config['agent_algorithm']['learning_rate']}")
    print(f"  gamma: {saved_config['agent_algorithm']['gamma']}")
    
    # Check that training actually completed (checkpoints directory exists)
    checkpoints_dir = training_run_dir / "checkpoints"
    assert checkpoints_dir.exists(), "Checkpoints directory not created"
    
    final_model_path = checkpoints_dir / "final_model.zip"
    assert final_model_path.exists(), "Final model not saved"
    
    print(f"[TEST] Integration test PASSED: tune → train workflow complete")


def test_skip_if_exists_prevents_duplicate_optimization(test_workspace):
    """Test that --skip-if-exists flag prevents duplicate optimization runs.
    
    This test:
    1. Runs tune.py once
    2. Runs tune.py again with --skip-if-exists
    3. Verifies second run is skipped (exit code 0, but no new optimization)
    """
    experiments_dir = test_workspace / "experiments"
    
    # Prepare minimal config
    umbrella_config_path = experiments_dir / "configs" / "umbrella_configs" / "base_experiment.yaml"
    config = load_config(config_path=str(umbrella_config_path))
    config['training']['total_num_training_episodes'] = 2
    config['environment']['max_time_steps'] = 5
    
    test_umbrella_path = experiments_dir / "configs" / "umbrella_configs" / "test_skip.yaml"
    with open(test_umbrella_path, 'w') as f:
        yaml.dump(config, f)
    
    tuning_config = {
        'optimization': {
            'n_trials': 1,
            'n_seeds_per_trial': 1,
            'truncated_episodes': 2,
            'direction': 'maximize',
            'sampler': 'Random'
        },
        'search_space': {
            'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-3, 'log': True}
        }
    }
    
    tuning_config_path = experiments_dir / "tuning_configs" / "test_skip.yaml"
    with open(tuning_config_path, 'w') as f:
        yaml.dump(tuning_config, f)
    
    optimization_dir = test_workspace / "optimization"
    optimization_dir.mkdir()
    results_dir = test_workspace / "results"
    results_dir.mkdir()
    
    # Run tune.py first time
    tune_cmd = [
        sys.executable, '-m', 'abx_amr_simulator.training.tune',
        '--umbrella-config', str(test_umbrella_path),
        '--tuning-config', str(tuning_config_path),
        '--run-name', 'skip_test',
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir)
    ]
    
    result1 = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=60)
    assert result1.returncode == 0, "First tune.py run failed"
    
    # Check optimization folder exists after first execution
    skip_test_dir = optimization_dir / "skip_test"
    assert skip_test_dir.exists(), f"Expected optimization folder not found: {skip_test_dir}"
    
    # Run tune.py second time with --skip-if-exists
    tune_cmd_skip = tune_cmd + ['--skip-if-exists']
    
    result2 = subprocess.run(tune_cmd_skip, capture_output=True, text=True, timeout=60)
    assert result2.returncode == 0, "Second tune.py run should exit successfully (skipped)"
    assert "SKIPPING" in result2.stdout, "Expected SKIPPING message in output"
    
    # Verify folder still exists (not recreated)
    assert skip_test_dir.exists(), "Optimization folder should still exist after skip"
    
    print(f"[TEST] --skip-if-exists correctly prevented duplicate optimization")
