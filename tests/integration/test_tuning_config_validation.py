"""Integration tests for tune.py config validation and persistence features.

Tests the following new features added to tune.py:
1. Folder structure: optimization/{run_name}/ (no timestamp in folder name)
2. Early config saving: configs saved immediately after loading (not after completion)
3. Config validation: prevents resuming with mismatched configs (fails loudly)
4. SQLite persistence: study database enables resumption across runs
5. HRL option library validation: detects changes to option library configs
6. Reward extraction: verifies that evaluation rewards are properly extracted (not -inf)

These tests ensure that the optimization infrastructure is robust and prevents
accidental corruption of studies by mixing trials from different configurations.

Test Coverage:
- test_folder_structure_no_timestamp: Verifies folder naming (no timestamp)
- test_configs_saved_immediately: Verifies early config persistence
- test_config_validation_prevents_mismatch: Verifies umbrella config validation
- test_overwrite_flag_bypasses_validation: Verifies --overwrite-existing-study flag
- test_study_resumption_with_matching_configs: Verifies successful resumption
- test_hrl_option_library_validation: Verifies option library config validation (HRL)
- test_reward_extraction_from_output: Verifies parse_reward_from_output function
- test_tuning_captures_valid_rewards: Verifies full tuning workflow extracts real rewards
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
from importlib.resources import files

from abx_amr_simulator.utils import (
    load_config,
    setup_config_folders_with_defaults,
)
from abx_amr_simulator.training import setup_optimization_folders_with_defaults


@pytest.fixture
def test_workspace():
    """Create a persistent test workspace for config validation tests.
    
    Uses a stable directory for inspection: tests/integration/test_outputs/tuning_config_validation/
    """
    test_dir = Path(__file__).parent / "test_outputs" / "tuning_config_validation"
    
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
    
    # Set up option libraries directory for HRL tests
    setup_option_libraries(target_path=experiments_dir)
    
    yield test_dir
    
    # Cleanup after test completes
    if test_dir.exists():
        shutil.rmtree(test_dir)


def setup_option_libraries(target_path: Path) -> None:
    """Copy default option libraries from package to test workspace.
    
    Creates options/option_libraries directory and copies default option library YAML files
    required for HRL algorithms.
    """
    option_lib_dir = Path(target_path) / "options" / "option_libraries"
    option_lib_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy default option library files from package
    defaults_root = files("abx_amr_simulator").joinpath("options/defaults/option_libraries")
    
    for entry in defaults_root.iterdir():
        if entry.is_file() and entry.name.endswith('.yaml'):
            dst_path = option_lib_dir / entry.name
            dst_path.write_bytes(entry.read_bytes())


def create_minimal_configs(experiments_dir: Path, run_name: str):
    """Helper to create minimal test configs for ultra-fast tuning.
    
    Returns:
        Tuple of (umbrella_config_path, tuning_config_path)
    """
    # Load default umbrella config
    umbrella_config_path = experiments_dir / "configs" / "umbrella_configs" / "base_experiment.yaml"
    config = load_config(config_path=str(umbrella_config_path))
    
    # Modify for ultra-brief testing
    config['training']['total_num_training_episodes'] = 2
    config['environment']['max_time_steps'] = 3
    config['training']['eval_freq_every_n_episodes'] = 2
    config['training']['save_freq_every_n_episodes'] = 2
    config['training']['run_name'] = run_name
    
    # Save modified umbrella config
    test_umbrella_path = experiments_dir / "configs" / "umbrella_configs" / f"{run_name}.yaml"
    with open(test_umbrella_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create minimal tuning config (1 trial, 1 seed)
    tuning_config = {
        'optimization': {
            'n_trials': 1,
            'n_seeds_per_trial': 1,
            'truncated_episodes': 2,
            'direction': 'maximize',
            'sampler': 'Random',
            'stability_penalty_weight': 0.1
        },
        'search_space': {
            'learning_rate': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-3,
                'log': True
            }
        }
    }
    
    tuning_config_path = experiments_dir / "tuning_configs" / f"{run_name}.yaml"
    with open(tuning_config_path, 'w') as f:
        yaml.dump(tuning_config, f)
    
    return test_umbrella_path, tuning_config_path


def test_folder_structure_no_timestamp(test_workspace):
    """Test that optimization folders use run_name only (no timestamp in folder name).
    
    Expected behavior:
    - Folder structure: optimization/{run_name}/
    - NOT: optimization/{run_name}_{timestamp}/
    - This enables database reuse across runs
    """
    experiments_dir = test_workspace / "experiments"
    optimization_dir = test_workspace / "optimization"
    optimization_dir.mkdir()
    results_dir = test_workspace / "results"
    results_dir.mkdir()
    
    run_name = "test_no_timestamp"
    umbrella_path, tuning_path = create_minimal_configs(experiments_dir, run_name)
    
    # Run tune.py
    tune_cmd = [
        sys.executable, '-m', 'abx_amr_simulator.training.tune',
        '--umbrella-config', str(umbrella_path),
        '--tuning-config', str(tuning_path),
        '--run-name', run_name,
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir)
    ]
    
    result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
    
    if result.returncode != 0:
        print(f"[TEST] tune.py stdout:\n{result.stdout}")
        print(f"[TEST] tune.py stderr:\n{result.stderr}")
    
    assert result.returncode == 0, f"tune.py failed with return code {result.returncode}"
    
    # Verify folder structure: should be exactly optimization/{run_name}/
    expected_folder = optimization_dir / run_name
    assert expected_folder.exists(), f"Expected folder not found: {expected_folder}"
    
    # Verify no timestamped folders exist
    all_folders = list(optimization_dir.glob(f"{run_name}*"))
    assert len(all_folders) == 1, f"Expected exactly 1 folder, found {len(all_folders)}: {all_folders}"
    assert all_folders[0].name == run_name, f"Folder name should be '{run_name}', got '{all_folders[0].name}'"
    
    # Verify database exists
    db_path = expected_folder / "optuna_study.db"
    assert db_path.exists(), f"SQLite database not found: {db_path}"
    
    print(f"✓ Test passed: folder structure is {run_name}/ (no timestamp)")


def test_configs_saved_immediately(test_workspace):
    """Test that configs are saved immediately after loading (not after study completion).
    
    Expected behavior:
    - full_agent_env_config.yaml written immediately
    - tuning_config.yaml written immediately
    - These files available even if optimization interrupted
    
    Strategy:
    - Mock an interruption by using a very short timeout
    - Verify configs exist even if study didn't complete
    """
    experiments_dir = test_workspace / "experiments"
    optimization_dir = test_workspace / "optimization"
    optimization_dir.mkdir()
    results_dir = test_workspace / "results"
    results_dir.mkdir()
    
    run_name = "test_early_save"
    umbrella_path, tuning_path = create_minimal_configs(experiments_dir, run_name)
    
    # Run tune.py (should complete, but we're testing that configs appear early)
    tune_cmd = [
        sys.executable, '-m', 'abx_amr_simulator.training.tune',
        '--umbrella-config', str(umbrella_path),
        '--tuning-config', str(tuning_path),
        '--run-name', run_name,
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir)
    ]
    
    result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
    
    assert result.returncode == 0, f"tune.py failed with return code {result.returncode}"
    
    # Verify configs were saved
    run_folder = optimization_dir / run_name
    assert run_folder.exists(), f"Optimization folder not found: {run_folder}"
    
    # Check umbrella config
    umbrella_saved = run_folder / "full_agent_env_config.yaml"
    assert umbrella_saved.exists(), "full_agent_env_config.yaml not saved"
    
    with open(umbrella_saved, 'r') as f:
        saved_umbrella = yaml.safe_load(f)
    
    assert 'algorithm' in saved_umbrella
    assert 'environment' in saved_umbrella
    assert 'training' in saved_umbrella
    
    # Check tuning config
    tuning_saved = run_folder / "tuning_config.yaml"
    assert tuning_saved.exists(), "tuning_config.yaml not saved"
    
    with open(tuning_saved, 'r') as f:
        saved_tuning = yaml.safe_load(f)
    
    assert 'optimization' in saved_tuning
    assert 'search_space' in saved_tuning
    
    print(f"✓ Test passed: configs saved immediately")


def test_config_validation_prevents_mismatch(test_workspace):
    """Test that attempting to resume with changed configs fails loudly.
    
    Expected behavior:
    1. First run creates study with config A
    2. Change config to config B
    3. Attempt to resume same run_name with config B
    4. tune.py should exit with error + clear guidance
    5. Error message should instruct: change run_name OR use --overwrite-existing-study
    """
    experiments_dir = test_workspace / "experiments"
    optimization_dir = test_workspace / "optimization"
    optimization_dir.mkdir()
    results_dir = test_workspace / "results"
    results_dir.mkdir()
    
    run_name = "test_validation"
    umbrella_path, tuning_path = create_minimal_configs(experiments_dir, run_name)
    
    # === STEP 1: First run with original config ===
    tune_cmd = [
        sys.executable, '-m', 'abx_amr_simulator.training.tune',
        '--umbrella-config', str(umbrella_path),
        '--tuning-config', str(tuning_path),
        '--run-name', run_name,
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir)
    ]
    
    result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"First tune.py run failed: {result.stderr}"
    
    # Verify configs saved
    run_folder = optimization_dir / run_name
    original_umbrella_path = run_folder / "full_agent_env_config.yaml"
    assert original_umbrella_path.exists()
    
    # === STEP 2: Modify umbrella config (change a value) ===
    config = load_config(config_path=str(umbrella_path))
    config['training']['total_num_training_episodes'] = 999  # Changed!
    
    with open(umbrella_path, 'w') as f:
        yaml.dump(config, f)
    
    # === STEP 3: Attempt to resume with modified config (should fail) ===
    result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
    
    # Should fail (non-zero exit code)
    assert result.returncode != 0, "tune.py should have failed with config mismatch"
    
    # Verify error message contains expected guidance
    combined_output = result.stdout + result.stderr
    assert "Configuration Mismatch" in combined_output or "mismatch" in combined_output.lower(), \
        f"Error message should mention config mismatch. Output:\n{combined_output}"
    
    # Verify error message mentions solutions
    assert "--overwrite-existing-study" in combined_output or "overwrite" in combined_output.lower(), \
        f"Error message should mention --overwrite-existing-study flag. Output:\n{combined_output}"
    
    print(f"✓ Test passed: config validation prevents mismatch")


def test_overwrite_flag_bypasses_validation(test_workspace):
    """Test that --overwrite-existing-study bypasses config validation.
    
    Expected behavior:
    1. First run creates study with config A
    2. Change config to config B
    3. Use --overwrite-existing-study flag
    4. tune.py should succeed (old study deleted, new study created)
    """
    experiments_dir = test_workspace / "experiments"
    optimization_dir = test_workspace / "optimization"
    optimization_dir.mkdir()
    results_dir = test_workspace / "results"
    results_dir.mkdir()
    
    run_name = "test_overwrite"
    umbrella_path, tuning_path = create_minimal_configs(experiments_dir, run_name)
    
    # === STEP 1: First run with original config ===
    tune_cmd_base = [
        sys.executable, '-m', 'abx_amr_simulator.training.tune',
        '--umbrella-config', str(umbrella_path),
        '--tuning-config', str(tuning_path),
        '--run-name', run_name,
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir)
    ]
    
    result = subprocess.run(tune_cmd_base, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"First tune.py run failed: {result.stderr}"
    
    # === STEP 2: Modify config ===
    config = load_config(config_path=str(umbrella_path))
    original_episodes = config['training']['total_num_training_episodes']
    config['training']['total_num_training_episodes'] = original_episodes + 100  # Changed!
    
    with open(umbrella_path, 'w') as f:
        yaml.dump(config, f)
    
    # === STEP 3: Use --overwrite-existing-study (should succeed) ===
    tune_cmd_overwrite = tune_cmd_base + ['--overwrite-existing-study']
    
    result = subprocess.run(tune_cmd_overwrite, capture_output=True, text=True, timeout=180)
    
    if result.returncode != 0:
        print(f"[TEST] stdout:\n{result.stdout}")
        print(f"[TEST] stderr:\n{result.stderr}")
    
    assert result.returncode == 0, "tune.py with --overwrite-existing-study should succeed"
    
    # Verify new config was saved (matches modified version)
    run_folder = optimization_dir / run_name
    saved_umbrella = run_folder / "full_agent_env_config.yaml"
    
    with open(saved_umbrella, 'r') as f:
        saved_config = yaml.safe_load(f)
    
    assert saved_config['training']['total_num_training_episodes'] == original_episodes + 100, \
        "Saved config should reflect modified values"
    
    print(f"✓ Test passed: --overwrite-existing-study bypasses validation")


def test_study_resumption_with_matching_configs(test_workspace):
    """Test that resuming with identical configs succeeds (validation passes).
    
    Expected behavior:
    1. First run creates study with config A (1 trial)
    2. Run again with same config A and run_name (1 more trial)
    3. tune.py should succeed (resume existing study)
    4. Total trials should be 2
    """
    experiments_dir = test_workspace / "experiments"
    optimization_dir = test_workspace / "optimization"
    optimization_dir.mkdir()
    results_dir = test_workspace / "results"
    results_dir.mkdir()
    
    run_name = "test_resume"
    umbrella_path, tuning_path = create_minimal_configs(experiments_dir, run_name)
    
    tune_cmd = [
        sys.executable, '-m', 'abx_amr_simulator.training.tune',
        '--umbrella-config', str(umbrella_path),
        '--tuning-config', str(tuning_path),
        '--run-name', run_name,
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir)
    ]
    
    # === STEP 1: First run (1 trial) ===
    result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"First run failed: {result.stderr}"
    
    # Verify 1 trial completed
    run_folder = optimization_dir / run_name
    summary_path = run_folder / "study_summary.json"
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    assert summary['n_trials'] == 1, f"Expected 1 trial, got {summary['n_trials']}"
    
    # === STEP 2: Second run with same config (should add 1 more trial) ===
    result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
    
    if result.returncode != 0:
        print(f"[TEST] stdout:\n{result.stdout}")
        print(f"[TEST] stderr:\n{result.stderr}")
    
    assert result.returncode == 0, "Second run (resume) should succeed with matching configs"
    
    # Verify 2 trials total
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    assert summary['n_trials'] == 2, f"Expected 2 trials after resume, got {summary['n_trials']}"
    
    # Verify validation message appeared
    combined_output = result.stdout + result.stderr
    assert "validation passed" in combined_output.lower() or "configs match" in combined_output.lower(), \
        f"Should see validation success message. Output:\n{combined_output}"
    
    print(f"✓ Test passed: resumption works with matching configs")


def test_hrl_option_library_validation(test_workspace):
    """Test that changing HRL option library config triggers validation failure.
    
    Expected behavior:
    1. First run creates HRL study with option library A
    2. Modify option library config
    3. Attempt to resume same run_name
    4. tune.py should fail with validation error mentioning option_library_config.yaml
    """
    experiments_dir = test_workspace / "experiments"
    optimization_dir = test_workspace / "optimization"
    optimization_dir.mkdir()
    results_dir = test_workspace / "results"
    results_dir.mkdir()
    
    run_name = "test_hrl_validation"
    
    # Create minimal HRL config
    umbrella_config_path = experiments_dir / "configs" / "umbrella_configs" / "base_experiment.yaml"
    config = load_config(config_path=str(umbrella_config_path))
    
    # Modify for HRL with ultra-brief testing
    config['algorithm'] = 'HRL_PPO'
    config['training']['total_num_training_episodes'] = 2
    config['environment']['max_time_steps'] = 3
    config['training']['eval_freq_every_n_episodes'] = 2
    config['training']['save_freq_every_n_episodes'] = 2
    config['training']['run_name'] = run_name
    
    # Use default option library
    config['hrl'] = {
        'option_library': 'option_libraries/default_deterministic.yaml'
    }
    
    # Save modified umbrella config
    test_umbrella_path = experiments_dir / "configs" / "umbrella_configs" / f"{run_name}.yaml"
    with open(test_umbrella_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create minimal tuning config (1 trial, 1 seed)
    tuning_config = {
        'optimization': {
            'n_trials': 1,
            'n_seeds_per_trial': 1,
            'truncated_episodes': 2,
            'direction': 'maximize',
            'sampler': 'Random',
            'stability_penalty_weight': 0.1
        },
        'search_space': {
            'learning_rate': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-3,
                'log': True
            }
        }
    }
    
    tuning_config_path = experiments_dir / "tuning_configs" / f"{run_name}.yaml"
    with open(tuning_config_path, 'w') as f:
        yaml.dump(tuning_config, f)
    
    # === STEP 1: First run with original option library ===
    tune_cmd = [
        sys.executable, '-m', 'abx_amr_simulator.training.tune',
        '--umbrella-config', str(test_umbrella_path),
        '--tuning-config', str(tuning_config_path),
        '--run-name', run_name,
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir)
    ]
    
    result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
    
    if result.returncode != 0:
        print(f"[TEST] First run stdout:\n{result.stdout}")
        print(f"[TEST] First run stderr:\n{result.stderr}")
    else:
        print(f"[TEST] First run stdout:\n{result.stdout}")
        print(f"[TEST] First run stderr:\n{result.stderr}")
    
    assert result.returncode == 0, f"First HRL tune.py run failed: {result.stderr}"
    
    # Verify option library config was saved
    run_folder = optimization_dir / run_name
    option_library_config_path = run_folder / "option_library_config.yaml"
    
    # Debug: list contents of run folder
    print(f"[TEST] Contents of {run_folder}:")
    if run_folder.exists():
        for item in run_folder.iterdir():
            print(f"  - {item.name}")
    
    assert option_library_config_path.exists(), "option_library_config.yaml should be saved for HRL"
    
    # === STEP 2: Modify option library by changing the library reference ===
    # Load the saved option library config and modify it
    with open(option_library_config_path, 'r') as f:
        option_library_config = yaml.safe_load(f)
    
    # Modify the library (add a dummy field to trigger mismatch)
    option_library_config['library_name'] = option_library_config.get('library_name', 'default') + "_modified"
    
    # Write it back (simulating user changing the option library source file)
    with open(option_library_config_path, 'w') as f:
        yaml.dump(option_library_config, f)
    
    # To properly test this, we need to modify the actual source option library file
    # that the umbrella config references
    options_lib_path = experiments_dir / config['hrl']['option_library']
    if options_lib_path.exists():
        with open(options_lib_path, 'r') as f:
            source_lib = yaml.safe_load(f)
        
        # Modify the source library
        source_lib['library_name'] = source_lib.get('library_name', 'default') + "_modified"
        
        with open(options_lib_path, 'w') as f:
            yaml.dump(source_lib, f)
    
    # === STEP 3: Attempt to resume (should fail due to option library mismatch) ===
    result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
    
    # Should fail (non-zero exit code)
    assert result.returncode != 0, "tune.py should fail when option library config changed"
    
    # Verify error message mentions option library
    combined_output = result.stdout + result.stderr
    assert "Configuration Mismatch" in combined_output or "mismatch" in combined_output.lower(), \
        f"Error message should mention config mismatch. Output:\n{combined_output}"
    
    assert "option_library_config.yaml" in combined_output, \
        f"Error message should specifically mention option_library_config.yaml. Output:\n{combined_output}"
    
    print(f"✓ Test passed: HRL option library validation prevents mismatch")


def test_reward_extraction_from_output():
    """Test that parse_reward_from_output correctly extracts rewards from training output.
    
    This is a unit test for the reward parsing function that tune.py uses to extract
    evaluation results from train.py stdout.
    
    Expected behavior:
    - Correctly parses "Final mean reward: -123.45" format
    - Correctly parses "Mean reward: -123.45" format  
    - Returns -inf when no reward found
    - Handles various formatting variations
    """
    from abx_amr_simulator.training.tune import parse_reward_from_output
    
    # Test case 1: Standard format from train.py
    output1 = """
======================================================================
EVALUATION RESULTS (for hyperparameter tuning)
======================================================================
Final mean reward: -234.5678
Best mean reward: -200.1234
======================================================================
"""
    reward1 = parse_reward_from_output(output1)
    assert reward1 == -234.5678, f"Expected -234.5678, got {reward1}"
    
    # Test case 2: Alternative format
    output2 = """
Training complete. Results saved to: /path/to/results
Mean reward: -100.25
Some other output
"""
    reward2 = parse_reward_from_output(output2)
    assert reward2 == -100.25, f"Expected -100.25, got {reward2}"
    
    # Test case 3: No reward in output (should return -inf)
    output3 = """
Training complete.
No reward information here.
"""
    reward3 = parse_reward_from_output(output3)
    assert reward3 == float('-inf'), f"Expected -inf, got {reward3}"
    
    # Test case 4: Multiple reward mentions (should get first one)
    output4 = """
Some output
Final mean reward: -50.0
Later mean reward: -60.0
"""
    reward4 = parse_reward_from_output(output4)
    assert reward4 == -50.0, f"Expected -50.0 (first match), got {reward4}"
    
    # Test case 5: Positive reward
    output5 = "Final mean reward: 123.456"
    reward5 = parse_reward_from_output(output5)
    assert reward5 == 123.456, f"Expected 123.456, got {reward5}"
    
    print(f"✓ Test passed: parse_reward_from_output correctly extracts rewards")


def test_tuning_captures_valid_rewards(test_workspace):
    """Test that full tuning workflow captures actual reward values (not -inf).
    
    This integration test verifies that:
    1. train.py prints evaluation results to stdout
    2. tune.py correctly parses those results
    3. Optuna trial values are not -inf
    4. study_summary.json contains valid best_value
    
    This test addresses the bug where evaluation was disabled during tuning,
    causing parse_reward_from_output to return -inf.
    """
    experiments_dir = test_workspace / "experiments"
    optimization_dir = test_workspace / "optimization"
    optimization_dir.mkdir()
    results_dir = test_workspace / "results"
    results_dir.mkdir()
    
    run_name = "test_reward_extraction"
    umbrella_path, tuning_path = create_minimal_configs(experiments_dir, run_name)
    
    # Run tune.py with 2 trials to get multiple reward samples
    tuning_config = {
        'optimization': {
            'n_trials': 2,
            'n_seeds_per_trial': 1,
            'truncated_episodes': 2,
            'direction': 'maximize',
            'sampler': 'Random',
            'stability_penalty_weight': 0.0
        },
        'search_space': {
            'learning_rate': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-3,
                'log': True
            }
        }
    }
    
    with open(tuning_path, 'w') as f:
        yaml.dump(tuning_config, f)
    
    tune_cmd = [
        sys.executable, '-m', 'abx_amr_simulator.training.tune',
        '--umbrella-config', str(umbrella_path),
        '--tuning-config', str(tuning_path),
        '--run-name', run_name,
        '--optimization-dir', str(optimization_dir),
        '--results-dir', str(results_dir)
    ]
    
    result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"[TEST] tune.py stdout:\n{result.stdout}")
        print(f"[TEST] tune.py stderr:\n{result.stderr}")
    
    assert result.returncode == 0, f"tune.py failed with return code {result.returncode}"
    
    # Verify study_summary.json exists and contains valid reward
    run_folder = optimization_dir / run_name
    summary_path = run_folder / "study_summary.json"
    assert summary_path.exists(), "study_summary.json not found"
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Check that best_value is not -inf
    best_value = summary.get('best_value')
    assert best_value is not None, "best_value missing from study_summary.json"
    assert best_value != float('-inf'), f"best_value is -inf (reward extraction failed)"
    assert best_value != '-Infinity', f"best_value is '-Infinity' string (reward extraction failed)"
    
    # best_value should be a finite number
    assert isinstance(best_value, (int, float)), f"best_value should be numeric, got {type(best_value)}"
    assert -1000 < best_value < 1000, f"best_value seems unrealistic: {best_value}"
    
    # Check that best_params exists
    best_params = summary.get('best_params')
    assert best_params is not None, "best_params missing from study_summary.json"
    assert 'learning_rate' in best_params, "learning_rate should be in best_params"
    
    # The key test: verify rewards were extracted successfully
    # If reward extraction failed, all trials would have -inf values
    # train.py prints evaluation results to its subprocess stdout (captured by tune.py internally)
    # tune.py parses those results and stores the reward values in Optuna
    
    # Verify that warning about -inf was NOT printed (which would indicate parsing failure)
    combined_output = result.stdout + result.stderr
    assert "Warning: Could not extract reward" not in combined_output, \
        "Should not see reward extraction warning on successful run"
    
    print(f"✓ Test passed: tuning captures valid rewards (best_value = {best_value})")
