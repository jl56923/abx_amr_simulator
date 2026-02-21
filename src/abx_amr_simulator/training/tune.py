"""
Hyperparameter tuning script using Optuna for ABXAMREnv agents.

Usage:
    # Basic tuning run
    python -m abx_amr_simulator.training.tune \\
        --umbrella-config /abs/path/to/base_experiment.yaml \\
        --tuning-config /abs/path/to/ppo_tuning.yaml \\
        --run-name exp_tuning_test
    
    # With custom optimization directory
    python -m abx_amr_simulator.training.tune \\
        --umbrella-config base_experiment.yaml \\
        --tuning-config ppo_tuning.yaml \\
        --run-name exp1_tuning \\
        --optimization-dir /path/to/workspace/optimization
    
    # With subconfig and param overrides
    python -m abx_amr_simulator.training.tune \\
        --umbrella-config base_experiment.yaml \\
        --tuning-config ppo_tuning.yaml \\
        --run-name exp1_tuning \\
        -s environment-subconfig=/abs/path/custom_env.yaml \\
        -p "reward_calculator.lambda_weight=0.5"
    
    # Skip if tuning already completed
    python -m abx_amr_simulator.training.tune \\
        --umbrella-config base_experiment.yaml \\
        --tuning-config ppo_tuning.yaml \\
        --run-name exp1_tuning \\
        --skip-if-exists
    
    # Continue existing study (default behavior)
    python -m abx_amr_simulator.training.tune \\
        --umbrella-config base_experiment.yaml \\
        --tuning-config ppo_tuning.yaml \\
        --run-name exp1_tuning
    
    # Start fresh study (ignore existing trials)
    python -m abx_amr_simulator.training.tune \\
        --umbrella-config base_experiment.yaml \\
        --tuning-config ppo_tuning.yaml \\
        --run-name exp1_tuning \\
        --overwrite-existing-study

    # Parallel tuning with PostgreSQL storage (shared study)
    python -m abx_amr_simulator.training.tune \\
        --umbrella-config /abs/path/to/base_experiment.yaml \\
        --tuning-config /abs/path/to/ppo_tuning.yaml \\
        --run-name exp1_tuning \\
        --use-postgres \\
        --study-name exp1_tuning

OPTIMIZATION REGISTRY (.optimization_completed.txt):
    Similar to training registry, tracks completed tuning runs:
        <optimization_dir>/.optimization_completed.txt
    
    Format: CSV with columns (run_name, timestamp, best_value, n_trials)
    
    Purpose:
    - Tracks successfully completed tuning runs
    - Enables --skip-if-exists to avoid re-running completed optimizations
    - Enables train.py to load best params by experiment name (resolves to most recent timestamp)
    
    Workflow:
    1. Before tuning: Check if run_name already in registry (if --skip-if-exists set)
    2. During tuning: Optuna study persisted to SQLite database (optuna_study.db)
    3. After tuning: Add run_name + timestamp + best_value + n_trials to registry
    4. During training: train.py loads best_params.json from most recent run_name timestamp
    
    Study Persistence:
    - Each optimization run uses folder: <run_name>/optuna_study.db (no timestamp in path)
    - Default: Continues existing study if database found (resume interrupted runs)
    - With --overwrite-existing-study: Deletes old database and starts fresh
    - Registry tracks completion timestamps for train.py lookup (most recent run)

TUNING CONFIG STRUCTURE:
    tuning_config:
      optimization:
        n_trials: 50                    # Number of Optuna trials
        n_seeds_per_trial: 3            # Seeds to aggregate per trial
        truncated_episodes: 50          # Reduced episodes for fast evaluation
        direction: maximize             # maximize or minimize
        sampler: TPE                    # TPE, Random, or CMAES
        stability_penalty_weight: 0.1   # λ for mean-variance penalty (0.0 = pure mean, higher = penalize inconsistency)
      
      search_space:
        learning_rate:
          type: float
          low: 1.0e-5
          high: 1.0e-3
          log: true
        
        n_steps:
          type: int
          low: 128
          high: 2048
          step: 128
        
        gamma:
          type: float
          low: 0.9
          high: 0.999
        
        ent_coef:
          type: categorical
          choices: [0.0, 0.01, 0.1]

"""

import argparse
import json
import os
import sys
import tempfile
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import optuna
import yaml
import numpy as np

from abx_amr_simulator.utils import (
    load_config,
    apply_param_overrides,
    apply_subconfig_overrides,
    save_training_config,
    save_option_library_config,
)
from abx_amr_simulator.utils.registry import (
    load_registry,
    update_registry,
    validate_and_clean_registry,
)


def load_and_save_hrl_option_library(
    config: Dict[str, Any],
    umbrella_config_path: str,
    optimization_dir: str
) -> Optional[Dict[str, Any]]:
    """Load HRL option library YAML and save to optimization directory if HRL algorithm.
    
    For validation purposes, we save the raw option library YAML file (not the fully
    resolved config, which would require creating an environment). This is sufficient
    to detect if the user changed the option library between runs.
    
    Args:
        config: Resolved umbrella config
        umbrella_config_path: Path to umbrella config (for resolving relative paths)
        optimization_dir: Directory to save option library config
    
    Returns:
        Dict with option library config (or None if not HRL)
    """
    algorithm = config.get('algorithm', 'PPO')
    
    if algorithm not in ['HRL_PPO', 'HRL_RPPO']:
        return None
    
    # Resolve option library path (same logic as train.py)
    hrl_config = config.get('hrl', {})
    option_library_relative = hrl_config.get('option_library')
    
    if not option_library_relative:
        print("Warning: HRL algorithm selected but no option_library specified in hrl config")
        return None
    
    if 'option_library_path' not in config:
        options_folder = config.get('options_folder_location', '../../options')
        config_dir = Path(umbrella_config_path).parent
        resolved_path = (config_dir / options_folder / option_library_relative).resolve()
        config['option_library_path'] = str(resolved_path)
    
    # Load raw option library YAML
    option_library_path = config['option_library_path']
    
    if not os.path.exists(option_library_path):
        print(f"Warning: Option library file not found: {option_library_path}")
        return None
    
    with open(option_library_path, 'r') as f:
        option_library_config = yaml.safe_load(f)
    
    # Save option library config to optimization directory
    saved_path = os.path.join(optimization_dir, 'option_library_config.yaml')
    with open(saved_path, 'w') as f:
        yaml.dump(option_library_config, f, default_flow_style=False)
    
    print(f"✓ Saved option_library_config.yaml")
    
    return option_library_config


def _normalize_config(config: Any) -> Any:
    """Recursively normalize a config dict to handle YAML parsing differences.
    
    Converts dicts to a canonical form with sorted keys to ensure consistent 
    comparison regardless of key ordering from YAML parsing.
    
    Args:
        config: Configuration to normalize (dict, list, or scalar)
    
    Returns:
        Normalized config with consistent key ordering
    """
    if isinstance(config, dict):
        return {k: _normalize_config(v) for k, v in sorted(config.items())}
    elif isinstance(config, list):
        return [_normalize_config(item) for item in config]
    else:
        return config


def validate_configs_match_existing(
    current_umbrella_config: Dict[str, Any],
    current_tuning_config: Dict[str, Any],
    current_option_library_config: Optional[Dict[str, Any]],
    optimization_dir: str,
    run_name: str
):
    """Validate that current configs match existing saved configs.
    
    Fails loudly if mismatch detected, instructing user to either change
    run_name or use --overwrite-existing-study flag.
    
    Args:
        current_umbrella_config: Resolved umbrella config from current run
        current_tuning_config: Tuning config from current run
        current_option_library_config: Raw option library config (or None if not HRL)
        optimization_dir: Directory containing saved configs
        run_name: Name of the optimization run
    
    Raises:
        SystemExit: If configs don't match
    """
    # Check if saved configs exist
    umbrella_path = os.path.join(optimization_dir, 'full_agent_env_config.yaml')
    tuning_path = os.path.join(optimization_dir, 'tuning_config.yaml')
    options_path = os.path.join(optimization_dir, 'option_library_config.yaml')
    
    if not os.path.exists(umbrella_path):
        # No saved config means fresh start (edge case: folder exists but no config)
        print("⚠️  Warning: optimization folder exists but no saved config found. Proceeding as new study.")
        return
    
    # Load saved configs
    with open(umbrella_path, 'r') as f:
        saved_umbrella_config = yaml.safe_load(f)
    
    with open(tuning_path, 'r') as f:
        saved_tuning_config = yaml.safe_load(f)
    
    saved_option_library_config = None
    if os.path.exists(options_path):
        with open(options_path, 'r') as f:
            saved_option_library_config = yaml.safe_load(f)
    
    # Normalize both current and saved configs to handle YAML parsing differences
    # (key ordering can differ but content is identical)
    normalized_current_umbrella = _normalize_config(current_umbrella_config)
    normalized_saved_umbrella = _normalize_config(saved_umbrella_config)
    normalized_current_tuning = _normalize_config(current_tuning_config)
    normalized_saved_tuning = _normalize_config(saved_tuning_config)
    normalized_current_options = (
        _normalize_config(current_option_library_config) 
        if current_option_library_config is not None else None
    )
    normalized_saved_options = (
        _normalize_config(saved_option_library_config) 
        if saved_option_library_config is not None else None
    )
    
    # Compare configs
    configs_match = True
    mismatches = []
    
    if normalized_current_umbrella != normalized_saved_umbrella:
        configs_match = False
        mismatches.append(("full_agent_env_config.yaml", current_umbrella_config, saved_umbrella_config))
    
    if normalized_current_tuning != normalized_saved_tuning:
        configs_match = False
        mismatches.append(("tuning_config.yaml", current_tuning_config, saved_tuning_config))
    
    # Check option library if applicable
    if normalized_current_options is not None or normalized_saved_options is not None:
        if normalized_current_options != normalized_saved_options:
            configs_match = False
            mismatches.append(("option_library_config.yaml", current_option_library_config, saved_option_library_config))
    
    if not configs_match:
        print(f"\n{'='*70}")
        print("❌ ERROR: Configuration Mismatch Detected")
        print(f"{'='*70}")
        print(f"\nYou are attempting to resume optimization run '{run_name}',")
        print(f"but the current configuration does not match the saved configuration.")
        print(f"\nMismatched files and their differences:")
        print(f"{'-'*70}")
        
        for filename, current, saved in mismatches:
            print(f"\n📄 {filename}")
            print(f"\n  CURRENT (from command line):")
            print(f"  {yaml.dump(current, default_flow_style=False, sort_keys=False).rstrip()}")
            print(f"\n  SAVED (from {optimization_dir}):")
            print(f"  {yaml.dump(saved, default_flow_style=False, sort_keys=False).rstrip()}")
            print(f"\n  {'-'*70}")
        
        print(f"\nThis likely means you changed values in your umbrella config,")
        print(f"tuning config, or options library since starting this optimization.")
        print(f"\nTo proceed, you must either:")
        print(f"\n  1. Change the run name to create a new optimization study:")
        print(f"       --run-name {run_name}_v2")
        print(f"\n  2. Overwrite the existing study (discard old trials):")
        print(f"       --overwrite-existing-study")
        print(f"\n{'='*70}\n")
        sys.exit(1)
    
    print("✓ Configuration validation passed: current configs match saved configs")


def build_storage_url(
    use_postgres: bool,
    run_name: str,
    optimization_dir: str
) -> str:
    """Build Optuna storage URL based on backend choice.

    Args:
        use_postgres: If True, use PostgreSQL; else use SQLite
        run_name: Experiment run name (used for SQLite path)
        optimization_dir: Optimization directory (used for SQLite path)

    Returns:
        Storage URL string (postgresql://... or sqlite:///...)
    """
    if use_postgres:
        pg_username = os.environ.get("PG_USERNAME", os.environ.get("USER", "postgres"))
        pg_port = os.environ.get("PG_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "optuna_tuning")
        storage_url = f"postgresql://{pg_username}@localhost:{pg_port}/{db_name}"
        print(f"Using PostgreSQL storage: {storage_url}")
        return storage_url

    storage_path = os.path.join(optimization_dir, "optuna_study.db")
    storage_url = f"sqlite:///{storage_path}"
    print(f"Using SQLite storage: {storage_url}")
    return storage_url


def suggest_hyperparameters(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest hyperparameters for the current trial based on search space config.
    
    Args:
        trial: Optuna trial object
        search_space: Dictionary defining parameter types and ranges
    
    Returns:
        Dict mapping parameter names to suggested values
    
    Raises:
        ValueError: If parameter type is unknown
    """
    suggested_params = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config.get('type')
        
        if param_type == 'float':
            suggested_params[param_name] = trial.suggest_float(
                name=param_name,
                low=param_config['low'],
                high=param_config['high'],
                log=param_config.get('log', False)
            )
        elif param_type == 'int':
            suggested_params[param_name] = trial.suggest_int(
                name=param_name,
                low=param_config['low'],
                high=param_config['high'],
                step=param_config.get('step', 1)
            )
        elif param_type == 'categorical':
            suggested_params[param_name] = trial.suggest_categorical(
                name=param_name,
                choices=param_config['choices']
            )
        else:
            raise ValueError(f"Unknown parameter type '{param_type}' for '{param_name}'")
    
    return suggested_params


def run_training_trial(
    umbrella_config_path: str,
    suggested_params: Dict[str, Any],
    truncated_episodes: int,
    seeds: List[int],
    subconfig_overrides: Dict[str, str],
    base_param_overrides: Dict[str, Any],
    results_dir: str,
    trial_run_prefix: str
) -> List[float]:
    """Run training for multiple seeds with suggested hyperparameters.
    
    Args:
        umbrella_config_path: Path to umbrella config file
        suggested_params: Hyperparameters suggested by Optuna
        truncated_episodes: Number of episodes to train (reduced for speed)
        seeds: List of seeds to run
        subconfig_overrides: Subconfig overrides from CLI
        base_param_overrides: Parameter overrides from CLI (not including suggested params)
        results_dir: Base directory for results
    
    Returns:
        List of final mean rewards (one per seed)
    """
    rewards = []
    
    # Determine working directory: parent of src/abx_amr_simulator (the package root)
    import abx_amr_simulator
    package_root = Path(abx_amr_simulator.__file__).parent.parent.parent
    
    for seed in seeds:
        # Merge suggested params with base overrides and seed
        param_overrides = {**base_param_overrides, **suggested_params}
        param_overrides['training.seed'] = seed
        param_overrides['training.total_num_training_episodes'] = truncated_episodes
        param_overrides['training.run_name'] = f"{trial_run_prefix}_seed{seed}"
        
        # Optimize evaluation for tuning: do ONE final evaluation to get reward metric
        # Set eval_freq to trigger only at the very end (truncated_episodes + 1 ensures one eval at end)
        param_overrides['training.eval_freq_every_n_episodes'] = truncated_episodes  # Eval at end only
        param_overrides['training.save_freq_every_n_episodes'] = 999999  # Save only final model
        param_overrides['training.log_patient_trajectories'] = False  # Disable trajectory logging
        
        # Build command to run training
        cmd = [
            sys.executable, '-m', 'abx_amr_simulator.training.train',
            '--umbrella-config', umbrella_config_path,
            '--results-dir', results_dir
        ]
        
        # Add subconfig overrides
        for key, value in subconfig_overrides.items():
            cmd.extend(['-s', f"{key}={value}"])
        
        # Add parameter overrides
        for key, value in param_overrides.items():
            cmd.extend(['-p', f"{key}={value}"])
        
        # Run training subprocess with correct working directory
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout per seed
                cwd=str(package_root)  # Set working directory to package root
            )
            
            if result.returncode != 0:
                print(f"\n{'='*80}")
                print(f"WARNING: Training failed for seed {seed}")
                print(f"Return code: {result.returncode}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Working directory: {package_root}")
                print(f"\nstderr (last 20 lines):")
                stderr_lines = result.stderr.split('\n')
                for line in stderr_lines[-20:]:
                    if line.strip():
                        print(f"  {line}")
                print(f"\nstdout (last 30 lines):")
                stdout_lines = result.stdout.split('\n')
                for line in stdout_lines[-30:]:
                    if line.strip():
                        print(f"  {line}")
                print(f"{'='*80}\n")
                rewards.append(float('-inf'))
                continue
            
            # Parse output to extract final mean reward
            reward = parse_reward_from_output(result.stdout)
            if reward == float('-inf'):
                print(f"\nWarning: Could not extract reward from training output for seed {seed}")
                print(f"stdout (last 20 lines):")
                stdout_lines = result.stdout.split('\n')
                for line in stdout_lines[-20:]:
                    if line.strip():
                        print(f"  {line}")
            rewards.append(reward)
            
        except subprocess.TimeoutExpired:
            print(f"Warning: Training timed out for seed {seed}")
            rewards.append(float('-inf'))
        except Exception as e:
            print(f"Warning: Training raised exception for seed {seed}: {e}")
            rewards.append(float('-inf'))
    
    return rewards


def parse_reward_from_output(output: str) -> float:
    """Extract final mean reward from training output.
    
    Looks for lines like:
    - "Mean reward: -123.45"
    - "Final mean reward: -123.45"
    - "mean_reward: -123.45"
    
    Args:
        output: stdout from training script
    
    Returns:
        float: Extracted reward value, or -inf if not found
    """
    for line in output.split('\n'):
        line_lower = line.lower().strip()
        
        if 'mean reward' in line_lower or 'mean_reward' in line_lower:
            try:
                # Extract number after colon or equals
                if ':' in line:
                    parts = line.split(':')
                elif '=' in line:
                    parts = line.split('=')
                else:
                    continue
                
                if len(parts) >= 2:
                    return float(parts[-1].strip())
            except (ValueError, IndexError):
                continue
    
    # If we didn't find a reward, return -inf
    return float('-inf')


def create_objective_function(
    umbrella_config_path: str,
    tuning_config: Dict[str, Any],
    subconfig_overrides: Dict[str, str],
    base_param_overrides: Dict[str, Any],
    results_dir: str,
    base_seed: int,
    tuning_run_name: str
):
    """Create Optuna objective function closure.
    
    Args:
        umbrella_config_path: Path to umbrella config
        tuning_config: Tuning configuration dictionary
        subconfig_overrides: Subconfig overrides from CLI
        base_param_overrides: Parameter overrides from CLI
        results_dir: Base directory for results
        base_seed: Base seed for generating trial seeds
    
    Returns:
        Callable objective function for Optuna
    """
    optimization_config = tuning_config.get('optimization', {})
    search_space = tuning_config.get('search_space', {})
    
    n_seeds_per_trial = optimization_config.get('n_seeds_per_trial', 3)
    truncated_episodes = optimization_config.get('truncated_episodes', 50)
    stability_penalty_weight = optimization_config.get('stability_penalty_weight', 0.1)
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function with mean-variance penalty.
        
        Objective = mean(rewards) - λ × std(rewards)
        
        This balances reward maximization with consistency across seeds.
        λ=0.0 reduces to pure mean optimization; higher λ penalizes variance.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            float: Mean reward minus stability penalty (higher is better)
        """
        # Suggest hyperparameters
        suggested_params = suggest_hyperparameters(trial=trial, search_space=search_space)
        
        # Generate seeds for this trial (deterministic based on trial number)
        trial_seeds = [base_seed + trial.number * 1000 + i for i in range(n_seeds_per_trial)]
        
        # Run training for all seeds
        trial_run_prefix = f"{tuning_run_name}_trial{trial.number}"
        rewards = run_training_trial(
            umbrella_config_path=umbrella_config_path,
            suggested_params=suggested_params,
            truncated_episodes=truncated_episodes,
            seeds=trial_seeds,
            subconfig_overrides=subconfig_overrides,
            base_param_overrides=base_param_overrides,
            results_dir=results_dir,
            trial_run_prefix=trial_run_prefix
        )
        
        # Filter out failed trials (-inf)
        valid_rewards = [r for r in rewards if r != float('-inf')]
        
        if not valid_rewards:
            # All seeds failed
            return float('-inf')
        
        # Compute mean-variance penalty: mean(rewards) - λ * std(rewards)
        mean_reward = float(np.mean(valid_rewards))
        std_reward = float(np.std(valid_rewards))
        
        # Apply stability penalty (penalize high variance across seeds)
        objective_value = mean_reward - stability_penalty_weight * std_reward
        
        # Log components for debugging
        trial.set_user_attr('mean_reward', mean_reward)
        trial.set_user_attr('std_reward', std_reward)
        trial.set_user_attr('stability_penalty', stability_penalty_weight * std_reward)
        
        return objective_value
    
    return objective


def save_best_params(
    study: optuna.Study,
    tuning_config: Dict[str, Any],
    optimization_dir: str
):
    """Save best hyperparameters and study summary to optimization directory.
    
    Args:
        study: Completed Optuna study
        tuning_config: Tuning configuration (for metadata)
        optimization_dir: Directory to save results
    """
    # Save best parameters
    best_params = study.best_params
    best_params_path = os.path.join(optimization_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n✓ Saved best parameters to: {best_params_path}")
    
    # Save study summary
    summary = {
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'best_params': best_params,
        'study_name': study.study_name,
        'direction': study.direction.name,
    }
    
    summary_path = os.path.join(optimization_dir, 'study_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved study summary to: {summary_path}")
    
    # Save tuning config for reproducibility
    tuning_config_path = os.path.join(optimization_dir, 'tuning_config.yaml')
    with open(tuning_config_path, 'w') as f:
        yaml.dump(tuning_config, f, default_flow_style=False)
    
    print(f"✓ Saved tuning config to: {tuning_config_path}")


def main():
    parser = argparse.ArgumentParser(description='Tune RL agent hyperparameters using Optuna')
    parser.add_argument(
        '--umbrella-config',
        type=str,
        required=True,
        help='Path to umbrella config YAML file (absolute path required)'
    )
    parser.add_argument(
        '--tuning-config',
        type=str,
        required=True,
        help='Path to tuning config YAML file defining search space (absolute path required)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Name for this optimization run (used for directory and registry)'
    )
    parser.add_argument(
        '--optimization-dir',
        type=str,
        default=None,
        help='Directory where optimization results should be saved. If not specified, uses current working directory + "optimization"'
    )
    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help='Skip this tuning run if an optimization with the same run name already exists in the registry'
    )
    parser.add_argument(
        '--overwrite-existing-study',
        action='store_true',
        help='Start a new Optuna study from scratch instead of continuing an existing one. Default: continue existing study if found.'
    )
    parser.add_argument(
        '--use-postgres',
        action='store_true',
        default=False,
        help='Use PostgreSQL backend for parallel workers (requires PostgreSQL running locally)'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default=None,
        help='Shared Optuna study name (recommended for parallel workers)'
    )
    parser.add_argument(
        '-p',
        '--param-override',
        action='append',
        default=[],
        help='Change config values using dot notation. Example: -p reward_calculator.lambda_weight=0.8'
    )
    parser.add_argument(
        '-s',
        '--subconfig-override',
        action='append',
        default=[],
        help='Replace entire subconfig with explicit path. Example: -s environment-subconfig=/abs/path/to/custom_environment.yaml'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory where training results should be created during tuning trials (temporary). If not specified, uses system temp directory.'
    )
    parser.add_argument(
        '--discard-trial-results',
        action='store_true',
        help='Run Optuna trial training in a temporary directory and delete outputs after tuning'
    )
    
    args = parser.parse_args()
    
    # Enforce absolute paths
    if not os.path.isabs(args.umbrella_config):
        print("Error: --umbrella-config must be an absolute path")
        sys.exit(1)
    
    if not os.path.exists(args.umbrella_config):
        print(f"Error: Umbrella config file not found: {args.umbrella_config}")
        sys.exit(1)
    
    if not os.path.isabs(args.tuning_config):
        print("Error: --tuning-config must be an absolute path")
        sys.exit(1)
    
    if not os.path.exists(args.tuning_config):
        print(f"Error: Tuning config file not found: {args.tuning_config}")
        sys.exit(1)

    if args.use_postgres and args.overwrite_existing_study:
        print(f"\n{'='*70}")
        print("ERROR: --overwrite-existing-study is unsafe with --use-postgres")
        print(f"{'='*70}")
        print("\nParallel workers can race to delete studies, causing data loss.")
        print("To start a fresh PostgreSQL study:")
        print("  1) Delete the study in a wrapper script before launching workers")
        print("  2) Remove --overwrite-existing-study from worker commands")
        print("\nUse --overwrite-existing-study only for single-worker SQLite runs.")
        print(f"{'='*70}\n")
        sys.exit(1)
    
    # Parse parameter overrides
    param_overrides = {}
    for override_str in args.param_override:
        if '=' not in override_str:
            print(f"Warning: Ignoring invalid parameter override format '{override_str}' (expected key=value)")
            continue
        key, value = override_str.split('=', 1)
        # Try to parse as number, boolean, or keep as string
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
        param_overrides[key] = value
    
    # Parse subconfig overrides
    subconfig_overrides = {}
    for override_str in args.subconfig_override:
        if '=' not in override_str or not override_str.endswith('.yaml'):
            print(f"Warning: Ignoring invalid subconfig override format '{override_str}'")
            continue
        key_part, path = override_str.split('=', 1)
        
        if not key_part.endswith('-subconfig'):
            print(f"Warning: Ignoring subconfig override with invalid key '{key_part}'")
            continue
        
        if not os.path.isabs(path):
            print(f"Error: Subconfig path for '{key_part}' must be absolute: {path}")
            sys.exit(1)
        if not os.path.exists(path):
            print(f"Error: Subconfig file not found: {path}")
            sys.exit(1)
        
        subconfig_overrides[key_part] = path
    
    # Determine optimization directory
    if args.optimization_dir:
        optimization_base_dir = args.optimization_dir
        if not os.path.isabs(optimization_base_dir):
            optimization_base_dir = os.path.abspath(optimization_base_dir)
    else:
        optimization_base_dir = os.path.join(os.getcwd(), 'optimization')
    
    # Create optimization base directory if it doesn't exist
    Path(optimization_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up registry
    completion_registry_path = os.path.join(optimization_base_dir, '.optimization_completed.txt')
    run_name = args.run_name
    
    # Check if optimization already exists (if --skip-if-exists flag is set)
    if args.skip_if_exists:
        # Validate and clean completion registry
        stale_entries = validate_and_clean_registry(
            completion_registry_path,
            optimization_base_dir,
            exclude_prefix=run_name
        )
        if stale_entries:
            print(f"\n⚠️  Registry cleanup: Removed {len(stale_entries)} stale entry/entries:")
            for entry in stale_entries:
                print(f"    - {entry[0]} (timestamp {entry[1]})")
            print()
        
        completed_prefixes = load_registry(completion_registry_path)
        
        if run_name in completed_prefixes:
            print(f"\n{'='*70}")
            print("SKIPPING: Optimization already completed")
            print(f"{'='*70}")
            print(f"Run name: {run_name}")
            print(f"Found in completion registry: {completion_registry_path}")
            print(f"\nTo force re-run, either:")
            print(f"  1. Remove the --skip-if-exists flag, or")
            print(f"  2. Delete the optimization folder(s), or")
            print(f"  3. Edit {completion_registry_path} to remove this entry\n")
            sys.exit(0)
    
    # Create optimization directory (no timestamp in folder name for study persistence)
    # Timestamp only used for registry tracking
    optimization_dir = os.path.join(optimization_base_dir, run_name)
    Path(optimization_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for registry only
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*70}")
    print("STARTING HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Run name: {run_name}")
    print(f"Optimization directory: {optimization_dir}")
    print(f"Registry timestamp: {timestamp}")
    print(f"Registry: {completion_registry_path}")
    print(f"{'='*70}\n")
    
    # Load configs
    print("Loading configurations...")
    umbrella_config = load_config(args.umbrella_config)
    
    with open(args.tuning_config, 'r') as f:
        tuning_config = yaml.safe_load(f)
    
    # Apply overrides to umbrella config
    if subconfig_overrides:
        print(f"\nApplying subconfig overrides...")
        for key, path in subconfig_overrides.items():
            # Extract config type from key (e.g., 'environment-subconfig' -> 'environment')
            config_type = key.replace('-subconfig', '')
            print(f"  {key}: {path}")
            
            # Load the new subconfig
            try:
                with open(path, 'r') as f:
                    subconfig = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading subconfig from {path}: {e}")
                sys.exit(1)
            
            # Place subconfig in the right location
            if config_type == 'environment':
                umbrella_config['environment'] = subconfig
            elif config_type == 'reward_calculator':
                umbrella_config['reward_calculator'] = subconfig
            elif config_type == 'patient_generator':
                umbrella_config['patient_generator'] = subconfig
            elif config_type == 'agent_algorithm':
                umbrella_config.update(subconfig)
            else:
                print(f"Error: Unknown subconfig type '{config_type}' (expected environment, reward_calculator, patient_generator, or agent_algorithm)")
                sys.exit(1)
    
    if param_overrides:
        umbrella_config = apply_param_overrides(umbrella_config, param_overrides)
    
    # Validate configs match existing if resuming (unless --overwrite-existing-study)
    umbrella_path = os.path.join(optimization_dir, 'full_agent_env_config.yaml')
    if os.path.exists(umbrella_path) and not args.overwrite_existing_study:
        print(f"\n{'='*70}")
        print("VALIDATING CONFIGURATION CONSISTENCY")
        print(f"{'='*70}")
        print(f"Existing optimization folder detected: {optimization_dir}")
        print(f"Checking if current configs match saved configs...\\n")
        
        # Load HRL option library if applicable (for validation)
        temp_option_library_config = load_and_save_hrl_option_library(
            config=umbrella_config,
            umbrella_config_path=args.umbrella_config,
            optimization_dir=tempfile.mkdtemp()  # Temp dir for validation only
        )
        
        # Validate configs match
        validate_configs_match_existing(
            current_umbrella_config=umbrella_config,
            current_tuning_config=tuning_config,
            current_option_library_config=temp_option_library_config,
            optimization_dir=optimization_dir,
            run_name=run_name
        )
        print(f"{'='*70}\\n")
    
    # Save resolved configs immediately (before optimization starts)
    config_already_saved = os.path.exists(umbrella_path)
    if args.use_postgres and config_already_saved:
        print("Configs already exist (saved by another worker), skipping save.")
    else:
        print(f"Saving resolved configuration to: {optimization_dir}")
        save_training_config(config=umbrella_config, run_dir=optimization_dir)
        print(f"✓ Saved full_agent_env_config.yaml")

        # Save tuning config for reproducibility (before optimization starts)
        tuning_config_path = os.path.join(optimization_dir, 'tuning_config.yaml')
        with open(tuning_config_path, 'w') as f:
            yaml.dump(tuning_config, f, default_flow_style=False)
        print(f"✓ Saved tuning_config.yaml")

        # Load and save HRL option library if applicable
        load_and_save_hrl_option_library(
            config=umbrella_config,
            umbrella_config_path=args.umbrella_config,
            optimization_dir=optimization_dir
        )
    
    # Extract optimization settings
    optimization_config = tuning_config.get('optimization', {})
    n_trials = optimization_config.get('n_trials', 50)
    direction = optimization_config.get('direction', 'maximize')
    sampler_name = optimization_config.get('sampler', 'TPE')
    
    # Create Optuna sampler
    if sampler_name == 'TPE':
        sampler = optuna.samplers.TPESampler()
    elif sampler_name == 'Random':
        sampler = optuna.samplers.RandomSampler()
    elif sampler_name == 'CMAES':
        sampler = optuna.samplers.CmaEsSampler()
    else:
        print(f"Warning: Unknown sampler '{sampler_name}', using TPE")
        sampler = optuna.samplers.TPESampler()
    
    # Determine results directory for training trials
    trial_results_dir_to_cleanup = None
    if args.discard_trial_results:
        results_dir = tempfile.mkdtemp(prefix='optuna_trials_')
        trial_results_dir_to_cleanup = results_dir
    elif args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(optimization_dir, 'trial_runs')
        Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Get base seed from umbrella config
    base_seed = umbrella_config.get('training', {}).get('seed', 42)
    
    storage_url = build_storage_url(
        use_postgres=args.use_postgres,
        run_name=run_name,
        optimization_dir=optimization_dir
    )
    storage_path = None

    # Determine whether to load existing study or start fresh
    load_if_exists = not args.overwrite_existing_study

    if not args.use_postgres:
        storage_path = os.path.join(optimization_dir, 'optuna_study.db')
        if args.overwrite_existing_study and os.path.exists(storage_path):
            print(f"\n⚠️  Overwriting existing study database: {storage_path}")
            os.remove(storage_path)
        elif load_if_exists and os.path.exists(storage_path):
            print(f"\n✓ Found existing study database, will continue: {storage_path}")

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=args.study_name or run_name,
        storage=storage_url,
        load_if_exists=load_if_exists
    )
    
    # Report study status
    n_existing_trials = len(study.trials)
    n_completed_trials_with_results = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    # Determine if this is a new study or a resumed study
    is_resumed_study = n_existing_trials > 0
    
    # For NEW studies only: check if we've already reached the target number of completed trials
    # For RESUMED studies: always allow adding n_trials more (configs already validated as matching)
    if not is_resumed_study and n_completed_trials_with_results >= n_trials:
        print(f"✓ Loaded existing study with {n_existing_trials} trial(s)")
        print(f"  Completed trials with results: {n_completed_trials_with_results}")
        print(f"  Target n_trials: {n_trials}")
        print(f"\n✓ Study has already reached the target! Skipping additional trials.\n")
        
        # Print results
        print(f"{'='*70}")
        print("OPTIMIZATION ALREADY COMPLETE")
        print(f"{'='*70}")
        print(f"Best value: {study.best_value:.4f}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Total trials completed: {n_completed_trials_with_results}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*70}\n")
        
        # Save results
        save_best_params(
            study=study,
            tuning_config=tuning_config,
            optimization_dir=optimization_dir
        )

        if trial_results_dir_to_cleanup:
            shutil.rmtree(trial_results_dir_to_cleanup, ignore_errors=True)
        
        storage_label = storage_path or storage_url
        print(f"✓ Study database at: {storage_label}")
        
        # Update registry
        update_registry(
            registry_path=completion_registry_path,
            run_name=run_name,
            timestamp=timestamp
        )
        print(f"\n✓ Recorded successful completion in registry: {completion_registry_path}\n")
        
        print(f"Optimization results saved to: {optimization_dir}")
        print(f"  - Study database: {storage_label}")
        print(f"  - Best params: {os.path.join(optimization_dir, 'best_params.json')}")
        print(f"  - Study summary: {os.path.join(optimization_dir, 'study_summary.json')}")
        sys.exit(0)
    
    # Not yet complete, continue or start optimization
    if is_resumed_study:
        print(f"✓ Loaded existing study with {n_existing_trials} trial(s)")
        print(f"  Completed trials with results: {n_completed_trials_with_results}")
        print(f"  Best value so far: {study.best_value:.4f}")
        # For resumed studies, always add n_trials more
        remaining_trials = n_trials
        print(f"  Will run {remaining_trials} additional trial(s) (total will be {n_existing_trials + n_trials})")
    else:
        print(f"✓ Starting new study")
        remaining_trials = n_trials
    
    # Create objective function
    objective = create_objective_function(
        umbrella_config_path=args.umbrella_config,
        tuning_config=tuning_config,
        subconfig_overrides=subconfig_overrides,
        base_param_overrides=param_overrides,
        results_dir=results_dir,
        base_seed=base_seed,
        tuning_run_name=run_name
    )
    
    # Run optimization
    print(f"\nStarting optimization with {remaining_trials} trials...")
    print(f"Direction: {direction}")
    print(f"Sampler: {sampler_name}")
    print(f"Seeds per trial: {optimization_config.get('n_seeds_per_trial', 3)}")
    print(f"Truncated episodes: {optimization_config.get('truncated_episodes', 50)}\n")
    
    study.optimize(objective, n_trials=remaining_trials)
    
    # Print results
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Total trials completed: {len(study.trials)}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    # Save results
    save_best_params(
        study=study,
        tuning_config=tuning_config,
        optimization_dir=optimization_dir
    )

    if trial_results_dir_to_cleanup:
        shutil.rmtree(trial_results_dir_to_cleanup, ignore_errors=True)
    
    storage_label = storage_path or storage_url
    print(f"✓ Study database saved to: {storage_label}")
    
    # Update registry
    update_registry(
        registry_path=completion_registry_path,
        run_name=run_name,
        timestamp=timestamp
    )
    print(f"\n✓ Recorded successful completion in registry: {completion_registry_path}\n")
    
    print(f"Optimization results saved to: {optimization_dir}")
    print(f"  - Study database: {storage_label}")
    print(f"  - Best params: {os.path.join(optimization_dir, 'best_params.json')}")
    print(f"  - Study summary: {os.path.join(optimization_dir, 'study_summary.json')}")
    print(f"\nTo use these parameters in training, run:")
    print(f"  python -m abx_amr_simulator.training.train \\")
    print(f"    --umbrella-config {args.umbrella_config} \\")
    print(f"    --load-best-params-by-experiment-name {run_name}")


if __name__ == '__main__':
    main()
