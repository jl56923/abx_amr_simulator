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
    2. After tuning: Add run_name + timestamp + best_value + n_trials to registry
    3. During training: train.py loads best_params.json from most recent run_name timestamp

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
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import optuna
import yaml
import numpy as np

from abx_amr_simulator.utils import (
    load_config,
    apply_param_overrides,
    apply_subconfig_overrides,
)
from abx_amr_simulator.utils.registry import (
    load_registry,
    update_registry,
    validate_and_clean_registry,
)


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
    results_dir: str
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
    base_seed: int
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
        rewards = run_training_trial(
            umbrella_config_path=umbrella_config_path,
            suggested_params=suggested_params,
            truncated_episodes=truncated_episodes,
            seeds=trial_seeds,
            subconfig_overrides=subconfig_overrides,
            base_param_overrides=base_param_overrides,
            results_dir=results_dir
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
    
    # Create timestamped optimization directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    optimization_dir = os.path.join(optimization_base_dir, f"{run_name}_{timestamp}")
    Path(optimization_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("STARTING HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Run name: {run_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Optimization directory: {optimization_dir}")
    print(f"Registry: {completion_registry_path}")
    print(f"{'='*70}\n")
    
    # Load configs
    print("Loading configurations...")
    umbrella_config = load_config(args.umbrella_config)
    
    with open(args.tuning_config, 'r') as f:
        tuning_config = yaml.safe_load(f)
    
    # Apply overrides to umbrella config
    if subconfig_overrides:
        umbrella_config = apply_subconfig_overrides(umbrella_config, subconfig_overrides)
    if param_overrides:
        umbrella_config = apply_param_overrides(umbrella_config, param_overrides)
    
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
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Create temporary directory for trial results
        results_dir = tempfile.mkdtemp(prefix='optuna_trials_')
    
    # Get base seed from umbrella config
    base_seed = umbrella_config.get('training', {}).get('seed', 42)
    
    # Create Optuna study
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=run_name
    )
    
    # Create objective function
    objective = create_objective_function(
        umbrella_config_path=args.umbrella_config,
        tuning_config=tuning_config,
        subconfig_overrides=subconfig_overrides,
        base_param_overrides=param_overrides,
        results_dir=results_dir,
        base_seed=base_seed
    )
    
    # Run optimization
    print(f"\nStarting optimization with {n_trials} trials...")
    print(f"Direction: {direction}")
    print(f"Sampler: {sampler_name}")
    print(f"Seeds per trial: {optimization_config.get('n_seeds_per_trial', 3)}")
    print(f"Truncated episodes: {optimization_config.get('truncated_episodes', 50)}\n")
    
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best trial: {study.best_trial.number}")
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
    
    # Update registry
    update_registry(
        registry_path=completion_registry_path,
        run_name=run_name,
        timestamp=timestamp
    )
    print(f"\n✓ Recorded successful completion in registry: {completion_registry_path}\n")
    
    print(f"Optimization results saved to: {optimization_dir}")
    print(f"To use these parameters in training, run:")
    print(f"  python -m abx_amr_simulator.training.train \\")
    print(f"    --umbrella-config {args.umbrella_config} \\")
    print(f"    --load-best-params-by-experiment-name {run_name}")


if __name__ == '__main__':
    main()
