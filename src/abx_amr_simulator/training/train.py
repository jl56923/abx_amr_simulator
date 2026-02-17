"""
Main training script for ABXAMREnv with Stable Baselines3.

Usage:
    # Train from scratch with parameter overrides
    python -m abx_amr_simulator.training.train --umbrella-config experiments/configs/umbrella_configs/defaults/umbrella/base_experiment.yaml
    python -m abx_amr_simulator.training.train --umbrella-config base_experiment.yaml -p "reward_calculator.lambda_weight=0.8"
    
    # Train with subconfig override (replace entire subconfig with external file)
    python -m abx_amr_simulator.training.train --umbrella-config base_experiment.yaml -s "environment-subconfig=path/to/custom_environment.yaml"
    
    # Train with both parameter and subconfig overrides
    python -m abx_amr_simulator.training.train --umbrella-config base_experiment.yaml \
      -s "environment-subconfig=configs/env_ablation.yaml" \
      -p "reward_calculator.lambda_weight=0.5"
    
    # Continue training from prior results
    python -m abx_amr_simulator.training.train --train-from-prior-results results/ppo_baseline_20260108_120000 --additional-training-episodes 100
    
    # Skip training if experiment with same name already completed successfully (useful for resuming interrupted shell scripts)
    python -m abx_amr_simulator.training.train --umbrella-config base_experiment.yaml --skip-if-exists

COMPLETION REGISTRY (.training_completed.txt):
    The completion registry is a critical part of the training workflow. Located at:
        <results_dir>/<output_dir>/.training_completed.txt
    
    Purpose:
    - Tracks successfully completed training run names (one per line)
    - Enables --skip-if-exists to distinguish between:
        * Completed experiments (in registry): skipped on re-run
        * Interrupted experiments (folder exists but NOT in registry): retried on re-run
    
    How it works:
    1. Before training starts, if --skip-if-exists is set:
       - Registry is validated and cleaned (stale entries for deleted folders removed)
       - Current run_name is excluded from validation (won't be removed if not completed yet)
       - Registry is checked to see if this run_name is already completed
       - If found: training is skipped; if not found: training proceeds
    
    2. After training completes successfully:
       - Run name is automatically added to the registry
       - Confirmation message printed: "✓ Recorded successful completion in registry"
    
    Registry cleaning:
    - Automatic: Stale entries (folders deleted but registry not updated) are cleaned on each training
    - Manual: Edit .training_completed.txt directly to add/remove entries, or delete it to reset
    
    Integration with analysis tools:
    - diagnostic_analysis.py and evaluative_plots.py use a similar registry pattern
    - Both training and analysis registries work independently; they're designed to track
      different stages of the experiment workflow (training completion vs. analysis completion)

Example shell script pattern (resuming interrupted runs):
    #!/bin/bash
    for SEED in 1 2 3 4 5; do
      for LAMBDA in 0.0 0.1 0.2; do
        python -m abx_amr_simulator.training.train \\
          --umbrella-config experiments/configs/umbrella_configs/defaults/umbrella/base_experiment.yaml \\
          --results-dir /path/to/results \\
          --skip-if-exists \\
          -p "training.run_name=exp_lambda${LAMBDA}_seed${SEED}" \\
          -p "reward_calculator.lambda_weight=$LAMBDA" \\
          -p "training.seed=$SEED"
      done
    done
    
    If interrupted mid-run: simply re-run the script. Completed experiments are skipped,
    interrupted ones are retried, and new experiments are started.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import json
import glob
from typing import Dict, Any, Optional, Tuple

import pdb

from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO

from abx_amr_simulator.utils import (
    load_config,
    apply_param_overrides,
    create_reward_calculator,
    create_patient_generator,
    create_environment,
    create_agent,
    setup_callbacks,
    create_run_directory,
    save_training_config,
    save_training_summary,
    plot_metrics_trained_agent,
)
from abx_amr_simulator.utils.registry import (
    extract_experiment_prefix,
    extract_timestamp_from_run_folder,
    load_registry,
    update_registry,
    validate_and_clean_registry,
)


def load_best_params_from_optimization(
    experiment_name: str,
    optimization_base_dir: str = "optimization"
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Find and load best hyperparameters from most recent optimization run.
    
    Args:
        experiment_name: Name of the optimization experiment (without timestamp)
        optimization_base_dir: Base directory where optimization runs are stored
    
    Returns:
        Tuple of (best_params dict or None, optimization_dir path or None)
        Returns (None, None) if no optimization run found or if loading fails
    """
    # New tune.py structure: optimization/{run_name}/ (no timestamp in folder name)
    # Each run_name can resume/continue, with registry tracking timestamps
    direct_path = os.path.join(optimization_base_dir, experiment_name)
    
    if os.path.exists(direct_path) and os.path.isdir(direct_path):
        # Found direct match - use it
        most_recent_dir = direct_path
    else:
        # Legacy support: Also try matching {experiment_name}_* pattern for backwards compatibility
        # (older code may have used timestamped folder names)
        pattern = os.path.join(optimization_base_dir, f"{experiment_name}_*")
        matching_dirs = glob.glob(pattern)
        
        if not matching_dirs:
            print(f"Error: No optimization runs found matching '{experiment_name}' in {optimization_base_dir}")
            return None, None
        
        # Sort by timestamp (embedded in directory name) to get most recent
        # Directory format: <experiment_name>_<timestamp> where timestamp is YYYYMMDD_HHMMSS
        matching_dirs.sort(reverse=True)  # Most recent first
        most_recent_dir = matching_dirs[0]
    
    # Load best_params.json from optimization directory
    best_params_path = os.path.join(most_recent_dir, 'best_params.json')
    
    if not os.path.exists(best_params_path):
        print(f"Error: best_params.json not found in {most_recent_dir}")
        return None, None
    
    try:
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        print(f"\n{'='*70}")
        print("LOADED BEST HYPERPARAMETERS FROM OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Optimization run: {os.path.basename(most_recent_dir)}")
        print(f"Parameters loaded from: {best_params_path}")
        print(f"Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*70}\n")
        return best_params, most_recent_dir
    except Exception as e:
        print(f"Error loading best_params.json: {e}")
        return None, None


def validate_training_config(config: Dict[str, Any], loaded_best_params_from: Optional[str] = None) -> None:
    """Validate training configuration for common issues and mismatches.
    
    Performs sanity checks on the training configuration:
    - Verifies required config sections exist
    - Checks algorithm compatibility with action mode
    - Validates HRL-specific requirements (option library path)
    - Warns about potential environment/agent mismatches
    - Validates best params if loaded from optimization
    
    Args:
        config: Full training configuration dictionary
        loaded_best_params_from: Path to optimization run if best params were loaded
    
    Raises:
        SystemExit: If critical validation errors are found
    """
    errors = []
    warnings = []
    
    # Check required sections
    required_sections = ['environment', 'training']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required config section: '{section}'")
    
    # Check algorithm
    algorithm = config.get('algorithm', 'PPO')
    supported_algorithms = ['PPO', 'RecurrentPPO', 'A2C', 'HRL_PPO', 'HRL_RPPO']
    if algorithm not in supported_algorithms:
        errors.append(f"Unsupported algorithm: '{algorithm}'. Supported: {supported_algorithms}")
    
    # HRL-specific validation
    if algorithm in ['HRL_PPO', 'HRL_RPPO']:
        # Check that HRL config section exists
        hrl_config = config.get('hrl', {})
        
        # Check that option library path exists
        option_library_path = hrl_config.get('option_library')
        if not option_library_path:
            errors.append(f"HRL algorithm '{algorithm}' requires 'hrl.option_library' in config")
        elif not os.path.exists(option_library_path):
            errors.append(f"Option library path does not exist: {option_library_path}")
        
        # Check that option_gamma is specified if using HRL
        option_gamma = hrl_config.get('option_gamma')
        if option_gamma is None:
            warnings.append("HRL algorithm should specify 'hrl.option_gamma' in config (discount factor for OptionsWrapper)")
    
    # Validate best params if loaded
    if loaded_best_params_from:
        print(f"\n{'='*70}")
        print("VALIDATING LOADED BEST PARAMETERS")
        print(f"{'='*70}")
        
        # Check that best params are appropriate for algorithm
        # PPO/HRL_PPO expect: learning_rate, n_steps, gamma, etc.
        # DQN expects: learning_rate, buffer_size, learning_starts, etc.
        agent_config = config.get('agent_algorithm', {})
        
        if algorithm in ['PPO', 'HRL_PPO', 'A2C', 'HRL_RPPO']:
            # Check for PPO-specific hyperparams
            ppo_params = ['learning_rate', 'n_steps', 'gamma']
            missing_params = [p for p in ppo_params if p not in agent_config]
            if missing_params:
                warnings.append(f"Expected PPO hyperparameters not found in agent_algorithm config: {missing_params}")
        
        print(f"Loaded best parameters from: {loaded_best_params_from}")
        print(f"Target algorithm: {algorithm}")
        print(f"✓ Validation complete")
        print(f"{'='*70}\n")
    
    # Print warnings
    if warnings:
        print(f"\n{'='*70}")
        print("CONFIG VALIDATION WARNINGS")
        print(f"{'='*70}")
        for warning in warnings:
            print(f"⚠️  {warning}")
        print(f"{'='*70}\n")
    
    # Print errors and exit if critical issues found
    if errors:
        print(f"\n{'='*70}")
        print("CONFIG VALIDATION ERRORS")
        print(f"{'='*70}")
        for error in errors:
            print(f"❌ {error}")
        print(f"{'='*70}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train RL agent on ABXAMREnv')
    parser.add_argument(
        '--umbrella-config',
        type=str,
        required=False,
        help='Path to umbrella config YAML file. Example: experiments/configs/umbrella_configs/base_experiment.yaml (required for new training)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config if provided)'
    )
    parser.add_argument(
        '--train-from-prior-results',
        type=str,
        default=None,
        help='Path to prior results folder to continue training from'
    )
    parser.add_argument(
        '--additional-training-episodes',
        type=int,
        default=None,
        help='Number of additional episodes to train (only used with --train-from-prior-results)'
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
        help='Replace entire subconfig with explicit path. Example: -s environment-subconfig=path/to/custom_environment.yaml'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory where results folder should be created. If not specified, uses current working directory. Example: --results-dir /path/to/workspace/results'
    )
    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help='Skip this training run if an experiment with the same run name (ignoring timestamp) already exists in the results directory. Useful for resuming interrupted shell scripts without overwriting completed experiments.'
    )
    parser.add_argument(
        '--load-best-params-by-experiment-name',
        type=str,
        default=None,
        help='Load best hyperparameters from most recent optimization run with this experiment name. Looks in optimization/ directory for <name>_<timestamp>/ folders and loads best_params.json from the most recent one. Example: --load-best-params-by-experiment-name my_tuning_run'
    )
    parser.add_argument(
        '--optimization-dir',
        type=str,
        default='optimization',
        help='Base directory where optimization runs are stored (default: optimization)'
    )
    parser.add_argument(
        '--skip-registry-validation',
        action='store_true',
        help='Skip registry validation (do not check if experiment folders still exist). Useful when moving completed experiments to different storage. Still checks if run completed, just does not validate folder existence.'
    )
    args = parser.parse_args()
    
    # Use current working directory as the base for configs (define early for subconfig path resolution)
    cwd = os.getcwd()
    
    # Enforce absolute paths for umbrella config (only if not continuing from prior results)
    if args.umbrella_config:
        if not os.path.isabs(args.umbrella_config):
            print("Error: --umbrella-config must be an absolute path. Example: /abs/path/configs/umbrella_configs/base_experiment.yaml")
            sys.exit(1)
        if not os.path.exists(args.umbrella_config):
            print(f"Error: Umbrella config file not found: {args.umbrella_config}")
            sys.exit(1)
    
    # Parse parameter overrides from -p flags
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
    
    # Parse subconfig overrides from -s flags (format: <component>-subconfig=/abs/path/to/file.yaml)
    subconfig_overrides = {}
    for override_str in args.subconfig_override:
        if '=' not in override_str or not override_str.endswith('.yaml'):
            print(f"Warning: Ignoring invalid subconfig override format '{override_str}' (expected <component>-subconfig=/abs/path/to/file.yaml)")
            continue
        key_part, path = override_str.split('=', 1)
        
        # Validate that key_part ends with '-subconfig'
        if not key_part.endswith('-subconfig'):
            print(f"Warning: Ignoring subconfig override with invalid key '{key_part}' (expected <component>-subconfig)")
            continue
        
        # Enforce absolute paths for subconfigs; remove implicit resolution
        if not os.path.isabs(path):
            print(f"Error: Subconfig path for '{key_part}' must be absolute: {path}\nExample: -s environment-subconfig=/abs/path/configs/environment/custom.yaml")
            sys.exit(1)
        if not os.path.exists(path):
            print(f"Error: Subconfig file not found: {path}")
            sys.exit(1)
        
        subconfig_overrides[key_part] = path
    
    # Determine results directory: use --results-dir if provided, otherwise current working directory
    if args.results_dir:
        results_base_dir = args.results_dir
        if not os.path.isabs(results_base_dir):
            results_base_dir = os.path.abspath(results_base_dir)
    else:
        results_base_dir = os.getcwd()
    
    # Package directory (for locating default configs if needed)
    package_root = Path(__file__).parent.parent
    
    # Determine if continuing from prior results or starting fresh
    if args.train_from_prior_results:
        # Continue training from prior results
        print(f"\n{'='*70}")
        print("CONTINUING TRAINING FROM PRIOR RESULTS")
        print(f"{'='*70}\n")
        
        prior_results_path = args.train_from_prior_results
        if not os.path.isabs(prior_results_path):
            prior_results_path = os.path.join(cwd, prior_results_path)
        
        if not os.path.exists(prior_results_path):
            print(f"Error: Prior results folder not found at {prior_results_path}")
            sys.exit(1)
        
        # Load config from prior results
        prior_config_path = os.path.join(prior_results_path, 'full_agent_env_config.yaml')
        if not os.path.exists(prior_config_path):
            print(f"Error: full_agent_env_config.yaml not found in {prior_results_path}")
            sys.exit(1)
        
        config = load_config(prior_config_path)
        print(f"Loaded config from prior results: {prior_config_path}")
        
        # Apply overrides, but warn user that only --additional-training-episodes should be changed
        # when continuing training (other parameters should match the original training run)
        if param_overrides:
            # Filter to only allow additional_training_episodes override
            allowed_keys = {'training.total_num_training_episodes', 'seed'}
            disallowed_overrides = {k: v for k, v in param_overrides.items() if k not in allowed_keys}
            if disallowed_overrides:
                print(f"\n⚠️  WARNING: When continuing training, only 'training.total_num_training_episodes' and 'seed' can be overridden.")
                print(f"   Ignoring parameter overrides: {list(disallowed_overrides.keys())}")
                print(f"   To change environment, reward, or patient parameters, start fresh training instead.\n")
                param_overrides = {k: v for k, v in param_overrides.items() if k in allowed_keys}
            if param_overrides:
                print(f"Applying allowed parameter overrides: {param_overrides}")
                config = apply_param_overrides(config, param_overrides)
        
        # Warn about subconfig overrides (not allowed during continue training)
        if subconfig_overrides:
            print(f"\n⚠️  WARNING: Subconfig overrides are not allowed when continuing training.")
            print(f"   Ignoring subconfig overrides: {list(subconfig_overrides.keys())}")
            print(f"   To change environment, reward, or patient configurations, start fresh training instead.\n")
        
        # Convert additional episodes to timesteps for SB3
        if not args.additional_training_episodes:
            print("Error: --additional-training-episodes is required when continuing training")
            sys.exit(1)
        
        max_time_steps = config['environment'].get('max_time_steps', 1000)
        additional_episodes = args.additional_training_episodes
        additional_steps = additional_episodes * max_time_steps
        print(f"Training for {additional_episodes} additional episodes ({additional_steps} timesteps, episode length: {max_time_steps})")
        
        # Override seed if provided
        if args.seed is not None:
            config['training']['seed'] = args.seed
        
        seed = config['training'].get('seed', 42)
        algorithm = config.get('algorithm', 'PPO')
        
        print(f"Algorithm: {algorithm}")
        print(f"Action mode: {config.get('action_mode', 'multidiscrete')}")
        print(f"Seed: {seed}")
        print(f"Additional training timesteps: {additional_steps}")
        
        # Load the best model from prior results (or final model if best not available)
        best_model_path = os.path.join(prior_results_path, 'checkpoints', 'best_model.zip')
        final_model_path = os.path.join(prior_results_path, 'checkpoints', 'final_model.zip')
        
        if os.path.exists(best_model_path):
            model_path = best_model_path
            print(f"Loading best model from: {model_path}")
        elif os.path.exists(final_model_path):
            model_path = final_model_path
            print(f"Note: best_model.zip not found, using final_model.zip instead")
            print(f"Loading final model from: {model_path}")
        else:
            print(f"Error: Neither best_model.zip nor final_model.zip found in {os.path.join(prior_results_path, 'checkpoints')}")
            sys.exit(1)
        
        # Map algorithm names to classes
        algorithm_map = {
            'PPO': PPO,
            'A2C': A2C,
            'HRL_PPO': PPO,
            'HRL_RPPO': RecurrentPPO,
        }
        if algorithm not in algorithm_map:
            print(f"Error: Unsupported algorithm '{algorithm}'")
            sys.exit(1)
        
        AgentClass = algorithm_map[algorithm]
        
        # Create new run directory for continued training
        original_run_name = config.get('training', {}).get('run_name', 'experiment')
        if 'training' not in config:
            config['training'] = {}
        config['training']['run_name'] = f"{original_run_name}_continued"
        run_dir, _ = create_run_directory(config, results_base_dir)
        print(f"Output directory for continued training: {run_dir}")
        
        # Create reward calculator
        print("Creating reward calculator...")
        reward_calculator = create_reward_calculator(config)
        
        # Create patient generator
        print("Creating patient generator...")
        patient_generator = create_patient_generator(config)
        
        # Create environment
        print("Creating environment...")
        env = create_environment(config=config, reward_calculator=reward_calculator, patient_generator=patient_generator)
        env.reset(seed=seed)
        
        # Wrap with OptionsWrapper if using HRL
        if algorithm in ['HRL_PPO', 'HRL_RPPO']:
            from abx_amr_simulator.utils import wrap_environment_for_hrl, save_option_library_config
            print("Wrapping environment with OptionsWrapper for HRL...")
            env = wrap_environment_for_hrl(env, config)
            # Save resolved option library config for reproducibility
            if hasattr(env, 'resolved_option_library_config'):
                save_option_library_config(env.resolved_option_library_config, run_dir)
        
        # Now load the agent with the environment
        agent = AgentClass.load(path=model_path, env=env)

        # Persist action mapping for analysis (abx -> index and index -> abx)
        try:
            config['abx_name_to_index'] = reward_calculator.abx_name_to_index
            config['index_to_abx_name'] = reward_calculator.index_to_abx_name
        except Exception:
            pass

        # Save updated config to new run directory (after mapping attached)
        config['training']['total_num_training_episodes'] = additional_episodes
        config['training']['continued_from'] = prior_results_path
        save_training_config(config, run_dir)
        
        print("Creating evaluation environment...")
        eval_env = create_environment(
            config=config,
            reward_calculator=reward_calculator,
            patient_generator=patient_generator,
            wrap_monitor=True,
        )
        eval_env.reset(seed=seed + 1)
                # Wrap eval env with OptionsWrapper if using HRL
        if algorithm in ['HRL_PPO', 'HRL_RPPO']:
            from abx_amr_simulator.utils import wrap_environment_for_hrl
            eval_env = wrap_environment_for_hrl(eval_env, config)
                # Set up callbacks for continued training
        callbacks = setup_callbacks(config, run_dir, eval_env=eval_env)
        
        print(f"\nContinuing training for {additional_steps} additional timesteps...")
        print(f"TensorBoard logs: {os.path.join(run_dir, 'logs')}")
        
        # Continue training
        training_config = config.get('training', {})
        log_interval = training_config.get('log_interval', 1)
        
        agent.learn(
            total_timesteps=additional_steps,
            log_interval=log_interval,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False,  # Continue from previous timestep count
        )
        
        # Save final model
        final_model_path = os.path.join(run_dir, 'checkpoints', 'final_model')
        agent.save(final_model_path)
        print(f"\nFinal model (end of continued training) saved to: {final_model_path}.zip")
        print(f"Best model (highest eval reward) saved to: {os.path.join(run_dir, 'checkpoints', 'best_model')}.zip")
        
        # Print evaluation results for Optuna tuning (tune.py parses this output)
        if eval_env is not None and len(callbacks) > 1:
            # Find EvalCallback in callbacks list
            eval_callback = None
            for cb in callbacks:
                if hasattr(cb, 'best_mean_reward'):
                    eval_callback = cb
                    break
            
            if eval_callback is not None and hasattr(eval_callback, 'last_mean_reward'):
                # Print in format that tune.py can parse
                print(f"\n{'='*70}")
                print(f"EVALUATION RESULTS (for hyperparameter tuning)")
                print(f"{'='*70}")
                print(f"Final mean reward: {eval_callback.last_mean_reward:.4f}")
                print(f"Best mean reward: {eval_callback.best_mean_reward:.4f}")
                print(f"{'='*70}\n")
        
        # Save summary
        save_training_summary(config, run_dir, additional_steps, 0)
        print(f"Continued training complete. Results saved to: {run_dir}")
        
        # Plot metrics for final agent (agent in memory after training)
        print("\n" + "="*70)
        print("Generating diagnostics for FINAL agent (end of training)")
        print("="*70)
        final_figs_dir = os.path.join(run_dir, 'figures_final_agent')
        os.makedirs(final_figs_dir, exist_ok=True)
        plot_metrics_trained_agent(model=agent, env=eval_env, experiment_folder=final_figs_dir, deterministic=True, figures_folder_name=None)
        
        # Plot metrics for best agent (highest eval reward during training)
        best_model_checkpoint = os.path.join(run_dir, 'checkpoints', 'best_model.zip')
        if os.path.exists(best_model_checkpoint):
            print("\n" + "="*70)
            print("Generating diagnostics for BEST agent (highest eval reward)")
            print("="*70)
            best_agent = AgentClass.load(best_model_checkpoint, env=eval_env)
            best_figs_dir = os.path.join(run_dir, 'figures_best_agent')
            os.makedirs(best_figs_dir, exist_ok=True)
            plot_metrics_trained_agent(model=best_agent, env=eval_env, experiment_folder=best_figs_dir, deterministic=True, figures_folder_name=None)
        else:
            print("\nWarning: best_model.zip not found, skipping best agent diagnostics")

        # Close environments
        env.close()
        eval_env.close()
        
        # Record successful training completion in registry
        output_dir = config.get('output_dir', 'results')
        results_folder = os.path.join(prior_results_path, '..')
        results_folder = os.path.normpath(os.path.join(results_folder, output_dir))
        completion_registry_path = os.path.join(results_folder, '.training_completed.txt')
        run_name = config.get('training', {}).get('run_name', 'experiment')
        run_timestamp = extract_timestamp_from_run_folder(os.path.basename(prior_results_path))
        if not run_timestamp:
            raise ValueError("Expected timestamp suffix in prior results folder name.")
        update_registry(completion_registry_path, run_name, run_timestamp)
        print(f"✓ Recorded successful completion in registry: {completion_registry_path} (timestamp {run_timestamp})")
        
    else:
        # Start fresh training
        if not args.umbrella_config:
            print("Error: --umbrella-config is required when starting fresh training (not continuing from prior results)")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print("STARTING FRESH TRAINING")
        print(f"{'='*70}\n")
        
        # Load configuration
        config_path = args.umbrella_config
        if not os.path.isabs(config_path):
            # Try relative to CWD first
            config_path_cwd = os.path.join(cwd, args.umbrella_config)
            if os.path.exists(config_path_cwd):
                config_path = config_path_cwd
            else:
                # Fall back to package default configs
                config_path = os.path.join(str(package_root), args.umbrella_config)
        
        if not os.path.exists(config_path):
            print(f"Error: Umbrella config file not found at {config_path}")
            sys.exit(1)
        
        config = load_config(config_path)
        print(f"Loading config from: {config_path}")
        
        # Load best hyperparameters from optimization if requested
        if args.load_best_params_by_experiment_name:
            best_params, optimization_dir = load_best_params_from_optimization(
                experiment_name=args.load_best_params_by_experiment_name,
                optimization_base_dir=args.optimization_dir
            )
            
            if best_params is None:
                print("Error: Failed to load best parameters from optimization")
                sys.exit(1)
            
            # Convert best_params to param_overrides format (dotted notation)
            # Best params from Optuna are typically flat keys like 'learning_rate', 'gamma', etc.
            # We need to prefix them with the appropriate config section (e.g., 'agent_algorithm.')
            print("\nApplying best hyperparameters as config overrides...")
            for key, value in best_params.items():
                # Best params are agent hyperparameters, so prefix with 'agent_algorithm.'
                dotted_key = f'agent_algorithm.{key}'
                param_overrides[dotted_key] = value
                print(f"  {dotted_key} = {value}")
            
            # Store reference to optimization run in config for provenance
            if 'training' not in config:
                config['training'] = {}
            config['training']['loaded_best_params_from'] = optimization_dir
        
        # First, apply subconfig overrides with explicit paths (no path inference)
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
                    config['environment'] = subconfig
                elif config_type == 'reward_calculator':
                    config['reward_calculator'] = subconfig
                elif config_type == 'patient_generator':
                    config['patient_generator'] = subconfig
                elif config_type == 'agent_algorithm':
                    config.update(subconfig)
                else:
                    print(f"Error: Unknown subconfig type '{config_type}' (expected environment, reward_calculator, patient_generator, or agent_algorithm)")
                    sys.exit(1)
        
        # Apply parameter overrides
        if param_overrides:
            print(f"\nApplying parameter overrides:")
            for key, value in param_overrides.items():
                print(f"  {key} = {value}")
            config = apply_param_overrides(config=config, overrides=param_overrides)
        
        # Resolve HRL option library path (if applicable)
        algorithm = config.get('algorithm', 'PPO')
        if algorithm in ['HRL_PPO', 'HRL_RPPO']:
            hrl_config = config.get('hrl', {})
            option_library_relative = hrl_config.get('option_library')
            if option_library_relative and 'option_library_path' not in config:
                # Resolve relative path to absolute path
                options_folder = config.get('options_folder_location', '../../options')
                config_dir = Path(args.umbrella_config).parent
                resolved_path = (config_dir / options_folder / option_library_relative).resolve()
                config['option_library_path'] = str(resolved_path)
        
        # Validate configuration
        validate_training_config(
            config=config,
            loaded_best_params_from=config.get('training', {}).get('loaded_best_params_from')
        )
        
        # Override seed if provided via --seed flag
        if args.seed is not None:
            config['training']['seed'] = args.seed
        
        seed = config['training'].get('seed', 42)
                    
        print(f"Algorithm: {config.get('algorithm', 'PPO')}")
        print(f"Action mode: {config.get('action_mode', 'multidiscrete')}")
        print(f"Seed: {seed}")
        
        # Convert episode-based training config to timesteps for SB3
        training_config = config.get('training', {})
        max_time_steps = config['environment'].get('max_time_steps')
        
        if max_time_steps is None:
            print("Error: environment.max_time_steps must be defined for episode-based training config")
            sys.exit(1)
        
        # Convert episodes to timesteps
        total_num_training_episodes = training_config.get('total_num_training_episodes')
        if total_num_training_episodes is not None:
            total_timesteps = total_num_training_episodes * max_time_steps
            print(f"Training for {total_num_training_episodes} episodes ({total_timesteps} timesteps, episode length: {max_time_steps})")
        else:
            # Fallback for legacy configs
            total_timesteps = training_config.get('total_timesteps', 1000)
            print(f"Warning: Using legacy timestep-based config. Total timesteps: {total_timesteps}")
        
        save_freq_episodes = training_config.get('save_freq_every_n_episodes')
        if save_freq_episodes is not None:
            save_freq = save_freq_episodes * max_time_steps
        else:
            save_freq = training_config.get('save_freq', 10000)
        
        eval_freq_episodes = training_config.get('eval_freq_every_n_episodes')
        if eval_freq_episodes is not None:
            eval_freq = eval_freq_episodes * max_time_steps
        else:
            eval_freq = training_config.get('eval_freq', 5000)
        
        # Update config with converted timesteps for callbacks
        config['training']['_converted_total_timesteps'] = total_timesteps
        config['training']['_converted_save_freq'] = save_freq
        config['training']['_converted_eval_freq'] = eval_freq
        
        # Prepare registry paths
        output_dir = config.get('output_dir', 'results')
        results_folder = os.path.join(results_base_dir, output_dir)
        completion_registry_path = os.path.join(results_folder, '.training_completed.txt')
        run_name = config.get('training', {}).get('run_name', 'experiment')
        
        # Check if experiment already exists (if --skip-if-exists flag is set)
        if args.skip_if_exists:
            # Validate and clean completion registry ONLY when checking skip-if-exists
            # Don't clean the current run_name - it might legitimately not have a folder yet
            if not args.skip_registry_validation:
                stale_entries = validate_and_clean_registry(
                    completion_registry_path,
                    results_folder,
                    exclude_prefix=run_name  # Don't remove current run from registry
                )
                if stale_entries:
                    print(f"\n⚠️  Registry cleanup: Removed {len(stale_entries)} stale entry/entries for deleted experiments:")
                    for entry in stale_entries:
                        print(f"    - {entry[0]} (timestamp {entry[1]})")
                    print()
            else:
                stale_entries = []
                print(f"\n⚠️  Registry validation skipped (--skip-registry-validation flag set)")
                print(f"Registry will NOT check if experiment folders exist.\n")
            
            completed_prefixes = load_registry(completion_registry_path)

            # Check if this run_name has been completed (any timestamp)
            is_completed = run_name in completed_prefixes
            
            if is_completed:
                print(f"\n{'='*70}")
                print("SKIPPING: Experiment successfully completed")
                print(f"{'='*70}")
                print(f"Run name: {run_name}")
                print(f"Found in completion registry: {completion_registry_path}")
                print(f"\nSkipping this training run (--skip-if-exists flag is set).")
                print(f"To force re-run, either:")
                print(f"  1. Remove the --skip-if-exists flag, or")
                print(f"  2. Delete the experiment folder(s), or")
                print(f"  3. Edit {completion_registry_path} to remove this entry\n")
                sys.exit(0)  # Exit successfully (not an error)
        
        # Create run directory
        run_dir, run_timestamp = create_run_directory(config, results_base_dir)
        print(f"Output directory: {run_dir}")
        
        # Create reward model
        print("Creating reward calculator...")
        reward_calculator = create_reward_calculator(config)
        
        # Create patient generator
        print("Creating patient generator...")
        patient_generator = create_patient_generator(config)
        
        # Create environment
        print("Creating environment...")
        env = create_environment(config=config, reward_calculator=reward_calculator, patient_generator=patient_generator)
        env.reset(seed=seed)
        
        # Wrap with OptionsWrapper if using HRL
        algorithm = config.get('algorithm', 'PPO')
        if algorithm in ['HRL_PPO', 'HRL_RPPO']:
            from abx_amr_simulator.utils import wrap_environment_for_hrl, save_option_library_config
            print("Wrapping environment with OptionsWrapper for HRL...")
            env = wrap_environment_for_hrl(env, config)
            print(f"Manager observation space: {env.observation_space}")
            print(f"Manager action space (option selection): {env.action_space}")
            # Save resolved option library config for reproducibility
            if hasattr(env, 'resolved_option_library_config'):
                save_option_library_config(env.resolved_option_library_config, run_dir)

        # Persist action mapping for analysis (abx -> index and index -> abx)
        try:
            config['abx_name_to_index'] = reward_calculator.abx_name_to_index
            config['index_to_abx_name'] = reward_calculator.index_to_abx_name
        except Exception:
            pass
        # Save config to run directory after mappings are attached
        save_training_config(config, run_dir)

        print("Creating evaluation environment")
        eval_env = create_environment(
            config=config,
            reward_calculator=reward_calculator,
            patient_generator=patient_generator,
            wrap_monitor=True,
        )
        eval_env.reset(seed=seed + 1) # Different seed for eval env
        
        # Wrap eval env with OptionsWrapper if using HRL
        if algorithm in ['HRL_PPO', 'HRL_RPPO']:
            from abx_amr_simulator.utils import wrap_environment_for_hrl
            eval_env = wrap_environment_for_hrl(eval_env, config)
        
        # Create agent
        print("Creating agent...")
        agent = create_agent(config, env, tb_log_path=os.path.join(run_dir, 'logs'))
        
        # Set up callbacks
        callbacks = setup_callbacks(config, run_dir, eval_env=eval_env)
        
        # Train agent
        training_config = config.get('training', {})
        total_timesteps = training_config.get('_converted_total_timesteps', 1000)
        log_interval = training_config.get('log_interval', 1)
        
        print(f"\nStarting training for {total_timesteps} timesteps...")
        print(f"TensorBoard logs: {os.path.join(run_dir, 'logs')}")
        
        agent.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callbacks,
            progress_bar=True,
        )
        
        # Save final model to checkpoints folder (separate from best_model saved by EvalCallback)
        # best_model: highest eval reward (saved by EvalCallback if eval_env provided)
        # final_model: model at end of training (useful for debugging overfitting)
        final_model_path = os.path.join(run_dir, 'checkpoints', 'final_model')
        agent.save(final_model_path)
        print(f"\nFinal model (end of training) saved to: {final_model_path}.zip")
        print(f"Best model (highest eval reward) saved to: {os.path.join(run_dir, 'checkpoints', 'best_model')}.zip")
        
        # Print evaluation results for Optuna tuning (tune.py parses this output)
        if eval_env is not None and len(callbacks) > 1:
            # Find EvalCallback in callbacks list (should be second callback after PatientStatsLoggingCallback)
            eval_callback = None
            for cb in callbacks:
                if hasattr(cb, 'best_mean_reward'):
                    eval_callback = cb
                    break
            
            if eval_callback is not None and hasattr(eval_callback, 'last_mean_reward'):
                # Print in format that tune.py can parse
                print(f"\n{'='*70}")
                print(f"EVALUATION RESULTS (for hyperparameter tuning)")
                print(f"{'='*70}")
                print(f"Final mean reward: {eval_callback.last_mean_reward:.4f}")
                print(f"Best mean reward: {eval_callback.best_mean_reward:.4f}")
                print(f"{'='*70}\n")
        
        # Save summary
        save_training_summary(config, run_dir, total_timesteps, 0)
        print(f"Training complete. Results saved to: {run_dir}")
        
        # Map algorithm names to classes for loading best model
        algorithm = config.get('algorithm', 'PPO')
        algorithm_map = {
            'PPO': PPO,
            'A2C': A2C,
            'HRL_PPO': PPO,
            'HRL_RPPO': RecurrentPPO,
        }
        AgentClass = algorithm_map.get(algorithm, PPO)
        
        # Plot metrics for final agent (agent in memory after training)
        print("\n" + "="*70)
        print("Generating diagnostics for FINAL agent (end of training)")
        print("="*70)
        final_figs_dir = os.path.join(run_dir, 'figures_final_agent')
        os.makedirs(final_figs_dir, exist_ok=True)
        plot_metrics_trained_agent(model=agent, env=eval_env, experiment_folder=final_figs_dir, deterministic=True, figures_folder_name=None)
        
        # Plot metrics for best agent (highest eval reward during training)
        best_model_checkpoint = os.path.join(run_dir, 'checkpoints', 'best_model.zip')
        if os.path.exists(best_model_checkpoint):
            print("\n" + "="*70)
            print("Generating diagnostics for BEST agent (highest eval reward)")
            print("="*70)
            best_agent = AgentClass.load(best_model_checkpoint, env=eval_env)
            best_figs_dir = os.path.join(run_dir, 'figures_best_agent')
            os.makedirs(best_figs_dir, exist_ok=True)
            plot_metrics_trained_agent(model=best_agent, env=eval_env, experiment_folder=best_figs_dir, deterministic=True, figures_folder_name=None)
        else:
            print("\nWarning: best_model.zip not found, skipping best agent diagnostics")
        
        # Close environment
        env.close()
        eval_env.close()
        
        # Record successful training completion in registry
        output_dir = config.get('output_dir', 'results')
        results_folder = os.path.join(results_base_dir, output_dir)
        completion_registry_path = os.path.join(results_folder, '.training_completed.txt')
        run_name = config.get('training', {}).get('run_name', 'experiment')
        update_registry(completion_registry_path, run_name, run_timestamp)
        print(f"\n✓ Recorded successful completion in registry: {completion_registry_path} (timestamp {run_timestamp})")

if __name__ == '__main__':
    main()
