"""
Training script for MBPO (Model-Based Policy Optimization) on ABXAMREnv.

MBPO combines model-based learning (DynamicsModel) with model-free policy improvement (PPO).
This script manages the episode-based training loop, which differs fundamentally from
the step-based loop in train.py.

Usage:
    # Train MBPO from scratch
    python -m abx_amr_simulator.training.train_mbpo --umbrella-config base_experiment.yaml
    python -m abx_amr_simulator.training.train_mbpo --umbrella-config base_experiment.yaml -p "mbpo.rollout_length=10"
    
    # Train with subconfig override
    python -m abx_amr_simulator.training.train_mbpo --umbrella-config base_experiment.yaml \
      -s "agent_algorithm-subconfig=/abs/path/to/mbpo.yaml"
    
    # Override seed for reproducibility
    python -m abx_amr_simulator.training.train_mbpo --umbrella-config base_experiment.yaml --seed 123

Note:
    - MBPO training is episode-based (agent.train(total_episodes))
    - For standard agents (PPO/DQN/A2C), use train.py instead
    - Configuration must specify algorithm='MBPO' in agent config
    - All config override mechanisms (-p, -s flags) work identically to train.py
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

from abx_amr_simulator.utils import (
    load_config,
    apply_param_overrides,
    create_reward_calculator,
    create_patient_generator,
    create_environment,
    create_agent,
    create_run_directory,
    save_training_config,
    save_training_summary,
    plot_metrics_trained_agent,
)


def main():
    parser = argparse.ArgumentParser(description='Train MBPO (Model-Based Policy Optimization) on ABXAMREnv')
    parser.add_argument(
        '--umbrella-config',
        type=str,
        required=True,
        help='Path to umbrella config YAML file. Example: experiments/configs/umbrella_configs/base_experiment.yaml'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config if provided)'
    )
    parser.add_argument(
        '-p',
        '--param-override',
        action='append',
        default=[],
        help='Change config values using dot notation. Example: -p mbpo.rollout_length=10'
    )
    parser.add_argument(
        '-s',
        '--subconfig-override',
        action='append',
        default=[],
        help='Replace entire subconfig with explicit path. Example: -s agent_algorithm-subconfig=/abs/path/to/mbpo.yaml'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory where results folder should be created. If not specified, uses current working directory.'
    )
    
    args = parser.parse_args()
    cwd = os.getcwd()
    
    # Parse parameter overrides from -p flags (format: dot.notation.key=value)
    param_overrides = {}
    for override_str in args.param_override:
        if '=' not in override_str:
            print(f"Warning: Ignoring invalid parameter override format '{override_str}' (expected key=value)")
            continue
        key, value = override_str.split('=', 1)
        
        # Attempt type conversion
        if value.lower() in ['none', 'null']:
            value = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
        param_overrides[key] = value
    
    # Parse subconfig overrides from -s flags
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
        
        # Enforce absolute paths for subconfigs
        if not os.path.isabs(path):
            print(f"Error: Subconfig path for '{key_part}' must be absolute: {path}\nExample: -s agent_algorithm-subconfig=/abs/path/configs/agent_algorithm/mbpo.yaml")
            sys.exit(1)
        if not os.path.exists(path):
            print(f"Error: Subconfig file not found: {path}")
            sys.exit(1)
        
        subconfig_overrides[key_part] = path
    
    # Determine results directory
    if args.results_dir:
        results_base_dir = args.results_dir
        if not os.path.isabs(results_base_dir):
            results_base_dir = os.path.abspath(results_base_dir)
    else:
        results_base_dir = os.getcwd()
    
    # Package directory for locating default configs
    package_root = Path(__file__).parent.parent
    
    print(f"\n{'='*70}")
    print("STARTING MBPO TRAINING")
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
    
    # Apply subconfig overrides
    if subconfig_overrides:
        print(f"\nApplying subconfig overrides...")
        for key, path in subconfig_overrides.items():
            # Extract config type from key (e.g., 'agent_algorithm-subconfig' -> 'agent_algorithm')
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
    
    # Override seed if provided via --seed flag
    if args.seed is not None:
        config['training']['seed'] = args.seed
    
    seed = config['training'].get('seed', 42)
    
    # Validate algorithm
    algorithm = config.get('algorithm', 'PPO')
    if algorithm != 'MBPO':
        print(f"Warning: This script is designed for MBPO. Config specifies algorithm='{algorithm}'")
        print(f"For standard agents (PPO/DQN/A2C), use train.py instead")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(0)
    
    print(f"Algorithm: {algorithm}")
    print(f"Seed: {seed}")
    
    # Get training parameters
    training_config = config.get('training', {})
    total_training_episodes = training_config.get('total_num_training_episodes', 100)
    
    print(f"Training for {total_training_episodes} episodes...")
    
    # Create run directory
    run_dir, run_timestamp = create_run_directory(config, results_base_dir)
    print(f"Output directory: {run_dir}")
    
    # Create reward calculator
    print("Creating reward calculator...")
    reward_calculator = create_reward_calculator(config)
    
    # Create patient generator
    print("Creating patient generator...")
    patient_generator = create_patient_generator(config)
    
    # Create environment
    print("Creating environment...")
    env = create_environment(
        config=config,
        reward_calculator=reward_calculator,
        patient_generator=patient_generator
    )
    env.reset(seed=seed)
    
    # Persist action mapping for analysis
    try:
        config['abx_name_to_index'] = reward_calculator.abx_name_to_index
        config['index_to_abx_name'] = reward_calculator.index_to_abx_name
    except Exception:
        pass
    
    # Save config to run directory
    save_training_config(config, run_dir)
    
    # Create agent (MBPO-specific)
    print("Creating MBPO agent...")
    agent = create_agent(config=config, env=env, tb_log_path=os.path.join(run_dir, 'logs'))
    
    # Verify agent is MBPOAgent
    from abx_amr_simulator.mbpo.mbpo_agent import MBPOAgent
    if not isinstance(agent, MBPOAgent):
        print(f"Error: Expected MBPOAgent but got {type(agent).__name__}")
        sys.exit(1)
    
    # Train agent (episode-based, unlike train.py which is step-based)
    print(f"\nStarting MBPO training for {total_training_episodes} episodes...")
    print(f"TensorBoard logs: {os.path.join(run_dir, 'logs')}")
    
    try:
        agent.train(total_episodes=total_training_episodes)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    # Save final checkpoint
    final_model_path = os.path.join(run_dir, 'checkpoints', 'final_model')
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    agent.policy.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}.zip")
    
    # Save summary
    save_training_summary(config, run_dir, total_training_episodes, 0)
    print(f"MBPO training complete. Results saved to: {run_dir}")
    
    # Close environment
    env.close()


if __name__ == '__main__':
    main()
