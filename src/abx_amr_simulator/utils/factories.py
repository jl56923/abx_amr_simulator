"""
Factory functions for instantiating core components from configuration.

Creates reward calculators, patient generators, environments, agents, and callbacks
from configuration dictionaries.
"""

import os
import json
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pdb

import yaml
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from abx_amr_simulator.core import (
    ABXAMREnv,
    RewardCalculator,
    PatientGenerator,
    PatientGeneratorMixer,
)
from abx_amr_simulator.core.reward_calculator import BalancedReward


def create_reward_calculator(config: Dict[str, Any]) -> RewardCalculator:
    """Instantiate RewardCalculator from experiment configuration.
    
    Extracts 'reward_calculator' section from config, injects seed from 'training.seed',
    and passes config dict directly to RewardCalculator constructor. This factory ensures
    seed synchronization across environment components for reproducibility.
    
    Args:
        config (Dict[str, Any]): Full experiment config dictionary. Must contain:
            - 'reward_calculator': Dict with RewardCalculator parameters (see
              RewardCalculator.default_config() for required keys)
            - 'training.seed' (optional): Random seed to inject into reward_calculator
    
    Returns:
        RewardCalculator: Initialized reward calculator instance with synchronized seed.
    
    Example:
        >>> config = load_config('base_experiment.yaml')
        >>> rc = create_reward_calculator(config)
        >>> # rc.seed now matches config['training']['seed']
    """
    reward_config = config.get('reward_calculator', {}).copy()
    # Get seed from training config if available
    training_config = config.get('training', {})
    reward_config['seed'] = training_config.get('seed', None)
    
    return RewardCalculator(config=reward_config)


def create_patient_generator(config: Dict[str, Any]) -> PatientGenerator:
    """
    Instantiate PatientGenerator or PatientGeneratorMixer from config.
    
    Supports two configuration styles:
    1. Regular PatientGenerator (default):
       patient_generator:
         prob_infected_dist: {...}
         benefit_value_multiplier_dist: {...}
         ... (other distribution configs)
         visible_patient_attributes: ['prob_infected', ...]
    
    2. PatientGeneratorMixer:
       patient_generator:
         type: mixer
         generators:
           - config_file: path/to/low_risk.yaml
             proportion: 0.8
           - config_file: path/to/high_risk.yaml
             proportion: 0.2
         visible_patient_attributes: ['prob_infected', ...]
    
    Returns:
        PatientGenerator or PatientGeneratorMixer instance
        
    Raises:
        ValueError: If configuration is invalid or missing required fields
    """
    patient_gen_config = config.get('patient_generator', {})
    
    if not patient_gen_config:
        raise ValueError(
            "Missing 'patient_generator' config section. "
            "PatientGenerator must be configured with required parameters including 'visible_patient_attributes'."
        )
    
    # Validate that visibility is configured in patient_generator section (not environment)
    if 'visible_patient_attributes' not in patient_gen_config:
        raise ValueError(
            "'visible_patient_attributes' must be specified in patient_generator config, not environment config. "
            "PatientGenerator owns visibility configuration."
        )
    
    # Get seed from training config if available
    training_config = config.get('training', {})
    seed = training_config.get('seed', None)
    
    # Check if this is a mixer configuration
    if patient_gen_config.get('type') == 'mixer':
        # Load and instantiate child generators from config files
        child_generators = []
        proportions = []
        
        generators_config = patient_gen_config.get('generators', [])
        if not generators_config:
            raise ValueError(
                "Mixer patient_generator requires 'generators' list with config files and proportions"
            )
        
        for i, gen_spec in enumerate(generators_config):
            if 'config_file' not in gen_spec:
                raise ValueError(f"generators[{i}] missing 'config_file'")
            if 'proportion' not in gen_spec:
                raise ValueError(f"generators[{i}] missing 'proportion'")
            
            # Load child config file
            config_file = gen_spec['config_file']
            
            # Resolve relative paths from project root
            project_root = Path(__file__).parent.parent
            if not os.path.isabs(config_file):
                config_file = os.path.join(str(project_root), config_file)
            
            if not os.path.exists(config_file):
                raise ValueError(f"generators[{i}] config_file not found: {config_file}")
            
            # Load the child config
            with open(config_file, 'r') as f:
                child_config = yaml.safe_load(f)
            
            # Set seed for child generator
            child_config['seed'] = seed
            
            # Instantiate child generator
            child_gen = PatientGenerator(config=child_config)
            child_generators.append(child_gen)
            proportions.append(gen_spec['proportion'])
        
        # Create mixer config for PatientGeneratorMixer
        mixer_config = {
            'generators': child_generators,
            'proportions': proportions,
            'visible_patient_attributes': patient_gen_config['visible_patient_attributes'],
            'seed': seed,
        }
        
        return PatientGeneratorMixer(config=mixer_config)
    
    else:
        # Regular PatientGenerator
        patient_gen_config['seed'] = seed
        return PatientGenerator(config=patient_gen_config)


def create_environment(config: Dict[str, Any], reward_calculator: RewardCalculator, patient_generator: PatientGenerator, wrap_monitor: bool = False, monitor_dir: Optional[str] = None) -> gym.Env:
    """Create ABXAMREnv from pre-instantiated RewardCalculator and PatientGenerator.
    
    Follows the clean orchestration pattern: core components (RewardCalculator,
    PatientGenerator) are instantiated externally and injected into the environment.
    This function extracts environment-specific parameters from config and creates
    the environment instance.
    
    Args:
        config (Dict[str, Any]): Full experiment config dictionary. Must contain:
            - 'environment': Dict with ABXAMREnv parameters (antibiotics_AMR_dict,
              num_patients_per_time_step, max_time_steps, etc.)
        reward_calculator (RewardCalculator): Pre-created RewardCalculator instance
            (created via create_reward_calculator).
        patient_generator (PatientGenerator): Pre-created PatientGenerator instance
            (created via create_patient_generator).
        wrap_monitor (bool): If True, wrap environment in stable-baselines3 Monitor
            for automatic episode logging. Useful for evaluation. Default: False.
        monitor_dir (str, optional): Directory for Monitor logs (unused, kept for
            API compatibility).
    
    Returns:
        gym.Env: Initialized environment instance, optionally wrapped in Monitor.
    
    Raises:
        ValueError: If 'patient_generator' or 'visible_patient_attributes' found in
            environment config (these belong in patient_generator config, not environment).
    
    Example:
        >>> config = load_config('base_experiment.yaml')
        >>> rc = create_reward_calculator(config)
        >>> pg = create_patient_generator(config)
        >>> env = create_environment(config, reward_calculator=rc, patient_generator=pg)
    """
    # Make a deep copy of env_kwargs to avoid modifying the original config
    env_kwargs = copy.deepcopy(config.get('environment', {}))
    
    # Remove patient_generator config section if present (should not be in env config)
    if 'patient_generator' in env_kwargs:
        raise ValueError(
            "'patient_generator' should not be in environment config. "
            "PatientGenerator must be instantiated separately and passed to create_environment()."
        )
    
    # Remove visible_patient_attributes if present (belongs to PG, not env)
    if 'visible_patient_attributes' in env_kwargs:
        raise ValueError(
            "'visible_patient_attributes' should not be in environment config. "
            "This parameter belongs in patient_generator config. PatientGenerator owns visibility."
        )
    
    # Add the pre-created reward_calculator and patient_generator to env_kwargs
    env_kwargs['reward_calculator'] = reward_calculator
    env_kwargs['patient_generator'] = patient_generator
    
    # Create the environment
    env = ABXAMREnv(**env_kwargs)
    
    if wrap_monitor:
        env = Monitor(env)
        
    return env


def wrap_environment_for_hrl(env: ABXAMREnv, config: Dict[str, Any]) -> gym.Env:
    """Wrap ABXAMREnv with OptionsWrapper for hierarchical RL.
    
    Creates OptionsWrapper with option library and manager observation configuration.
    Only called when algorithm is 'HRL_PPO' or 'HRL_DQN'.
    
    Args:
        env (ABXAMREnv): Base environment to wrap.
        config (Dict[str, Any]): Full experiment config. Must contain 'hrl' section with:
            - option_library: 'default' or path to custom library
            - option_gamma: Discount factor for macro-reward aggregation
                        - front_edge_use_full_vector: If True, manager gets full boundary cohort
                            vector. If False, manager gets mean + std for each visible attribute.
    
    Returns:
        gym.Env: OptionsWrapper-wrapped environment ready for manager training.
    
    Raises:
        ValueError: If HRL config section missing or option library invalid.
    
    Example:
        >>> base_env = create_environment(config, rc, pg)
        >>> hrl_env = wrap_environment_for_hrl(base_env, config)
        >>> agent = create_agent(config, hrl_env, tb_log_path=...)
    """
    # Import HRL components from abx_amr_simulator.hrl
    from abx_amr_simulator.hrl import OptionsWrapper, OptionLibraryLoader
    
    hrl_config = config.get('hrl', {})
    if not hrl_config:
        raise ValueError("HRL algorithm selected but 'hrl' config section missing")
    
    # Get option library
    option_library_spec = hrl_config.get('option_library', 'default')
    project_root = Path(__file__).resolve().parents[5]

    if option_library_spec == 'default':
        library_path = project_root / "workspace" / "experiments" / "options" / "option_libraries" / "default_deterministic.yaml"
    else:
        library_path = Path(option_library_spec)
        if not library_path.is_absolute():
            library_path = (project_root / library_path).resolve()
        else:
            library_path = library_path.resolve()

    if not library_path.exists():
        raise ValueError(f"Option library config not found: {library_path}")

    option_library, _ = OptionLibraryLoader.load_library(
        library_config_path=str(library_path),
        env=env,
    )
    
    # Get HRL wrapper parameters
    gamma = hrl_config.get('option_gamma', 0.99)
    front_edge_use_full_vector = hrl_config.get('front_edge_use_full_vector', False)
    
    # Wrap environment
    wrapped_env = OptionsWrapper(
        env=env,
        option_library=option_library,
        gamma=gamma,
        front_edge_use_full_vector=front_edge_use_full_vector,
    )
    
    return wrapped_env


def create_agent(config: Dict[str, Any], env: gym.Env, tb_log_path: Optional[str] = None, verbose: int = 0) -> Any:
    """Instantiate RL agent (PPO, DQN, A2C, RecurrentPPO, or MBPO) from config.
    
    Extracts algorithm type and hyperparameters from config, then creates the
    appropriate agent class. For standard agents (PPO/DQN/A2C/RecurrentPPO), uses
    'MlpPolicy' (or 'MlpLstmPolicy' for RecurrentPPO). For MBPO, returns an MBPOAgent
    instance that orchestrates model-based policy optimization.
    
    Args:
        config (Dict[str, Any]): Full experiment config dictionary. Must contain:
            - 'algorithm': 'PPO' | 'DQN' | 'A2C' | 'RecurrentPPO' | 'MBPO'
            - '{algorithm_lowercase}': Dict with algorithm-specific hyperparameters
              (e.g., 'ppo': {'learning_rate': 3e-4, 'n_steps': 2048, ...})
              (e.g., 'mbpo': {...}, 'dynamics_model': {...} for MBPO)
        env (gym.Env): Training environment instance (from create_environment).
        tb_log_path (str, optional): Path for tensorboard logs. If None, no logging.
        verbose (int): Verbosity level for stable-baselines3 output. Default: 0 (silent).
    
    Returns:
        PPO | DQN | A2C | RecurrentPPO | MBPOAgent: Initialized agent ready for training.
        For standard agents, use via .learn(total_timesteps).
        For MBPO, use via .train(total_episodes).
    
    Raises:
        ValueError: If algorithm is unknown or config missing required hyperparameters.
    
    Example:
        >>> config = load_config('ppo_baseline.yaml')
        >>> env = create_environment(config, rc, pg)
        >>> agent = create_agent(config, env, tb_log_path='results/run_1/logs')
        >>> agent.learn(total_timesteps=100000)  # For standard agents
        
        >>> config = load_config('mbpo_baseline.yaml')
        >>> agent = create_agent(config, env, tb_log_path='results/run_1/logs')
        >>> agent.train(total_episodes=200)  # For MBPO
    """
    algorithm = config.get('algorithm', 'PPO')
    action_mode = config.get('action_mode', 'multidiscrete')
    
    # Common parameters
    learning_rate = config.get('learning_rate', 3.0e-4) # Overridden later if needed
    policy_kwargs = config.get('policy_kwargs', {})
    seed = config.get('training', {}).get('seed', None)  # Get seed from training config
    verbose = verbose
    
    if algorithm == 'PPO':
        ppo_config = config.get('ppo', {})
        learning_rate = ppo_config.get('learning_rate', 3.0e-4)
        
        agent = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            ent_coef=ppo_config.get('ent_coef', 0.0),
            vf_coef=ppo_config.get('vf_coef', 0.5),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            verbose=ppo_config.get('verbose', 1),
            tensorboard_log=tb_log_path,
            seed=seed,
        )
    elif algorithm == 'DQN':
        dqn_config = config.get('dqn', {})
        learning_rate = dqn_config.get('learning_rate', 1.0e-4)
        
        agent = DQN(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            buffer_size=dqn_config.get('buffer_size', 100000),
            learning_starts=dqn_config.get('learning_starts', 1000),
            batch_size=dqn_config.get('batch_size', 32),
            tau=dqn_config.get('tau', 1.0),
            gamma=dqn_config.get('gamma', 0.99),
            train_freq=dqn_config.get('train_freq', 4),
            target_update_interval=dqn_config.get('target_update_interval', 10000),
            exploration_fraction=dqn_config.get('exploration_fraction', 0.1),
            exploration_initial_eps=dqn_config.get('exploration_initial_eps', 1.0),
            exploration_final_eps=dqn_config.get('exploration_final_eps', 0.05),
            policy_kwargs=policy_kwargs,
            verbose=dqn_config.get('verbose', 1),
            tensorboard_log=tb_log_path,
            seed=seed,
        )
    elif algorithm == 'A2C':
        a2c_config = config.get('a2c', {})
        learning_rate = a2c_config.get('learning_rate', 7.0e-4)
        
        agent = A2C(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=a2c_config.get('n_steps', 5),
            gamma=a2c_config.get('gamma', 0.99),
            gae_lambda=a2c_config.get('gae_lambda', 1.0),
            ent_coef=a2c_config.get('ent_coef', 0.0),
            vf_coef=a2c_config.get('vf_coef', 0.5),
            max_grad_norm=a2c_config.get('max_grad_norm', 0.5),
            rms_prop_eps=a2c_config.get('rms_prop_eps', 1.0e-5),
            use_rms_prop=a2c_config.get('use_rms_prop', True),
            policy_kwargs=policy_kwargs,
            verbose=a2c_config.get('verbose', 1),
            tensorboard_log=tb_log_path,
            seed=seed,
        )
    elif algorithm == 'RecurrentPPO':
        recurrent_ppo_config = config.get('recurrent_ppo', {})
        lstm_kwargs_config = config.get('lstm_kwargs', {})
        learning_rate = recurrent_ppo_config.get('learning_rate', 3.0e-4)

        # Build LSTM-specific policy_kwargs
        lstm_policy_kwargs = policy_kwargs.copy()
        lstm_policy_kwargs.update({
            'lstm_hidden_size': lstm_kwargs_config.get('lstm_hidden_size', 64),
            'n_lstm_layers': lstm_kwargs_config.get('n_lstm_layers', 1),
            'enable_critic_lstm': lstm_kwargs_config.get('enable_critic_lstm', True),
        })

        agent = RecurrentPPO(
            policy='MlpLstmPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=recurrent_ppo_config.get('n_steps', 2048),
            batch_size=recurrent_ppo_config.get('batch_size', 64),
            n_epochs=recurrent_ppo_config.get('n_epochs', 10),
            gamma=recurrent_ppo_config.get('gamma', 0.99),
            gae_lambda=recurrent_ppo_config.get('gae_lambda', 0.95),
            clip_range=recurrent_ppo_config.get('clip_range', 0.2),
            ent_coef=recurrent_ppo_config.get('ent_coef', 0.0),
            vf_coef=recurrent_ppo_config.get('vf_coef', 0.5),
            max_grad_norm=recurrent_ppo_config.get('max_grad_norm', 0.5),
            policy_kwargs=lstm_policy_kwargs,
            verbose=recurrent_ppo_config.get('verbose', 1),
            tensorboard_log=tb_log_path,
            seed=seed,
        )
    elif algorithm == 'MBPO':
        from abx_amr_simulator.mbpo.mbpo_agent import MBPOAgent
        
        # Instantiate MBPOAgent with full config dict
        # MBPOAgent expects: env, config (containing 'ppo', 'mbpo', 'dynamics_model' sections)
        agent = MBPOAgent(env=env, config=config)
    elif algorithm == 'HRL_PPO':
        # Hierarchical RL with options-based wrapper
        # Env is already wrapped with OptionsWrapper before reaching here
        # Manager uses PPO to select options
        ppo_config = config.get('ppo', {})
        learning_rate = ppo_config.get('learning_rate', 3.0e-4)
        
        agent = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=ppo_config.get('n_steps', 256),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            ent_coef=ppo_config.get('ent_coef', 0.02),
            vf_coef=ppo_config.get('vf_coef', 0.5),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            verbose=ppo_config.get('verbose', 1),
            tensorboard_log=tb_log_path,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent


def setup_callbacks(config: Dict[str, Any], run_dir: str, eval_env: Optional[gym.Env] = None) -> list:
    """Set up stable-baselines3 training callbacks.
    
    Creates PatientStatsLoggingCallback (always active), DetailedEvalCallback or
    standard EvalCallback (if eval_env provided), and CheckpointCallback. Automatically
    creates subdirectories in run_dir for logs, checkpoints, and eval_logs.
    
    Args:
        config (Dict[str, Any]): Full experiment config dictionary. Uses 'training'
            section for callback parameters:
            - eval_freq: Evaluation frequency in timesteps (default: 5000)
            - num_eval_episodes: Number of episodes per evaluation (default: 10)
            - save_freq: Checkpoint save frequency in timesteps (default: 10000)
            - log_patient_trajectories: If True, use DetailedEvalCallback to
              capture full patient trajectories during evaluation (default: False)
        run_dir (str): Run output directory (from create_run_directory).
        eval_env (gym.Env, optional): Evaluation environment instance. If None, no
            evaluation callback created. Should be a separate env instance from
            training env to avoid state pollution.
    
    Returns:
        list: List of callback instances to pass to agent.learn(callback=...).
    
    Example:
        >>> callbacks = setup_callbacks(config, run_dir, eval_env=eval_env)
        >>> agent.learn(total_timesteps=100000, callback=callbacks)
    """
    from abx_amr_simulator.callbacks import PatientStatsLoggingCallback, DetailedEvalCallback
    
    callbacks = []
    
    training_config = config.get('training', {})
    # Use converted timesteps if available (from episode-based config)
    eval_freq = training_config.get('_converted_eval_freq', training_config.get('eval_freq', 5000))
    num_eval_episodes = training_config.get('num_eval_episodes', 10)
    save_freq = training_config.get('_converted_save_freq', training_config.get('save_freq', 10000))
    log_patient_trajectories = training_config.get('log_patient_trajectories', False)
    
    # Patient stats logging callback (always active during training)
    patient_stats_callback = PatientStatsLoggingCallback()
    callbacks.append(patient_stats_callback)
    
    # Evaluation callback (only if an eval_env is provided)
    if eval_env is not None:
        # Wrap eval_env in DummyVecEnv if needed
        vec_eval_env = DummyVecEnv([lambda env=eval_env: env])
        
        if log_patient_trajectories:
            # Use DetailedEvalCallback when we want full patient trajectories
            # DetailedEvalCallback will create run_dir/eval_logs/ automatically
            eval_callback = DetailedEvalCallback(
                eval_env=vec_eval_env,
                n_eval_episodes=num_eval_episodes,
                eval_freq=eval_freq,
                log_path=run_dir,
                best_model_save_path=os.path.join(run_dir, 'checkpoints'),
                deterministic=True,
                render=False,
            )
        else:
            # Use standard EvalCallback
            eval_callback = EvalCallback(
                eval_env=vec_eval_env,
                n_eval_episodes=num_eval_episodes,
                eval_freq=eval_freq,
                log_path=os.path.join(run_dir, 'eval_logs'),
                best_model_save_path=os.path.join(run_dir, 'checkpoints'),
                deterministic=True,
                render=False,
            )
        callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(run_dir, 'checkpoints'),
        name_prefix='model',
        save_replay_buffer=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback (optional)
    early_stopping_config = training_config.get('early_stopping', {})
    if early_stopping_config.get('enabled', False):
        from abx_amr_simulator.callbacks import EarlyStoppingCallback
        
        early_stopping_callback = EarlyStoppingCallback(
            patience=early_stopping_config.get('patience', 5),
            min_delta=early_stopping_config.get('min_delta', 0.0),
            metric_name=early_stopping_config.get('metric_name', 'eval/mean_reward'),
            verbose=training_config.get('verbose', 0),
        )
        callbacks.append(early_stopping_callback)
    
    return callbacks


def create_run_directory(config: Dict[str, Any], project_root: str) -> tuple:
    """Create timestamped run directory with standard subdirectories.
    
    Generates directory structure: <output_dir>/<run_name>_<timestamp>/ with
    subdirectories: logs/, checkpoints/, eval_logs/. Timestamp format: YYYYMMDD_HHMMSS.
    
    Args:
        config (Dict[str, Any]): Experiment config dictionary. Uses:
            - 'output_dir' (str): Output base directory (default: 'results')
            - 'training.run_name' (str): Experiment name prefix (default: 'experiment')
        project_root (str): Absolute path to project root directory.
    
    Returns:
        tuple: (run_dir_path, timestamp) where:
            - run_dir_path (str): Absolute path to created run directory
            - timestamp (str): Timestamp string in format YYYYMMDD_HHMMSS
    
    Example:
        >>> run_dir, ts = create_run_directory(config, project_root='/path/to/project')
        >>> # Returns: ('/path/to/project/results/my_experiment_20250115_143614', '20250115_143614')
    """
    output_dir = config.get('output_dir', 'results')
    run_name = config.get('training', {}).get('run_name', 'experiment')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(project_root, output_dir, f"{run_name}_{timestamp}")
    
    # Create directory structure
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(run_dir, 'logs')).mkdir(exist_ok=True)
    Path(os.path.join(run_dir, 'checkpoints')).mkdir(exist_ok=True)
    Path(os.path.join(run_dir, 'eval_logs')).mkdir(exist_ok=True)
    
    return run_dir, timestamp


def save_training_config(config: Dict[str, Any], run_dir: str):
    """Save full experiment configuration as YAML in run directory.
    
    Writes config dictionary to <run_dir>/config.yaml for reproducibility and
    experiment record-keeping. This enables exact replication of training runs.
    
    Args:
        config (Dict[str, Any]): Full experiment configuration dictionary (as loaded
            by load_config, including any command-line overrides).
        run_dir (str): Absolute path to run directory (from create_run_directory).
    
    Example:
        >>> save_training_config(config, run_dir)
        >>> # Creates: <run_dir>/config.yaml
    """
    config_path = os.path.join(run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_training_summary(config: Dict[str, Any], run_dir: str, total_timesteps: int, total_episodes: int):
    """Save training run summary statistics as JSON.
    
    Records run_name, algorithm, timesteps, episodes, timestamp, and action_mode
    in <run_dir>/summary.json. Enables quick experiment identification and
    comparison without loading full config or tensorboard logs.
    
    Args:
        config (Dict[str, Any]): Full experiment configuration dictionary.
        run_dir (str): Absolute path to run directory.
        total_timesteps (int): Total environment timesteps trained.
        total_episodes (int): Total episodes completed during training.
    
    Example:
        >>> save_training_summary(config, run_dir, total_timesteps=100000, total_episodes=250)
        >>> # Creates: <run_dir>/summary.json with run metadata
    """
    summary = {
        'run_name': config.get('run_name', 'experiment'),
        'algorithm': config.get('algorithm', 'PPO'),
        'total_timesteps': total_timesteps,
        'total_episodes': total_episodes,
        'timestamp': datetime.now().isoformat(),
        'action_mode': config.get('action_mode', 'multidiscrete'),
    }
    
    summary_path = os.path.join(run_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
