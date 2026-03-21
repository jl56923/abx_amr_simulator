"""
Factory functions for instantiating core components from configuration.

Creates reward calculators, patient generators, environments, agents, and callbacks
from configuration dictionaries.
"""

import os
import json
import copy
import importlib
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

import pdb

import yaml
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import masked variants for HRL
from abx_amr_simulator.hrl.rl_algorithms import PPO_Masked, RecurrentPPO_Masked
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from abx_amr_simulator.core import (
    ABXAMREnv,
    RewardCalculator,
    PatientGenerator,
    PatientGeneratorMixer,
)
from abx_amr_simulator.core.base_patient_generator import PatientGeneratorBase
from abx_amr_simulator.core.base_reward_calculator import RewardCalculatorBase
from abx_amr_simulator.core.base_amr_dynamics import AMRDynamicsBase
from abx_amr_simulator.core.leaky_balloon import AMR_LeakyBalloon
from abx_amr_simulator.core.reward_calculator import BalancedReward
from abx_amr_simulator.utils.plugin_loader import load_plugin_component

# Import for type hints only (avoid circular imports at runtime)
if TYPE_CHECKING:
    from abx_amr_simulator.hrl import OptionsWrapper


def create_reward_calculator(config: Dict[str, Any]) -> RewardCalculator:
    """Instantiate RewardCalculator from experiment configuration.

    If `reward_calculator.plugin.loader_module` is configured, this factory calls
    `load_reward_calculator_component(config: Dict[str, Any]) -> RewardCalculatorBase`
    through the shared plugin loader utility and returns the plugin-provided
    instance directly.
    
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
    reward_config_for_plugin = config.get('reward_calculator', {})
    plugin_result = load_plugin_component(
        component_config=reward_config_for_plugin,
        expected_base_class=RewardCalculatorBase,
        default_loader_fn_name='load_reward_calculator_component',
        config_dir_hint=config.get('_umbrella_config_dir'),
    )
    if plugin_result is not None:
        return plugin_result

    reward_config = config.get('reward_calculator', {}).copy()
    # Get seed from training config if available
    training_config = config.get('training', {})
    reward_config['seed'] = training_config.get('seed', None)
    
    return RewardCalculator(config=reward_config)


def create_patient_generator(config: Dict[str, Any]) -> PatientGenerator:
    """
    Instantiate PatientGenerator or PatientGeneratorMixer from config.

    If `patient_generator.plugin.loader_module` is configured, this factory calls
    `load_patient_generator_component(config: Dict[str, Any]) -> PatientGeneratorBase`
    through the shared plugin loader utility and returns the plugin-provided
    instance directly.
    
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

    plugin_result = load_plugin_component(
        component_config=patient_gen_config,
        expected_base_class=PatientGeneratorBase,
        default_loader_fn_name='load_patient_generator_component',
        config_dir_hint=config.get('_umbrella_config_dir'),
    )
    if plugin_result is not None:
        return plugin_result
    
    # Get seed from training config if available
    training_config = config.get('training', {})
    seed = training_config.get('seed', None)
    
    # Check if this is a mixer configuration
    if patient_gen_config.get('type') == 'mixer':
        # For mixers, we'll derive visible_patient_attributes from sub-generators
        # Load and instantiate child generators from config files
        child_generators = []
        proportions = []
        all_visible_attrs = []  # Collect all visible attributes from sub-generators
        
        generators_config = patient_gen_config.get('generators', [])
        if not generators_config:
            raise ValueError(
                "Mixer patient_generator requires 'generators' list with config files and proportions"
            )
        
        # Determine base directory for child generator config resolution
        # Following the same pattern as umbrella configs use config_folder_location
        if 'patient_generator_config_folder_location' in patient_gen_config:
            # Modern format: use explicit folder location (relative to umbrella config dir)
            pg_config_folder_location = patient_gen_config['patient_generator_config_folder_location']
            
            # Get umbrella config directory from config (injected by load_config)
            umbrella_config_dir = config.get('_umbrella_config_dir')
            if umbrella_config_dir is None:
                raise ValueError(
                    "Mixer config specifies 'patient_generator_config_folder_location' but "
                    "'_umbrella_config_dir' not found in config. This should be set by load_config()."
                )
            
            umbrella_dir = Path(umbrella_config_dir)
            if not Path(pg_config_folder_location).is_absolute():
                # Resolve relative to umbrella config location
                # First resolve config_folder_location to get patient_generator config dir
                config_folder_location = config.get('config_folder_location', '../')
                if not Path(config_folder_location).is_absolute():
                    config_base_dir = (umbrella_dir / config_folder_location).resolve()
                else:
                    config_base_dir = Path(config_folder_location).resolve()
                
                # Then resolve patient_generator subfolder
                pg_base_dir = (config_base_dir / 'patient_generator').resolve()
                
                # Finally resolve the mixer's child generator folder
                child_gen_base_dir = (pg_base_dir / pg_config_folder_location).resolve()
            else:
                child_gen_base_dir = Path(pg_config_folder_location).resolve()
        else:
            # Legacy fallback: resolve from project root (package location)
            project_root = Path(__file__).parent.parent
            child_gen_base_dir = project_root
        
        for i, gen_spec in enumerate(generators_config):
            if 'config_file' not in gen_spec:
                raise ValueError(f"generators[{i}] missing 'config_file'")
            if 'proportion' not in gen_spec:
                raise ValueError(f"generators[{i}] missing 'proportion'")
            
            # Load child config file
            config_file = gen_spec['config_file']
            
            # Resolve relative paths from determined base directory
            if not os.path.isabs(config_file):
                config_file = str(child_gen_base_dir / config_file)
            
            if not os.path.exists(config_file):
                raise ValueError(f"generators[{i}] config_file not found: {config_file}")
            
            # Load the child config
            with open(config_file, 'r') as f:
                child_config = yaml.safe_load(f)
            
            # Collect visible attributes from this child config
            if 'visible_patient_attributes' not in child_config:
                raise ValueError(
                    f"generators[{i}] config file missing 'visible_patient_attributes': {config_file}"
                )
            all_visible_attrs.extend(child_config['visible_patient_attributes'])
            
            # Set seed for child generator
            child_config['seed'] = seed
            
            # Instantiate child generator
            child_gen = PatientGenerator(config=child_config)
            child_generators.append(child_gen)
            proportions.append(gen_spec['proportion'])
        
        # Create union of all visible attributes (preserves order, removes duplicates)
        visible_attrs_union = []
        seen = set()
        for attr in all_visible_attrs:
            if attr not in seen:
                visible_attrs_union.append(attr)
                seen.add(attr)
        
        # Create mixer config for PatientGeneratorMixer
        mixer_config = {
            'generators': child_generators,
            'proportions': proportions,
            'visible_patient_attributes': visible_attrs_union,
            'seed': seed,
        }
        
        return PatientGeneratorMixer(config=mixer_config)
    
    else:
        # Regular PatientGenerator - validate visibility config
        if 'visible_patient_attributes' not in patient_gen_config:
            raise ValueError(
                "'visible_patient_attributes' must be specified in patient_generator config, not environment config. "
                "PatientGenerator owns visibility configuration."
            )
        patient_gen_config['seed'] = seed
        return PatientGenerator(config=patient_gen_config)


def create_amr_dynamics(config: Dict[str, Any]) -> Dict[str, AMRDynamicsBase]:
    """Instantiate AMR dynamics mapping from experiment configuration.

    If `amr_dynamics.plugin.loader_module` is configured, this factory calls
    `load_amr_dynamics_component(config: Dict[str, Any]) -> Dict[str, AMRDynamicsBase]`
    through the shared plugin loader utility and validates that all returned
    values are `AMRDynamicsBase` instances.

    Without plugin config, this factory constructs canonical `AMR_LeakyBalloon`
    instances from `environment.antibiotics_AMR_dict` using the same defaults as
    the historical internal `ABXAMREnv` fallback construction path.

    Args:
        config (Dict[str, Any]): Full merged experiment config.

    Returns:
        Dict[str, AMRDynamicsBase]: Mapping of antibiotic name to AMR dynamics instance.

    Raises:
        ValueError: If required config sections are missing.
        TypeError: If plugin return payload has invalid type/contents.
    """
    amr_dynamics_config = config.get('amr_dynamics', {})
    plugin_result = load_plugin_component(
        component_config=amr_dynamics_config,
        expected_base_class=dict,
        default_loader_fn_name='load_amr_dynamics_component',
        config_dir_hint=config.get('_umbrella_config_dir'),
    )
    if plugin_result is not None:
        for abx_name, dynamics_instance in plugin_result.items():
            if not isinstance(abx_name, str):
                raise TypeError(
                    "AMR dynamics plugin returned invalid dict key type. "
                    f"Expected str, got '{type(abx_name).__name__}'."
                )
            if not isinstance(dynamics_instance, AMRDynamicsBase):
                raise TypeError(
                    "AMR dynamics plugin returned invalid dict value type. "
                    f"Expected AMRDynamicsBase for key '{abx_name}', got '{type(dynamics_instance).__name__}'."
                )
        return plugin_result

    environment_config = config.get('environment', {})
    antibiotics_amr_dict = environment_config.get('antibiotics_AMR_dict')
    if not isinstance(antibiotics_amr_dict, dict) or not antibiotics_amr_dict:
        raise ValueError(
            "Missing or invalid 'environment.antibiotics_AMR_dict'. "
            "Expected non-empty dictionary of AMR dynamics parameters."
        )

    default_params = {
        'leak': 0.05,
        'flatness_parameter': 1.0,
        'permanent_residual_volume': 0.0,
        'initial_amr_level': 0.0,
    }
    amr_dynamics_instances: Dict[str, AMRDynamicsBase] = {}
    for abx_name, abx_params in antibiotics_amr_dict.items():
        if not isinstance(abx_params, dict):
            raise TypeError(
                "Each entry in environment.antibiotics_AMR_dict must be a dict of parameters. "
                f"Key '{abx_name}' has type '{type(abx_params).__name__}'."
            )
        leak = abx_params.get('leak', default_params['leak'])
        flatness_parameter = abx_params.get('flatness_parameter', default_params['flatness_parameter'])
        permanent_residual_volume = abx_params.get('permanent_residual_volume', default_params['permanent_residual_volume'])
        initial_amr_level = abx_params.get('initial_amr_level', default_params['initial_amr_level'])

        amr_dynamics_instances[abx_name] = AMR_LeakyBalloon(
            leak=leak,
            flatness_parameter=flatness_parameter,
            permanent_residual_volume=permanent_residual_volume,
            initial_amr_level=initial_amr_level,
        )

    return amr_dynamics_instances


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
    
    amr_dynamics = create_amr_dynamics(config=config)

    reward_antibiotic_order = list(reward_calculator.antibiotic_names)
    environment_antibiotics_config = env_kwargs.get('antibiotics_AMR_dict')
    if not isinstance(environment_antibiotics_config, dict):
        raise ValueError(
            "Environment config must define 'antibiotics_AMR_dict' as a dictionary."
        )

    environment_antibiotic_order = list(environment_antibiotics_config.keys())
    if reward_antibiotic_order != environment_antibiotic_order:
        raise ValueError(
            "RewardCalculator antibiotic order must match environment antibiotics_AMR_dict order exactly. "
            f"reward_calculator.antibiotic_names={reward_antibiotic_order}, "
            f"environment.antibiotics_AMR_dict.keys()={environment_antibiotic_order}"
        )

    if hasattr(patient_generator, 'bind_antibiotic_order'):
        patient_generator.bind_antibiotic_order(
            antibiotic_names=reward_antibiotic_order,
        )

    # Add the pre-created reward_calculator and patient_generator to env_kwargs
    env_kwargs['reward_calculator'] = reward_calculator
    env_kwargs['patient_generator'] = patient_generator
    env_kwargs['amr_dynamics_instances'] = amr_dynamics
    
    # Create the environment
    env = ABXAMREnv(**env_kwargs)
    
    if wrap_monitor:
        env = Monitor(env)
        
    return env


def wrap_environment_for_hrl(env: ABXAMREnv, config: Dict[str, Any]) -> "OptionsWrapper":
    """Wrap ABXAMREnv with OptionsWrapper for hierarchical RL.
    
    Creates OptionsWrapper with option library and manager observation configuration.
    Only called when algorithm is 'HRL_PPO' or 'HRL_RPPO'.
    
    Args:
        env (ABXAMREnv): Base environment to wrap.
        config (Dict[str, Any]): Full experiment config. Must contain 'hrl' section with:
            - option_library: path to option library config (relative to options_folder_location)
            - option_gamma: Discount factor for macro-reward aggregation
            - front_edge_use_full_vector: If True, manager gets full boundary cohort
                vector. If False, manager gets mean + std for each visible attribute.
            And should contain 'options_folder_location' (default: '../../options' relative
            to configs/umbrella_configs).
    
    Returns:
        OptionsWrapper: Hierarchy-aware environment wrapping ABXAMREnv with option selection.
    
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
    
    # Get option library path
    option_library_path = hrl_config.get('option_library')
    if not option_library_path:
        raise ValueError("HRL config must specify 'option_library' path")
    
    # Determine base directory for option library resolution
    options_folder_location = config.get('options_folder_location', '../../options')
    
    # Get the umbrella config directory for relative path resolution
    umbrella_config_dir = config.get('_umbrella_config_dir')
    if umbrella_config_dir:
        umbrella_dir = Path(umbrella_config_dir)
    else:
        # Fallback if _umbrella_config_dir not available
        umbrella_dir = Path.cwd()
    
    # Resolve options_folder_location
    if Path(options_folder_location).is_absolute():
        options_base_dir = Path(options_folder_location).resolve()
    else:
        # Relative to umbrella config directory
        options_base_dir = (umbrella_dir / options_folder_location).resolve()
    
    # Construct full library path
    library_path = (options_base_dir / option_library_path).resolve()
    
    if not library_path.exists():
        raise ValueError(
            f"Option library config not found: {library_path}\n"
            f"  umbrella_config_dir: {umbrella_dir}\n"
            f"  options_folder_location: {options_folder_location}\n"
            f"  option_library: {option_library_path}"
        )

    option_library, resolved_option_library_config = OptionLibraryLoader.load_library(
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
    
    # Attach resolved config to wrapped environment for saving
    # Include the absolute library path for reproducibility
    resolved_option_library_config['library_config_path'] = str(library_path)
    wrapped_env.resolved_option_library_config = resolved_option_library_config
    
    return wrapped_env


def create_agent(config: Dict[str, Any], env: gym.Env, tb_log_path: Optional[str] = None, verbose: int = 0) -> Any:
    """Instantiate RL agent (PPO, A2C, RecurrentPPO, HRL_PPO, HRL_RPPO, or MBPO) from config.
    
    Extracts algorithm type and hyperparameters from config, then creates the
    appropriate agent class. For standard agents (PPO/A2C/RecurrentPPO), uses
    'MlpPolicy' (or 'MlpLstmPolicy' for RecurrentPPO). For MBPO, returns an MBPOAgent
    instance that orchestrates model-based policy optimization.
    
    Args:
        config (Dict[str, Any]): Full experiment config dictionary. Must contain:
            - 'algorithm': 'PPO' | 'A2C' | 'RecurrentPPO' | 'HRL_PPO' | 'HRL_RPPO' | 'MBPO'
            - '{algorithm_lowercase}': Dict with algorithm-specific hyperparameters
              (e.g., 'ppo': {'learning_rate': 3e-4, 'n_steps': 2048, ...})
              (e.g., 'mbpo': {...}, 'dynamics_model': {...} for MBPO)
        env (gym.Env): Training environment instance (from create_environment).
        tb_log_path (str, optional): Path for tensorboard logs. If None, no logging.
        verbose (int): Verbosity level for stable-baselines3 output. Default: 0 (silent).
    
    Returns:
        PPO | A2C | RecurrentPPO | MBPOAgent: Initialized agent ready for training.
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
        
        # Check if masking should be used for clipped transitions
        hrl_config = config.get('hrl', {})
        exclude_clipped = hrl_config.get('exclude_clipped_manager_steps_from_training', False)
        
        PPOClass = PPO_Masked if exclude_clipped else PPO
        
        agent = PPOClass(
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
    elif algorithm == 'HRL_RPPO':
        # Hierarchical RL with options-based wrapper and recurrent manager
        # Env is already wrapped with OptionsWrapper before reaching here
        recurrent_ppo_config = config.get('recurrent_ppo', {})
        lstm_kwargs_config = config.get('lstm_kwargs', {})
        learning_rate = recurrent_ppo_config.get('learning_rate', 3.0e-4)

        lstm_policy_kwargs = policy_kwargs.copy()
        lstm_policy_kwargs.update({
            'lstm_hidden_size': lstm_kwargs_config.get('lstm_hidden_size', 64),
            'n_lstm_layers': lstm_kwargs_config.get('n_lstm_layers', 1),
            'enable_critic_lstm': lstm_kwargs_config.get('enable_critic_lstm', True),
        })

        # Check if masking should be used for clipped transitions
        hrl_config = config.get('hrl', {})
        exclude_clipped = hrl_config.get('exclude_clipped_manager_steps_from_training', False)
        
        RecurrentPPOClass = RecurrentPPO_Masked if exclude_clipped else RecurrentPPO

        agent = RecurrentPPOClass(
            policy='MlpLstmPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=recurrent_ppo_config.get('n_steps', 256),
            batch_size=recurrent_ppo_config.get('batch_size', 64),
            n_epochs=recurrent_ppo_config.get('n_epochs', 10),
            gamma=recurrent_ppo_config.get('gamma', 0.99),
            gae_lambda=recurrent_ppo_config.get('gae_lambda', 0.95),
            clip_range=recurrent_ppo_config.get('clip_range', 0.2),
            ent_coef=recurrent_ppo_config.get('ent_coef', 0.02),
            vf_coef=recurrent_ppo_config.get('vf_coef', 0.5),
            max_grad_norm=recurrent_ppo_config.get('max_grad_norm', 0.5),
            policy_kwargs=lstm_policy_kwargs,
            verbose=recurrent_ppo_config.get('verbose', 1),
            tensorboard_log=tb_log_path,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent


def setup_callbacks(config: Dict[str, Any], run_dir: str, eval_env: Optional[gym.Env] = None, stop_after_n_episodes: Optional[int] = None) -> list:
    """Set up stable-baselines3 training callbacks.
    
    Creates PatientStatsLoggingCallback (always active), EpisodeCounterCallback
    (always active; optionally stops training after a target number of actual
    episodes — correct under variable-length/boundary-clipped episodes),
    DetailedEvalCallback or standard EvalCallback (if eval_env provided), and
    CheckpointCallback. Automatically creates subdirectories in run_dir for
    logs, checkpoints, and eval_logs.
    
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
        stop_after_n_episodes (int, optional): If provided, an EpisodeCounterCallback
            will stop training after this many actual completed episodes. This makes
            the training budget accurate under variable-length episodes (e.g. when
            HRL boundary clipping terminates episodes early). When None, the
            EpisodeCounterCallback is still added but only counts/logs episodes;
            training budget is controlled solely by model.learn(total_timesteps=...).
    
    Returns:
        list: List of callback instances to pass to agent.learn(callback=...).
    
    Example:
        >>> callbacks = setup_callbacks(config, run_dir, eval_env=eval_env,
        ...                             stop_after_n_episodes=total_episodes)
        >>> agent.learn(total_timesteps=total_episodes * max_time_steps, callback=callbacks)
    """
    from abx_amr_simulator.callbacks import PatientStatsLoggingCallback, DetailedEvalCallback, EpisodeCounterCallback

    def _resolve_detailed_eval_callback_class() -> type:
        """Resolve optional callback-class override for trajectory evaluation.

        Expected training config keys:
        - training.detailed_eval_callback_module: import path or filesystem path
        - training.detailed_eval_callback_class: class name in that module

        If unset, returns canonical DetailedEvalCallback.
        """
        module_spec = training_config.get('detailed_eval_callback_module', None)
        class_name = training_config.get('detailed_eval_callback_class', None)

        if module_spec is None and class_name is None:
            return DetailedEvalCallback

        if not module_spec or not class_name:
            raise ValueError(
                "Both training.detailed_eval_callback_module and "
                "training.detailed_eval_callback_class must be provided when "
                "overriding the detailed evaluation callback."
            )

        resolved_module = None
        import_error: Optional[Exception] = None

        try:
            resolved_module = importlib.import_module(module_spec)
        except Exception as exc:
            import_error = exc

        if resolved_module is None:
            module_path = Path(module_spec)
            if not module_path.is_absolute():
                umbrella_dir = config.get('_umbrella_config_dir')
                if umbrella_dir is None:
                    raise ValueError(
                        "Callback override module path is relative but _umbrella_config_dir "
                        "is missing from merged config. "
                        f"module='{module_spec}', original_import_error={import_error}"
                    )
                module_path = (Path(umbrella_dir) / module_path).resolve()
            else:
                module_path = module_path.resolve()

            if not module_path.exists():
                raise ValueError(
                    "Detailed eval callback override module is not importable and filesystem "
                    f"path does not exist: '{module_spec}' -> '{module_path}'. "
                    f"Original import error: {import_error}"
                )

            dynamic_module_name = f"abx_amr_simulator_eval_callback_plugin_{module_path.stem}"
            module_loader_spec = importlib.util.spec_from_file_location(
                name=dynamic_module_name,
                location=str(module_path),
            )
            if module_loader_spec is None or module_loader_spec.loader is None:
                raise ImportError(
                    f"Failed to create import spec for callback module path '{module_path}'."
                )
            resolved_module = importlib.util.module_from_spec(module_loader_spec)
            module_loader_spec.loader.exec_module(resolved_module)

        if not hasattr(resolved_module, class_name):
            raise AttributeError(
                "Detailed eval callback class not found in override module: "
                f"module='{module_spec}', class='{class_name}'."
            )

        resolved_class = getattr(resolved_module, class_name)
        if not isinstance(resolved_class, type):
            raise TypeError(
                "Detailed eval callback override must reference a class type, got "
                f"'{type(resolved_class).__name__}'."
            )
        if not issubclass(resolved_class, DetailedEvalCallback):
            raise TypeError(
                "Detailed eval callback override class must inherit from DetailedEvalCallback; "
                f"got '{resolved_class.__name__}'."
            )
        return resolved_class
    
    callbacks = []
    
    training_config = config.get('training', {})
    # Use converted timesteps if available (from episode-based config)
    eval_freq = training_config.get('_converted_eval_freq', training_config.get('eval_freq', 5000))
    num_eval_episodes = training_config.get('num_eval_episodes', 10)
    save_freq = training_config.get('_converted_save_freq', training_config.get('save_freq', 10000))
    log_patient_trajectories = training_config.get('log_patient_trajectories', False)
    if training_config.get('log_personalized_patient_attributes', False):
        raise ValueError(
            "training.log_personalized_patient_attributes is no longer supported in canonical "
            "abx_amr_simulator callbacks. Configure a custom detailed eval callback subclass via "
            "training.detailed_eval_callback_module + training.detailed_eval_callback_class."
        )
    
    # Patient stats logging callback (always active during training)
    patient_stats_callback = PatientStatsLoggingCallback()
    callbacks.append(patient_stats_callback)

    # Episode counter callback (always active).
    # When stop_after_n_episodes is set, training terminates after exactly that
    # many actual episodes — important under HRL boundary clipping where episodes
    # can be shorter than max_time_steps.
    episode_counter_callback = EpisodeCounterCallback(
        stop_after_n_episodes=stop_after_n_episodes,
        verbose=1,
    )
    callbacks.append(episode_counter_callback)
    
    # Evaluation callback (only if an eval_env is provided)
    if eval_env is not None:
        # Wrap eval_env in DummyVecEnv if needed
        vec_eval_env = DummyVecEnv([lambda env=eval_env: env])
        
        if log_patient_trajectories:
            detailed_eval_callback_class = _resolve_detailed_eval_callback_class()
            detailed_eval_callback_kwargs = training_config.get('detailed_eval_callback_kwargs', {})
            if not isinstance(detailed_eval_callback_kwargs, dict):
                raise ValueError(
                    "training.detailed_eval_callback_kwargs must be a dictionary when provided."
                )
            # Use DetailedEvalCallback when we want full patient trajectories
            # DetailedEvalCallback will create run_dir/eval_logs/ automatically
            eval_callback = detailed_eval_callback_class(
                eval_env=vec_eval_env,
                n_eval_episodes=num_eval_episodes,
                eval_freq=eval_freq,
                log_path=run_dir,
                best_model_save_path=os.path.join(run_dir, 'checkpoints'),
                deterministic=True,
                render=False,
                **detailed_eval_callback_kwargs,
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
    
    Writes the fully resolved agent and environment configuration to 
    <run_dir>/full_agent_env_config.yaml for reproducibility and experiment 
    record-keeping. This is the complete, merged umbrella config (all subconfigs 
    resolved and all parameter overrides applied).
    
    Args:
        config (Dict[str, Any]): Full experiment configuration dictionary (as loaded
            by load_config, including any command-line overrides).
        run_dir (str): Absolute path to run directory (from create_run_directory).
    
    Example:
        >>> save_training_config(config, run_dir)
        >>> # Creates: <run_dir>/full_agent_env_config.yaml
    """
    # Ensure run directory exists
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # Primary (new) filename
    config_path = os.path.join(run_dir, 'full_agent_env_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Note: do not write legacy `config.yaml` here. Tests and tools should
    # rely on the canonical `full_agent_env_config.yaml` filename. If
    # legacy compatibility is required, update calling code or tests.


def save_option_library_config(resolved_config: Dict[str, Any], run_dir: str):
    """Save fully resolved option library configuration as YAML in run directory.
    
    Writes the resolved option library configuration (with absolute paths and 
    all option specs merged) to <run_dir>/full_options_library.yaml. Only used 
    for HRL experiments.
    
    Args:
        resolved_config (Dict[str, Any]): Resolved option library configuration
            (returned by OptionLibraryLoader.load_library as second return value).
        run_dir (str): Absolute path to run directory (from create_run_directory).
    
    Example:
        >>> # After wrapping environment for HRL:
        >>> save_option_library_config(wrapped_env.resolved_option_library_config, run_dir)
        >>> # Creates: <run_dir>/full_options_library.yaml
    """
    config_path = os.path.join(run_dir, 'full_options_library.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(resolved_config, f, default_flow_style=False)


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
