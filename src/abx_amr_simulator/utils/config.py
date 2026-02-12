"""
Configuration loading and merging utilities.

Handles both legacy and nested YAML config formats with support for:
- Single config files with all sections
- Hierarchical configs that reference component configs
- Command-line parameter overrides with dot notation
"""

import os
import pdb
import yaml
import copy
from importlib.resources import files
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file.
    
    Supports three config formats:
    1. **Flat format**: Single YAML file containing all config sections (environment,
       reward_calculator, patient_generator, training, etc.) as nested dictionaries.
    2. **Nested format (legacy)**: Base YAML that references component config files with
       relative paths (e.g., 'environment: ../environment/default.yaml'). Component 
       configs are loaded and merged into a single dictionary.
    3. **Nested format (modern)**: Base YAML that explicitly specifies folder locations:
       - `config_folder_location`: Path to folder containing component config folders
       - `options_folder_location`: Path to folder containing option libraries
       Component paths are then relative to these folders (no `../` needed).
    
    Modern nested format enables better path management: all paths are relative to
    explicitly declared locations rather than implicit relative navigation.
    
    Args:
        config_path (str): Absolute or relative path to YAML config file.
    
    Returns:
        Dict[str, Any]: Merged configuration dictionary with all component configs
            resolved and loaded. Contains keys: 'environment', 'reward_calculator',
            'patient_generator', 'training', 'algorithm', algorithm-specific hyperparams,
            and optionally 'config_folder_location', 'options_folder_location'.
    
    Example:
        >>> config = load_config('experiments/configs/umbrella_configs/base_experiment.yaml')
        >>> print(config['environment']['num_patients_per_time_step'])
        10
    """
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if this is a nested config
    is_nested = (
        isinstance(config.get('environment'), str) or
        isinstance(config.get('reward_calculator'), str) or
        isinstance(config.get('agent_algorithm'), str)
    )
    
    if not is_nested:
        # Flat format - return as-is
        return config
    
    # Nested format - load and merge component configs
    umbrella_dir = Path(config_path).parent
    
    # Determine base directory for config resolution
    if 'config_folder_location' in config:
        # Modern format: use explicit config folder location
        config_folder_location = config['config_folder_location']
        if not Path(config_folder_location).is_absolute():
            # Resolve relative to umbrella config location
            config_base_dir = (umbrella_dir / config_folder_location).resolve()
        else:
            config_base_dir = Path(config_folder_location).resolve()
    else:
        # Legacy format: use umbrella config's parent directory
        config_base_dir = umbrella_dir
    
    merged_config = {}
    
    # Store the umbrella config path for later use (e.g., by HRL wrapper)
    merged_config['_umbrella_config_dir'] = str(umbrella_dir.resolve())
    
    # Load environment config
    if 'environment' in config and isinstance(config['environment'], str):
        env_path = config_base_dir / config['environment']
        with open(env_path, 'r') as f:
            env_config = yaml.safe_load(f)
        merged_config['environment'] = env_config
    
    # Load reward_calculator config
    if 'reward_calculator' in config and isinstance(config['reward_calculator'], str):
        reward_path = config_base_dir / config['reward_calculator']
        with open(reward_path, 'r') as f:
            reward_config = yaml.safe_load(f)
        merged_config['reward_calculator'] = reward_config
    
    # Load patient_generator config
    if 'patient_generator' in config and isinstance(config['patient_generator'], str):
        patient_gen_path = config_base_dir / config['patient_generator']
        with open(patient_gen_path, 'r') as f:
            patient_gen_config = yaml.safe_load(f)
            
        # Merge patient_generator config at top level
        merged_config['patient_generator'] = patient_gen_config
    
    # Load agent_algorithm config
    if 'agent_algorithm' in config and isinstance(config['agent_algorithm'], str):
        algo_path = config_base_dir / config['agent_algorithm']
        with open(algo_path, 'r') as f:
            algo_config = yaml.safe_load(f)
        # Merge algorithm config at top level
        merged_config.update(algo_config)
    
    # Copy metadata and training from base config when provided directly
    for key in ['seed', 'output_dir', 'run_name', 'config_folder_location', 'options_folder_location', 'hrl']:
        if key in config and not isinstance(config[key], str):
            merged_config[key] = config[key]
        elif key in config and isinstance(config[key], str):
            # For string keys like config_folder_location, preserve them for later use
            merged_config[key] = config[key]

    # Training block (non-string) should be merged through
    if 'training' in config and not isinstance(config['training'], str):
        merged_config['training'] = copy.deepcopy(config['training'])
    # Handle seed at training level
    if 'seed' in config:
        if 'training' not in merged_config:
            merged_config['training'] = {}
        merged_config['training']['seed'] = config['seed']
    
    return merged_config


def apply_subconfig_overrides(configs_dir: str, orig_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply command-line subconfig overrides (swap component configs at runtime).
    
    Enables swapping entire component config files via CLI, e.g.:
        python train.py --config base.yaml -o "environment_subconfig=environment/high_amr.yaml"
    
    Filters overrides dict for keys containing 'subconfig', loads the referenced YAML
    files, and replaces the corresponding sections in the original config.
    
    Args:
        configs_dir (str): Base directory containing component config folders
            (environment/, reward_calculator/, etc.).
        orig_config (Dict[str, Any]): Original config dict (from load_config).
        overrides (Dict[str, Any]): Override dict from command-line parsing. Keys
            containing 'subconfig' trigger component replacement:
            - 'environment_subconfig': Replaces 'environment' section
            - 'reward_calculator_subconfig': Replaces 'reward_calculator' section
            - 'patient_generator_subconfig': Replaces 'patient_generator' section
            - 'agent_algorithm_subconfig': Replaces algorithm section
    
    Returns:
        Dict[str, Any]: Updated config with subconfig overrides applied.
    
    Example:
        >>> overrides = {'environment_subconfig': 'environment/high_amr.yaml'}
        >>> config = apply_subconfig_overrides('experiments/configs', config, overrides)
    """
    
    # Check if overrides contains keys that have the word 'subconfig' in them, and if so, only keep those key/value pairs:
    subconfig_overrides = {k: v for k, v in overrides.items() if 'subconfig' in k}
    
    # Now that we have the subconfig key/value pairs, use the value to fetch the override subconfig.
    # First, make a copy of the original configuration:
    overridden_config = copy.deepcopy(orig_config)
    
    # Next, iterate through each subconfig override:
    for key, value in subconfig_overrides.items():
        # The key will be something like 'environment_subconfig' - we need to extract the actual config type:
        config_type = key.replace('_subconfig', '')
        
        # Now, load the new subconfig from the provided path:
        config_dir = Path(configs_dir)
        subconfig_path = config_dir / value
        
        with open(subconfig_path, 'r') as f:
            subconfig = yaml.safe_load(f)
        
        # Now, depending on the config_type, we need to place this subconfig in the right location in overridden_config:
        if config_type == 'environment':
            overridden_config['environment'] = subconfig
        elif config_type == 'reward_calculator':
            overridden_config['reward_calculator'] = subconfig
        elif config_type == 'patient_generator':
            if 'environment' not in overridden_config:
                overridden_config['environment'] = {}
            overridden_config['environment']['patient_generator'] = subconfig
        elif config_type == 'agent_algorithm':
            overridden_config.update(subconfig)
        else:
            raise ValueError(f"Unknown subconfig type: {config_type}")
    
    return overridden_config


def apply_param_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply command-line parameter overrides using dot notation.
    
    Modifies config in-place by setting nested dictionary values via dot-separated
    keys. Automatically creates intermediate dictionaries if they don't exist.
    Ignores keys containing 'subconfig' (handled by apply_subconfig_overrides).
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to modify.
        overrides (Dict[str, Any]): Override dict with dot-notation keys:
            - 'reward_calculator.lambda_weight': 0.8 → config['reward_calculator']['lambda_weight'] = 0.8
            - 'environment.num_patients_per_time_step': 20
            - 'training.seed': 42
    
    Returns:
        Dict[str, Any]: Modified config dictionary (same object as input).
    
    Example:
        >>> overrides = {'reward_calculator.lambda_weight': 0.8, 'training.seed': 42}
        >>> config = apply_param_overrides(config, overrides)
        >>> assert config['reward_calculator']['lambda_weight'] == 0.8
    """
    # Remove any keys that contain 'subconfig' - those are handled separately
    overrides = {k: v for k, v in overrides.items() if 'subconfig' not in k}
    
    for key, value in overrides.items():
        
        # Parse dot notation
        keys = key.split('.')
        current = config
        
        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
    
    return config


def get_example_config(name: str) -> Path:
    """Return the path to a bundled example config.

    Args:
        name: Base filename without extension (e.g., "base_experiment" or "ppo_baseline").

    Returns:
        Path to the example config inside the installed package.
    """

    examples_dir = files("abx_amr_simulator").joinpath("configs/examples")
    candidate = examples_dir / f"{name}.yaml"
    return Path(str(candidate))


def setup_config_folders_with_defaults(target_path: Path) -> None:
    """Create config directory scaffold populated with bundled default configs.
    
    Copies bundled default component configs from package into user's target directory.
    Creates nested structure:
        target_path/configs/
            umbrella_configs/base_experiment.yaml, hrl_ppo_default.yaml
            agent_algorithm/default.yaml, ppo.yaml, a2c.yaml, hrl_ppo.yaml, hrl_rppo.yaml
            environment/default.yaml
            patient_generator/default.yaml, patient_generator/default_mixer.yaml
            reward_calculator/default.yaml
    
    Additionally, recursively copies any YAML files whose filenames contain the word
    'default' from the package's configs/defaults/* subfolders into the corresponding
    target subfolders. This ensures new defaults (e.g., patient_generator/default_mixer.yaml)
    are included without having to update this mapping.
    
    Useful for initializing new experiment directories with working baseline configs.
    
    Args:
        target_path (Path): Base directory where configs/ folder should be created.
            Typically project root or experiments/ directory.
    
    Example:
        >>> setup_config_folders_with_defaults(Path('experiments'))
        >>> # Creates: experiments/configs/umbrella_configs/base_experiment.yaml, etc.
    """

    base = Path(target_path) / "configs"
    umbrella_dir = base / "umbrella_configs"
    agent_dir = base / "agent_algorithm"
    env_dir = base / "environment"
    pg_dir = base / "patient_generator"
    rc_dir = base / "reward_calculator"

    for d in [umbrella_dir, agent_dir, env_dir, pg_dir, rc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    defaults_root = files("abx_amr_simulator").joinpath("configs/defaults")

    copy_map = {
        # Umbrella config that stitches together the component defaults
        defaults_root.joinpath("umbrella/base_experiment.yaml"): umbrella_dir / "base_experiment.yaml",
        defaults_root.joinpath("umbrella/hrl_ppo_default.yaml"): umbrella_dir / "hrl_ppo_default.yaml",
        # Agent algorithm extras (not matched by 'default*' copy below)
        defaults_root.joinpath("agent_algorithm/ppo.yaml"): agent_dir / "ppo.yaml",
        defaults_root.joinpath("agent_algorithm/a2c.yaml"): agent_dir / "a2c.yaml",
        defaults_root.joinpath("agent_algorithm/hrl_ppo.yaml"): agent_dir / "hrl_ppo.yaml",
        defaults_root.joinpath("agent_algorithm/hrl_rppo.yaml"): agent_dir / "hrl_rppo.yaml",
        # Explicit core defaults (will also be covered by recursive copy below)
        defaults_root.joinpath("agent_algorithm/default.yaml"): agent_dir / "default.yaml",
        defaults_root.joinpath("environment/default.yaml"): env_dir / "default.yaml",
        defaults_root.joinpath("patient_generator/default.yaml"): pg_dir / "default.yaml",
        defaults_root.joinpath("reward_calculator/default.yaml"): rc_dir / "default.yaml",
    }

    for src, dst in copy_map.items():
        dst.write_bytes(src.read_bytes())

    # Recursively copy any YAML files with 'default' in the filename from
    # configs/defaults/* into the corresponding target configs subfolders.
    # This future-proofs the scaffold (e.g., patient_generator/default_mixer.yaml).
    def _copy_defaults_recursively(src_dir, rel_parts):
        # src_dir is a Traversable (importlib.resources); use iterdir for recursion
        for entry in src_dir.iterdir():
            if entry.is_dir():
                _copy_defaults_recursively(entry, rel_parts + [entry.name])
            elif entry.is_file() and entry.name.endswith('.yaml') and 'default' in entry.name:
                # Expect structure configs/defaults/<component>/.../*.yaml
                if not rel_parts:
                    # No component folder context; skip defensively
                    continue
                component = rel_parts[0]
                sub_rel = rel_parts[1:]
                dst_dir = base / component
                for part in sub_rel:
                    dst_dir = dst_dir / part
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / entry.name
                dst_path.write_bytes(entry.read_bytes())

    _copy_defaults_recursively(defaults_root, [])
