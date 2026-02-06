"""
Utility functions for training and evaluation of RL agents on ABXAMREnv.

This package is organized into focused modules:
- config: Configuration loading and merging
- factories: Core component instantiation (rewards, patients, envs, agents, callbacks)
- registry: Experiment discovery and tracking
- metrics: Evaluation and visualization

For backward compatibility, all functions are re-exported from this module.
New code can also use: from abx_amr_simulator.utils.config import load_config
"""

# Re-export from config module
from .config import (
    load_config,
    apply_subconfig_overrides,
    apply_param_overrides,
    setup_config_folders_with_defaults,
)

# Re-export from factories module
from .factories import (
    create_reward_calculator,
    create_patient_generator,
    create_environment,
    wrap_environment_for_hrl,
    create_agent,
    setup_callbacks,
    create_run_directory,
    save_training_config,
    save_training_summary,
)

# Re-export from registry module
from .registry import (
    extract_experiment_prefix,
    extract_timestamp_from_run_folder,
    find_experiment_runs,
    load_registry,
    load_registry_csv,
    update_registry,
    update_registry_csv,
    scan_for_experiments,
    identify_new_experiments,
    clear_registry,
    validate_and_clean_registry,
    get_all_timestamps_for_run,
    is_run_completed,
)

# Re-export from metrics module
from .metrics import (
    run_episode_and_get_trajectory,
    plot_metrics_trained_agent,
    plot_metrics_ensemble_agents,
)

# Re-export from visualization module
from .visualization import (
    visualize_environment_behavior,
)

__all__ = [
    # Config
    'load_config',
    'apply_subconfig_overrides',
    'apply_param_overrides',
    # Factories
    'create_reward_calculator',
    'create_patient_generator',
    'create_environment',
    'create_agent',
    'setup_callbacks',
    'create_run_directory',
    'save_training_config',
    'save_training_summary',
    # Registry
    'extract_experiment_prefix',
    'extract_timestamp_from_run_folder',
    'find_experiment_runs',
    'load_registry',
    'load_registry_csv',
    'update_registry',
    'update_registry_csv',
    'scan_for_experiments',
    'identify_new_experiments',
    'clear_registry',
    'validate_and_clean_registry',
    'get_all_timestamps_for_run',
    'is_run_completed',
    # Metrics
    'run_episode_and_get_trajectory',
    'plot_metrics_trained_agent',
    'plot_metrics_ensemble_agents',
]
