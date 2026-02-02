"""Utilities for visualizing environment behavior and AMR dynamics.

This module provides tools to explore and understand environment parameters before running
full training experiments. Key features:

- Visualize leaky balloon response curves to different puff sequences
- Sample random environment trajectories and inspect observations/rewards
- Understand how parameter changes affect dynamics

Example:
    >>> from abx_amr_simulator.utils.visualization import visualize_environment_behavior
    >>> visualize_environment_behavior(
    ...     config_path='configs/umbrella_configs/base_experiment.yaml',
    ...     output_folder='visualization_outputs'
    ... )
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

from abx_amr_simulator.utils import (
    load_config,
    create_reward_calculator,
    create_patient_generator,
    create_environment,
)


def visualize_environment_behavior(config_path: str, output_folder: str = 'visualization_outputs') -> None:
    """Visualize ABXAMREnv behavior with given environment parameters.
    
    Creates visualizations of:
    1. Leaky balloon response curves for each antibiotic showing AMR evolution
    2. Random trajectory samples showing observations, actions, and rewards
    
    Args:
        config_path (str): Path to experiment config YAML file
        output_folder (str): Folder to save visualization outputs (default: 'visualization_outputs')
    """
    
    # Use current working directory for config resolution
    cwd = Path.cwd()
    
    # Try to resolve config path: absolute, relative to CWD, then package
    config_path_str = config_path
    if not os.path.isabs(config_path_str):
        config_path_cwd = cwd / config_path_str
        if os.path.exists(config_path_cwd):
            config_path_str = str(config_path_cwd)
        else:
            # Fall back to package directory
            project_root = Path(__file__).parent.parent
            config_path_str = str(project_root / config_path_str)
    
    if not os.path.exists(config_path_str):
        print(f"Error: Config file not found at {config_path_str}")
        sys.exit(1)
    
    config = load_config(config_path_str)
    
    print(f"Loading config from: {config_path_str}")
    
    # Create reward model and environment
    reward_calculator = create_reward_calculator(config)
    patient_generator = create_patient_generator(config)
    env = create_environment(
        config=config,
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        wrap_monitor=False
    )
    
    # If output folder does not exist, create it
    output_folder_path = output_folder
    if not os.path.isabs(output_folder_path):
        output_folder_path = str(cwd / output_folder_path)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # Visualize environment behavior. First, show how AMR levels evolve in response to puff sequences.
    # The AMR leaky balloons are an attribute of the environment, contained in the 'amr_balloon_models'
    # attribute of the environment.
    for abx_name, abx_amr_leaky_balloon in env.amr_balloon_models.items():
        print(f"Visualizing behavior for ABX: {abx_name}")
        
        # Define a puff sequence to visualize
        puff_sequence = [1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1]
        
        # Plot and save the leaky balloon response
        title = f"Leaky Balloon Response for {abx_name}"
        fname = f"{abx_name}_leaky_balloon_response.png"
        
        abx_amr_leaky_balloon.plot_leaky_balloon_response_to_puff_sequence(
            puff_sequence=puff_sequence,
            title=title,
            fname=fname,
            save_plot_folder=output_folder_path,
            show_plot=False
        )
        
        print(f"Saved plot to {os.path.join(output_folder_path, fname)}")
    
    # Now, randomly sample different actions, feed them through the environment, and create a CSV file
    # with the observations and rewards.
    
    num_states_and_actions_to_sample = 10
    list_of_states = []
    list_of_actions = []
    list_of_info_dicts = []
    
    for i in range(num_states_and_actions_to_sample):

        # Store the states
        env_state, _ = env.reset()
        list_of_states.append(env_state)
        
        random_action = env.action_space.sample()
        list_of_actions.append(random_action)
        
        # Step the environment
        obs, reward, truncated, done, info = env.step(random_action)
        list_of_info_dicts.append(info)
    
    # Create a dataframe to store the observations, actions, and information from the info dict.
    # The states and actions can be made into dataframes without issue, but the info dicts are nested,
    # and some key/value pairs are not useful.
    
    # Number the columns for the observations and actions
    observations_cols = [f"obs_{i}" for i in range(len(list_of_states[0]))]
    actions_cols = [f"action_{i}" for i in range(len(list_of_actions[0]))]
    
    observations_df = pd.DataFrame(list_of_states, columns=observations_cols)
    actions_df = pd.DataFrame(list_of_actions, columns=actions_cols)
    
    # Iterate through each info dict, remove some key/value pairs that are not useful, and flatten.
    
    list_of_outcome_breakdown_dicts = []
    
    for info in list_of_info_dicts:
        # Remove the following keys if they exist:
        keys_to_remove = [
            'delta_amr_per_antibiotic',
            'individual_rewards',
            'count_clinical_benefits',
            'count_adverse_events',
            'patients_actually_infected'
        ]
        
        for key in keys_to_remove:
            if key in info:
                del info[key]
        
        # Also remove the 'outcomes_breakdown' value, but put it into its own list
        if 'outcomes_breakdown' in info:
            list_of_outcome_breakdown_dicts.append(info['outcomes_breakdown'])
            del info['outcomes_breakdown']
    
    # Now flatten the remaining info dicts into a dataframe
    info_df = pd.json_normalize(list_of_info_dicts)
    
    # Also create a dataframe for the outcomes breakdowns (which are also nested, so flatten them)
    outcomes_breakdown_df = pd.json_normalize(list_of_outcome_breakdown_dicts)
    
    # Concatenate all dataframes into a single dataframe (axis=1 for side-by-side)
    final_df = pd.concat([observations_df, actions_df, info_df, outcomes_breakdown_df], axis=1)
    
    # Save the final dataframe to a CSV file
    final_csv_path = os.path.join(output_folder_path, 'env_behavior_samples.csv')
    final_df.to_csv(final_csv_path, index=False)
    print(f"Saved environment behavior samples to {final_csv_path}")


def main():
    """CLI entry point for environment visualization."""
    parser = argparse.ArgumentParser(description='Visualize ABXAMREnv behavior with given environment parameters')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config YAML file'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        default='visualization_outputs',
        help='Folder to save visualization outputs'
    )
    
    args = parser.parse_args()
    visualize_environment_behavior(config_path=args.config, output_folder=args.output_folder)


if __name__ == "__main__":
    main()
