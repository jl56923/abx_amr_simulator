# The purpose of this script is to allow the user to visualize the behavior of the ABXAMREnv environment given different environment parameters. Specifically, it saves plots of how the AMR levels evolve in response to a puff sequence, and also what the reward looks like for different actions that are chosen.

"""Environment parameters ABXAMREnv
Use with: python visualize_env_with_params_behavior.py --config experiments/configs/sample_environment_params.yaml
The script visualize_env_with_params_behavior.py visualizes environment behavior given these parameters. This is to allow the user to get a feel for how different environment parameters affect dynamics, and what kind of observations/rewards are generated. Specifically, it saves plots of how the AMR levels evolve in response to a puff sequence, and also what the reward looks like for different actions that are chosen.
"""

# %% Import libraries
import os
import sys
import argparse
from pathlib import Path
import pandas as pd

import pdb

from abx_amr_simulator.utils import (
    load_config,
    create_reward_calculator,
    create_patient_generator,
    create_environment,
)

# %% Parse arguments and load config
def main():
    parser = argparse.ArgumentParser(description='Visualize ABXAMREnv behavior with given environment parameters')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config YAML file'
    )
    
    # Add another argument for where the output plots should be saved
    parser.add_argument(
        '--output_folder',
        type=str,
        default='visualization_outputs',
        help='Folder to save visualization outputs'
    )
    
    args = parser.parse_args()
    
    # Use current working directory for config resolution
    cwd = Path.cwd()
    
    # Try to resolve config path: absolute, relative to CWD, then package
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path_cwd = cwd / args.config
        if os.path.exists(config_path_cwd):
            config_path = str(config_path_cwd)
        else:
            # Fall back to package directory
            project_root = Path(__file__).parent.parent
            config_path = str(project_root / args.config)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    print(f"Loading config from: {config_path}")
    
    # Create reward model and environment
    reward_calculator = create_reward_calculator(config)
    patient_generator = create_patient_generator(config)
    env = create_environment(config=config, reward_calculator=reward_calculator, patient_generator=patient_generator, wrap_monitor=False)
    
    # If output folder does not exist, create it
    output_folder = args.output_folder
    if not os.path.isabs(output_folder):
        output_folder = str(cwd / output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Visualize environment behavior. The first thing that I am interested in is how the AMR levels evolve in response to a puff sequence. The AMR leaky ballooons are an attribute of the environment, contained in the 'amr_balloon_models' attribute of the environment.
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
            save_plot_folder=output_folder,
            show_plot=False
        )
        
        print(f"Saved plot to {os.path.join(output_folder, fname)}")
        
    # Now, let's randomly sample different actions, feed them through the environment, and create a JSON file with the observations and rewards.
    
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
    
    # Create a dataframe to store the observations, the actions, and the information in the info dict.
    # The states and actions can be made into dataframes without issue, but the info dicts are nested, and some of the key/value pairs are not useful.
    
    # Just number the columns for the observations and actions
    observations_cols = [f"obs_{i}" for i in range(len(list_of_states[0]))]
    actions_cols = [f"action_{i}" for i in range(len(list_of_actions[0]))]
    
    observations_df = pd.DataFrame(list_of_states, columns=observations_cols)
    actions_df = pd.DataFrame(list_of_actions, columns=actions_cols)
    
    # Iterate through each info dict, remove some key/value pairs that are not useful, and flatten the dict.
    
    list_of_outcome_breakdown_dicts = []
    
    for info in list_of_info_dicts:
        # Remove the following keys if they exist:
        keys_to_remove = ['delta_amr_per_antibiotic', 'individual_rewards', 'count_clinical_benefits', 'count_adverse_events', 'patients_actually_infected']
        
        for key in keys_to_remove:
            if key in info:
                del info[key]
                
        # Also remove the 'outcomes_breakdown' value, but put it into its own list
        if 'outcomes_breakdown' in info:
            list_of_outcome_breakdown_dicts.append(info['outcomes_breakdown'])
            del info['outcomes_breakdown']
    
    # Now flatten the remaining info dicts into a dataframe
    info_df = pd.json_normalize(list_of_info_dicts)
    
    # Also create a dataframe for the outcomes breakdowns, although the outcome breakdown dicts are also nested., so they also have to be flattened.
    outcomes_breakdown_df = pd.json_normalize(list_of_outcome_breakdown_dicts)
    
    # Concatenate all dataframes into a single dataframe; these all have to be concatenated side by side (axis=1)
    final_df = pd.concat([observations_df, actions_df, info_df, outcomes_breakdown_df], axis=1)
    
    # Save the final dataframe to a CSV file
    final_csv_path = os.path.join(output_folder, 'env_behavior_samples.csv')
    final_df.to_csv(final_csv_path, index=False)
    print(f"Saved environment behavior samples to {final_csv_path}")
    

if __name__ == "__main__":
    main()