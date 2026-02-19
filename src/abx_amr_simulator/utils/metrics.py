"""
Evaluation and metrics visualization utilities.

Functions for running episodes, collecting trajectories, and generating
publication-quality plots from trained agents and ensemble results.
"""

import os
from typing import Optional
import json

import pdb

import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

from abx_amr_simulator.core import AMR_LeakyBalloon

# Convert numpy types to native Python types recursively
def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj


# ====================== HRL Diagnostic Helpers ======================

def _extract_option_sequences_from_trajectories(trajectories: list) -> list:
    """Extract sequence of option names from trajectory infos.
    
    Args:
        trajectories: List of trajectory dicts with 'infos' key
    
    Returns:
        List of option name sequences (one per trajectory)
    """
    option_sequences = []
    for traj in trajectories:
        option_seq = []
        for info in traj.get('infos', [])[:-1]:  # Exclude final info
            if isinstance(info, dict) and 'option_name' in info:
                option_seq.append(info['option_name'])
        if option_seq:
            option_sequences.append(option_seq)
    return option_sequences


def _compute_inter_decision_times(trajectories: list) -> list:
    """Compute distribution of steps between macro-decisions.
    
    Args:
        trajectories: List of trajectory dicts with 'infos' key
    
    Returns:
        List of inter-decision times (spans across all trajectories)
    """
    inter_times = []
    for traj in trajectories:
        option_seq = []
        durations = []
        for info in traj.get('infos', [])[:-1]:  # Exclude final info
            if isinstance(info, dict) and 'option_name' in info:
                opt_name = info.get('option_name')
                opt_duration = info.get('option_duration', 1)
                option_seq.append(opt_name)
                durations.append(opt_duration)
        
        # Each duration is the inter-decision time
        inter_times.extend(durations)
    
    return inter_times if inter_times else [0]


def _compute_macro_rewards_per_option(trajectories: list) -> dict:
    """Compute distribution of macro-rewards per option.
    
    Args:
        trajectories: List of trajectory dicts with 'rewards' and 'infos' keys
    
    Returns:
        Dict mapping option_name -> list of macro-rewards
    """
    macro_rewards_per_option = {}
    
    for traj in trajectories:
        rewards = traj.get('rewards', [])
        infos = traj.get('infos', [])[:-1]  # Exclude final info
        
        for reward, info in zip(rewards, infos):
            if isinstance(info, dict) and 'option_name' in info:
                opt_name = info['option_name']
                if opt_name not in macro_rewards_per_option:
                    macro_rewards_per_option[opt_name] = []
                macro_rewards_per_option[opt_name].append(float(reward))
    
    return macro_rewards_per_option


def _bin_amr_states(amr_levels_list: list, num_bins: int = 4) -> list:
    """Bin continuous AMR levels into discrete bins.
    
    Args:
        amr_levels_list: List of (amr_a, amr_b) tuples or AMR dicts
        num_bins: Number of bins (default 4 for quartiles)
    
    Returns:
        List of bin indices
    """
    bin_indices = []
    for amr in amr_levels_list:
        if isinstance(amr, dict):
            # Assume dict with antibiotic names as keys
            values = list(amr.values())
            amr_mean = np.mean(values) if values else 0.5
        else:
            # Tuple or list
            amr_mean = np.mean(amr) if amr else 0.5
        
        bin_idx = min(int(amr_mean * num_bins), num_bins - 1)
        bin_indices.append(bin_idx)
    
    return bin_indices


def _compute_option_amr_contingency(trajectories: list, amr_extractor, num_amr_bins: int = 4) -> dict:
    """Build contingency table: (amr_bin, option) -> frequency.
    
    Args:
        trajectories: List of trajectory dicts
        amr_extractor: Function to extract AMR from trajectory at step i
        num_amr_bins: Number of AMR bins
    
    Returns:
        Dict: {amr_bin: {option_name: count}}
    """
    contingency = {i: {} for i in range(num_amr_bins)}
    
    for traj in trajectories:
        infos = traj.get('infos', [])[:-1]  # Exclude final
        
        for step_idx, info in enumerate(infos):
            if isinstance(info, dict) and 'option_name' in info:
                opt_name = info['option_name']
                
                # Extract AMR at this step
                try:
                    amr_at_step = amr_extractor(traj, step_idx)
                    amr_bin = _bin_amr_states([amr_at_step], num_amr_bins)[0]
                except:
                    amr_bin = 0  # Default bin on error
                
                # Increment contingency count
                if opt_name not in contingency[amr_bin]:
                    contingency[amr_bin][opt_name] = 0
                contingency[amr_bin][opt_name] += 1
    
    return contingency


def _compute_option_transitions(trajectories: list) -> tuple:
    """Compute option transition statistics (bigrams and trigrams).
    
    Args:
        trajectories: List of trajectory dicts with option sequences
    
    Returns:
        Tuple of (bigram_freq_matrix, top_trigrams)
        - bigram_freq_matrix: Dict[option_from][option_to] -> count
        - top_trigrams: List of (trigram, count) sorted by count descending
    """
    # Collect all option names
    all_options = set()
    all_bigrams = {}
    all_trigrams = {}
    
    for traj in trajectories:
        option_seq = []
        for info in traj.get('infos', [])[:-1]:  # Exclude final
            if isinstance(info, dict) and 'option_name' in info:
                option_seq.append(info['option_name'])
                all_options.add(info['option_name'])
        
        # Collect bigrams
        for i in range(len(option_seq) - 1):
            bigram = (option_seq[i], option_seq[i + 1])
            all_bigrams[bigram] = all_bigrams.get(bigram, 0) + 1
        
        # Collect trigrams
        for i in range(len(option_seq) - 2):
            trigram = (option_seq[i], option_seq[i + 1], option_seq[i + 2])
            all_trigrams[trigram] = all_trigrams.get(trigram, 0) + 1
    
    # Build bigram matrix
    all_options_list = sorted(all_options)
    bigram_matrix = {opt: {opt2: 0 for opt2 in all_options_list} for opt in all_options_list}
    for (from_opt, to_opt), count in all_bigrams.items():
        bigram_matrix[from_opt][to_opt] = count
    
    # Sort trigrams by frequency
    top_trigrams = sorted(all_trigrams.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return bigram_matrix, top_trigrams


def create_overall_outcomes_summary_dict(
    count_clinical_benefits_cumsum,
    count_clinical_failures_cumsum,
    count_adverse_events_cumsum,
    not_infected_no_treatment_cumsum,
    not_infected_treated_cumsum,
    infected_no_treatment_cumsum,
    infected_treated_overall_cumsum,
    infected_treated_sensitive_per_abx_cumsum,
    infected_treated_resistant_per_abx_cumsum,
    count_prescriptions_per_abx_cumsum,
    total_reward_cumsum,
    antibiotic_names,
):
    """Create summary dictionary of end-of-episode outcome counts.
    
    Extracts final cumulative values from time series and structures them into
    a summary dictionary for JSON serialization. Used by all plotting functions
    to create consistent outcome summaries.
    
    Args:
        count_clinical_benefits_cumsum (np.ndarray): Cumulative clinical benefits over time
        count_clinical_failures_cumsum (np.ndarray): Cumulative clinical failures over time
        count_adverse_events_cumsum (np.ndarray): Cumulative adverse events over time
        not_infected_no_treatment_cumsum (np.ndarray): Cumulative count of uninfected untreated patients
        not_infected_treated_cumsum (np.ndarray): Cumulative count of uninfected treated patients
        infected_no_treatment_cumsum (np.ndarray): Cumulative count of infected untreated patients
        infected_treated_overall_cumsum (np.ndarray): Cumulative count of infected treated patients (all antibiotics)
        infected_treated_sensitive_per_abx_cumsum (dict): Dict mapping abx_name -> cumulative sensitive infections treated
        infected_treated_resistant_per_abx_cumsum (dict): Dict mapping abx_name -> cumulative resistant infections treated
        count_prescriptions_per_abx_cumsum (dict): Dict mapping abx_name -> cumulative prescriptions
            total_reward_cumsum (np.ndarray): Cumulative total reward over time
        antibiotic_names (list): List of antibiotic names
    
    Returns:
        dict: Summary dictionary with end-of-episode counts (native Python types, JSON-serializable)
    
    Example:
        >>> summary = create_overall_outcomes_summary_dict(
        ...     count_clinical_benefits_cumsum=np.cumsum([1, 2, 3]),
        ...     count_clinical_failures_cumsum=np.cumsum([0, 1, 1]),
        ...     count_adverse_events_cumsum=np.cumsum([0, 0, 1]),
        ...     not_infected_no_treatment_cumsum=np.cumsum([1, 0, 0]),
        ...     not_infected_treated_cumsum=np.cumsum([0, 1, 0]),
        ...     infected_no_treatment_cumsum=np.cumsum([0, 0, 1]),
        ...     infected_treated_overall_cumsum=np.cumsum([0, 1, 1]),
        ...     infected_treated_sensitive_per_abx_cumsum={'A': np.cumsum([0, 1, 1]), 'B': np.cumsum([0, 0, 0])},
        ...     infected_treated_resistant_per_abx_cumsum={'A': np.cumsum([0, 0, 0]), 'B': np.cumsum([0, 0, 0])},
        ...     count_prescriptions_per_abx_cumsum={'A': np.cumsum([0, 1, 1]), 'B': np.cumsum([0, 0, 0])},
        ...     total_reward_cumsum=np.cumsum([1.0, 0.5, -0.25]),
        ...     antibiotic_names=['A', 'B']
        ... )
        >>> summary['overall_count_clinical_benefits']
        6
    """
    # Extract final values (end of episode)
    overall_count_clinical_benefits = count_clinical_benefits_cumsum[-1]
    overall_count_clinical_failures = count_clinical_failures_cumsum[-1]
    overall_count_adverse_events = count_adverse_events_cumsum[-1]
    
    overall_not_infected_no_treatment_count = not_infected_no_treatment_cumsum[-1]
    overall_not_infected_treated_count = not_infected_treated_cumsum[-1]
    overall_infected_no_treatment_count = infected_no_treatment_cumsum[-1]
    overall_infected_treated_count = infected_treated_overall_cumsum[-1]
    
    # Per-antibiotic breakdowns
    overall_sensitive_infection_treated_count_per_abx_dict = {
        abx_name: infected_treated_sensitive_per_abx_cumsum[abx_name][-1] 
        for abx_name in antibiotic_names
    }
    
    overall_resistant_infection_treated_count_per_abx_dict = {
        abx_name: infected_treated_resistant_per_abx_cumsum[abx_name][-1] 
        for abx_name in antibiotic_names
    }
    
    overall_abx_prescriptions_count_per_abx = {
        abx_name: count_prescriptions_per_abx_cumsum[abx_name][-1] 
        for abx_name in antibiotic_names
    }
    
    # Aggregate across antibiotics
    overall_sensitive_infection_treated_count = sum(overall_sensitive_infection_treated_count_per_abx_dict.values())
    overall_resistant_infection_treated_count = sum(overall_resistant_infection_treated_count_per_abx_dict.values())
    overall_abx_prescriptions_count = sum(overall_abx_prescriptions_count_per_abx.values())
    # Extract total reward
    overall_total_reward = total_reward_cumsum[-1]
    
    
    # Build summary dictionary
    overall_outcomes_summary_dict = {
            'overall_total_reward': overall_total_reward,
        'overall_count_clinical_benefits': overall_count_clinical_benefits,
        'overall_count_clinical_failures': overall_count_clinical_failures,
        'overall_count_adverse_events': overall_count_adverse_events,
        'overall_not_infected_no_treatment_count': overall_not_infected_no_treatment_count,
        'overall_not_infected_treated_count': overall_not_infected_treated_count,
        'overall_infected_no_treatment_count': overall_infected_no_treatment_count,
        'overall_infected_treated_count': overall_infected_treated_count,
        'overall_sensitive_infection_treated_count_per_abx_dict': overall_sensitive_infection_treated_count_per_abx_dict,
        'overall_resistant_infection_treated_count_per_abx_dict': overall_resistant_infection_treated_count_per_abx_dict,
        'overall_sensitive_infection_treated_count': overall_sensitive_infection_treated_count,
        'overall_resistant_infection_treated_count': overall_resistant_infection_treated_count,
        'overall_abx_prescriptions_count_per_abx': overall_abx_prescriptions_count_per_abx,
        'overall_abx_prescriptions_count': overall_abx_prescriptions_count,
    }
    
    # Convert numpy types to native Python types for JSON serialization
    return convert_to_native_types(overall_outcomes_summary_dict)


# ====================== HRL Diagnostic Plots ======================

def plot_hrl_option_selection_histogram(trajectories: list, option_names: list, output_path: str):
    """Plot [1.1] Option Selection Histogram - frequency of options over trajectories.
    
    Args:
        trajectories: List of trajectory dicts from evaluation episodes
        option_names: Sorted list of option names
        output_path: Path to save PNG
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [WARN] matplotlib not available, skipping option selection histogram")
        return
    
    # Count option frequencies
    option_counts = {opt: 0 for opt in option_names}
    for traj in trajectories:
        for info in traj.get('infos', [])[:-1]:
            if isinstance(info, dict) and 'option_name' in info:
                opt_name = info['option_name']
                if opt_name in option_counts:
                    option_counts[opt_name] += 1
    
    # Compute frequencies (as fractions)
    total_steps = sum(option_counts.values())
    if total_steps == 0:
        print(f"  [WARN] No option data found in trajectories")
        return
    
    frequencies = {opt: count / total_steps for opt, count in option_counts.items()}
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    opts = list(frequencies.keys())
    freqs = list(frequencies.values())
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(opts)))
    ax.bar(opts, freqs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Option Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (fraction of steps)', fontsize=12, fontweight='bold')
    ax.set_title('Option Selection Histogram', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved option selection histogram to {output_path}")


def plot_hrl_macro_decision_frequency(trajectories: list, output_path: str):
    """Plot [1.2] Macro-Decision Frequency - distribution of inter-decision times.
    
    Args:
        trajectories: List of trajectory dicts from evaluation episodes
        output_path: Path to save PNG
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [WARN] matplotlib not available, skipping macro decision frequency")
        return
    
    inter_times = _compute_inter_decision_times(trajectories)
    
    if not inter_times:
        print(f"  [WARN] No inter-decision time data found")
        return
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(inter_times, bins=max(10, len(set(inter_times))), 
            alpha=0.7, edgecolor='black', color='steelblue')
    
    ax.set_xlabel('Inter-Decision Time (steps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Macro-Decision Frequencies', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    mean_time = np.mean(inter_times)
    std_time = np.std(inter_times)
    ax.text(0.98, 0.97, f'Mean: {mean_time:.2f}\nStd: {std_time:.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved macro decision frequency plot to {output_path}")


def plot_hrl_option_effectiveness(trajectories: list, option_names: list, output_path: str):
    """Plot [1.3] Option Effectiveness Over Training - macro-reward per option.
    
    Args:
        trajectories: List of trajectory dicts from evaluation episodes
        option_names: Sorted list of option names
        output_path: Path to save PNG
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [WARN] matplotlib not available, skipping option effectiveness")
        return
    
    macro_rewards = _compute_macro_rewards_per_option(trajectories)
    
    if not macro_rewards:
        print(f"  [WARN] No reward data found")
        return
    
    # Prepare data for box plot (only options that appeared)
    opts_with_data = [opt for opt in option_names if opt in macro_rewards]
    reward_data = [macro_rewards[opt] for opt in opts_with_data]
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(reward_data, labels=opts_with_data, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.tab20(np.linspace(0, 1, len(opts_with_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Option Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro-Reward', fontsize=12, fontweight='bold')
    ax.set_title('Option Effectiveness (Macro-Reward Distribution)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved option effectiveness plot to {output_path}")


def plot_hrl_option_amr_strategy(trajectories: list, option_names: list, output_path: str):
    """Plot [2.1] Option Selection by AMR State - conditional strategy.
    
    Args:
        trajectories: List of trajectory dicts from evaluation episodes
        option_names: Sorted list of option names
        output_path: Path to save PNG
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [WARN] matplotlib not available, skipping option-AMR strategy")
        return
    
    # Need to extract AMR levels from trajectory infos
    # This is a simplified version - assumes info contains 'amr_levels' or we compute from patient data
    def extract_amr(traj, idx):
        """Extract AMR at trajectory step idx."""
        try:
            # Try to get from info dict first
            if idx < len(traj.get('infos', [])):
                info = traj['infos'][idx]
                if isinstance(info, dict):
                    # Look for patient trajectories (if available)
                    if 'patient_trajectories' in info:
                        # Average across patients if available
                        pts = info['patient_trajectories']
                        if pts:
                            return {k: np.mean([p.get(k, 0.5) for p in pts]) 
                                    for k in ['amr_a', 'amr_b']}
            return {'amr_a': 0.5, 'amr_b': 0.5}  # Default fallback
        except:
            return {'amr_a': 0.5, 'amr_b': 0.5}
    
    contingency = _compute_option_amr_contingency(trajectories, extract_amr, num_amr_bins=4)
    
    # Convert contingency to conditional probabilities
    amr_bins = ['Low\n[0-0.25)', 'Med-Low\n[0.25-0.5)', 'Med-High\n[0.5-0.75)', 'High\n[0.75-1.0]']
    
    # Build matrix: rows=AMR bins, cols=options
    matrix = np.zeros((4, len(option_names)))
    for bin_idx in range(4):
        bin_counts = contingency.get(bin_idx, {})
        total = sum(bin_counts.values())
        if total > 0:
            for opt_idx, opt_name in enumerate(option_names):
                matrix[bin_idx, opt_idx] = bin_counts.get(opt_name, 0) / total
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(option_names)))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(option_names, rotation=45, ha='right')
    ax.set_yticklabels(amr_bins)
    
    ax.set_xlabel('Option Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('AMR State', fontsize=12, fontweight='bold')
    ax.set_title('Option Selection by AMR State (Conditional Probability)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(Option | AMR)', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(4):
        for j in range(len(option_names)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved option-AMR strategy plot to {output_path}")


def plot_hrl_option_transitions(trajectories: list, option_names: list, output_path_heatmap: str, 
                               output_path_csv: str):
    """Plot [3.1] Option Transition Frequency - Markov analysis.
    
    Creates:
    - Heatmap of bigram transitions
    - CSV of top-10 trigrams
    
    Args:
        trajectories: List of trajectory dicts from evaluation episodes
        option_names: Sorted list of option names
        output_path_heatmap: Path to save heatmap PNG
        output_path_csv: Path to save trigrams CSV
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [WARN] matplotlib not available, skipping option transitions")
        return
    
    bigram_matrix, top_trigrams = _compute_option_transitions(trajectories)
    
    # Create bigram heatmap
    matrix = np.zeros((len(option_names), len(option_names)))
    for i, from_opt in enumerate(option_names):
        for j, to_opt in enumerate(option_names):
            matrix[i, j] = bigram_matrix.get(from_opt, {}).get(to_opt, 0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(len(option_names)))
    ax.set_yticks(np.arange(len(option_names)))
    ax.set_xticklabels(option_names, rotation=45, ha='right')
    ax.set_yticklabels(option_names)
    
    ax.set_xlabel('Next Option', fontsize=12, fontweight='bold')
    ax.set_ylabel('Current Option', fontsize=12, fontweight='bold')
    ax.set_title('Option Transition Frequency (Bigrams)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Count', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(option_names)):
        for j in range(len(option_names)):
            count = int(matrix[i, j])
            if count > 0:
                text = ax.text(j, i, str(count), ha="center", va="center", 
                             color="white" if count > matrix.max()/2 else "black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path_heatmap, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved option transition heatmap to {output_path_heatmap}")
    
    # Save top trigrams to CSV
    try:
        import csv
        with open(output_path_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Trigram', 'Count'])
            for rank, (trigram, count) in enumerate(top_trigrams, 1):
                trigram_str = ' → '.join(trigram)
                writer.writerow([rank, trigram_str, int(count)])
        print(f"  ✓ Saved top trigrams to {output_path_csv}")
    except Exception as e:
        print(f"  [WARN] Failed to save trigrams CSV: {e}")


def plot_hrl_diagnostics_single_run(model, env, experiment_folder: str, option_library, 
                                    num_eval_episodes: int = 5, figures_folder_name: str = "figures_hrl"):
    """Main wrapper for HRL diagnostic plots - single trained agent.
    
    Orchestrates all 5 HRL diagnostic plots for a single run.
    
    Args:
        model: Trained agent with predict() method
        env: Wrapped environment (OptionsWrapper)
        experiment_folder: Path to experiment results folder
        option_library: OptionLibrary instance
        num_eval_episodes: Number of evaluation episodes to run
        figures_folder_name: Subfolder for figures
    """
    import os
    
    # Create figures folder
    figures_folder = os.path.join(experiment_folder, figures_folder_name)
    os.makedirs(figures_folder, exist_ok=True)
    
    print(f"\n  Running HRL diagnostics ({num_eval_episodes} episodes)...")
    
    # Collect trajectories
    trajectories = []
    for ep in range(num_eval_episodes):
        try:
            traj = run_episode_and_get_trajectory(model=model, env=env, deterministic=True)
            trajectories.append(traj)
        except Exception as e:
            print(f"    [WARN] Episode {ep} failed: {e}")
            continue
    
    if not trajectories:
        print(f"  [ERROR] No successful trajectories collected")
        return
    
    # Get option names
    option_names = sorted(list(option_library.list_options()))
    
    # Generate all plots
    print(f"  Generating plots...")
    
    # Tier 1: Critical
    plot_hrl_option_selection_histogram(
        trajectories, option_names,
        os.path.join(figures_folder, '1_1_option_selection_histogram.png')
    )
    
    plot_hrl_macro_decision_frequency(
        trajectories,
        os.path.join(figures_folder, '1_2_macro_decision_frequency.png')
    )
    
    plot_hrl_option_effectiveness(
        trajectories, option_names,
        os.path.join(figures_folder, '1_3_option_effectiveness.png')
    )
    
    # Tier 2: Important
    plot_hrl_option_amr_strategy(
        trajectories, option_names,
        os.path.join(figures_folder, '2_1_option_amr_strategy.png')
    )
    
    plot_hrl_option_transitions(
        trajectories, option_names,
        os.path.join(figures_folder, '3_1_option_transitions.png'),
        os.path.join(figures_folder, '3_1_option_trigrams.csv')
    )
    
    print(f"\n  ✓ All HRL diagnostics complete! Results saved to {figures_folder}")


def run_episode_and_get_trajectory(model, env, deterministic=True):
    """Run a single episode with a trained agent and collect full trajectory.
    
    Executes one episode from reset to termination/truncation, recording observations,
    actions, rewards, and info dicts at each timestep. Includes safety fallback
    (10000 steps max) to prevent infinite loops if environment doesn't terminate.
    
    Args:
        model: Trained stable-baselines3 agent with predict() method (PPO, DQN, A2C).
        env (gym.Env): Gymnasium environment to run episode in.
        deterministic (bool): If True, use deterministic policy (argmax); if False,
            sample from policy distribution. Default: True.
    
    Returns:
        dict: Trajectory dictionary with keys:
            - 'obs': List of observations (numpy arrays), length = num_steps + 1
            - 'actions': List of actions (numpy arrays), length = num_steps
            - 'rewards': List of scalar rewards, length = num_steps
            - 'infos': List of info dicts, length = num_steps + 1 (includes initial)
    
    Example:
        >>> trajectory = run_episode_and_get_trajectory(model, env, deterministic=True)
        >>> print(len(trajectory['rewards']))  # Number of timesteps in episode
        50
    """
    obs, info = env.reset()
    # Ensure obs is a proper numpy array with explicit dtype for PyTorch compatibility
    obs = np.asarray(obs, dtype=np.float32)
    trajectory = {"obs": [obs], "actions": [], "rewards": [], "infos": [info]}
    
    # Fallback as a safety check
    max_steps_fallback = (env.unwrapped.max_time_steps + 1) if hasattr(env.unwrapped, 'max_time_steps') else 10000
    
    print(env.unwrapped.max_time_steps if hasattr(env.unwrapped, 'max_time_steps') else "No max_time_steps attribute")
    
    # If environment doesn't have max_time_steps, print a warning message:
    if not hasattr(env.unwrapped, 'max_time_steps'):
        print(f"[WARN] Environment does not have 'max_time_steps' attribute; using fallback of {max_steps_fallback} steps.")
    
    step_count = 0
    
    while step_count < max_steps_fallback:
        # Ensure obs is a proper numpy array with explicit dtype for PyTorch compatibility
        obs = np.asarray(obs, dtype=np.float32)
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Normalize action shape/type to match env action space expectations
        # SB3 returns numpy arrays even for scalar actions; MultiDiscrete needs arrays.
        if isinstance(env.action_space, spaces.MultiDiscrete):
            if isinstance(action, np.ndarray):
                action = action.astype(int)
            else:
                action = np.array([int(action)], dtype=int)
        elif isinstance(env.action_space, spaces.Discrete):
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.size == 1 else int(action.squeeze())
            else:
                action = int(action)
        elif isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["infos"].append(info)
        trajectory["obs"].append(obs)
        
        step_count += 1
        
        if terminated or truncated:
            break
    
    if step_count >= max_steps_fallback:
        print(f"[WARN] Episode reached safety limit of {max_steps_fallback} steps without termination/truncation")
    
    return trajectory


def plot_metrics_trained_agent(model, env, experiment_folder, deterministic=True, figures_folder_name: Optional[str] = "figures"):
    """Generate publication-quality plots from a single trained agent trajectory.
    
    Runs one episode, extracts metrics from trajectory, and creates diagnostic plots:
    - Leaky balloon AMR responses (for each antibiotic)
    - AMR levels over time (actual vs. observable)
    - Reward components (total, individual, community)
    - Clinical outcomes (benefits, failures, adverse events)
    - Patient outcome breakdowns
    - Antibiotic prescriptions over time
    
    All figures saved to <experiment_folder>/<figures_folder_name>/ as PNG files.
    
    Args:
        model: Trained stable-baselines3 agent with predict() method.
        env (gym.Env): ABXAMREnv instance to run episode in.
        experiment_folder (str): Absolute path to experiment result directory
            (e.g., 'results/exp_1.a_..._seed1_20260115_143614').
        deterministic (bool): If True, use deterministic policy. Default: True.
        figures_folder_name (str, optional): Subfolder name for figures. If None,
            saves directly to experiment_folder. Default: 'figures'.
    
    Raises:
        ValueError: If experiment_folder doesn't exist.
    
    Example:
        >>> model = PPO.load('results/my_run/checkpoints/best_model.zip')
        >>> env = create_environment(config, rc, pg)
        >>> plot_metrics_trained_agent(model, env, 'results/my_run')
        >>> # Creates: results/my_run/figures/*.png
    """
    
    if not os.path.exists(experiment_folder):
        raise ValueError(f"Experiment folder {experiment_folder} does not exist.")
    
    if figures_folder_name is None:
        # Save figures directly to experiment_folder
        experiment_figures_folder = experiment_folder
    else:
        # Create subfolder for figures
        experiment_figures_folder = os.path.join(experiment_folder, figures_folder_name)
        if not os.path.exists(experiment_figures_folder):
            os.mkdir(experiment_figures_folder)
    
    # First, create a plot of the AMR response for every antibiotic, for a steady dose sequence of [1, 1, 1, ..., 1].
    
    dose_sequence = [1] * 10 + [0] * 10 + [2] * 10
    
    for antibiotic_name, AMR_leakyballoon_params in env.unwrapped.antibiotics_AMR_dict.items():
        # Create a leaky balloon instance with the given parameters
        leaky_balloon = AMR_LeakyBalloon(**AMR_leakyballoon_params)
        leaky_balloon.plot_leaky_balloon_response_to_dose_sequence(
            dose_sequence,
            title=f"AMR Response for {antibiotic_name}",
            fname=f"leaky_balloon_response_{antibiotic_name}.png",
            save_plot_folder=experiment_figures_folder,
            show_plot=False,
        )
        
    # Next, run an episode with the trained agent, and get the traj dictionary:
    trajectory_dict = run_episode_and_get_trajectory(model, env, deterministic=deterministic)
    
    # All the necessary info is in trajectory_dict, in the 'infos' key.
    trajectory_infos = trajectory_dict['infos']
    
    # For convenience, get the antibiotic names from the env:
    antibiotic_names = env.unwrapped.antibiotic_names
    
    # Iterate through the list of info dictionaries in order to extract relevant metrics for plotting.
    actual_AMR_levels_over_time = {abx_name: [] for abx_name in antibiotic_names}
    visible_AMR_levels_over_time = {abx_name: [] for abx_name in antibiotic_names}
    total_reward_over_time = []
    individual_reward_over_time = []
    normalized_individual_reward_over_time = []
    community_reward_over_time = []
    normalized_community_reward_over_time = []
    count_clinical_benefits_over_time = []
    count_clinical_failures_over_time = []
    count_adverse_events_over_time = []
    
    not_infected_no_treatment_count_over_time = []
    not_infected_treated_over_time = []
    infected_no_treatment_count_over_time = []
    infected_treated_count_over_time = {abx_name: {'sensitive_infection_treated': [], 'resistant_infection_treated': []} for abx_name in antibiotic_names}
    count_of_abx_prescriptions_over_time = {abx_name: [] for abx_name in antibiotic_names}
    
    # Skip the first info (from reset) which lacks step-level fields like 'actual_amr_levels'
    for info in trajectory_infos[1:]:
        
        for abx_name in antibiotic_names:
            actual_AMR_levels_over_time[abx_name].append(info['actual_amr_levels'][abx_name])
            visible_AMR_levels_over_time[abx_name].append(info['visible_amr_levels'][abx_name])
            
        total_reward_over_time.append(info['total_reward'])
        individual_reward_over_time.append(info['overall_individual_reward_component'])
        normalized_individual_reward_over_time.append(info['normalized_individual_reward'])
        community_reward_over_time.append(info['overall_community_reward_component'])
        normalized_community_reward_over_time.append(info['normalized_community_reward'])
        count_clinical_benefits_over_time.append(info['count_clinical_benefits'])
        count_clinical_failures_over_time.append(info['count_clinical_failures'])
        count_adverse_events_over_time.append(info['count_adverse_events'])
        
        # Also extract the outcomes_breakdown dictionary, and flatten out the information in it.
        outcomes_dict = info['outcomes_breakdown']
        
        # Extract counts for each outcome type:
        not_infected_no_treatment_count_over_time.append(outcomes_dict['not_infected_no_treatment'])
        not_infected_treated_over_time.append(outcomes_dict['not_infected_treated'])
        infected_no_treatment_count_over_time.append(outcomes_dict['infected_no_treatment'])
        for abx_name in antibiotic_names:
            infected_treated_count_over_time[abx_name]['sensitive_infection_treated'].append(outcomes_dict['infected_treated'][abx_name]['sensitive_infection_treated'])
            infected_treated_count_over_time[abx_name]['resistant_infection_treated'].append(outcomes_dict['infected_treated'][abx_name]['resistant_infection_treated'])
            
            count_of_abx_prescriptions_over_time[abx_name].append((outcomes_dict['infected_treated'][abx_name]['sensitive_infection_treated'] + outcomes_dict['infected_treated'][abx_name]['resistant_infection_treated']))
    
    # The fourth plot in outcome_counts_over_time.png should actually be 'Infected Treated' over time, with both the summed overall numbers of these patients over time, then also per antibiotic. This fourth plot does not care about whether or not the infection was sensitive or resistant, it only cares about the count of 'infected treated'. The images titled 'infected_treated_counts' will instead plot the sensitive vs resistant counts per antibiotic.
    # Therefore, let's aggregate the info in infected_treated_count_over_time in order to get the total infected treated counts over time.
    infected_treated_overall_over_time = []
    infected_treated_per_antibiotic_over_time = {abx_name: [] for abx_name in antibiotic_names}
    
    for abx_name in antibiotic_names:
        infected_treated_per_antibiotic_over_time[abx_name] = [
            s + r for s, r in zip(
                infected_treated_count_over_time[abx_name]['sensitive_infection_treated'],
                infected_treated_count_over_time[abx_name]['resistant_infection_treated'],
            )
        ]

    # I want to take the sum across all antibiotics at each timestep:
    for abx_name in antibiotic_names:
        if not infected_treated_overall_over_time:
            infected_treated_overall_over_time = infected_treated_per_antibiotic_over_time[abx_name]
        else:
            # Add the two lists element by element:
            infected_treated_overall_over_time = [x + y for x, y in zip(infected_treated_overall_over_time, infected_treated_per_antibiotic_over_time[abx_name])]
    
    # Once all the data is extracted, plot each of these into separate figures; similar datastreams will get plotted in the same figure.
    
    # First, plot actual AMR levels and visible AMR levels in two side by side plots; each plot will have all antibiotic AMR levels over time plotted.
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for abx_name in antibiotic_names:
        plt.plot(actual_AMR_levels_over_time[abx_name], label=f"{abx_name} Actual AMR Level")
    plt.xlabel('Timestep')
    plt.ylabel('Actual AMR Level')
    plt.title('Actual AMR Levels Over Time')
    # Add grid:
    plt.grid(True)
    # Set ylim to be -0.05 to 1.05:
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for abx_name in antibiotic_names:
        plt.plot(visible_AMR_levels_over_time[abx_name], label=f"{abx_name} Visible AMR Level")
    plt.xlabel('Timestep')
    plt.ylabel('Visible AMR Level')
    plt.title('Visible AMR Levels Over Time')
    # Add grid:
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "amr_levels_over_time.png"))
    plt.close()
    
    # Next, plot the individual and community reward components over time in one plot, the normalized_individual_reward and normalized community_reward in another plot (next to the first one), and then the total reward in a third plot.
    # What I'm actually interested in is the cumulative reward over time, so I'll create cumulative sums of each of these reward components.
    individual_reward_over_time = np.cumsum(individual_reward_over_time)
    normalized_individual_reward_over_time = np.cumsum(normalized_individual_reward_over_time)
    community_reward_over_time = np.cumsum(community_reward_over_time)
    normalized_community_reward_over_time = np.cumsum(normalized_community_reward_over_time)
    total_reward_over_time = np.cumsum(total_reward_over_time)
    
    # All three of these subplots should have the same y-axis limits for easier comparison.
    ylim_min = min(
        min(individual_reward_over_time),
        min(normalized_individual_reward_over_time),
        min(community_reward_over_time),
        min(normalized_community_reward_over_time),
        min(total_reward_over_time),
    ) - 0.5  # Add some padding
    ylim_max = max(
        max(individual_reward_over_time),
        max(normalized_individual_reward_over_time),
        max(community_reward_over_time),
        max(normalized_community_reward_over_time),
        max(total_reward_over_time),
    ) + 0.5  # Add some padding
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(individual_reward_over_time, label='Individual Reward Component')
    plt.plot(community_reward_over_time, label='Community Reward Component')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Individual and Community Reward Components Over Time')
    plt.ylim(ylim_min, ylim_max)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(normalized_individual_reward_over_time, label='Normalized Individual Reward')
    plt.plot(normalized_community_reward_over_time, label='Normalized Community Reward')
    plt.xlabel('Timestep')
    plt.ylabel('Normalized Reward')
    plt.title('Normalized Individual and Community Rewards Over Time')
    plt.ylim(ylim_min, ylim_max)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(total_reward_over_time, label='Total Reward', color='purple')
    plt.xlabel('Timestep')
    plt.ylabel('Total Reward')
    plt.title('Total Reward Over Time')
    plt.grid(True)
    plt.legend()
    plt.ylim(ylim_min, ylim_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "reward_components_over_time.png"))
    plt.close()
    
    # Next, plot the counts of clinical benefits and adverse events over time in the same plot.
    # Again, what I'm actually interested in is the cumulative counts over time. Also, for all of the outcome counts, the lower y limit is -0.05 to give some padding.
    count_clinical_benefits_over_time = np.cumsum(count_clinical_benefits_over_time)
    count_clinical_failures_over_time = np.cumsum(count_clinical_failures_over_time)
    count_adverse_events_over_time = np.cumsum(count_adverse_events_over_time)
    
    plt.figure(figsize=(10, 5))
    plt.plot(count_clinical_benefits_over_time, label='Count Clinical Benefits', color='green')
    plt.plot(count_clinical_failures_over_time, label='Count Clinical Failures', color='red')
    plt.plot(count_adverse_events_over_time, label='Count Adverse Events', color='purple')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title('Counts of Clinical Benefits, Clinical Failures, and Adverse Events Over Time')
    plt.legend()
    # Add grid:
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-0.05, max(max(count_clinical_benefits_over_time), max(count_clinical_failures_over_time), max(count_adverse_events_over_time)) + 1)
    plt.savefig(os.path.join(experiment_figures_folder, "clinical_benefits_failures_adverse_events_over_time.png"))
    plt.close()
    
    # # Finally, plot the various outcome counts over time in a single figure with multiple subplots. The first plot with include 4 subfigures: not infected no treatment count over time, not infected treated over time, infected no treatment count over time, and count of abx prescriptions over time (each as a separate line per antibiotic, in the same subplot).
    # 
    # Then, each individual antibiotic will get their own figure, which will plot the sensitive and resistant infection treated counts over time.
    
    # First figure with 4 subplots:
    # Again, what I'm interested in is the cumulative counts over time.
    not_infected_no_treatment_count_over_time = np.cumsum(not_infected_no_treatment_count_over_time)
    not_infected_treated_over_time = np.cumsum(not_infected_treated_over_time)
    infected_no_treatment_count_over_time = np.cumsum(infected_no_treatment_count_over_time)
    infected_treated_overall_over_time = np.cumsum(infected_treated_overall_over_time)
    # Do this also per antibiotic:
    for abx_name in antibiotic_names:
        infected_treated_per_antibiotic_over_time[abx_name] = np.cumsum(infected_treated_per_antibiotic_over_time[abx_name])
    
    # The lower limit of y axis will be -0.05 for padding. However, if I also need to set the upper limit, I will set it to be max count + 1 for padding.
    
    for abx_name in antibiotic_names:
        count_of_abx_prescriptions_over_time[abx_name] = np.cumsum(count_of_abx_prescriptions_over_time[abx_name])
    
    # Actually, the upper y limit for all four of these subplots should be the maximum count across all four metrics, to allow for easier comparison.
    overall_max_count = max(
        max(not_infected_no_treatment_count_over_time),
        max(not_infected_treated_over_time),
        max(infected_no_treatment_count_over_time),
        max(infected_treated_overall_over_time),
    ) + 1  # Add some padding
    
    plt.figure(figsize=(18, 12))
    plt.subplot(2, 2, 1)
    plt.plot(not_infected_no_treatment_count_over_time, label='Not Infected No Treatment', color='blue')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    # Add grid:
    plt.grid(True)
    plt.title('Not Infected No Treatment Count Over Time')
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(not_infected_treated_over_time, label='Not Infected Treated', color='orange')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title('Not Infected Treated Count Over Time')
    # Add grid:
    plt.grid(True)
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(infected_no_treatment_count_over_time, label='Infected No Treatment', color='red')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    # Add grid:
    plt.grid(True)
    plt.title('Infected No Treatment Count Over Time')
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Here, plot 'infected_treated_overall_over_time' as a single line, then also plot each antibiotic's infected treated counts as separate lines. This should all be on the same plot.
    plt.plot(infected_treated_overall_over_time, label='Infected Treated Overall', color='purple', linewidth=2)
    for abx_name in antibiotic_names:
        plt.plot(infected_treated_per_antibiotic_over_time[abx_name], label=f'Infected Treated {abx_name}')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title('Infected Treated Counts Over Time')
    # Add grid:
    plt.grid(True)
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "outcome_counts_over_time.png"))
    plt.close()
    
    # Create a separate plot for count of abx prescriptions over time:
    plt.figure(figsize=(10, 5))
    for abx_name in antibiotic_names:
        plt.plot(count_of_abx_prescriptions_over_time[abx_name], label=f"{abx_name} Prescriptions")
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title('Antibiotic Prescriptions Over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, overall_max_count)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "abx_prescriptions_over_time.png"))
    plt.close()
    
    # Individual antibiotic figures:
    for abx_name in antibiotic_names:
        # Get the cumulative sums first
        infected_treated_count_over_time[abx_name]['sensitive_infection_treated'] = np.cumsum(infected_treated_count_over_time[abx_name]['sensitive_infection_treated'])
        infected_treated_count_over_time[abx_name]['resistant_infection_treated'] = np.cumsum(infected_treated_count_over_time[abx_name]['resistant_infection_treated'])
        
        plt.figure(figsize=(10, 5))
        plt.plot(infected_treated_count_over_time[abx_name]['sensitive_infection_treated'], label='Sensitive Infection Treated', color='green')
        plt.plot(infected_treated_count_over_time[abx_name]['resistant_infection_treated'], label='Resistant Infection Treated', color='red')
        plt.xlabel('Timestep')
        plt.ylabel('Count')
        plt.title(f'Infected Treated Counts Over Time for {abx_name}')
        plt.legend()
        # Add grid:
        plt.grid(True)
        plt.ylim(-0.05, max(max(infected_treated_count_over_time[abx_name]['sensitive_infection_treated']), max(infected_treated_count_over_time[abx_name]['resistant_infection_treated'])) + 1)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_figures_folder, f"infected_treated_counts_{abx_name}_over_time.png"))
        plt.close()
    
    # Create and save overall outcomes summary dictionary
    overall_outcomes_summary_dict = create_overall_outcomes_summary_dict(
        count_clinical_benefits_cumsum=count_clinical_benefits_over_time,
        count_clinical_failures_cumsum=count_clinical_failures_over_time,
        count_adverse_events_cumsum=count_adverse_events_over_time,
        not_infected_no_treatment_cumsum=not_infected_no_treatment_count_over_time,
        not_infected_treated_cumsum=not_infected_treated_over_time,
        infected_no_treatment_cumsum=infected_no_treatment_count_over_time,
        infected_treated_overall_cumsum=infected_treated_overall_over_time,
        infected_treated_sensitive_per_abx_cumsum={
            abx_name: infected_treated_count_over_time[abx_name]['sensitive_infection_treated'] 
            for abx_name in antibiotic_names
        },
        infected_treated_resistant_per_abx_cumsum={
            abx_name: infected_treated_count_over_time[abx_name]['resistant_infection_treated'] 
            for abx_name in antibiotic_names
        },
        count_prescriptions_per_abx_cumsum=count_of_abx_prescriptions_over_time,
        total_reward_cumsum=total_reward_over_time,
        antibiotic_names=antibiotic_names,
    )
    
    # Also compute the final AMR levels at the end of the episode, and add to the summary dict.
    final_actual_amr_levels = {abx_name: sum(actual_AMR_levels_over_time[abx_name][-10:])/len(actual_AMR_levels_over_time[abx_name][-10:]) for abx_name in antibiotic_names}
    
    overall_outcomes_summary_dict['final_actual_amr_levels'] = final_actual_amr_levels
    
    final_visible_amr_levels = {abx_name: sum(visible_AMR_levels_over_time[abx_name][-10:])/len(visible_AMR_levels_over_time[abx_name][-10:]) for abx_name in antibiotic_names}
    
    overall_outcomes_summary_dict['final_visible_amr_levels'] = final_visible_amr_levels
     
    # Save as json file:
    with open(os.path.join(experiment_figures_folder, "overall_outcomes_summary.json"), 'w') as f:
        json.dump(overall_outcomes_summary_dict, f, indent=4)
    
    return


def plot_metrics_from_trajectory(trajectory, env, experiment_folder, figures_folder_name: Optional[str] = "figures"):
    """Generate publication-quality plots from a direct dose sequence trajectory.
    
    Takes a pre-computed trajectory (e.g., from a specific dose sequence or manual strategy)
    and creates the same diagnostic plots as plot_metrics_trained_agent:
    - Leaky balloon AMR responses (for each antibiotic)
    - AMR levels over time (actual vs. observable)
    - Reward components (total, individual, community)
    - Clinical outcomes (benefits, failures, adverse events)
    - Patient outcome breakdowns
    - Antibiotic prescriptions over time
    
    This allows visualization of arbitrary dose sequences without requiring a trained agent.
    
    Args:
        trajectory (dict): Trajectory dictionary with keys:
            - 'obs': List of observations (numpy arrays)
            - 'actions': List of actions (numpy arrays)
            - 'rewards': List of scalar rewards
            - 'infos': List of info dicts from environment
        env (gym.Env): ABXAMREnv instance (used to access antibiotic names and AMR params).
        experiment_folder (str): Absolute path to experiment result directory
            (e.g., 'results/exp_1.a_..._seed1_20260115_143614').
        figures_folder_name (str, optional): Subfolder name for figures. If None,
            saves directly to experiment_folder. Default: 'figures'.
    
    Raises:
        ValueError: If experiment_folder doesn't exist or trajectory format invalid.
    
    Example:
        >>> trajectory = run_episode_and_get_trajectory(model, env, deterministic=True)
        >>> plot_metrics_from_trajectory(trajectory, env, 'results/my_run')
        >>> # Creates: results/my_run/figures/*.png
    """
    
    if not os.path.exists(experiment_folder):
        raise ValueError(f"Experiment folder {experiment_folder} does not exist.")
    
    # Validate trajectory format
    required_keys = ['obs', 'actions', 'rewards', 'infos']
    for key in required_keys:
        if key not in trajectory:
            raise ValueError(f"Trajectory missing required key: {key}")
    
    if figures_folder_name is None:
        # Save figures directly to experiment_folder
        experiment_figures_folder = experiment_folder
    else:
        # Create subfolder for figures
        experiment_figures_folder = os.path.join(experiment_folder, figures_folder_name)
        if not os.path.exists(experiment_figures_folder):
            os.makedirs(experiment_figures_folder)
    
    # First, create a plot of the AMR response for every antibiotic, for a steady dose sequence of [1, 1, 1, ..., 1].
    dose_sequence = [1] * 10 + [0] * 10 + [2] * 10
    
    for antibiotic_name, AMR_leakyballoon_params in env.unwrapped.antibiotics_AMR_dict.items():
        # Create a leaky balloon instance with the given parameters
        leaky_balloon = AMR_LeakyBalloon(**AMR_leakyballoon_params)
        leaky_balloon.plot_leaky_balloon_response_to_dose_sequence(
            dose_sequence,
            title=f"AMR Response for {antibiotic_name}",
            fname=f"leaky_balloon_response_{antibiotic_name}.png",
            save_plot_folder=experiment_figures_folder,
            show_plot=False,
        )
    
    # Extract trajectory components
    trajectory_infos = trajectory['infos']
    
    # For convenience, get the antibiotic names from the env:
    antibiotic_names = env.unwrapped.antibiotic_names
    
    # Iterate through the list of info dictionaries in order to extract relevant metrics for plotting.
    actual_AMR_levels_over_time = {abx_name: [] for abx_name in antibiotic_names}
    visible_AMR_levels_over_time = {abx_name: [] for abx_name in antibiotic_names}
    total_reward_over_time = []
    individual_reward_over_time = []
    normalized_individual_reward_over_time = []
    community_reward_over_time = []
    normalized_community_reward_over_time = []
    count_clinical_benefits_over_time = []
    count_clinical_failures_over_time = []
    count_adverse_events_over_time = []
    
    not_infected_no_treatment_count_over_time = []
    not_infected_treated_over_time = []
    infected_no_treatment_count_over_time = []
    infected_treated_count_over_time = {abx_name: {'sensitive_infection_treated': [], 'resistant_infection_treated': []} for abx_name in antibiotic_names}
    count_of_abx_prescriptions_over_time = {abx_name: [] for abx_name in antibiotic_names}
    
    # Skip the first info (from reset) which lacks step-level fields like 'actual_amr_levels'
    for info in trajectory_infos[1:]:
        
        for abx_name in antibiotic_names:
            actual_AMR_levels_over_time[abx_name].append(info['actual_amr_levels'][abx_name])
            visible_AMR_levels_over_time[abx_name].append(info['visible_amr_levels'][abx_name])
            
        total_reward_over_time.append(info['total_reward'])
        individual_reward_over_time.append(info['overall_individual_reward_component'])
        normalized_individual_reward_over_time.append(info['normalized_individual_reward'])
        community_reward_over_time.append(info['overall_community_reward_component'])
        normalized_community_reward_over_time.append(info['normalized_community_reward'])
        count_clinical_benefits_over_time.append(info['count_clinical_benefits'])
        count_clinical_failures_over_time.append(info['count_clinical_failures'])
        count_adverse_events_over_time.append(info['count_adverse_events'])
        
        # Also extract the outcomes_breakdown dictionary, and flatten out the information in it.
        outcomes_dict = info['outcomes_breakdown']
        
        # Extract counts for each outcome type:
        not_infected_no_treatment_count_over_time.append(outcomes_dict['not_infected_no_treatment'])
        not_infected_treated_over_time.append(outcomes_dict['not_infected_treated'])
        infected_no_treatment_count_over_time.append(outcomes_dict['infected_no_treatment'])
        for abx_name in antibiotic_names:
            infected_treated_count_over_time[abx_name]['sensitive_infection_treated'].append(outcomes_dict['infected_treated'][abx_name]['sensitive_infection_treated'])
            infected_treated_count_over_time[abx_name]['resistant_infection_treated'].append(outcomes_dict['infected_treated'][abx_name]['resistant_infection_treated'])
            
            count_of_abx_prescriptions_over_time[abx_name].append((outcomes_dict['infected_treated'][abx_name]['sensitive_infection_treated'] + outcomes_dict['infected_treated'][abx_name]['resistant_infection_treated']))
    
    # The fourth plot in outcome_counts_over_time.png should actually be 'Infected Treated' over time, with both the summed overall numbers of these patients over time, then also per antibiotic. This fourth plot does not care about whether or not the infection was sensitive or resistant, it only cares about the count of 'infected treated'. The images titled 'infected_treated_counts' will instead plot the sensitive vs resistant counts per antibiotic.
    # Therefore, let's aggregate the info in infected_treated_count_over_time in order to get the total infected treated counts over time.
    infected_treated_overall_over_time = []
    infected_treated_per_antibiotic_over_time = {abx_name: [] for abx_name in antibiotic_names}
    
    for abx_name in antibiotic_names:
        infected_treated_per_antibiotic_over_time[abx_name] = [
            s + r for s, r in zip(
                infected_treated_count_over_time[abx_name]['sensitive_infection_treated'],
                infected_treated_count_over_time[abx_name]['resistant_infection_treated'],
            )
        ]

    # I want to take the sum across all antibiotics at each timestep:
    for abx_name in antibiotic_names:
        if not infected_treated_overall_over_time:
            infected_treated_overall_over_time = infected_treated_per_antibiotic_over_time[abx_name]
        else:
            # Add the two lists element by element:
            infected_treated_overall_over_time = [x + y for x, y in zip(infected_treated_overall_over_time, infected_treated_per_antibiotic_over_time[abx_name])]
    
    # Once all the data is extracted, plot each of these into separate figures; similar datastreams will get plotted in the same figure.
    
    # First, plot actual AMR levels and visible AMR levels in two side by side plots; each plot will have all antibiotic AMR levels over time plotted.
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for abx_name in antibiotic_names:
        plt.plot(actual_AMR_levels_over_time[abx_name], label=f"{abx_name} Actual AMR Level")
    plt.xlabel('Timestep')
    plt.ylabel('Actual AMR Level')
    plt.title('Actual AMR Levels Over Time')
    # Add grid:
    plt.grid(True)
    # Set ylim to be -0.05 to 1.05:
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for abx_name in antibiotic_names:
        plt.plot(visible_AMR_levels_over_time[abx_name], label=f"{abx_name} Visible AMR Level")
    plt.xlabel('Timestep')
    plt.ylabel('Visible AMR Level')
    plt.title('Visible AMR Levels Over Time')
    # Add grid:
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "amr_levels_over_time.png"))
    plt.close()
    
    # Next, plot the individual and community reward components over time in one plot, the normalized_individual_reward and normalized community_reward in another plot (next to the first one), and then the total reward in a third plot.
    # What I'm actually interested in is the cumulative reward over time, so I'll create cumulative sums of each of these reward components.
    individual_reward_over_time = np.cumsum(individual_reward_over_time)
    normalized_individual_reward_over_time = np.cumsum(normalized_individual_reward_over_time)
    community_reward_over_time = np.cumsum(community_reward_over_time)
    normalized_community_reward_over_time = np.cumsum(normalized_community_reward_over_time)
    total_reward_over_time = np.cumsum(total_reward_over_time)
    
    # All three of these subplots should have the same y-axis limits for easier comparison.
    ylim_min = min(
        min(individual_reward_over_time),
        min(normalized_individual_reward_over_time),
        min(community_reward_over_time),
        min(normalized_community_reward_over_time),
        min(total_reward_over_time),
    ) - 0.5  # Add some padding
    ylim_max = max(
        max(individual_reward_over_time),
        max(normalized_individual_reward_over_time),
        max(community_reward_over_time),
        max(normalized_community_reward_over_time),
        max(total_reward_over_time),
    ) + 0.5  # Add some padding
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(individual_reward_over_time, label='Individual Reward Component')
    plt.plot(community_reward_over_time, label='Community Reward Component')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Individual and Community Reward Components Over Time')
    plt.ylim(ylim_min, ylim_max)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(normalized_individual_reward_over_time, label='Normalized Individual Reward')
    plt.plot(normalized_community_reward_over_time, label='Normalized Community Reward')
    plt.xlabel('Timestep')
    plt.ylabel('Normalized Reward')
    plt.title('Normalized Individual and Community Rewards Over Time')
    plt.ylim(ylim_min, ylim_max)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(total_reward_over_time, label='Total Reward', color='purple')
    plt.xlabel('Timestep')
    plt.ylabel('Total Reward')
    plt.title('Total Reward Over Time')
    plt.grid(True)
    plt.legend()
    plt.ylim(ylim_min, ylim_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "reward_components_over_time.png"))
    plt.close()
    
    # Next, plot the counts of clinical benefits and adverse events over time in the same plot.
    # Again, what I'm actually interested in is the cumulative counts over time. Also, for all of the outcome counts, the lower y limit is -0.05 to give some padding.
    count_clinical_benefits_over_time = np.cumsum(count_clinical_benefits_over_time)
    count_clinical_failures_over_time = np.cumsum(count_clinical_failures_over_time)
    count_adverse_events_over_time = np.cumsum(count_adverse_events_over_time)
    
    plt.figure(figsize=(10, 5))
    plt.plot(count_clinical_benefits_over_time, label='Count Clinical Benefits', color='green')
    plt.plot(count_clinical_failures_over_time, label='Count Clinical Failures', color='red')
    plt.plot(count_adverse_events_over_time, label='Count Adverse Events', color='purple')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title('Counts of Clinical Benefits, Clinical Failures, and Adverse Events Over Time')
    plt.legend()
    # Add grid:
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-0.05, max(max(count_clinical_benefits_over_time), max(count_clinical_failures_over_time), max(count_adverse_events_over_time)) + 1)
    plt.savefig(os.path.join(experiment_figures_folder, "clinical_benefits_failures_adverse_events_over_time.png"))
    plt.close()
    
    # # Finally, plot the various outcome counts over time in a single figure with multiple subplots. The first plot with include 4 subfigures: not infected no treatment count over time, not infected treated over time, infected no treatment count over time, and count of abx prescriptions over time (each as a separate line per antibiotic, in the same subplot).
    # 
    # Then, each individual antibiotic will get their own figure, which will plot the sensitive and resistant infection treated counts over time.
    
    # First figure with 4 subplots:
    # Again, what I'm interested in is the cumulative counts over time.
    not_infected_no_treatment_count_over_time = np.cumsum(not_infected_no_treatment_count_over_time)
    not_infected_treated_over_time = np.cumsum(not_infected_treated_over_time)
    infected_no_treatment_count_over_time = np.cumsum(infected_no_treatment_count_over_time)
    infected_treated_overall_over_time = np.cumsum(infected_treated_overall_over_time)
    # Do this also per antibiotic:
    for abx_name in antibiotic_names:
        infected_treated_per_antibiotic_over_time[abx_name] = np.cumsum(infected_treated_per_antibiotic_over_time[abx_name])
    
    # The lower limit of y axis will be -0.05 for padding. However, if I also need to set the upper limit, I will set it to be max count + 1 for padding.
    
    for abx_name in antibiotic_names:
        count_of_abx_prescriptions_over_time[abx_name] = np.cumsum(count_of_abx_prescriptions_over_time[abx_name])
    
    # Actually, the upper y limit for all four of these subplots should be the maximum count across all four metrics, to allow for easier comparison.
    overall_max_count = max(
        max(not_infected_no_treatment_count_over_time),
        max(not_infected_treated_over_time),
        max(infected_no_treatment_count_over_time),
        max(infected_treated_overall_over_time),
    ) + 1  # Add some padding
    
    plt.figure(figsize=(18, 12))
    plt.subplot(2, 2, 1)
    plt.plot(not_infected_no_treatment_count_over_time, label='Not Infected No Treatment', color='blue')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    # Add grid:
    plt.grid(True)
    plt.title('Not Infected No Treatment Count Over Time')
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(not_infected_treated_over_time, label='Not Infected Treated', color='orange')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title('Not Infected Treated Count Over Time')
    # Add grid:
    plt.grid(True)
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(infected_no_treatment_count_over_time, label='Infected No Treatment', color='red')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    # Add grid:
    plt.grid(True)
    plt.title('Infected No Treatment Count Over Time')
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Here, plot 'infected_treated_overall_over_time' as a single line, then also plot each antibiotic's infected treated counts as separate lines. This should all be on the same plot.
    plt.plot(infected_treated_overall_over_time, label='Infected Treated Overall', color='purple', linewidth=2)
    for abx_name in antibiotic_names:
        plt.plot(infected_treated_per_antibiotic_over_time[abx_name], label=f'Infected Treated {abx_name}')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title('Infected Treated Counts Over Time')
    # Add grid:
    plt.grid(True)
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "outcome_counts_over_time.png"))
    plt.close()
    
    # Create a separate plot for count of abx prescriptions over time:
    plt.figure(figsize=(10, 5))
    for abx_name in antibiotic_names:
        plt.plot(count_of_abx_prescriptions_over_time[abx_name], label=f"{abx_name} Prescriptions")
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title('Antibiotic Prescriptions Over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, overall_max_count)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "abx_prescriptions_over_time.png"))
    plt.close()
    
    # Individual antibiotic figures:
    for abx_name in antibiotic_names:
        # Get the cumulative sums first
        infected_treated_count_over_time[abx_name]['sensitive_infection_treated'] = np.cumsum(infected_treated_count_over_time[abx_name]['sensitive_infection_treated'])
        infected_treated_count_over_time[abx_name]['resistant_infection_treated'] = np.cumsum(infected_treated_count_over_time[abx_name]['resistant_infection_treated'])
        
        plt.figure(figsize=(10, 5))
        plt.plot(infected_treated_count_over_time[abx_name]['sensitive_infection_treated'], label='Sensitive Infection Treated', color='green')
        plt.plot(infected_treated_count_over_time[abx_name]['resistant_infection_treated'], label='Resistant Infection Treated', color='red')
        plt.xlabel('Timestep')
        plt.ylabel('Count')
        plt.title(f'Infected Treated Counts Over Time for {abx_name}')
        plt.legend()
        # Add grid:
        plt.grid(True)
        plt.ylim(-0.05, max(max(infected_treated_count_over_time[abx_name]['sensitive_infection_treated']), max(infected_treated_count_over_time[abx_name]['resistant_infection_treated'])) + 1)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_figures_folder, f"infected_treated_counts_{abx_name}_over_time.png"))
        plt.close()
    
    # Create and save overall outcomes summary dictionary
    overall_outcomes_summary_dict = create_overall_outcomes_summary_dict(
        count_clinical_benefits_cumsum=count_clinical_benefits_over_time,
        count_clinical_failures_cumsum=count_clinical_failures_over_time,
        count_adverse_events_cumsum=count_adverse_events_over_time,
        not_infected_no_treatment_cumsum=not_infected_no_treatment_count_over_time,
        not_infected_treated_cumsum=not_infected_treated_over_time,
        infected_no_treatment_cumsum=infected_no_treatment_count_over_time,
        infected_treated_overall_cumsum=infected_treated_overall_over_time,
        infected_treated_sensitive_per_abx_cumsum={
            abx_name: infected_treated_count_over_time[abx_name]['sensitive_infection_treated'] 
            for abx_name in antibiotic_names
        },
        infected_treated_resistant_per_abx_cumsum={
            abx_name: infected_treated_count_over_time[abx_name]['resistant_infection_treated'] 
            for abx_name in antibiotic_names
        },
        count_prescriptions_per_abx_cumsum=count_of_abx_prescriptions_over_time,
        antibiotic_names=antibiotic_names,
        total_reward_cumsum=total_reward_over_time,
    )
    
    final_actual_amr_levels = {}
    
    for antibiotic_name in antibiotic_names:
        final_actual_amr_levels[antibiotic_name] = sum(actual_AMR_levels_over_time[antibiotic_name][-10:])/len(actual_AMR_levels_over_time[antibiotic_name][-10:])
    
    overall_outcomes_summary_dict['final_actual_amr_levels'] = final_actual_amr_levels
    
    final_visible_amr_levels = {}
    
    for antibiotic_name in antibiotic_names:
        final_visible_amr_levels[antibiotic_name] = sum(visible_AMR_levels_over_time[antibiotic_name][-10:])/len(visible_AMR_levels_over_time[antibiotic_name][-10:])
    
    overall_outcomes_summary_dict['final_visible_amr_levels'] = final_visible_amr_levels
    
    # Save as json file:
    with open(os.path.join(experiment_figures_folder, "overall_outcomes_summary.json"), 'w') as f:
        json.dump(overall_outcomes_summary_dict, f, indent=4)
    
    return


def plot_metrics_ensemble_agents(
    models,
    env,
    experiment_folder,
    n_episodes_per_agent=10,
    deterministic=True,
    figures_folder_name="ensemble_figures",
    per_seed_figures=True,
    episode_seed_start=0,
):
    """Plot aggregated metrics from multiple trained agents across multiple episodes.
    
    Runs each agent (representing different training seeds) through multiple episodes,
    aggregates results, and generates plots with median curves and 10-90 percentile bands.
    Enables robust evaluation of agent performance under environment stochasticity
    (heterogeneous patient generation).
    
    Creates ensemble figures (aggregated across all agents/episodes) and optionally
    per-seed figures (showing individual agent performance).
    
    Args:
        models (list): List of trained stable-baselines3 agents (different training seeds).
        env (gym.Env): ABXAMREnv instance to run episodes in. Will be reset between
            episodes with different seeds for environment stochasticity.
        experiment_folder (str): Absolute path to experiment result directory. Figures
            saved to <experiment_folder>/<figures_folder_name>/.
        n_episodes_per_agent (int): Number of episodes to run per agent. Default: 10.
            Total episodes = len(models) * n_episodes_per_agent.
        deterministic (bool): If True, use deterministic policy. Default: True.
            Even with deterministic=True, episode outcomes vary due to environment
            stochasticity (patient generation randomness).
        figures_folder_name (str): Subfolder name for ensemble figures. Default:
            'ensemble_figures'.
        per_seed_figures (bool): If True, create per-agent visualization folders
            showing individual agent performance. Default: True.
        episode_seed_start (int): Starting seed for environment resets (incremented
            for each episode). Default: 0.
    
    Raises:
        ValueError: If experiment_folder doesn't exist.
    
    Example:
        >>> models = [PPO.load(f'results/run_seed{i}/best_model.zip') for i in range(1, 6)]
        >>> env = create_environment(config, rc, pg)
        >>> plot_metrics_ensemble_agents(models, env, 'results/ensemble_analysis')
        >>> # Creates: results/ensemble_analysis/ensemble_figures/*.png with median ± percentiles
    """
    
    if not os.path.exists(experiment_folder):
        raise ValueError(f"Experiment folder {experiment_folder} does not exist.")
    
    # Create main figures folder
    experiment_figures_folder = os.path.join(experiment_folder, figures_folder_name)
    if not os.path.exists(experiment_figures_folder):
        os.makedirs(experiment_figures_folder)
    
    # Get antibiotic names from environment
    antibiotic_names = env.unwrapped.antibiotic_names
    
    # Plot leaky balloon responses once (deterministic, independent of agent)
    dose_sequence = [1] * 10 + [0] * 10 + [2] * 10
    for antibiotic_name, AMR_leakyballoon_params in env.unwrapped.antibiotics_AMR_dict.items():
        leaky_balloon = AMR_LeakyBalloon(**AMR_leakyballoon_params)
        leaky_balloon.plot_leaky_balloon_response_to_dose_sequence(
            dose_sequence,
            title=f"AMR Response for {antibiotic_name}",
            fname=f"leaky_balloon_response_{antibiotic_name}.png",
            save_plot_folder=experiment_figures_folder,
            show_plot=False,
        )
    
    # ====== PHASE 1: Data Collection ======
    print(f"Collecting data from {len(models)} agents × {n_episodes_per_agent} episodes = {len(models) * n_episodes_per_agent} total trajectories...")
    
    # Storage for all trajectories (will be lists of lists)
    all_trajectories_data = {
        'actual_AMR_levels': {abx_name: [] for abx_name in antibiotic_names},
        'visible_AMR_levels': {abx_name: [] for abx_name in antibiotic_names},
        'total_reward': [],
        'individual_reward': [],
        'normalized_individual_reward': [],
        'community_reward': [],
        'normalized_community_reward': [],
        'count_clinical_benefits': [],
        'count_clinical_failures': [],
        'count_adverse_events': [],
        'not_infected_no_treatment': [],
        'not_infected_treated': [],
        'infected_no_treatment': [],
        'infected_treated_sensitive': {abx_name: [] for abx_name in antibiotic_names},
        'infected_treated_resistant': {abx_name: [] for abx_name in antibiotic_names},
        'count_prescriptions': {abx_name: [] for abx_name in antibiotic_names},
    }
    
    # Also track per-seed data for per-seed visualizations
    per_seed_data = []
    
    episode_seed = episode_seed_start
    
    for model_idx, model in enumerate(models):
        print(f"  Processing model {model_idx + 1}/{len(models)}...")
        
        # Storage for this seed's trajectories
        seed_trajectories_data = {
            'actual_AMR_levels': {abx_name: [] for abx_name in antibiotic_names},
            'visible_AMR_levels': {abx_name: [] for abx_name in antibiotic_names},
            'total_reward': [],
            'individual_reward': [],
            'normalized_individual_reward': [],
            'community_reward': [],
            'normalized_community_reward': [],
            'count_clinical_benefits': [],
            'count_clinical_failures': [],
            'count_adverse_events': [],
            'not_infected_no_treatment': [],
            'not_infected_treated': [],
            'infected_no_treatment': [],
            'infected_treated_sensitive': {abx_name: [] for abx_name in antibiotic_names},
            'infected_treated_resistant': {abx_name: [] for abx_name in antibiotic_names},
            'count_prescriptions': {abx_name: [] for abx_name in antibiotic_names},
        }
        
        for episode_idx in range(n_episodes_per_agent):
            # Reset environment with unique seed for stochasticity
            env.reset(seed=episode_seed)
            episode_seed += 1
            
            # Run episode
            trajectory_dict = run_episode_and_get_trajectory(model, env, deterministic=deterministic)
            trajectory_infos = trajectory_dict['infos']
            
            # Extract metrics from this trajectory (skip first info from reset)
            traj_data = {
                'actual_AMR_levels': {abx_name: [] for abx_name in antibiotic_names},
                'visible_AMR_levels': {abx_name: [] for abx_name in antibiotic_names},
                'total_reward': [],
                'individual_reward': [],
                'normalized_individual_reward': [],
                'community_reward': [],
                'normalized_community_reward': [],
                'count_clinical_benefits': [],
                'count_clinical_failures': [],
                'count_adverse_events': [],
                'not_infected_no_treatment': [],
                'not_infected_treated': [],
                'infected_no_treatment': [],
                'infected_treated_sensitive': {abx_name: [] for abx_name in antibiotic_names},
                'infected_treated_resistant': {abx_name: [] for abx_name in antibiotic_names},
                'count_prescriptions': {abx_name: [] for abx_name in antibiotic_names},
            }
            
            for info in trajectory_infos[1:]:
                for abx_name in antibiotic_names:
                    traj_data['actual_AMR_levels'][abx_name].append(info['actual_amr_levels'][abx_name])
                    traj_data['visible_AMR_levels'][abx_name].append(info['visible_amr_levels'][abx_name])
                
                traj_data['total_reward'].append(info['total_reward'])
                traj_data['individual_reward'].append(info['overall_individual_reward_component'])
                traj_data['normalized_individual_reward'].append(info['normalized_individual_reward'])
                traj_data['community_reward'].append(info['overall_community_reward_component'])
                traj_data['normalized_community_reward'].append(info['normalized_community_reward'])
                traj_data['count_clinical_benefits'].append(info['count_clinical_benefits'])
                traj_data['count_clinical_failures'].append(info['count_clinical_failures'])
                traj_data['count_adverse_events'].append(info['count_adverse_events'])
                
                outcomes_dict = info['outcomes_breakdown']
                traj_data['not_infected_no_treatment'].append(outcomes_dict['not_infected_no_treatment'])
                traj_data['not_infected_treated'].append(outcomes_dict['not_infected_treated'])
                traj_data['infected_no_treatment'].append(outcomes_dict['infected_no_treatment'])
                
                for abx_name in antibiotic_names:
                    traj_data['infected_treated_sensitive'][abx_name].append(
                        outcomes_dict['infected_treated'][abx_name]['sensitive_infection_treated']
                    )
                    traj_data['infected_treated_resistant'][abx_name].append(
                        outcomes_dict['infected_treated'][abx_name]['resistant_infection_treated']
                    )
                    traj_data['count_prescriptions'][abx_name].append(
                        outcomes_dict['infected_treated'][abx_name]['sensitive_infection_treated'] +
                        outcomes_dict['infected_treated'][abx_name]['resistant_infection_treated']
                    )
            
            # Add this trajectory to global and seed-specific storage
            for key in ['total_reward', 'individual_reward', 'normalized_individual_reward', 
                       'community_reward', 'normalized_community_reward', 'count_clinical_benefits',
                       'count_clinical_failures', 'count_adverse_events', 'not_infected_no_treatment',
                       'not_infected_treated', 'infected_no_treatment']:
                all_trajectories_data[key].append(traj_data[key])
                seed_trajectories_data[key].append(traj_data[key])
            
            for abx_name in antibiotic_names:
                for key in ['actual_AMR_levels', 'visible_AMR_levels', 'infected_treated_sensitive',
                           'infected_treated_resistant', 'count_prescriptions']:
                    all_trajectories_data[key][abx_name].append(traj_data[key][abx_name])
                    seed_trajectories_data[key][abx_name].append(traj_data[key][abx_name])
        
        # Store this seed's data
        per_seed_data.append(seed_trajectories_data)
    
    print("Data collection complete!")
    
    # ====== PHASE 2: Aggregation ======
    print("Aggregating trajectories...")
    
    def aggregate_trajectories(trajectories_list, apply_cumsum=False):
        """
        Aggregate list of trajectories into mean, percentiles, and IQM.
        
        Parameters:
        -----------
        trajectories_list : list of lists
            Each element is a trajectory (list of values over timesteps)
        apply_cumsum : bool
            Whether to apply cumulative sum before aggregation
        
        Returns:
        --------
        dict with keys: mean, median, p10, p90, p25, p75, iqm, timesteps
        """
        # Convert to numpy array (trajectories × timesteps)
        arr = np.array(trajectories_list)
        
        if apply_cumsum:
            arr = np.cumsum(arr, axis=1)
        
        # Compute statistics across trajectories (axis=0)
        mean = np.mean(arr, axis=0)
        median = np.percentile(arr, 50, axis=0)  # 50th percentile
        p10 = np.percentile(arr, 10, axis=0)
        p90 = np.percentile(arr, 90, axis=0)
        
        # Interquartile mean (IQM): mean of values between 25th and 75th percentile
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        # For each timestep, filter values within IQR and compute mean
        iqm = np.zeros(arr.shape[1])
        for t in range(arr.shape[1]):
            values_t = arr[:, t]
            mask = (values_t >= p25[t]) & (values_t <= p75[t])
            iqm[t] = np.mean(values_t[mask]) if mask.sum() > 0 else mean[t]
        
        timesteps = np.arange(len(median))
        
        return {
            'mean': mean,
            'median': median,
            'p10': p10,
            'p90': p90,
            'p25': p25,
            'p75': p75,
            'iqm': iqm,
            'timesteps': timesteps,
        }
    
    # Aggregate all metrics
    aggregated = {}
    
    # AMR levels (no cumsum)
    aggregated['actual_AMR_levels'] = {}
    aggregated['visible_AMR_levels'] = {}
    for abx_name in antibiotic_names:
        aggregated['actual_AMR_levels'][abx_name] = aggregate_trajectories(
            all_trajectories_data['actual_AMR_levels'][abx_name], apply_cumsum=False
        )
        aggregated['visible_AMR_levels'][abx_name] = aggregate_trajectories(
            all_trajectories_data['visible_AMR_levels'][abx_name], apply_cumsum=False
        )
    
    # Rewards (apply cumsum)
    for key in ['total_reward', 'individual_reward', 'normalized_individual_reward',
                'community_reward', 'normalized_community_reward']:
        aggregated[key] = aggregate_trajectories(all_trajectories_data[key], apply_cumsum=True)
    
    # Outcome counts (apply cumsum)
    for key in ['count_clinical_benefits', 'count_clinical_failures', 'count_adverse_events',
                'not_infected_no_treatment', 'not_infected_treated', 'infected_no_treatment']:
        aggregated[key] = aggregate_trajectories(all_trajectories_data[key], apply_cumsum=True)
    
    # Per-antibiotic counts (apply cumsum)
    aggregated['infected_treated_sensitive'] = {}
    aggregated['infected_treated_resistant'] = {}
    aggregated['count_prescriptions'] = {}
    for abx_name in antibiotic_names:
        aggregated['infected_treated_sensitive'][abx_name] = aggregate_trajectories(
            all_trajectories_data['infected_treated_sensitive'][abx_name], apply_cumsum=True
        )
        aggregated['infected_treated_resistant'][abx_name] = aggregate_trajectories(
            all_trajectories_data['infected_treated_resistant'][abx_name], apply_cumsum=True
        )
        aggregated['count_prescriptions'][abx_name] = aggregate_trajectories(
            all_trajectories_data['count_prescriptions'][abx_name], apply_cumsum=True
        )
    
    # Compute infected treated overall (sum across antibiotics)
    infected_treated_overall_trajs = []
    for traj_idx in range(len(all_trajectories_data['infected_treated_sensitive'][antibiotic_names[0]])):
        traj_sum = np.zeros(len(all_trajectories_data['infected_treated_sensitive'][antibiotic_names[0]][traj_idx]))
        for abx_name in antibiotic_names:
            traj_sum += np.array(all_trajectories_data['infected_treated_sensitive'][abx_name][traj_idx])
            traj_sum += np.array(all_trajectories_data['infected_treated_resistant'][abx_name][traj_idx])
        infected_treated_overall_trajs.append(traj_sum.tolist())
    aggregated['infected_treated_overall'] = aggregate_trajectories(infected_treated_overall_trajs, apply_cumsum=True)
    
    print("Aggregation complete!")
    
    # ====== PHASE 3: Plotting ======
    print("Generating ensemble plots...")
    
    def plot_with_bands(ax, data_dict, label, color, linestyle='-', linewidth=1.5):
        """Helper function to plot median with percentile bands (p10-p90) and IQR bounds (p25-p75).
        
        Uses median (50th percentile) instead of mean because:
        - Median is guaranteed to be between p25-p75 by definition
        - More robust to outlier trajectories
        - More intuitive: "50% of seeds achieved at least this value by this timestep"
        """
        timesteps = data_dict['timesteps']
        median = data_dict['median']
        p10 = data_dict['p10']
        p90 = data_dict['p90']
        p25 = data_dict['p25']
        p75 = data_dict['p75']
        
        # Plot median line and capture its color (important when color=None for auto-coloring)
        line = ax.plot(timesteps, median, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
        line_color = line[0].get_color()
        
        # Use the same color for shading and IQR bounds
        ax.fill_between(timesteps, p10, p90, color=line_color, alpha=0.2)
        
        # Plot p25 and p75 with different line styles for better distinction
        # Use dotted for one and dashed for the other
        ax.plot(timesteps, p25, color=line_color, linestyle=':', linewidth=1.5, alpha=0.7, label=f'{label} (IQR: p25-p75)')
        ax.plot(timesteps, p75, color=line_color, linestyle='--', linewidth=1.0, alpha=0.7)
    
    # Figure 1: AMR levels over time
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for abx_name in antibiotic_names:
        plot_with_bands(plt.gca(), aggregated['actual_AMR_levels'][abx_name], 
                       f"{abx_name} Actual AMR", color=None)
    plt.xlabel('Timestep')
    plt.ylabel('Actual AMR Level')
    plt.title('Actual AMR Levels Over Time (Median ± 10-90%)')
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for abx_name in antibiotic_names:
        plot_with_bands(plt.gca(), aggregated['visible_AMR_levels'][abx_name],
                       f"{abx_name} Visible AMR", color=None)
    plt.xlabel('Timestep')
    plt.ylabel('Visible AMR Level')
    plt.title('Visible AMR Levels Over Time (Median ± 10-90%)')
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "amr_levels_over_time.png"))
    plt.close()
    
    # Figure 2: Reward components
    ylim_min = min(
        aggregated['individual_reward']['p10'].min(),
        aggregated['normalized_individual_reward']['p10'].min(),
        aggregated['community_reward']['p10'].min(),
        aggregated['normalized_community_reward']['p10'].min(),
        aggregated['total_reward']['p10'].min(),
    ) - 0.5
    ylim_max = max(
        aggregated['individual_reward']['p90'].max(),
        aggregated['normalized_individual_reward']['p90'].max(),
        aggregated['community_reward']['p90'].max(),
        aggregated['normalized_community_reward']['p90'].max(),
        aggregated['total_reward']['p90'].max(),
    ) + 0.5
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plot_with_bands(plt.gca(), aggregated['individual_reward'], 'Individual Reward', 'blue')
    plot_with_bands(plt.gca(), aggregated['community_reward'], 'Community Reward', 'orange')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Reward')
    plt.title('Individual and Community Reward Components (Median ± 10-90%)')
    plt.ylim(ylim_min, ylim_max)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plot_with_bands(plt.gca(), aggregated['normalized_individual_reward'], 'Normalized Individual', 'blue')
    plot_with_bands(plt.gca(), aggregated['normalized_community_reward'], 'Normalized Community', 'orange')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Normalized Reward')
    plt.title('Normalized Rewards (Median ± 10-90%)')
    plt.ylim(ylim_min, ylim_max)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plot_with_bands(plt.gca(), aggregated['total_reward'], 'Total Reward', 'purple')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Total Reward')
    plt.title('Total Reward (Median ± 10-90%)')
    plt.grid(True)
    plt.legend()
    plt.ylim(ylim_min, ylim_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "reward_components_over_time.png"))
    plt.close()
    
    # Figure 3: Clinical outcomes
    plt.figure(figsize=(10, 5))
    plot_with_bands(plt.gca(), aggregated['count_clinical_benefits'], 'Clinical Benefits', 'green')
    plot_with_bands(plt.gca(), aggregated['count_clinical_failures'], 'Clinical Failures', 'red')
    plot_with_bands(plt.gca(), aggregated['count_adverse_events'], 'Adverse Events', 'purple')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Count')
    plt.title('Clinical Outcomes Over Time (Median ± 10-90%)')
    plt.legend()
    plt.grid(True)
    overall_max = max(
        aggregated['count_clinical_benefits']['p90'].max(),
        aggregated['count_clinical_failures']['p90'].max(),
        aggregated['count_adverse_events']['p90'].max(),
    )
    plt.ylim(-0.05, overall_max + 1)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "clinical_benefits_failures_adverse_events_over_time.png"))
    plt.close()
    
    # Figure 4: Outcome counts (4 subplots)
    overall_max_count = max(
        aggregated['not_infected_no_treatment']['p90'].max(),
        aggregated['not_infected_treated']['p90'].max(),
        aggregated['infected_no_treatment']['p90'].max(),
        aggregated['infected_treated_overall']['p90'].max(),
    ) + 1
    
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 2, 1)
    plot_with_bands(plt.gca(), aggregated['not_infected_no_treatment'], 'Not Infected No Treatment', 'blue')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Count')
    plt.grid(True)
    plt.title('Not Infected No Treatment (Mean ± 10-90%)')
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plot_with_bands(plt.gca(), aggregated['not_infected_treated'], 'Not Infected Treated', 'orange')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Count')
    plt.title('Not Infected Treated (Mean ± 10-90%)')
    plt.grid(True)
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plot_with_bands(plt.gca(), aggregated['infected_no_treatment'], 'Infected No Treatment', 'red')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Count')
    plt.grid(True)
    plt.title('Infected No Treatment (Mean ± 10-90%)')
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plot_with_bands(plt.gca(), aggregated['infected_treated_overall'], 'Infected Treated Overall', 'purple', linewidth=2)
    for abx_name in antibiotic_names:
        # Compute per-antibiotic infected treated (sensitive + resistant)
        per_abx_trajs = []
        for traj_idx in range(len(all_trajectories_data['infected_treated_sensitive'][abx_name])):
            traj = (np.array(all_trajectories_data['infected_treated_sensitive'][abx_name][traj_idx]) +
                   np.array(all_trajectories_data['infected_treated_resistant'][abx_name][traj_idx]))
            per_abx_trajs.append(traj.tolist())
        per_abx_agg = aggregate_trajectories(per_abx_trajs, apply_cumsum=True)
        plot_with_bands(plt.gca(), per_abx_agg, f'Infected Treated {abx_name}', color=None)
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Count')
    plt.title('Infected Treated Counts (Mean ± 10-90%)')
    plt.grid(True)
    plt.ylim(-0.05, overall_max_count)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "outcome_counts_over_time.png"))
    plt.close()
    
    # Figure 5: Antibiotic prescriptions
    plt.figure(figsize=(10, 5))
    for abx_name in antibiotic_names:
        plot_with_bands(plt.gca(), aggregated['count_prescriptions'][abx_name], 
                       f"{abx_name} Prescriptions", color=None)
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Count')
    plt.title('Antibiotic Prescriptions Over Time (Mean ± 10-90%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, overall_max_count)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_figures_folder, "abx_prescriptions_over_time.png"))
    plt.close()
    
    # Figure 6: Per-antibiotic infected treated (sensitive vs resistant)
    for abx_name in antibiotic_names:
        plt.figure(figsize=(10, 5))
        plot_with_bands(plt.gca(), aggregated['infected_treated_sensitive'][abx_name],
                       'Sensitive Infection Treated', 'green')
        plot_with_bands(plt.gca(), aggregated['infected_treated_resistant'][abx_name],
                       'Resistant Infection Treated', 'red')
        plt.xlabel('Timestep')
        plt.ylabel('Cumulative Count')
        plt.title(f'Infected Treated Counts for {abx_name} (Mean ± 10-90%)')
        plt.legend()
        plt.grid(True)
        overall_max_abx = max(
            aggregated['infected_treated_sensitive'][abx_name]['p90'].max(),
            aggregated['infected_treated_resistant'][abx_name]['p90'].max(),
        )
        plt.ylim(-0.05, overall_max_abx + 1)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_figures_folder, f"infected_treated_counts_{abx_name}_over_time.png"))
        plt.close()
    
    print(f"Ensemble plots saved to {experiment_figures_folder}")
    
    # ====== PHASE 4: Per-Seed Visualizations (Optional) ======
    if per_seed_figures and len(per_seed_data) > 1:
        print("Generating per-seed visualizations...")
        
        for seed_idx, seed_data in enumerate(per_seed_data):
            seed_folder = os.path.join(experiment_figures_folder, f"seed_{seed_idx}")
            if not os.path.exists(seed_folder):
                os.makedirs(seed_folder)
            
            # Aggregate this seed's trajectories
            seed_aggregated = {}
            
            # AMR levels
            seed_aggregated['actual_AMR_levels'] = {}
            seed_aggregated['visible_AMR_levels'] = {}
            for abx_name in antibiotic_names:
                seed_aggregated['actual_AMR_levels'][abx_name] = aggregate_trajectories(
                    seed_data['actual_AMR_levels'][abx_name], apply_cumsum=False
                )
                seed_aggregated['visible_AMR_levels'][abx_name] = aggregate_trajectories(
                    seed_data['visible_AMR_levels'][abx_name], apply_cumsum=False
                )
            
            # Rewards
            for key in ['total_reward', 'individual_reward', 'normalized_individual_reward',
                        'community_reward', 'normalized_community_reward']:
                seed_aggregated[key] = aggregate_trajectories(seed_data[key], apply_cumsum=True)
            
            # Outcome counts
            for key in ['count_clinical_benefits', 'count_clinical_failures', 'count_adverse_events',
                        'not_infected_no_treatment', 'not_infected_treated', 'infected_no_treatment']:
                seed_aggregated[key] = aggregate_trajectories(seed_data[key], apply_cumsum=True)
            
            # Per-antibiotic
            seed_aggregated['infected_treated_sensitive'] = {}
            seed_aggregated['infected_treated_resistant'] = {}
            seed_aggregated['count_prescriptions'] = {}
            for abx_name in antibiotic_names:
                seed_aggregated['infected_treated_sensitive'][abx_name] = aggregate_trajectories(
                    seed_data['infected_treated_sensitive'][abx_name], apply_cumsum=True
                )
                seed_aggregated['infected_treated_resistant'][abx_name] = aggregate_trajectories(
                    seed_data['infected_treated_resistant'][abx_name], apply_cumsum=True
                )
                seed_aggregated['count_prescriptions'][abx_name] = aggregate_trajectories(
                    seed_data['count_prescriptions'][abx_name], apply_cumsum=True
                )
            
            # Quick plot: just AMR and total reward for each seed
            plt.figure(figsize=(14, 6))
            
            plt.subplot(1, 2, 1)
            for abx_name in antibiotic_names:
                plot_with_bands(plt.gca(), seed_aggregated['actual_AMR_levels'][abx_name],
                               f"{abx_name}", color=None)
            plt.xlabel('Timestep')
            plt.ylabel('Actual AMR Level')
            plt.title(f'Seed {seed_idx}: Actual AMR (Mean ± 10-90%)')
            plt.grid(True)
            plt.ylim(-0.05, 1.05)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plot_with_bands(plt.gca(), seed_aggregated['total_reward'], 'Total Reward', 'purple')
            plt.xlabel('Timestep')
            plt.ylabel('Cumulative Total Reward')
            plt.title(f'Seed {seed_idx}: Total Reward (Mean ± 10-90%)')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(seed_folder, f"seed_{seed_idx}_summary.png"))
            plt.close()
        
        print(f"Per-seed plots saved to {experiment_figures_folder}/seed_*")
    
    # ====== PHASE 5: Overall Outcomes Summary ======
    print("Collecting overall outcomes summary across all runs...")
    
    # Helper to flatten per-antibiotic dictionaries for percentile computation
    def _flatten_summary(summary_dict):
        flat_dict = {}
        for key, value in summary_dict.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    flat_dict[f"{key}_{subkey}"] = subval
            else:
                flat_dict[key] = value
        return flat_dict
    
    overall_outcomes_per_trajectory = []
    num_trajectories = len(all_trajectories_data['total_reward'])
    
    # Also create a dictionary that will record what the final AMR levels (both true and visible) were at the end of each trajectory, per antibiotic, over the last 10 timesteps.
    
    list_of_final_amr_levels_per_trajectory_per_abx = []
    
    for traj_idx in range(num_trajectories):
            
        # Store the final AMR levels for this trajectory
        list_of_final_amr_levels_per_trajectory_per_abx.append({
            abx_name: {
                'actual': all_trajectories_data['actual_AMR_levels'][abx_name][traj_idx][-10:],
                'visible': all_trajectories_data['visible_AMR_levels'][abx_name][traj_idx][-10:],
            }
            for abx_name in antibiotic_names
        })
        
        # Build cumulative trajectories for this run
        count_clinical_benefits_cumsum = np.cumsum(all_trajectories_data['count_clinical_benefits'][traj_idx])
        count_clinical_failures_cumsum = np.cumsum(all_trajectories_data['count_clinical_failures'][traj_idx])
        count_adverse_events_cumsum = np.cumsum(all_trajectories_data['count_adverse_events'][traj_idx])
        not_infected_no_treatment_cumsum = np.cumsum(all_trajectories_data['not_infected_no_treatment'][traj_idx])
        not_infected_treated_cumsum = np.cumsum(all_trajectories_data['not_infected_treated'][traj_idx])
        infected_no_treatment_cumsum = np.cumsum(all_trajectories_data['infected_no_treatment'][traj_idx])
        total_reward_cumsum = np.cumsum(all_trajectories_data['total_reward'][traj_idx])
        
        infected_treated_sensitive_per_abx_cumsum = {
            abx_name: np.cumsum(all_trajectories_data['infected_treated_sensitive'][abx_name][traj_idx])
            for abx_name in antibiotic_names
        }
        infected_treated_resistant_per_abx_cumsum = {
            abx_name: np.cumsum(all_trajectories_data['infected_treated_resistant'][abx_name][traj_idx])
            for abx_name in antibiotic_names
        }
        count_prescriptions_per_abx_cumsum = {
            abx_name: np.cumsum(all_trajectories_data['count_prescriptions'][abx_name][traj_idx])
            for abx_name in antibiotic_names
        }
        infected_treated_overall_cumsum = None
        for abx_name in antibiotic_names:
            combined = infected_treated_sensitive_per_abx_cumsum[abx_name] + infected_treated_resistant_per_abx_cumsum[abx_name]
            infected_treated_overall_cumsum = combined if infected_treated_overall_cumsum is None else infected_treated_overall_cumsum + combined
        
        overall_outcomes_summary_dict = create_overall_outcomes_summary_dict(
            count_clinical_benefits_cumsum=count_clinical_benefits_cumsum,
            count_clinical_failures_cumsum=count_clinical_failures_cumsum,
            count_adverse_events_cumsum=count_adverse_events_cumsum,
            not_infected_no_treatment_cumsum=not_infected_no_treatment_cumsum,
            not_infected_treated_cumsum=not_infected_treated_cumsum,
            infected_no_treatment_cumsum=infected_no_treatment_cumsum,
            infected_treated_overall_cumsum=infected_treated_overall_cumsum,
            infected_treated_sensitive_per_abx_cumsum=infected_treated_sensitive_per_abx_cumsum,
            infected_treated_resistant_per_abx_cumsum=infected_treated_resistant_per_abx_cumsum,
            count_prescriptions_per_abx_cumsum=count_prescriptions_per_abx_cumsum,
            total_reward_cumsum=total_reward_cumsum,
            antibiotic_names=antibiotic_names,
        )
        overall_outcomes_per_trajectory.append(overall_outcomes_summary_dict)
    
    # Build raw values dictionary (flattened for JSON + percentiles)
    if not overall_outcomes_per_trajectory:
        raise ValueError("No trajectories collected; cannot compute overall outcomes summary.")
    
    raw_template = _flatten_summary(overall_outcomes_per_trajectory[0])
    overall_outcomes_summary_raw_vals_dict = {key: [] for key in raw_template.keys()}
    
    for summary_dict in overall_outcomes_per_trajectory:
        flat_summary = _flatten_summary(summary_dict)
        for key, value in flat_summary.items():
            overall_outcomes_summary_raw_vals_dict[key].append(value)
    
    # Also add final AMR levels per antibiotic (both actual and visible) to the raw values dict
    for abx_name in antibiotic_names:
        for level_type in ['actual', 'visible']:
            final_levels_list = [
                sum(traj_final_levels[abx_name][level_type])/len(traj_final_levels[abx_name][level_type])
                for traj_final_levels in list_of_final_amr_levels_per_trajectory_per_abx
            ]
            overall_outcomes_summary_raw_vals_dict[f'final_amr_{level_type}_{abx_name}'] = final_levels_list
            
    overall_outcomes_summary_raw_vals_dict = convert_to_native_types(overall_outcomes_summary_raw_vals_dict)
    
    with open(os.path.join(experiment_figures_folder, "overall_outcomes_summary_raw_vals.json"), 'w') as f:
        json.dump(overall_outcomes_summary_raw_vals_dict, f, indent=4)
    
    print(f"Raw values saved to {experiment_figures_folder}/overall_outcomes_summary_raw_vals.json")
    
    # Compute summary statistics (percentiles) for each outcome (only for scalar series)
    overall_outcomes_summary_summary_stats_dict = {}
    for outcome_key, values_list in overall_outcomes_summary_raw_vals_dict.items():
        values_array = np.array(values_list, dtype=float)
        overall_outcomes_summary_summary_stats_dict[outcome_key] = {
            'p10': float(np.percentile(values_array, 10)),
            'p25': float(np.percentile(values_array, 25)),
            'p50': float(np.percentile(values_array, 50)),
            'p75': float(np.percentile(values_array, 75)),
            'p90': float(np.percentile(values_array, 90)),
        }
        
    # Compute summary statistics (percentiles) for final AMR levels per antibiotic, add them to the summary stats dict
    for abx_name in antibiotic_names:
        for level_type in ['actual', 'visible']:
            final_levels_list = [
                sum(traj_final_levels[abx_name][level_type])/len(traj_final_levels[abx_name][level_type])
                for traj_final_levels in list_of_final_amr_levels_per_trajectory_per_abx
            ]
            final_levels_array = np.array(final_levels_list, dtype=float)
            overall_outcomes_summary_summary_stats_dict[f'final_amr_{level_type}_{abx_name}'] = {
                'p10': float(np.percentile(final_levels_array, 10)),
                'p25': float(np.percentile(final_levels_array, 25)),
                'p50': float(np.percentile(final_levels_array, 50)),
                'p75': float(np.percentile(final_levels_array, 75)),
                'p90': float(np.percentile(final_levels_array, 90)),
            }
    
    with open(os.path.join(experiment_figures_folder, "overall_outcomes_summary_summary_stats.json"), 'w') as f:
        json.dump(overall_outcomes_summary_summary_stats_dict, f, indent=4)
    
    print(f"Summary statistics saved to {experiment_figures_folder}/overall_outcomes_summary_summary_stats.json")
    
    print("All plots complete!")
    return
