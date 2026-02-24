"""
Evaluative Plots Script

Generates publication-quality evaluation plots from trained policies.
Merges ensemble performance plotting with action-attribute association analysis.
Automatically detects new experiments and generates plots for evaluation-ready policies.

How to use:
    # Auto mode: Generate plots for all new exp_* experiments
    python -m abx_amr_simulator.analysis.evaluative_plots
    
    # Manual mode: Generate plots for specific single run (by full name including timestamp)
    python -m abx_amr_simulator.analysis.evaluative_plots --experiment-name example_run_20260115_143614
    
    # Generate ensemble plots from multiple seeds
    python -m abx_amr_simulator.analysis.evaluative_plots --experiment-name example_run --aggregate-by-seed
    
    # Force regeneration with custom eval episodes
    python -m abx_amr_simulator.analysis.evaluative_plots --force --num-eval-episodes 100

Behavior:
    By default (without --aggregate-by-seed), finds and evaluates a single run matching
    the exact prefix (e.g., 'example_run_20260115_143614'). Ensemble plots show variation
    across evaluation episodes for that single agent.
    
    With --aggregate-by-seed flag, looks for multiple seed runs matching pattern
    <prefix>_seed*_* and creates ensemble plots with statistics across seeds
    (e.g., --prefix example_run finds example_run_seed1_*, example_run_seed2_*, etc.).

Naming convention for multi-seed experiments:
    When running multiple seeds of the same experiment, use the pattern:
        <run_name>_seed1, <run_name>_seed2, <run_name>_seed3, etc.
    
    Then analyze with: --prefix <run_name> --aggregate-by-seed
"""

import argparse
import os
import re
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import pdb
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from stable_baselines3 import PPO, A2C

from abx_amr_simulator.utils import (
    load_config,
    create_reward_calculator,
    create_patient_generator,
    create_environment,
    extract_experiment_prefix,
    extract_timestamp_from_run_folder,
    find_experiment_runs,
    load_registry,
    update_registry_csv,
    identify_new_experiments,
    clear_registry,
    plot_metrics_ensemble_agents,
)
# ==================== Model Loading ====================

from pathlib import Path

def load_best_model_from_run(run_folder: Path) -> Optional[Any]:
    """Load the best model from a run folder."""
    model_path = run_folder / "checkpoints" / "best_model.zip"
    if not model_path.exists():
        print(f"    [WARN] No best_model.zip found in {run_folder.name}")
        return None
    
    # Detect algorithm type from config
    cfg_path = run_folder / "full_agent_env_config.yaml"
    if not cfg_path.exists():
        print(f"    [WARN] No full_agent_env_config.yaml found; assuming PPO")
        agent_class = PPO
    else:
        try:
            import yaml
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
            algo = cfg.get("agent_algorithm", {}).get("algorithm", "PPO")
            agent_class = {"PPO": PPO, "A2C": A2C}.get(algo, PPO)
        except Exception:
            agent_class = PPO
    
    try:
        model = agent_class.load(str(model_path))
        return model
    except Exception as e:
        print(f"    [ERROR] Failed to load model: {e}")
        return None


def load_config_from_run(run_folder: Path) -> Optional[Dict[str, Any]]:
    """Load config from run folder."""
    cfg_path = run_folder / "full_agent_env_config.yaml"
    if not cfg_path.exists():
        return None
    
    try:
        import yaml
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def is_hrl_run(config: Dict[str, Any]) -> bool:
    """Check if config indicates an HRL run."""
    try:
        # Check flat structure first (most common after load_config())
        if "algorithm" in config:
            algo = config.get("algorithm", "")
            if "HRL" in str(algo).upper():
                return True
        
        # Check nested structure
        agent_algo = config.get("agent_algorithm", {})
        if isinstance(agent_algo, dict):
            algo = agent_algo.get("algorithm", "")
            if "HRL" in str(algo).upper():
                return True
        
        return False
    except Exception:
        return False


def wrap_environment_for_hrl(
    env: Any,
    config: Dict[str, Any],
    run_dir: Path,
) -> Optional[Any]:
    """Wrap environment with OptionsWrapper for HRL evaluation.
    
    Returns wrapped env if HRL, None if wrapping failed or not HRL.
    """
    try:
        from abx_amr_simulator.hrl import OptionLibraryLoader, OptionsWrapper
    except ImportError:
        print(f"      [WARN] Could not import HRL modules; skipping HRL wrapping")
        return None
    
    try:
        # Try to use resolved option library config first (contains absolute paths)
        resolved_lib_config_path = run_dir / "full_options_library.yaml"
        if resolved_lib_config_path.exists():
            import yaml
            with open(resolved_lib_config_path, 'r') as f:
                resolved_config = yaml.safe_load(f)
            option_lib_path = resolved_config.get("library_config_path")
        else:
            # Fall back to regular config
            option_lib_path = config.get("hrl", {}).get("option_library", None)
        
        if not option_lib_path:
            print(f"      [WARN] No HRL option_library specified in config")
            return None
        
        loader = OptionLibraryLoader()
        option_library, _ = loader.load_library(library_config_path=option_lib_path, env=env)
        
        wrapped_env = OptionsWrapper(env=env, option_library=option_library)
        return wrapped_env
    except Exception as e:
        print(f"      [WARN] Failed to wrap environment for HRL: {e}")
        return None


# ==================== Evaluation ====================

def run_evaluation_episodes(
    model: Any,
    env: Any,
    num_episodes: int = 50,
) -> Dict[str, Any]:
    """
    Run evaluation episodes and collect trajectory data.
    
    Returns dict with episode metrics and trajectory data.
    """
    episode_rewards = []
    episode_lengths = []
    episode_data = []
    
    obs, info = env.reset()
    for ep in range(num_episodes):
        ep_reward = 0.0
        ep_length = 0
        ep_trajectory = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "amr_levels": [],
        }
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    action = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += float(reward)
            ep_length += 1
            
            ep_trajectory["obs"].append(obs.copy())
            ep_trajectory["actions"].append(action)
            ep_trajectory["rewards"].append(float(reward))
            
            # Capture AMR levels if available in info
            if "amr_levels" in info:
                ep_trajectory["amr_levels"].append(info["amr_levels"].copy())
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_data.append(ep_trajectory)
    
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episodes": episode_data,
    }


# ==================== Action-Attribute Associations ====================

def _normalize_actions_array(actions: np.ndarray) -> np.ndarray:
    """Normalize actions array shape to (steps, patients)."""
    arr = np.array(actions)
    if arr.size == 0:
        return arr
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return np.array([])
    return arr


def _compute_mutual_information(bins: np.ndarray, prescribed: np.ndarray) -> Optional[float]:
    """Compute mutual information between binned attribute values and binary prescribing."""
    if bins.size < 2 or bins.size != prescribed.size:
        return None
    total = float(bins.size)
    bin_vals, bin_counts = np.unique(bins, return_counts=True)
    pres_vals, pres_counts = np.unique(prescribed, return_counts=True)
    if bin_vals.size < 2 or pres_vals.size < 2:
        return 0.0

    joint_counts = np.zeros((bin_vals.size, pres_vals.size), dtype=float)
    for b, p in zip(bins, prescribed):
        bi = np.where(bin_vals == b)[0][0]
        pj = np.where(pres_vals == p)[0][0]
        joint_counts[bi, pj] += 1.0

    mi = 0.0
    for i, bi_count in enumerate(bin_counts):
        for j, pj_count in enumerate(pres_counts):
            joint = joint_counts[i, j]
            if joint == 0:
                continue
            p_xy = joint / total
            p_x = bi_count / total
            p_y = pj_count / total
            mi += p_xy * np.log(p_xy / (p_x * p_y))
    return float(mi)


def _build_bins(values: np.ndarray, num_bins: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Create quantile-based bins and return bin indices plus edge list."""
    if values.size == 0:
        return np.array([]), []
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([float(values.min()), float(values.max())])
    bin_indices = np.digitize(values, edges[1:-1], right=False)
    bins_meta: List[Tuple[float, float]] = []
    for i in range(len(edges) - 1):
        bins_meta.append((float(edges[i]), float(edges[i + 1])))
    return bin_indices, bins_meta


def compute_action_attribute_associations(
    all_eval_data: List[Dict[str, Any]],
    antibiotic_names: List[str],
    num_bins: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute prescribing frequencies conditioned on patient attributes and mutual information.
    
    Returns: (action_rows, assoc_rows)
    """
    action_rows: List[Dict[str, Any]] = []
    assoc_rows: List[Dict[str, Any]] = []

    abx_names = [n for n in antibiotic_names if n != "no_treatment"]
    if not abx_names:
        return [], []

    # Collect per-attribute data from evaluation episodes
    attr_data: Dict[str, Dict[str, list]] = {}

    for eval_data in all_eval_data:
        for ep_trajectory in eval_data.get("episodes", []):
            obs = np.array(ep_trajectory.get("obs", []), dtype=float)
            actions = np.array(ep_trajectory.get("actions", []), dtype=int)
            rewards = np.array(ep_trajectory.get("rewards", []), dtype=float)
            
            if obs.size == 0 or actions.size == 0 or rewards.size == 0:
                continue
            
            # Normalize shapes
            if obs.ndim == 2:  # (steps, features)
                # Assume single patient per step for evaluation
                obs = obs[:, np.newaxis, :]  # (steps, 1, features)
            
            steps = min(obs.shape[0], len(actions), len(rewards))
            
            for s in range(steps):
                action_val = int(actions[s])
                step_reward = float(rewards[s])
                
                # For each attribute in observation
                for attr_idx in range(obs.shape[2]):
                    attr_name = f"attr_{attr_idx}"
                    obs_val = float(obs[s, 0, attr_idx]) if obs.shape[1] > 0 else 0.0
                    
                    if attr_name not in attr_data:
                        attr_data[attr_name] = {"values": [], "actions": [], "rewards": []}
                    attr_data[attr_name]["values"].append(obs_val)
                    attr_data[attr_name]["actions"].append(action_val)
                    attr_data[attr_name]["rewards"].append(step_reward)

    # Compute associations per attribute
    for attr_name, series in attr_data.items():
        values = np.array(series["values"], dtype=float)
        actions_arr = np.array(series["actions"], dtype=int)
        rewards_arr = np.array(series["rewards"], dtype=float)
        
        if values.size == 0:
            continue
        
        bin_indices, bins_meta = _build_bins(values, num_bins)

        for abx_idx, abx_name in enumerate(abx_names):
            # Bin-wise aggregation
            for bin_id, (low, high) in enumerate(bins_meta):
                mask = bin_indices == bin_id
                count = int(mask.sum())
                if count < 1:
                    continue
                prescribed_mask = mask & (actions_arr == abx_idx)
                count_prescribe = int(prescribed_mask.sum())
                prescribe_rate = count_prescribe / count if count > 0 else 0.0
                mean_reward = float(rewards_arr[mask].mean()) if count > 0 else None
                action_rows.append({
                    "attribute": attr_name,
                    "bin_label": f"bin_{bin_id}",
                    "bin_low": low,
                    "bin_high": high,
                    "antibiotic": abx_name,
                    "count_samples": count,
                    "count_prescribe": count_prescribe,
                    "prescribe_rate": prescribe_rate,
                    "mean_reward": mean_reward,
                })

            # Mutual information
            pres_flag = (actions_arr == abx_idx).astype(int)
            mi = _compute_mutual_information(bin_indices, pres_flag)
            assoc_rows.append({
                "attribute": attr_name,
                "antibiotic": abx_name,
                "mutual_information": mi,
                "samples": int(values.size),
            })

    return action_rows, assoc_rows


def write_csv_action_drivers(rows: List[Dict[str, Any]], out_path: Path):
    """Write action-attribute associations to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "attribute",
        "bin_label",
        "bin_low",
        "bin_high",
        "antibiotic",
        "count_samples",
        "count_prescribe",
        "prescribe_rate",
        "mean_reward",
    ]
    lines = [",".join(headers)]
    for r in rows:
        line = (
            f"{r['attribute']},{r['bin_label']},{r['bin_low']},{r['bin_high']}"
            f",{r['antibiotic']},{r['count_samples']},{r['count_prescribe']},"
            f"{r['prescribe_rate']},{r['mean_reward']}"
        )
        lines.append(line)
    out_path.write_text("\n".join(lines))


def write_csv_action_attribute_associations(rows: List[Dict[str, Any]], out_path: Path):
    """Write mutual information associations to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["attribute", "antibiotic", "mutual_information", "samples"]
    lines = [",".join(headers)]
    for r in rows:
        line = f"{r['attribute']},{r['antibiotic']},{r['mutual_information']},{r['samples']}"
        lines.append(line)
    out_path.write_text("\n".join(lines))


# ==================== Auto-Detection ====================

def analyze_experiment(
    prefix: str,
    results_dir: str = "results",
    analysis_dir: str = "analysis_output",
    num_eval_episodes: int = 10,
    aggregate_by_seed: bool = False,
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Analyze experiment run(s): load best models and generate evaluation plots.
    
    Behavior depends on aggregate_by_seed:
    - If aggregate_by_seed=False (default): Finds single run matching exact prefix.
      Generates single-agent plots (not ensemble, just multiple episodes of one agent).
    - If aggregate_by_seed=True: Finds all seed runs matching <prefix>_seed*_* pattern.
      Generates ensemble plots aggregating across multiple seeds.
    
    Generates:
    - Ensemble/single-agent plots: episode rewards, episode lengths, AMR dynamics, clinical outcomes
    - Action-attribute associations: prescribing rates and mutual information per patient attribute
    
    Returns (success, output_dir)
    """
    # Find runs based on aggregation mode
    runs = find_experiment_runs(
        prefix=prefix, 
        results_dir=results_dir,
        aggregate_by_seed=aggregate_by_seed
    )
    if not runs:
        print(f"  [WARN] No runs found for prefix: {prefix}")
        return False, None, None
    
    if aggregate_by_seed:
        print(f"  Found {len(runs)} seed runs for {prefix}")
    else:
        if len(runs) > 1:
            print(f"  [WARN] Found {len(runs)} runs for prefix {prefix}. Using only the first one.")
            print(f"         (Use --aggregate-by-seed to analyze multiple seeds together)")
            runs = runs[:1]
        print(f"  Found 1 run for {prefix}")
    
    # Load models and configs
    models_and_configs = []
    representative_timestamp: Optional[str] = None
    
    for run_dir in runs:
        match = re.search(r'_seed(\d+)_', run_dir.name)
        seed = int(match.group(1)) if match else -1
        
        # Capture timestamp from first run (representative for registry)
        if representative_timestamp is None:
            representative_timestamp = extract_timestamp_from_run_folder(run_dir.name)
        
        label = f"seed {seed}" if seed >= 0 else "run"
        print(f"    Loading {label}...")
        
        # Check if training is complete: prefer per-run sentinel, fallback to root registry
        training_completed_file = run_dir / ".training_completed.txt"
        if not training_completed_file.exists():
            fallback_ok = False
            try:
                registry_file = Path(results_dir) / ".training_completed.txt"
                if registry_file.exists():
                    prefixes = load_registry(str(registry_file))
                    m2 = re.match(r'^(exp_.+?_seed\d+)_', run_dir.name)
                    seeded_run_name = m2.group(1) if m2 else None
                    if seeded_run_name:
                        if seeded_run_name in prefixes or any(p.startswith(f"{seeded_run_name}_") for p in prefixes):
                            fallback_ok = True
            except Exception:
                fallback_ok = False
            if not fallback_ok:
                print(f"      [WARN] Training may not be complete (.training_completed.txt not found)")
                print(f"             Analysis will proceed but results may be incomplete")
        
        model = load_best_model_from_run(run_folder=run_dir)
        config = load_config_from_run(run_folder=run_dir)
        
        if model is None or config is None:
            print(f"      [SKIP] Could not load model or config")
            continue
        
        models_and_configs.append((seed, model, config, run_dir))
    
    if not models_and_configs:
        print(f"  [ERROR] Failed to load any models for {prefix}")
        return False, None, None
    
    analysis_root = Path(analysis_dir)
    if analysis_root.name != "analysis_output":
        analysis_root = analysis_root / "analysis_output"
    analysis_root.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = analysis_root / prefix / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== Part 1: Generate ensemble/single-agent plots =====
    print(f"  Generating {'ensemble' if aggregate_by_seed else 'single-agent'} plots...")
    
    # Check if this is an HRL run
    first_config = models_and_configs[0][2]
    is_hrl = is_hrl_run(first_config)
    if is_hrl:
        print(f"  Detected HRL run; using OptionsWrapper for environment")
    
    try:
        # Extract just the models
        models = [model for _, model, _, _ in models_and_configs]
        
        # Create environment from first seed's config (all should be identical)
        _, _, config, run_dir = models_and_configs[0]
        rc = create_reward_calculator(config=config)
        pg = create_patient_generator(config=config)
        env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
        
        # Wrap environment for HRL if needed
        if is_hrl:
            wrapped_env = wrap_environment_for_hrl(env=env, config=config, run_dir=run_dir)
            if wrapped_env is not None:
                env = wrapped_env
                print(f"  Environment successfully wrapped for HRL evaluation")
            else:
                print(f"  [WARN] HRL wrapping failed; using base environment (may cause shape mismatch)")
        
        # Call ensemble plotting utility
        plot_metrics_ensemble_agents(
            models=models,
            env=env,
            experiment_folder=str(output_dir),
            n_episodes_per_agent=num_eval_episodes,
            deterministic=True,
            figures_folder_name="ensemble" if aggregate_by_seed else "plots",
            per_seed_figures=False,
            episode_seed_start=1000,
        )
        
        env.close()
        plot_type = "ensemble plots" if aggregate_by_seed else "agent plots"
        print(f"    Saved {plot_type} to {output_dir}/{'ensemble' if aggregate_by_seed else 'plots'}/")
    except Exception as e:
        print(f"    [WARN] Plot generation failed: {e}")
        # Continue anyway - plots are optional
    
    # ===== Part 2: Run individual evaluations for action-attribute associations =====
    print(f"  Running individual evaluations...")
    all_eval_data = []
    seed_labels = []
    
    for seed, model, config, run_dir in models_and_configs:
        label = f"seed {seed}" if seed >= 0 else "run"
        print(f"    Evaluating {label}...")
        try:
            rc = create_reward_calculator(config=config)
            pg = create_patient_generator(config=config)
            env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
            
            # Wrap environment for HRL if needed
            if is_hrl:
                wrapped_env = wrap_environment_for_hrl(env=env, config=config, run_dir=run_dir)
                if wrapped_env is not None:
                    env = wrapped_env
            
            eval_data = run_evaluation_episodes(model=model, env=env, num_episodes=num_eval_episodes)
            all_eval_data.append(eval_data)
            seed_labels.append(f"seed_{seed}" if seed >= 0 else "run")
            
            env.close()
        except Exception as e:
            print(f"      [ERROR] Evaluation failed: {e}")
            continue
    
    if not all_eval_data:
        print(f"  [ERROR] No successful evaluations for {prefix}")
        return False, None, None
    
    # ===== Part 3: Compute action-attribute associations (skip for HRL) =====
    if is_hrl:
        print(f"  Skipping action-attribute associations for HRL run (actions are option IDs, not antibiotics)")
        action_rows = []
        assoc_rows = []
    else:
        print(f"  Computing action-attribute associations...")
        antibiotic_names = models_and_configs[0][2].get("environment", {}).get("antibiotic_names", []) if models_and_configs else []
        
        action_rows, assoc_rows = compute_action_attribute_associations(
            all_eval_data=all_eval_data,
            antibiotic_names=antibiotic_names,
            num_bins=5,
        )
        
        if action_rows:
            write_csv_action_drivers(rows=action_rows, out_path=output_dir / "action_drivers.csv")
            print(f"    Saved action_drivers.csv")
        
        if assoc_rows:
            write_csv_action_attribute_associations(rows=assoc_rows, out_path=output_dir / "action_attribute_associations.csv")
            print(f"    Saved action_attribute_associations.csv")
    
    # Save metadata
    metadata = {
        "prefix": prefix,
        "num_runs": len(all_eval_data),
        "num_episodes_per_run": num_eval_episodes,
        "aggregate_by_seed": aggregate_by_seed,
        "seed_labels": seed_labels,
        "has_action_attributes": bool(action_rows),
        "has_plots": True,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved all outputs to {output_dir}")
    return True, output_dir, representative_timestamp


# ==================== Main Pipeline ====================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluative plots from trained policies. Generates ensemble plots and action-attribute associations."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Manual mode: Generate plots for specific experiment. For single run: include full name with timestamp. For multi-seed: use base name with --aggregate-by-seed"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all exp_* experiments"
    )
    parser.add_argument(
        "--aggregate-by-seed",
        action="store_true",
        help="Group multiple seed runs together (looks for <prefix>_seed*_* pattern). Default: False (exact prefix match)"
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per seed (default: 10)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to results directory"
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="analysis_output",
        help="Base path where analysis_output will be created"
    )
    
    args = parser.parse_args()
    
    analysis_root = Path(args.analysis_dir)
    if analysis_root.name != "analysis_output":
        analysis_root = analysis_root / "analysis_output"
    analysis_root.mkdir(parents=True, exist_ok=True)

    registry_path = str(analysis_root / ".evaluative_plots_completed.txt")
    
    if args.force:
        print("[INFO] --force flag: Clearing registry and regenerating all experiments")
        clear_registry(registry_path=registry_path)
        prefixes_to_analyze = identify_new_experiments(registry_path=registry_path, results_dir=args.results_dir)
        if not prefixes_to_analyze:
            from abx_amr_simulator.utils import scan_for_experiments
            prefixes_to_analyze = scan_for_experiments(results_dir=args.results_dir)
    elif args.experiment_name:
        print(f"[INFO] Manual mode: Generating plots for experiment '{args.experiment_name}'")
        prefixes_to_analyze = {args.experiment_name}
    else:
        print("[INFO] Auto mode: Scanning for new experiments...")
        prefixes_to_analyze = identify_new_experiments(registry_path=registry_path, results_dir=args.results_dir)
    
    if not prefixes_to_analyze:
        print("[INFO] No experiments to analyze.")
        return
    
    print(f"[INFO] Found {len(prefixes_to_analyze)} experiment(s) to analyze")
    
    for prefix in sorted(prefixes_to_analyze):
        print(f"\nAnalyzing: {prefix}")
        success, output_dir, timestamp = analyze_experiment(
            prefix=prefix,
            results_dir=args.results_dir,
            analysis_dir=str(analysis_root),
            num_eval_episodes=args.num_eval_episodes,
            aggregate_by_seed=args.aggregate_by_seed,
        )
        
        if success and timestamp:
            update_registry_csv(registry_path=registry_path, run_name=prefix, timestamp=timestamp)
            print(f"✓ Updated registry for {prefix} (timestamp {timestamp})")
        elif success:
            print(f"✗ Plot generation succeeded but could not extract timestamp from run folder for {prefix}")
        else:
            print(f"✗ Failed to generate plots for {prefix}")
    
    print("\n[INFO] Evaluative plots complete!")


if __name__ == "__main__":
    main()
