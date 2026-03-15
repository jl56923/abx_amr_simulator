"""
Evaluative Plots Script

Generates publication-quality evaluation plots from trained policies.
Merges ensemble performance plotting with action-attribute association analysis.
Automatically detects new experiments and generates plots for evaluation-ready policies.

Variable-length episode support:
    When HRL boundary clipping is active, episodes may terminate before
    ``env.max_time_steps``.  All evaluation functions in this module are
    designed to handle episodes of heterogeneous realized lengths:
    - ``run_evaluation_episodes()`` resets the environment at the start of
      every episode and records the true realized step count.
    - ``compute_action_attribute_associations()`` uses the actual length of
      each episode's trajectory rather than assuming a fixed horizon.
    Downstream analysis (e.g. ``plot_metrics_ensemble_agents`` in metrics.py)
    uses NaN-padding to aggregate trajectories of different lengths.

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


def _extract_algorithm_name(config: Dict[str, Any]) -> str:
    """Extract configured algorithm name from flat or nested config structures."""
    if not isinstance(config, dict):
        return ""

    if "algorithm" in config and config.get("algorithm") is not None:
        return str(config.get("algorithm", ""))

    agent_algo = config.get("agent_algorithm", {})
    if isinstance(agent_algo, dict):
        return str(agent_algo.get("algorithm", ""))

    return ""


def is_hrl_run(config: Dict[str, Any]) -> bool:
    """Check if config indicates an HRL run."""
    try:
        algo_upper = _extract_algorithm_name(config=config).upper()
        return "HRL" in algo_upper
    except Exception:
        return False


def is_recurrent_run(config: Dict[str, Any]) -> bool:
    """Check if config indicates recurrent-policy usage (canonical or HRL-recurrent)."""
    try:
        algo_upper = _extract_algorithm_name(config=config).upper()
        if not algo_upper:
            return False
        return ("RECURRENT" in algo_upper) or ("RPPO" in algo_upper)
    except Exception:
        return False


def detect_analysis_branches(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect canonical branch applicability across one experiment prefix's seed configs."""
    if not configs:
        raise ValueError("detect_analysis_branches() requires at least one config")

    per_seed = []
    for cfg in configs:
        algo = _extract_algorithm_name(config=cfg)
        per_seed.append(
            {
                "algorithm": algo,
                "is_hrl": is_hrl_run(config=cfg),
                "is_recurrent": is_recurrent_run(config=cfg),
            }
        )

    unique_algorithms = sorted({item["algorithm"] for item in per_seed})
    unique_hrl = {item["is_hrl"] for item in per_seed}
    unique_recurrent = {item["is_recurrent"] for item in per_seed}

    if len(unique_algorithms) > 1:
        print(
            f"  [WARN] Mixed algorithm labels across seeds: {unique_algorithms}. "
            "Branch detection will use any-seed applicability."
        )
    if len(unique_hrl) > 1:
        print(
            "  [WARN] Mixed HRL detection across seeds; using any-seed applicability "
            "for canonical branch gating."
        )
    if len(unique_recurrent) > 1:
        print(
            "  [WARN] Mixed recurrent detection across seeds; using any-seed applicability "
            "for canonical branch gating."
        )

    is_hrl = any(item["is_hrl"] for item in per_seed)
    is_recurrent = any(item["is_recurrent"] for item in per_seed)

    return {
        "algorithms": unique_algorithms,
        "is_hrl": is_hrl,
        "is_recurrent": is_recurrent,
        "hrl_branch": {
            "should_run": is_hrl,
            "reason": "detected HRL algorithm in saved seed config"
            if is_hrl
            else "no HRL algorithm detected in saved seed config",
        },
        "lstm_probe_branch": {
            "should_run": is_recurrent,
            "reason": "detected recurrent algorithm in saved seed config"
            if is_recurrent
            else "no recurrent algorithm detected in saved seed config",
        },
    }


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


# ==================== HRL Diagnostics ====================

def compute_hrl_option_stats(eval_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-option statistics from a single seed's evaluation trajectories.

    Args:
        eval_data: Output dict from ``run_evaluation_episodes()`` for one seed.

    Returns:
        Dict containing:
        - ``option_counts`` (Dict[int, int]): raw selection counts per option ID.
        - ``option_frequencies`` (Dict[int, float]): relative frequency per option ID.
        - ``option_reward_stats`` (Dict[int, Dict]): mean/std reward per option ID.
        - ``episode_length_stats`` (Dict): mean/std/min/max of episode lengths.
        - ``total_steps`` (int): total steps across all episodes.
        - ``num_episodes`` (int): number of evaluation episodes.
    """
    episode_lengths = eval_data.get("episode_lengths", [])
    episodes = eval_data.get("episodes", [])

    total_steps = sum(episode_lengths)
    option_counts: Dict[int, int] = {}
    option_reward_totals: Dict[int, float] = {}
    option_reward_counts: Dict[int, int] = {}

    for episode in episodes:
        actions = episode.get("actions", [])
        rewards = episode.get("rewards", [])

        for step_idx, action in enumerate(actions):
            # Actions for HRL are manager-level option IDs (int or 1-element array)
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    action_id = int(action.item())
                else:
                    action_id = int(action.flat[0])
            else:
                action_id = int(action)

            option_counts[action_id] = option_counts.get(action_id, 0) + 1

            step_reward = float(rewards[step_idx]) if step_idx < len(rewards) else 0.0
            option_reward_totals[action_id] = option_reward_totals.get(action_id, 0.0) + step_reward
            option_reward_counts[action_id] = option_reward_counts.get(action_id, 0) + 1

    # Relative frequencies
    option_frequencies: Dict[int, float] = {}
    for opt_id, count in option_counts.items():
        option_frequencies[opt_id] = count / total_steps if total_steps > 0 else 0.0

    # Per-option reward stats
    option_reward_stats: Dict[int, Dict[str, float]] = {}
    for opt_id in option_counts:
        cnt = option_reward_counts.get(opt_id, 0)
        mean_reward = option_reward_totals.get(opt_id, 0.0) / cnt if cnt > 0 else 0.0
        option_reward_stats[opt_id] = {"mean_reward": mean_reward, "count": cnt}

    # Episode length statistics
    ep_lengths_arr = np.array(episode_lengths, dtype=float) if episode_lengths else np.array([0.0])
    episode_length_stats: Dict[str, float] = {
        "mean": float(np.mean(ep_lengths_arr)),
        "std": float(np.std(ep_lengths_arr)),
        "min": float(np.min(ep_lengths_arr)),
        "max": float(np.max(ep_lengths_arr)),
    }

    return {
        "option_counts": {str(k): v for k, v in option_counts.items()},
        "option_frequencies": {str(k): v for k, v in option_frequencies.items()},
        "option_reward_stats": {str(k): v for k, v in option_reward_stats.items()},
        "episode_length_stats": episode_length_stats,
        "total_steps": total_steps,
        "num_episodes": len(episode_lengths),
    }


def aggregate_hrl_option_stats(per_seed_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed HRL option statistics into ensemble summary.

    Args:
        per_seed_stats: List of dicts from ``compute_hrl_option_stats()``.

    Returns:
        Dict with ensemble option statistics (mean/std/median/min/max per option).
    """
    if not per_seed_stats:
        return {
            "num_seeds": 0,
            "options": [],
            "option_statistics": {},
            "episode_length_statistics": {},
        }

    # Collect all option IDs seen across seeds
    all_option_ids = sorted(
        {opt_id for seed_stat in per_seed_stats for opt_id in seed_stat.get("option_frequencies", {}).keys()}
    )

    option_statistics: Dict[str, Any] = {}
    for opt_id in all_option_ids:
        freq_values = np.array(
            [seed_stat.get("option_frequencies", {}).get(opt_id, 0.0) for seed_stat in per_seed_stats]
        )
        reward_values = np.array(
            [
                seed_stat.get("option_reward_stats", {}).get(opt_id, {}).get("mean_reward", 0.0)
                for seed_stat in per_seed_stats
            ]
        )
        option_statistics[opt_id] = {
            "frequency_mean": float(np.mean(freq_values)),
            "frequency_std": float(np.std(freq_values)),
            "frequency_median": float(np.median(freq_values)),
            "frequency_min": float(np.min(freq_values)),
            "frequency_max": float(np.max(freq_values)),
            "frequency_p25": float(np.percentile(freq_values, 25)),
            "frequency_p75": float(np.percentile(freq_values, 75)),
            "mean_reward_mean": float(np.mean(reward_values)),
            "mean_reward_std": float(np.std(reward_values)),
        }

    # Aggregate episode length statistics across seeds
    all_ep_means = np.array([s.get("episode_length_stats", {}).get("mean", 0.0) for s in per_seed_stats])
    episode_length_statistics: Dict[str, float] = {
        "mean_of_seed_means": float(np.mean(all_ep_means)),
        "std_of_seed_means": float(np.std(all_ep_means)),
    }

    return {
        "num_seeds": len(per_seed_stats),
        "options": all_option_ids,
        "option_statistics": option_statistics,
        "episode_length_statistics": episode_length_statistics,
    }


def run_hrl_diagnostics(
    all_eval_data: List[Dict[str, Any]],
    seed_labels: List[str],
    output_dir: Path,
) -> bool:
    """Run HRL diagnostics branch and write aggregated stats to ``hrl_stats/`` subfolder.

    Processes pre-collected evaluation trajectories for each seed with
    warning-and-continue behavior: if a single seed's diagnostics fail, a warning
    is emitted and aggregation proceeds from the remaining successful seeds.
    The branch succeeds as long as at least one seed produces valid diagnostics.

    Args:
        all_eval_data: List of eval data dicts (one per seed) as returned by
            ``run_evaluation_episodes()``.  Must be the same length as
            ``seed_labels``.
        seed_labels: Human-readable label for each entry in ``all_eval_data``
            (e.g. ``["seed_1", "seed_2"]``).
        output_dir: Base evaluation output directory
            (``analysis_output/<prefix>/evaluation/``).  The ``hrl_stats/``
            subfolder will be created inside it.

    Returns:
        True if at least one seed's diagnostics were successfully collected
        and written; False otherwise.
    """
    hrl_stats_dir = output_dir / "hrl_stats"
    hrl_stats_dir.mkdir(parents=True, exist_ok=True)

    per_seed_stats: List[Dict[str, Any]] = []
    succeeded_labels: List[str] = []
    failed_labels: List[str] = []

    for eval_data, label in zip(all_eval_data, seed_labels):
        try:
            seed_stats = compute_hrl_option_stats(eval_data=eval_data)
            per_seed_stats.append(seed_stats)
            succeeded_labels.append(label)

            # Write per-seed diagnostics file
            per_seed_path = hrl_stats_dir / f"hrl_stats_{label}.json"
            with open(per_seed_path, "w") as fh:
                json.dump(seed_stats, fp=fh, indent=2)

        except Exception as exc:
            print(f"      [WARN] HRL diagnostics failed for {label}: {exc}; skipping seed")
            failed_labels.append(label)
            continue

    if not per_seed_stats:
        print(f"    [ERROR] HRL diagnostics: no seeds succeeded; skipping hrl_stats output")
        return False

    if failed_labels:
        print(
            f"    [WARN] HRL diagnostics: {len(failed_labels)} seed(s) failed "
            f"({', '.join(failed_labels)}); aggregating from {len(per_seed_stats)} successful seed(s)"
        )

    # Aggregate across seeds
    aggregated = aggregate_hrl_option_stats(per_seed_stats=per_seed_stats)
    aggregated["prefix_seed_labels"] = succeeded_labels
    aggregated["failed_seeds"] = failed_labels

    summary_path = hrl_stats_dir / "hrl_stats_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(aggregated, fp=fh, indent=2)

    # Write option_usage.csv for convenient downstream inspection
    option_usage_path = hrl_stats_dir / "option_usage.csv"
    option_stats = aggregated.get("option_statistics", {})
    csv_headers = [
        "option_id",
        "frequency_mean",
        "frequency_std",
        "frequency_median",
        "frequency_min",
        "frequency_max",
        "frequency_p25",
        "frequency_p75",
        "mean_reward_mean",
        "mean_reward_std",
    ]
    csv_lines = [",".join(csv_headers)]
    for opt_id in aggregated.get("options", []):
        stats = option_stats.get(opt_id, {})
        row = [
            str(opt_id),
            str(stats.get("frequency_mean", "")),
            str(stats.get("frequency_std", "")),
            str(stats.get("frequency_median", "")),
            str(stats.get("frequency_min", "")),
            str(stats.get("frequency_max", "")),
            str(stats.get("frequency_p25", "")),
            str(stats.get("frequency_p75", "")),
            str(stats.get("mean_reward_mean", "")),
            str(stats.get("mean_reward_std", "")),
        ]
        csv_lines.append(",".join(row))
    option_usage_path.write_text("\n".join(csv_lines))

    # Optional: generate a simple bar-chart for option frequencies when matplotlib is available
    if plt is not None and option_stats:
        try:
            options_list = aggregated.get("options", [])
            means = [option_stats[opt].get("frequency_mean", 0.0) for opt in options_list]
            stds = [option_stats[opt].get("frequency_std", 0.0) for opt in options_list]

            fig, ax = plt.subplots(figsize=(max(8, len(options_list) * 1.2), 5))
            x_pos = np.arange(len(options_list))
            ax.bar(
                x_pos,
                means,
                yerr=stds,
                capsize=5,
                alpha=0.75,
                edgecolor="black",
                linewidth=1.2,
                error_kw={"linewidth": 1.5},
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"opt_{o}" for o in options_list], rotation=45, ha="right")
            ax.set_ylabel("Mean Selection Frequency (across seeds)")
            ax.set_xlabel("Option ID")
            ax.set_title("HRL Option Selection Frequency (Ensemble)")
            ax.set_ylim(0.0, min(1.0, max(means) * 1.3 + 0.05) if means else 1.0)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plot_path = hrl_stats_dir / "option_frequency_ensemble.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved option frequency plot to hrl_stats/option_frequency_ensemble.png")
        except Exception as exc:
            print(f"    [WARN] Could not generate HRL option frequency plot: {exc}")

    print(
        f"    Saved HRL diagnostics for {len(per_seed_stats)} seed(s) to {hrl_stats_dir}"
    )
    return True


# ==================== Recurrent LSTM Probe Diagnostics ====================

def _safe_metric_mean(values: List[Any]) -> Optional[float]:
    numeric_values = [float(v) for v in values if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
    if not numeric_values:
        return None
    return float(np.mean(np.array(numeric_values, dtype=float)))


def _safe_metric_std(values: List[Any]) -> Optional[float]:
    numeric_values = [float(v) for v in values if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
    if not numeric_values:
        return None
    return float(np.std(np.array(numeric_values, dtype=float)))


def compute_lstm_probe_stats(
    log_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute per-seed LSTM probe statistics from one run's ``lstm_logs`` directory."""
    from abx_amr_simulator.analysis.probe_hidden_belief import fit_probe, load_episodes

    episode_files = sorted(log_dir.glob("episode_*.npz"))
    if len(episode_files) == 0:
        raise ValueError(f"No episode_*.npz logs found in {log_dir}")

    hidden_states, true_amr, _ = load_episodes(log_dir=log_dir)
    if hidden_states.ndim < 2 or hidden_states.shape[0] == 0:
        raise ValueError("LSTM probe failed: hidden states must be non-empty with shape (N, features)")
    if true_amr.ndim < 2 or true_amr.shape[0] == 0:
        raise ValueError("LSTM probe failed: true AMR must be non-empty with shape (N, num_antibiotics)")
    if hidden_states.shape[0] != true_amr.shape[0]:
        raise ValueError("LSTM probe failed: hidden-state and true-AMR timestep counts do not match")
    if not np.all(np.isfinite(hidden_states)):
        raise ValueError("LSTM probe failed: hidden states contain non-finite values")
    if not np.all(np.isfinite(true_amr)):
        raise ValueError("LSTM probe failed: true AMR contains non-finite values")

    probe_results = fit_probe(
        hidden_states=hidden_states,
        true_amr=true_amr,
        test_size=test_size,
        random_state=random_state,
    )

    per_antibiotic: List[Dict[str, Any]] = []
    for result in probe_results.get("results", []):
        per_antibiotic.append(
            {
                "antibiotic_idx": int(result.get("antibiotic_idx", -1)),
                "train_r2": float(result.get("train_r2", 0.0)),
                "test_r2": float(result.get("test_r2", 0.0)),
                "train_mae": float(result.get("train_mae", 0.0)),
                "test_mae": float(result.get("test_mae", 0.0)),
            }
        )

    mean_test_r2 = _safe_metric_mean(values=[item.get("test_r2") for item in per_antibiotic])
    mean_test_mae = _safe_metric_mean(values=[item.get("test_mae") for item in per_antibiotic])

    seed_stats: Dict[str, Any] = {
        "log_dir": str(log_dir),
        "num_logged_episodes": len(episode_files),
        "num_timesteps": int(hidden_states.shape[0]) if hasattr(hidden_states, "shape") and hidden_states.shape else 0,
        "hidden_state_shape": list(hidden_states.shape) if hasattr(hidden_states, "shape") else [],
        "true_amr_shape": list(true_amr.shape) if hasattr(true_amr, "shape") else [],
        "num_antibiotics": len(per_antibiotic),
        "per_antibiotic": per_antibiotic,
        "metrics": {
            "mean_test_r2": mean_test_r2,
            "mean_test_mae": mean_test_mae,
        },
    }

    return seed_stats, probe_results


def aggregate_lstm_probe_stats(per_seed_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed recurrent LSTM probe stats into ensemble summary."""
    if not per_seed_stats:
        return {
            "num_seeds": 0,
            "mean_test_r2": None,
            "std_test_r2": None,
            "mean_test_mae": None,
            "std_test_mae": None,
            "per_antibiotic": {},
        }

    mean_test_r2_vals = [seed_stat.get("metrics", {}).get("mean_test_r2") for seed_stat in per_seed_stats]
    mean_test_mae_vals = [seed_stat.get("metrics", {}).get("mean_test_mae") for seed_stat in per_seed_stats]

    per_antibiotic_metrics: Dict[str, Dict[str, Any]] = {}
    antibiotic_ids = sorted(
        {
            int(abx_stat.get("antibiotic_idx", -1))
            for seed_stat in per_seed_stats
            for abx_stat in seed_stat.get("per_antibiotic", [])
            if int(abx_stat.get("antibiotic_idx", -1)) >= 0
        }
    )

    for abx_idx in antibiotic_ids:
        test_r2_vals = [
            abx_stat.get("test_r2")
            for seed_stat in per_seed_stats
            for abx_stat in seed_stat.get("per_antibiotic", [])
            if int(abx_stat.get("antibiotic_idx", -1)) == abx_idx
        ]
        test_mae_vals = [
            abx_stat.get("test_mae")
            for seed_stat in per_seed_stats
            for abx_stat in seed_stat.get("per_antibiotic", [])
            if int(abx_stat.get("antibiotic_idx", -1)) == abx_idx
        ]
        per_antibiotic_metrics[str(abx_idx)] = {
            "mean_test_r2": _safe_metric_mean(values=test_r2_vals),
            "std_test_r2": _safe_metric_std(values=test_r2_vals),
            "mean_test_mae": _safe_metric_mean(values=test_mae_vals),
            "std_test_mae": _safe_metric_std(values=test_mae_vals),
        }

    return {
        "num_seeds": len(per_seed_stats),
        "mean_test_r2": _safe_metric_mean(values=mean_test_r2_vals),
        "std_test_r2": _safe_metric_std(values=mean_test_r2_vals),
        "mean_test_mae": _safe_metric_mean(values=mean_test_mae_vals),
        "std_test_mae": _safe_metric_std(values=mean_test_mae_vals),
        "per_antibiotic": per_antibiotic_metrics,
    }


def run_lstm_probe_diagnostics(
    log_dirs: List[Path],
    seed_labels: List[str],
    output_dir: Path,
    num_eval_episodes_per_seed: Optional[int] = None,
) -> bool:
    """Run recurrent LSTM probe branch and write aggregated stats to ``lstm_probe/``.

    Warning-and-continue behavior: if one seed fails, remaining seeds continue.
    Branch succeeds if at least one seed probe succeeds.
    """
    lstm_probe_dir = output_dir / "lstm_probe"
    lstm_probe_dir.mkdir(parents=True, exist_ok=True)

    per_seed_stats: List[Dict[str, Any]] = []
    succeeded_labels: List[str] = []
    failed_labels: List[str] = []

    for log_dir, label in zip(log_dirs, seed_labels):
        try:
            if not log_dir.exists() or not log_dir.is_dir():
                raise FileNotFoundError(f"missing lstm_logs directory at {log_dir}")

            seed_stats, probe_results = compute_lstm_probe_stats(log_dir=log_dir)
            seed_stats["seed_label"] = label
            per_seed_stats.append(seed_stats)
            succeeded_labels.append(label)

            per_seed_path = lstm_probe_dir / f"lstm_probe_{label}.json"
            with open(per_seed_path, "w") as fh:
                json.dump(seed_stats, fp=fh, indent=2)

            try:
                from abx_amr_simulator.analysis.probe_hidden_belief import plot_predictions

                plot_predictions(
                    probe_results=probe_results,
                    output_dir=lstm_probe_dir / label,
                )
            except Exception as exc:
                print(f"      [WARN] LSTM probe plotting failed for {label}: {exc}; continuing without plot")

        except Exception as exc:
            print(f"      [WARN] LSTM probe failed for {label}: {exc}; skipping seed")
            failed_labels.append(label)
            continue

    if not per_seed_stats:
        print(f"    [ERROR] LSTM probe diagnostics: no seeds succeeded; skipping lstm_probe output")
        return False

    if failed_labels:
        print(
            f"    [WARN] LSTM probe diagnostics: {len(failed_labels)} seed(s) failed "
            f"({', '.join(failed_labels)}); aggregating from {len(per_seed_stats)} successful seed(s)"
        )

    num_logged_episodes_total = int(
        sum(int(seed_stat.get("num_logged_episodes", 0)) for seed_stat in per_seed_stats)
    )

    raw_vals_payload: Dict[str, Any] = {
        "num_seeds_requested": len(seed_labels),
        "num_seeds_succeeded": len(per_seed_stats),
        "num_seeds_failed": len(failed_labels),
        "num_eval_episodes_per_seed": num_eval_episodes_per_seed,
        "num_logged_episodes_total": num_logged_episodes_total,
        "seed_labels_requested": seed_labels,
        "seed_labels_succeeded": succeeded_labels,
        "seed_labels_failed": failed_labels,
        "per_seed": per_seed_stats,
    }
    raw_vals_path = lstm_probe_dir / "lstm_probe_raw_vals.json"
    with open(raw_vals_path, "w") as fh:
        json.dump(raw_vals_payload, fp=fh, indent=2)

    aggregated = aggregate_lstm_probe_stats(per_seed_stats=per_seed_stats)
    aggregated["prefix_seed_labels"] = succeeded_labels
    aggregated["failed_seeds"] = failed_labels
    aggregated["num_seeds_requested"] = len(seed_labels)
    aggregated["num_seeds_succeeded"] = len(per_seed_stats)
    aggregated["num_eval_episodes_per_seed"] = num_eval_episodes_per_seed
    aggregated["num_logged_episodes_total"] = num_logged_episodes_total

    summary_path = lstm_probe_dir / "lstm_probe_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(aggregated, fp=fh, indent=2)

    print(
        f"    Saved LSTM probe diagnostics for {len(per_seed_stats)} seed(s) to {lstm_probe_dir}"
    )
    return True


# ==================== Evaluation ====================

def run_evaluation_episodes(
    model: Any,
    env: Any,
    num_episodes: int = 50,
    is_hrl: bool = False,
    is_recurrent: bool = False,
    recurrent_log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run evaluation episodes and collect trajectory data.

    Each episode starts with a fresh ``env.reset()``, ensuring episode
    independence. Episodes are not assumed to have equal lengths: when boundary
    clipping is active in HRL (or the environment terminates early for any
    reason) the realized episode length may be less than ``env.max_time_steps``.
    The returned ``episode_lengths`` list reflects the true realized length of
    each episode; downstream callers must not assume all lengths are equal.

    Args:
        model: Trained policy that implements ``predict(obs, deterministic)``.
        env: Gymnasium-compatible environment. Must support the
            ``(obs, info) = env.reset()`` and
            ``(obs, reward, terminated, truncated, info) = env.step(action)``
            interfaces.
        num_episodes: Number of independent evaluation episodes to run.
        is_hrl: If True, unwraps scalar array actions produced by the HRL
            manager before passing them to ``env.step``.

    Returns:
        Dict with keys:
            - ``episode_rewards`` (List[float]): total undiscounted reward per episode.
            - ``episode_lengths`` (List[int]): realized step count per episode
              (may vary across episodes if early termination occurs).
            - ``episodes`` (List[Dict]): per-episode trajectory dicts with keys
              ``obs``, ``actions``, ``rewards``, ``amr_levels``.
    """
    episode_rewards = []
    episode_lengths = []
    episode_data = []

    if is_recurrent and recurrent_log_dir is not None:
        recurrent_log_dir.mkdir(parents=True, exist_ok=True)
    
    for ep in range(num_episodes):
        # Reset at the START of each episode to ensure independence
        obs, info = env.reset()
        
        ep_reward = 0.0
        ep_length = 0
        ep_trajectory = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "amr_levels": [],
        }
        
        done = False
        recurrent_state = None
        recurrent_episode_start = np.array([True], dtype=bool)
        recurrent_hidden_states: List[np.ndarray] = []
        recurrent_true_amr: List[np.ndarray] = []
        recurrent_timesteps: List[int] = []
        while not done:
            if is_recurrent:
                action, recurrent_state = model.predict(
                    obs,
                    state=recurrent_state,
                    episode_start=recurrent_episode_start,
                    deterministic=True,
                )
            else:
                action, _ = model.predict(obs, deterministic=True)

            if is_hrl and isinstance(action, np.ndarray):
                if action.size == 1:
                    action = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if is_recurrent:
                if recurrent_state is None:
                    raise ValueError("Recurrent evaluation failed: model did not return recurrent state")

                hidden_state = recurrent_state[0] if isinstance(recurrent_state, tuple) else recurrent_state
                hidden_state_array = np.array(hidden_state, dtype=float)
                if hidden_state_array.size == 0:
                    raise ValueError("Recurrent evaluation failed: empty hidden state")

                if hidden_state_array.ndim == 3 and hidden_state_array.shape[1] > 0:
                    hidden_state_array = hidden_state_array[:, 0, :]
                hidden_state_array = hidden_state_array.reshape(-1)
                if not np.all(np.isfinite(hidden_state_array)):
                    raise ValueError("Recurrent evaluation failed: hidden state contains non-finite values")

                if not isinstance(info, dict):
                    raise ValueError("Recurrent evaluation failed: env info is not a dict")

                amr_source = info.get("actual_amr_levels", info.get("amr_levels", None))
                if not isinstance(amr_source, dict) or not amr_source:
                    raise ValueError(
                        "Recurrent evaluation failed: env info missing actual_amr_levels/amr_levels dict"
                    )

                amr_values = np.array(
                    [float(amr_source[key]) for key in sorted(amr_source.keys())],
                    dtype=float,
                )
                if amr_values.size == 0:
                    raise ValueError("Recurrent evaluation failed: true AMR vector is empty")
                if not np.all(np.isfinite(amr_values)):
                    raise ValueError("Recurrent evaluation failed: true AMR contains non-finite values")

                recurrent_hidden_states.append(hidden_state_array)
                recurrent_true_amr.append(amr_values)
                recurrent_timesteps.append(ep_length)
                recurrent_episode_start = np.array([done], dtype=bool)
            
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

        if is_recurrent and recurrent_log_dir is not None:
            if not recurrent_hidden_states:
                raise ValueError(f"Recurrent evaluation failed: no hidden states captured for episode {ep}")
            if len(recurrent_hidden_states) != len(recurrent_true_amr):
                raise ValueError("Recurrent evaluation failed: hidden-state and AMR lengths are mismatched")

            hidden_arr = np.array(recurrent_hidden_states, dtype=float)
            true_amr_arr = np.array(recurrent_true_amr, dtype=float)
            if hidden_arr.ndim < 2 or hidden_arr.shape[0] == 0:
                raise ValueError("Recurrent evaluation failed: invalid hidden-state array shape")
            if true_amr_arr.ndim < 2 or true_amr_arr.shape[0] == 0:
                raise ValueError("Recurrent evaluation failed: invalid true-AMR array shape")
            if hidden_arr.shape[0] != true_amr_arr.shape[0]:
                raise ValueError(
                    "Recurrent evaluation failed: hidden-state and true-AMR timestep counts differ"
                )

            np.savez_compressed(
                recurrent_log_dir / f"episode_{ep:04d}.npz",
                hidden_states=hidden_arr,
                true_amr=true_amr_arr,
                timesteps=np.array(recurrent_timesteps, dtype=int),
            )
    
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
            # Note: episodes may have different lengths (variable-length support);
            # `steps` is the realized step count for this specific episode.
            
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
    
    # Detect canonical branch applicability from saved seed configs
    branch_detection = detect_analysis_branches(
        configs=[cfg for _, _, cfg, _ in models_and_configs]
    )
    is_hrl = branch_detection["is_hrl"]
    is_recurrent = branch_detection["is_recurrent"]
    if is_hrl:
        print(f"  Detected HRL run; using OptionsWrapper for environment")
    else:
        print(f"  [SKIP] Canonical HRL branch: {branch_detection['hrl_branch']['reason']}")

    if is_recurrent:
        print(f"  Detected recurrent run; canonical LSTM probe branch is eligible")
    else:
        print(f"  [SKIP] Canonical recurrent/LSTM branch: {branch_detection['lstm_probe_branch']['reason']}")
    
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
    recurrent_probe_log_dirs: List[Path] = []
    
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
            
            eval_data = run_evaluation_episodes(
                model=model,
                env=env,
                num_episodes=num_eval_episodes,
                is_hrl=is_hrl,
                is_recurrent=is_recurrent,
                recurrent_log_dir=(
                    output_dir / "lstm_probe" / "logs" / (f"seed_{seed}" if seed >= 0 else "run")
                    if is_recurrent
                    else None
                ),
            )
            all_eval_data.append(eval_data)
            seed_labels.append(f"seed_{seed}" if seed >= 0 else "run")
            if is_recurrent:
                recurrent_probe_log_dirs.append(
                    output_dir / "lstm_probe" / "logs" / (f"seed_{seed}" if seed >= 0 else "run")
                )
            
            env.close()
        except Exception as e:
            print(f"      [ERROR] Evaluation failed: {e}")
            continue
    
    if not all_eval_data:
        print(f"  [ERROR] No successful evaluations for {prefix}")
        return False, None, None
    
    # ===== Part 3: HRL diagnostics branch =====
    hrl_diagnostics_ok = False
    if is_hrl:
        print(f"  Running HRL diagnostics branch...")
        hrl_diagnostics_ok = run_hrl_diagnostics(
            all_eval_data=all_eval_data,
            seed_labels=seed_labels,
            output_dir=output_dir,
        )

    # ===== Part 3b: Recurrent LSTM probe branch =====
    lstm_probe_ok = False
    if is_recurrent:
        print(f"  Running recurrent LSTM probe branch...")
        lstm_probe_ok = run_lstm_probe_diagnostics(
            log_dirs=recurrent_probe_log_dirs,
            seed_labels=seed_labels,
            output_dir=output_dir,
            num_eval_episodes_per_seed=num_eval_episodes,
        )

    # ===== Part 4: Compute action-attribute associations (skip for HRL) =====
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
        "branch_detection": branch_detection,
        "hrl_diagnostics_ok": hrl_diagnostics_ok,
        "lstm_probe_ok": lstm_probe_ok,
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
