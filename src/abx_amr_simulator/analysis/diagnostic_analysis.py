"""
Diagnostic Analysis Script

Analyzes training behavior and environment dynamics from evaluation trajectories.
Automatically detects new experiments and performs analysis on single or aggregated seeds.

How to use:
    # Auto mode: Analyze all new exp_* experiments
    python -m abx_amr_simulator.analysis.diagnostic_analysis
    
    # Manual mode: Analyze specific single run (by full name including timestamp)
    python -m abx_amr_simulator.analysis.diagnostic_analysis --experiment-name example_run_20260115_143614
    
    # Analyze and aggregate multiple seeds
    python -m abx_amr_simulator.analysis.diagnostic_analysis --experiment-name example_run --aggregate-by-seed
    
    # Force re-analysis of all experiments
    python -m abx_amr_simulator.analysis.diagnostic_analysis --force

Behavior:
    By default (without --aggregate-by-seed), finds and analyzes a single run matching 
    the exact prefix (e.g., 'example_run_20260115_143614').
    
    With --aggregate-by-seed flag, looks for multiple seed runs matching pattern
    <prefix>_seed*_* and aggregates analysis across seeds (e.g., --experiment-name example_run
    finds example_run_seed1_*, example_run_seed2_*, etc.).

Naming convention for multi-seed experiments:
    When running multiple seeds of the same experiment, use the pattern:
        <run_name>_seed1, <run_name>_seed2, <run_name>_seed3, etc.
    
    Then analyze with: --experiment-name <run_name> --aggregate-by-seed
"""

import argparse
import json
import os
import re
import struct
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional

import numpy as np
try:
    from tensorboard.backend.event_processing import event_accumulator as _tb_event_accumulator
    _HAS_TB_EVENT_ACC = True
except Exception:
    _HAS_TB_EVENT_ACC = False
try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None

try:
    import yaml
except Exception:
    yaml = None

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

from abx_amr_simulator.utils import (
    extract_experiment_prefix,
    extract_timestamp_from_run_folder,
    find_experiment_runs,
    load_registry,
    update_registry_csv,
    identify_new_experiments,
    clear_registry,
)

# ==================== TensorBoard Event File Parsers ====================

def read_tfrecord_simple(event_file_path: Path, metric_name: str) -> List[Tuple[int, float]]:
    """Simple TFRecord reader without TensorFlow dependency.
    
    Reads protobuf-encoded event records and extracts scalar metrics.
    Gracefully handles parsing errors.
    
    Args:
        event_file_path: Path to events.out.tfevents.* file
        metric_name: Scalar tag to extract (e.g., 'eval/mean_reward')
    
    Returns:
        List of (global_step, value) tuples in chronological order
    """
    data = []
    
    try:
        with open(event_file_path, 'rb') as f:
            while True:
                # Read record length (8 bytes: uint64)
                len_bytes = f.read(8)
                if not len_bytes or len(len_bytes) < 8:
                    break
                
                record_len = struct.unpack('<Q', len_bytes)[0]
                record = f.read(record_len)
                if len(record) < record_len:
                    break
                
                # Skip CRC bytes (4 bytes after record)
                f.read(4)
                
                # Parse the record as a protobuf Event
                try:
                    event_dict = parse_event_protobuf(record, metric_name)
                    if event_dict:
                        data.append((event_dict['step'], event_dict['value']))
                except Exception:
                    # Malformed record, continue
                    pass
    except Exception as e:
        print(f"Warning: Error reading event file: {e}")
    
    return sorted(data, key=lambda x: x[0])


def parse_event_protobuf(data: bytes, target_tag: str) -> Optional[Dict[str, Any]]:
    """Minimal protobuf parser for Event message.
    
    Extracts step and scalar values. Very basic implementation, just
    enough to handle TensorBoard event format.
    """
    step = None
    tag = None
    value = None
    
    pos = 0
    while pos < len(data):
        try:
            field_num, wire_type, field_data, new_pos = parse_protobuf_field(data, pos)
            pos = new_pos
            
            # Field 1: step (int64)
            if field_num == 1 and wire_type == 0:
                step = parse_varint(field_data)
            
            # Field 5: summary (nested message)
            elif field_num == 5 and wire_type == 2:
                # Parse summary for scalar values
                tag, value = parse_summary_message(field_data, target_tag)
        except:
            break
    
    if step is not None and tag == target_tag and value is not None:
        return {'step': step, 'value': value}
    return None


def parse_protobuf_field(data: bytes, pos: int) -> Tuple[int, int, Any, int]:
    """Parse a single protobuf field.
    
    Returns: (field_number, wire_type, field_value_bytes, next_position)
    """
    # Read field header (varint)
    field_header, pos = read_varint(data, pos)
    field_num = field_header >> 3
    wire_type = field_header & 0x7
    
    # Read field value based on wire type
    if wire_type == 0:  # Varint
        value, pos = read_varint(data, pos)
        return field_num, wire_type, value, pos
    elif wire_type == 2:  # Length-delimited
        length, pos = read_varint(data, pos)
        value = data[pos:pos + length]
        return field_num, wire_type, value, pos + length
    else:
        # Unsupported wire type, skip
        return field_num, wire_type, b'', pos


def read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    """Read a varint from data starting at pos.
    
    Returns: (value, next_position)
    """
    value = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7f) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return value, pos


def parse_varint(data: bytes) -> int:
    """Parse a single varint from bytes."""
    value = 0
    shift = 0
    for byte in data:
        value |= (byte & 0x7f) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return value


def parse_summary_message(data: bytes, target_tag: str) -> Tuple[Optional[str], Optional[float]]:
    """Parse summary protobuf message, extract scalar value for target_tag."""
    pos = 0
    while pos < len(data):
        try:
            field_num, wire_type, field_data, pos = parse_protobuf_field(data, pos)
            
            # Field 1: value (repeated, nested message)
            if field_num == 1 and wire_type == 2:
                tag, value = parse_value_message(field_data, target_tag)
                if tag == target_tag:
                    return tag, value
        except:
            break
    return None, None


def parse_value_message(data: bytes, target_tag: str) -> Tuple[Optional[str], Optional[float]]:
    """Parse Value protobuf message (contains tag and simple_value)."""
    tag = None
    value = None
    pos = 0
    
    while pos < len(data):
        try:
            field_num, wire_type, field_data, pos = parse_protobuf_field(data, pos)
            
            # Field 1: tag (string)
            if field_num == 1 and wire_type == 2:
                tag = field_data.decode('utf-8', errors='ignore')
            
            # Field 2: simple_value (float)
            elif field_num == 2 and wire_type == 5:
                if len(field_data) >= 4:
                    value = struct.unpack('<f', field_data[:4])[0]
        except:
            break
    
    return tag, value


def find_event_files(run_dir: Path, logs_subdir: str = "logs") -> List[Path]:
    """Find TensorBoard event files in run directory.
    
    Searches recursively under logs_subdir for events.out.tfevents.* files.
    """
    logs_dir = run_dir / logs_subdir
    if not logs_dir.exists():
        return []
    
    # Search recursively for events.out.tfevents.* files
    event_files = list(logs_dir.rglob("events.out.tfevents.*"))
    return sorted(event_files)


def extract_eval_metrics(
    run_dir: Path,
    metric_name: str = "eval/mean_reward",
    fallback_metric_names: Optional[List[str]] = None,
) -> List[Tuple[int, float]]:
    """Extract evaluation metrics from TensorBoard logs with graceful fallbacks.
    
    Primary path: TensorBoard event_accumulator (if available).
    Secondary path: Minimal TFRecord parser (legacy).
    Tries metric_name then fallback_metric_names. Returns first non-empty series.
    """
    event_files = find_event_files(run_dir)
    if not event_files:
        print(f"  [WARN] No TensorBoard event files found under {run_dir}/logs")
        return []
    print(f"  [DEBUG] Found {len(event_files)} event file(s) under {run_dir}/logs")
    for ef in event_files[:3]:
        print(f"    [DEBUG] event file: {ef}")

    fallback_metric_names = fallback_metric_names or [
        "rollout/ep_rew_mean",
        "train/episode_reward",
        "train/rollout/ep_rew_mean",
    ]
    metric_tags = [metric_name] + [tag for tag in fallback_metric_names if tag != metric_name]

    def _extract_with_event_accumulator(tag: str) -> List[Tuple[int, float]]:
        if not _HAS_TB_EVENT_ACC:
            return []
        data: List[Tuple[int, float]] = []
        for event_file in event_files:
            try:
                acc = _tb_event_accumulator.EventAccumulator(str(event_file))
                acc.Reload()
                scalars = acc.Scalars(tag)
                data.extend([(s.step, float(s.value)) for s in scalars])
            except Exception as e:
                print(f"    [DEBUG] event_accumulator failed for {event_file} tag '{tag}': {e}")
        return sorted(set(data), key=lambda x: x[0])

    def _extract_with_legacy_parser(tag: str) -> List[Tuple[int, float]]:
        data: List[Tuple[int, float]] = []
        for event_file in event_files:
            parsed = read_tfrecord_simple(event_file, tag)
            data.extend(parsed)
        return sorted(set(data), key=lambda x: x[0])

    for tag in metric_tags:
        all_data = _extract_with_event_accumulator(tag)
        if not all_data:
            all_data = _extract_with_legacy_parser(tag)
        print(f"    [DEBUG] tag='{tag}' yielded {len(all_data)} points from {len(event_files)} file(s)")
        if all_data:
            if tag != metric_name:
                print(f"  [INFO] Using fallback metric tag '{tag}' for convergence curve (primary '{metric_name}' not found)")
            return all_data

    print(f"  [WARN] No metric data found for tags {metric_tags} in {run_dir}/logs")
    return []


def extract_eval_metrics_from_eval_logs(run_dir: Path, eval_logs_subdir: str = "eval_logs") -> List[Tuple[int, float]]:
    """Fallback: build convergence data from eval_logs npz files.
    
    Filters out macOS hidden files and handles corrupted files gracefully.
    """
    eval_dir = run_dir / eval_logs_subdir
    if not eval_dir.exists():
        print(f"  [WARN] eval_logs directory not found at {eval_dir}")
        return []
    # Filter out macOS hidden/resource fork files (._*)
    all_files = eval_dir.glob("eval_*_step_*.npz")
    npz_files = sorted([f for f in all_files if not f.name.startswith('._') and not f.name.startswith('.')])
    if not npz_files:
        print(f"  [WARN] No eval_*.npz files found in {eval_dir}")
        return []
    data: List[Tuple[int, float]] = []
    skipped = 0
    for npz_file in npz_files:
        try:
            arr = np.load(npz_file)
            if "episode_rewards" not in arr or "timestep" not in arr:
                skipped += 1
                continue
            rewards = arr["episode_rewards"].reshape(-1)
            timestep = int(arr["timestep"])
            data.append((timestep, float(np.mean(rewards))))
        except Exception as e:
            skipped += 1
            print(f"    [DEBUG] Failed to parse {npz_file.name}: {e}")
    data = sorted(data, key=lambda x: x[0])
    if skipped > 0:
        print(f"  [INFO] Skipped {skipped} corrupted/invalid .npz files")
    print(f"  [INFO] Built convergence data from eval_logs ({len(data)} points)")
    return data


def plot_convergence_curve(
    metric_data: List[Tuple[int, float]],
    output_path: Path,
    title: str = "Training Convergence",
    xlabel: str = "Timestep",
    ylabel: str = "Eval Mean Reward"
):
    """Plot convergence curve from evaluation metrics.
    
    Args:
        metric_data: List of (timestep, value) tuples
        output_path: Where to save the plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if not _HAS_MATPLOTLIB:
        print("  [WARN] matplotlib not available, skipping convergence plot")
        return
    
    if not metric_data:
        print("  [WARN] No metric data to plot")
        return
    
    steps, values = zip(*metric_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, values, linewidth=2, alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at final value
    final_value = values[-1]
    ax.axhline(y=final_value, color='red', linestyle='--', alpha=0.5, label=f'Final: {final_value:.3f}')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_multi_seed_convergence(
    per_seed_data: Dict[int, List[Tuple[int, float]]],
    output_path: Path,
    title: str = "Training Convergence (Multi-Seed)",
    xlabel: str = "Timestep",
    ylabel: str = "Eval Mean Reward"
):
    """Plot convergence curves for multiple seeds with mean and std band.
    
    Args:
        per_seed_data: Dict mapping seed -> list of (timestep, value) tuples
        output_path: Where to save the plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if not _HAS_MATPLOTLIB:
        print("  [WARN] matplotlib not available, skipping multi-seed convergence plot")
        return
    
    if not per_seed_data:
        print("  [WARN] No seed data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual seeds with low alpha
    for seed, data in sorted(per_seed_data.items()):
        if not data:
            continue
        steps, values = zip(*data)
        ax.plot(steps, values, alpha=0.3, linewidth=1, label=f'Seed {seed}')
    
    # Compute mean and std across seeds
    # First, align all seeds to common timesteps
    all_steps = sorted(set(step for data in per_seed_data.values() for step, _ in data))
    
    # Interpolate each seed to common timesteps
    seed_arrays = {}
    for seed, data in per_seed_data.items():
        if not data:
            continue
        steps, values = zip(*data)
        # Simple nearest-neighbor interpolation for missing points
        seed_arrays[seed] = np.interp(all_steps, steps, values)
    
    if seed_arrays:
        # Compute mean and std
        values_matrix = np.array(list(seed_arrays.values()))
        mean_values = np.mean(values_matrix, axis=0)
        std_values = np.std(values_matrix, axis=0)
        
        # Plot mean with std band
        ax.plot(all_steps, mean_values, color='black', linewidth=3, label='Mean', zorder=10)
        ax.fill_between(
            all_steps,
            mean_values - std_values,
            mean_values + std_values,
            color='black',
            alpha=0.2,
            label='±1 Std',
            zorder=5
        )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)




def find_npz_files(run_dir: Path, eval_logs_subdir: str = "eval_logs") -> List[Path]:
    """Find .npz files containing evaluation trajectories.
    
    Filters out macOS hidden files (._*) and other system files.
    """
    eval_dir = run_dir / eval_logs_subdir
    if not eval_dir.exists():
        return []
    # Filter out macOS hidden/resource fork files (._*) and other hidden files
    all_files = eval_dir.glob("*.npz")
    return sorted([f for f in all_files if not f.name.startswith('._') and not f.name.startswith('.')])


def load_npz_episodes(npz_path: Path) -> Dict[str, Any]:
    """
    Load a single trajectory .npz produced by DetailedEvalCallback._save_trajectories.

    Returns a dict with:
    - meta: {episode_rewards, episode_lengths, num_episodes, timestep, antibiotic_names}
    - episodes: list of episode dicts {patient_true, patient_observed, patient_attrs, actions, rewards, actual_amr_levels, visible_amr_levels}
    
    Raises:
        Exception: If file is corrupted or not a valid .npz file (caller should handle)
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        raise Exception(f"Failed to load {npz_path.name}: {e}")
    keys = list(data.keys())

    antibiotic_names = []
    if "antibiotic_names" in data:
        try:
            antibiotic_names = list(data["antibiotic_names"].tolist())
        except Exception:
            antibiotic_names = [str(x) for x in data["antibiotic_names"]]

    meta = {
        "episode_rewards": data["episode_rewards"].astype(float).tolist() if "episode_rewards" in data else [],
        "episode_lengths": data["episode_lengths"].astype(int).tolist() if "episode_lengths" in data else [],
        "num_episodes": int(data["num_episodes"]) if "num_episodes" in data else None,
        "timestep": int(data["timestep"]) if "timestep" in data else None,
        "antibiotic_names": antibiotic_names,
    }

    # Group episode keys by prefix 'episode_X/' where X is a number
    episode_prefixes = sorted({m.group(1) for k in keys for m in [re.match(r"^(episode_\d+)/", k)] if m})
    episodes: List[Dict[str, Any]] = []

    for ep_prefix in episode_prefixes:
        ep: Dict[str, Any] = {}
        # Optional patient data
        true_key = f"{ep_prefix}/patient_true"
        obs_key = f"{ep_prefix}/patient_observed"
        attrs_key = f"{ep_prefix}/patient_attrs"
        if true_key in data and obs_key in data:
            ep["patient_true"] = np.array(data[true_key])
            ep["patient_observed"] = np.array(data[obs_key])
            if attrs_key in data:
                try:
                    ep["patient_attrs"] = list(data[attrs_key].tolist())
                except Exception:
                    ep["patient_attrs"] = [str(x) for x in data[attrs_key]]
        # Actions and rewards (always saved)
        actions_key = f"{ep_prefix}/actions"
        rewards_key = f"{ep_prefix}/rewards"
        ep["actions"] = np.array(data[actions_key]) if actions_key in data else np.array([])
        ep["rewards"] = np.array(data[rewards_key]) if rewards_key in data else np.array([])

        actual_amr_key = f"{ep_prefix}/actual_amr_levels"
        visible_amr_key = f"{ep_prefix}/visible_amr_levels"
        if actual_amr_key in data:
            ep["actual_amr_levels"] = np.array(data[actual_amr_key])
        if visible_amr_key in data:
            ep["visible_amr_levels"] = np.array(data[visible_amr_key])

        if antibiotic_names:
            ep["antibiotic_names"] = list(antibiotic_names)
        episodes.append(ep)

    return {"meta": meta, "episodes": episodes, "source_file": str(npz_path)}


def compute_observation_error_metrics(all_runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-attribute observation error metrics: bias, MAE, RMSE.
    
    Returns dict: {attr_name: {bias, mae, rmse, samples}}
    """
    accum: Dict[str, Dict[str, Any]] = {}

    for run in all_runs:
        for ep in run["episodes"]:
            true = ep.get("patient_true")
            obs = ep.get("patient_observed")
            attrs = ep.get("patient_attrs") or []
            if true is None or obs is None or not attrs:
                continue
            steps, patients, num_attrs = true.shape
            true_flat = true.reshape(steps * patients, num_attrs)
            obs_flat = obs.reshape(steps * patients, num_attrs)
            err = obs_flat - true_flat
            for idx, name in enumerate(attrs):
                e = err[:, idx]
                if name not in accum:
                    accum[name] = {"sum": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "count": 0}
                accum[name]["sum"] += float(e.sum())
                accum[name]["sum_abs"] += float(np.abs(e).sum())
                accum[name]["sum_sq"] += float((e ** 2).sum())
                accum[name]["count"] += int(e.size)

    metrics: Dict[str, Dict[str, Any]] = {}
    for name, a in accum.items():
        c = max(a["count"], 1)
        bias = a["sum"] / c
        mae = a["sum_abs"] / c
        rmse = float(np.sqrt(a["sum_sq"] / c))
        metrics[name] = {"bias": bias, "mae": mae, "rmse": rmse, "samples": a["count"]}
    return metrics


def compute_reward_observation_error_correlations(all_runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute correlation between observation error and step rewards.
    
    Returns dict: {attr_name: {pearson, spearman, samples}}
    """
    accum: Dict[str, Dict[str, List[float]]] = {}

    for run in all_runs:
        for ep in run["episodes"]:
            true = ep.get("patient_true")
            obs = ep.get("patient_observed")
            attrs = ep.get("patient_attrs") or []
            rewards = ep.get("rewards")
            if true is None or obs is None or not attrs or rewards is None or rewards.size == 0:
                continue

            steps = true.shape[0]
            step_count = min(steps, int(rewards.shape[0]))
            for s in range(step_count):
                for idx, name in enumerate(attrs):
                    err_vec = obs[s, :, idx] - true[s, :, idx]
                    err_mean = float(np.mean(err_vec))
                    if name not in accum:
                        accum[name] = {"errors": [], "rewards": []}
                    accum[name]["errors"].append(err_mean)
                    accum[name]["rewards"].append(float(rewards[s]))

    results: Dict[str, Dict[str, Any]] = {}
    for name, series in accum.items():
        x = np.array(series["errors"], dtype=float)
        y = np.array(series["rewards"], dtype=float)
        n = int(min(x.size, y.size))
        if n < 2:
            results[name] = {"pearson": None, "spearman": None, "samples": n}
            continue

        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            pearson = None
        else:
            pearson = float(np.corrcoef(x, y)[0, 1])

        if _scipy_stats is not None and x.size > 1 and y.size > 1:
            try:
                spearman = float(_scipy_stats.spearmanr(x, y, nan_policy="omit").correlation)
            except Exception:
                spearman = None
        else:
            spearman = None

        results[name] = {"pearson": pearson, "spearman": spearman, "samples": n}

    return results


# ==================== CSV Writers ====================

def write_json(obj: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)


def write_csv_observation_error_metrics(metrics: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["attribute", "bias", "mae", "rmse", "samples"]
    lines = [",".join(headers)]
    for name, m in sorted(metrics.items()):
        line = f"{name},{m['bias']},{m['mae']},{m['rmse']},{m['samples']}"
        lines.append(line)
    out_path.write_text("\n".join(lines))


def write_csv_reward_error_correlations(corrs: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["attribute", "pearson", "spearman", "samples"]
    lines = [",".join(headers)]
    for name, m in sorted(corrs.items()):
        p = "" if m["pearson"] is None else m["pearson"]
        s = "" if m["spearman"] is None else m["spearman"]
        line = f"{name},{p},{s},{m['samples']}"
        lines.append(line)
    out_path.write_text("\n".join(lines))


def write_csv_per_seed_summary(
    prefix: str,
    per_seed_metrics: Dict[int, Dict[str, Any]],
    out_path: Path
):
    """
    Write per-seed diagnostic metrics to CSV.
    
    Columns: seed, obs_error_mae_mean, reward_error_pearson_mean, ...
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all metric keys from first seed
    if not per_seed_metrics:
        return
    
    first_seed_data = next(iter(per_seed_metrics.values()))
    
    # Build header: seed, then all metric names
    headers = ["seed", "mean_obs_error_mae", "mean_reward_error_pearson"]
    lines = [",".join(headers)]
    
    for seed in sorted(per_seed_metrics.keys()):
        data = per_seed_metrics[seed]
        mean_mae = data.get("mean_obs_error_mae", "")
        mean_pearson = data.get("mean_reward_error_pearson", "")
        line = f"{seed},{mean_mae},{mean_pearson}"
        lines.append(line)
    
    out_path.write_text("\n".join(lines))


# ==================== Per-Seed Aggregation ====================

def analyze_single_run(run_dir: Path, eval_logs_subdir: str = "eval_logs") -> Dict[str, Any]:
    """
    Analyze a single seed run and return aggregated metrics.
    """
    files = find_npz_files(run_dir, eval_logs_subdir)
    if not files:
        return {
            "error": f"No .npz files found in {run_dir}",
            "run_dir": str(run_dir),
        }
    
    # Load all npz files, skipping corrupted ones
    all_runs: List[Dict[str, Any]] = []
    skipped_files = 0
    for p in files:
        try:
            all_runs.append(load_npz_episodes(p))
        except Exception as e:
            skipped_files += 1
            print(f"  [WARN] Skipping corrupted file {p.name}: {e}")
    
    if not all_runs:
        return {
            "error": f"All .npz files in {run_dir} were corrupted or invalid",
            "run_dir": str(run_dir),
        }
    
    if skipped_files > 0:
        print(f"  [INFO] Successfully loaded {len(all_runs)}/{len(files)} .npz files ({skipped_files} skipped)")
    
    obs_error_metrics = compute_observation_error_metrics(all_runs)
    reward_error_corrs = compute_reward_observation_error_correlations(all_runs)
    
    # Compute means for aggregation
    mae_values = [m.get("mae", 0) for m in obs_error_metrics.values()]
    mean_mae = float(np.mean(mae_values)) if mae_values else None
    
    pearson_values = [c.get("pearson", 0) for c in reward_error_corrs.values() if c.get("pearson") is not None]
    mean_pearson = float(np.mean(pearson_values)) if pearson_values else None
    
    # Extract convergence data from TensorBoard logs with fallback to eval_logs
    eval_metrics = []
    try:
        eval_metrics = extract_eval_metrics(run_dir=run_dir, metric_name="eval/mean_reward")
    except Exception as e:
        print(f"  [WARN] Failed to extract metrics from TensorBoard logs: {e}")
    
    if not eval_metrics:
        print("  [INFO] Attempting fallback to eval_logs npz files for convergence data")
        try:
            eval_metrics = extract_eval_metrics_from_eval_logs(run_dir=run_dir, eval_logs_subdir=eval_logs_subdir)
            if eval_metrics:
                print(f"  [SUCCESS] Extracted {len(eval_metrics)} convergence points from eval_logs")
        except Exception as e:
            print(f"  [WARN] Fallback to eval_logs also failed: {e}")
    
    return {
        "run_dir": str(run_dir),
        "obs_error_metrics": obs_error_metrics,
        "reward_error_corrs": reward_error_corrs,
        "mean_obs_error_mae": mean_mae,
        "mean_reward_error_pearson": mean_pearson,
        "convergence_data": eval_metrics,  # Add convergence data
    }


# ====================== HRL-Specific Analysis ======================

def is_hrl_run(run_dir: Path) -> bool:
    """Check if a run is HRL by examining config.
    
    Checks for HRL algorithm in both flat and nested config structures:
    - config["algorithm"] (flat structure after load_config())
    - config["agent_algorithm"]["algorithm"] (nested structure)
    """
    config_path = run_dir / "full_agent_env_config.yaml"
    if not config_path.exists():
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
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


def analyze_hrl_single_run(run_dir: Path, max_eval_episodes: int = 5) -> Dict[str, Any]:
    """Run HRL-specific diagnostics for a single seed run.
    
    Generates diagnostic plots for manager behavior analysis.
    
    Returns dict with:
        - error: if analysis failed
        - hrl_diagnostics_dir: path where plots were saved
    """
    try:
        from abx_amr_simulator.utils import (
            create_reward_calculator,
            create_patient_generator,
            create_environment,
            load_config,
        )
        from abx_amr_simulator.hrl import OptionLibraryLoader
        from abx_amr_simulator.utils.metrics import plot_hrl_diagnostics_single_run
        from stable_baselines3 import PPO, DQN, A2C
    except ImportError as e:
        return {"error": f"Failed to import required modules: {e}"}
    
    # Load config
    config_path = run_dir / "full_agent_env_config.yaml"
    if not config_path.exists():
        return {"error": f"full_agent_env_config.yaml not found in {run_dir}"}
    
    try:
        config = load_config(str(config_path))
    except Exception as e:
        return {"error": f"Failed to load config: {e}"}
    
    # Create base environment FIRST (needed for option library loading)
    try:
        rc = create_reward_calculator(config=config)
        pg = create_patient_generator(config=config)
        base_env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
    except Exception as e:
        return {"error": f"Failed to create environment: {e}"}
    
    # Load option library (requires env for validation)
    try:
        # Try to use resolved option library config first (contains absolute paths)
        resolved_lib_config_path = run_dir / "full_options_library.yaml"
        if resolved_lib_config_path.exists():
            # Use the resolved config with absolute paths
            import yaml
            with open(resolved_lib_config_path, 'r') as f:
                resolved_config = yaml.safe_load(f)
            # Extract the library path from resolved config
            option_lib_path = resolved_config.get("library_config_path")
        else:
            # Fall back to regular config
            option_lib_path = config.get("hrl", {}).get("option_library", None)
        
        if not option_lib_path:
            return {"error": f"No HRL option_library specified in config"}
        
        loader = OptionLibraryLoader()
        option_library, _ = loader.load_library(library_config_path=option_lib_path, env=base_env)
    except Exception as e:
        return {"error": f"Failed to load option library: {e}"}
    
    # Wrap environment with options
    try:
        from abx_amr_simulator.hrl import OptionsWrapper
        env = OptionsWrapper(env=base_env, option_library=option_library)
    except Exception as e:
        return {"error": f"Failed to wrap environment: {e}"}
    
    # Load best model
    model_path = run_dir / "best_model.zip"
    if not model_path.exists():
        # Try checkpoints folder
        model_path = run_dir / "checkpoints" / "best_model.zip"
    
    if not model_path.exists():
        return {"error": f"best_model.zip not found in {run_dir} or {run_dir}/checkpoints"}
    
    try:
        # Infer model class from config (check flat structure first)
        algo_name = config.get("algorithm", "")
        if not algo_name:
            # Fall back to nested structure
            algo_name = config.get("agent_algorithm", {}).get("algorithm", "PPO")
        
        if "HRL_PPO" in algo_name:
            model_class = PPO
        else:
            model_class = PPO  # Default
        
        model = model_class.load(str(model_path), env=env)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}
    
    # Generate HRL diagnostics
    try:
        print(f"      Generating HRL diagnostics ({max_eval_episodes} episodes)...")
        plot_hrl_diagnostics_single_run(
            model=model,
            env=env,
            experiment_folder=str(run_dir),
            option_library=option_library,
            num_eval_episodes=max_eval_episodes,
            figures_folder_name="figures_hrl"
        )
        return {"hrl_diagnostics_dir": str(run_dir / "figures_hrl")}
    except Exception as e:
        return {"error": f"Failed to generate HRL diagnostics: {e}"}


def analyze_experiment(
    prefix: str,
    results_dir: str = "results",
    analysis_dir: str = "analysis_output",
    eval_logs_subdir: str = "eval_logs",
    aggregate_by_seed: bool = False,
) -> Tuple[Dict[int, Dict[str, Any]], bool, Optional[str]]:
    """
    Analyze experiment run(s) and return per-seed metrics.
    
    Behavior depends on aggregate_by_seed:
    - If aggregate_by_seed=False (default): Finds single run matching exact prefix.
      Analyzes that one run and returns metrics with seed -1 (indicating no seed).
    - If aggregate_by_seed=True: Finds all seed runs matching <prefix>_seed*_* pattern.
      Analyzes each seed separately and returns metrics keyed by seed number.
    
    Returns (per_seed_metrics, success) where success=True if analysis completed.
    """
    # Find runs based on aggregation mode
    runs = find_experiment_runs(
        prefix, 
        results_dir=results_dir, 
        aggregate_by_seed=aggregate_by_seed
    )
    if not runs:
        print(f"[WARN] No runs found for prefix: {prefix}")
        return {}, False, None
    
    if aggregate_by_seed:
        print(f"  Found {len(runs)} seed runs for {prefix}")
    else:
        if len(runs) > 1:
            print(f"  [WARN] Found {len(runs)} runs for prefix {prefix}. Using only the first one.")
            print(f"         (Use --aggregate-by-seed to analyze multiple seeds together)")
            runs = runs[:1]
        print(f"  Found 1 run for {prefix}")
    
    per_seed_metrics: Dict[int, Dict[str, Any]] = {}
    representative_timestamp: Optional[str] = None
    
    for run_dir in runs:
        # Extract seed number from folder name (if it matches seed pattern)
        match = re.search(r'_seed(\d+)_', run_dir.name)
        if match:
            seed = int(match.group(1))
        else:
            seed = -1  # No seed in name, use -1 to indicate single run
        
        # Capture timestamp from first run (representative for registry)
        if representative_timestamp is None:
            representative_timestamp = extract_timestamp_from_run_folder(run_dir.name)
        
        print(f"    Analyzing {'seed ' + str(seed) if seed >= 0 else 'run'}...")
        
        # Check if this is an HRL run
        is_hrl = is_hrl_run(run_dir)
        if is_hrl:
            print(f"      [INFO] Detected HRL agent - running HRL-specific diagnostics...")
            hrl_result = analyze_hrl_single_run(run_dir, max_eval_episodes=5)
            if "error" in hrl_result:
                print(f"      [WARN] HRL diagnostics failed: {hrl_result['error']}")
            else:
                print(f"      ✓ HRL diagnostics saved to {hrl_result['hrl_diagnostics_dir']}")
        
        # Check if training is complete: prefer per-run sentinel, fallback to root registry
        training_completed_file = run_dir / ".training_completed.txt"
        if not training_completed_file.exists():
            fallback_ok = False
            # Fallback: look for entry in root-level registry
            try:
                registry_file = Path(results_dir) / ".training_completed.txt"
                if registry_file.exists():
                    prefixes = load_registry(str(registry_file))
                    # Extract run_name without timestamp but including seed
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
        
        result = analyze_single_run(run_dir, eval_logs_subdir)
        if "error" in result:
            print(f"      [ERROR] {result['error']}")
            continue
        
        per_seed_metrics[seed] = result
    
    if not per_seed_metrics:
        print(f"  [ERROR] No runs analyzed successfully for {prefix}")
        return {}, False, None
    
    analysis_root = Path(analysis_dir)
    if analysis_root.name != "analysis_output":
        analysis_root = analysis_root / "analysis_output"
    analysis_root.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = analysis_root / prefix / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write per-seed and aggregate data
    write_csv_per_seed_summary(prefix, per_seed_metrics, output_dir / "summary_metrics.csv")
    
    # Collect convergence data for multi-seed plot
    per_seed_convergence = {}
    
    # Write individual metrics for each seed
    for seed, metrics in per_seed_metrics.items():
        if seed >= 0:
            seed_dir = output_dir / f"seed_{seed}"
        else:
            seed_dir = output_dir  # For non-seeded runs, save directly to output_dir
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        write_json(metrics["obs_error_metrics"], seed_dir / "observation_error_metrics.json")
        write_csv_observation_error_metrics(
            metrics["obs_error_metrics"],
            seed_dir / "observation_error_metrics.csv"
        )
        write_json(metrics["reward_error_corrs"], seed_dir / "reward_error_correlations.json")
        write_csv_reward_error_correlations(
            metrics["reward_error_corrs"],
            seed_dir / "reward_error_correlations.csv"
        )
        
        # Plot convergence curve
        convergence_data = metrics.get("convergence_data", [])
        if convergence_data:
            per_seed_convergence[seed] = convergence_data
            seed_label = f"seed {seed}" if seed >= 0 else "run"
            plot_convergence_curve(
                metric_data=convergence_data,
                output_path=seed_dir / "convergence_curve.png",
                title=f"Training Convergence - {prefix} ({seed_label})",
                xlabel="Timestep",
                ylabel="Eval Mean Reward"
            )
        else:
            print(f"  [WARN] No convergence data extracted for seed {seed}; skipping convergence plot")
    
    # Generate multi-seed convergence plot if we have multiple seeds
    if len(per_seed_convergence) > 1 and aggregate_by_seed:
        plot_multi_seed_convergence(
            per_seed_data=per_seed_convergence,
            output_path=output_dir / "convergence_curve_aggregated.png",
            title=f"Training Convergence - {prefix} (All Seeds)",
            xlabel="Timestep",
            ylabel="Eval Mean Reward"
        )
    
    print(f"  Saved diagnostics to {output_dir}")
    return per_seed_metrics, True, representative_timestamp


# ==================== Main Pipeline ====================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic analysis of training behavior and environment dynamics."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Manual mode: Analyze specific experiment. For single run: include full name with timestamp. For multi-seed: use base name with --aggregate-by-seed"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analysis of all exp_* experiments"
    )
    parser.add_argument(
        "--aggregate-by-seed",
        action="store_true",
        help="Group multiple seed runs together (looks for <prefix>_seed*_* pattern). Default: False (exact prefix match)"
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

    registry_path = str(analysis_root / ".diagnostic_analysis_completed.txt")
    
    if args.force:
        print("[INFO] --force flag: Clearing registry and re-analyzing all experiments")
        clear_registry(registry_path)
        prefixes_to_analyze = identify_new_experiments(registry_path, results_dir=args.results_dir)
        if not prefixes_to_analyze:
            # If registry cleared but no new experiments found, get all exp_ prefixes
            from abx_amr_simulator.utils import scan_for_experiments
            prefixes_to_analyze = scan_for_experiments(results_dir=args.results_dir)
    elif args.experiment_name:
        print(f"[INFO] Manual mode: Analyzing experiment '{args.experiment_name}'")
        prefixes_to_analyze = {args.experiment_name}
    else:
        print("[INFO] Auto mode: Scanning for new experiments...")
        prefixes_to_analyze = identify_new_experiments(registry_path, results_dir=args.results_dir)
    
    if not prefixes_to_analyze:
        print("[INFO] No experiments to analyze.")
        return
    
    print(f"[INFO] Found {len(prefixes_to_analyze)} experiment(s) to analyze")
    
    for prefix in sorted(prefixes_to_analyze):
        print(f"\nAnalyzing: {prefix}")
        per_seed_metrics, success, timestamp = analyze_experiment(
            prefix,
            results_dir=args.results_dir,
            analysis_dir=str(analysis_root),
            aggregate_by_seed=args.aggregate_by_seed,
        )
        
        if success and timestamp:
            update_registry_csv(registry_path, prefix, timestamp)
            print(f"✓ Updated registry for {prefix} (timestamp {timestamp})")
        elif success:
            print(f"✗ Analysis succeeded but could not extract timestamp from run folder for {prefix}")
        else:
            print(f"✗ Failed to analyze {prefix} - not adding to registry")
    
    print("\n[INFO] Diagnostic analysis complete!")


if __name__ == "__main__":
    main()
