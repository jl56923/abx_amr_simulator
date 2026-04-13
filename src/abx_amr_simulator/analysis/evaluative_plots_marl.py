from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for evaluative plotting") from exc

from abx_amr_simulator.utils.metrics import aggregate_trajectories, plot_with_bands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate general-purpose MARL evaluative plots from per-agent granular NPZs."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing MARL run folders.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        required=True,
        help=(
            "Directory where outputs are written as "
            "<analysis-dir>/<prefix>/evaluation/evaluative_plots_marl/."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing artifacts.",
    )
    parser.add_argument(
        "--prefix-filter",
        type=str,
        default=None,
        help="Optional regex filter applied to experiment prefix names.",
    )
    return parser.parse_args()


def discover_prefix_runs(*, results_dir: Path, prefix_filter: Optional[str]) -> Dict[str, List[Path]]:
    pattern = re.compile(pattern=r"^(exp_.+?)_seed\d+(?:_\d+.*)?$")
    prefix_regex = re.compile(pattern=prefix_filter) if prefix_filter is not None else None

    grouped: Dict[str, List[Path]] = {}
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        is_marl = (
            (run_dir / "marl_full_agents_env_config.yaml").exists()
            or len(list((run_dir / "eval_logs").glob("eval_granular_best_model_*.npz"))) > 0
            or len(list((run_dir / "checkpoints").glob("final_model_agent_*.zip"))) > 0
        )
        if not is_marl:
            continue

        match = pattern.match(string=run_dir.name)
        if match is None:
            continue

        prefix = match.group(1)
        if prefix_regex is not None and prefix_regex.search(prefix) is None:
            continue

        grouped.setdefault(prefix, []).append(run_dir)
    return grouped


def discover_agent_npz_files(*, run_dir: Path) -> Dict[str, Path]:
    eval_dir = run_dir / "eval_logs"
    if not eval_dir.exists():
        return {}

    pattern = re.compile(pattern=r"^eval_granular_best_model_(?P<agent>[^.]+)\.npz$")
    found: Dict[str, Path] = {}
    for path in sorted(eval_dir.glob(pattern="eval_granular_best_model_*.npz")):
        match = pattern.match(string=path.name)
        if match is None:
            continue
        found[match.group("agent")] = path
    return found


def _episode_indices_from_keys(*, keys: Sequence[str]) -> List[int]:
    indices: List[int] = []
    pattern = re.compile(pattern=r"^episode_(\d+)/")
    for key in keys:
        match = pattern.match(string=key)
        if match is None:
            continue
        indices.append(int(match.group(1)))
    return sorted(set(indices))


def _flatten_substeps(
    *,
    primitive_values: np.ndarray,
    primitive_substep_counts: np.ndarray,
) -> np.ndarray:
    slices: List[np.ndarray] = []
    for macro_idx in range(len(primitive_substep_counts)):
        num_substeps = int(primitive_substep_counts[macro_idx])
        for substep_idx in range(num_substeps):
            slices.append(primitive_values[macro_idx, substep_idx])
    if len(slices) == 0:
        raise ValueError("No primitive substeps found while flattening arrays")
    return np.stack(slices)


def _aggregate_lines(*, trajectories: List[List[float]], apply_cumsum: bool) -> Dict[str, np.ndarray]:
    if len(trajectories) == 0:
        raise ValueError("Cannot aggregate empty trajectory list")
    return aggregate_trajectories(trajectories_list=trajectories, apply_cumsum=apply_cumsum)


def _save_json(*, output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8") as handle:
        json.dump(obj=payload, fp=handle, indent=2)


def _build_agent_trajectory_payload(*, npz_path: Path) -> Tuple[List[str], Dict[str, Any]]:
    with np.load(file=npz_path, allow_pickle=True) as data:
        if "antibiotic_names" not in data:
            raise ValueError(f"Missing antibiotic_names in artifact {npz_path}")
        antibiotic_names = [str(name) for name in data["antibiotic_names"].tolist()]
        if len(antibiotic_names) == 0:
            raise ValueError(f"Empty antibiotic_names in artifact {npz_path}")

        episode_indices = _episode_indices_from_keys(keys=list(data.keys()))
        if len(episode_indices) == 0:
            raise ValueError(f"No episode data found in artifact {npz_path}")

        payload: Dict[str, Any] = {
            "total_reward": [],
            "individual_reward": [],
            "normalized_individual_reward": [],
            "community_reward": [],
            "normalized_community_reward": [],
            "count_clinical_benefits": [],
            "count_clinical_failures": [],
            "count_adverse_events": [],
            "not_infected_no_treatment": [],
            "not_infected_treated": [],
            "infected_no_treatment": [],
            "count_prescriptions": {abx: [] for abx in antibiotic_names},
            "infected_treated_sensitive": {abx: [] for abx in antibiotic_names},
            "infected_treated_resistant": {abx: [] for abx in antibiotic_names},
            "actual_AMR_levels": {abx: [] for abx in antibiotic_names},
            "visible_AMR_levels": {abx: [] for abx in antibiotic_names},
        }

        no_treatment_index = len(antibiotic_names)

        required_keys = [
            "primitive_individual_rewards",
            "primitive_patients_actually_infected",
            "primitive_substep_counts",
            "primitive_actions",
            "primitive_actual_amr_levels",
            "primitive_visible_amr_levels",
        ]

        for episode_idx in episode_indices:
            ep_prefix = f"episode_{episode_idx}"
            for key_suffix in required_keys:
                full_key = f"{ep_prefix}/{key_suffix}"
                if full_key not in data:
                    raise ValueError(
                        f"Missing required field '{full_key}' in artifact {npz_path}"
                    )

            primitive_rewards = np.asarray(data[f"{ep_prefix}/primitive_individual_rewards"], dtype=float)
            primitive_infected = np.asarray(
                data[f"{ep_prefix}/primitive_patients_actually_infected"], dtype=float
            )
            primitive_substep_counts = np.asarray(
                data[f"{ep_prefix}/primitive_substep_counts"], dtype=int
            )
            primitive_actions = data[f"{ep_prefix}/primitive_actions"]
            primitive_actual_amr = np.asarray(
                data[f"{ep_prefix}/primitive_actual_amr_levels"], dtype=float
            )
            primitive_visible_amr = np.asarray(
                data[f"{ep_prefix}/primitive_visible_amr_levels"], dtype=float
            )

            rewards = _flatten_substeps(
                primitive_values=primitive_rewards,
                primitive_substep_counts=primitive_substep_counts,
            )
            infected = _flatten_substeps(
                primitive_values=primitive_infected,
                primitive_substep_counts=primitive_substep_counts,
            ) > 0.5
            actual_amr = _flatten_substeps(
                primitive_values=primitive_actual_amr,
                primitive_substep_counts=primitive_substep_counts,
            )
            visible_amr = _flatten_substeps(
                primitive_values=primitive_visible_amr,
                primitive_substep_counts=primitive_substep_counts,
            )

            action_slices: List[np.ndarray] = []
            for macro_idx in range(len(primitive_substep_counts)):
                macro_actions = np.asarray(primitive_actions[macro_idx], dtype=int)
                num_substeps = int(primitive_substep_counts[macro_idx])
                for substep_idx in range(num_substeps):
                    action_slices.append(macro_actions[substep_idx])
            if len(action_slices) == 0:
                raise ValueError(f"No primitive actions found in {npz_path} ({ep_prefix})")
            actions_per_patient = np.stack(action_slices)

            if actions_per_patient.shape != infected.shape:
                raise ValueError(
                    "Shape mismatch between actions and infected arrays: "
                    f"{actions_per_patient.shape} vs {infected.shape}"
                )

            total_reward_step = np.sum(rewards, axis=1)
            mean_reward_step = np.mean(rewards, axis=1)
            payload["total_reward"].append(total_reward_step.tolist())
            payload["individual_reward"].append(total_reward_step.tolist())
            payload["normalized_individual_reward"].append(mean_reward_step.tolist())
            payload["community_reward"].append(np.zeros_like(total_reward_step).tolist())
            payload["normalized_community_reward"].append(np.zeros_like(total_reward_step).tolist())

            payload["count_clinical_benefits"].append(np.zeros_like(total_reward_step).tolist())
            payload["count_clinical_failures"].append(np.zeros_like(total_reward_step).tolist())
            payload["count_adverse_events"].append(np.zeros_like(total_reward_step).tolist())

            not_infected = ~infected
            no_treatment = actions_per_patient == no_treatment_index
            payload["not_infected_no_treatment"].append(
                np.sum(not_infected & no_treatment, axis=1).astype(float).tolist()
            )
            payload["not_infected_treated"].append(
                np.sum(not_infected & ~no_treatment, axis=1).astype(float).tolist()
            )
            payload["infected_no_treatment"].append(
                np.sum(infected & no_treatment, axis=1).astype(float).tolist()
            )

            for abx_idx, abx_name in enumerate(antibiotic_names):
                prescribed = actions_per_patient == abx_idx
                infected_and_treated = infected & prescribed
                payload["count_prescriptions"][abx_name].append(
                    np.sum(prescribed, axis=1).astype(float).tolist()
                )
                payload["infected_treated_sensitive"][abx_name].append(
                    np.sum(infected_and_treated, axis=1).astype(float).tolist()
                )
                payload["infected_treated_resistant"][abx_name].append(
                    np.zeros(infected_and_treated.shape[0], dtype=float).tolist()
                )
                payload["actual_AMR_levels"][abx_name].append(actual_amr[:, abx_idx].tolist())
                payload["visible_AMR_levels"][abx_name].append(visible_amr[:, abx_idx].tolist())

        return antibiotic_names, payload


def _write_agent_plots(
    *,
    output_dir: Path,
    antibiotic_names: List[str],
    payload: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_individual = _aggregate_lines(trajectories=payload["individual_reward"], apply_cumsum=True)
    agg_total = _aggregate_lines(trajectories=payload["total_reward"], apply_cumsum=True)

    plt.figure(figsize=(12, 5))
    plot_with_bands(
        ax=plt.gca(),
        data_dict=agg_individual,
        label="Individual reward (cumulative)",
        add_iqr_legend=False,
    )
    plot_with_bands(
        ax=plt.gca(),
        data_dict=agg_total,
        label="Total reward (cumulative)",
        add_iqr_legend=False,
    )
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative reward")
    plt.title("Reward Components Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_components_over_time.png")
    plt.close()

    agg_not_infected_no_treatment = _aggregate_lines(
        trajectories=payload["not_infected_no_treatment"], apply_cumsum=True
    )
    agg_not_infected_treated = _aggregate_lines(
        trajectories=payload["not_infected_treated"], apply_cumsum=True
    )
    agg_infected_no_treatment = _aggregate_lines(
        trajectories=payload["infected_no_treatment"], apply_cumsum=True
    )

    infected_treated_overall: List[List[float]] = []
    n_traj = len(payload["not_infected_no_treatment"])
    for traj_idx in range(n_traj):
        first_abx = antibiotic_names[0]
        total = np.zeros(
            shape=len(payload["infected_treated_sensitive"][first_abx][traj_idx]),
            dtype=float,
        )
        for abx_name in antibiotic_names:
            total += np.asarray(payload["infected_treated_sensitive"][abx_name][traj_idx], dtype=float)
            total += np.asarray(payload["infected_treated_resistant"][abx_name][traj_idx], dtype=float)
        infected_treated_overall.append(total.tolist())
    agg_infected_treated_overall = _aggregate_lines(
        trajectories=infected_treated_overall,
        apply_cumsum=True,
    )

    plt.figure(figsize=(12, 5))
    plot_with_bands(
        ax=plt.gca(),
        data_dict=agg_not_infected_no_treatment,
        label="Not infected, no treatment",
        add_iqr_legend=False,
    )
    plot_with_bands(
        ax=plt.gca(),
        data_dict=agg_not_infected_treated,
        label="Not infected, treated",
        add_iqr_legend=False,
    )
    plot_with_bands(
        ax=plt.gca(),
        data_dict=agg_infected_no_treatment,
        label="Infected, no treatment",
        add_iqr_legend=False,
    )
    plot_with_bands(
        ax=plt.gca(),
        data_dict=agg_infected_treated_overall,
        label="Infected, treated",
        add_iqr_legend=False,
    )
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative count")
    plt.title("Outcome Counts Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "outcome_counts_over_time.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    for abx_name in antibiotic_names:
        agg = _aggregate_lines(
            trajectories=payload["count_prescriptions"][abx_name],
            apply_cumsum=True,
        )
        plot_with_bands(
            ax=plt.gca(),
            data_dict=agg,
            label=f"{abx_name} prescriptions",
            add_iqr_legend=False,
        )
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative prescriptions")
    plt.title("Antibiotic Prescriptions Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "abx_prescriptions_over_time.png")
    plt.close()

    for abx_name in antibiotic_names:
        agg_sensitive = _aggregate_lines(
            trajectories=payload["infected_treated_sensitive"][abx_name],
            apply_cumsum=True,
        )
        agg_resistant = _aggregate_lines(
            trajectories=payload["infected_treated_resistant"][abx_name],
            apply_cumsum=True,
        )

        plt.figure(figsize=(10, 5))
        plot_with_bands(
            ax=plt.gca(),
            data_dict=agg_sensitive,
            label=f"{abx_name} treated infected",
            add_iqr_legend=False,
        )
        plot_with_bands(
            ax=plt.gca(),
            data_dict=agg_resistant,
            label=f"{abx_name} treated resistant (unavailable -> 0)",
            add_iqr_legend=False,
        )
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative count")
        plt.title(f"Infected Treated Counts: {abx_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"infected_treated_counts_{abx_name}_over_time.png")
        plt.close()

    summary_raw = {
        "num_trajectories": len(payload["individual_reward"]),
        "antibiotic_names": antibiotic_names,
        "limitations": [
            "community and normalized community reward are not emitted in current MARL granular NPZs; plotted as zeros",
            "clinical benefit/failure/adverse-event counts are not emitted in current MARL granular NPZs; omitted from dedicated plot",
            "treated-resistant counts are not emitted in current MARL granular NPZs; plotted as zeros",
        ],
    }
    _save_json(output_path=output_dir / "overall_outcomes_summary_raw_vals.json", payload=summary_raw)

    summary_stats: Dict[str, Any] = {
        "final_individual_reward_median": float(agg_individual["median"][-1]),
        "final_total_reward_median": float(agg_total["median"][-1]),
    }
    for abx_name in antibiotic_names:
        agg_prescriptions = _aggregate_lines(
            trajectories=payload["count_prescriptions"][abx_name],
            apply_cumsum=True,
        )
        summary_stats[f"final_prescriptions_{abx_name}_median"] = float(
            agg_prescriptions["median"][-1]
        )
    _save_json(
        output_path=output_dir / "overall_outcomes_summary_summary_stats.json",
        payload=summary_stats,
    )


def _collect_shared_amr_payload(*, agent_npz_paths: List[Path]) -> Tuple[List[str], Dict[str, List[List[float]]]]:
    if len(agent_npz_paths) == 0:
        raise ValueError("Cannot build shared AMR payload from zero NPZ files")

    antibiotic_names_ref: Optional[List[str]] = None
    shared_payload: Dict[str, List[List[float]]] = {}

    for npz_path in agent_npz_paths:
        with np.load(file=npz_path, allow_pickle=True) as data:
            if "antibiotic_names" not in data:
                raise ValueError(f"Missing antibiotic_names in artifact {npz_path}")
            antibiotic_names = [str(name) for name in data["antibiotic_names"].tolist()]
            if antibiotic_names_ref is None:
                antibiotic_names_ref = antibiotic_names
                shared_payload = {
                    f"actual::{abx}": [] for abx in antibiotic_names
                }
                shared_payload.update(
                    {f"visible::{abx}": [] for abx in antibiotic_names}
                )
            elif antibiotic_names != antibiotic_names_ref:
                raise ValueError(
                    "Antibiotic name mismatch across shared AMR sources: "
                    f"{antibiotic_names_ref} vs {antibiotic_names}"
                )

            episode_indices = _episode_indices_from_keys(keys=list(data.keys()))
            if len(episode_indices) == 0:
                raise ValueError(f"No episode data found in artifact {npz_path}")

            for episode_idx in episode_indices:
                ep_prefix = f"episode_{episode_idx}"
                substep_counts_key = f"{ep_prefix}/primitive_substep_counts"
                actual_key = f"{ep_prefix}/primitive_actual_amr_levels"
                visible_key = f"{ep_prefix}/primitive_visible_amr_levels"
                if substep_counts_key not in data or actual_key not in data or visible_key not in data:
                    raise ValueError(
                        f"Missing shared AMR fields in artifact {npz_path} ({ep_prefix})"
                    )

                substep_counts = np.asarray(data[substep_counts_key], dtype=int)
                actual = _flatten_substeps(
                    primitive_values=np.asarray(data[actual_key], dtype=float),
                    primitive_substep_counts=substep_counts,
                )
                visible = _flatten_substeps(
                    primitive_values=np.asarray(data[visible_key], dtype=float),
                    primitive_substep_counts=substep_counts,
                )

                for abx_idx, abx_name in enumerate(antibiotic_names):
                    shared_payload[f"actual::{abx_name}"].append(actual[:, abx_idx].tolist())
                    shared_payload[f"visible::{abx_name}"].append(visible[:, abx_idx].tolist())

    if antibiotic_names_ref is None:
        raise ValueError("Failed to infer antibiotic names while collecting shared AMR")

    return antibiotic_names_ref, shared_payload


def _write_shared_amr_plot(
    *,
    output_dir: Path,
    antibiotic_names: List[str],
    shared_payload: Dict[str, List[List[float]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for abx_name in antibiotic_names:
        agg = _aggregate_lines(
            trajectories=shared_payload[f"actual::{abx_name}"],
            apply_cumsum=False,
        )
        plot_with_bands(
            ax=plt.gca(),
            data_dict=agg,
            label=f"{abx_name} actual AMR",
            add_iqr_legend=False,
        )
    plt.xlabel("Timestep")
    plt.ylabel("AMR level")
    plt.title("Shared Actual AMR Levels Over Time")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for abx_name in antibiotic_names:
        agg = _aggregate_lines(
            trajectories=shared_payload[f"visible::{abx_name}"],
            apply_cumsum=False,
        )
        plot_with_bands(
            ax=plt.gca(),
            data_dict=agg,
            label=f"{abx_name} visible AMR",
            add_iqr_legend=False,
        )
    plt.xlabel("Timestep")
    plt.ylabel("AMR level")
    plt.title("Shared Visible AMR Levels Over Time")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "amr_levels_over_time.png")
    plt.close()


def _build_prefix_outputs(
    *,
    prefix: str,
    run_dirs: Sequence[Path],
    analysis_dir: Path,
    force: bool,
) -> None:
    all_agents: List[str] = []
    run_to_agents: Dict[Path, Dict[str, Path]] = {}

    for run_dir in run_dirs:
        mapping = discover_agent_npz_files(run_dir=run_dir)
        run_to_agents[run_dir] = mapping
        for agent_id in sorted(mapping.keys()):
            if agent_id not in all_agents:
                all_agents.append(agent_id)

    if len(all_agents) == 0:
        print(f"[SKIP] {prefix}: no per-agent eval_granular_best_model_*.npz files found")
        return

    base_output_dir = analysis_dir / prefix / "evaluation" / "evaluative_plots_marl"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for agent_id in sorted(all_agents):
        agent_output_dir = base_output_dir / f"agent_{agent_id}"
        summary_path = agent_output_dir / "overall_outcomes_summary_summary_stats.json"
        if summary_path.exists() and not force:
            print(f"[SKIP] {prefix} / agent_{agent_id}: outputs exist (use --force)")
            continue

        agent_npzs: List[Path] = []
        for run_dir in run_dirs:
            maybe_path = run_to_agents[run_dir].get(agent_id)
            if maybe_path is not None:
                agent_npzs.append(maybe_path)

        if len(agent_npzs) == 0:
            print(f"[SKIP] {prefix} / agent_{agent_id}: no matching NPZs across seeds")
            continue

        merged_antibiotic_names: Optional[List[str]] = None
        merged_payload: Optional[Dict[str, Any]] = None
        for npz_path in agent_npzs:
            antibiotic_names, payload = _build_agent_trajectory_payload(npz_path=npz_path)
            if merged_antibiotic_names is None:
                merged_antibiotic_names = antibiotic_names
                merged_payload = payload
            else:
                if antibiotic_names != merged_antibiotic_names:
                    raise ValueError(
                        f"Antibiotic names mismatch for prefix {prefix}, agent {agent_id}: "
                        f"{merged_antibiotic_names} vs {antibiotic_names}"
                    )
                if merged_payload is None:
                    raise ValueError("Internal error: merged_payload unexpectedly None")

                for key in [
                    "total_reward",
                    "individual_reward",
                    "normalized_individual_reward",
                    "community_reward",
                    "normalized_community_reward",
                    "count_clinical_benefits",
                    "count_clinical_failures",
                    "count_adverse_events",
                    "not_infected_no_treatment",
                    "not_infected_treated",
                    "infected_no_treatment",
                ]:
                    merged_payload[key].extend(payload[key])

                for abx_name in merged_antibiotic_names:
                    merged_payload["count_prescriptions"][abx_name].extend(
                        payload["count_prescriptions"][abx_name]
                    )
                    merged_payload["infected_treated_sensitive"][abx_name].extend(
                        payload["infected_treated_sensitive"][abx_name]
                    )
                    merged_payload["infected_treated_resistant"][abx_name].extend(
                        payload["infected_treated_resistant"][abx_name]
                    )
                    merged_payload["actual_AMR_levels"][abx_name].extend(
                        payload["actual_AMR_levels"][abx_name]
                    )
                    merged_payload["visible_AMR_levels"][abx_name].extend(
                        payload["visible_AMR_levels"][abx_name]
                    )

        if merged_antibiotic_names is None or merged_payload is None:
            raise ValueError(
                f"Failed to build merged payload for prefix={prefix}, agent={agent_id}"
            )

        _write_agent_plots(
            output_dir=agent_output_dir,
            antibiotic_names=merged_antibiotic_names,
            payload=merged_payload,
        )
        print(f"[DONE] {prefix} / agent_{agent_id}: wrote evaluative plots")

    shared_output_dir = base_output_dir / "shared"
    shared_plot_path = shared_output_dir / "amr_levels_over_time.png"
    if shared_plot_path.exists() and not force:
        print(f"[SKIP] {prefix} / shared: AMR plot exists (use --force)")
        return

    canonical_agent_npzs: List[Path] = []
    for run_dir in run_dirs:
        per_agent = run_to_agents[run_dir]
        if len(per_agent) == 0:
            continue
        canonical_agent_id = sorted(per_agent.keys())[0]
        canonical_agent_npzs.append(per_agent[canonical_agent_id])

    if len(canonical_agent_npzs) == 0:
        raise ValueError(
            f"Cannot build shared AMR plot for {prefix}: no canonical per-seed NPZ files found"
        )

    antibiotic_names, shared_payload = _collect_shared_amr_payload(
        agent_npz_paths=canonical_agent_npzs
    )
    _write_shared_amr_plot(
        output_dir=shared_output_dir,
        antibiotic_names=antibiotic_names,
        shared_payload=shared_payload,
    )
    print(f"[DONE] {prefix} / shared: wrote AMR plot")


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = Path(args.analysis_dir)

    if not results_dir.exists() or not results_dir.is_dir():
        raise ValueError(f"--results-dir is not a directory: {results_dir}")

    grouped = discover_prefix_runs(
        results_dir=results_dir,
        prefix_filter=args.prefix_filter,
    )
    if len(grouped) == 0:
        print("No MARL run folders found for provided filters.")
        return

    done_prefixes = 0
    failed_prefixes = 0
    for prefix, run_dirs in sorted(grouped.items()):
        try:
            _build_prefix_outputs(
                prefix=prefix,
                run_dirs=run_dirs,
                analysis_dir=analysis_dir,
                force=bool(args.force),
            )
            done_prefixes += 1
        except Exception as exc:
            failed_prefixes += 1
            print(f"[ERROR] {prefix}: {exc}")

    total_prefixes = len(grouped)
    print(
        "\nMARL evaluative plotting complete: "
        f"done={done_prefixes}, errors={failed_prefixes}, total={total_prefixes}"
    )


if __name__ == "__main__":
    main()
