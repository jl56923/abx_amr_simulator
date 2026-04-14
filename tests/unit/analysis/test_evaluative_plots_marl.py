from __future__ import annotations

import os
from pathlib import Path
import json

import numpy as np
import pytest

from abx_amr_simulator.analysis import evaluative_plots_marl as module

# Ensure headless plotting during tests.
os.environ.setdefault("MPLBACKEND", "Agg")


def _build_minimal_marl_npz(*, output_path: Path, antibiotic_names: list[str], n_patients: int = 2) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_abx = len(antibiotic_names)
    num_macro_steps = 2
    max_substeps = 2

    primitive_patient_true = np.zeros((num_macro_steps, max_substeps, n_patients, 1), dtype=float)
    primitive_patient_observed = np.zeros((num_macro_steps, max_substeps, n_patients, 1), dtype=float)

    primitive_individual_rewards = np.array(
        [
            [[1.0, 0.5], [0.25, -0.1]],
            [[0.2, 0.3], [0.0, 0.0]],
        ],
        dtype=float,
    )
    primitive_patients_infected = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=float,
    )

    primitive_substep_counts = np.array([2, 1], dtype=np.int32)

    primitive_actions = np.empty(num_macro_steps, dtype=object)
    primitive_actions[0] = np.array([[0, num_abx], [1, 2 if num_abx > 2 else num_abx]], dtype=int)
    primitive_actions[1] = np.array([[0, 1]], dtype=int)

    primitive_actual_amr = np.zeros((num_macro_steps, max_substeps, num_abx), dtype=float)
    primitive_visible_amr = np.zeros((num_macro_steps, max_substeps, num_abx), dtype=float)

    for macro_idx in range(num_macro_steps):
        for sub_idx in range(max_substeps):
            primitive_actual_amr[macro_idx, sub_idx, :] = np.linspace(0.1, 0.3, num_abx) + 0.05 * (macro_idx + sub_idx)
            primitive_visible_amr[macro_idx, sub_idx, :] = np.clip(
                primitive_actual_amr[macro_idx, sub_idx, :] - 0.02,
                a_min=0.0,
                a_max=1.0,
            )

    # Scalar per-substep reward-calculator fields. Pad substeps (index 1 of
    # macro_idx=1) are NaN to match the writer's pad-with-NaN semantics.
    def _scalar_array(values_macro_0: list[float], value_macro_1_substep_0: float) -> np.ndarray:
        arr = np.full((num_macro_steps, max_substeps), np.nan, dtype=float)
        arr[0, :] = values_macro_0
        arr[1, 0] = value_macro_1_substep_0
        return arr

    total_reward = _scalar_array([0.5, 0.25], 0.1)
    overall_individual = _scalar_array([1.5, 0.15], 0.5)
    normalized_individual = _scalar_array([0.75, 0.075], 0.25)
    overall_community = _scalar_array([-0.1, -0.2], -0.05)
    normalized_community = _scalar_array([-0.05, -0.1], -0.025)
    count_clinical_benefits = _scalar_array([1.0, 0.0], 1.0)
    count_clinical_failures = _scalar_array([0.0, 1.0], 0.0)
    count_adverse_events = _scalar_array([0.0, 0.0], 1.0)
    not_infected_no_treatment = _scalar_array([1.0, 0.0], 0.0)
    not_infected_treated = _scalar_array([0.0, 0.0], 0.0)
    infected_no_treatment = _scalar_array([0.0, 1.0], 0.0)

    save_kwargs: dict = {
        "episode_0/primitive_patient_true": primitive_patient_true,
        "episode_0/primitive_patient_observed": primitive_patient_observed,
        "episode_0/primitive_patient_attrs": np.array(["prob_infected"], dtype=object),
        "episode_0/primitive_individual_rewards": primitive_individual_rewards,
        "episode_0/primitive_patients_actually_infected": primitive_patients_infected,
        "episode_0/primitive_substep_counts": primitive_substep_counts,
        "episode_0/primitive_actions": primitive_actions,
        "episode_0/primitive_actual_amr_levels": primitive_actual_amr,
        "episode_0/primitive_visible_amr_levels": primitive_visible_amr,
        "episode_0/primitive_total_reward": total_reward,
        "episode_0/primitive_overall_individual_reward_component": overall_individual,
        "episode_0/primitive_normalized_individual_reward": normalized_individual,
        "episode_0/primitive_overall_community_reward_component": overall_community,
        "episode_0/primitive_normalized_community_reward": normalized_community,
        "episode_0/primitive_count_clinical_benefits": count_clinical_benefits,
        "episode_0/primitive_count_clinical_failures": count_clinical_failures,
        "episode_0/primitive_count_adverse_events": count_adverse_events,
        "episode_0/primitive_not_infected_no_treatment": not_infected_no_treatment,
        "episode_0/primitive_not_infected_treated": not_infected_treated,
        "episode_0/primitive_infected_no_treatment": infected_no_treatment,
    }

    # Per-antibiotic sensitive / resistant treated counts. Distribute
    # nonzero values across antibiotics so downstream aggregations are
    # non-degenerate.
    for abx_idx, abx_name in enumerate(antibiotic_names):
        sensitive = _scalar_array(
            [1.0 if abx_idx == 0 else 0.0, 0.0], 0.0
        )
        resistant = _scalar_array(
            [0.0, 1.0 if abx_idx == 1 else 0.0], 0.0
        )
        save_kwargs[
            f"episode_0/primitive_sensitive_infection_treated/{abx_name}"
        ] = sensitive
        save_kwargs[
            f"episode_0/primitive_resistant_infection_treated/{abx_name}"
        ] = resistant

    np.savez_compressed(
        file=str(output_path),
        antibiotic_names=np.array(antibiotic_names, dtype=object),
        num_episodes=1,
        **save_kwargs,
    )


def test_build_prefix_outputs_writes_per_agent_and_shared_artifacts(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    analysis_dir = tmp_path / "analysis"
    prefix = "exp_3e_cov50_auroc60__marl_mlc50"

    run_dir = results_dir / f"{prefix}_seed1"
    eval_dir = run_dir / "eval_logs"

    antibiotic_names = ["A", "B", "C"]
    _build_minimal_marl_npz(
        output_path=eval_dir / "eval_granular_best_model_agent_n.npz",
        antibiotic_names=antibiotic_names,
    )
    _build_minimal_marl_npz(
        output_path=eval_dir / "eval_granular_best_model_agent_p.npz",
        antibiotic_names=antibiotic_names,
    )

    module._build_prefix_outputs(
        prefix=prefix,
        run_dirs=[run_dir],
        analysis_dir=analysis_dir,
        force=True,
    )

    base_output = analysis_dir / prefix / "evaluation" / "evaluative_plots_marl"

    assert (base_output / "agent_n" / "outcome_counts_over_time.png").exists()
    assert (base_output / "agent_p" / "reward_components_over_time.png").exists()
    assert (base_output / "agent_n" / "clinical_benefits_failures_adverse_events_over_time.png").exists()
    assert (base_output / "agent_n" / "amr_levels_over_time.png").exists() is False
    assert (base_output / "shared" / "amr_levels_over_time.png").exists()

    summary_stats_path = base_output / "agent_n" / "overall_outcomes_summary_summary_stats.json"
    with open(summary_stats_path, "r", encoding="utf-8") as handle:
        summary_stats = json.load(handle)
    assert "overall_total_reward" in summary_stats
    assert set(summary_stats["overall_total_reward"].keys()) == {"p10", "p25", "p50", "p75", "p90"}

    # Fixture seeds 2 clinical benefits / 1 failure / 1 adverse event across
    # the episode — make sure these flow through rather than being the old
    # hard-coded zeros.
    assert summary_stats["overall_count_clinical_benefits"]["p50"] == pytest.approx(2.0)
    assert summary_stats["overall_count_clinical_failures"]["p50"] == pytest.approx(1.0)
    assert summary_stats["overall_count_adverse_events"]["p50"] == pytest.approx(1.0)
    # Resistant infections are populated on antibiotic B (fixture: 1 per
    # episode); they must no longer be hard-coded to zero.
    assert (
        summary_stats["overall_resistant_infection_treated_count_per_abx_dict_B"]["p50"]
        == pytest.approx(1.0)
    )


def test_build_prefix_outputs_writes_single_shared_plot_for_multi_seed_prefix(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    analysis_dir = tmp_path / "analysis"
    prefix = "exp_3e_cov50_auroc60__marl_mlc50"

    run_dir_1 = results_dir / f"{prefix}_seed1"
    run_dir_2 = results_dir / f"{prefix}_seed2"

    antibiotic_names = ["A", "B", "C"]
    _build_minimal_marl_npz(
        output_path=run_dir_1 / "eval_logs" / "eval_granular_best_model_agent_n.npz",
        antibiotic_names=antibiotic_names,
    )
    _build_minimal_marl_npz(
        output_path=run_dir_2 / "eval_logs" / "eval_granular_best_model_agent_n.npz",
        antibiotic_names=antibiotic_names,
    )

    module._build_prefix_outputs(
        prefix=prefix,
        run_dirs=[run_dir_1, run_dir_2],
        analysis_dir=analysis_dir,
        force=True,
    )

    shared_dir = analysis_dir / prefix / "evaluation" / "evaluative_plots_marl" / "shared"
    shared_plots = list(shared_dir.glob("amr_levels_over_time.png"))
    assert len(shared_plots) == 1


def test_build_agent_payload_fails_loud_on_missing_required_field(tmp_path: Path) -> None:
    npz_path = tmp_path / "eval_logs" / "eval_granular_best_model_agent_n.npz"
    _build_minimal_marl_npz(output_path=npz_path, antibiotic_names=["A", "B"])

    with np.load(file=str(npz_path), allow_pickle=True) as data:
        payload = {k: data[k] for k in data.files if k != "episode_0/primitive_visible_amr_levels"}

    np.savez_compressed(file=str(npz_path), **payload)

    with pytest.raises(ValueError, match="primitive_visible_amr_levels"):
        module._build_agent_trajectory_payload(npz_path=npz_path)
