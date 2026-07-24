"""Real-instance tests for the fixed-prescribing (FP) MARL evaluator.

These exercise the full code path — a real ``ABXAMRParallelEnv`` built from the
minimal two-agent fixture config, real reward calculators and patient generators,
and the real ``expected_reward_greedy`` policy — and assert the granular NPZ schema.

The focus is the per-antibiotic ``primitive_sensitive_infection_treated/{abx}`` and
``primitive_resistant_infection_treated/{abx}`` keys: the DS-effective equity metric
needs them for BOTH covered and uncovered agents, and the FP evaluator must log them
in the same ``(macro_steps, max_substeps)`` shape as the trained-MARL granular eval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from abx_amr_simulator.analysis.eval_fixed_prescribing_marl import (
    run_fp_marl_eval,
    _SCALAR_INFO_KEYS,
    _OUTCOME_SCALAR_KEYS,
)

FIXTURE_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "marl_configs"
    / "minimal_two_agent.yaml"
)


def _run(tmp_path: Path):
    return run_fp_marl_eval(
        marl_config_path=str(FIXTURE_CONFIG),
        output_dir=str(tmp_path),
        policy_name="expected_reward_greedy",
        seed=1,
        n_eval_episodes=1,
    )


def test_fp_eval_logs_per_antibiotic_treated_counts_for_every_agent(tmp_path):
    written = _run(tmp_path)
    assert set(written) == {"agent_0", "agent_1"}

    for agent_id, npz_path in written.items():
        with np.load(npz_path, allow_pickle=True) as data:
            antibiotic_names = [str(n) for n in data["antibiotic_names"].tolist()]
            assert antibiotic_names, f"{agent_id}: no antibiotic_names recorded"

            substep_counts = data["episode_0/primitive_substep_counts"]
            num_steps = substep_counts.shape[0]
            # FP acts at the primitive level: exactly one substep per macro step.
            assert np.all(substep_counts == 1)

            for abx in antibiotic_names:
                sens_key = f"episode_0/primitive_sensitive_infection_treated/{abx}"
                res_key = f"episode_0/primitive_resistant_infection_treated/{abx}"
                assert sens_key in data, f"{agent_id}: missing {sens_key}"
                assert res_key in data, f"{agent_id}: missing {res_key}"

                sens = np.asarray(data[sens_key], dtype=float)
                res = np.asarray(data[res_key], dtype=float)
                # Shape must match the trained-MARL schema: (macro_steps, max_substeps).
                assert sens.shape == (num_steps, 1), f"{agent_id}/{abx}: {sens.shape}"
                assert res.shape == (num_steps, 1), f"{agent_id}/{abx}: {res.shape}"
                # Counts are non-negative integers stored as floats.
                assert np.all(sens >= 0) and np.all(res >= 0)


def test_fp_eval_logs_full_trained_marl_primitive_schema(tmp_path):
    """The FP evaluator must emit the same primitive schema the trained-MARL granular
    analyses (evaluative_plots_marl, equity) require: shared AMR levels plus per-substep
    scalar reward/outcome fields, each shaped like the trained-MARL (steps, substeps)."""
    written = _run(tmp_path)

    for agent_id, npz_path in written.items():
        with np.load(npz_path, allow_pickle=True) as data:
            antibiotic_names = [str(n) for n in data["antibiotic_names"].tolist()]
            num_abx = len(antibiotic_names)
            num_steps = data["episode_0/primitive_substep_counts"].shape[0]

            # Shared AMR levels: (num_steps, 1, num_abx).
            for amr_key in (
                "episode_0/primitive_actual_amr_levels",
                "episode_0/primitive_visible_amr_levels",
            ):
                assert amr_key in data, f"{agent_id}: missing {amr_key}"
                arr = np.asarray(data[amr_key], dtype=float)
                assert arr.shape == (num_steps, 1, num_abx), f"{agent_id}/{amr_key}: {arr.shape}"

            # Scalar reward + outcome tallies: (num_steps, 1).
            for key in (*_SCALAR_INFO_KEYS, *_OUTCOME_SCALAR_KEYS):
                full_key = f"episode_0/primitive_{key}"
                assert full_key in data, f"{agent_id}: missing {full_key}"
                arr = np.asarray(data[full_key], dtype=float)
                assert arr.shape == (num_steps, 1), f"{agent_id}/{full_key}: {arr.shape}"
                assert np.all(np.isfinite(arr)), f"{agent_id}/{full_key}: non-finite values"


def test_fp_treated_counts_are_bounded_by_patients_and_consistent(tmp_path):
    """Per-step treated counts across all antibiotics cannot exceed the infected count."""
    written = _run(tmp_path)

    for agent_id, npz_path in written.items():
        with np.load(npz_path, allow_pickle=True) as data:
            antibiotic_names = [str(n) for n in data["antibiotic_names"].tolist()]
            infected = np.asarray(
                data["episode_0/primitive_patients_actually_infected"], dtype=float
            )
            # infected has shape (steps, 1, patients); infected patients per step.
            infected_per_step = infected.reshape(infected.shape[0], -1).sum(axis=1)

            treated_per_step = np.zeros(infected.shape[0], dtype=float)
            for abx in antibiotic_names:
                sens = np.asarray(
                    data[f"episode_0/primitive_sensitive_infection_treated/{abx}"],
                    dtype=float,
                ).reshape(-1)
                res = np.asarray(
                    data[f"episode_0/primitive_resistant_infection_treated/{abx}"],
                    dtype=float,
                ).reshape(-1)
                treated_per_step += sens + res

            # Only infected patients can be "infection_treated"; treated <= infected.
            assert np.all(treated_per_step <= infected_per_step + 1e-9), (
                f"{agent_id}: treated infections exceed infected patients"
            )
