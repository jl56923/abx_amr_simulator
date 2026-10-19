"""eng_14 (hrl-eval-first-step-dropped): HRL evaluation summaries cover every primitive step.

``_flatten_hrl_trajectory`` used to drop the reset info, so the ``infos[1:]`` slice every summary
consumer applies skipped the first real primitive step of each HRL episode: evaluation summaries
counted 499 of 500 steps for HRL agents and 500 of 500 for flat agents.

These tests run the real evaluation path end to end, with no monkeypatching of the rollout:
* a real ``ABXAMREnv`` built from the package's default configs (short episodes);
* the real default deterministic option library through ``OptionLibraryLoader`` and ``OptionsWrapper``;
* real (untrained) SB3 PPO models for the manager and the flat agent.

Policy quality is irrelevant; only the step accounting is under test.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from stable_baselines3 import PPO

from abx_amr_simulator.hrl import OptionLibraryLoader, OptionsWrapper, setup_options_folders_with_defaults
from abx_amr_simulator.utils import (
    create_environment,
    create_patient_generator,
    create_reward_calculator,
    load_config,
    setup_config_folders_with_defaults,
)
from abx_amr_simulator.utils.metrics import plot_metrics_ensemble_agents, run_episode_and_get_trajectory

MAX_TIME_STEPS = 60
PATIENT_COUNT_KEYS = (
    "overall_not_infected_no_treatment_count",
    "overall_not_infected_treated_count",
    "overall_infected_no_treatment_count",
    "overall_infected_treated_count",
)


@pytest.fixture
def workspace():
    """A temporary workspace scaffolded with the package's real default configs and option libraries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        experiments_dir = Path(tmpdir) / "experiments"
        experiments_dir.mkdir()
        setup_config_folders_with_defaults(target_path=experiments_dir)
        setup_options_folders_with_defaults(target_path=experiments_dir)
        yield experiments_dir


def _base_env(workspace: Path):
    config = load_config(config_path=str(workspace / "configs" / "umbrella_configs" / "base_experiment.yaml"))
    config["environment"]["max_time_steps"] = MAX_TIME_STEPS
    reward_calculator = create_reward_calculator(config=config)
    patient_generator = create_patient_generator(config=config)
    return create_environment(config=config, reward_calculator=reward_calculator, patient_generator=patient_generator)


def _hrl_env(workspace: Path):
    env = _base_env(workspace)
    library, _ = OptionLibraryLoader.load_library_from_env(
        library_config_path=str(workspace / "options" / "option_libraries" / "default_deterministic.yaml"),
        env=env,
    )
    return OptionsWrapper(env=env, option_library=library, gamma=0.99)


def _summary_patient_totals(model, env, tmp_path: Path, label: str) -> list[int]:
    plot_metrics_ensemble_agents(
        models=[model], env=env, experiment_folder=str(tmp_path / label), n_episodes_per_agent=2,
        deterministic=True, figures_folder_name="ensemble", per_seed_figures=False, episode_seed_start=0,
    )
    summary = json.loads((tmp_path / label / "ensemble" / "overall_outcomes_summary_raw_vals.json").read_text())
    return [sum(values) for values in zip(*(summary[key] for key in PATIENT_COUNT_KEYS))]


def test_flattened_hrl_trajectory_counts_every_primitive_step(workspace: Path) -> None:
    env = _hrl_env(workspace)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
    trajectory = run_episode_and_get_trajectory(model=model, env=env, deterministic=True, flatten_hrl=True)
    # Consumers skip infos[0] (the reset info); what remains must be every primitive step.
    assert len(trajectory["infos"][1:]) == MAX_TIME_STEPS
    assert len(trajectory["actions"]) == MAX_TIME_STEPS
    env.close()


def test_hrl_and_flat_summaries_count_the_same_patients(workspace: Path, tmp_path: Path) -> None:
    hrl_env = _hrl_env(workspace)
    hrl_model = PPO("MlpPolicy", hrl_env, n_steps=64, batch_size=32, verbose=0)
    cohort = hrl_env.unwrapped.num_patients_per_time_step
    hrl_totals = _summary_patient_totals(hrl_model, hrl_env, tmp_path, "hrl")

    flat_env = _base_env(workspace)
    flat_model = PPO("MlpPolicy", flat_env, n_steps=64, batch_size=32, verbose=0)
    flat_totals = _summary_patient_totals(flat_model, flat_env, tmp_path, "flat")

    expected = MAX_TIME_STEPS * cohort
    assert hrl_totals == [expected, expected], "HRL summary must count every primitive step"
    assert flat_totals == [expected, expected]
    hrl_env.close()
    flat_env.close()
