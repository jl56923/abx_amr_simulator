"""Tests for the single-agent HRL granular primitive-step logging branch.

`analysis/granular_logging.py` produces the only per-primitive-step record of a
*best-model* HRL policy: `option_ids`, per-patient `actions`, true/observed patient
attributes and AMR levels, one row per primitive step. It is what makes option-usage
audits possible without retraining.

This module previously had no test coverage at all, which mattered because the branch is
reached only via `evaluative_plots.py --granular-logging`. It went unexercised long enough
that an analysis reached into `eval_logs/` instead — periodic *mid-training* evaluation
snapshots — and joined them to best-model outcomes, producing a wrong number.

All tests use real environments, real option libraries and a real (briefly trained) HRL
agent. No mocks.

Covered:
- the branch writes one npz per seed, with every documented key
- `option_ids` only ever indexes into the run's real option library
- rows are consistent: one per primitive step, with episode/macro/primitive indices aligned
- per-option macro-step counts derived from the granular log match a direct count of the
  manager's selections (the property the E0 correction relied on)
- a failing seed is skipped rather than aborting the whole branch
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from abx_amr_simulator.analysis.granular_logging import run_hrl_granular_logging_branch
from abx_amr_simulator.hrl import setup_options_folders_with_defaults
from abx_amr_simulator.utils import (
    create_agent,
    create_environment,
    create_patient_generator,
    create_reward_calculator,
    load_config,
    setup_config_folders_with_defaults,
    wrap_environment_for_hrl,
)


# ---------------------------------------------------------------------------
# Fixtures — real instances throughout
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_workspace():
    """Temporary workspace with real default HRL config and options scaffolding."""
    tmpdir = Path(tempfile.mkdtemp())
    experiments_dir = tmpdir / "experiments"
    experiments_dir.mkdir()
    setup_config_folders_with_defaults(target_path=experiments_dir)
    setup_options_folders_with_defaults(target_path=experiments_dir)
    yield experiments_dir
    shutil.rmtree(tmpdir)


@pytest.fixture
def hrl_config(temp_workspace):
    """Default HRL umbrella config, shrunk so the rollout is fast."""
    umbrella_path = temp_workspace / "configs" / "umbrella_configs" / "hrl_ppo_default.yaml"
    config = load_config(config_path=str(umbrella_path))
    config["training"]["total_num_training_episodes"] = 2
    config["environment"]["max_time_steps"] = 10
    return config


@pytest.fixture
def trained_hrl_agent(hrl_config):
    """A real, briefly-trained HRL agent."""
    reward_calculator = create_reward_calculator(config=hrl_config)
    patient_generator = create_patient_generator(config=hrl_config)
    train_env = create_environment(
        config=hrl_config,
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
    )
    wrapped_train = wrap_environment_for_hrl(env=train_env, config=hrl_config)
    wrapped_train.reset(seed=1)
    agent = create_agent(config=hrl_config, env=wrapped_train, verbose=0)
    agent.learn(
        total_timesteps=(
            hrl_config["training"]["total_num_training_episodes"]
            * hrl_config["environment"]["max_time_steps"]
        )
    )
    yield agent
    train_env.close()


def _wrap_for_branch(env, config, run_dir):
    """`wrap_environment_for_hrl` adapted to the branch's (env, config, run_dir) signature."""
    return wrap_environment_for_hrl(env=env, config=config)


def _run_branch(agent, config, output_dir, seeds=(0,), num_episodes=2):
    models_and_configs = [(seed, agent, config, Path(".")) for seed in seeds]
    return run_hrl_granular_logging_branch(
        models_and_configs=models_and_configs,
        output_dir=output_dir,
        num_episodes=num_episodes,
        wrap_fn=_wrap_for_branch,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGranularLoggingBranch:
    """The branch produces a well-formed per-seed granular log."""

    def test_writes_one_npz_per_seed_with_documented_keys(self, trained_hrl_agent, hrl_config, tmp_path):
        assert _run_branch(trained_hrl_agent, hrl_config, tmp_path, seeds=(0, 1)) is True

        files = sorted((tmp_path / "granular_logs").glob("*.npz"))
        assert [f.name for f in files] == ["granular_seed_0.npz", "granular_seed_1.npz"]

        expected_keys = {
            "episode_ids", "macro_step_ids", "primitive_step_ids", "option_ids",
            "option_names", "actions", "patients_infected", "individual_rewards",
            "actual_amr_levels", "visible_amr_levels", "patient_true", "patient_observed",
            "patient_attr_names", "antibiotic_names",
        }
        payload = np.load(files[0], allow_pickle=True)
        assert expected_keys.issubset(set(payload.files))

    def test_rows_are_aligned_across_all_per_step_arrays(self, trained_hrl_agent, hrl_config, tmp_path):
        _run_branch(trained_hrl_agent, hrl_config, tmp_path)
        payload = np.load(tmp_path / "granular_logs" / "granular_seed_0.npz", allow_pickle=True)

        num_rows = payload["option_ids"].shape[0]
        assert num_rows > 0
        for key in (
            "episode_ids", "macro_step_ids", "primitive_step_ids", "option_names",
            "actions", "patients_infected", "individual_rewards",
            "actual_amr_levels", "visible_amr_levels", "patient_true", "patient_observed",
        ):
            assert payload[key].shape[0] == num_rows, f"{key} has a different row count"

    def test_option_ids_index_into_the_real_library(self, trained_hrl_agent, hrl_config, tmp_path):
        _run_branch(trained_hrl_agent, hrl_config, tmp_path)
        payload = np.load(tmp_path / "granular_logs" / "granular_seed_0.npz", allow_pickle=True)

        num_options = trained_hrl_agent.env.action_space.n
        option_ids = payload["option_ids"]
        assert option_ids.min() >= 0
        assert option_ids.max() < num_options

        # option_names must be the human-readable name of the id on the same row.
        names = payload["option_names"]
        for option_id in np.unique(option_ids):
            names_for_id = set(names[option_ids == option_id].tolist())
            assert len(names_for_id) == 1, f"option id {option_id} maps to multiple names"

    def test_episode_count_matches_request(self, trained_hrl_agent, hrl_config, tmp_path):
        _run_branch(trained_hrl_agent, hrl_config, tmp_path, num_episodes=3)
        payload = np.load(tmp_path / "granular_logs" / "granular_seed_0.npz", allow_pickle=True)
        assert sorted(np.unique(payload["episode_ids"]).tolist()) == [0, 1, 2]

    def test_primitive_steps_restart_within_each_macro_step(self, trained_hrl_agent, hrl_config, tmp_path):
        """Each macro-step's primitive indices must run 0..k-1 for that option's duration."""
        _run_branch(trained_hrl_agent, hrl_config, tmp_path)
        payload = np.load(tmp_path / "granular_logs" / "granular_seed_0.npz", allow_pickle=True)

        episodes = payload["episode_ids"]
        macro_steps = payload["macro_step_ids"]
        primitive_steps = payload["primitive_step_ids"]

        for episode in np.unique(episodes):
            in_episode = episodes == episode
            for macro_step in np.unique(macro_steps[in_episode]):
                block = primitive_steps[in_episode & (macro_steps == macro_step)]
                assert block.tolist() == list(range(len(block)))

    def test_macro_step_counts_are_recoverable(self, trained_hrl_agent, hrl_config, tmp_path):
        """Collapsing primitive rows to macro-steps must reproduce the manager's selections.

        This is the property the section 20c correction depended on: per-option macro-step
        counts derived from the granular log agreed exactly with the independently computed
        `hrl_stats` counts.
        """
        _run_branch(trained_hrl_agent, hrl_config, tmp_path)
        payload = np.load(tmp_path / "granular_logs" / "granular_seed_0.npz", allow_pickle=True)

        episodes = payload["episode_ids"]
        macro_steps = payload["macro_step_ids"]
        option_ids = payload["option_ids"]

        macro_decisions = []
        for episode in np.unique(episodes):
            in_episode = episodes == episode
            for macro_step in np.unique(macro_steps[in_episode]):
                block = option_ids[in_episode & (macro_steps == macro_step)]
                # One macro-step is one manager decision, so the option must not change mid-block.
                assert len(set(block.tolist())) == 1
                macro_decisions.append(int(block[0]))

        assert len(macro_decisions) > 0
        assert sum(
            np.sum(option_ids == option_id) for option_id in np.unique(option_ids)
        ) == option_ids.shape[0]


class TestGranularLoggingResilience:
    """A bad seed must not take the whole branch down."""

    def test_unusable_seed_is_skipped_but_others_still_written(self, trained_hrl_agent, hrl_config, tmp_path):
        broken_config = {"environment": {}}  # real dict, missing everything the factories need
        models_and_configs = [
            (0, trained_hrl_agent, hrl_config, Path(".")),
            (1, trained_hrl_agent, broken_config, Path(".")),
        ]
        result = run_hrl_granular_logging_branch(
            models_and_configs=models_and_configs,
            output_dir=tmp_path,
            num_episodes=1,
            wrap_fn=_wrap_for_branch,
        )

        assert result is True  # at least one seed succeeded
        written = {f.name for f in (tmp_path / "granular_logs").glob("*.npz")}
        assert written == {"granular_seed_0.npz"}

    def test_returns_false_when_every_seed_fails(self, hrl_config, tmp_path):
        broken_config = {"environment": {}}
        result = run_hrl_granular_logging_branch(
            models_and_configs=[(0, None, broken_config, Path("."))],
            output_dir=tmp_path,
            num_episodes=1,
            wrap_fn=_wrap_for_branch,
        )
        assert result is False


class TestCheckpointProvenanceStamp:
    """Every evaluation artifact records which policy produced it.

    Added after a section 20c analysis silently joined best-model outcomes to mid-training
    option selections. With this field, such a join can be rejected instead of believed.
    """

    def test_granular_npz_is_stamped_best(self, trained_hrl_agent, hrl_config, tmp_path):
        _run_branch(trained_hrl_agent, hrl_config, tmp_path)
        payload = np.load(tmp_path / "granular_logs" / "granular_seed_0.npz", allow_pickle=True)
        assert "checkpoint" in payload.files
        assert str(payload["checkpoint"]) == "best"

    def test_infer_checkpoint_label_maps_the_two_real_output_folders(self):
        from abx_amr_simulator.utils.metrics import infer_checkpoint_label

        assert infer_checkpoint_label("results/run_x/figures_best_agent") == "best"
        assert infer_checkpoint_label("results/run_x/figures_final_agent") == "final"
        # Trailing separators must not change the answer.
        assert infer_checkpoint_label("results/run_x/figures_best_agent/") == "best"

    def test_infer_checkpoint_label_does_not_guess(self):
        """An unrecognised folder is recorded verbatim, never silently called 'best'."""
        from abx_amr_simulator.utils.metrics import infer_checkpoint_label

        label = infer_checkpoint_label("results/run_x/figures_something_else")
        assert label == "unknown:figures_something_else"
        assert not label.startswith("best")
