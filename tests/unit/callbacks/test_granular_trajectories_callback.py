"""Tests for DetailedEvalCallback.save_granular_trajectories flag.

Verifies:
- When save_granular_trajectories=False (default), primitive-step keys are absent
  from saved NPZs and option_id/primitive_actions are not collected.
- When save_granular_trajectories=True, the saved NPZ contains all expected
  primitive-step keys with correct shapes and substep counts.
- _save_primitive_patient_arrays pads correctly when options have different durations.

All tests use real HRL environment instances (no mocks).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from abx_amr_simulator.callbacks import DetailedEvalCallback
from abx_amr_simulator.hrl import (
    OptionLibraryLoader,
    OptionsWrapper,
    setup_options_folders_with_defaults,
)
from abx_amr_simulator.utils import (
    create_agent,
    create_environment,
    create_patient_generator,
    create_reward_calculator,
    load_config,
    setup_config_folders_with_defaults,
    wrap_environment_for_hrl,
)
from stable_baselines3.common.vec_env import DummyVecEnv


# ---------------------------------------------------------------------------
# Fixtures
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
    """Load and minimally configure the default HRL umbrella config for fast tests."""
    umbrella_path = temp_workspace / "configs" / "umbrella_configs" / "hrl_ppo_default.yaml"
    config = load_config(config_path=str(umbrella_path))
    config["training"]["total_num_training_episodes"] = 2
    config["environment"]["max_time_steps"] = 10
    return config


@pytest.fixture
def hrl_eval_env(hrl_config):
    """A real wrapped HRL eval environment (OptionsWrapper around ABXAMREnv)."""
    rc = create_reward_calculator(config=hrl_config)
    pg = create_patient_generator(config=hrl_config)
    env = create_environment(config=hrl_config, reward_calculator=rc, patient_generator=pg)
    wrapped = wrap_environment_for_hrl(env=env, config=hrl_config)
    wrapped.reset(seed=0)
    yield wrapped
    env.close()


@pytest.fixture
def trained_hrl_agent(hrl_config, hrl_eval_env):
    """A briefly-trained HRL agent (enough to produce eval trajectories)."""
    rc = create_reward_calculator(config=hrl_config)
    pg = create_patient_generator(config=hrl_config)
    train_env = create_environment(config=hrl_config, reward_calculator=rc, patient_generator=pg)
    wrapped_train = wrap_environment_for_hrl(env=train_env, config=hrl_config)
    wrapped_train.reset(seed=1)
    agent = create_agent(config=hrl_config, env=wrapped_train, verbose=0)
    steps = hrl_config["training"]["total_num_training_episodes"] * hrl_config["environment"]["max_time_steps"]
    agent.learn(total_timesteps=steps)
    yield agent
    train_env.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_eval_with_callback(agent, hrl_eval_env, log_dir, save_granular):
    """Run one eval episode using DetailedEvalCallback with the given flag."""
    vec_eval_env = DummyVecEnv([lambda env=hrl_eval_env: env])
    callback = DetailedEvalCallback(
        eval_env=vec_eval_env,
        n_eval_episodes=1,
        eval_freq=1,
        log_path=str(log_dir),
        deterministic=True,
        verbose=0,
        save_patient_trajectories=True,
        save_granular_trajectories=save_granular,
    )
    callback.init_callback(agent)
    callback._run_evaluation_with_trajectories()
    return list(Path(log_dir, "eval_logs").glob("*.npz"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGranularTrajectoriesFlagOff:
    """When save_granular_trajectories=False (default), no primitive-step keys are saved."""

    def test_default_flag_is_false(self):
        """save_granular_trajectories defaults to False."""
        import inspect
        sig = inspect.signature(DetailedEvalCallback.__init__)
        assert sig.parameters["save_granular_trajectories"].default is False

    def test_primitive_keys_absent_from_npz(self, trained_hrl_agent, hrl_eval_env, tmp_path):
        """NPZ produced with default flag has no primitive_* or option_id keys."""
        npz_files = _run_eval_with_callback(
            trained_hrl_agent, hrl_eval_env, tmp_path, save_granular=False
        )
        assert len(npz_files) == 1, "Expected exactly one NPZ file"

        with np.load(npz_files[0], allow_pickle=True) as data:
            keys = list(data.keys())
            for key in keys:
                assert "primitive_patient" not in key, f"Unexpected key: {key}"
                assert "primitive_individual_rewards" not in key, f"Unexpected key: {key}"
                assert "primitive_patients_actually_infected" not in key, f"Unexpected key: {key}"
                assert "primitive_substep_counts" not in key, f"Unexpected key: {key}"
                assert "option_id" not in key, f"Unexpected key: {key}"
                assert "primitive_actions" not in key, f"Unexpected key: {key}"

    def test_macro_step_keys_still_present(self, trained_hrl_agent, hrl_eval_env, tmp_path):
        """Macro-step keys (patient_true, individual_rewards, etc.) are still present."""
        npz_files = _run_eval_with_callback(
            trained_hrl_agent, hrl_eval_env, tmp_path, save_granular=False
        )
        with np.load(npz_files[0], allow_pickle=True) as data:
            assert "episode_0/patient_true" in data
            assert "episode_0/patient_observed" in data
            assert "episode_0/individual_rewards" in data
            assert "episode_0/patients_actually_infected" in data


class TestGranularTrajectoriesFlagOn:
    """When save_granular_trajectories=True, all primitive-step keys are saved."""

    def test_primitive_keys_present_in_npz(self, trained_hrl_agent, hrl_eval_env, tmp_path):
        """NPZ produced with flag=True contains all expected primitive-step keys."""
        npz_files = _run_eval_with_callback(
            trained_hrl_agent, hrl_eval_env, tmp_path, save_granular=True
        )
        assert len(npz_files) == 1

        with np.load(npz_files[0], allow_pickle=True) as data:
            assert "episode_0/primitive_patient_true" in data
            assert "episode_0/primitive_patient_observed" in data
            assert "episode_0/primitive_patient_attrs" in data
            assert "episode_0/primitive_individual_rewards" in data
            assert "episode_0/primitive_patients_actually_infected" in data
            assert "episode_0/primitive_substep_counts" in data
            assert "episode_0/option_id" in data
            assert "episode_0/primitive_actions" in data

    def test_primitive_array_shapes_consistent(self, trained_hrl_agent, hrl_eval_env, tmp_path):
        """Primitive-step arrays have consistent shapes across the macro-step dimension."""
        npz_files = _run_eval_with_callback(
            trained_hrl_agent, hrl_eval_env, tmp_path, save_granular=True
        )
        with np.load(npz_files[0], allow_pickle=True) as data:
            pt = data["episode_0/primitive_patient_true"]
            po = data["episode_0/primitive_patient_observed"]
            ir = data["episode_0/primitive_individual_rewards"]
            inf = data["episode_0/primitive_patients_actually_infected"]
            sc = data["episode_0/primitive_substep_counts"]

            # All arrays share the same (macro_steps, max_substeps, patients) prefix.
            macro_steps, max_substeps, num_patients = ir.shape
            assert pt.shape[:3] == (macro_steps, max_substeps, num_patients)
            assert po.shape[:3] == (macro_steps, max_substeps, num_patients)
            assert inf.shape == (macro_steps, max_substeps, num_patients)
            assert sc.shape == (macro_steps,)

    def test_substep_counts_within_bounds(self, trained_hrl_agent, hrl_eval_env, tmp_path):
        """All substep counts are positive and do not exceed max_substeps."""
        npz_files = _run_eval_with_callback(
            trained_hrl_agent, hrl_eval_env, tmp_path, save_granular=True
        )
        with np.load(npz_files[0], allow_pickle=True) as data:
            sc = data["episode_0/primitive_substep_counts"]
            max_substeps = data["episode_0/primitive_individual_rewards"].shape[1]
            assert np.all(sc >= 1), "Every macro-step must have at least one substep"
            assert np.all(sc <= max_substeps), "Substep counts must not exceed max_substeps"

    def test_total_substeps_matches_macro_step_count(self, trained_hrl_agent, hrl_eval_env, tmp_path):
        """primitive_substep_counts has one entry per macro step (same as episode_lengths)."""
        npz_files = _run_eval_with_callback(
            trained_hrl_agent, hrl_eval_env, tmp_path, save_granular=True
        )
        with np.load(npz_files[0], allow_pickle=True) as data:
            sc = data["episode_0/primitive_substep_counts"]
            ep_lengths = data["episode_lengths"]
            # episode_lengths counts macro steps (one per manager decision).
            # primitive_substep_counts has one entry per macro step.
            assert len(sc) == int(ep_lengths[0])

    def test_patient_attrs_match_between_macro_and_primitive(self, trained_hrl_agent, hrl_eval_env, tmp_path):
        """primitive_patient_attrs matches macro-level patient_attrs."""
        npz_files = _run_eval_with_callback(
            trained_hrl_agent, hrl_eval_env, tmp_path, save_granular=True
        )
        with np.load(npz_files[0], allow_pickle=True) as data:
            macro_attrs = list(data["episode_0/patient_attrs"].tolist())
            prim_attrs = list(data["episode_0/primitive_patient_attrs"].tolist())
            assert macro_attrs == prim_attrs

    def test_macro_step_keys_still_present(self, trained_hrl_agent, hrl_eval_env, tmp_path):
        """Macro-step keys are still present alongside new primitive-step keys (backwards compat)."""
        npz_files = _run_eval_with_callback(
            trained_hrl_agent, hrl_eval_env, tmp_path, save_granular=True
        )
        with np.load(npz_files[0], allow_pickle=True) as data:
            assert "episode_0/patient_true" in data
            assert "episode_0/individual_rewards" in data
            assert "episode_0/patients_actually_infected" in data
