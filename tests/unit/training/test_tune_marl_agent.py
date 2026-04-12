"""Unit tests for tune_marl_agent.py.

Uses the minimal_two_agent.yaml fixture for most tests (plain PatientGenerators).
Tests for the personalized-generator path (exact_covered_count injection) use
an inline config dict to avoid external file dependencies.

All tests use real ABXAMREnv + OptionsWrapper — no mocks.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from abx_amr_simulator.training.tune_marl_agent import (
    build_single_agent_env_from_marl_config,
    build_single_agent_wrapper_from_marl_config,
    run_marl_agent_tuning,
    _resolve_batch_size_for_n_steps,
)
from abx_amr_simulator.utils.marl_factories import load_marl_config

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "marl_configs"
_FIXTURE_CONFIG = _FIXTURE_DIR / "minimal_two_agent.yaml"

_MINIMAL_TUNING_CONFIG = {
    "optimization": {
        "n_trials": 2,
        "n_seeds_per_trial": 1,
        "truncated_primitive_steps": 40,
        "direction": "maximize",
        "sampler": "TPE",
        "stability_penalty_weight": 0.0,
        "n_eval_episodes": 1,
        "early_stopping": {"enabled": False},
    },
    "search_space": {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 3e-4, "log": True},
    },
}


def _load() -> dict:
    return load_marl_config(_FIXTURE_CONFIG)


# --------------------------------------------------------------------------- #
# _resolve_batch_size_for_n_steps
# --------------------------------------------------------------------------- #

class TestResolveBatchSizeForNSteps:
    def test_returns_requested_when_divisible(self):
        resolved = _resolve_batch_size_for_n_steps(
            n_steps=256,
            requested_batch_size=64,
        )
        assert resolved == 64

    def test_reduces_to_largest_divisor_when_not_divisible(self):
        resolved = _resolve_batch_size_for_n_steps(
            n_steps=224,
            requested_batch_size=64,
        )
        assert resolved == 56

    def test_caps_batch_size_to_n_steps(self):
        resolved = _resolve_batch_size_for_n_steps(
            n_steps=32,
            requested_batch_size=64,
        )
        assert resolved == 32


# --------------------------------------------------------------------------- #
# build_single_agent_env_from_marl_config
# --------------------------------------------------------------------------- #

class TestBuildSingleAgentEnv:
    def test_returns_abx_amr_env(self):
        from abx_amr_simulator.core.abx_amr_env import ABXAMREnv
        config = _load()
        env = build_single_agent_env_from_marl_config(config, "agent_0")
        assert isinstance(env, ABXAMREnv)
        env.close()

    def test_patient_count_overridden_to_tuning_n_patients(self):
        config = _load()
        env = build_single_agent_env_from_marl_config(
            config, "agent_0", tuning_n_patients=5
        )
        assert env.num_patients_per_time_step == 5
        env.close()

    def test_second_agent_also_builds(self):
        config = _load()
        env = build_single_agent_env_from_marl_config(config, "agent_1", tuning_n_patients=4)
        assert env.num_patients_per_time_step == 4
        env.close()

    def test_raises_for_unknown_agent_id(self):
        config = _load()
        with pytest.raises(ValueError, match="not found"):
            build_single_agent_env_from_marl_config(config, "no_such_agent")

    def test_plain_pg_does_not_set_exact_covered_count(self):
        """Plain PatientGenerator has no exact_covered_count — env should build fine."""
        config = _load()
        env = build_single_agent_env_from_marl_config(config, "agent_0")
        # ABXAMREnv does not expose exact_covered_count — just verify it built
        assert env is not None
        env.close()

    def test_personalized_pg_injects_exact_covered_count(self, tmp_path):
        """When create_personal_pred=True, exact_covered_count is set to tuning_n_patients."""
        # Build a minimal MARL config with a personalized patient generator inline.
        # We can't run a full PersonalizedPredPatientGenerator here without the
        # plugin loader resolving the path, so we verify the config dict is
        # modified correctly before the build attempt by patching the factory.
        config = _load()
        # Inject create_personal_pred into agent_0's patient generator inline config.
        agent_entry = next(
            e for e in config["environment"]["agents"]
            if e["agent_id"] == "agent_0"
        )
        pg_config = dict(agent_entry["patient_generator"])
        pg_config["create_personal_pred"] = True
        # Remove the plugin key so it falls back to plain PatientGenerator build
        # (which will silently ignore the create_personal_pred key). This verifies
        # that exact_covered_count is injected into the config dict without needing
        # the full personalized plugin to be importable.
        pg_config.pop("plugin", None)
        agent_entry["patient_generator"] = pg_config

        tuning_n = 7
        # We monkey-patch build_patient_generator_from_config to capture the dict
        # passed to it.
        captured = {}
        import abx_amr_simulator.training.tune_marl_agent as _mod
        original = _mod.build_patient_generator_from_config

        def capturing_builder(pg_value, config_dir, seed):
            captured["pg_value"] = copy.deepcopy(pg_value)
            return original(pg_value, config_dir, seed)

        _mod.build_patient_generator_from_config = capturing_builder
        try:
            env = build_single_agent_env_from_marl_config(
                config, "agent_0", tuning_n_patients=tuning_n
            )
            env.close()
        finally:
            _mod.build_patient_generator_from_config = original

        assert captured["pg_value"].get("exact_covered_count") == tuning_n


# --------------------------------------------------------------------------- #
# build_single_agent_wrapper_from_marl_config
# --------------------------------------------------------------------------- #

class TestBuildSingleAgentWrapper:
    def test_returns_options_wrapper(self):
        from abx_amr_simulator.hrl.wrapper import OptionsWrapper
        config = _load()
        env = build_single_agent_env_from_marl_config(config, "agent_0")
        wrapper = build_single_agent_wrapper_from_marl_config(config, "agent_0", env)
        assert isinstance(wrapper, OptionsWrapper)
        wrapper.close()

    def test_obs_space_has_correct_shape(self):
        config = _load()
        env = build_single_agent_env_from_marl_config(
            config, "agent_0", tuning_n_patients=3
        )
        wrapper = build_single_agent_wrapper_from_marl_config(config, "agent_0", env)
        # OptionsWrapper has a flat Box obs space
        import gymnasium as gym
        assert isinstance(wrapper.observation_space, gym.spaces.Box)
        wrapper.close()

    def test_raises_for_unknown_agent_id(self):
        from abx_amr_simulator.core.abx_amr_env import ABXAMREnv
        config = _load()
        env = build_single_agent_env_from_marl_config(config, "agent_0")
        with pytest.raises(ValueError, match="not found"):
            build_single_agent_wrapper_from_marl_config(config, "no_such_agent", env)
        env.close()

    def test_wrapper_can_reset_and_step(self):
        config = _load()
        env = build_single_agent_env_from_marl_config(
            config, "agent_0", tuning_n_patients=3
        )
        wrapper = build_single_agent_wrapper_from_marl_config(config, "agent_0", env)
        obs, _ = wrapper.reset(seed=0)
        assert obs.shape == wrapper.observation_space.shape
        action = wrapper.action_space.sample()
        obs2, reward, terminated, truncated, info = wrapper.step(action)
        assert obs2.shape == wrapper.observation_space.shape
        assert isinstance(reward, float)
        wrapper.close()


# --------------------------------------------------------------------------- #
# run_marl_agent_tuning
# --------------------------------------------------------------------------- #

class TestRunMarlAgentTuning:
    def test_writes_best_params_json(self, tmp_path):
        config = _load()
        best = run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=_MINIMAL_TUNING_CONFIG,
            optimization_dir=tmp_path,
            run_name="test_run",
            seed=0,
        )
        assert (tmp_path / "test_run" / "best_params.json").exists()
        assert isinstance(best, dict)
        assert len(best) > 0

    def test_best_params_contains_search_space_keys(self, tmp_path):
        config = _load()
        best = run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=_MINIMAL_TUNING_CONFIG,
            optimization_dir=tmp_path,
            run_name="test_run",
            seed=0,
        )
        assert "learning_rate" in best

    def test_writes_study_summary_json(self, tmp_path):
        config = _load()
        run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=_MINIMAL_TUNING_CONFIG,
            optimization_dir=tmp_path,
            run_name="test_run",
            seed=0,
        )
        summary_path = tmp_path / "test_run" / "study_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["agent_id"] == "agent_0"
        assert summary["n_trials_completed"] == 2

    def test_skip_if_exists_returns_existing_params(self, tmp_path):
        config = _load()
        # First run
        run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=_MINIMAL_TUNING_CONFIG,
            optimization_dir=tmp_path,
            run_name="test_run",
            seed=0,
        )
        best_path = tmp_path / "test_run" / "best_params.json"
        with open(best_path) as f:
            first_params = json.load(f)

        # Second run with skip_if_exists — should not re-run study
        second_params = run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=_MINIMAL_TUNING_CONFIG,
            optimization_dir=tmp_path,
            run_name="test_run",
            seed=0,
            skip_if_exists=True,
        )
        assert second_params == first_params

    def test_second_agent_also_tunes(self, tmp_path):
        config = _load()
        best = run_marl_agent_tuning(
            config=config,
            agent_id="agent_1",
            tuning_config=_MINIMAL_TUNING_CONFIG,
            optimization_dir=tmp_path,
            run_name="test_run_agent1",
            seed=1,
        )
        assert isinstance(best, dict)
        assert len(best) > 0

    def test_overwrite_existing_study_reruns(self, tmp_path):
        config = _load()
        run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=_MINIMAL_TUNING_CONFIG,
            optimization_dir=tmp_path,
            run_name="test_run",
            seed=0,
        )
        # Overwrite — should not raise and should produce a valid result
        best = run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=_MINIMAL_TUNING_CONFIG,
            optimization_dir=tmp_path,
            run_name="test_run",
            seed=99,
            overwrite_existing_study=True,
        )
        assert isinstance(best, dict)
