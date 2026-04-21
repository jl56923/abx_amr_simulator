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
    _build_tuning_agent,
    _compute_distributed_worker_quota,
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


class TestComputeDistributedWorkerQuota:
    def test_even_split(self):
        q0 = _compute_distributed_worker_quota(
            n_trials=6,
            worker_id=0,
            total_workers=3,
        )
        q1 = _compute_distributed_worker_quota(
            n_trials=6,
            worker_id=1,
            total_workers=3,
        )
        q2 = _compute_distributed_worker_quota(
            n_trials=6,
            worker_id=2,
            total_workers=3,
        )
        assert q0["quota"] == 2
        assert q1["quota"] == 2
        assert q2["quota"] == 2

    def test_remainder_goes_to_early_workers(self):
        q0 = _compute_distributed_worker_quota(
            n_trials=5,
            worker_id=0,
            total_workers=2,
        )
        q1 = _compute_distributed_worker_quota(
            n_trials=5,
            worker_id=1,
            total_workers=2,
        )
        assert q0["quota"] == 3
        assert q1["quota"] == 2
        assert q0["end"] == q1["start"]


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

    def test_raises_on_invalid_worker_id(self, tmp_path):
        config = _load()
        with pytest.raises(ValueError, match="worker_id"):
            run_marl_agent_tuning(
                config=config,
                agent_id="agent_0",
                tuning_config=_MINIMAL_TUNING_CONFIG,
                optimization_dir=tmp_path,
                run_name="bad_worker",
                worker_id=2,
                total_workers=2,
            )

    def test_distributed_two_workers_reaches_target_trials(self, tmp_path):
        config = _load()
        tuning_cfg = copy.deepcopy(_MINIMAL_TUNING_CONFIG)
        tuning_cfg["optimization"]["n_trials"] = 2

        run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=tuning_cfg,
            optimization_dir=tmp_path,
            run_name="dist_run",
            seed=0,
            worker_id=0,
            total_workers=2,
        )
        run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=tuning_cfg,
            optimization_dir=tmp_path,
            run_name="dist_run",
            seed=0,
            worker_id=1,
            total_workers=2,
        )

        summary_path = tmp_path / "dist_run" / "study_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["n_trials_completed"] == 2

    def test_distributed_workers_suggest_distinct_hyperparameters(self, tmp_path):
        """Each worker should explore different hyperparameters, not identical ones.

        This is a regression test for the bug where all distributed workers
        created TPESampler(seed=<same_seed>), causing every worker to suggest
        identical parameters during the startup random-sampling phase.
        """
        import optuna

        config = _load()
        # Use a wider search space so distinct seeds produce visibly different values.
        tuning_cfg = copy.deepcopy(_MINIMAL_TUNING_CONFIG)
        tuning_cfg["optimization"]["n_trials"] = 4
        tuning_cfg["search_space"] = {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "gamma": {"type": "float", "low": 0.9, "high": 0.999},
            "clip_range": {"type": "float", "low": 0.05, "high": 0.4},
        }

        # Run 4 workers sequentially, each contributing 1 trial.
        for wid in range(4):
            run_marl_agent_tuning(
                config=config,
                agent_id="agent_0",
                tuning_config=tuning_cfg,
                optimization_dir=tmp_path,
                run_name="distinct_hp_run",
                seed=42,
                worker_id=wid,
                total_workers=4,
            )

        # Load the study and inspect trial params.
        db_path = tmp_path / "distinct_hp_run" / "optuna_study.db"
        study = optuna.load_study(
            study_name="distinct_hp_run",
            storage=f"sqlite:///{db_path}",
        )
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        assert len(completed) == 4

        # Collect all parameter dicts and verify they are not all identical.
        param_sets = [tuple(sorted(t.params.items())) for t in completed]
        unique_param_sets = set(param_sets)
        assert len(unique_param_sets) > 1, (
            f"All {len(completed)} distributed workers produced identical "
            f"hyperparameters: {completed[0].params}. "
            "Each worker's TPE sampler should use a different seed."
        )


# --------------------------------------------------------------------------- #
# Algorithm dispatch: _build_tuning_agent and end-to-end HRL_RPPO tuning
# --------------------------------------------------------------------------- #

def _build_wrapper_for_agent(agent_id: str):
    """Helper: build a fresh single-agent wrapper from the fixture config."""
    config = _load()
    env = build_single_agent_env_from_marl_config(
        config=config,
        agent_id=agent_id,
        tuning_n_patients=3,
    )
    wrapper = build_single_agent_wrapper_from_marl_config(
        config=config,
        agent_id=agent_id,
        env=env,
    )
    return wrapper


class TestBuildTuningAgent:
    """Direct tests of the algorithm-dispatch helper.

    Uses real ABXAMREnv + OptionsWrapper instances (sociable testing)
    and inspects the resulting SB3 agent's type and policy attributes.
    """

    _PARAMS = {
        "learning_rate": 3e-4,
        "n_epochs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
    }

    def test_hrl_ppo_returns_ppo_instance(self):
        from stable_baselines3 import PPO
        wrapper = _build_wrapper_for_agent("agent_0")
        try:
            agent = _build_tuning_agent(
                algorithm="HRL_PPO",
                wrapper=wrapper,
                params=self._PARAMS,
                resolved_n_steps=16,
                resolved_batch_size=4,
                seed=0,
                lstm_kwargs={},
                net_arch=None,
            )
            assert isinstance(agent, PPO)
        finally:
            wrapper.close()

    def test_hrl_rppo_returns_recurrent_ppo_masked_instance(self):
        from abx_amr_simulator.hrl.rl_algorithms.recurrent_ppo_masked import (
            RecurrentPPO_Masked,
        )
        wrapper = _build_wrapper_for_agent("agent_0")
        try:
            agent = _build_tuning_agent(
                algorithm="HRL_RPPO",
                wrapper=wrapper,
                params=self._PARAMS,
                resolved_n_steps=16,
                resolved_batch_size=4,
                seed=0,
                lstm_kwargs={"lstm_hidden_size": 8, "n_lstm_layers": 1},
                net_arch=None,
            )
            assert isinstance(agent, RecurrentPPO_Masked)
        finally:
            wrapper.close()

    def test_hrl_rppo_honors_lstm_hidden_size(self):
        wrapper = _build_wrapper_for_agent("agent_0")
        try:
            agent = _build_tuning_agent(
                algorithm="HRL_RPPO",
                wrapper=wrapper,
                params=self._PARAMS,
                resolved_n_steps=16,
                resolved_batch_size=4,
                seed=0,
                lstm_kwargs={"lstm_hidden_size": 32, "n_lstm_layers": 2},
                net_arch=None,
            )
            assert agent.policy.lstm_actor.hidden_size == 32
            assert agent.policy.lstm_actor.num_layers == 2
        finally:
            wrapper.close()

    def test_hrl_rppo_honors_net_arch(self):
        wrapper = _build_wrapper_for_agent("agent_0")
        try:
            agent = _build_tuning_agent(
                algorithm="HRL_RPPO",
                wrapper=wrapper,
                params=self._PARAMS,
                resolved_n_steps=16,
                resolved_batch_size=4,
                seed=0,
                lstm_kwargs={"lstm_hidden_size": 8, "n_lstm_layers": 1},
                net_arch=[7, 11],
            )
            # SB3 stores policy_kwargs back onto the agent.
            assert agent.policy_kwargs.get("net_arch") == [7, 11]
        finally:
            wrapper.close()

    def test_hrl_ppo_honors_net_arch(self):
        wrapper = _build_wrapper_for_agent("agent_0")
        try:
            agent = _build_tuning_agent(
                algorithm="HRL_PPO",
                wrapper=wrapper,
                params=self._PARAMS,
                resolved_n_steps=16,
                resolved_batch_size=4,
                seed=0,
                lstm_kwargs={},
                net_arch=[7, 11],
            )
            assert agent.policy_kwargs.get("net_arch") == [7, 11]
        finally:
            wrapper.close()

    def test_hrl_ppo_net_arch_absent_when_none(self):
        """When net_arch is None, policy_kwargs is not populated with a net_arch key."""
        wrapper = _build_wrapper_for_agent("agent_0")
        try:
            agent = _build_tuning_agent(
                algorithm="HRL_PPO",
                wrapper=wrapper,
                params=self._PARAMS,
                resolved_n_steps=16,
                resolved_batch_size=4,
                seed=0,
                lstm_kwargs={},
                net_arch=None,
            )
            # SB3 normalises policy_kwargs to {} when nothing was passed.
            assert "net_arch" not in (agent.policy_kwargs or {})
        finally:
            wrapper.close()

    def test_unsupported_algorithm_raises(self):
        wrapper = _build_wrapper_for_agent("agent_0")
        try:
            with pytest.raises(ValueError, match="unsupported algorithm"):
                _build_tuning_agent(
                    algorithm="NOT_AN_ALGO",
                    wrapper=wrapper,
                    params=self._PARAMS,
                    resolved_n_steps=16,
                    resolved_batch_size=4,
                    seed=0,
                    lstm_kwargs={},
                    net_arch=None,
                )
        finally:
            wrapper.close()


class TestRunMarlAgentTuningHrlRppo:
    """End-to-end tuning run with an HRL_RPPO agent config entry.

    Patches the fixture config in-memory to set ``algorithm: HRL_RPPO`` +
    ``lstm_kwargs`` + ``policy_kwargs.net_arch`` on agent_0, then verifies
    that ``run_marl_agent_tuning`` completes and writes a best_params.json.
    """

    def test_hrl_rppo_tuning_runs_to_completion(self, tmp_path):
        config = _load()
        agent_entry = next(
            e for e in config["environment"]["agents"]
            if e["agent_id"] == "agent_0"
        )
        agent_entry["algorithm"] = "HRL_RPPO"
        agent_entry["lstm_kwargs"] = {
            "lstm_hidden_size": 8,
            "n_lstm_layers": 1,
            "enable_critic_lstm": True,
        }
        agent_entry["policy_kwargs"] = {"net_arch": [8, 8]}

        best = run_marl_agent_tuning(
            config=config,
            agent_id="agent_0",
            tuning_config=_MINIMAL_TUNING_CONFIG,
            optimization_dir=tmp_path,
            run_name="rppo_run",
            seed=0,
        )
        assert (tmp_path / "rppo_run" / "best_params.json").exists()
        assert isinstance(best, dict)
        assert "learning_rate" in best

    def test_hrl_ppo_regression_when_algorithm_omitted(self, tmp_path):
        """With no ``algorithm`` field, tuning should still use plain PPO.

        This is a regression guard: pre-existing MARL configs (without an
        explicit algorithm field) must continue to work as HRL_PPO.
        """
        from stable_baselines3 import PPO
        config = _load()

        # Spy on _build_tuning_agent to capture the produced agent's class.
        import abx_amr_simulator.training.tune_marl_agent as _mod
        original_builder = _mod._build_tuning_agent
        captured = {"instances": []}

        def spy_builder(*args, **kwargs):
            inst = original_builder(*args, **kwargs)
            captured["instances"].append(inst)
            return inst

        _mod._build_tuning_agent = spy_builder
        try:
            run_marl_agent_tuning(
                config=config,
                agent_id="agent_0",
                tuning_config=_MINIMAL_TUNING_CONFIG,
                optimization_dir=tmp_path,
                run_name="ppo_regression",
                seed=0,
            )
        finally:
            _mod._build_tuning_agent = original_builder

        assert len(captured["instances"]) > 0
        # All produced agents must be plain PPO (not RecurrentPPO_Masked).
        for inst in captured["instances"]:
            assert type(inst) is PPO
