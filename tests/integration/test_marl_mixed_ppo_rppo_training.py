"""End-to-end integration test for MARL training with mixed PPO + RPPO agents.

Validates the full MARLTrainer loop with a heterogeneous agent configuration:
one HRL_PPO agent and one HRL_RPPO agent. This exercises the complete code path
from environment construction through rollout collection, buffer storage (both
RolloutBuffer and RecurrentRolloutBuffer), GAE computation, policy updates,
evaluation, and checkpoint saving.

Also validates the post-training granular eval pipeline: after training completes,
runs ``run_granular_eval_for_marl_seed()`` on the seed folder and verifies that
per-agent NPZ artifacts are produced with the expected structure.

Uses real components throughout — no mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer

from abx_amr_simulator.analysis.run_granular_eval_best_models_marl import (
    run_granular_eval_for_marl_seed,
)
from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl import MARLOptionsWrapper, OptionBase, OptionLibrary
from abx_amr_simulator.hrl.rl_algorithms.recurrent_ppo_masked import RecurrentPPO_Masked
from abx_amr_simulator.training.train_marl import (
    MARLTrainer,
    make_ppo_for_agent,
    make_recurrent_ppo_for_agent,
    run_marl_training,
)

from test_reference_helpers import make_pg, make_rc  # type: ignore[import-not-found]


# Path to the fixture YAML — resolved relative to this test file.
_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "marl_configs"
_FIXTURE_CONFIG = _FIXTURE_DIR / "minimal_two_agent.yaml"


# --------------------------------------------------------------------------- #
# Minimal real option (same corrected pattern used across all RPPO tests)
# --------------------------------------------------------------------------- #

class ConstantOption(OptionBase):
    """Prescribes the same antibiotic for k steps (no_treatment if prob_infected=0)."""

    REQUIRES_OBSERVATION_ATTRIBUTES = ["prob_infected"]
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = False

    def __init__(self, name: str, action_name: str, k: int = 5):
        super().__init__(name=name, k=k)
        self._action_name = action_name

    def decide(self, env_state: dict) -> np.ndarray:
        patients = env_state.get("patients", [])
        n = env_state.get("num_patients", len(patients))
        actions = np.full(shape=(n,), fill_value=self._action_name, dtype=object)
        for i, patient in enumerate(patients):
            if patient.get("prob_infected", 1.0) == 0.0:
                actions[i] = "no_treatment"
        return actions

    def get_referenced_antibiotics(self) -> list:
        return [self._action_name] if self._action_name != "no_treatment" else []


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

N_STEPS = 8
BATCH_SIZE = 4
MAX_TIME_STEPS = 20
OPTION_K = 5


def _build_wrapper(
    antibiotic_names=None,
    n_patients_per_agent=None,
) -> MARLOptionsWrapper:
    """Build a real MARLOptionsWrapper with two agents and one option each."""
    if antibiotic_names is None:
        antibiotic_names = ["A", "B"]
    if n_patients_per_agent is None:
        n_patients_per_agent = [3, 3]

    agent_configs = []
    for i, n in enumerate(n_patients_per_agent):
        pg = make_pg()
        pg.visible_patient_attributes = ["prob_infected"]
        rc = make_rc(antibiotic_names=antibiotic_names)
        agent_configs.append({
            "agent_id": f"agent_{i}",
            "n_patients": n,
            "patient_generator": pg,
            "reward_calculator": rc,
        })

    shared_env_config = {
        "antibiotics_AMR_dict": {
            name: {
                "leak": 0.05,
                "flatness_parameter": 1.0,
                "permanent_residual_volume": 0.0,
                "initial_amr_level": 0.0,
            }
            for name in antibiotic_names
        },
        "max_time_steps": MAX_TIME_STEPS,
    }

    base_env = ABXAMRParallelEnv(
        agent_configs=agent_configs,
        shared_env_config=shared_env_config,
        seed=0,
    )

    option_libs = {}
    for aid in base_env.possible_agents:
        rc = base_env._reward_calculators[aid]
        lib = OptionLibrary(reward_calculator=rc, name=f"lib_{aid}")
        lib.add_option(
            ConstantOption(name="prescribe_A", action_name="A", k=OPTION_K)
        )
        option_libs[aid] = lib

    return MARLOptionsWrapper(
        base_env=base_env,
        option_libraries=option_libs,
        gamma=0.99,
    )


def _build_mixed_agents(wrapper: MARLOptionsWrapper) -> dict:
    """Build one PPO agent (agent_0) and one RPPO agent (agent_1)."""
    agent_ids = list(wrapper.base_env.possible_agents)
    return {
        agent_ids[0]: make_ppo_for_agent(
            wrapper=wrapper, agent_id=agent_ids[0],
            n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=1, seed=0,
        ),
        agent_ids[1]: make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=agent_ids[1],
            n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
        ),
    }


# --------------------------------------------------------------------------- #
# Integration tests
# --------------------------------------------------------------------------- #

class TestMARLMixedPPORPPOTraining:
    """End-to-end integration tests for MARLTrainer with mixed agent types."""

    def test_training_completes_without_error(self, tmp_path):
        """Full training loop runs to completion with mixed PPO + RPPO agents."""
        wrapper = _build_wrapper()
        agents = _build_mixed_agents(wrapper)

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=N_STEPS,
            total_primitive_steps=200,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=2,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

    def test_final_models_saved_for_all_agents(self, tmp_path):
        """Final model .zip files are created for both PPO and RPPO agents."""
        wrapper = _build_wrapper()
        agents = _build_mixed_agents(wrapper)
        ckpt_dir = tmp_path / "checkpoints"

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=N_STEPS,
            total_primitive_steps=200,
            checkpoint_dir=ckpt_dir,
            eval_freq_episodes=2,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

        for aid in wrapper.base_env.possible_agents:
            assert (ckpt_dir / f"final_model_{aid}.zip").exists(), (
                f"Final model not saved for {aid}"
            )

    def test_agent_types_preserved_after_training(self, tmp_path):
        """PPO agent stays PPO, RPPO agent stays RecurrentPPO_Masked after training."""
        wrapper = _build_wrapper()
        agents = _build_mixed_agents(wrapper)
        agent_ids = list(wrapper.base_env.possible_agents)

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=N_STEPS,
            total_primitive_steps=200,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=2,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

        assert isinstance(agents[agent_ids[0]], PPO)
        assert not isinstance(agents[agent_ids[0]], RecurrentPPO_Masked)
        assert isinstance(agents[agent_ids[1]], RecurrentPPO_Masked)

    def test_buffer_types_correct_throughout(self, tmp_path):
        """PPO agent uses RolloutBuffer, RPPO agent uses RecurrentRolloutBuffer."""
        wrapper = _build_wrapper()
        agents = _build_mixed_agents(wrapper)
        agent_ids = list(wrapper.base_env.possible_agents)

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=N_STEPS,
            total_primitive_steps=200,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=2,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

        assert isinstance(trainer._buffers[agent_ids[0]], RolloutBuffer)
        assert isinstance(trainer._buffers[agent_ids[1]], RecurrentRolloutBuffer)

    def test_evaluation_runs_with_mixed_agents(self, tmp_path):
        """Eval episodes complete and produce finite rewards for both agent types."""
        wrapper = _build_wrapper()
        agents = _build_mixed_agents(wrapper)

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=N_STEPS,
            total_primitive_steps=200,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=2,
            n_eval_episodes=2,
            verbose=0,
        )

        eval_rewards = trainer._run_eval_episodes()
        assert set(eval_rewards.keys()) == set(wrapper.base_env.possible_agents)
        for aid, reward in eval_rewards.items():
            assert np.isfinite(reward), f"Non-finite eval reward for {aid}: {reward}"

    def test_best_model_checkpoints_created(self, tmp_path):
        """Best-model checkpoints are saved for agents whose eval reward improves."""
        wrapper = _build_wrapper()
        agents = _build_mixed_agents(wrapper)
        ckpt_dir = tmp_path / "checkpoints"

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=N_STEPS,
            total_primitive_steps=200,
            checkpoint_dir=ckpt_dir,
            eval_freq_episodes=1,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

        # At least one eval should have run (episodes > 0), so best models
        # should exist for both agents (first eval always sets "best").
        for aid in wrapper.base_env.possible_agents:
            assert (ckpt_dir / f"best_model_{aid}.zip").exists(), (
                f"Best model checkpoint not saved for {aid}"
            )

    def test_multiple_episodes_with_buffer_flushes(self, tmp_path):
        """Training runs long enough for multiple buffer flushes without error.

        With n_steps=8 and option_k=5, each episode of max_time_steps=20
        produces ~4 manager transitions per agent. So 8 steps of buffer
        capacity fills after ~2 episodes, triggering a flush and policy
        update. Running for 400 primitive steps ensures multiple flushes.
        """
        wrapper = _build_wrapper()
        agents = _build_mixed_agents(wrapper)

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=N_STEPS,
            total_primitive_steps=400,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=5,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

    def test_deterministic_with_same_seed(self, tmp_path):
        """Two training runs with the same seed produce identical final models."""
        results = []
        for run_idx in range(2):
            wrapper = _build_wrapper()
            agents = _build_mixed_agents(wrapper)
            ckpt_dir = tmp_path / f"run_{run_idx}"

            trainer = MARLTrainer(
                wrapper=wrapper,
                agents=agents,
                n_steps=N_STEPS,
                total_primitive_steps=200,
                checkpoint_dir=ckpt_dir,
                eval_freq_episodes=2,
                n_eval_episodes=1,
                verbose=0,
            )
            trainer.train()

            # Extract final policy parameters as a fingerprint
            params = {}
            for aid in wrapper.base_env.possible_agents:
                params[aid] = {
                    name: p.detach().cpu().numpy().copy()
                    for name, p in agents[aid].policy.named_parameters()
                }
            results.append(params)

        for aid in results[0]:
            for name in results[0][aid]:
                np.testing.assert_array_equal(
                    results[0][aid][name],
                    results[1][aid][name],
                    err_msg=f"Parameter {name} for {aid} differs between runs",
                )


class TestMARLAllRPPOTraining:
    """Integration tests for MARLTrainer with all-RPPO agents."""

    def test_all_rppo_training_completes(self, tmp_path):
        """Training completes with two RPPO agents."""
        wrapper = _build_wrapper()
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper=wrapper, agent_id=aid,
                n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=N_STEPS,
            total_primitive_steps=200,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=2,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

        for aid in wrapper.base_env.possible_agents:
            assert (tmp_path / "checkpoints" / f"final_model_{aid}.zip").exists()

    def test_all_rppo_lstm_states_tracked(self, tmp_path):
        """LSTM states are non-None for all agents during training."""
        wrapper = _build_wrapper()
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper=wrapper, agent_id=aid,
                n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=N_STEPS,
            total_primitive_steps=100,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=100,
            n_eval_episodes=1,
            verbose=0,
        )

        # Verify LSTM states are initialized
        for aid in wrapper.base_env.possible_agents:
            assert trainer._is_recurrent[aid] is True
            assert trainer._lstm_states[aid] is not None

        trainer.train()

        # After training, LSTM states should still be tracked
        for aid in wrapper.base_env.possible_agents:
            assert trainer._lstm_states[aid] is not None


# --------------------------------------------------------------------------- #
# Helpers for config-based training + granular eval tests
# --------------------------------------------------------------------------- #

def _write_mixed_algorithm_config(dest_path: Path) -> Path:
    """Write a MARL config YAML with agent_0=HRL_PPO and agent_1=HRL_RPPO.

    Loads the fixture minimal_two_agent.yaml, sets per-agent algorithm fields,
    resolves the option_library paths to absolute so the config is self-contained
    when written to an arbitrary directory, and writes the result to dest_path.

    Returns the path to the written YAML.
    """
    with open(_FIXTURE_CONFIG) as f:
        config = yaml.safe_load(f)

    agents = config["environment"]["agents"]
    agents[0]["algorithm"] = "HRL_PPO"
    agents[1]["algorithm"] = "HRL_RPPO"

    # Make option_library paths absolute (they are relative to the fixture dir).
    for entry in agents:
        lib_path = Path(entry["option_library"])
        if not lib_path.is_absolute():
            entry["option_library"] = str((_FIXTURE_DIR / lib_path).resolve())

    # Set training params directly (override system only updates existing keys).
    training = config["training"]
    training["total_primitive_steps"] = 200
    training["eval_freq_episodes"] = 1
    training["save_freq_episodes"] = 1
    training["n_eval_episodes"] = 1

    config_path = dest_path / "marl_mixed_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


# --------------------------------------------------------------------------- #
# Granular eval integration tests
# --------------------------------------------------------------------------- #

# Expected NPZ fields that every episode must contain (shared across all agents).
_EXPECTED_EPISODE_FIELDS = [
    "primitive_patient_true",
    "primitive_patient_observed",
    "primitive_patient_attrs",
    "primitive_individual_rewards",
    "primitive_patients_actually_infected",
    "primitive_substep_counts",
    "primitive_actions",
    "primitive_actual_amr_levels",
    "primitive_visible_amr_levels",
    "primitive_total_reward",
    "primitive_overall_individual_reward_component",
    "primitive_normalized_individual_reward",
    "primitive_overall_community_reward_component",
    "primitive_normalized_community_reward",
    "primitive_count_clinical_benefits",
    "primitive_count_clinical_failures",
    "primitive_count_adverse_events",
    "primitive_not_infected_no_treatment",
    "primitive_not_infected_treated",
    "primitive_infected_no_treatment",
]


class TestMARLGranularEvalMixedAgents:
    """End-to-end: train mixed PPO+RPPO via run_marl_training, then run granular eval."""

    def _train_and_get_seed_folder(self, tmp_path: Path) -> Path:
        """Train a short mixed-algorithm run and return the seed folder path."""
        config_path = _write_mixed_algorithm_config(tmp_path)
        results_dir = tmp_path / "results"
        run_name = "mixed_granular_test"

        run_marl_training(
            marl_config_path=config_path,
            results_dir=results_dir,
            run_name=run_name,
            seed=42,
        )

        # Find the timestamped run folder.
        run_candidates = sorted(results_dir.glob(f"{run_name}_????????_??????"))
        assert len(run_candidates) == 1, (
            f"Expected exactly 1 run folder, found {len(run_candidates)}"
        )
        return run_candidates[0]

    def test_granular_eval_produces_npz_per_agent(self, tmp_path):
        """Granular eval creates one NPZ per agent in eval_logs/."""
        seed_folder = self._train_and_get_seed_folder(tmp_path)

        status = run_granular_eval_for_marl_seed(
            seed_folder=seed_folder,
            n_eval_episodes=2,
            force=True,
        )
        assert status == "done"

        eval_dir = seed_folder / "eval_logs"
        assert eval_dir.exists()
        assert (eval_dir / "eval_granular_best_model_agent_0.npz").exists()
        assert (eval_dir / "eval_granular_best_model_agent_1.npz").exists()

    def test_npz_contains_expected_top_level_fields(self, tmp_path):
        """Each NPZ has antibiotic_names, num_episodes, and episode data."""
        seed_folder = self._train_and_get_seed_folder(tmp_path)

        run_granular_eval_for_marl_seed(
            seed_folder=seed_folder,
            n_eval_episodes=2,
            force=True,
        )

        for aid in ["agent_0", "agent_1"]:
            npz_path = seed_folder / "eval_logs" / f"eval_granular_best_model_{aid}.npz"
            with np.load(npz_path, allow_pickle=True) as data:
                keys = list(data.keys())
                assert "antibiotic_names" in keys, f"{aid}: missing antibiotic_names"
                assert "num_episodes" in keys, f"{aid}: missing num_episodes"
                assert int(data["num_episodes"]) == 2, (
                    f"{aid}: expected 2 episodes, got {data['num_episodes']}"
                )

    def test_npz_episode_has_all_required_fields(self, tmp_path):
        """Each episode in the NPZ contains all primitive-level data arrays."""
        seed_folder = self._train_and_get_seed_folder(tmp_path)

        run_granular_eval_for_marl_seed(
            seed_folder=seed_folder,
            n_eval_episodes=1,
            force=True,
        )

        for aid in ["agent_0", "agent_1"]:
            npz_path = seed_folder / "eval_logs" / f"eval_granular_best_model_{aid}.npz"
            with np.load(npz_path, allow_pickle=True) as data:
                keys = list(data.keys())
                for field in _EXPECTED_EPISODE_FIELDS:
                    full_key = f"episode_0/{field}"
                    assert full_key in keys, (
                        f"{aid}: missing expected field '{full_key}'"
                    )

    def test_npz_per_antibiotic_outcome_fields(self, tmp_path):
        """Each episode has per-antibiotic sensitive/resistant outcome arrays."""
        seed_folder = self._train_and_get_seed_folder(tmp_path)

        run_granular_eval_for_marl_seed(
            seed_folder=seed_folder,
            n_eval_episodes=1,
            force=True,
        )

        for aid in ["agent_0", "agent_1"]:
            npz_path = seed_folder / "eval_logs" / f"eval_granular_best_model_{aid}.npz"
            with np.load(npz_path, allow_pickle=True) as data:
                antibiotic_names = [str(n) for n in data["antibiotic_names"].tolist()]
                assert len(antibiotic_names) == 2, (
                    f"{aid}: expected 2 antibiotics, got {len(antibiotic_names)}"
                )
                for abx in antibiotic_names:
                    for prefix in [
                        "primitive_sensitive_infection_treated",
                        "primitive_resistant_infection_treated",
                    ]:
                        full_key = f"episode_0/{prefix}/{abx}"
                        assert full_key in data, (
                            f"{aid}: missing '{full_key}'"
                        )

    def test_npz_arrays_have_finite_values(self, tmp_path):
        """Core numeric arrays contain finite values (no NaN/inf in data region)."""
        seed_folder = self._train_and_get_seed_folder(tmp_path)

        run_granular_eval_for_marl_seed(
            seed_folder=seed_folder,
            n_eval_episodes=1,
            force=True,
        )

        for aid in ["agent_0", "agent_1"]:
            npz_path = seed_folder / "eval_logs" / f"eval_granular_best_model_{aid}.npz"
            with np.load(npz_path, allow_pickle=True) as data:
                substep_counts = data["episode_0/primitive_substep_counts"]
                assert np.all(substep_counts > 0), (
                    f"{aid}: zero substep counts found"
                )

                patient_true = data["episode_0/primitive_patient_true"]
                assert np.all(np.isfinite(patient_true)), (
                    f"{aid}: non-finite values in patient_true"
                )

    def test_skip_when_npz_exists(self, tmp_path):
        """Granular eval returns 'skip' when all NPZs already exist and force=False."""
        seed_folder = self._train_and_get_seed_folder(tmp_path)

        # First run creates the NPZs.
        status1 = run_granular_eval_for_marl_seed(
            seed_folder=seed_folder,
            n_eval_episodes=1,
            force=True,
        )
        assert status1 == "done"

        # Second run should skip.
        status2 = run_granular_eval_for_marl_seed(
            seed_folder=seed_folder,
            n_eval_episodes=1,
            force=False,
        )
        assert status2 == "skip"
