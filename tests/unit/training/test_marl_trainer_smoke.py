"""Smoke tests for MARLTrainer.

Runs MARLTrainer for a very short budget with a minimal two-agent config and
verifies that the loop completes without errors, checkpoints are written, and
at least one gradient update happens per agent.

Uses real ABXAMRParallelEnv, MARLOptionsWrapper, and PPO objects — no mocks.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl import MARLOptionsWrapper, OptionBase, OptionLibrary
from abx_amr_simulator.training.train_marl import MARLTrainer, make_ppo_for_agent

# Import test helpers (sys.path configured in tests/conftest.py)
from test_reference_helpers import make_pg, make_rc  # type: ignore[import-not-found]


# --------------------------------------------------------------------------- #
# Minimal real option for testing
# --------------------------------------------------------------------------- #

class ConstantOption(OptionBase):
    """Always prescribes the same antibiotic for k steps."""

    REQUIRES_OBSERVATION_ATTRIBUTES = ["prob_infected"]
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = False

    def __init__(self, name: str, action_name: str, k: int = 5):
        super().__init__(name=name, k=k)
        self._action_name = action_name

    def decide(self, env_state: dict) -> np.ndarray:
        n = env_state.get("num_patients", 1)
        return np.full(shape=(n,), fill_value=self._action_name, dtype=object)

    def get_referenced_antibiotics(self) -> list:
        return [self._action_name] if self._action_name != "no_treatment" else []


# --------------------------------------------------------------------------- #
# Helpers that build real components
# --------------------------------------------------------------------------- #

def _build_minimal_wrapper(
    antibiotic_names=None,
    n_patients_per_agent=None,
    option_k: int = 5,
    max_time_steps: int = 20,
) -> MARLOptionsWrapper:
    """Build a real MARLOptionsWrapper with a single ConstantOption per agent."""
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
        "max_time_steps": max_time_steps,
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
            ConstantOption(
                name=f"prescribe_{antibiotic_names[0]}",
                action_name=antibiotic_names[0],
                k=option_k,
            )
        )
        option_libs[aid] = lib

    return MARLOptionsWrapper(
        base_env=base_env,
        option_libraries=option_libs,
        gamma=0.99,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestMARLTrainerSmoke:
    """Smoke tests: MARLTrainer runs without error and produces expected artefacts."""

    def test_train_completes_without_error(self, tmp_path):
        """MARLTrainer.train() runs to completion with no exceptions."""
        n_steps = 8
        option_k = 5
        max_time_steps = 20
        # Budget: enough for at least 2 buffer flushes per agent
        total_primitive_steps = n_steps * option_k * 3

        wrapper = _build_minimal_wrapper(
            option_k=option_k,
            max_time_steps=max_time_steps,
        )
        agents = {
            aid: make_ppo_for_agent(
                wrapper=wrapper,
                agent_id=aid,
                n_steps=n_steps,
                batch_size=4,
                n_epochs=2,
                seed=42,
            )
            for aid in wrapper.base_env.possible_agents
        }

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=n_steps,
            total_primitive_steps=total_primitive_steps,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=5,
            n_eval_episodes=2,
            verbose=0,
        )
        trainer.train()  # must not raise

    def test_final_model_files_created(self, tmp_path):
        """Final model .zip files are written for every agent."""
        n_steps = 8
        wrapper = _build_minimal_wrapper(option_k=5, max_time_steps=20)
        agents = {
            aid: make_ppo_for_agent(wrapper=wrapper, agent_id=aid, n_steps=n_steps,
                                    batch_size=4, n_epochs=1, seed=0)
            for aid in wrapper.base_env.possible_agents
        }
        checkpoint_dir = tmp_path / "checkpoints"
        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=n_steps,
            total_primitive_steps=n_steps * 5 * 3,
            checkpoint_dir=checkpoint_dir,
            eval_freq_episodes=100,  # skip eval
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

        for aid in wrapper.base_env.possible_agents:
            final = checkpoint_dir / f"final_model_{aid}.zip"
            assert final.exists(), f"Expected {final} but it does not exist"

    def test_primitive_step_count_within_one_option_of_budget(self, tmp_path):
        """Total primitive steps elapsed is within one option duration of the budget."""
        option_k = 5
        n_steps = 8
        budget = n_steps * option_k * 4

        wrapper = _build_minimal_wrapper(option_k=option_k, max_time_steps=40)
        agents = {
            aid: make_ppo_for_agent(wrapper=wrapper, agent_id=aid, n_steps=n_steps,
                                    batch_size=4, n_epochs=1, seed=0)
            for aid in wrapper.base_env.possible_agents
        }

        # Monkey-patch to track actual primitive steps
        original_step = wrapper.step
        actual_steps = [0]
        def tracking_step(selections):
            result = original_step(selections)
            m_info = result[4]
            if m_info:
                actual_steps[0] += max(
                    info["option_duration"] for info in m_info.values()
                )
            return result
        wrapper.step = tracking_step

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=n_steps,
            total_primitive_steps=budget,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=100,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

        assert actual_steps[0] >= budget, (
            f"Fewer steps than budget: {actual_steps[0]} < {budget}"
        )
        assert actual_steps[0] <= budget + option_k, (
            f"Steps exceeded budget by more than one option: "
            f"{actual_steps[0]} > {budget + option_k}"
        )

    def test_at_least_one_gradient_update_per_agent(self, tmp_path):
        """Each agent's SB3 update counter increments after training."""
        n_steps = 8
        option_k = 5

        wrapper = _build_minimal_wrapper(option_k=option_k, max_time_steps=20)
        agents = {
            aid: make_ppo_for_agent(wrapper=wrapper, agent_id=aid, n_steps=n_steps,
                                    batch_size=4, n_epochs=2, seed=42)
            for aid in wrapper.base_env.possible_agents
        }

        # Capture SB3's internal update counter before training
        initial_updates = {
            aid: agents[aid]._n_updates
            for aid in wrapper.base_env.possible_agents
        }

        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=n_steps,
            total_primitive_steps=n_steps * option_k * 5,
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=100,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

        for aid in wrapper.base_env.possible_agents:
            assert agents[aid]._n_updates > initial_updates[aid], (
                f"Agent '{aid}' had no gradient updates — "
                f"_n_updates stayed at {initial_updates[aid]}"
            )

    def test_best_model_saved_when_eval_runs(self, tmp_path):
        """best_model_{aid}.zip is created when eval_freq_episodes is reached."""
        n_steps = 8
        option_k = 5
        # Force eval every episode
        wrapper = _build_minimal_wrapper(option_k=option_k, max_time_steps=20)
        agents = {
            aid: make_ppo_for_agent(wrapper=wrapper, agent_id=aid, n_steps=n_steps,
                                    batch_size=4, n_epochs=1, seed=0)
            for aid in wrapper.base_env.possible_agents
        }
        checkpoint_dir = tmp_path / "checkpoints"
        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=n_steps,
            total_primitive_steps=n_steps * option_k * 5,
            checkpoint_dir=checkpoint_dir,
            eval_freq_episodes=1,   # eval every episode
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()

        for aid in wrapper.base_env.possible_agents:
            best = checkpoint_dir / f"best_model_{aid}.zip"
            assert best.exists(), f"Expected best_model_{aid}.zip but not found"

    def test_small_budget_terminates_gracefully(self, tmp_path):
        """A budget smaller than one full option runs without error."""
        option_k = 10
        n_steps = 32
        # Budget is less than one option's duration — should still terminate cleanly
        wrapper = _build_minimal_wrapper(option_k=option_k, max_time_steps=50)
        agents = {
            aid: make_ppo_for_agent(wrapper=wrapper, agent_id=aid, n_steps=n_steps,
                                    batch_size=8, n_epochs=1, seed=0)
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper,
            agents=agents,
            n_steps=n_steps,
            total_primitive_steps=3,  # less than option_k=10
            checkpoint_dir=tmp_path / "checkpoints",
            eval_freq_episodes=100,
            n_eval_episodes=1,
            verbose=0,
        )
        trainer.train()  # should not raise
