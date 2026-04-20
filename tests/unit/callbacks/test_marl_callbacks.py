"""Unit tests for run_marl_eval_episodes in callbacks.marl_callbacks.

Uses real ABXAMRParallelEnv, MARLOptionsWrapper, and PPO objects — no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from abx_amr_simulator.callbacks.marl_callbacks import run_marl_eval_episodes
from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl import MARLOptionsWrapper, OptionBase, OptionLibrary
from abx_amr_simulator.training.train_marl import (
    make_ppo_for_agent,
    make_recurrent_ppo_for_agent,
)

from test_reference_helpers import make_pg, make_rc  # type: ignore[import-not-found]


# --------------------------------------------------------------------------- #
# Minimal real option for testing
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

def _build_wrapper(
    antibiotic_names=None,
    n_patients_per_agent=None,
    option_k: int = 5,
    max_time_steps: int = 20,
) -> MARLOptionsWrapper:
    """Build a real MARLOptionsWrapper with one ConstantOption per agent."""
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


def _build_agents(wrapper: MARLOptionsWrapper, n_steps: int = 8) -> dict:
    return {
        aid: make_ppo_for_agent(wrapper=wrapper, agent_id=aid, n_steps=n_steps,
                                batch_size=4, n_epochs=1, seed=0)
        for aid in wrapper.base_env.possible_agents
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestRunMarlEvalEpisodes:
    """Tests for the run_marl_eval_episodes standalone function."""

    def test_returns_dict_with_all_agent_ids(self):
        """Result contains one entry per agent."""
        wrapper = _build_wrapper()
        agents = _build_agents(wrapper)

        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=2)

        assert set(result.keys()) == set(wrapper.base_env.possible_agents)

    def test_returns_float_rewards(self):
        """All values in the returned dict are plain floats."""
        wrapper = _build_wrapper()
        agents = _build_agents(wrapper)

        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=1)

        for aid, r in result.items():
            assert isinstance(r, float), f"Expected float for {aid}, got {type(r)}"

    def test_single_episode_returns_finite_rewards(self):
        """Rewards are finite (not NaN or inf) for a normal episode."""
        wrapper = _build_wrapper(max_time_steps=20)
        agents = _build_agents(wrapper)

        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=1)

        for aid, r in result.items():
            assert np.isfinite(r), f"Reward for {aid} is not finite: {r}"

    def test_multiple_episodes_runs_without_error(self):
        """Running more than one eval episode completes without error."""
        wrapper = _build_wrapper()
        agents = _build_agents(wrapper)

        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=3)

        assert len(result) == len(wrapper.base_env.possible_agents)

    def test_does_not_modify_rollout_buffers(self):
        """Buffer positions are unchanged after eval (no training state corruption)."""
        wrapper = _build_wrapper()
        agents = _build_agents(wrapper)

        before = {
            aid: agents[aid].rollout_buffer.pos
            for aid in wrapper.base_env.possible_agents
        }
        run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=2)
        after = {
            aid: agents[aid].rollout_buffer.pos
            for aid in wrapper.base_env.possible_agents
        }

        for aid in wrapper.base_env.possible_agents:
            assert before[aid] == after[aid], (
                f"Buffer position changed for {aid}: {before[aid]} → {after[aid]}"
            )

    def test_wrapper_resets_between_episodes(self):
        """Each episode starts fresh (wrapper.reset() called per episode).

        Verified indirectly: running n_episodes on a short-episode wrapper
        completes without the truncation flag bleeding across episodes.
        """
        wrapper = _build_wrapper(max_time_steps=5, option_k=5)
        agents = _build_agents(wrapper)

        # If episodes bled together this would hang or raise
        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=4)
        assert len(result) == len(wrapper.base_env.possible_agents)

    def test_deterministic_across_calls_with_same_seed(self):
        """Eval results are consistent across calls when wrapper seed is fixed."""
        wrapper1 = _build_wrapper()
        agents1 = _build_agents(wrapper1)
        result1 = run_marl_eval_episodes(wrapper=wrapper1, agents=agents1, n_episodes=3)

        wrapper2 = _build_wrapper()
        agents2 = _build_agents(wrapper2)
        result2 = run_marl_eval_episodes(wrapper=wrapper2, agents=agents2, n_episodes=3)

        for aid in wrapper1.base_env.possible_agents:
            assert result1[aid] == pytest.approx(result2[aid], abs=1e-6), (
                f"Eval reward for {aid} not reproducible: {result1[aid]} vs {result2[aid]}"
            )


# --------------------------------------------------------------------------- #
# Tests: RPPO eval
# --------------------------------------------------------------------------- #

def _build_rppo_agents(wrapper: MARLOptionsWrapper, n_steps: int = 8) -> dict:
    return {
        aid: make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=n_steps,
            batch_size=4, seed=0,
        )
        for aid in wrapper.base_env.possible_agents
    }


def _build_mixed_agents(wrapper: MARLOptionsWrapper, n_steps: int = 8) -> dict:
    agent_ids = list(wrapper.base_env.possible_agents)
    return {
        agent_ids[0]: make_ppo_for_agent(
            wrapper=wrapper, agent_id=agent_ids[0], n_steps=n_steps,
            batch_size=4, n_epochs=1, seed=0,
        ),
        agent_ids[1]: make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=agent_ids[1], n_steps=n_steps,
            batch_size=4, seed=0,
        ),
    }


class TestRunMarlEvalEpisodesRPPO:
    """Tests for run_marl_eval_episodes with recurrent agents."""

    def test_returns_dict_with_all_agent_ids(self):
        wrapper = _build_wrapper()
        agents = _build_rppo_agents(wrapper)
        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=2)
        assert set(result.keys()) == set(wrapper.base_env.possible_agents)

    def test_returns_finite_rewards(self):
        wrapper = _build_wrapper()
        agents = _build_rppo_agents(wrapper)
        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=1)
        for aid, r in result.items():
            assert np.isfinite(r), f"Reward for {aid} is not finite: {r}"

    def test_multiple_episodes_runs_without_error(self):
        wrapper = _build_wrapper()
        agents = _build_rppo_agents(wrapper)
        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=3)
        assert len(result) == len(wrapper.base_env.possible_agents)

    def test_does_not_modify_rollout_buffers(self):
        wrapper = _build_wrapper()
        agents = _build_rppo_agents(wrapper)
        before = {
            aid: agents[aid].rollout_buffer.pos
            for aid in wrapper.base_env.possible_agents
        }
        run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=2)
        after = {
            aid: agents[aid].rollout_buffer.pos
            for aid in wrapper.base_env.possible_agents
        }
        for aid in wrapper.base_env.possible_agents:
            assert before[aid] == after[aid]


class TestRunMarlEvalEpisodesMixed:
    """Tests for run_marl_eval_episodes with mixed PPO + RPPO agents."""

    def test_returns_all_agents(self):
        wrapper = _build_wrapper()
        agents = _build_mixed_agents(wrapper)
        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=2)
        assert set(result.keys()) == set(wrapper.base_env.possible_agents)

    def test_returns_finite_rewards(self):
        wrapper = _build_wrapper()
        agents = _build_mixed_agents(wrapper)
        result = run_marl_eval_episodes(wrapper=wrapper, agents=agents, n_episodes=1)
        for aid, r in result.items():
            assert np.isfinite(r)
