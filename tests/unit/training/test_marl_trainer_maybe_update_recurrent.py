"""Unit tests for MARLTrainer._maybe_update() with recurrent agents (Phase 3).

Verifies that _maybe_update() correctly passes LSTM states to predict_values()
for GAE bootstrap, triggers agent.train(), and resets the buffer for recurrent
agents — while leaving the non-recurrent path unchanged.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer

from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl import MARLOptionsWrapper, OptionBase, OptionLibrary
from abx_amr_simulator.training.train_marl import (
    MARLTrainer,
    make_ppo_for_agent,
    make_recurrent_ppo_for_agent,
)

from test_reference_helpers import make_pg, make_rc  # type: ignore[import-not-found]


# --------------------------------------------------------------------------- #
# Minimal real option
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
# Fixtures
# --------------------------------------------------------------------------- #

def _make_wrapper():
    """Build a real MARLOptionsWrapper with two agents and one option each."""
    antibiotic_names = ["A", "B"]
    agent_configs = []
    for i, n in enumerate([3, 3]):
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
        "max_time_steps": 20,
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
            ConstantOption(name="prescribe_A", action_name="A", k=5)
        )
        option_libs[aid] = lib

    return MARLOptionsWrapper(
        base_env=base_env,
        option_libraries=option_libs,
        gamma=0.99,
    )


@pytest.fixture
def wrapper():
    return _make_wrapper()


# Use a small n_steps so we can fill the buffer easily
N_STEPS = 4
BATCH_SIZE = 4
TOTAL_STEPS = 1000


def _fill_buffer(trainer, wrapper, aid):
    """Fill an agent's rollout buffer by running predict + store N_STEPS times.

    Returns the observation after the last stored transition (for use as
    next_obs in _maybe_update).
    """
    obs_dict, _ = wrapper.reset()
    for step in range(N_STEPS):
        trainer._predict_single(aid, obs_dict[aid], episode_start=(step == 0))
        trainer._store_transition(
            aid=aid,
            obs=obs_dict[aid],
            action=0,
            reward=1.0,
            episode_start=(step == 0),
            m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
        )
    assert trainer._buffers[aid].full
    return obs_dict[aid]


# --------------------------------------------------------------------------- #
# Tests: _maybe_update with RPPO agents
# --------------------------------------------------------------------------- #

class TestMaybeUpdateRPPO:
    """Verify _maybe_update works for recurrent agents."""

    def test_does_not_crash_when_buffer_full(self, wrapper):
        """_maybe_update should complete without error for a recurrent agent."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_update_rppo",
        )
        aid = trainer._agent_ids[0]
        next_obs = _fill_buffer(trainer, wrapper, aid)

        # Should not raise
        trainer._maybe_update(aid=aid, next_obs=next_obs, done=False)

    def test_buffer_reset_after_update(self, wrapper):
        """After _maybe_update triggers, the buffer should be reset (pos=0, not full)."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_update_rppo",
        )
        aid = trainer._agent_ids[0]
        next_obs = _fill_buffer(trainer, wrapper, aid)

        trainer._maybe_update(aid=aid, next_obs=next_obs, done=False)
        assert trainer._buffers[aid].pos == 0
        assert not trainer._buffers[aid].full

    def test_policy_parameters_change_after_update(self, wrapper):
        """agent.train() should modify policy parameters."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_update_rppo",
        )
        aid = trainer._agent_ids[0]

        # Snapshot parameters before update
        params_before = {
            name: p.clone()
            for name, p in trainer.agents[aid].policy.named_parameters()
        }

        next_obs = _fill_buffer(trainer, wrapper, aid)
        trainer._maybe_update(aid=aid, next_obs=next_obs, done=False)

        # At least some parameters should have changed
        changed = False
        for name, p in trainer.agents[aid].policy.named_parameters():
            if not torch.equal(p, params_before[name]):
                changed = True
                break
        assert changed, "Policy parameters should change after train()"

    def test_no_update_when_buffer_not_full(self, wrapper):
        """_maybe_update should be a no-op when the buffer is not full."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_update_rppo",
        )
        aid = trainer._agent_ids[0]
        obs_dict, _ = wrapper.reset()

        # Store fewer transitions than n_steps
        trainer._predict_single(aid, obs_dict[aid], episode_start=True)
        trainer._store_transition(
            aid=aid, obs=obs_dict[aid], action=0, reward=1.0,
            episode_start=True,
            m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
        )
        assert not trainer._buffers[aid].full

        # _maybe_update should not reset the buffer
        trainer._maybe_update(aid=aid, next_obs=obs_dict[aid], done=False)
        assert trainer._buffers[aid].pos == 1  # unchanged

    def test_works_with_done_true(self, wrapper):
        """_maybe_update should work when done=True (terminal episode)."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_update_rppo",
        )
        aid = trainer._agent_ids[0]
        next_obs = _fill_buffer(trainer, wrapper, aid)

        # Should not raise with done=True
        trainer._maybe_update(aid=aid, next_obs=next_obs, done=True)
        assert not trainer._buffers[aid].full

    def test_can_fill_and_update_twice(self, wrapper):
        """Buffer can be filled, updated, then filled and updated again."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_update_rppo",
        )
        aid = trainer._agent_ids[0]

        # First fill + update
        next_obs = _fill_buffer(trainer, wrapper, aid)
        trainer._maybe_update(aid=aid, next_obs=next_obs, done=False)
        assert not trainer._buffers[aid].full

        # Second fill + update
        next_obs = _fill_buffer(trainer, wrapper, aid)
        trainer._maybe_update(aid=aid, next_obs=next_obs, done=False)
        assert not trainer._buffers[aid].full


# --------------------------------------------------------------------------- #
# Tests: _maybe_update with PPO agents (backward compatibility)
# --------------------------------------------------------------------------- #

class TestMaybeUpdatePPO:
    """Verify _maybe_update still works for non-recurrent agents."""

    def test_buffer_reset_after_update(self, wrapper):
        agents = {
            aid: make_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_update_ppo",
        )
        aid = trainer._agent_ids[0]
        next_obs = _fill_buffer(trainer, wrapper, aid)

        trainer._maybe_update(aid=aid, next_obs=next_obs, done=False)
        assert not trainer._buffers[aid].full


# --------------------------------------------------------------------------- #
# Tests: mixed PPO + RPPO
# --------------------------------------------------------------------------- #

class TestMaybeUpdateMixed:
    """Verify _maybe_update works with mixed agent types."""

    def test_both_agents_can_update(self, wrapper):
        agent_ids = list(wrapper.base_env.possible_agents)
        agents = {
            agent_ids[0]: make_ppo_for_agent(
                wrapper, agent_ids[0], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
            ),
            agent_ids[1]: make_recurrent_ppo_for_agent(
                wrapper, agent_ids[1], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
            ),
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_update_mixed",
        )

        for aid in trainer._agent_ids:
            next_obs = _fill_buffer(trainer, wrapper, aid)
            trainer._maybe_update(aid=aid, next_obs=next_obs, done=False)
            assert not trainer._buffers[aid].full
