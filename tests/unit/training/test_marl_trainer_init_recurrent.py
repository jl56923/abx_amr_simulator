"""Unit tests for MARLTrainer.__init__() recurrent agent support (Phase 2a).

Verifies that MARLTrainer correctly detects recurrent vs. non-recurrent agents,
initializes LSTM state tracking dicts, accepts RecurrentRolloutBuffers, and
that _setup_learn() does not clobber LSTM states.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.buffers import RolloutBuffer

from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl import MARLOptionsWrapper, OptionBase, OptionLibrary
from abx_amr_simulator.hrl.rl_algorithms.recurrent_ppo_masked import RecurrentPPO_Masked
from abx_amr_simulator.training.train_marl import (
    MARLTrainer,
    make_ppo_for_agent,
    make_recurrent_ppo_for_agent,
)

from test_reference_helpers import make_pg, make_rc  # type: ignore[import-not-found]


# --------------------------------------------------------------------------- #
# Minimal real option (consistent with test_make_recurrent_ppo_for_agent.py)
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


N_STEPS = 16
BATCH_SIZE = 8
TOTAL_STEPS = 1000


# --------------------------------------------------------------------------- #
# Tests: all-PPO agents (backward compatibility)
# --------------------------------------------------------------------------- #

class TestMARLTrainerInitAllPPO:
    """Verify __init__ still works correctly with all non-recurrent PPO agents."""

    def test_is_recurrent_all_false(self, wrapper):
        agents = {
            aid: make_ppo_for_agent(wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0)
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_ppo_init",
        )
        for aid in trainer._agent_ids:
            assert trainer._is_recurrent[aid] is False

    def test_lstm_states_all_none(self, wrapper):
        agents = {
            aid: make_ppo_for_agent(wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0)
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_ppo_init",
        )
        for aid in trainer._agent_ids:
            assert trainer._lstm_states[aid] is None
            assert trainer._lstm_states_at_action[aid] is None

    def test_buffers_are_rollout_buffers(self, wrapper):
        agents = {
            aid: make_ppo_for_agent(wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0)
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_ppo_init",
        )
        for aid in trainer._agent_ids:
            assert isinstance(trainer._buffers[aid], RolloutBuffer)


# --------------------------------------------------------------------------- #
# Tests: all-RPPO agents
# --------------------------------------------------------------------------- #

class TestMARLTrainerInitAllRPPO:
    """Verify __init__ correctly handles all recurrent agents."""

    def test_is_recurrent_all_true(self, wrapper):
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_rppo_init",
        )
        for aid in trainer._agent_ids:
            assert trainer._is_recurrent[aid] is True

    def test_lstm_states_initialized_as_rnn_states(self, wrapper):
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_rppo_init",
        )
        for aid in trainer._agent_ids:
            lstm_states = trainer._lstm_states[aid]
            assert isinstance(lstm_states, RNNStates)
            # RNNStates has .pi and .vf fields
            assert hasattr(lstm_states, "pi")
            assert hasattr(lstm_states, "vf")

    def test_lstm_states_are_zero_tensors(self, wrapper):
        """LSTM states should be initialized to zeros (matching SB3's _last_lstm_states)."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_rppo_init",
        )
        for aid in trainer._agent_ids:
            lstm_states = trainer._lstm_states[aid]
            for tensor in lstm_states.pi:
                assert torch.all(tensor == 0.0)
            for tensor in lstm_states.vf:
                assert torch.all(tensor == 0.0)

    def test_lstm_states_are_independent_copies(self, wrapper):
        """LSTM states must be deep copies, not aliases to agent._last_lstm_states."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_rppo_init",
        )
        for aid in trainer._agent_ids:
            trainer_states = trainer._lstm_states[aid]
            agent_states = agents[aid]._last_lstm_states
            # Modify the trainer's copy — should not affect the agent's copy
            trainer_states.pi[0].fill_(999.0)
            assert not torch.all(agent_states.pi[0] == 999.0)

    def test_lstm_states_at_action_all_none(self, wrapper):
        """_lstm_states_at_action should be None for all agents initially."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_rppo_init",
        )
        for aid in trainer._agent_ids:
            assert trainer._lstm_states_at_action[aid] is None

    def test_buffers_are_recurrent_rollout_buffers(self, wrapper):
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_rppo_init",
        )
        for aid in trainer._agent_ids:
            assert isinstance(trainer._buffers[aid], RecurrentRolloutBuffer)

    def test_setup_learn_does_not_clobber_lstm_states(self, wrapper):
        """_setup_learn() is called in __init__; verify it doesn't reset _last_lstm_states."""
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }
        # Capture _last_lstm_states before MARLTrainer.__init__ calls _setup_learn
        pre_states = {
            aid: deepcopy(agents[aid]._last_lstm_states)
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_rppo_init",
        )
        # After __init__ (which calls _setup_learn), _last_lstm_states should still exist
        for aid in trainer._agent_ids:
            assert agents[aid]._last_lstm_states is not None
            # Shape should be preserved
            for pre_t, post_t in zip(
                pre_states[aid].pi, agents[aid]._last_lstm_states.pi
            ):
                assert pre_t.shape == post_t.shape

    def test_lstm_state_shape_matches_agent_config(self, wrapper):
        """LSTM state tensor shape should reflect (n_layers, n_envs=1, hidden_size)."""
        hidden_size = 32
        n_layers = 2
        agents = {
            aid: make_recurrent_ppo_for_agent(
                wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE,
                lstm_hidden_size=hidden_size, n_lstm_layers=n_layers, seed=0,
            )
            for aid in wrapper.base_env.possible_agents
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_rppo_init",
        )
        for aid in trainer._agent_ids:
            lstm_states = trainer._lstm_states[aid]
            # Each pi/vf tuple has (hidden, cell), each of shape (n_layers, 1, hidden_size)
            for tensor in lstm_states.pi:
                assert tensor.shape == (n_layers, 1, hidden_size)
            for tensor in lstm_states.vf:
                assert tensor.shape == (n_layers, 1, hidden_size)


# --------------------------------------------------------------------------- #
# Tests: mixed PPO + RPPO agents
# --------------------------------------------------------------------------- #

class TestMARLTrainerInitMixed:
    """Verify __init__ handles a mix of PPO and RPPO agents."""

    def test_mixed_is_recurrent(self, wrapper):
        agent_ids = list(wrapper.base_env.possible_agents)
        agents = {
            agent_ids[0]: make_ppo_for_agent(
                wrapper, agent_ids[0], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            ),
            agent_ids[1]: make_recurrent_ppo_for_agent(
                wrapper, agent_ids[1], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            ),
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_mixed_init",
        )
        assert trainer._is_recurrent[agent_ids[0]] is False
        assert trainer._is_recurrent[agent_ids[1]] is True

    def test_mixed_lstm_states(self, wrapper):
        agent_ids = list(wrapper.base_env.possible_agents)
        agents = {
            agent_ids[0]: make_ppo_for_agent(
                wrapper, agent_ids[0], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            ),
            agent_ids[1]: make_recurrent_ppo_for_agent(
                wrapper, agent_ids[1], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            ),
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_mixed_init",
        )
        assert trainer._lstm_states[agent_ids[0]] is None
        assert isinstance(trainer._lstm_states[agent_ids[1]], RNNStates)

    def test_mixed_buffer_types(self, wrapper):
        agent_ids = list(wrapper.base_env.possible_agents)
        agents = {
            agent_ids[0]: make_ppo_for_agent(
                wrapper, agent_ids[0], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            ),
            agent_ids[1]: make_recurrent_ppo_for_agent(
                wrapper, agent_ids[1], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=0,
            ),
        }
        trainer = MARLTrainer(
            wrapper=wrapper, agents=agents, n_steps=N_STEPS,
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_mixed_init",
        )
        assert isinstance(trainer._buffers[agent_ids[0]], RolloutBuffer)
        assert isinstance(trainer._buffers[agent_ids[1]], RecurrentRolloutBuffer)
