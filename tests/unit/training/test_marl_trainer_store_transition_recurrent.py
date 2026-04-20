"""Unit tests for MARLTrainer._store_transition() with recurrent agents (Phase 2c).

Verifies that _store_transition() correctly passes LSTM states to
predict_values(), evaluate_actions(), and buffer.add() for recurrent agents,
while leaving the non-recurrent path unchanged.
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


N_STEPS = 16
BATCH_SIZE = 8
TOTAL_STEPS = 1000


def _make_rppo_trainer(wrapper):
    """Build a MARLTrainer with all-RPPO agents."""
    agents = {
        aid: make_recurrent_ppo_for_agent(
            wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
        )
        for aid in wrapper.base_env.possible_agents
    }
    return MARLTrainer(
        wrapper=wrapper, agents=agents, n_steps=N_STEPS,
        total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_store_rppo",
    )


def _make_ppo_trainer(wrapper):
    """Build a MARLTrainer with all-PPO agents."""
    agents = {
        aid: make_ppo_for_agent(
            wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
        )
        for aid in wrapper.base_env.possible_agents
    }
    return MARLTrainer(
        wrapper=wrapper, agents=agents, n_steps=N_STEPS,
        total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_store_ppo",
    )


def _make_mixed_trainer(wrapper):
    """Build a MARLTrainer with agent_0=PPO, agent_1=RPPO."""
    agent_ids = list(wrapper.base_env.possible_agents)
    agents = {
        agent_ids[0]: make_ppo_for_agent(
            wrapper, agent_ids[0], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
        ),
        agent_ids[1]: make_recurrent_ppo_for_agent(
            wrapper, agent_ids[1], n_steps=N_STEPS, batch_size=BATCH_SIZE, seed=42,
        ),
    }
    return MARLTrainer(
        wrapper=wrapper, agents=agents, n_steps=N_STEPS,
        total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_store_mixed",
    )


def _do_predict_and_get_obs(trainer, wrapper):
    """Reset the wrapper, run _predict_all, and return obs_dict + selections."""
    obs_dict, _ = wrapper.reset()
    episode_starts = {aid: True for aid in trainer._agent_ids}
    selections = trainer._predict_all(obs_dict, episode_starts)
    return obs_dict, selections


# --------------------------------------------------------------------------- #
# Tests: _store_transition with RPPO agents
# --------------------------------------------------------------------------- #

class TestStoreTransitionRPPO:
    """Verify _store_transition works for recurrent agents."""

    def test_buffer_position_advances(self, wrapper):
        """After storing one transition, the buffer position should advance."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, selections = _do_predict_and_get_obs(trainer, wrapper)
        aid = trainer._agent_ids[0]

        assert trainer._buffers[aid].pos == 0
        trainer._store_transition(
            aid=aid,
            obs=obs_dict[aid],
            action=selections[aid],
            reward=1.0,
            episode_start=True,
            m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
        )
        assert trainer._buffers[aid].pos == 1

    def test_lstm_states_stored_in_buffer(self, wrapper):
        """The buffer's hidden/cell state arrays should be populated after add()."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, selections = _do_predict_and_get_obs(trainer, wrapper)
        aid = trainer._agent_ids[0]

        buf = trainer._buffers[aid]
        trainer._store_transition(
            aid=aid,
            obs=obs_dict[aid],
            action=selections[aid],
            reward=1.0,
            episode_start=True,
            m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
        )

        # RecurrentRolloutBuffer stores hidden/cell states as numpy arrays
        assert hasattr(buf, "hidden_states_pi")
        assert hasattr(buf, "cell_states_pi")
        assert hasattr(buf, "hidden_states_vf")
        assert hasattr(buf, "cell_states_vf")
        # Position 0 should have been written
        assert buf.hidden_states_pi[0] is not None
        assert buf.cell_states_pi[0] is not None

    def test_stored_lstm_states_match_pre_forward_snapshot(self, wrapper):
        """The LSTM states in the buffer should match _lstm_states_at_action."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, selections = _do_predict_and_get_obs(trainer, wrapper)
        aid = trainer._agent_ids[0]

        # Capture the pre-forward snapshot that _predict_single saved
        saved_states = deepcopy(trainer._lstm_states_at_action[aid])

        trainer._store_transition(
            aid=aid,
            obs=obs_dict[aid],
            action=selections[aid],
            reward=1.0,
            episode_start=True,
            m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
        )

        buf = trainer._buffers[aid]
        # Compare buffer's stored states with the snapshot
        np.testing.assert_array_almost_equal(
            buf.hidden_states_pi[0].flatten(),
            saved_states.pi[0].cpu().numpy().flatten(),
        )
        np.testing.assert_array_almost_equal(
            buf.cell_states_pi[0].flatten(),
            saved_states.pi[1].cpu().numpy().flatten(),
        )

    def test_non_trainable_transition_skipped(self, wrapper):
        """Non-trainable transitions should not advance buffer position."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, selections = _do_predict_and_get_obs(trainer, wrapper)
        aid = trainer._agent_ids[0]

        assert trainer._buffers[aid].pos == 0
        trainer._store_transition(
            aid=aid,
            obs=obs_dict[aid],
            action=selections[aid],
            reward=1.0,
            episode_start=True,
            m_info_entry={"manager_transition_trainable": False, "option_duration": 5},
        )
        assert trainer._buffers[aid].pos == 0

    def test_multiple_transitions_accumulate(self, wrapper):
        """Multiple successive transitions should fill the buffer sequentially."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, _ = _do_predict_and_get_obs(trainer, wrapper)
        aid = trainer._agent_ids[0]

        for step in range(3):
            # Need a fresh prediction to populate _lstm_states_at_action
            trainer._predict_single(aid, obs_dict[aid], episode_start=(step == 0))
            trainer._store_transition(
                aid=aid,
                obs=obs_dict[aid],
                action=0,
                reward=float(step),
                episode_start=(step == 0),
                m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
            )
        assert trainer._buffers[aid].pos == 3


# --------------------------------------------------------------------------- #
# Tests: _store_transition with PPO agents (backward compatibility)
# --------------------------------------------------------------------------- #

class TestStoreTransitionPPO:
    """Verify _store_transition still works for non-recurrent agents."""

    def test_buffer_position_advances(self, wrapper):
        trainer = _make_ppo_trainer(wrapper)
        obs_dict, selections = _do_predict_and_get_obs(trainer, wrapper)
        aid = trainer._agent_ids[0]

        assert trainer._buffers[aid].pos == 0
        trainer._store_transition(
            aid=aid,
            obs=obs_dict[aid],
            action=selections[aid],
            reward=1.0,
            episode_start=True,
            m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
        )
        assert trainer._buffers[aid].pos == 1

    def test_non_trainable_transition_skipped(self, wrapper):
        trainer = _make_ppo_trainer(wrapper)
        obs_dict, selections = _do_predict_and_get_obs(trainer, wrapper)
        aid = trainer._agent_ids[0]

        trainer._store_transition(
            aid=aid,
            obs=obs_dict[aid],
            action=selections[aid],
            reward=1.0,
            episode_start=True,
            m_info_entry={"manager_transition_trainable": False, "option_duration": 5},
        )
        assert trainer._buffers[aid].pos == 0


# --------------------------------------------------------------------------- #
# Tests: mixed PPO + RPPO
# --------------------------------------------------------------------------- #

class TestStoreTransitionMixed:
    """Verify _store_transition works with mixed agent types."""

    def test_both_agents_can_store(self, wrapper):
        trainer = _make_mixed_trainer(wrapper)
        obs_dict, selections = _do_predict_and_get_obs(trainer, wrapper)

        for aid in trainer._agent_ids:
            trainer._store_transition(
                aid=aid,
                obs=obs_dict[aid],
                action=selections[aid],
                reward=1.0,
                episode_start=True,
                m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
            )
            assert trainer._buffers[aid].pos == 1

    def test_ppo_buffer_has_no_lstm_states(self, wrapper):
        """PPO agent's RolloutBuffer should not have LSTM state arrays."""
        trainer = _make_mixed_trainer(wrapper)
        obs_dict, selections = _do_predict_and_get_obs(trainer, wrapper)
        ppo_aid = trainer._agent_ids[0]  # PPO agent

        trainer._store_transition(
            aid=ppo_aid,
            obs=obs_dict[ppo_aid],
            action=selections[ppo_aid],
            reward=1.0,
            episode_start=True,
            m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
        )
        assert not hasattr(trainer._buffers[ppo_aid], "hidden_states_pi")

    def test_rppo_buffer_has_lstm_states(self, wrapper):
        """RPPO agent's RecurrentRolloutBuffer should have LSTM state arrays."""
        trainer = _make_mixed_trainer(wrapper)
        obs_dict, selections = _do_predict_and_get_obs(trainer, wrapper)
        rppo_aid = trainer._agent_ids[1]  # RPPO agent

        trainer._store_transition(
            aid=rppo_aid,
            obs=obs_dict[rppo_aid],
            action=selections[rppo_aid],
            reward=1.0,
            episode_start=True,
            m_info_entry={"manager_transition_trainable": True, "option_duration": 5},
        )
        buf = trainer._buffers[rppo_aid]
        assert hasattr(buf, "hidden_states_pi")
        assert buf.hidden_states_pi[0] is not None
