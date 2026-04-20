"""Unit tests for MARLTrainer._predict_single() and _predict_all() with recurrent agents (Phase 2b).

Verifies that _predict_single() correctly dispatches between policy.predict()
(PPO) and policy.forward() (RecurrentPPO), manages LSTM state tracking, and
that _predict_all() aggregates results for all agents.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

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
        total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_predict_rppo",
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
        total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_predict_ppo",
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
        total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_predict_mixed",
    )


# --------------------------------------------------------------------------- #
# Tests: _predict_single with PPO agents
# --------------------------------------------------------------------------- #

class TestPredictSinglePPO:
    """Verify _predict_single works for non-recurrent agents."""

    def test_returns_valid_action(self, wrapper):
        trainer = _make_ppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        aid = trainer._agent_ids[0]
        action = trainer._predict_single(aid, obs_dict[aid], episode_start=True)
        n_options = wrapper.action_spaces[aid].n
        assert isinstance(action, int)
        assert 0 <= action < n_options

    def test_lstm_states_unchanged(self, wrapper):
        """For PPO agents, _lstm_states should remain None."""
        trainer = _make_ppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        aid = trainer._agent_ids[0]
        trainer._predict_single(aid, obs_dict[aid], episode_start=True)
        assert trainer._lstm_states[aid] is None

    def test_lstm_states_at_action_unchanged(self, wrapper):
        """For PPO agents, _lstm_states_at_action should remain None."""
        trainer = _make_ppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        aid = trainer._agent_ids[0]
        trainer._predict_single(aid, obs_dict[aid], episode_start=True)
        assert trainer._lstm_states_at_action[aid] is None


# --------------------------------------------------------------------------- #
# Tests: _predict_single with RPPO agents
# --------------------------------------------------------------------------- #

class TestPredictSingleRPPO:
    """Verify _predict_single works for recurrent agents."""

    def test_returns_valid_action(self, wrapper):
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        aid = trainer._agent_ids[0]
        action = trainer._predict_single(aid, obs_dict[aid], episode_start=True)
        n_options = wrapper.action_spaces[aid].n
        assert isinstance(action, int)
        assert 0 <= action < n_options

    def test_lstm_states_updated_after_prediction(self, wrapper):
        """LSTM states should change after a forward pass (not remain all zeros)."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        aid = trainer._agent_ids[0]

        states_before = deepcopy(trainer._lstm_states[aid])
        trainer._predict_single(aid, obs_dict[aid], episode_start=False)
        states_after = trainer._lstm_states[aid]

        # At least one tensor in the LSTM states should have changed
        # (the observation is non-trivial, so the LSTM should produce
        # different hidden states than all-zeros)
        changed = False
        for before_t, after_t in zip(states_before.pi, states_after.pi):
            if not torch.equal(before_t, after_t):
                changed = True
                break
        if not changed:
            for before_t, after_t in zip(states_before.vf, states_after.vf):
                if not torch.equal(before_t, after_t):
                    changed = True
                    break
        assert changed, "LSTM states should change after forward pass"

    def test_lstm_states_at_action_set_after_prediction(self, wrapper):
        """_lstm_states_at_action should be populated after _predict_single."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        aid = trainer._agent_ids[0]

        assert trainer._lstm_states_at_action[aid] is None
        trainer._predict_single(aid, obs_dict[aid], episode_start=True)
        assert trainer._lstm_states_at_action[aid] is not None
        assert isinstance(trainer._lstm_states_at_action[aid], RNNStates)

    def test_lstm_states_at_action_captures_pre_forward_states(self, wrapper):
        """_lstm_states_at_action should hold the states BEFORE the forward pass."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        aid = trainer._agent_ids[0]

        # States start as zeros
        states_before = deepcopy(trainer._lstm_states[aid])
        trainer._predict_single(aid, obs_dict[aid], episode_start=False)

        # _lstm_states_at_action should match the pre-forward (zero) states
        saved_states = trainer._lstm_states_at_action[aid]
        for saved_t, before_t in zip(saved_states.pi, states_before.pi):
            assert torch.equal(saved_t, before_t)
        for saved_t, before_t in zip(saved_states.vf, states_before.vf):
            assert torch.equal(saved_t, before_t)

    def test_lstm_states_at_action_is_independent_copy(self, wrapper):
        """_lstm_states_at_action should not alias _lstm_states."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        aid = trainer._agent_ids[0]

        trainer._predict_single(aid, obs_dict[aid], episode_start=True)

        # Mutate _lstm_states — should not affect _lstm_states_at_action
        trainer._lstm_states[aid].pi[0].fill_(999.0)
        assert not torch.all(trainer._lstm_states_at_action[aid].pi[0] == 999.0)

    def test_successive_predictions_accumulate_lstm_state(self, wrapper):
        """Two successive predictions should produce different LSTM state snapshots."""
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        aid = trainer._agent_ids[0]

        trainer._predict_single(aid, obs_dict[aid], episode_start=True)
        states_after_first = deepcopy(trainer._lstm_states[aid])
        saved_at_first = deepcopy(trainer._lstm_states_at_action[aid])

        trainer._predict_single(aid, obs_dict[aid], episode_start=False)
        saved_at_second = trainer._lstm_states_at_action[aid]

        # The second call's "at action" snapshot should equal the post-first states
        for s2_t, s1_t in zip(saved_at_second.pi, states_after_first.pi):
            assert torch.equal(s2_t, s1_t)


# --------------------------------------------------------------------------- #
# Tests: _predict_all
# --------------------------------------------------------------------------- #

class TestPredictAll:
    """Verify _predict_all aggregates _predict_single across agents."""

    def test_returns_all_agents_rppo(self, wrapper):
        trainer = _make_rppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        episode_starts = {aid: True for aid in trainer._agent_ids}
        selections = trainer._predict_all(obs_dict, episode_starts)
        assert set(selections.keys()) == set(trainer._agent_ids)
        for aid, action in selections.items():
            n_options = wrapper.action_spaces[aid].n
            assert 0 <= action < n_options

    def test_returns_all_agents_ppo(self, wrapper):
        trainer = _make_ppo_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        episode_starts = {aid: True for aid in trainer._agent_ids}
        selections = trainer._predict_all(obs_dict, episode_starts)
        assert set(selections.keys()) == set(trainer._agent_ids)

    def test_mixed_agents(self, wrapper):
        trainer = _make_mixed_trainer(wrapper)
        obs_dict, _ = wrapper.reset()
        episode_starts = {aid: True for aid in trainer._agent_ids}
        selections = trainer._predict_all(obs_dict, episode_starts)

        assert set(selections.keys()) == set(trainer._agent_ids)
        # PPO agent should have None LSTM states
        assert trainer._lstm_states[trainer._agent_ids[0]] is None
        # RPPO agent should have populated LSTM states
        assert trainer._lstm_states_at_action[trainer._agent_ids[1]] is not None
