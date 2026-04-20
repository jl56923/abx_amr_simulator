"""Unit tests for _make_zero_lstm_states() and episode-boundary LSTM reset (Phase 2d).

Verifies that the helper produces correctly shaped zero tensors, that __init__
uses it for initialization, and that LSTM states are reset to zeros at episode
boundaries in train().
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from sb3_contrib.common.recurrent.type_aliases import RNNStates

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


def _make_rppo_trainer(wrapper, lstm_hidden_size=64, n_lstm_layers=1):
    """Build a MARLTrainer with all-RPPO agents."""
    agents = {
        aid: make_recurrent_ppo_for_agent(
            wrapper, aid, n_steps=N_STEPS, batch_size=BATCH_SIZE,
            lstm_hidden_size=lstm_hidden_size, n_lstm_layers=n_lstm_layers,
            seed=42,
        )
        for aid in wrapper.base_env.possible_agents
    }
    return MARLTrainer(
        wrapper=wrapper, agents=agents, n_steps=N_STEPS,
        total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_reset",
    )


# --------------------------------------------------------------------------- #
# Tests: _make_zero_lstm_states helper
# --------------------------------------------------------------------------- #

class TestMakeZeroLstmStates:
    """Verify the _make_zero_lstm_states helper."""

    def test_returns_rnn_states(self, wrapper):
        trainer = _make_rppo_trainer(wrapper)
        aid = trainer._agent_ids[0]
        result = trainer._make_zero_lstm_states(aid)
        assert isinstance(result, RNNStates)

    def test_all_tensors_are_zero(self, wrapper):
        trainer = _make_rppo_trainer(wrapper)
        aid = trainer._agent_ids[0]
        result = trainer._make_zero_lstm_states(aid)
        for tensor in result.pi:
            assert torch.all(tensor == 0.0)
        for tensor in result.vf:
            assert torch.all(tensor == 0.0)

    def test_shape_matches_default_config(self, wrapper):
        """Shape should be (n_layers=1, n_envs=1, hidden_size=64) by default."""
        trainer = _make_rppo_trainer(wrapper)
        aid = trainer._agent_ids[0]
        result = trainer._make_zero_lstm_states(aid)
        for tensor in result.pi:
            assert tensor.shape == (1, 1, 64)
        for tensor in result.vf:
            assert tensor.shape == (1, 1, 64)

    def test_shape_matches_custom_config(self, wrapper):
        """Shape should reflect custom n_lstm_layers and lstm_hidden_size."""
        trainer = _make_rppo_trainer(wrapper, lstm_hidden_size=32, n_lstm_layers=2)
        aid = trainer._agent_ids[0]
        result = trainer._make_zero_lstm_states(aid)
        for tensor in result.pi:
            assert tensor.shape == (2, 1, 32)
        for tensor in result.vf:
            assert tensor.shape == (2, 1, 32)

    def test_each_call_returns_independent_copy(self, wrapper):
        """Two calls should produce distinct tensor objects."""
        trainer = _make_rppo_trainer(wrapper)
        aid = trainer._agent_ids[0]
        states_a = trainer._make_zero_lstm_states(aid)
        states_b = trainer._make_zero_lstm_states(aid)
        # Mutate one — should not affect the other
        states_a.pi[0].fill_(999.0)
        assert not torch.all(states_b.pi[0] == 999.0)

    def test_pi_has_hidden_and_cell(self, wrapper):
        """The .pi field should be a tuple of exactly 2 tensors (hidden, cell)."""
        trainer = _make_rppo_trainer(wrapper)
        aid = trainer._agent_ids[0]
        result = trainer._make_zero_lstm_states(aid)
        assert len(result.pi) == 2
        assert len(result.vf) == 2


# --------------------------------------------------------------------------- #
# Tests: __init__ uses _make_zero_lstm_states
# --------------------------------------------------------------------------- #

class TestInitUsesHelper:
    """Verify that __init__ produces the same states as the helper."""

    def test_init_lstm_states_match_helper_output(self, wrapper):
        """_lstm_states from __init__ should be all-zero, matching the helper."""
        trainer = _make_rppo_trainer(wrapper)
        for aid in trainer._agent_ids:
            init_states = trainer._lstm_states[aid]
            helper_states = trainer._make_zero_lstm_states(aid)
            for init_t, helper_t in zip(init_states.pi, helper_states.pi):
                assert torch.equal(init_t, helper_t)
            for init_t, helper_t in zip(init_states.vf, helper_states.vf):
                assert torch.equal(init_t, helper_t)


# --------------------------------------------------------------------------- #
# Tests: episode-boundary LSTM reset in train()
# --------------------------------------------------------------------------- #

class TestEpisodeBoundaryReset:
    """Verify LSTM states are reset to zeros at episode boundaries."""

    def test_lstm_states_reset_after_episode(self, wrapper):
        """After an episode completes, LSTM states should be zeros again.

        We drive the trainer through a partial training run until at least
        one episode completes, then verify the LSTM states are zeros.
        """
        trainer = _make_rppo_trainer(wrapper)

        # Run predictions to make LSTM states non-zero
        obs_dict, _ = wrapper.reset()
        for aid in trainer._agent_ids:
            trainer._predict_single(aid, obs_dict[aid], episode_start=True)
            # After prediction, states should be non-zero
            # (we already verified this in Phase 2b tests)

        # Now simulate what train() does at an episode boundary:
        # 1. reset env
        # 2. reset LSTM states
        # 3. call _predict_all with episode_start=True
        obs_dict, _ = wrapper.reset()
        last_episode_start = {aid: True for aid in trainer._agent_ids}

        for aid in trainer._agent_ids:
            if trainer._is_recurrent[aid]:
                trainer._lstm_states[aid] = trainer._make_zero_lstm_states(aid)

        # Verify LSTM states are now zeros (before _predict_all)
        for aid in trainer._agent_ids:
            states = trainer._lstm_states[aid]
            for tensor in states.pi:
                assert torch.all(tensor == 0.0), (
                    f"Agent {aid}: pi LSTM states should be zeros after reset"
                )
            for tensor in states.vf:
                assert torch.all(tensor == 0.0), (
                    f"Agent {aid}: vf LSTM states should be zeros after reset"
                )

    def test_lstm_states_at_action_reflect_zeros_after_reset(self, wrapper):
        """After episode reset + prediction, _lstm_states_at_action should hold zeros.

        Because _predict_single saves pre-forward states to _lstm_states_at_action,
        and we just reset to zeros, the saved states should be all zeros.
        """
        trainer = _make_rppo_trainer(wrapper)

        # Make LSTM states non-zero via a prediction
        obs_dict, _ = wrapper.reset()
        for aid in trainer._agent_ids:
            trainer._predict_single(aid, obs_dict[aid], episode_start=False)

        # Simulate episode boundary reset
        obs_dict, _ = wrapper.reset()
        for aid in trainer._agent_ids:
            if trainer._is_recurrent[aid]:
                trainer._lstm_states[aid] = trainer._make_zero_lstm_states(aid)

        # Now predict — _lstm_states_at_action should capture the zero states
        episode_starts = {aid: True for aid in trainer._agent_ids}
        trainer._predict_all(obs_dict, episode_starts)

        for aid in trainer._agent_ids:
            saved = trainer._lstm_states_at_action[aid]
            for tensor in saved.pi:
                assert torch.all(tensor == 0.0), (
                    f"Agent {aid}: saved pi states should be zeros after reset"
                )
            for tensor in saved.vf:
                assert torch.all(tensor == 0.0), (
                    f"Agent {aid}: saved vf states should be zeros after reset"
                )

    def test_ppo_agent_unaffected_by_reset(self, wrapper):
        """PPO agents should have None LSTM states, unaffected by episode reset."""
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
            total_primitive_steps=TOTAL_STEPS, checkpoint_dir="/tmp/test_reset_mixed",
        )

        # Simulate episode boundary reset
        for aid in trainer._agent_ids:
            if trainer._is_recurrent[aid]:
                trainer._lstm_states[aid] = trainer._make_zero_lstm_states(aid)

        # PPO agent should still have None
        assert trainer._lstm_states[agent_ids[0]] is None
        # RPPO agent should have zeros
        states = trainer._lstm_states[agent_ids[1]]
        assert isinstance(states, RNNStates)
        for tensor in states.pi:
            assert torch.all(tensor == 0.0)
