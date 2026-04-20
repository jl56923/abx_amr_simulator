"""Unit tests for make_recurrent_ppo_for_agent().

Verifies that the constructor produces a RecurrentPPO_Masked agent with the
correct policy type, LSTM dimensions, buffer type, and observation/action
spaces. Uses real ABXAMRParallelEnv, MARLOptionsWrapper, and option library
instances — no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl import MARLOptionsWrapper, OptionBase, OptionLibrary
from abx_amr_simulator.hrl.rl_algorithms.recurrent_ppo_masked import RecurrentPPO_Masked
from abx_amr_simulator.training.train_marl import (
    make_ppo_for_agent,
    make_recurrent_ppo_for_agent,
)

from test_reference_helpers import make_pg, make_rc  # type: ignore[import-not-found]


# --------------------------------------------------------------------------- #
# Minimal real option (same as test_marl_trainer_smoke.py)
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
# Fixture: build a minimal MARLOptionsWrapper
# --------------------------------------------------------------------------- #

@pytest.fixture
def wrapper():
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


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestMakeRecurrentPPOForAgent:
    """Verify make_recurrent_ppo_for_agent() produces a correctly configured agent."""

    def test_returns_recurrent_ppo_masked_instance(self, wrapper):
        """The returned object is a RecurrentPPO_Masked."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
        )
        assert isinstance(agent, RecurrentPPO_Masked)

    def test_policy_is_recurrent_actor_critic(self, wrapper):
        """The agent's policy is a RecurrentActorCriticPolicy (MlpLstmPolicy)."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
        )
        assert isinstance(agent.policy, RecurrentActorCriticPolicy)

    def test_rollout_buffer_is_recurrent(self, wrapper):
        """The agent's rollout buffer is a RecurrentRolloutBuffer."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
        )
        assert isinstance(agent.rollout_buffer, RecurrentRolloutBuffer)

    def test_rollout_buffer_size_matches_n_steps(self, wrapper):
        """The rollout buffer capacity matches the requested n_steps."""
        aid = wrapper.base_env.possible_agents[0]
        n_steps = 32
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=n_steps, batch_size=8, seed=0,
        )
        assert agent.rollout_buffer.buffer_size == n_steps

    def test_lstm_hidden_size_default(self, wrapper):
        """Default LSTM hidden size is 64."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
        )
        assert agent.policy.lstm_actor.hidden_size == 64

    def test_lstm_hidden_size_custom(self, wrapper):
        """Custom LSTM hidden size is respected."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8,
            lstm_hidden_size=128, seed=0,
        )
        assert agent.policy.lstm_actor.hidden_size == 128

    def test_n_lstm_layers_default(self, wrapper):
        """Default number of LSTM layers is 1."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
        )
        assert agent.policy.lstm_actor.num_layers == 1

    def test_n_lstm_layers_custom(self, wrapper):
        """Custom number of LSTM layers is respected."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8,
            n_lstm_layers=2, seed=0,
        )
        assert agent.policy.lstm_actor.num_layers == 2

    def test_critic_lstm_enabled_by_default(self, wrapper):
        """The critic LSTM is enabled by default (separate from actor)."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
        )
        assert agent.policy.lstm_critic is not None

    def test_critic_lstm_can_be_disabled(self, wrapper):
        """Setting enable_critic_lstm=False results in no critic LSTM."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8,
            enable_critic_lstm=False, seed=0,
        )
        # When critic LSTM is disabled, the policy uses a shared LSTM or
        # feedforward critic. Either way, lstm_critic should be None.
        assert agent.policy.lstm_critic is None

    def test_observation_space_matches_wrapper(self, wrapper):
        """The agent's observation space matches the wrapper's space for that agent."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
        )
        expected_space = wrapper.observation_spaces[aid]
        assert agent.observation_space.shape == expected_space.shape

    def test_action_space_matches_wrapper(self, wrapper):
        """The agent's action space matches the wrapper's space for that agent."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
        )
        expected_space = wrapper.action_spaces[aid]
        assert agent.action_space.n == expected_space.n

    def test_last_lstm_states_initialized(self, wrapper):
        """The agent's _last_lstm_states is initialized (not None) after construction."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
        )
        assert agent._last_lstm_states is not None

    def test_ppo_hyperparams_passed_through(self, wrapper):
        """PPO hyperparameters (learning_rate, gamma, etc.) are set correctly."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8,
            learning_rate=1e-4, gamma=0.95, ent_coef=0.05, seed=0,
        )
        assert agent.learning_rate == 1e-4
        assert agent.gamma == 0.95
        assert agent.ent_coef == 0.05

    def test_both_agents_can_be_constructed(self, wrapper):
        """make_recurrent_ppo_for_agent works for all agents in the wrapper."""
        for aid in wrapper.base_env.possible_agents:
            agent = make_recurrent_ppo_for_agent(
                wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=0,
            )
            assert isinstance(agent, RecurrentPPO_Masked)

    def test_predict_produces_valid_action(self, wrapper):
        """The agent can predict an action from a real observation."""
        aid = wrapper.base_env.possible_agents[0]
        agent = make_recurrent_ppo_for_agent(
            wrapper=wrapper, agent_id=aid, n_steps=16, batch_size=8, seed=42,
        )

        obs_dict, _ = wrapper.reset()
        obs = obs_dict[aid][np.newaxis, :]
        action, state = agent.policy.predict(obs, deterministic=True)

        n_options = wrapper.action_spaces[aid].n
        action_int = int(action.item()) if hasattr(action, 'item') else int(action)
        assert 0 <= action_int < n_options
