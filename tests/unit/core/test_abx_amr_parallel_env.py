"""
Unit tests for ABXAMRParallelEnv
=================================

Validates the PettingZoo ParallelEnv multi-agent environment:
- Observation and action space shapes per agent
- reset() returns correct observation shapes
- step() returns per-agent rewards and observations with correct shapes
- Shared AMR dynamics reflect combined prescriptions from all agents
- Episode terminates at max_time_steps
- agents list is cleared on episode end
- Heterogeneous agent configs (different n_patients) are handled correctly
"""

import numpy as np
import pytest

from abx_amr_simulator.core import (
    ABXAMRParallelEnv,
    PatientGenerator,
    RewardCalculator,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_rc_config(abx_names=("A", "B")):
    return {
        "abx_clinical_reward_penalties_info_dict": {
            "clinical_benefit_reward": 10.0,
            "clinical_benefit_probability": 0.7,
            "clinical_failure_penalty": -5.0,
            "clinical_failure_probability": 0.1,
            "abx_adverse_effects_info": {
                abx: {"adverse_effect_penalty": -1.0, "adverse_effect_probability": 0.05}
                for abx in abx_names
            },
        },
        "lambda_weight": 0.3,
        "seed": 42,
    }


def _make_pg_config():
    return {
        "prob_infected": {
            "prob_dist": {"type": "constant", "value": 0.7, "mu": None, "sigma": None},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, 1.0],
        },
        "benefit_value_multiplier": {
            "prob_dist": {"type": "constant", "value": 1.0, "mu": None, "sigma": None},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "failure_value_multiplier": {
            "prob_dist": {"type": "constant", "value": 1.0, "mu": None, "sigma": None},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "benefit_probability_multiplier": {
            "prob_dist": {"type": "constant", "value": 0.7, "mu": None, "sigma": None},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "failure_probability_multiplier": {
            "prob_dist": {"type": "constant", "value": 0.9, "mu": None, "sigma": None},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "recovery_without_treatment_prob": {
            "prob_dist": {"type": "constant", "value": 0.05, "mu": None, "sigma": None},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, 1.0],
        },
        "visible_patient_attributes": ["prob_infected"],
        "seed": 42,
    }


def _make_shared_config(max_time_steps=10, abx_names=("A", "B")):
    return {
        "antibiotics_AMR_dict": {
            abx: {
                "leak": 0.05,
                "flatness_parameter": 1.0,
                "permanent_residual_volume": 0.0,
                "initial_amr_level": 0.1,
            }
            for abx in abx_names
        },
        "max_time_steps": max_time_steps,
        "update_visible_AMR_levels_every_n_timesteps": 1,
    }


def _make_two_agent_env(n_p=6, n_n=14, max_time_steps=10, seed=0):
    """Create a standard two-agent env (agent_p, agent_n) with real components."""
    abx_names = ("A", "B")
    agent_configs = [
        {
            "agent_id": "agent_p",
            "n_patients": n_p,
            "patient_generator": PatientGenerator(config=_make_pg_config()),
            "reward_calculator": RewardCalculator(config=_make_rc_config(abx_names)),
        },
        {
            "agent_id": "agent_n",
            "n_patients": n_n,
            "patient_generator": PatientGenerator(config=_make_pg_config()),
            "reward_calculator": RewardCalculator(config=_make_rc_config(abx_names)),
        },
    ]
    return ABXAMRParallelEnv(
        agent_configs=agent_configs,
        shared_env_config=_make_shared_config(max_time_steps=max_time_steps, abx_names=abx_names),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Observation and action space shapes
# ---------------------------------------------------------------------------

def test_observation_space_shapes():
    """Each agent's obs space = n_patients * n_visible_attrs + n_abx."""
    env = _make_two_agent_env(n_p=6, n_n=14)
    # 1 visible attr (prob_infected), 2 antibiotics
    assert env.observation_spaces["agent_p"].shape == (6 * 1 + 2,)
    assert env.observation_spaces["agent_n"].shape == (14 * 1 + 2,)


def test_action_space_shapes():
    """Each agent's action space = MultiDiscrete([n_abx+1] * n_patients)."""
    env = _make_two_agent_env(n_p=6, n_n=14)
    assert env.action_spaces["agent_p"].shape == (6,)
    assert env.action_spaces["agent_n"].shape == (14,)
    # Each entry should allow 3 choices: no_treatment + 2 antibiotics
    assert all(v == 3 for v in env.action_spaces["agent_p"].nvec)
    assert all(v == 3 for v in env.action_spaces["agent_n"].nvec)


def test_observation_space_method_matches_dict():
    """observation_space(agent) method returns same space as observation_spaces dict."""
    env = _make_two_agent_env()
    for aid in env.possible_agents:
        assert env.observation_space(aid) is env.observation_spaces[aid]


def test_action_space_method_matches_dict():
    """action_space(agent) method returns same space as action_spaces dict."""
    env = _make_two_agent_env()
    for aid in env.possible_agents:
        assert env.action_space(aid) is env.action_spaces[aid]


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_returns_correct_obs_shapes():
    """reset() returns obs arrays matching each agent's observation space."""
    env = _make_two_agent_env(n_p=6, n_n=14)
    obs, info = env.reset()

    assert set(obs.keys()) == {"agent_p", "agent_n"}
    assert obs["agent_p"].shape == env.observation_spaces["agent_p"].shape
    assert obs["agent_n"].shape == env.observation_spaces["agent_n"].shape


def test_reset_restores_agents_list():
    """After running to termination, reset() repopulates the agents list."""
    env = _make_two_agent_env(max_time_steps=2)
    env.reset()
    actions = {
        "agent_p": np.zeros(6, dtype=int),
        "agent_n": np.zeros(14, dtype=int),
    }
    env.step(actions)
    env.step(actions)  # episode ends
    assert env.agents == []

    env.reset()
    assert set(env.agents) == {"agent_p", "agent_n"}


def test_reset_is_deterministic_with_seed():
    """Two resets with the same seed produce identical observations."""
    env = _make_two_agent_env()
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    for aid in env.possible_agents:
        np.testing.assert_array_equal(obs1[aid], obs2[aid])


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

def test_step_returns_correct_shapes():
    """step() returns obs, reward, termination, truncation, info with correct keys and shapes."""
    env = _make_two_agent_env(n_p=6, n_n=14)
    env.reset()
    no_treatment = 2  # With 2 antibiotics: A=0, B=1, no_treatment=2
    actions = {
        "agent_p": np.full(6, no_treatment, dtype=int),
        "agent_n": np.full(14, no_treatment, dtype=int),
    }
    obs, rewards, terminations, truncations, infos = env.step(actions)

    for aid in ["agent_p", "agent_n"]:
        assert aid in obs
        assert obs[aid].shape == env.observation_spaces[aid].shape
        assert isinstance(rewards[aid], float)
        assert isinstance(terminations[aid], bool)
        assert isinstance(truncations[aid], bool)
        assert "visible_amr_levels" in infos[aid]


def test_episode_terminates_at_max_time_steps():
    """truncated=True is returned for all agents when max_time_steps is reached."""
    env = _make_two_agent_env(max_time_steps=3)
    env.reset()
    no_treatment = 2  # With 2 antibiotics: A=0, B=1, no_treatment=2
    actions = {
        "agent_p": np.full(6, no_treatment, dtype=int),
        "agent_n": np.full(14, no_treatment, dtype=int),
    }
    truncated_step = None
    for step_idx in range(1, 5):
        _, _, terminations, truncations, _ = env.step(actions)
        if any(truncations.values()):
            truncated_step = step_idx
            break

    assert truncated_step == 3, f"Expected truncation at step 3, got {truncated_step}"
    assert env.agents == []


def test_shared_amr_reflects_combined_prescriptions():
    """AMR rises more when both agents prescribe than when only one does.

    Uses small patient counts (n_p=2, n_n=3) so the balloon sigmoid does not
    saturate on a single step and the additive effect of combined prescriptions
    is numerically visible.
    """
    def _run_one_step_get_amr(prescriber_ids, n_p=2, n_n=3):
        """Run one step where given agents prescribe antibiotic A; others use no_treatment.

        With 2 antibiotics: A=0, B=1, no_treatment=2.
        """
        no_treatment = 2
        prescribe_a = 0
        env = _make_two_agent_env(n_p=n_p, n_n=n_n, seed=0)
        env.reset()
        actions = {}
        if "agent_p" in prescriber_ids:
            actions["agent_p"] = np.full(n_p, prescribe_a, dtype=int)
        else:
            actions["agent_p"] = np.full(n_p, no_treatment, dtype=int)
        if "agent_n" in prescriber_ids:
            actions["agent_n"] = np.full(n_n, prescribe_a, dtype=int)
        else:
            actions["agent_n"] = np.full(n_n, no_treatment, dtype=int)
        env.step(actions)
        return env.amr_balloon_models["A"].get_volume()

    amr_both = _run_one_step_get_amr({"agent_p", "agent_n"})
    amr_p_only = _run_one_step_get_amr({"agent_p"})
    amr_n_only = _run_one_step_get_amr({"agent_n"})
    amr_neither = _run_one_step_get_amr(set())

    assert amr_both > amr_p_only, "Combined prescriptions should raise AMR more than agent_p alone"
    assert amr_both > amr_n_only, "Combined prescriptions should raise AMR more than agent_n alone"
    assert amr_p_only > amr_neither, "Prescribing should raise AMR compared to no treatment"
    assert amr_n_only > amr_neither


def test_per_agent_rewards_are_independent():
    """Agents receive separate scalar rewards, not a shared single value."""
    env = _make_two_agent_env()
    env.reset()
    no_treatment = 2  # With 2 antibiotics: A=0, B=1, no_treatment=2
    actions = {
        "agent_p": np.full(6, no_treatment, dtype=int),
        "agent_n": np.full(14, no_treatment, dtype=int),
    }
    _, rewards, _, _, _ = env.step(actions)

    assert "agent_p" in rewards
    assert "agent_n" in rewards
    # Rewards are scalars
    assert isinstance(rewards["agent_p"], float)
    assert isinstance(rewards["agent_n"], float)


# ---------------------------------------------------------------------------
# Heterogeneous agents
# ---------------------------------------------------------------------------

def test_unequal_patient_counts_supported():
    """Agents with different n_patients produce different-shaped obs."""
    env = _make_two_agent_env(n_p=3, n_n=17)
    obs, _ = env.reset()
    assert obs["agent_p"].shape == (3 * 1 + 2,)
    assert obs["agent_n"].shape == (17 * 1 + 2,)


# ---------------------------------------------------------------------------
# Validation / error cases
# ---------------------------------------------------------------------------

def test_empty_agent_configs_raises():
    with pytest.raises((ValueError, KeyError)):
        ABXAMRParallelEnv(
            agent_configs=[],
            shared_env_config=_make_shared_config(),
        )


def test_duplicate_agent_ids_raise():
    pg = PatientGenerator(config=_make_pg_config())
    rc = RewardCalculator(config=_make_rc_config())
    with pytest.raises(ValueError, match="unique"):
        ABXAMRParallelEnv(
            agent_configs=[
                {"agent_id": "agent_a", "n_patients": 5, "patient_generator": pg, "reward_calculator": rc},
                {"agent_id": "agent_a", "n_patients": 5, "patient_generator": pg, "reward_calculator": rc},
            ],
            shared_env_config=_make_shared_config(),
        )


def test_crossresistance_matrix_applied():
    """Off-diagonal crossresistance causes antibiotic A prescriptions to also raise AMR for B.

    Uses n_patients=3 to keep prescription counts low enough that the sigmoid
    balloon does not saturate — preserving the inequality amr_b < amr_a.
    """
    abx_names = ("A", "B")
    agent_configs = [
        {
            "agent_id": "agent_p",
            "n_patients": 3,
            "patient_generator": PatientGenerator(config=_make_pg_config()),
            "reward_calculator": RewardCalculator(config=_make_rc_config(abx_names)),
        },
    ]
    shared_config_with_cross = _make_shared_config(abx_names=abx_names)
    shared_config_with_cross["crossresistance_matrix"] = {"A": {"B": 0.5}}

    env = ABXAMRParallelEnv(
        agent_configs=agent_configs,
        shared_env_config=shared_config_with_cross,
        seed=0,
    )
    env.reset()
    # Prescribe antibiotic A (index 0) to all patients; no_treatment would be index 2
    env.step({"agent_p": np.zeros(3, dtype=int)})

    amr_a = env.amr_balloon_models["A"].get_volume()
    amr_b = env.amr_balloon_models["B"].get_volume()

    # B should have risen due to crossresistance from A prescriptions
    assert amr_b > env._antibiotics_AMR_dict["B"]["initial_amr_level"], (
        "Crossresistance should cause antibiotic B's AMR to rise when A is prescribed"
    )
    # B's AMR should be less than A's (crossresistance ratio = 0.5)
    assert amr_b < amr_a, "B's AMR rise should be less than A's (crossresistance ratio 0.5)"
