"""Unit tests for granular trajectory logging in ABXAMRParallelEnv.

Verifies that:
- episode_log is empty when save_granular_trajectories is False (default)
- episode_log is populated with one entry per agent per step when True
- Each log entry has 'true' and 'observed' sub-dicts with per-patient lists
- Log is cleared on reset()
- patient_full_data is included in step() infos when logging is enabled
- Schema matches the single-agent ABXAMREnv format
"""

from __future__ import annotations

import numpy as np
import pytest

from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv

from test_reference_helpers import make_pg, make_rc  # type: ignore[import-not-found]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_ABX = ["A", "B"]
_N_PATIENTS = [3, 4]


def _make_env(seed: int = 0) -> ABXAMRParallelEnv:
    """Build a minimal two-agent parallel env."""
    agent_configs = []
    for i, n in enumerate(_N_PATIENTS):
        pg = make_pg()
        pg.visible_patient_attributes = ["prob_infected"]
        rc = make_rc(antibiotic_names=_ABX)
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
            for name in _ABX
        },
        "max_time_steps": 10,
    }

    return ABXAMRParallelEnv(
        agent_configs=agent_configs,
        shared_env_config=shared_env_config,
        seed=seed,
    )


def _make_random_actions(env: ABXAMRParallelEnv, rng: np.random.Generator) -> dict:
    return {
        aid: rng.integers(0, env.action_spaces[aid].nvec[0], size=env.action_spaces[aid].shape)
        for aid in env.agents
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestGranularLoggingDisabled:
    """episode_log is empty when save_granular_trajectories is False (default)."""

    def test_default_flag_is_false(self):
        env = _make_env()
        assert env.save_granular_trajectories is False

    def test_episode_log_empty_after_reset(self):
        env = _make_env()
        env.reset()
        for aid in env.possible_agents:
            assert env.episode_log[aid] == []

    def test_episode_log_stays_empty_after_steps(self):
        env = _make_env()
        rng = np.random.default_rng(1)
        env.reset()
        for _ in range(3):
            actions = _make_random_actions(env, rng)
            env.step(actions)
        for aid in env.possible_agents:
            assert env.episode_log[aid] == []

    def test_infos_have_no_patient_full_data(self):
        env = _make_env()
        rng = np.random.default_rng(2)
        env.reset()
        actions = _make_random_actions(env, rng)
        _, _, _, _, infos = env.step(actions)
        for aid in env.possible_agents:
            assert "patient_full_data" not in infos[aid]


class TestGranularLoggingEnabled:
    """episode_log is populated correctly when save_granular_trajectories is True."""

    def test_episode_log_has_one_entry_per_step(self):
        env = _make_env()
        env.save_granular_trajectories = True
        rng = np.random.default_rng(3)
        env.reset()

        n_steps = 4
        for _ in range(n_steps):
            actions = _make_random_actions(env, rng)
            env.step(actions)

        for aid in env.possible_agents:
            assert len(env.episode_log[aid]) == n_steps, (
                f"Expected {n_steps} log entries for {aid}, "
                f"got {len(env.episode_log[aid])}"
            )

    def test_log_entries_have_true_and_observed_keys(self):
        env = _make_env()
        env.save_granular_trajectories = True
        rng = np.random.default_rng(4)
        env.reset()
        actions = _make_random_actions(env, rng)
        env.step(actions)

        for aid in env.possible_agents:
            entry = env.episode_log[aid][0]
            assert "true" in entry, f"Missing 'true' key for {aid}"
            assert "observed" in entry, f"Missing 'observed' key for {aid}"

    def test_log_entry_true_attrs_have_correct_patient_count(self):
        env = _make_env()
        env.save_granular_trajectories = True
        rng = np.random.default_rng(5)
        env.reset()
        actions = _make_random_actions(env, rng)
        env.step(actions)

        for i, aid in enumerate(env.possible_agents):
            expected_n = _N_PATIENTS[i]
            entry = env.episode_log[aid][0]
            for attr_name, values in entry["true"].items():
                assert len(values) == expected_n, (
                    f"Agent {aid}, attr '{attr_name}': expected {expected_n} "
                    f"patients, got {len(values)}"
                )

    def test_log_entry_observed_attrs_have_correct_patient_count(self):
        env = _make_env()
        env.save_granular_trajectories = True
        rng = np.random.default_rng(6)
        env.reset()
        actions = _make_random_actions(env, rng)
        env.step(actions)

        for i, aid in enumerate(env.possible_agents):
            expected_n = _N_PATIENTS[i]
            entry = env.episode_log[aid][0]
            for attr_name, values in entry["observed"].items():
                assert len(values) == expected_n, (
                    f"Agent {aid}, attr '{attr_name}': expected {expected_n} "
                    f"patients, got {len(values)}"
                )

    def test_patient_full_data_in_step_infos(self):
        env = _make_env()
        env.save_granular_trajectories = True
        rng = np.random.default_rng(7)
        env.reset()
        actions = _make_random_actions(env, rng)
        _, _, _, _, infos = env.step(actions)

        for aid in env.possible_agents:
            assert "patient_full_data" in infos[aid], (
                f"Expected patient_full_data in infos for {aid}"
            )
            pfd = infos[aid]["patient_full_data"]
            assert "true" in pfd and "observed" in pfd

    def test_patient_full_data_in_infos_matches_episode_log(self):
        """Info dict and episode_log entry point to the same data."""
        env = _make_env()
        env.save_granular_trajectories = True
        rng = np.random.default_rng(8)
        env.reset()
        actions = _make_random_actions(env, rng)
        _, _, _, _, infos = env.step(actions)

        for aid in env.possible_agents:
            log_entry = env.episode_log[aid][0]
            info_entry = infos[aid]["patient_full_data"]
            assert log_entry == info_entry, (
                f"episode_log and infos['patient_full_data'] differ for {aid}"
            )

    def test_log_cleared_on_reset(self):
        env = _make_env()
        env.save_granular_trajectories = True
        rng = np.random.default_rng(9)
        env.reset()
        actions = _make_random_actions(env, rng)
        env.step(actions)
        env.step(actions)

        # Confirm log has entries before reset
        for aid in env.possible_agents:
            assert len(env.episode_log[aid]) > 0

        env.reset()

        for aid in env.possible_agents:
            assert env.episode_log[aid] == [], (
                f"episode_log not cleared on reset for {aid}"
            )

    def test_log_grows_over_full_episode(self):
        """episode_log length equals max_time_steps after a complete episode."""
        max_steps = 5
        env = _make_env()
        env.save_granular_trajectories = True
        rng = np.random.default_rng(10)
        env.reset()

        # Force max_time_steps to 5 for this test
        env.max_time_steps = max_steps

        done = False
        step_count = 0
        while not done:
            actions = _make_random_actions(env, rng)
            _, _, m_term, m_trunc, _ = env.step(actions)
            step_count += 1
            done = any(m_term.values()) or any(m_trunc.values())

        assert step_count == max_steps
        for aid in env.possible_agents:
            assert len(env.episode_log[aid]) == max_steps, (
                f"Expected {max_steps} log entries for {aid}, "
                f"got {len(env.episode_log[aid])}"
            )

    def test_prob_infected_values_are_in_unit_interval(self):
        """True prob_infected values are in [0, 1]."""
        env = _make_env()
        env.save_granular_trajectories = True
        rng = np.random.default_rng(11)
        env.reset()
        actions = _make_random_actions(env, rng)
        env.step(actions)

        for aid in env.possible_agents:
            entry = env.episode_log[aid][0]
            if "prob_infected" in entry["true"]:
                for val in entry["true"]["prob_infected"]:
                    assert 0.0 <= val <= 1.0, (
                        f"prob_infected out of [0,1] for {aid}: {val}"
                    )
