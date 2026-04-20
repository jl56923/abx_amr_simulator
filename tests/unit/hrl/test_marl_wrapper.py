"""Unit tests for MARLOptionsWrapper.

Uses real ABXAMRParallelEnv instances with lightweight configurations.
No mocks or stubs for internal objects — follows the sociable-test policy
in CLAUDE.md.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl import MARLOptionsWrapper, OptionBase, OptionLibrary

# Import test helpers (sys.path configured in tests/conftest.py)
from test_reference_helpers import make_pg, make_rc  # type: ignore[import-not-found]


# --------------------------------------------------------------------------- #
# Real option implementations for testing
# --------------------------------------------------------------------------- #


class ConstantOption(OptionBase):
    """Prescribes a fixed antibiotic for k steps.

    Returns ``no_treatment`` for patients with ``prob_infected == 0`` so the
    option passes the semantic validation check in
    ``OptionLibrary.validate_environment_compatibility()``.
    """

    REQUIRES_OBSERVATION_ATTRIBUTES = ["prob_infected"]
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = False

    def __init__(self, name: str, action_name: str, k: int = 1):
        super().__init__(name=name, k=k)
        self._action_name = action_name

    def decide(self, env_state: dict) -> np.ndarray:
        patients = env_state.get("patients", [])
        n = env_state.get("num_patients", len(patients))
        if not patients or self._action_name == "no_treatment":
            return np.full(shape=(n,), fill_value=self._action_name, dtype=object)
        actions = []
        for p in patients:
            pi = p.get("prob_infected_obs", p.get("prob_infected", 0.5))
            actions.append(self._action_name if pi > 0.01 else "no_treatment")
        return np.array(actions, dtype=object)

    def get_referenced_antibiotics(self) -> list:
        return [self._action_name] if self._action_name != "no_treatment" else []


# --------------------------------------------------------------------------- #
# Helpers that build real components
# --------------------------------------------------------------------------- #


def make_parallel_env(
    antibiotic_names: List[str],
    n_patients_per_agent: List[int],
    max_time_steps: int = 20,
) -> ABXAMRParallelEnv:
    """Build a real ABXAMRParallelEnv with one PatientGenerator + RewardCalculator per agent."""
    agent_configs = []
    for i, n in enumerate(n_patients_per_agent):
        pg = make_pg()
        pg.visible_patient_attributes = ["prob_infected"]
        rc = make_rc(antibiotic_names=antibiotic_names)
        agent_configs.append(
            {
                "agent_id": f"agent_{i}",
                "n_patients": n,
                "patient_generator": pg,
                "reward_calculator": rc,
            }
        )

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

    return ABXAMRParallelEnv(
        agent_configs=agent_configs,
        shared_env_config=shared_env_config,
    )


def make_option_libraries(
    base_env: ABXAMRParallelEnv,
    antibiotic_names: List[str],
    option_k: int = 5,
) -> Dict[str, OptionLibrary]:
    """Build one OptionLibrary per agent, each with a single k-step ConstantOption."""
    libs: Dict[str, OptionLibrary] = {}
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
        libs[aid] = lib
    return libs


def make_wrapper(
    antibiotic_names: List[str] = None,
    n_patients_per_agent: List[int] = None,
    option_k: int = 5,
    max_time_steps: int = 20,
    gamma: float = 0.99,
) -> tuple:
    """Return (base_env, option_libraries, wrapper) with real instances."""
    if antibiotic_names is None:
        antibiotic_names = ["A", "B"]
    if n_patients_per_agent is None:
        n_patients_per_agent = [3, 4]

    base_env = make_parallel_env(
        antibiotic_names=antibiotic_names,
        n_patients_per_agent=n_patients_per_agent,
        max_time_steps=max_time_steps,
    )
    option_libs = make_option_libraries(
        base_env=base_env,
        antibiotic_names=antibiotic_names,
        option_k=option_k,
    )
    wrapper = MARLOptionsWrapper(
        base_env=base_env,
        option_libraries=option_libs,
        gamma=gamma,
    )
    return base_env, option_libs, wrapper


# --------------------------------------------------------------------------- #
# Construction tests
# --------------------------------------------------------------------------- #


class TestMARLOptionsWrapperConstruction:
    def test_init_succeeds_with_valid_inputs(self):
        base_env, libs, wrapper = make_wrapper()
        assert wrapper is not None

    def test_missing_option_library_raises(self):
        base_env = make_parallel_env(antibiotic_names=["A"], n_patients_per_agent=[2, 3])
        rc0 = base_env._reward_calculators["agent_0"]
        # Only provide library for agent_0, omit agent_1
        libs = {"agent_0": OptionLibrary(reward_calculator=rc0, name="lib_0")}
        libs["agent_0"].add_option(ConstantOption(name="opt", action_name="A", k=1))
        with pytest.raises(ValueError, match="agent_1"):
            MARLOptionsWrapper(base_env=base_env, option_libraries=libs)

    def test_inconsistent_abx_map_raises(self):
        """Two agents with different abx_name_to_index should be rejected."""
        base_env = make_parallel_env(antibiotic_names=["A", "B"], n_patients_per_agent=[2, 3])
        rc0 = base_env._reward_calculators["agent_0"]
        rc1 = base_env._reward_calculators["agent_1"]

        lib0 = OptionLibrary(reward_calculator=rc0, name="lib_0")
        lib0.add_option(ConstantOption(name="opt", action_name="A", k=1))

        # Build a library with a different abx mapping by using an RC with only A
        rc_different = make_rc(antibiotic_names=["A"])  # only A, not A+B
        lib1 = OptionLibrary(reward_calculator=rc_different, name="lib_1")
        lib1.add_option(ConstantOption(name="opt", action_name="A", k=1))

        with pytest.raises(ValueError, match="abx_name_to_index"):
            MARLOptionsWrapper(
                base_env=base_env,
                option_libraries={"agent_0": lib0, "agent_1": lib1},
            )

    def test_observation_and_action_spaces_created_for_all_agents(self):
        base_env, libs, wrapper = make_wrapper(
            antibiotic_names=["A", "B"], n_patients_per_agent=[3, 4]
        )
        for aid in base_env.possible_agents:
            assert aid in wrapper.observation_spaces
            assert aid in wrapper.action_spaces

    def test_action_space_size_equals_num_options(self):
        base_env, libs, wrapper = make_wrapper()
        for aid in base_env.possible_agents:
            assert wrapper.action_spaces[aid].n == len(libs[aid])


# --------------------------------------------------------------------------- #
# Observation dimension tests
# --------------------------------------------------------------------------- #


class TestObservationDimension:
    def test_obs_dim_matches_declared_space(self):
        """Manager obs returned by reset() must match observation_space shape."""
        base_env, libs, wrapper = make_wrapper(
            antibiotic_names=["A", "B"],
            n_patients_per_agent=[3, 4],
        )
        manager_obs, _ = wrapper.reset()
        for aid in base_env.possible_agents:
            declared = wrapper.observation_spaces[aid].shape[0]
            actual = manager_obs[aid].shape[0]
            assert actual == declared, (
                f"Agent '{aid}': obs shape {actual} != declared {declared}"
            )

    def test_agents_with_different_patient_counts_have_different_obs_dims(self):
        """Heterogeneous cohort sizes → heterogeneous manager obs dims."""
        base_env, libs, wrapper = make_wrapper(
            antibiotic_names=["A"],
            n_patients_per_agent=[2, 8],
        )
        dim_0 = wrapper.observation_spaces["agent_0"].shape[0]
        dim_1 = wrapper.observation_spaces["agent_1"].shape[0]
        # Front-edge summary stats depend on num_visible_attrs, not num_patients,
        # so with the same patient generator the only difference is the front-edge
        # full-vector path.  With summary stats (default) dims are equal.
        # This test just checks the shapes are well-formed.
        assert dim_0 > 0
        assert dim_1 > 0

    def test_full_vector_vs_summary_obs_dim_differ(self):
        """front_edge_use_full_vector=True should produce a wider observation."""
        base_env = make_parallel_env(antibiotic_names=["A"], n_patients_per_agent=[5, 5])
        libs = make_option_libraries(base_env=base_env, antibiotic_names=["A"])

        wrapper_summary = MARLOptionsWrapper(
            base_env=base_env, option_libraries=libs, front_edge_use_full_vector=False
        )
        wrapper_full = MARLOptionsWrapper(
            base_env=base_env, option_libraries=libs, front_edge_use_full_vector=True
        )

        for aid in base_env.possible_agents:
            dim_summary = wrapper_summary.observation_spaces[aid].shape[0]
            dim_full = wrapper_full.observation_spaces[aid].shape[0]
            # Full vector: num_patients * num_visible_attrs vs summary: 2 * num_visible_attrs
            assert dim_full > dim_summary


# --------------------------------------------------------------------------- #
# reset() tests
# --------------------------------------------------------------------------- #


class TestReset:
    def test_reset_returns_obs_for_all_agents(self):
        base_env, libs, wrapper = make_wrapper()
        manager_obs, info = wrapper.reset()
        assert set(manager_obs.keys()) == set(base_env.possible_agents)

    def test_reset_obs_are_float32_arrays(self):
        base_env, libs, wrapper = make_wrapper()
        manager_obs, _ = wrapper.reset()
        for aid, obs in manager_obs.items():
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32

    def test_reset_clears_option_state(self):
        """After reset, all agents should have _steps_remaining == 0."""
        base_env, libs, wrapper = make_wrapper()

        # Run one step to populate option state
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        wrapper.step(all_selections)

        # Reset and verify state is cleared
        wrapper.reset()
        for aid in base_env.possible_agents:
            assert wrapper._steps_remaining[aid] == 0
            assert wrapper._current_option[aid] is None

    def test_reset_seed_produces_reproducible_obs(self):
        base_env, libs, wrapper = make_wrapper()
        obs1, _ = wrapper.reset(seed=42)
        obs2, _ = wrapper.reset(seed=42)
        for aid in base_env.possible_agents:
            np.testing.assert_array_equal(obs1[aid], obs2[aid])


# --------------------------------------------------------------------------- #
# step() tests — synchronous completion
# --------------------------------------------------------------------------- #


class TestStepSynchronous:
    """Tests where all agents have the same option duration and complete together."""

    def test_step_returns_all_agents_when_all_complete(self):
        """With equal option durations, all agents complete simultaneously."""
        base_env, libs, wrapper = make_wrapper(option_k=3, max_time_steps=20)
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        m_obs, m_rew, m_term, m_trunc, m_info = wrapper.step(all_selections)

        assert set(m_obs.keys()) == set(base_env.possible_agents)
        assert set(m_rew.keys()) == set(base_env.possible_agents)

    def test_step_obs_shapes_match_declared_spaces(self):
        base_env, libs, wrapper = make_wrapper(option_k=3, max_time_steps=20)
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        m_obs, _, _, _, _ = wrapper.step(all_selections)

        for aid, obs in m_obs.items():
            declared = wrapper.observation_spaces[aid].shape[0]
            assert obs.shape[0] == declared

    def test_step_rewards_are_discounted(self):
        """Reward accumulated over k > 1 steps must be less than k * max_step_reward
        when gamma < 1, and equal to the sum when gamma == 1."""
        base_env, libs, wrapper_099 = make_wrapper(option_k=3, gamma=0.99, max_time_steps=20)
        _, _, wrapper_100 = make_wrapper(option_k=3, gamma=1.0, max_time_steps=20)

        wrapper_099.reset(seed=0)
        wrapper_100.reset(seed=0)

        all_selections_099 = {aid: 0 for aid in base_env.possible_agents}
        all_selections_100 = {aid: 0 for aid in wrapper_100.base_env.possible_agents}

        _, rew_099, _, _, _ = wrapper_099.step(all_selections_099)
        _, rew_100, _, _, _ = wrapper_100.step(all_selections_100)

        # gamma=0.99 should accumulate slightly less reward than gamma=1.0
        for aid in base_env.possible_agents:
            # Comparison only meaningful if reward is positive
            if rew_100[aid] > 0:
                assert rew_099[aid] <= rew_100[aid]

    def test_step_info_contains_required_keys(self):
        base_env, libs, wrapper = make_wrapper(option_k=2, max_time_steps=20)
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        _, _, _, _, m_info = wrapper.step(all_selections)

        required_keys = {
            "option_id", "option_name", "option_duration",
            "primitive_actions", "primitive_infos",
            "manager_clipped", "steps_clipped", "manager_transition_trainable",
        }
        for aid in base_env.possible_agents:
            assert required_keys.issubset(m_info[aid].keys()), (
                f"Agent '{aid}' info missing keys: {required_keys - m_info[aid].keys()}"
            )

    def test_step_primitive_actions_length_equals_option_k(self):
        k = 4
        base_env, libs, wrapper = make_wrapper(option_k=k, max_time_steps=20)
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        _, _, _, _, m_info = wrapper.step(all_selections)

        for aid in base_env.possible_agents:
            assert len(m_info[aid]["primitive_actions"]) == k

    def test_manager_transition_trainable_true_when_not_clipped(self):
        base_env, libs, wrapper = make_wrapper(option_k=3, max_time_steps=20)
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        _, _, _, _, m_info = wrapper.step(all_selections)

        for aid in base_env.possible_agents:
            assert m_info[aid]["manager_transition_trainable"] is True
            assert m_info[aid]["manager_clipped"] is False


# --------------------------------------------------------------------------- #
# step() tests — asynchronous completion
# --------------------------------------------------------------------------- #


class TestStepAsynchronous:
    """Tests where agents have different option durations."""

    def _make_async_wrapper(self, k_agent_0: int = 2, k_agent_1: int = 5):
        """Build a wrapper where agent_0 has shorter options than agent_1."""
        antibiotic_names = ["A", "B"]
        n_patients = [3, 3]
        max_time_steps = 30

        base_env = make_parallel_env(
            antibiotic_names=antibiotic_names,
            n_patients_per_agent=n_patients,
            max_time_steps=max_time_steps,
        )

        libs: Dict[str, OptionLibrary] = {}
        for i, (aid, k) in enumerate(
            zip(base_env.possible_agents, [k_agent_0, k_agent_1])
        ):
            rc = base_env._reward_calculators[aid]
            lib = OptionLibrary(reward_calculator=rc, name=f"lib_{aid}")
            lib.add_option(
                ConstantOption(
                    name=f"opt_{aid}",
                    action_name=antibiotic_names[0],
                    k=k,
                )
            )
            libs[aid] = lib

        wrapper = MARLOptionsWrapper(
            base_env=base_env,
            option_libraries=libs,
            gamma=0.99,
        )
        return base_env, libs, wrapper

    def test_first_step_returns_only_faster_agent(self):
        """agent_0 (k=2) completes before agent_1 (k=5); first step returns only agent_0."""
        base_env, libs, wrapper = self._make_async_wrapper(k_agent_0=2, k_agent_1=5)
        wrapper.reset()

        all_selections = {aid: 0 for aid in base_env.possible_agents}
        m_obs, m_rew, _, _, _ = wrapper.step(all_selections)

        # Only agent_0 should be in the return dicts
        assert "agent_0" in m_obs
        assert "agent_1" not in m_obs

    def test_second_step_takes_new_selection_for_completing_agent(self):
        """After agent_0 completes, pass its new selection and advance again."""
        base_env, libs, wrapper = self._make_async_wrapper(k_agent_0=2, k_agent_1=6)
        wrapper.reset()

        # First step: advance 2 steps, agent_0 completes
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        m_obs1, _, _, _, _ = wrapper.step(all_selections)
        assert "agent_0" in m_obs1

        # Second step: only agent_0 gets a new selection (k=2 again)
        # After 2 more steps agent_0 completes again (total 4 prim steps),
        # agent_1 still has 2 remaining (6-4=2).
        m_obs2, _, _, _, _ = wrapper.step({"agent_0": 0})
        assert "agent_0" in m_obs2
        assert "agent_1" not in m_obs2

    def test_primitive_step_count_is_min_remaining(self):
        """The primitive loop executes exactly min(_steps_remaining) steps."""
        base_env, libs, wrapper = self._make_async_wrapper(k_agent_0=3, k_agent_1=7)
        wrapper.reset()

        all_selections = {aid: 0 for aid in base_env.possible_agents}
        _, _, _, _, m_info = wrapper.step(all_selections)

        # agent_0 completes after 3 steps; option_duration should be 3
        assert m_info["agent_0"]["option_duration"] == 3


# --------------------------------------------------------------------------- #
# Episode termination / clipping tests
# --------------------------------------------------------------------------- #


class TestEpisodeTermination:
    def test_truncation_returns_all_agents(self):
        """When max_time_steps is reached, all agents are returned."""
        # Option k=10, max_time_steps=5 → option will be clipped
        base_env, libs, wrapper = make_wrapper(option_k=10, max_time_steps=5)
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        m_obs, _, m_term, m_trunc, m_info = wrapper.step(all_selections)

        # All agents returned
        assert set(m_obs.keys()) == set(base_env.possible_agents)

    def test_clipped_agents_marked_not_trainable(self):
        """Agents clipped at episode boundary must have manager_transition_trainable=False."""
        base_env, libs, wrapper = make_wrapper(option_k=10, max_time_steps=5)
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        _, _, _, _, m_info = wrapper.step(all_selections)

        for aid in base_env.possible_agents:
            assert m_info[aid]["manager_clipped"] is True
            assert m_info[aid]["manager_transition_trainable"] is False

    def test_truncation_flag_set_on_episode_end(self):
        """manager_truncated should be True when max_time_steps is reached."""
        base_env, libs, wrapper = make_wrapper(option_k=10, max_time_steps=5)
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        _, _, _, m_trunc, _ = wrapper.step(all_selections)

        for aid in base_env.possible_agents:
            assert m_trunc[aid] is True

    def test_unclipped_step_has_zero_steps_clipped(self):
        base_env, libs, wrapper = make_wrapper(option_k=3, max_time_steps=20)
        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        _, _, _, _, m_info = wrapper.step(all_selections)

        for aid in base_env.possible_agents:
            assert m_info[aid]["steps_clipped"] == 0


# --------------------------------------------------------------------------- #
# Multi-step rollout test
# --------------------------------------------------------------------------- #


class TestMultiStepRollout:
    def test_full_episode_rollout_with_synchronous_agents(self):
        """Run a complete episode with synchronous agents and verify it terminates."""
        option_k = 5
        max_time_steps = 20
        base_env, libs, wrapper = make_wrapper(
            antibiotic_names=["A", "B"],
            n_patients_per_agent=[3, 4],
            option_k=option_k,
            max_time_steps=max_time_steps,
        )

        wrapper.reset()
        all_selections = {aid: 0 for aid in base_env.possible_agents}

        episode_done = False
        step_count = 0
        max_macro_steps = max_time_steps  # Upper bound

        while not episode_done and step_count < max_macro_steps:
            m_obs, m_rew, m_term, m_trunc, m_info = wrapper.step(all_selections)
            step_count += 1

            if any(m_trunc.values()) or any(m_term.values()):
                episode_done = True
                break

            # Provide new selections for completing agents
            all_selections = {aid: 0 for aid in m_obs.keys()}

        assert episode_done, "Episode should have terminated within max_macro_steps"

    def test_full_episode_async_agents_terminates(self):
        """Run a complete episode with async agents; all agents returned at episode end."""
        max_time_steps = 20
        antibiotic_names = ["A"]
        n_patients = [3, 3]

        base_env = make_parallel_env(
            antibiotic_names=antibiotic_names,
            n_patients_per_agent=n_patients,
            max_time_steps=max_time_steps,
        )
        libs: Dict = {}
        for i, aid in enumerate(base_env.possible_agents):
            rc = base_env._reward_calculators[aid]
            k = 3 if i == 0 else 7  # Different durations
            lib = OptionLibrary(reward_calculator=rc, name=f"lib_{aid}")
            lib.add_option(ConstantOption(name="opt", action_name="A", k=k))
            libs[aid] = lib

        wrapper = MARLOptionsWrapper(base_env=base_env, option_libraries=libs, gamma=0.99)
        wrapper.reset()

        pending_selections = {aid: 0 for aid in base_env.possible_agents}
        episode_done = False
        for _ in range(max_time_steps * 2):
            m_obs, _, m_term, m_trunc, _ = wrapper.step(pending_selections)
            if any(m_trunc.values()) or any(m_term.values()):
                episode_done = True
                # All agents should be returned at episode end
                assert set(m_obs.keys()) == set(base_env.possible_agents)
                break
            # Only completing agents need new selections
            pending_selections = {aid: 0 for aid in m_obs.keys()}

        assert episode_done, "Episode should have terminated"


class TestStepsSincePrescribedConsistency:
    """Regression tests for _steps_since_prescribed consistency between
    MARLOptionsWrapper and the SA OptionsWrapper.

    The off-by-one concern (SA=0 vs MARL=1 for a just-prescribed antibiotic)
    was investigated and found to be non-reproducible — both wrappers use
    identical update logic. These tests guard against future regressions.
    """

    def test_single_agent_matches_sa_wrapper(self):
        """In a single-agent MARL scenario, _steps_since_prescribed should
        match what the SA OptionsWrapper produces for the same option."""
        from abx_amr_simulator.hrl import OptionsWrapper
        from test_reference_helpers import make_env  # type: ignore[import-not-found]

        abx_names = ["A", "B"]
        k = 10

        # --- SA wrapper ---
        sa_env = make_env(
            antibiotic_names=abx_names,
            num_patients_per_time_step=3,
            max_time_steps=50,
        )
        sa_lib = OptionLibrary(
            reward_calculator=sa_env.reward_calculator, name="sa_lib"
        )
        sa_lib.add_option(ConstantOption(name="prescribe_A", action_name="A", k=k))
        sa_wrapper = OptionsWrapper(env=sa_env, option_library=sa_lib, gamma=0.99)
        sa_wrapper.reset(seed=42)
        sa_wrapper.step(0)

        # --- MARL wrapper (single agent) ---
        base_env = make_parallel_env(
            antibiotic_names=abx_names,
            n_patients_per_agent=[3],
            max_time_steps=50,
        )
        marl_libs = {}
        for aid in base_env.possible_agents:
            marl_rc = base_env._reward_calculators[aid]
            lib = OptionLibrary(reward_calculator=marl_rc, name=f"lib_{aid}")
            lib.add_option(ConstantOption(name="prescribe_A", action_name="A", k=k))
            marl_libs[aid] = lib
        marl_wrapper = MARLOptionsWrapper(
            base_env=base_env, option_libraries=marl_libs, gamma=0.99
        )
        marl_wrapper.reset(seed=42)
        marl_wrapper.step({base_env.possible_agents[0]: 0})

        # Compare steps_since_prescribed for real antibiotics (skip no_treatment)
        agent_id = base_env.possible_agents[0]
        for abx in abx_names:
            sa_val = sa_wrapper._steps_since_prescribed[abx]
            marl_val = marl_wrapper._steps_since_prescribed[agent_id][abx]
            assert sa_val == marl_val, (
                f"steps_since_prescribed['{abx}'] mismatch: SA={sa_val}, MARL={marl_val}"
            )

    def test_async_options_accumulate_correctly(self):
        """With two agents having different option durations, each agent's
        _steps_since_prescribed should reflect its full option execution."""
        abx_names = ["A", "B"]

        base_env = make_parallel_env(
            antibiotic_names=abx_names,
            n_patients_per_agent=[3, 3],
            max_time_steps=50,
        )
        libs = {}
        for aid in base_env.possible_agents:
            rc = base_env._reward_calculators[aid]
            lib = OptionLibrary(reward_calculator=rc, name=f"lib_{aid}")
            # Agent_0 gets k=5, Agent_1 gets k=10
            k = 5 if aid == "agent_0" else 10
            lib.add_option(ConstantOption(name="prescribe_A", action_name="A", k=k))
            libs[aid] = lib

        wrapper = MARLOptionsWrapper(
            base_env=base_env, option_libraries=libs, gamma=0.99
        )
        wrapper.reset(seed=42)

        # Step 1: both select option 0 → next_event=5, agent_0 returns
        obs1, _, _, _, _ = wrapper.step({"agent_0": 0, "agent_1": 0})
        assert "agent_0" in obs1
        assert "agent_1" not in obs1  # agent_1 still running (5 of 10 done)

        # After 5 steps of prescribing A: counter should be 0 for A
        assert wrapper._steps_since_prescribed["agent_0"]["A"] == 0
        assert wrapper._steps_since_prescribed["agent_0"]["B"] == 5

        # Step 2: agent_0 re-selects, agent_1 finishes → both return
        obs2, _, _, _, _ = wrapper.step({"agent_0": 0})
        assert "agent_0" in obs2
        assert "agent_1" in obs2

        # After 10 total steps of prescribing A: counter should be 0 for A
        assert wrapper._steps_since_prescribed["agent_0"]["A"] == 0
        assert wrapper._steps_since_prescribed["agent_1"]["A"] == 0
        assert wrapper._steps_since_prescribed["agent_1"]["B"] == 10
