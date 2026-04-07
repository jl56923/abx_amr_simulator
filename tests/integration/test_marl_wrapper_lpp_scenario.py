"""Integration test for MARLOptionsWrapper with real LPP option libraries.

Uses the three-antibiotic environment config and the heuristic_three_abx_mixed_vis
option library from the workspace to run a full two-agent episode through
MARLOptionsWrapper. This exercises the real heuristic option decision logic
(not synthetic ConstantOption stubs) in the full async option-execution loop.

Both agents use the same heuristic library (mixed-vis) with real PatientGenerators
and RewardCalculators built from the workspace's three-antibiotic configs. The test
validates that:
- Both option libraries load and are compatible with MARLOptionsWrapper
- Manager observations have the correct shapes throughout
- The episode terminates within max_time_steps
- Rewards accumulate without NaN/Inf
- Async option completion (10-step vs 20-step options) works end-to-end
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import yaml

from abx_amr_simulator.core import PatientGenerator, RewardCalculator
from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl import MARLOptionsWrapper, OptionLibraryLoader

# Locate the workspace root relative to this test file:
# external/abx_amr_simulator/tests/integration/ → root is 4 levels up
_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKSPACE = _REPO_ROOT / "workspace" / "experiments"

# Paths to real workspace configs and option libraries
_ENV_CONFIG_PATH = _WORKSPACE / "configs" / "environment" / (
    "three_abx_with_mod_asym_crossresistance_environment.yaml"
)
_RC_CONFIG_PATH = _WORKSPACE / "configs" / "reward_calculator" / (
    "three_abx_reward_calculator.yaml"
)
_MIXED_VIS_LIBRARY_PATH = _WORKSPACE / "options" / "option_libraries" / (
    "heuristic_three_abx_mixed_vis.yaml"
)


def _configs_available() -> bool:
    """Return True if the workspace config files exist."""
    return (
        _ENV_CONFIG_PATH.exists()
        and _RC_CONFIG_PATH.exists()
        and _MIXED_VIS_LIBRARY_PATH.exists()
    )


pytestmark = pytest.mark.skipif(
    not _configs_available(),
    reason="Workspace LPP configs not found; skipping LPP integration test.",
)


def _build_three_abx_parallel_env(
    max_time_steps: int = 100,
) -> ABXAMRParallelEnv:
    """Build a real ABXAMRParallelEnv using the three-antibiotic workspace config.

    Two agents, each with 10 patients, using real PatientGenerator and
    RewardCalculator instances loaded from the workspace config files.

    Args:
        max_time_steps: Episode length override (shorter than prod for test speed).

    Returns:
        A ready-to-use ABXAMRParallelEnv.
    """
    with _ENV_CONFIG_PATH.open(encoding="utf-8") as f:
        env_config = yaml.safe_load(f)

    with _RC_CONFIG_PATH.open(encoding="utf-8") as f:
        rc_config = yaml.safe_load(f)
    rc_config["seed"] = 42

    antibiotic_names = list(env_config["antibiotics_AMR_dict"].keys())

    agent_configs = []
    for i in range(2):
        pg_config = PatientGenerator.default_config()
        pg_config["visible_patient_attributes"] = ["prob_infected"]
        pg = PatientGenerator(config=pg_config)

        rc = RewardCalculator(config=rc_config)

        agent_configs.append(
            {
                "agent_id": f"agent_{i}",
                "n_patients": 10,
                "patient_generator": pg,
                "reward_calculator": rc,
            }
        )

    shared_env_config = {
        "antibiotics_amr_dict": {
            name: params
            for name, params in env_config["antibiotics_AMR_dict"].items()
        },
        "max_time_steps": max_time_steps,
        "update_visible_amr_levels_every_n_timesteps": env_config.get(
            "update_visible_AMR_levels_every_n_timesteps", 1
        ),
        "add_noise_to_visible_amr_levels": env_config.get(
            "add_noise_to_visible_AMR_levels", 0.0
        ),
        "add_bias_to_visible_amr_levels": env_config.get(
            "add_bias_to_visible_AMR_levels", 0.0
        ),
        "crossresistance_matrix": env_config.get("crossresistance_matrix", None),
    }

    return ABXAMRParallelEnv(
        agent_configs=agent_configs,
        shared_env_config=shared_env_config,
        seed=42,
    )


def _build_marl_wrapper_with_lpp_libraries(
    base_env: ABXAMRParallelEnv,
) -> MARLOptionsWrapper:
    """Load the real LPP mixed-vis option library and wrap base_env.

    Both agents use the heuristic_three_abx_mixed_vis library, which contains
    6 options with durations of 10 or 20 steps. This exercises the async
    option-completion logic within MARLOptionsWrapper.

    Args:
        base_env: Pre-instantiated ABXAMRParallelEnv.

    Returns:
        MARLOptionsWrapper wrapping base_env with real heuristic option libraries.
    """
    option_libs: Dict = {}
    for aid in base_env.possible_agents:
        rc = base_env._reward_calculators[aid]
        lib, _ = OptionLibraryLoader.load_library(
            library_config_path=str(_MIXED_VIS_LIBRARY_PATH),
            reward_calculator=rc,
            library_name=f"mixed_vis_{aid}",
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


class TestLPPLibraryLoading:
    def test_lpp_mixed_vis_library_loads_for_each_agent(self):
        """Both agents load the mixed-vis library with 6 options each."""
        base_env = _build_three_abx_parallel_env()
        wrapper = _build_marl_wrapper_with_lpp_libraries(base_env)

        for aid in base_env.possible_agents:
            assert len(wrapper.option_libraries[aid]) == 6

    def test_lpp_action_spaces_match_library_size(self):
        base_env = _build_three_abx_parallel_env()
        wrapper = _build_marl_wrapper_with_lpp_libraries(base_env)

        for aid in base_env.possible_agents:
            assert wrapper.action_spaces[aid].n == 6

    def test_lpp_obs_dims_are_positive_and_consistent(self):
        """Both agents have the same PatientGenerator, so obs dims should match."""
        base_env = _build_three_abx_parallel_env()
        wrapper = _build_marl_wrapper_with_lpp_libraries(base_env)

        dims = [
            wrapper.observation_spaces[aid].shape[0]
            for aid in base_env.possible_agents
        ]
        assert all(d > 0 for d in dims)
        # Both agents have identical PG config → same obs dim
        assert dims[0] == dims[1]


class TestLPPFullEpisode:
    def test_reset_returns_correct_shaped_obs(self):
        base_env = _build_three_abx_parallel_env(max_time_steps=50)
        wrapper = _build_marl_wrapper_with_lpp_libraries(base_env)

        manager_obs, _ = wrapper.reset()

        assert set(manager_obs.keys()) == set(base_env.possible_agents)
        for aid, obs in manager_obs.items():
            declared = wrapper.observation_spaces[aid].shape[0]
            assert obs.shape == (declared,), (
                f"Agent '{aid}': obs shape {obs.shape} != declared ({declared},)"
            )
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs)), f"Agent '{aid}' reset obs contains non-finite values"

    def test_step_obs_shapes_and_finiteness(self):
        """Step through one macro-step and verify all outputs are well-formed."""
        base_env = _build_three_abx_parallel_env(max_time_steps=50)
        wrapper = _build_marl_wrapper_with_lpp_libraries(base_env)
        wrapper.reset()

        # Select option 0 (10-step) for both agents
        all_selections = {aid: 0 for aid in base_env.possible_agents}
        m_obs, m_rew, _, _, m_info = wrapper.step(all_selections)

        for aid in m_obs:
            declared = wrapper.observation_spaces[aid].shape[0]
            assert m_obs[aid].shape == (declared,)
            assert m_obs[aid].dtype == np.float32
            assert np.all(np.isfinite(m_obs[aid])), (
                f"Agent '{aid}' step obs contains non-finite values"
            )
            assert np.isfinite(m_rew[aid]), (
                f"Agent '{aid}' reward {m_rew[aid]} is not finite"
            )

    def test_async_completion_different_option_durations(self):
        """Verify that selecting 10-step and 20-step options produces async completions."""
        base_env = _build_three_abx_parallel_env(max_time_steps=100)
        wrapper = _build_marl_wrapper_with_lpp_libraries(base_env)
        wrapper.reset()

        # Options 0 and 1 should have different durations (10 vs 20 from yaml)
        opt0_k = wrapper.option_libraries["agent_0"].get_option(0).k
        opt1_k = wrapper.option_libraries["agent_0"].get_option(1).k
        assert opt0_k != opt1_k, "Test requires options with different durations"

        # Give agent_0 the 10-step option, agent_1 the 20-step option
        selections = {"agent_0": 0, "agent_1": 1}  # 10-step vs 20-step
        m_obs, _, _, _, m_info = wrapper.step(selections)

        # Only the faster agent (agent_0, k=10) should complete first
        assert "agent_0" in m_obs
        assert "agent_1" not in m_obs
        assert m_info["agent_0"]["option_duration"] == opt0_k

    def test_full_episode_terminates_and_rewards_finite(self):
        """Run a complete episode with real LPP heuristic options; verify termination."""
        max_time_steps = 100
        base_env = _build_three_abx_parallel_env(max_time_steps=max_time_steps)
        wrapper = _build_marl_wrapper_with_lpp_libraries(base_env)

        wrapper.reset(seed=0)

        pending_selections = {aid: 0 for aid in base_env.possible_agents}
        episode_done = False
        total_rewards: Dict[str, float] = {aid: 0.0 for aid in base_env.possible_agents}
        macro_step_count = 0

        # Upper bound: at most max_time_steps macro-steps
        for _ in range(max_time_steps):
            m_obs, m_rew, m_term, m_trunc, m_info = wrapper.step(pending_selections)
            macro_step_count += 1

            for aid, r in m_rew.items():
                total_rewards[aid] += r

            # Verify observations and rewards are finite
            for aid in m_obs:
                assert np.all(np.isfinite(m_obs[aid])), (
                    f"Agent '{aid}': non-finite obs at macro step {macro_step_count}"
                )
                assert np.isfinite(m_rew[aid]), (
                    f"Agent '{aid}': non-finite reward {m_rew[aid]} at macro step {macro_step_count}"
                )

            if any(m_trunc.values()) or any(m_term.values()):
                episode_done = True
                break

            # Collect new selections for completing agents
            pending_selections = {aid: 0 for aid in m_obs.keys()}

        assert episode_done, (
            f"Episode did not terminate within {max_time_steps} macro-steps "
            f"(primitive steps remaining: {wrapper._steps_remaining})"
        )

        # Cumulative rewards must be finite
        for aid, total_r in total_rewards.items():
            assert np.isfinite(total_r), f"Agent '{aid}' cumulative reward {total_r} is not finite"
