"""Unit tests for MARL factory functions in utils/marl_factories.py.

Uses the minimal two-agent fixture YAML in tests/fixtures/marl_configs/.
All tests use real components — no mocks.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import yaml
from gymnasium import spaces
from stable_baselines3 import PPO

from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl import MARLOptionsWrapper
from abx_amr_simulator.utils.marl_factories import (
    build_marl_env_from_config,
    build_marl_managers_from_config,
    build_marl_training_run_from_config,
    build_marl_wrapper_from_config,
    load_marl_config,
)

# Path to the fixture YAML — resolved relative to this test file.
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "marl_configs"
_FIXTURE_CONFIG = _FIXTURE_DIR / "minimal_two_agent.yaml"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load() -> dict:
    return load_marl_config(_FIXTURE_CONFIG)


# --------------------------------------------------------------------------- #
# load_marl_config
# --------------------------------------------------------------------------- #

class TestLoadMarlConfig:
    def test_returns_dict(self):
        config = _load()
        assert isinstance(config, dict)

    def test_injects_config_dir(self):
        config = _load()
        assert "_config_dir" in config
        assert Path(config["_config_dir"]) == _FIXTURE_DIR

    def test_has_required_top_level_keys(self):
        config = _load()
        assert "environment" in config
        assert "training" in config

    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_marl_config("/nonexistent/path/config.yaml")

    def test_raises_for_missing_environment_key(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("training:\n  n_steps: 8\n")
        with pytest.raises(ValueError, match="environment"):
            load_marl_config(bad)

    def test_raises_for_missing_training_key(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("environment:\n  shared: {}\n")
        with pytest.raises(ValueError, match="training"):
            load_marl_config(bad)


# --------------------------------------------------------------------------- #
# build_marl_env_from_config
# --------------------------------------------------------------------------- #

class TestBuildMarlEnvFromConfig:
    def test_returns_parallel_env(self):
        env = build_marl_env_from_config(_load())
        assert isinstance(env, ABXAMRParallelEnv)

    def test_agent_count(self):
        env = build_marl_env_from_config(_load())
        assert len(env.possible_agents) == 2

    def test_agent_ids(self):
        env = build_marl_env_from_config(_load())
        assert set(env.possible_agents) == {"agent_0", "agent_1"}

    def test_n_patients_per_agent(self):
        env = build_marl_env_from_config(_load())
        assert env._agent_n_patients["agent_0"] == 3
        assert env._agent_n_patients["agent_1"] == 4

    def test_observation_spaces_are_box(self):
        env = build_marl_env_from_config(_load())
        for aid in env.possible_agents:
            assert isinstance(env.observation_spaces[aid], spaces.Box)

    def test_antibiotic_names_from_config(self):
        env = build_marl_env_from_config(_load())
        assert set(env.antibiotic_names) == {"A", "B"}

    def test_max_time_steps(self):
        env = build_marl_env_from_config(_load())
        assert env.max_time_steps == 20

    def test_env_resets_without_error(self):
        env = build_marl_env_from_config(_load())
        obs, infos = env.reset()
        assert set(obs.keys()) == set(env.possible_agents)

    def test_raises_for_missing_antibiotics_amr_dict(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "environment:\n  shared:\n    max_time_steps: 10\n  agents: []\n"
            "training:\n  n_steps: 8\n"
        )
        with pytest.raises(ValueError, match="antibiotics_AMR_dict"):
            build_marl_env_from_config(load_marl_config(bad))

    def test_plugin_loader_module_resolves_from_component_yaml_dir(self, tmp_path):
        """Relative plugin.loader_module should resolve from component YAML location."""
        plugin_dir = tmp_path / "shared" / "plugins"
        plugin_dir.mkdir(parents=True)
        plugin_loader = plugin_dir / "stub_pg_loader.py"
        plugin_loader.write_text(
            "from __future__ import annotations\n"
            "import numpy as np\n"
            "from abx_amr_simulator.core.base_patient_generator import PatientGeneratorBase\n"
            "from abx_amr_simulator.core.types import Patient\n"
            "\n"
            "class StubPG(PatientGeneratorBase):\n"
            "    PROVIDES_ATTRIBUTES = ['prob_infected']\n"
            "    visible_patient_attributes = ['prob_infected']\n"
            "\n"
            "    def sample(self, n, true_amr_levels, rng):\n"
            "        return [Patient(prob_infected=0.5, prob_infected_obs=0.5) for _ in range(n)]\n"
            "\n"
            "    def observe(self, patients):\n"
            "        obs = [float(getattr(p, 'prob_infected_obs')) for p in patients]\n"
            "        return np.asarray(obs, dtype=np.float32)\n"
            "\n"
            "    def obs_dim(self, num_patients):\n"
            "        return int(num_patients)\n"
            "\n"
            "def load_patient_generator_component(config):\n"
            "    return StubPG()\n"
        )

        component_dir = tmp_path / "group" / "marl_configs" / "patient_generators"
        component_dir.mkdir(parents=True)
        patient_yaml = component_dir / "pg_with_plugin.yaml"
        patient_yaml.write_text(
            "plugin:\n"
            "  loader_module: ../../../shared/plugins/stub_pg_loader.py\n"
            "  loader_function: load_patient_generator_component\n"
        )

        marl_config_dir = tmp_path / "results_scratch" / "seed_run"
        marl_config_dir.mkdir(parents=True)
        marl_config_path = marl_config_dir / "marl.yaml"

        fixture = _load()
        rc_config = fixture["environment"]["agents"][0]["reward_calculator"]

        config_payload = {
            "environment": {
                "shared": fixture["environment"]["shared"],
                "agents": [
                    {
                        "agent_id": "agent_0",
                        "n_patients": 2,
                        "patient_generator": str(patient_yaml),
                        "reward_calculator": rc_config,
                    }
                ],
            },
            "training": {
                "n_steps": 8,
                "batch_size": 4,
                "n_epochs": 1,
                "learning_rate": 3e-4,
                "total_primitive_steps": 10,
                "option_gamma": 0.99,
                "seed": 7,
            },
        }
        marl_config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))

        loaded = load_marl_config(marl_config_path)
        env = build_marl_env_from_config(loaded)
        assert isinstance(env, ABXAMRParallelEnv)


# --------------------------------------------------------------------------- #
# build_marl_wrapper_from_config
# --------------------------------------------------------------------------- #

class TestBuildMarlWrapperFromConfig:
    def test_returns_marl_options_wrapper(self):
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        assert isinstance(wrapper, MARLOptionsWrapper)

    def test_wrapper_has_all_agents(self):
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        assert set(wrapper.base_env.possible_agents) == {"agent_0", "agent_1"}

    def test_wrapper_has_options_per_agent(self):
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        for aid in wrapper.base_env.possible_agents:
            lib = wrapper.option_libraries[aid]
            assert len(lib) > 0, f"Agent '{aid}' has no options"

    def test_default_deterministic_library_option_count(self):
        """default_deterministic.yaml defines 12 options."""
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        for aid in wrapper.base_env.possible_agents:
            assert len(wrapper.option_libraries[aid]) == 12

    def test_option_gamma_from_config(self):
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        assert wrapper.gamma == pytest.approx(0.99)

    def test_wrapper_resets_without_error(self):
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        obs, _ = wrapper.reset()
        assert set(obs.keys()) == set(wrapper.base_env.possible_agents)

    def test_raises_for_missing_option_library_file(self, tmp_path):
        """Raises FileNotFoundError if option_library filename does not exist."""
        config = _load()
        # Corrupt one agent's option_library path
        config["environment"]["agents"][0]["option_library"] = "nonexistent.yaml"
        env = build_marl_env_from_config(config)
        with pytest.raises(FileNotFoundError):
            build_marl_wrapper_from_config(config, env)


# --------------------------------------------------------------------------- #
# build_marl_managers_from_config
# --------------------------------------------------------------------------- #

class TestBuildMarlManagersFromConfig:
    def _make_wrapper(self):
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        return config, wrapper

    def test_returns_dict_of_ppo(self):
        config, wrapper = self._make_wrapper()
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            assert isinstance(agent, PPO), f"Expected PPO for {aid}"

    def test_agent_ids_match_wrapper(self):
        config, wrapper = self._make_wrapper()
        agents = build_marl_managers_from_config(config, wrapper)
        assert set(agents.keys()) == set(wrapper.base_env.possible_agents)

    def test_ppo_obs_space_matches_wrapper(self):
        config, wrapper = self._make_wrapper()
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            expected_obs_dim = wrapper.observation_spaces[aid].shape[0]
            actual_obs_dim = agent.observation_space.shape[0]
            assert actual_obs_dim == expected_obs_dim, (
                f"Agent '{aid}': obs dim {actual_obs_dim} != {expected_obs_dim}"
            )

    def test_ppo_n_steps_from_config(self):
        config, wrapper = self._make_wrapper()
        agents = build_marl_managers_from_config(config, wrapper)
        expected_n_steps = config["training"]["n_steps"]
        for aid, agent in agents.items():
            assert agent.n_steps == expected_n_steps

    def test_raises_for_unsupported_algorithm(self):
        config, wrapper = self._make_wrapper()
        config["environment"]["agents"][0]["algorithm"] = "HRL_RPPO"
        with pytest.raises(ValueError, match="unsupported algorithm"):
            build_marl_managers_from_config(config, wrapper)

    def test_hrl_ppo_algorithm_key_accepted(self):
        """Explicitly setting algorithm: HRL_PPO should not raise."""
        config, wrapper = self._make_wrapper()
        for entry in config["environment"]["agents"]:
            entry["algorithm"] = "HRL_PPO"
        agents = build_marl_managers_from_config(config, wrapper)
        assert len(agents) == 2


# --------------------------------------------------------------------------- #
# build_marl_training_run_from_config (round-trip)
# --------------------------------------------------------------------------- #

class TestBuildMarlTrainingRunFromConfig:
    def test_returns_wrapper_and_agents(self):
        wrapper, agents = build_marl_training_run_from_config(_FIXTURE_CONFIG)
        assert isinstance(wrapper, MARLOptionsWrapper)
        assert isinstance(agents, dict)
        assert len(agents) == 2

    def test_wrapper_and_agents_are_consistent(self):
        """All agent IDs in agents dict match wrapper.base_env.possible_agents."""
        wrapper, agents = build_marl_training_run_from_config(_FIXTURE_CONFIG)
        assert set(agents.keys()) == set(wrapper.base_env.possible_agents)

    def test_wrapper_can_reset_and_step(self):
        """Round-trip result supports a full reset+step without error."""
        wrapper, agents = build_marl_training_run_from_config(_FIXTURE_CONFIG)
        obs, _ = wrapper.reset()

        # Get first option selections from agents
        pending = {}
        for aid in wrapper.base_env.possible_agents:
            obs_arr = obs[aid][np.newaxis, :]
            action, _ = agents[aid].policy.predict(obs_arr, deterministic=True)
            pending[aid] = int(action[0])

        m_obs, m_rew, m_term, m_trunc, m_info = wrapper.step(pending)
        assert len(m_obs) > 0
