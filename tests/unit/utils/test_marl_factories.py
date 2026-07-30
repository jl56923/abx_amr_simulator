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
from abx_amr_simulator.core.patient_generator import PatientGenerator, PatientGeneratorMixer
from abx_amr_simulator.hrl import MARLOptionsWrapper
from abx_amr_simulator.utils.marl_factories import (
    build_marl_env_from_config,
    build_marl_managers_from_config,
    build_marl_training_run_from_config,
    build_marl_wrapper_from_config,
    build_patient_generator_from_config,
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
        config["environment"]["agents"][0]["algorithm"] = "DQN"
        with pytest.raises(ValueError, match="unsupported algorithm"):
            build_marl_managers_from_config(config, wrapper)

    # --- manager_gamma vs option_gamma -------------------------------------- #
    #
    # These are two different horizons. option_gamma discounts primitive rewards
    # WITHIN an option as the wrapper aggregates them; manager_gamma discounts
    # ACROSS options and governs whether long-run consequences reach the manager's
    # objective. The MARL path used to read option_gamma for both, so the manager's
    # horizon had no config key of its own. manager_gamma now takes precedence and
    # falls back to option_gamma, keeping pre-existing configs bit-identical.

    def test_manager_gamma_takes_precedence_over_option_gamma(self):
        config, wrapper = self._make_wrapper()
        config["training"]["option_gamma"] = 0.99
        config["training"]["manager_gamma"] = 0.5
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            assert agent.gamma == pytest.approx(0.5), (
                f"Agent '{aid}': manager_gamma should win over option_gamma"
            )

    def test_falls_back_to_option_gamma_when_manager_gamma_absent(self):
        """Backward compatibility: configs written before manager_gamma existed."""
        config, wrapper = self._make_wrapper()
        config["training"].pop("manager_gamma", None)
        config["training"]["option_gamma"] = 0.77
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            assert agent.gamma == pytest.approx(0.77), (
                f"Agent '{aid}': should fall back to option_gamma"
            )

    def test_falls_back_to_default_when_neither_specified(self):
        config, wrapper = self._make_wrapper()
        config["training"].pop("manager_gamma", None)
        config["training"].pop("option_gamma", None)
        agents = build_marl_managers_from_config(config, wrapper)
        for agent in agents.values():
            assert agent.gamma == pytest.approx(0.99)

    def test_per_agent_hyperparams_still_override_manager_gamma(self):
        """Tuned per-agent params outrank the shared default, as they always have.

        This is what the LPP gamma diagnostic sweep relies on: it pins each cell's
        manager gamma through a reused best_params.json regardless of what the
        config-level fallback resolves to.
        """
        config, wrapper = self._make_wrapper()
        config["training"]["manager_gamma"] = 0.5
        target = sorted(wrapper.base_env.possible_agents)[0]
        agents = build_marl_managers_from_config(
            config, wrapper, agent_hyperparams={target: {"gamma": 0.123}}
        )
        assert agents[target].gamma == pytest.approx(0.123)
        for aid, agent in agents.items():
            if aid != target:
                assert agent.gamma == pytest.approx(0.5), (
                    f"Agent '{aid}' had no per-agent params and should keep manager_gamma"
                )

    def test_manager_gamma_does_not_change_wrapper_option_discount(self):
        """The two horizons must move independently — that is the point of the key."""
        config = _load()
        config["training"]["option_gamma"] = 0.99
        config["training"]["manager_gamma"] = 0.25
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        agents = build_marl_managers_from_config(config, wrapper)

        assert wrapper.gamma == pytest.approx(0.99), (
            "wrapper's within-option discount must still come from option_gamma"
        )
        for agent in agents.values():
            assert agent.gamma == pytest.approx(0.25)

    def test_hrl_ppo_algorithm_key_accepted(self):
        """Explicitly setting algorithm: HRL_PPO should not raise."""
        config, wrapper = self._make_wrapper()
        for entry in config["environment"]["agents"]:
            entry["algorithm"] = "HRL_PPO"
        agents = build_marl_managers_from_config(config, wrapper)
        assert len(agents) == 2

    def test_hrl_rppo_algorithm_creates_recurrent_agents(self):
        """Setting algorithm: HRL_RPPO should create RecurrentPPO_Masked agents."""
        from abx_amr_simulator.hrl.rl_algorithms.recurrent_ppo_masked import (
            RecurrentPPO_Masked,
        )
        config, wrapper = self._make_wrapper()
        for entry in config["environment"]["agents"]:
            entry["algorithm"] = "HRL_RPPO"
        agents = build_marl_managers_from_config(config, wrapper)
        assert len(agents) == 2
        for aid, agent in agents.items():
            assert isinstance(agent, RecurrentPPO_Masked), (
                f"Expected RecurrentPPO_Masked for {aid}"
            )

    def test_hrl_rppo_default_lstm_kwargs(self):
        """HRL_RPPO without lstm_kwargs should use defaults (hidden=64, layers=1)."""
        config, wrapper = self._make_wrapper()
        for entry in config["environment"]["agents"]:
            entry["algorithm"] = "HRL_RPPO"
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            assert agent.policy.lstm_actor.hidden_size == 64
            assert agent.policy.lstm_actor.num_layers == 1

    def test_hrl_rppo_custom_lstm_kwargs(self):
        """HRL_RPPO with custom lstm_kwargs should respect them."""
        config, wrapper = self._make_wrapper()
        for entry in config["environment"]["agents"]:
            entry["algorithm"] = "HRL_RPPO"
            entry["lstm_kwargs"] = {
                "lstm_hidden_size": 32,
                "n_lstm_layers": 2,
                "enable_critic_lstm": False,
            }
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            assert agent.policy.lstm_actor.hidden_size == 32
            assert agent.policy.lstm_actor.num_layers == 2
            assert agent.policy.lstm_critic is None

    def test_mixed_ppo_and_rppo(self):
        """One agent HRL_PPO, another HRL_RPPO — both should be constructed."""
        from abx_amr_simulator.hrl.rl_algorithms.recurrent_ppo_masked import (
            RecurrentPPO_Masked,
        )
        config, wrapper = self._make_wrapper()
        entries = config["environment"]["agents"]
        entries[0]["algorithm"] = "HRL_PPO"
        entries[1]["algorithm"] = "HRL_RPPO"
        agents = build_marl_managers_from_config(config, wrapper)
        aid_0 = str(entries[0]["agent_id"])
        aid_1 = str(entries[1]["agent_id"])
        assert isinstance(agents[aid_0], PPO)
        assert not isinstance(agents[aid_0], RecurrentPPO_Masked)
        assert isinstance(agents[aid_1], RecurrentPPO_Masked)

    def test_algorithm_defaults_to_hrl_ppo(self):
        """Omitting algorithm key should default to HRL_PPO (backward compat)."""
        config, wrapper = self._make_wrapper()
        for entry in config["environment"]["agents"]:
            entry.pop("algorithm", None)
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            assert isinstance(agent, PPO)

    def test_hrl_ppo_net_arch_absent_by_default(self):
        """Without policy_kwargs.net_arch, HRL_PPO agents use SB3 defaults."""
        config, wrapper = self._make_wrapper()
        for entry in config["environment"]["agents"]:
            entry["algorithm"] = "HRL_PPO"
            entry.pop("policy_kwargs", None)
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            assert "net_arch" not in (agent.policy_kwargs or {})

    def test_hrl_ppo_honors_policy_kwargs_net_arch(self):
        """HRL_PPO reads policy_kwargs.net_arch from the agent entry."""
        config, wrapper = self._make_wrapper()
        for entry in config["environment"]["agents"]:
            entry["algorithm"] = "HRL_PPO"
            entry["policy_kwargs"] = {"net_arch": [7, 11]}
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            assert agent.policy_kwargs.get("net_arch") == [7, 11]

    def test_hrl_rppo_honors_policy_kwargs_net_arch(self):
        """HRL_RPPO reads policy_kwargs.net_arch alongside lstm_kwargs."""
        config, wrapper = self._make_wrapper()
        for entry in config["environment"]["agents"]:
            entry["algorithm"] = "HRL_RPPO"
            entry["lstm_kwargs"] = {"lstm_hidden_size": 16, "n_lstm_layers": 1}
            entry["policy_kwargs"] = {"net_arch": [13, 17]}
        agents = build_marl_managers_from_config(config, wrapper)
        for aid, agent in agents.items():
            assert agent.policy_kwargs.get("net_arch") == [13, 17]
            # LSTM kwargs should still be honored when net_arch is also set.
            assert agent.policy.lstm_actor.hidden_size == 16

    def test_mixed_net_arch_across_agents(self):
        """Per-agent net_arch values are applied independently."""
        config, wrapper = self._make_wrapper()
        entries = config["environment"]["agents"]
        entries[0]["algorithm"] = "HRL_PPO"
        entries[0]["policy_kwargs"] = {"net_arch": [5, 5]}
        entries[1]["algorithm"] = "HRL_RPPO"
        entries[1]["lstm_kwargs"] = {"lstm_hidden_size": 8}
        entries[1]["policy_kwargs"] = {"net_arch": [9, 9]}
        agents = build_marl_managers_from_config(config, wrapper)
        aid_0 = str(entries[0]["agent_id"])
        aid_1 = str(entries[1]["agent_id"])
        assert agents[aid_0].policy_kwargs.get("net_arch") == [5, 5]
        assert agents[aid_1].policy_kwargs.get("net_arch") == [9, 9]


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


# --------------------------------------------------------------------------- #
# build_patient_generator_from_config — mixer branch
# --------------------------------------------------------------------------- #

def _minimal_pg_attrs() -> dict:
    """Minimal inline PatientGenerator attribute config for use in mixer children."""
    return {
        "prob_infected": {
            "prob_dist": {"type": "constant", "value": 0.7, "mu": None, "sigma": None},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, 1.0],
        },
        "visible_patient_attributes": ["prob_infected"],
    }


class TestBuildPatientGeneratorFromConfigMixer:
    def test_mixer_inline_returns_patient_generator_mixer(self, tmp_path):
        """An inline type:mixer config is dispatched to PatientGeneratorMixer."""
        config = {
            "type": "mixer",
            "generators": [
                {"proportion": 0.6, **_minimal_pg_attrs()},
                {"proportion": 0.4, **_minimal_pg_attrs()},
            ],
        }
        pg = build_patient_generator_from_config(config, str(tmp_path), seed=0)
        assert isinstance(pg, PatientGeneratorMixer)

    def test_mixer_file_reference_returns_patient_generator_mixer(self, tmp_path):
        """A mixer YAML referenced by filename is resolved and returns PatientGeneratorMixer."""
        child_config = _minimal_pg_attrs()
        child_yaml = tmp_path / "child_pg.yaml"
        child_yaml.write_text(yaml.safe_dump(child_config))

        mixer_config = {
            "type": "mixer",
            "generators": [
                {"proportion": 0.5, "config_file": "child_pg.yaml"},
                {"proportion": 0.5, "config_file": "child_pg.yaml"},
            ],
        }
        mixer_yaml = tmp_path / "mixer_pg.yaml"
        mixer_yaml.write_text(yaml.safe_dump(mixer_config))

        pg = build_patient_generator_from_config(str(mixer_yaml), str(tmp_path), seed=1)
        assert isinstance(pg, PatientGeneratorMixer)

    def test_mixer_proportions_sum_to_one(self, tmp_path):
        config = {
            "type": "mixer",
            "generators": [
                {"proportion": 0.3, **_minimal_pg_attrs()},
                {"proportion": 0.7, **_minimal_pg_attrs()},
            ],
        }
        pg = build_patient_generator_from_config(config, str(tmp_path), seed=0)
        assert isinstance(pg, PatientGeneratorMixer)
        assert abs(pg.proportions.sum() - 1.0) < 1e-6

    def test_mixer_child_count(self, tmp_path):
        config = {
            "type": "mixer",
            "generators": [
                {"proportion": 0.5, **_minimal_pg_attrs()},
                {"proportion": 0.5, **_minimal_pg_attrs()},
            ],
        }
        pg = build_patient_generator_from_config(config, str(tmp_path), seed=0)
        assert len(pg.generators) == 2

    def test_non_mixer_inline_still_returns_patient_generator(self, tmp_path):
        """A plain (non-mixer) inline config still produces a PatientGenerator."""
        config = _minimal_pg_attrs()
        pg = build_patient_generator_from_config(config, str(tmp_path), seed=0)
        assert isinstance(pg, PatientGenerator)
        assert not isinstance(pg, PatientGeneratorMixer)

    def test_mixer_with_config_base_folder_prefix_resolves(self, tmp_path, monkeypatch):
        """Inline mixer with $CONFIG_BASE_FOLDER/ config_file resolves when env var is set."""
        child_yaml = tmp_path / "child_pg.yaml"
        child_yaml.write_text(yaml.safe_dump(_minimal_pg_attrs()))
        monkeypatch.setenv("ABX_AMR_CONFIG_BASE_FOLDER", str(tmp_path))

        config = {
            "type": "mixer",
            "generators": [
                {"proportion": 1.0, "config_file": "$CONFIG_BASE_FOLDER/child_pg.yaml"},
            ],
        }
        pg = build_patient_generator_from_config(config, ".", seed=0)
        assert isinstance(pg, PatientGeneratorMixer)


# --------------------------------------------------------------------------- #
# Inline config_file absolutization (pre-save robustness for run-dir reload)
# --------------------------------------------------------------------------- #

class TestInlineConfigFileAbsolutization:
    """Verify that inline patient_generator config_file paths are absolutized before saving.

    This simulates the pre-save step in train_marl.py: after absolutization,
    reloading from a different directory (the run dir) must not break path resolution.
    """

    def _make_inline_mixer_config(self, config_dir: Path, child_yaml: Path) -> dict:
        """Return a minimal MARL config with an inline mixer patient_generator."""
        return {
            "_config_dir": str(config_dir),
            "environment": {
                "shared": {
                    "antibiotics_AMR_dict": {
                        "A": {"leak": 0.05, "flatness_parameter": 1.0,
                              "permanent_residual_volume": 0.0, "initial_amr_level": 0.0}
                    },
                    "max_time_steps": 5,
                    "num_patients_per_time_step": 1,
                },
                "agents": [
                    {
                        "agent_id": "agent_0",
                        "n_patients": 2,
                        "patient_generator": {
                            "type": "mixer",
                            "generators": [
                                {"proportion": 0.5, "config_file": "child_pg.yaml"},
                                {"proportion": 0.5, "config_file": "child_pg.yaml"},
                            ],
                        },
                        "reward_calculator": _load()["environment"]["agents"][0]["reward_calculator"],
                        "option_library": str(
                            Path(_FIXTURE_DIR / "minimal_two_agent.yaml").resolve().parent
                            / "default_deterministic.yaml"
                        ),
                    }
                ],
            },
            "training": {"n_steps": 4, "batch_size": 2, "n_epochs": 1,
                         "learning_rate": 3e-4, "total_primitive_steps": 10,
                         "option_gamma": 0.99, "seed": 0},
        }

    def test_inline_config_file_paths_absolutized(self, tmp_path):
        """After applying the pre-save absolutization logic, config_file paths become absolute."""
        from abx_amr_simulator.utils.factories import resolve_config_path

        child_yaml = tmp_path / "child_pg.yaml"
        child_yaml.write_text(yaml.safe_dump(_minimal_pg_attrs()))

        config = self._make_inline_mixer_config(tmp_path, child_yaml)
        config_dir = Path(config["_config_dir"])

        # Replicate the absolutization logic from train_marl.py
        def absolutize_config_file_paths(obj):
            if isinstance(obj, dict):
                if "config_file" in obj:
                    val = obj["config_file"]
                    if isinstance(val, str):
                        obj["config_file"] = str(resolve_config_path(val, base_dir=config_dir))
                for v in obj.values():
                    absolutize_config_file_paths(v)
            elif isinstance(obj, list):
                for item in obj:
                    absolutize_config_file_paths(item)

        pg_value = config["environment"]["agents"][0]["patient_generator"]
        absolutize_config_file_paths(pg_value)

        for gen_entry in pg_value["generators"]:
            assert Path(gen_entry["config_file"]).is_absolute(), (
                f"config_file should be absolute after pre-save, got: {gen_entry['config_file']}"
            )

    def test_reloaded_config_builds_correctly_after_absolutization(self, tmp_path):
        """After saving and reloading config, inline mixer resolves correctly from run dir."""
        from abx_amr_simulator.utils.factories import resolve_config_path

        config_dir = tmp_path / "marl_configs"
        config_dir.mkdir()
        child_yaml = config_dir / "child_pg.yaml"
        child_yaml.write_text(yaml.safe_dump(_minimal_pg_attrs()))

        config = self._make_inline_mixer_config(config_dir, child_yaml)
        cfg_dir_path = Path(config["_config_dir"])

        def absolutize_config_file_paths(obj):
            if isinstance(obj, dict):
                if "config_file" in obj:
                    val = obj["config_file"]
                    if isinstance(val, str):
                        obj["config_file"] = str(resolve_config_path(val, base_dir=cfg_dir_path))
                for v in obj.values():
                    absolutize_config_file_paths(v)
            elif isinstance(obj, list):
                for item in obj:
                    absolutize_config_file_paths(item)

        pg_value = config["environment"]["agents"][0]["patient_generator"]
        absolutize_config_file_paths(pg_value)

        # Simulate reload from run dir (different directory than config_dir)
        run_dir = tmp_path / "run" / "deep" / "path"
        run_dir.mkdir(parents=True)
        saved_config_path = run_dir / "marl_full_agents_env_config.yaml"
        config_to_save = {k: v for k, v in config.items() if not k.startswith("_")}
        with open(saved_config_path, "w") as f:
            yaml.dump(config_to_save, f)

        reloaded = load_marl_config(saved_config_path)
        # _config_dir is now run_dir — but config_file paths are absolute, so no error
        pg = build_patient_generator_from_config(
            reloaded["environment"]["agents"][0]["patient_generator"],
            reloaded["_config_dir"],
            seed=0,
        )
        assert isinstance(pg, PatientGeneratorMixer)
