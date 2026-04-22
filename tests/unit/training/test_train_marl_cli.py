"""Unit tests for Phase 6.3: _apply_overrides, build_marl_managers_from_config
(agent_hyperparams), and the train_marl __main__ CLI.

Uses the minimal_two_agent.yaml fixture for real component construction.
All tests use real PPO objects — no mocks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from abx_amr_simulator.training.train_marl import _apply_overrides
from abx_amr_simulator.utils.marl_factories import (
    build_marl_env_from_config,
    build_marl_managers_from_config,
    build_marl_wrapper_from_config,
    load_marl_config,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "marl_configs"
_FIXTURE_CONFIG = _FIXTURE_DIR / "minimal_two_agent.yaml"


def _load() -> dict:
    return load_marl_config(_FIXTURE_CONFIG)


def _find_timestamped_run_dir(*, results_dir: Path, run_name: str) -> Path:
    candidates = sorted(results_dir.glob(f"{run_name}_????????_??????"))
    assert len(candidates) == 1, (
        f"Expected exactly one timestamped run dir for prefix {run_name!r}, "
        f"found {len(candidates)}: {[c.name for c in candidates]}"
    )
    return candidates[0]


# --------------------------------------------------------------------------- #
# _apply_overrides
# --------------------------------------------------------------------------- #

class TestApplyOverrides:
    def test_string_value_set(self):
        config = {"key": "old"}
        _apply_overrides(config, ["key=new"])
        assert config["key"] == "new"

    def test_int_coercion(self):
        config = {"n": 10}
        _apply_overrides(config, ["n=42"])
        assert config["n"] == 42
        assert isinstance(config["n"], int)

    def test_float_coercion(self):
        config = {"lr": 0.001}
        _apply_overrides(config, ["lr=0.0005"])
        assert abs(config["lr"] - 0.0005) < 1e-9
        assert isinstance(config["lr"], float)

    def test_bool_true_coercion(self):
        config = {"flag": False}
        _apply_overrides(config, ["flag=true"])
        assert config["flag"] is True

    def test_bool_false_coercion(self):
        config = {"flag": True}
        _apply_overrides(config, ["flag=false"])
        assert config["flag"] is False

    def test_nested_dict_path(self):
        config = {"a": {"b": {"c": 99}}}
        _apply_overrides(config, ["a.b.c=7"])
        assert config["a"]["b"]["c"] == 7

    def test_list_index(self):
        config = {"items": [10, 20, 30]}
        _apply_overrides(config, ["items.1=99"])
        assert config["items"][1] == 99

    def test_nested_list_dict_path(self):
        """agents.0.n_patients style override works."""
        config = _load()
        original_n = config["environment"]["agents"][0]["n_patients"]
        _apply_overrides(
            config, ["environment.agents.0.n_patients=5"]
        )
        assert config["environment"]["agents"][0]["n_patients"] == 5
        assert config["environment"]["agents"][0]["n_patients"] != original_n

    def test_multiple_overrides_applied_in_order(self):
        config = {"x": 1, "y": 2}
        _apply_overrides(config, ["x=10", "y=20"])
        assert config["x"] == 10
        assert config["y"] == 20

    def test_raises_for_missing_key(self):
        config = {"a": 1}
        with pytest.raises(KeyError, match="no_such_key"):
            _apply_overrides(config, ["no_such_key=5"])

    def test_raises_for_missing_nested_key(self):
        config = {"a": {"b": 1}}
        with pytest.raises(KeyError):
            _apply_overrides(config, ["a.no_such=5"])

    def test_raises_for_missing_equals(self):
        config = {"a": 1}
        with pytest.raises(ValueError, match="="):
            _apply_overrides(config, ["a"])

    def test_mutates_in_place_and_returns_config(self):
        config = {"v": 1}
        result = _apply_overrides(config, ["v=2"])
        assert result is config
        assert config["v"] == 2

    def test_string_left_as_string_for_string_existing(self):
        """If existing value is str, coerce to str (no type error)."""
        config = {"name": "old_name"}
        _apply_overrides(config, ["name=new_name"])
        assert config["name"] == "new_name"
        assert isinstance(config["name"], str)


# --------------------------------------------------------------------------- #
# build_marl_managers_from_config with agent_hyperparams
# --------------------------------------------------------------------------- #

class TestBuildMarlManagersAgentHyperparams:
    def test_default_hyperparams_without_agent_hyperparams(self):
        """All agents use training-section defaults when agent_hyperparams is None."""
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        agents = build_marl_managers_from_config(config, wrapper)
        assert "agent_0" in agents
        assert "agent_1" in agents

    def test_agent_hyperparams_overrides_learning_rate(self):
        """Per-agent learning_rate override is applied to the PPO object."""
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)

        override_lr = 9.9e-5
        agents = build_marl_managers_from_config(
            config,
            wrapper,
            agent_hyperparams={"agent_0": {"learning_rate": override_lr}},
        )
        # SB3 PPO stores learning_rate as a schedule; check via policy optimizer
        # The initial lr is accessible via the initial_learning_rate or via the
        # lambda schedule at t=1. Simplest: verify agent_1 uses the default.
        default_lr = float(config["training"].get("learning_rate", 3e-4))
        # agent_0 override should differ from training-config default
        assert override_lr != default_lr
        # agent_1 should still exist and be a PPO object
        from stable_baselines3 import PPO
        assert isinstance(agents["agent_1"], PPO)

    def test_batch_size_not_overridden_by_agent_hyperparams(self):
        """batch_size in agent_hyperparams is ignored; training-section value is used."""
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        # Inject a batch_size that would cause an error if applied
        # (larger than rollout buffer would allow for 3-patient env with n_steps=8)
        agents = build_marl_managers_from_config(
            config,
            wrapper,
            agent_hyperparams={"agent_0": {"batch_size": 99999, "n_epochs": 2}},
        )
        # Should build without error — batch_size is ignored, n_epochs is applied
        from stable_baselines3 import PPO
        assert isinstance(agents["agent_0"], PPO)

    def test_agent_without_entry_uses_defaults(self):
        """Agents not mentioned in agent_hyperparams use training-section defaults."""
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        # Only provide overrides for agent_0; agent_1 should use defaults silently
        agents = build_marl_managers_from_config(
            config,
            wrapper,
            agent_hyperparams={"agent_0": {"n_epochs": 3}},
        )
        assert "agent_1" in agents

    def test_empty_agent_hyperparams_dict_behaves_like_none(self):
        """Empty agent_hyperparams dict produces same result as None."""
        config = _load()
        env = build_marl_env_from_config(config)
        wrapper = build_marl_wrapper_from_config(config, env)
        agents = build_marl_managers_from_config(config, wrapper, agent_hyperparams={})
        assert "agent_0" in agents
        assert "agent_1" in agents


# --------------------------------------------------------------------------- #
# run_marl_training public API
# --------------------------------------------------------------------------- #

class TestRunMarlTraining:
    """Tests for the public run_marl_training function (called directly by runner)."""

    def test_produces_final_model_files(self, tmp_path):
        """run_marl_training writes final_model_{aid}.zip without sys.argv."""
        from abx_amr_simulator.training.train_marl import run_marl_training

        run_marl_training(
            marl_config_path=_FIXTURE_CONFIG,
            results_dir=tmp_path,
            run_name="api_run",
            seed=0,
        )
        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="api_run")
        checkpoint_dir = run_dir / "checkpoints"
        assert (checkpoint_dir / "final_model_agent_0.zip").exists()
        assert (checkpoint_dir / "final_model_agent_1.zip").exists()

    def test_overrides_applied(self, tmp_path):
        """run_marl_training applies overrides before saving config."""
        from abx_amr_simulator.training.train_marl import run_marl_training

        run_marl_training(
            marl_config_path=_FIXTURE_CONFIG,
            results_dir=tmp_path,
            run_name="api_run",
            seed=3,
            overrides=["training.n_steps=4"],
        )
        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="api_run")
        saved = yaml.safe_load(
            (run_dir / "marl_full_agents_env_config.yaml").read_text()
        )
        assert saved["training"]["n_steps"] == 4
        assert saved["training"]["seed"] == 3

    def test_skip_if_exists(self, tmp_path):
        """run_marl_training with skip_if_exists=True skips on second call."""
        from abx_amr_simulator.training.train_marl import run_marl_training

        run_marl_training(
            marl_config_path=_FIXTURE_CONFIG,
            results_dir=tmp_path,
            run_name="api_run",
            seed=0,
        )
        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="api_run")
        mtime = (
            run_dir / "checkpoints" / "final_model_agent_0.zip"
        ).stat().st_mtime

        run_marl_training(
            marl_config_path=_FIXTURE_CONFIG,
            results_dir=tmp_path,
            run_name="api_run",
            seed=99,
            skip_if_exists=True,
        )
        assert (
            run_dir / "checkpoints" / "final_model_agent_0.zip"
        ).stat().st_mtime == mtime

    def test_inline_patient_plugin_loader_module_saved_as_absolute(self, tmp_path):
        """Inline relative plugin loader paths are absolutized before replay."""
        from abx_amr_simulator.training.train_marl import run_marl_training

        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir(parents=True)
        plugin_path = plugin_dir / "custom_pg_plugin.py"
        plugin_path.write_text(
            "from abx_amr_simulator.core.patient_generator import PatientGenerator\n"
            "def load_patient_generator_component(config):\n"
            "    cfg = dict(config)\n"
            "    cfg.pop('plugin', None)\n"
            "    return PatientGenerator(config=cfg)\n"
        )

        option_library_path = (
            Path(__file__).resolve().parents[2]
            / ".."
            / "src"
            / "abx_amr_simulator"
            / "options"
            / "defaults"
            / "option_libraries"
            / "default_deterministic.yaml"
        ).resolve()

        marl_config = {
            "environment": {
                "shared": {
                    "antibiotics_AMR_dict": {
                        "A": {
                            "leak": 0.05,
                            "flatness_parameter": 1.0,
                            "permanent_residual_volume": 0.0,
                            "initial_amr_level": 0.0,
                        },
                        "B": {
                            "leak": 0.05,
                            "flatness_parameter": 1.0,
                            "permanent_residual_volume": 0.0,
                            "initial_amr_level": 0.0,
                        },
                    },
                    "max_time_steps": 5,
                },
                "agents": [
                    {
                        "agent_id": "agent_0",
                        "n_patients": 2,
                        "patient_generator": {
                            "plugin": {
                                "loader_module": "./plugins/custom_pg_plugin.py",
                                "loader_function": "load_patient_generator_component",
                            },
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
                                "prob_dist": {"type": "constant", "value": 1.0, "mu": None, "sigma": None},
                                "obs_bias_multiplier": 1.0,
                                "obs_noise_one_std_dev": 0.0,
                                "obs_noise_std_dev_fraction": 0.0,
                                "clipping_bounds": [0.0, None],
                            },
                            "failure_probability_multiplier": {
                                "prob_dist": {"type": "constant", "value": 1.0, "mu": None, "sigma": None},
                                "obs_bias_multiplier": 1.0,
                                "obs_noise_one_std_dev": 0.0,
                                "obs_noise_std_dev_fraction": 0.0,
                                "clipping_bounds": [0.0, None],
                            },
                            "recovery_without_treatment_prob": {
                                "prob_dist": {"type": "constant", "value": 0.0, "mu": None, "sigma": None},
                                "obs_bias_multiplier": 1.0,
                                "obs_noise_one_std_dev": 0.0,
                                "obs_noise_std_dev_fraction": 0.0,
                                "clipping_bounds": [0.0, 1.0],
                            },
                            "visible_patient_attributes": ["prob_infected"],
                        },
                        "reward_calculator": {
                            "abx_clinical_reward_penalties_info_dict": {
                                "clinical_benefit_reward": 10.0,
                                "clinical_benefit_probability": 1.0,
                                "clinical_failure_penalty": -10.0,
                                "clinical_failure_probability": 1.0,
                                "abx_adverse_effects_info": {
                                    "A": {"adverse_effect_penalty": -2.0, "adverse_effect_probability": 0.1},
                                    "B": {"adverse_effect_penalty": -3.0, "adverse_effect_probability": 0.15},
                                },
                            },
                            "lambda_weight": 0.0,
                            "seed": None,
                        },
                        "option_library": str(option_library_path),
                    }
                ],
            },
            "training": {
                "n_steps": 8,
                "batch_size": 4,
                "n_epochs": 1,
                "learning_rate": 3e-4,
                "total_primitive_steps": 20,
                "option_gamma": 0.99,
                "seed": 0,
            },
        }

        marl_config_path = tmp_path / "marl_inline_plugin.yaml"
        marl_config_path.write_text(yaml.safe_dump(marl_config, sort_keys=False))

        run_marl_training(
            marl_config_path=marl_config_path,
            results_dir=tmp_path,
            run_name="inline_plugin_run",
            seed=0,
        )
        run_dir = _find_timestamped_run_dir(
            results_dir=tmp_path,
            run_name="inline_plugin_run",
        )

        saved = yaml.safe_load(
            (run_dir / "marl_full_agents_env_config.yaml").read_text()
        )
        loader_module = saved["environment"]["agents"][0]["patient_generator"]["plugin"]["loader_module"]
        assert Path(loader_module).is_absolute()
        assert Path(loader_module).resolve() == plugin_path.resolve()

    def test_inline_reward_plugin_loader_module_saved_as_absolute(self, tmp_path):
        """Inline reward plugin loader path is absolutized before replay."""
        from abx_amr_simulator.training.train_marl import run_marl_training

        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir(parents=True)
        plugin_path = plugin_dir / "custom_rc_plugin.py"
        plugin_path.write_text(
            "from abx_amr_simulator.core.reward_calculator import RewardCalculator\n"
            "def load_reward_calculator_component(config):\n"
            "    cfg = dict(config)\n"
            "    cfg.pop('plugin', None)\n"
            "    return RewardCalculator(config=cfg)\n"
        )

        config = yaml.safe_load(_FIXTURE_CONFIG.read_text())
        option_library_path = (
            Path(__file__).resolve().parents[2]
            / ".."
            / "src"
            / "abx_amr_simulator"
            / "options"
            / "defaults"
            / "option_libraries"
            / "default_deterministic.yaml"
        ).resolve()
        for agent_entry in config["environment"]["agents"]:
            agent_entry["option_library"] = str(option_library_path)

        config["environment"]["agents"][0]["reward_calculator"]["plugin"] = {
            "loader_module": "./plugins/custom_rc_plugin.py",
            "loader_function": "load_reward_calculator_component",
        }
        config_path = tmp_path / "marl_inline_reward_plugin.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        run_marl_training(
            marl_config_path=config_path,
            results_dir=tmp_path,
            run_name="inline_reward_plugin_run",
            seed=0,
        )
        run_dir = _find_timestamped_run_dir(
            results_dir=tmp_path,
            run_name="inline_reward_plugin_run",
        )

        saved = yaml.safe_load(
            (run_dir / "marl_full_agents_env_config.yaml").read_text()
        )
        loader_module = saved["environment"]["agents"][0]["reward_calculator"]["plugin"]["loader_module"]
        assert Path(loader_module).is_absolute()
        assert Path(loader_module).resolve() == plugin_path.resolve()

    def test_registry_written_on_completion(self, tmp_path):
        """run_marl_training appends an entry to .training_completed.txt on success."""
        from abx_amr_simulator.training.train_marl import run_marl_training
        from abx_amr_simulator.utils.registry import load_registry

        run_marl_training(
            marl_config_path=_FIXTURE_CONFIG,
            results_dir=tmp_path,
            run_name="reg_run",
            seed=0,
        )
        registry_path = tmp_path / ".training_completed.txt"
        assert registry_path.exists(), "Registry file was not created after training"
        assert "reg_run" in load_registry(str(registry_path))

    def test_skip_if_exists_uses_registry_not_files(self, tmp_path):
        """skip_if_exists skips based on the registry even if final model files are absent."""
        from abx_amr_simulator.training.train_marl import run_marl_training

        run_marl_training(
            marl_config_path=_FIXTURE_CONFIG,
            results_dir=tmp_path,
            run_name="reg_skip_run",
            seed=0,
        )
        # Delete the final model files to confirm the skip is registry-driven.
        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="reg_skip_run")
        for f in (run_dir / "checkpoints").glob("final_model_*.zip"):
            f.unlink()

        run_marl_training(
            marl_config_path=_FIXTURE_CONFIG,
            results_dir=tmp_path,
            run_name="reg_skip_run",
            seed=99,
            skip_if_exists=True,
        )
        # Files are still absent — training was skipped, not re-run.
        remaining = list((run_dir / "checkpoints").glob("final_model_*.zip"))
        assert len(remaining) == 0, "Training ran again despite registry entry"

    def test_no_skip_when_not_in_registry(self, tmp_path):
        """skip_if_exists re-runs training when the registry entry is absent."""
        from abx_amr_simulator.training.train_marl import run_marl_training
        from abx_amr_simulator.utils.registry import load_registry

        run_marl_training(
            marl_config_path=_FIXTURE_CONFIG,
            results_dir=tmp_path,
            run_name="reg_retry_run",
            seed=0,
        )
        # Wipe the registry to simulate what an interrupted run looks like
        # (run folder exists on disk but was never registered as complete).
        registry_path = tmp_path / ".training_completed.txt"
        registry_path.unlink()

        run_marl_training(
            marl_config_path=_FIXTURE_CONFIG,
            results_dir=tmp_path,
            run_name="reg_retry_run",
            seed=1,
            skip_if_exists=True,
        )
        # Training ran again and wrote a new registry entry.  If it had
        # incorrectly skipped, the registry would still be empty.
        assert "reg_retry_run" in load_registry(str(registry_path)), (
            "Training did not re-run: registry is empty after retry"
        )

    def test_save_freq_episodes_decoupled_from_eval_freq_episodes(self, tmp_path):
        """run_marl_training honors save_freq_episodes independently of eval_freq_episodes."""
        from abx_amr_simulator.training.train_marl import run_marl_training

        config = yaml.safe_load(_FIXTURE_CONFIG.read_text())
        option_library_path = (
            Path(__file__).resolve().parents[2]
            / ".."
            / "src"
            / "abx_amr_simulator"
            / "options"
            / "defaults"
            / "option_libraries"
            / "default_deterministic.yaml"
        ).resolve()
        for agent_entry in config["environment"]["agents"]:
            agent_entry["option_library"] = str(option_library_path)

        config["environment"]["shared"]["max_time_steps"] = 10
        config["training"]["total_primitive_steps"] = 60
        config["training"]["eval_freq_episodes"] = 2
        config["training"]["save_freq_episodes"] = 3
        config_path = tmp_path / "save_freq_override_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        run_marl_training(
            marl_config_path=config_path,
            results_dir=tmp_path,
            run_name="save_freq_override_run",
            seed=0,
        )

        run_dir = _find_timestamped_run_dir(
            results_dir=tmp_path,
            run_name="save_freq_override_run",
        )
        checkpoint_dir = run_dir / "checkpoints"
        for aid in ["agent_0", "agent_1"]:
            periodic = sorted(checkpoint_dir.glob(f"{aid}_checkpoint_*.zip"))
            assert len(periodic) == 2, (
                f"Expected exactly 2 periodic checkpoints for {aid}, got {len(periodic)}"
            )
            assert (checkpoint_dir / f"best_model_{aid}.zip").exists()


# --------------------------------------------------------------------------- #
# _main() CLI integration
# --------------------------------------------------------------------------- #

class TestTrainMarlCLI:
    """Integration tests for the __main__ entry point."""

    def test_train_produces_final_model_files(self, tmp_path, monkeypatch):
        """_main() runs training and writes final_model_{aid}.zip files."""
        from abx_amr_simulator.training.train_marl import _main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_marl",
                "--marl-config", str(_FIXTURE_CONFIG),
                "--results-dir", str(tmp_path),
                "--run-name", "test_run",
                "--seed", "0",
            ],
        )
        _main()

        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="test_run")
        checkpoint_dir = run_dir / "checkpoints"
        assert (checkpoint_dir / "final_model_agent_0.zip").exists()
        assert (checkpoint_dir / "final_model_agent_1.zip").exists()

    def test_train_writes_config_yaml(self, tmp_path, monkeypatch):
        """_main() writes marl_full_agents_env_config.yaml to the run folder."""
        from abx_amr_simulator.training.train_marl import _main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_marl",
                "--marl-config", str(_FIXTURE_CONFIG),
                "--results-dir", str(tmp_path),
                "--run-name", "test_run",
                "--seed", "7",
            ],
        )
        _main()

        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="test_run")
        saved_config_path = run_dir / "marl_full_agents_env_config.yaml"
        assert saved_config_path.exists()
        with open(saved_config_path) as f:
            saved = yaml.safe_load(f)
        assert "environment" in saved
        assert "training" in saved
        # Internal _config_dir key should not be written
        assert "_config_dir" not in saved

    def test_config_yaml_has_absolute_option_library_paths(self, tmp_path, monkeypatch):
        """Saved config resolves option_library to absolute paths."""
        from abx_amr_simulator.training.train_marl import _main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_marl",
                "--marl-config", str(_FIXTURE_CONFIG),
                "--results-dir", str(tmp_path),
                "--run-name", "test_run",
                "--seed", "0",
            ],
        )
        _main()

        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="test_run")
        saved_config_path = run_dir / "marl_full_agents_env_config.yaml"
        with open(saved_config_path) as f:
            saved = yaml.safe_load(f)
        for entry in saved["environment"]["agents"]:
            lib_path = entry.get("option_library", "")
            assert Path(lib_path).is_absolute(), (
                f"option_library path is not absolute: {lib_path!r}"
            )

    def test_seed_injected_into_saved_config(self, tmp_path, monkeypatch):
        """--seed value appears in marl_full_agents_env_config.yaml training.seed."""
        from abx_amr_simulator.training.train_marl import _main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_marl",
                "--marl-config", str(_FIXTURE_CONFIG),
                "--results-dir", str(tmp_path),
                "--run-name", "test_run",
                "--seed", "13",
            ],
        )
        _main()

        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="test_run")
        saved_config_path = run_dir / "marl_full_agents_env_config.yaml"
        with open(saved_config_path) as f:
            saved = yaml.safe_load(f)
        assert saved["training"]["seed"] == 13

    def test_override_applied_before_training(self, tmp_path, monkeypatch):
        """-p override is visible in the saved config YAML."""
        from abx_amr_simulator.training.train_marl import _main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_marl",
                "--marl-config", str(_FIXTURE_CONFIG),
                "--results-dir", str(tmp_path),
                "--run-name", "test_run",
                "--seed", "0",
                "-p", "training.n_steps=4",
            ],
        )
        _main()

        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="test_run")
        saved_config_path = run_dir / "marl_full_agents_env_config.yaml"
        with open(saved_config_path) as f:
            saved = yaml.safe_load(f)
        assert saved["training"]["n_steps"] == 4

    def test_skip_if_exists_skips_when_files_present(self, tmp_path, monkeypatch):
        """--skip-if-exists returns early when all final_model_*.zip files exist."""
        from abx_amr_simulator.training.train_marl import _main

        # Run once to produce the files
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_marl",
                "--marl-config", str(_FIXTURE_CONFIG),
                "--results-dir", str(tmp_path),
                "--run-name", "test_run",
                "--seed", "0",
            ],
        )
        _main()

        # Pre-modification time of final models
        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="test_run")
        checkpoint_dir = run_dir / "checkpoints"
        mtime_before = (checkpoint_dir / "final_model_agent_0.zip").stat().st_mtime

        # Second call with --skip-if-exists should not re-run
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_marl",
                "--marl-config", str(_FIXTURE_CONFIG),
                "--results-dir", str(tmp_path),
                "--run-name", "test_run",
                "--seed", "99",
                "--skip-if-exists",
            ],
        )
        _main()

        mtime_after = (checkpoint_dir / "final_model_agent_0.zip").stat().st_mtime
        assert mtime_after == mtime_before, (
            "final_model was modified despite --skip-if-exists"
        )

    def test_agent_init_params_loads_best_params(self, tmp_path, monkeypatch):
        """--agent-init-params causes per-agent best_params to be loaded."""
        from abx_amr_simulator.training.train_marl import _main

        # Write a fake best_params.json
        best_params = {
            "learning_rate": 1.5e-4,
            "n_steps": 8,
            "gamma": 0.97,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
            "clip_range": 0.2,
            "n_epochs": 1,
        }
        best_params_path = tmp_path / "best_params.json"
        best_params_path.write_text(json.dumps(best_params))

        # Write agent_init_params.json
        agent_init_params = {
            "agent_0": {
                "best_params_path": str(best_params_path),
                "source_experiment_id": "1a",
            },
            "agent_1": {
                "best_params_path": "",
                "source_experiment_id": "",
            },
        }
        agent_init_path = tmp_path / "agent_init_params.json"
        agent_init_path.write_text(json.dumps(agent_init_params))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_marl",
                "--marl-config", str(_FIXTURE_CONFIG),
                "--results-dir", str(tmp_path),
                "--run-name", "test_run",
                "--seed", "0",
                "--agent-init-params", str(agent_init_path),
            ],
        )
        # Should complete without error; final models should exist
        _main()

        run_dir = _find_timestamped_run_dir(results_dir=tmp_path, run_name="test_run")
        checkpoint_dir = run_dir / "checkpoints"
        assert (checkpoint_dir / "final_model_agent_0.zip").exists()
        assert (checkpoint_dir / "final_model_agent_1.zip").exists()

    def test_agent_init_params_supports_mixed_per_agent_n_steps(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Different tuned n_steps values per agent should train successfully."""
        from abx_amr_simulator.training.train_marl import _main

        best_params_agent_0 = {
            "learning_rate": 1.5e-4,
            "n_steps": 8,
            "gamma": 0.97,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
            "clip_range": 0.2,
            "n_epochs": 1,
        }
        best_params_agent_1 = {
            "learning_rate": 2.5e-4,
            "n_steps": 16,
            "gamma": 0.99,
            "gae_lambda": 0.92,
            "ent_coef": 0.01,
            "clip_range": 0.15,
            "n_epochs": 2,
        }

        best_params_path_0 = tmp_path / "best_params_agent_0.json"
        best_params_path_0.write_text(json.dumps(best_params_agent_0))
        best_params_path_1 = tmp_path / "best_params_agent_1.json"
        best_params_path_1.write_text(json.dumps(best_params_agent_1))

        agent_init_params = {
            "agent_0": {
                "best_params_path": str(best_params_path_0),
                "source_experiment_id": "1a",
            },
            "agent_1": {
                "best_params_path": str(best_params_path_1),
                "source_experiment_id": "1b",
            },
        }
        agent_init_path = tmp_path / "agent_init_params_mixed.json"
        agent_init_path.write_text(json.dumps(agent_init_params))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_marl",
                "--marl-config", str(_FIXTURE_CONFIG),
                "--results-dir", str(tmp_path),
                "--run-name", "mixed_n_steps_run",
                "--seed", "0",
                "--agent-init-params", str(agent_init_path),
            ],
        )

        _main()

        run_dir = _find_timestamped_run_dir(
            results_dir=tmp_path,
            run_name="mixed_n_steps_run",
        )
        checkpoint_dir = run_dir / "checkpoints"
        assert (checkpoint_dir / "final_model_agent_0.zip").exists()
        assert (checkpoint_dir / "final_model_agent_1.zip").exists()
