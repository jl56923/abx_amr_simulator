"""Integration tests for plugin seams through canonical train/tune flows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from abx_amr_simulator.core.leaky_balloon import AMR_LeakyBalloon
from abx_amr_simulator.core.patient_generator import PatientGenerator
from abx_amr_simulator.core.reward_calculator import RewardCalculator
from abx_amr_simulator.training import setup_optimization_folders_with_defaults
from abx_amr_simulator.utils import (
    create_amr_dynamics,
    create_patient_generator,
    create_reward_calculator,
    load_config,
    setup_config_folders_with_defaults,
)


def _write_yaml(path: Path, content: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data=content, stream=handle, sort_keys=False)


def _prepare_workspace(tmp_path: Path) -> Dict[str, Path]:
    experiments_dir = tmp_path / "experiments"
    setup_config_folders_with_defaults(target_path=experiments_dir)
    setup_optimization_folders_with_defaults(target_path=experiments_dir)

    results_dir = tmp_path / "results"
    optimization_dir = tmp_path / "optimization"
    results_dir.mkdir(parents=True, exist_ok=True)
    optimization_dir.mkdir(parents=True, exist_ok=True)

    return {
        "experiments_dir": experiments_dir,
        "results_dir": results_dir,
        "optimization_dir": optimization_dir,
    }


def _base_train_config(experiments_dir: Path, run_name: str) -> Dict[str, Any]:
    umbrella_path = (
        experiments_dir
        / "configs"
        / "umbrella_configs"
        / "base_experiment.yaml"
    )
    config = load_config(config_path=str(umbrella_path))

    config["training"]["run_name"] = run_name
    config["training"]["total_num_training_episodes"] = 1
    config["training"]["save_freq_every_n_episodes"] = 1
    config["training"]["eval_freq_every_n_episodes"] = 1
    config["training"]["num_eval_episodes"] = 1
    config["training"]["log_patient_trajectories"] = False

    config["environment"]["max_time_steps"] = 2
    config["environment"]["num_patients_per_time_step"] = 1

    config.setdefault("ppo", {})
    config["ppo"].update(
        {
            "n_steps": 2,
            "batch_size": 2,
            "n_epochs": 1,
            "verbose": 0,
        }
    )

    config["output_dir"] = "results"
    return config


def _base_tuning_config() -> Dict[str, Any]:
    return {
        "optimization": {
            "n_trials": 1,
            "n_seeds_per_trial": 1,
            "truncated_episodes": 1,
            "direction": "maximize",
            "sampler": "Random",
        },
        "search_space": {
            "learning_rate": {
                "type": "float",
                "low": 0.0001,
                "high": 0.001,
                "log": True,
            }
        },
    }


def _run_subprocess(command: list[str], cwd: Path) -> None:
    package_src = str((Path(__file__).resolve().parents[2] / "src").resolve())
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{package_src}:{existing_pythonpath}" if existing_pythonpath else package_src
    )

    result = subprocess.run(
        args=command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _find_single_run_dir(results_dir: Path, run_name: str) -> Path:
    matches = [
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith(f"{run_name}_")
    ]
    assert len(matches) == 1
    return matches[0]


def _plugin_path(module_filename: str) -> str:
    return str((Path(__file__).parent / "fixtures" / module_filename).resolve())


def _assert_runtime_marker(marker_path: Path, expected_token: str) -> None:
    assert marker_path.exists(), f"Expected runtime marker file to exist: {marker_path}"
    marker_text = marker_path.read_text(encoding="utf-8")
    assert expected_token in marker_text


def _build_config_with_plugin(
    experiments_dir: Path,
    run_name: str,
    plugin_section: str,
    plugin_module_filename: str,
    runtime_marker_path: Path,
) -> Dict[str, Any]:
    config = _base_train_config(experiments_dir=experiments_dir, run_name=run_name)

    if plugin_section == "patient_generator":
        config["patient_generator"]["plugin"] = {
            "loader_module": _plugin_path(module_filename=plugin_module_filename),
        }
        config["patient_generator"]["runtime_marker_path"] = str(runtime_marker_path)
    elif plugin_section == "reward_calculator":
        config["reward_calculator"]["plugin"] = {
            "loader_module": _plugin_path(module_filename=plugin_module_filename),
        }
        config["reward_calculator"]["runtime_marker_path"] = str(runtime_marker_path)
    elif plugin_section == "amr_dynamics":
        config["amr_dynamics"] = {
            "antibiotics_AMR_dict": dict(config["environment"]["antibiotics_AMR_dict"]),
            "runtime_marker_path": str(runtime_marker_path),
            "plugin": {
                "loader_module": _plugin_path(module_filename=plugin_module_filename),
            }
        }
    else:
        raise ValueError(f"Unsupported plugin section: {plugin_section}")

    return config


def test_patient_generator_plugin_via_train(tmp_path: Path) -> None:
    paths = _prepare_workspace(tmp_path=tmp_path)
    marker_path = tmp_path / "markers" / "patient_generator_train.marker"
    config = _build_config_with_plugin(
        experiments_dir=paths["experiments_dir"],
        run_name="plugin_pg_train",
        plugin_section="patient_generator",
        plugin_module_filename="custom_patient_generator_plugin.py",
        runtime_marker_path=marker_path,
    )
    config_path = tmp_path / "configs" / "plugin_pg_train.yaml"
    _write_yaml(path=config_path, content=config)

    _run_subprocess(
        command=[
            sys.executable,
            "-m",
            "abx_amr_simulator.training.train",
            "--umbrella-config",
            str(config_path),
            "--results-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).parent.parent,
    )

    _find_single_run_dir(results_dir=paths["results_dir"], run_name="plugin_pg_train")
    _assert_runtime_marker(
        marker_path=marker_path,
        expected_token="CustomPatientGeneratorPlugin",
    )


def test_patient_generator_plugin_via_tune(tmp_path: Path) -> None:
    paths = _prepare_workspace(tmp_path=tmp_path)
    run_name = "plugin_pg_tune"
    marker_path = tmp_path / "markers" / "patient_generator_tune.marker"
    config = _build_config_with_plugin(
        experiments_dir=paths["experiments_dir"],
        run_name=run_name,
        plugin_section="patient_generator",
        plugin_module_filename="custom_patient_generator_plugin.py",
        runtime_marker_path=marker_path,
    )
    config_path = tmp_path / "configs" / "plugin_pg_tune.yaml"
    _write_yaml(path=config_path, content=config)

    tuning_config_path = tmp_path / "configs" / "plugin_pg_tune_tuning.yaml"
    _write_yaml(path=tuning_config_path, content=_base_tuning_config())

    _run_subprocess(
        command=[
            sys.executable,
            "-m",
            "abx_amr_simulator.training.tune",
            "--umbrella-config",
            str(config_path),
            "--tuning-config",
            str(tuning_config_path),
            "--run-name",
            run_name,
            "--optimization-dir",
            str(paths["optimization_dir"]),
            "--results-dir",
            str(paths["results_dir"]),
        ],
        cwd=Path(__file__).parent.parent,
    )

    assert (paths["optimization_dir"] / run_name / "best_params.json").exists()
    _assert_runtime_marker(
        marker_path=marker_path,
        expected_token="CustomPatientGeneratorPlugin",
    )


def test_reward_calculator_plugin_via_train(tmp_path: Path) -> None:
    paths = _prepare_workspace(tmp_path=tmp_path)
    marker_path = tmp_path / "markers" / "reward_calculator_train.marker"
    config = _build_config_with_plugin(
        experiments_dir=paths["experiments_dir"],
        run_name="plugin_rc_train",
        plugin_section="reward_calculator",
        plugin_module_filename="custom_reward_calculator_plugin.py",
        runtime_marker_path=marker_path,
    )
    config_path = tmp_path / "configs" / "plugin_rc_train.yaml"
    _write_yaml(path=config_path, content=config)

    _run_subprocess(
        command=[
            sys.executable,
            "-m",
            "abx_amr_simulator.training.train",
            "--umbrella-config",
            str(config_path),
            "--results-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).parent.parent,
    )

    _find_single_run_dir(results_dir=paths["results_dir"], run_name="plugin_rc_train")
    _assert_runtime_marker(
        marker_path=marker_path,
        expected_token="CustomRewardCalculatorPlugin",
    )


def test_amr_dynamics_plugin_via_train(tmp_path: Path) -> None:
    paths = _prepare_workspace(tmp_path=tmp_path)
    marker_path = tmp_path / "markers" / "amr_dynamics_train.marker"
    config = _build_config_with_plugin(
        experiments_dir=paths["experiments_dir"],
        run_name="plugin_amr_train",
        plugin_section="amr_dynamics",
        plugin_module_filename="custom_amr_dynamics_plugin.py",
        runtime_marker_path=marker_path,
    )
    config_path = tmp_path / "configs" / "plugin_amr_train.yaml"
    _write_yaml(path=config_path, content=config)

    _run_subprocess(
        command=[
            sys.executable,
            "-m",
            "abx_amr_simulator.training.train",
            "--umbrella-config",
            str(config_path),
            "--results-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).parent.parent,
    )

    _find_single_run_dir(results_dir=paths["results_dir"], run_name="plugin_amr_train")
    _assert_runtime_marker(
        marker_path=marker_path,
        expected_token="CustomAMRDynamicsPlugin",
    )


def test_canonical_patient_generator_unchanged(tmp_path: Path) -> None:
    paths = _prepare_workspace(tmp_path=tmp_path)
    config = _base_train_config(experiments_dir=paths["experiments_dir"], run_name="canonical_pg")
    patient_generator = create_patient_generator(config=config)
    assert type(patient_generator) is PatientGenerator


def test_canonical_reward_calculator_unchanged(tmp_path: Path) -> None:
    paths = _prepare_workspace(tmp_path=tmp_path)
    config = _base_train_config(experiments_dir=paths["experiments_dir"], run_name="canonical_rc")
    reward_calculator = create_reward_calculator(config=config)
    assert type(reward_calculator) is RewardCalculator


def test_canonical_amr_dynamics_unchanged(tmp_path: Path) -> None:
    paths = _prepare_workspace(tmp_path=tmp_path)
    config = _base_train_config(experiments_dir=paths["experiments_dir"], run_name="canonical_amr")
    amr_dynamics = create_amr_dynamics(config=config)
    assert all(type(model) is AMR_LeakyBalloon for model in amr_dynamics.values())
