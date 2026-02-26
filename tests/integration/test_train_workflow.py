"""Integration tests for train.py workflows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from abx_amr_simulator.training import train as train_module
from abx_amr_simulator.utils import load_config, setup_config_folders_with_defaults
from abx_amr_simulator.hrl import setup_options_folders_with_defaults


def _write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def _prepare_base_config(tmp_path: Path, run_name: str) -> dict:
    experiments_dir = tmp_path / "experiments"
    setup_config_folders_with_defaults(target_path=experiments_dir)

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


def _find_run_dir(results_dir: Path, run_name: str) -> Path:
    matches = [
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith(f"{run_name}_")
    ]
    assert len(matches) == 1
    return matches[0]


def _disable_plotting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_module, "plot_metrics_trained_agent", lambda **_: None)


def test_fresh_training_creates_artifacts_and_cleans_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_name = "train_smoke"
    config = _prepare_base_config(tmp_path=tmp_path, run_name=run_name)
    config_path = tmp_path / "configs" / "train_smoke.yaml"
    _write_config(path=config_path, config=config)

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    registry_path = results_dir / ".training_completed.txt"
    registry_path.write_text("stale_run,20260101_000000\n", encoding="utf-8")

    _disable_plotting(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--umbrella-config",
            str(config_path),
            "--results-dir",
            str(tmp_path),
            "--skip-if-exists",
        ],
    )

    train_module.main()

    run_dir = _find_run_dir(results_dir=results_dir, run_name=run_name)
    assert (run_dir / "logs").is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "eval_logs").is_dir()
    assert (run_dir / "full_agent_env_config.yaml").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "checkpoints" / "final_model.zip").exists()

    registry_contents = registry_path.read_text(encoding="utf-8")
    assert "stale_run" not in registry_contents
    assert run_name in registry_contents


def test_resume_training_creates_continued_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_name = "train_resume"
    config = _prepare_base_config(tmp_path=tmp_path, run_name=run_name)
    config_path = tmp_path / "configs" / "train_resume.yaml"
    _write_config(path=config_path, config=config)

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _disable_plotting(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--umbrella-config",
            str(config_path),
            "--results-dir",
            str(tmp_path),
        ],
    )
    train_module.main()

    prior_run_dir = _find_run_dir(results_dir=results_dir, run_name=run_name)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--train-from-prior-results",
            str(prior_run_dir),
            "--additional-training-episodes",
            "1",
            "--results-dir",
            str(tmp_path),
        ],
    )

    train_module.main()

    continued_run_dir = _find_run_dir(
        results_dir=results_dir,
        run_name=f"{run_name}_continued",
    )

    continued_config_path = continued_run_dir / "full_agent_env_config.yaml"
    assert continued_config_path.exists()
    continued_config = yaml.safe_load(continued_config_path.read_text(encoding="utf-8"))
    assert continued_config["training"]["continued_from"] == str(prior_run_dir)
    assert (continued_run_dir / "checkpoints" / "final_model.zip").exists()


def test_hrl_training_saves_option_library_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_name = "train_hrl"
    config = _prepare_base_config(tmp_path=tmp_path, run_name=run_name)

    experiments_dir = tmp_path / "experiments"
    options_dir = setup_options_folders_with_defaults(target_path=experiments_dir)
    option_library_path = options_dir / "option_libraries" / "default_deterministic.yaml"

    config["algorithm"] = "HRL_PPO"
    config["hrl"] = {
        "option_library": "option_libraries/default_deterministic.yaml",
        "option_gamma": 0.99,
    }
    config["option_gamma"] = 0.99
    config["option_library_path"] = str(option_library_path)

    config_path = tmp_path / "configs" / "train_hrl.yaml"
    _write_config(path=config_path, config=config)

    _disable_plotting(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--umbrella-config",
            str(config_path),
            "--results-dir",
            str(tmp_path),
        ],
    )

    train_module.main()

    results_dir = tmp_path / "results"
    run_dir = _find_run_dir(results_dir=results_dir, run_name=run_name)
    assert (run_dir / "full_options_library.yaml").exists()
