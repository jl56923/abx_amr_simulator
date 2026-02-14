"""Unit tests for training train.py helpers and CLI guardrails."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from abx_amr_simulator.training import train as train_module


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(file=path, mode="w", encoding="utf-8") as handle:
        yaml.safe_dump(data=data, stream=handle)


def test_validate_training_config_requires_environment(capsys: pytest.CaptureFixture[str]) -> None:
    config = {
        "training": {},
    }

    with pytest.raises(SystemExit) as excinfo:
        train_module.validate_training_config(config=config)

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Missing required config section: 'environment'" in captured.out


def test_validate_training_config_rejects_unsupported_algorithm(capsys: pytest.CaptureFixture[str]) -> None:
    config = {
        "environment": {},
        "training": {},
        "algorithm": "DQN",
    }

    with pytest.raises(SystemExit) as excinfo:
        train_module.validate_training_config(config=config)

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Unsupported algorithm: 'DQN'" in captured.out


def test_validate_training_config_requires_option_library_for_hrl(capsys: pytest.CaptureFixture[str]) -> None:
    config = {
        "environment": {},
        "training": {},
        "algorithm": "HRL_PPO",
    }

    with pytest.raises(SystemExit) as excinfo:
        train_module.validate_training_config(config=config)

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "requires 'option_library_path'" in captured.out


def test_validate_training_config_warns_on_missing_option_gamma(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    option_library_path = tmp_path / "options.yaml"
    option_library_path.write_text("options: []\n", encoding="utf-8")

    config = {
        "environment": {},
        "training": {},
        "algorithm": "HRL_PPO",
        "option_library_path": str(option_library_path),
    }

    train_module.validate_training_config(config=config)
    captured = capsys.readouterr()
    assert "CONFIG VALIDATION WARNINGS" in captured.out
    assert "option_gamma" in captured.out


def test_main_requires_absolute_umbrella_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--umbrella-config", "relative.yaml"])

    with pytest.raises(SystemExit) as excinfo:
        train_module.main()

    assert excinfo.value.code == 1


def test_main_skip_if_exists_exits_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_name = "skip_me"
    timestamp = "20260101_000000"

    config_path = tmp_path / "test_config.yaml"
    config = {
        "environment": {
            "max_time_steps": 2,
        },
        "training": {
            "total_num_training_episodes": 1,
            "run_name": run_name,
        },
        "output_dir": "results",
    }
    _write_yaml(path=config_path, data=config)

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    registry_path = results_dir / ".training_completed.txt"
    registry_path.write_text(f"{run_name},{timestamp}\n", encoding="utf-8")

    completed_run_dir = results_dir / f"{run_name}_{timestamp}"
    completed_run_dir.mkdir(parents=True, exist_ok=True)

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

    with pytest.raises(SystemExit) as excinfo:
        train_module.main()

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "SKIPPING: Experiment successfully completed" in captured.out
