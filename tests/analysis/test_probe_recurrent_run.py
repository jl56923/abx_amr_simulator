"""Tests for recurrent run probe entrypoint helper logic."""

from pathlib import Path

import pytest
import yaml

from abx_amr_simulator.analysis.probe_recurrent_run import (
    extract_algorithm_name,
    is_recurrent_algorithm,
    load_agent_and_environment,
    resolve_run_artifacts,
)


def test_extract_algorithm_name_flat_config() -> None:
    """Algorithm extraction should support flat config shape."""
    config = {"algorithm": "RecurrentPPO"}

    algorithm = extract_algorithm_name(config=config)

    assert algorithm == "RecurrentPPO"


def test_extract_algorithm_name_nested_config() -> None:
    """Algorithm extraction should support nested config shape."""
    config = {"agent_algorithm": {"algorithm": "HRL_RPPO"}}

    algorithm = extract_algorithm_name(config=config)

    assert algorithm == "HRL_RPPO"


def test_is_recurrent_algorithm_detection() -> None:
    """Recurrent detection should recognize recurrent and RPPO variants."""
    assert is_recurrent_algorithm(algorithm="RecurrentPPO") is True
    assert is_recurrent_algorithm(algorithm="HRL_RPPO") is True
    assert is_recurrent_algorithm(algorithm="PPO") is False


def test_resolve_run_artifacts_missing_model_raises(tmp_path: Path) -> None:
    """Artifact resolution should fail loudly when best_model.zip is missing."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "full_agent_env_config.yaml"
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(data={"algorithm": "RecurrentPPO"}, stream=config_file)

    with pytest.raises(FileNotFoundError, match="best_model.zip"):
        resolve_run_artifacts(run_dir=run_dir)


def test_resolve_run_artifacts_missing_config_raises(tmp_path: Path) -> None:
    """Artifact resolution should fail loudly when config yaml is missing."""
    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "best_model.zip").write_text(data="placeholder", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="full_agent_env_config.yaml"):
        resolve_run_artifacts(run_dir=run_dir)


def test_load_agent_and_environment_non_recurrent_fails_loudly(tmp_path: Path) -> None:
    """Loader should fail before model loading when algorithm is non-recurrent."""
    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # File only needs to exist for artifact checks; it is not loaded in this failure path.
    (checkpoints_dir / "best_model.zip").write_text(data="placeholder", encoding="utf-8")

    config_path = run_dir / "full_agent_env_config.yaml"
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(data={"algorithm": "PPO"}, stream=config_file)

    with pytest.raises(ValueError, match="requires recurrent algorithm"):
        load_agent_and_environment(run_dir=run_dir)
