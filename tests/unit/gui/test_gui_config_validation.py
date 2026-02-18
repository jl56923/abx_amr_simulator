"""Unit tests for GUI umbrella config validation helpers."""

from pathlib import Path
import yaml

from abx_amr_simulator.gui.config_validation import validate_umbrella_config_references


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False)


def test_validate_umbrella_config_references_ok(tmp_path: Path) -> None:
    """Valid nested config with explicit config folder location should pass."""
    config_root = tmp_path / "configs"
    _write_yaml(config_root / "environment" / "default.yaml", {"num_patients_per_time_step": 1})
    _write_yaml(config_root / "reward_calculator" / "default.yaml", {"lambda_weight": 0.5})
    _write_yaml(config_root / "patient_generator" / "default.yaml", {"visible_patient_attributes": ["prob_infected"]})
    _write_yaml(config_root / "agent_algorithm" / "default.yaml", {"algorithm": "PPO"})

    umbrella = {
        "config_folder_location": "../configs",
        "environment": "environment/default.yaml",
        "reward_calculator": "reward_calculator/default.yaml",
        "patient_generator": "patient_generator/default.yaml",
        "agent_algorithm": "agent_algorithm/default.yaml",
    }
    umbrella_path = tmp_path / "umbrella_configs" / "base.yaml"
    _write_yaml(umbrella_path, umbrella)

    errors, warnings = validate_umbrella_config_references(umbrella_path)

    assert errors == []
    assert warnings == []


def test_validate_umbrella_config_references_missing_file(tmp_path: Path) -> None:
    """Missing referenced component config should be reported as an error."""
    config_root = tmp_path / "configs"
    _write_yaml(config_root / "environment" / "default.yaml", {"num_patients_per_time_step": 1})

    umbrella = {
        "config_folder_location": "../configs",
        "environment": "environment/default.yaml",
        "reward_calculator": "reward_calculator/missing.yaml",
    }
    umbrella_path = tmp_path / "umbrella_configs" / "base.yaml"
    _write_yaml(umbrella_path, umbrella)

    errors, _ = validate_umbrella_config_references(umbrella_path)

    assert any("reward_calculator" in error for error in errors)


def test_validate_umbrella_config_references_legacy_warning(tmp_path: Path) -> None:
    """Legacy nested config without config_folder_location should warn but not error if paths exist."""
    umbrella_dir = tmp_path / "umbrella_configs"
    _write_yaml(umbrella_dir / "environment.yaml", {"num_patients_per_time_step": 1})

    umbrella = {
        "environment": "environment.yaml",
    }
    umbrella_path = umbrella_dir / "legacy.yaml"
    _write_yaml(umbrella_path, umbrella)

    errors, warnings = validate_umbrella_config_references(umbrella_path)

    assert errors == []
    assert warnings
