"""Helpers for validating GUI configuration inputs."""

from pathlib import Path
from typing import Any
import yaml


def validate_umbrella_config_references(config_path: Path) -> tuple[list[str], list[str]]:
    """Validate that umbrella config references exist on disk.

    Returns:
        (errors, warnings): lists of message strings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f) or {}
    except Exception as exc:
        return [f"Failed to read config: {exc}"], warnings

    is_nested = any(
        isinstance(raw_config.get(key), str)
        for key in ["environment", "reward_calculator", "patient_generator", "agent_algorithm"]
    )
    if not is_nested:
        return errors, warnings

    umbrella_dir = config_path.parent
    if "config_folder_location" in raw_config:
        config_folder_location = raw_config["config_folder_location"]
        base_dir = Path(config_folder_location)
        if not base_dir.is_absolute():
            base_dir = (umbrella_dir / base_dir).resolve()
        else:
            base_dir = base_dir.resolve()
    else:
        base_dir = umbrella_dir
        warnings.append(
            "Umbrella config uses legacy relative paths without 'config_folder_location'. "
            "Consider adding config_folder_location for modern path resolution."
        )

    for key in ["environment", "reward_calculator", "patient_generator", "agent_algorithm"]:
        ref = raw_config.get(key)
        if isinstance(ref, str):
            ref_path = (base_dir / ref).resolve()
            if not ref_path.exists():
                errors.append(f"Missing {key} config: {ref_path}")

    return errors, warnings
