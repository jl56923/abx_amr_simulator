"""Utilities for creating hyperparameter tuning scaffolding."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path
from typing import Union


def setup_optimization_folders_with_defaults(target_path: Union[str, Path]) -> Path:
    """Create tuning config scaffold populated with bundled default tuning configs.

    Creates a top-level 'tuning_configs' folder with bundled default configs:
        target_path/tuning_configs/ppo_tuning_default.yaml
        target_path/tuning_configs/hrl_ppo_tuning_default.yaml

    Copies bundled default tuning configs from package into user's target directory.
    Useful for initializing new experiment directories with working baseline tuning configs.

    Args:
        target_path: Base directory where the top-level 'tuning_configs' folder will be created.

    Returns:
        Path to the created tuning_configs directory.

    Raises:
        ValueError: If target_path is None.

    Example:
        >>> from abx_amr_simulator.training import setup_optimization_folders_with_defaults
        >>> tuning_path = setup_optimization_folders_with_defaults('experiments')
        >>> # Creates: experiments/tuning_configs/ppo_tuning_default.yaml, etc.
    
    CLI Usage:
        >>> python -m abx_amr_simulator.training.setup_tuning --target-path experiments
    """
    if target_path is None:
        raise ValueError("target_path must be provided.")

    base = Path(target_path)
    tuning_configs_dir = base / "tuning_configs"
    tuning_configs_dir.mkdir(parents=True, exist_ok=True)

    # Get bundled defaults from package
    defaults_root = files("abx_amr_simulator").joinpath("tuning/defaults")

    # Copy tuning config files
    ppo_tuning_src = defaults_root.joinpath("ppo_tuning_default.yaml")
    ppo_tuning_dst = tuning_configs_dir / "ppo_tuning_default.yaml"
    ppo_tuning_dst.write_bytes(ppo_tuning_src.read_bytes())

    hrl_ppo_tuning_src = defaults_root.joinpath("hrl_ppo_tuning_default.yaml")
    hrl_ppo_tuning_dst = tuning_configs_dir / "hrl_ppo_tuning_default.yaml"
    hrl_ppo_tuning_dst.write_bytes(hrl_ppo_tuning_src.read_bytes())

    return tuning_configs_dir


def main() -> None:
    """CLI entry point for tuning config scaffolding."""
    parser = argparse.ArgumentParser(
        description="Create tuning config scaffold with default templates."
    )
    parser.add_argument(
        "--target-path",
        type=str,
        required=True,
        help="Directory where tuning config scaffolding will be created.",
    )

    args = parser.parse_args()
    created_path = setup_optimization_folders_with_defaults(target_path=args.target_path)
    print(f"Created tuning config scaffolding at: {created_path}")


if __name__ == "__main__":
    main()
