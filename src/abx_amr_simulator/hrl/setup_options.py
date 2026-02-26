"""Utilities for creating option library scaffolding."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path
from typing import Union


def setup_options_folders_with_defaults(target_path: Union[str, Path]) -> Path:
    """Create option library scaffold populated with bundled default options.

    Creates a top-level 'options' folder with nested structure:
        target_path/options/option_libraries/default_deterministic.yaml
        target_path/options/option_types/block/block_option_default_config.yaml
        target_path/options/option_types/block/block_option_loader.py
        target_path/options/option_types/alternation/alternation_option_default_config.yaml
        target_path/options/option_types/alternation/alternation_option_loader.py
        target_path/options/option_types/heuristic/heuristic_option_default_config.yaml
        target_path/options/option_types/heuristic/heuristic_option_loader.py

    Copies bundled default option configs and loaders from package into user's target directory.
    Useful for initializing new experiment directories with working baseline option libraries.

    Args:
        target_path: Base directory where the top-level 'options' folder will be created.

    Returns:
        Path to the created options directory.

    Example:
        >>> from abx_amr_simulator.hrl import setup_options_folders_with_defaults
        >>> options_path = setup_options_folders_with_defaults('experiments')
        >>> # Creates: experiments/options/option_libraries/default_deterministic.yaml, etc.
    """
    if target_path is None:
        raise ValueError("target_path must be provided.")

    base = Path(target_path)
    options_dir = base / "options"
    option_libraries_dir = options_dir / "option_libraries"
    option_types_dir = options_dir / "option_types"
    block_dir = option_types_dir / "block"
    alternation_dir = option_types_dir / "alternation"
    heuristic_dir = option_types_dir / "heuristic"

    for directory in [option_libraries_dir, block_dir, alternation_dir, heuristic_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Get bundled defaults from package
    defaults_root = files("abx_amr_simulator").joinpath("options/defaults")

    # Copy option library configs
    library_src = defaults_root.joinpath("option_libraries/default_deterministic.yaml")
    library_dst = option_libraries_dir / "default_deterministic.yaml"
    library_dst.write_bytes(library_src.read_bytes())

    # Copy block option files
    block_config_src = defaults_root.joinpath("option_types/block/block_option_default_config.yaml")
    block_config_dst = block_dir / "block_option_default_config.yaml"
    block_config_dst.write_bytes(block_config_src.read_bytes())

    block_loader_src = defaults_root.joinpath("option_types/block/block_option_loader.py")
    block_loader_dst = block_dir / "block_option_loader.py"
    block_loader_dst.write_bytes(block_loader_src.read_bytes())

    # Copy alternation option files
    alt_config_src = defaults_root.joinpath("option_types/alternation/alternation_option_default_config.yaml")
    alt_config_dst = alternation_dir / "alternation_option_default_config.yaml"
    alt_config_dst.write_bytes(alt_config_src.read_bytes())

    alt_loader_src = defaults_root.joinpath("option_types/alternation/alternation_option_loader.py")
    alt_loader_dst = alternation_dir / "alternation_option_loader.py"
    alt_loader_dst.write_bytes(alt_loader_src.read_bytes())

    # Copy heuristic option files
    heur_config_src = defaults_root.joinpath("option_types/heuristic/heuristic_option_default_config.yaml")
    heur_config_dst = heuristic_dir / "heuristic_option_default_config.yaml"
    heur_config_dst.write_bytes(heur_config_src.read_bytes())

    heur_loader_src = defaults_root.joinpath("option_types/heuristic/heuristic_option_loader.py")
    heur_loader_dst = heuristic_dir / "heuristic_option_loader.py"
    heur_loader_dst.write_bytes(heur_loader_src.read_bytes())

    return options_dir


def main() -> None:
    """CLI entry point for option scaffolding."""
    parser = argparse.ArgumentParser(
        description="Create option library scaffold with default templates."
    )
    parser.add_argument(
        "--target-path",
        type=str,
        required=True,
        help="Directory where option scaffolding will be created.",
    )

    args = parser.parse_args()
    created_path = setup_options_folders_with_defaults(target_path=args.target_path)
    print(f"Created option scaffolding at: {created_path}")


if __name__ == "__main__":
    main()
