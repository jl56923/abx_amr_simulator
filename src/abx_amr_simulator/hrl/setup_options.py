"""Utilities for creating option library scaffolding."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path
from typing import Union


def setup_options_folders_with_defaults(target_path: Union[str, Path]) -> Path:
    """Create option library scaffold populated with bundled default options.

    Copies bundled default option configs and loaders from package into user's target directory.
    Creates nested structure:
        target_path/option_libraries/default_deterministic.yaml
        target_path/option_types/block/block_option_default_config.yaml
        target_path/option_types/block/block_option_loader.py
        target_path/option_types/alternation/alternation_option_default_config.yaml
        target_path/option_types/alternation/alternation_option_loader.py

    Useful for initializing new experiment directories with working baseline option libraries.

    Args:
        target_path: Base directory where option_libraries/ and option_types/ are created.

    Returns:
        Path to the created option_libraries directory.

    Example:
        >>> from abx_amr_simulator.hrl import setup_options_folders_with_defaults
        >>> library_path = setup_options_folders_with_defaults('experiments/options')
        >>> # Creates: experiments/options/option_libraries/default_deterministic.yaml, etc.
    """
    if target_path is None:
        raise ValueError("target_path must be provided.")

    base = Path(target_path)
    option_libraries_dir = base / "option_libraries"
    option_types_dir = base / "option_types"
    block_dir = option_types_dir / "block"
    alternation_dir = option_types_dir / "alternation"

    for directory in [option_libraries_dir, block_dir, alternation_dir]:
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

    return option_libraries_dir


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
    print(f"Created option scaffolding at: {created_path.parent}")


if __name__ == "__main__":
    main()
