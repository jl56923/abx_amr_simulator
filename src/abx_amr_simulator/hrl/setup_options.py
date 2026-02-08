"""Utilities for creating option library scaffolding."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union


def setup_options_folders_with_defaults(target_path: Union[str, Path]) -> Path:
    """Create option library scaffold populated with minimal defaults.

    Creates nested structure:
        target_path/option_libraries/
        target_path/option_types/block/
        target_path/option_types/alternation/

    Also writes minimal default option type configs if they do not exist.

    Args:
        target_path: Base directory where option_libraries/ and option_types/ are created.

    Returns:
        Path to the created option_libraries directory.
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

    block_template = block_dir / "block_option_default_config.yaml"
    if not block_template.exists():
        block_template.write_text(
            "# Default config for block option\n"
            "antibiotic: \"A\"\n"
            "duration: 5\n",
            encoding="utf-8",
        )

    alternation_template = alternation_dir / "alternation_option_default_config.yaml"
    if not alternation_template.exists():
        alternation_template.write_text(
            "# Default config for alternation option\n"
            "sequence:\n"
            "  - \"A\"\n"
            "  - \"B\"\n",
            encoding="utf-8",
        )

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
