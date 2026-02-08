"""Tests for setup_options_folders_with_defaults() helper."""

import tempfile
from pathlib import Path

import yaml

from abx_amr_simulator.hrl import setup_options_folders_with_defaults


def test_creates_option_folder_structure():
    """Ensure option scaffold directories are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        options_root = Path(tmpdir) / "options"

        setup_options_folders_with_defaults(target_path=options_root)

        assert (options_root / "option_libraries").exists()
        assert (options_root / "option_types" / "block").exists()
        assert (options_root / "option_types" / "alternation").exists()


def test_creates_default_template_files():
    """Ensure default option template configs are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        options_root = Path(tmpdir) / "options"

        setup_options_folders_with_defaults(target_path=options_root)

        assert (options_root / "option_types" / "block" / "block_option_default_config.yaml").exists()
        assert (
            options_root / "option_types" / "alternation" / "alternation_option_default_config.yaml"
        ).exists()


def test_templates_are_valid_yaml():
    """Ensure template files contain valid YAML content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        options_root = Path(tmpdir) / "options"

        setup_options_folders_with_defaults(target_path=options_root)

        templates = [
            options_root / "option_types" / "block" / "block_option_default_config.yaml",
            options_root
            / "option_types"
            / "alternation"
            / "alternation_option_default_config.yaml",
        ]

        for template in templates:
            with open(file=template, mode="r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
                assert config is not None


def test_idempotent_operation():
    """Ensure setup function can run multiple times without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        options_root = Path(tmpdir) / "options"

        setup_options_folders_with_defaults(target_path=options_root)
        setup_options_folders_with_defaults(target_path=options_root)

        assert (options_root / "option_libraries").exists()


def test_returns_option_libraries_path():
    """Ensure setup returns path to option_libraries directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        options_root = Path(tmpdir) / "options"

        created_path = setup_options_folders_with_defaults(target_path=options_root)

        assert created_path == options_root / "option_libraries"
