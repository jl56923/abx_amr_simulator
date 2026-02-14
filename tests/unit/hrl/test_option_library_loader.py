"""Tests for OptionLibraryLoader error paths and config resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from abx_amr_simulator.hrl import OptionLibraryLoader

# Import test helpers (sys.path configured in tests/conftest.py)
from test_reference_helpers import create_mock_environment  # type: ignore[import-not-found]


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data))


def _write_loader_module(path: Path, loader_body: str) -> None:
    path.write_text(loader_body)


def _build_library_config(
    option_specs: list,
    library_name: str = "test_library",
) -> dict:
    return {
        "library_name": library_name,
        "description": "Test library",
        "options": option_specs,
    }


def test_load_library_missing_config_file_raises(tmp_path: Path) -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        OptionLibraryLoader.load_library(
            library_config_path=str(missing_path),
            env=env,
        )


def test_load_library_empty_config_raises(tmp_path: Path) -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    config_path = tmp_path / "library.yaml"
    config_path.write_text("")

    with pytest.raises(ValueError, match="empty"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_library_no_options_raises(tmp_path: Path) -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[]))

    with pytest.raises(ValueError, match="no options"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_library_missing_option_name_raises(tmp_path: Path) -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    config_path = tmp_path / "library.yaml"
    option_specs = [
        {
            "option_type": "block",
            "option_subconfig_file": "block.yaml",
            "loader_module": "block_loader.py",
        }
    ]
    _write_yaml(path=config_path, data=_build_library_config(option_specs=option_specs))

    with pytest.raises(RuntimeError, match="missing 'option_name'"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_library_missing_option_type_raises(tmp_path: Path) -> None:
    env = create_mock_environment(antibiotic_names=["A"])
    config_path = tmp_path / "library.yaml"
    option_specs = [
        {
            "option_name": "A_1",
            "option_subconfig_file": "block.yaml",
            "loader_module": "block_loader.py",
        }
    ]
    _write_yaml(path=config_path, data=_build_library_config(option_specs=option_specs))

    with pytest.raises(RuntimeError, match="missing 'option_type'"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_single_option_missing_subconfig_raises(tmp_path: Path) -> None:
    option_spec = {
        "option_name": "A_1",
        "option_type": "block",
        "loader_module": "block_loader.py",
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="missing 'option_subconfig_file'"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_single_option_missing_loader_module_raises(tmp_path: Path) -> None:
    option_spec = {
        "option_name": "A_1",
        "option_type": "block",
        "option_subconfig_file": "block.yaml",
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="missing 'loader_module'"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_single_option_missing_paths_raises(tmp_path: Path) -> None:
    option_spec = {
        "option_name": "A_1",
        "option_type": "block",
        "option_subconfig_file": "block.yaml",
        "loader_module": "block_loader.py",
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="Option subconfig not found"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_import_loader_function_missing_loader_raises(tmp_path: Path) -> None:
    loader_path = tmp_path / "block_loader.py"
    loader_path.write_text(
        """

def not_a_loader(name, config):
    return None
"""
    )

    with pytest.raises(RuntimeError, match="missing 'load_block_option'"):
        OptionLibraryLoader._import_loader_function(
            option_type="block",
            loader_module_path=loader_path,
        )


def test_import_loader_function_syntax_error_raises(tmp_path: Path) -> None:
    loader_path = tmp_path / "bad_loader.py"
    loader_path.write_text("def broken(:\n    pass\n")

    with pytest.raises(SyntaxError, match="Syntax error"):
        OptionLibraryLoader._import_loader_function(
            option_type="block",
            loader_module_path=loader_path,
        )


def test_load_single_option_loader_returns_wrong_type(tmp_path: Path) -> None:
    subconfig_path = tmp_path / "block.yaml"
    _write_yaml(path=subconfig_path, data={"duration": 2})

    loader_path = tmp_path / "block_loader.py"
    _write_loader_module(
        path=loader_path,
        loader_body="""

def load_block_option(name, config):
    return "not_an_option"
""",
    )

    option_spec = {
        "option_name": "A_2",
        "option_type": "block",
        "option_subconfig_file": str(subconfig_path),
        "loader_module": str(loader_path),
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="expected OptionBase"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_single_option_loader_raises_runtime_error(tmp_path: Path) -> None:
    subconfig_path = tmp_path / "block.yaml"
    _write_yaml(path=subconfig_path, data={"duration": 2})

    loader_path = tmp_path / "block_loader.py"
    _write_loader_module(
        path=loader_path,
        loader_body="""

def load_block_option(name, config):
    raise ValueError("boom")
""",
    )

    option_spec = {
        "option_name": "A_2",
        "option_type": "block",
        "option_subconfig_file": str(subconfig_path),
        "loader_module": str(loader_path),
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="raised error"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_single_option_merges_config_and_resolves_paths(tmp_path: Path) -> None:
    subconfig_path = tmp_path / "block.yaml"
    _write_yaml(path=subconfig_path, data={"duration": 3, "antibiotic": "A"})

    loader_path = tmp_path / "block_loader.py"
    _write_loader_module(
        path=loader_path,
        loader_body="""
import numpy as np
from abx_amr_simulator.hrl.base_option import OptionBase

class DummyOption(OptionBase):
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = False

    def __init__(self, name, k):
        super().__init__(name=name, k=k)

    def decide(self, env_state):
        num_patients = env_state.get("num_patients", 1)
        return np.zeros(shape=num_patients, dtype=np.int32)

    def get_referenced_antibiotics(self):
        return ["A"]


def load_block_option(name, config):
    return DummyOption(name=name, k=config.get("duration", 1))
""",
    )

    option_spec = {
        "option_name": "A_3",
        "option_type": "block",
        "option_subconfig_file": subconfig_path.name,
        "loader_module": loader_path.name,
        "config_params_override": {"duration": 5},
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    library, resolved = OptionLibraryLoader.load_library(
        library_config_path=str(config_path),
        env=env,
    )

    assert len(library) == 1
    assert resolved["options"][0]["merged_config"]["duration"] == 5
    assert resolved["options"][0]["default_config"]["duration"] == 3
    assert resolved["options"][0]["overrides"]["duration"] == 5
    assert resolved["options"][0]["k"] == 5


def test_load_library_handles_multiple_options(tmp_path: Path) -> None:
    subconfig_path = tmp_path / "block.yaml"
    _write_yaml(path=subconfig_path, data={"duration": 2, "antibiotic": "A"})

    loader_path = tmp_path / "block_loader.py"
    _write_loader_module(
        path=loader_path,
        loader_body="""
import numpy as np
from abx_amr_simulator.hrl.base_option import OptionBase

class DummyOption(OptionBase):
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = False

    def __init__(self, name, k):
        super().__init__(name=name, k=k)

    def decide(self, env_state):
        num_patients = env_state.get("num_patients", 1)
        return np.zeros(shape=num_patients, dtype=np.int32)

    def get_referenced_antibiotics(self):
        return ["A"]


def load_block_option(name, config):
    return DummyOption(name=name, k=config.get("duration", 1))
""",
    )

    option_specs = [
        {
            "option_name": "A_1",
            "option_type": "block",
            "option_subconfig_file": subconfig_path.name,
            "loader_module": loader_path.name,
            "config_params_override": {"duration": 2},
        },
        {
            "option_name": "A_3",
            "option_type": "block",
            "option_subconfig_file": subconfig_path.name,
            "loader_module": loader_path.name,
            "config_params_override": {"duration": 3},
        },
    ]
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=option_specs))

    env = create_mock_environment(antibiotic_names=["A"])
    library, resolved = OptionLibraryLoader.load_library(
        library_config_path=str(config_path),
        env=env,
    )

    assert len(library) == 2
    assert resolved["num_options"] == 2
    assert resolved["options"][0]["option_name"] == "A_1"
    assert resolved["options"][1]["option_name"] == "A_3"
