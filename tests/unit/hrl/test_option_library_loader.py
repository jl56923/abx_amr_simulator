"""Tests for OptionLibraryLoader error paths and config resolution."""

from __future__ import annotations

import sys
import types
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
    subconfig_path = tmp_path / "custom.yaml"
    _write_yaml(path=subconfig_path, data={"duration": 2})

    option_spec = {
        "option_name": "A_1",
        "option_type": "custom",
        "option_subconfig_file": str(subconfig_path),
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="must include a 'plugin' mapping"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_single_option_missing_paths_raises(tmp_path: Path) -> None:
    option_spec = {
        "option_name": "A_1",
        "option_type": "block",
        "option_subconfig_file": "block.yaml",
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="Option subconfig not found"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_import_loader_function_missing_loader_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subconfig_path = tmp_path / "block.yaml"
    _write_yaml(path=subconfig_path, data={"antibiotic": "A", "duration": 2})

    option_spec = {
        "option_name": "A_2",
        "option_type": "block",
        "option_subconfig_file": str(subconfig_path),
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    monkeypatch.setitem(
        OptionLibraryLoader._CANONICAL_LOADER_TARGETS,
        "block",
        ("abx_amr_simulator.options", "load_block_option_does_not_exist"),
    )

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="missing expected function 'load_block_option_does_not_exist'"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_import_loader_function_syntax_error_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subconfig_path = tmp_path / "block.yaml"
    _write_yaml(path=subconfig_path, data={"antibiotic": "A", "duration": 2})

    option_spec = {
        "option_name": "A_2",
        "option_type": "block",
        "option_subconfig_file": str(subconfig_path),
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    fake_module = types.ModuleType(name="_test_noncallable_loader_module")
    setattr(fake_module, "load_block_option", "not_callable")
    monkeypatch.setitem(sys.modules, "_test_noncallable_loader_module", fake_module)
    monkeypatch.setitem(
        OptionLibraryLoader._CANONICAL_LOADER_TARGETS,
        "block",
        ("_test_noncallable_loader_module", "load_block_option"),
    )

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="is not callable"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_single_option_with_module_loader(tmp_path: Path) -> None:
    subconfig_path = tmp_path / "heuristic.yaml"
    _write_yaml(
        path=subconfig_path,
        data={
            "duration": 2,
            "action_thresholds": {"prescribe_A": 0.5, "no_treatment": 0.0},
        },
    )

    option_spec = {
        "option_name": "HEURISTIC_test",
        "option_type": "heuristic",
        "option_subconfig_file": str(subconfig_path),
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    library, resolved_config = OptionLibraryLoader.load_library(
        library_config_path=str(config_path),
        env=env,
    )

    assert len(library) == 1
    assert resolved_config["num_options"] == 1


def test_module_loader_missing_expected_function_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subconfig_path = tmp_path / "block.yaml"
    _write_yaml(path=subconfig_path, data={"duration": 2})

    option_spec = {
        "option_name": "A_2",
        "option_type": "block",
        "option_subconfig_file": str(subconfig_path),
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    monkeypatch.setitem(
        OptionLibraryLoader._CANONICAL_LOADER_TARGETS,
        "block",
        ("abx_amr_simulator.options", "load_block_option"),
    )

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="missing expected function 'load_block_option'"):
        OptionLibraryLoader.load_library(
            library_config_path=str(config_path),
            env=env,
        )


def test_load_single_option_loader_returns_wrong_type(tmp_path: Path) -> None:
    subconfig_path = tmp_path / "block.yaml"
    _write_yaml(path=subconfig_path, data={"duration": 2})

    loader_path = tmp_path / "block_loader.py"
    _write_loader_module(
        path=loader_path,
        loader_body="""

def load_custom_option(config):
    return "not_an_option"
""",
    )

    option_spec = {
        "option_name": "A_2",
        "option_type": "custom",
        "option_subconfig_file": str(subconfig_path),
        "plugin": {
            "loader_module": str(loader_path),
            "loader_function": "load_custom_option",
        },
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="Plugin loader returned invalid type"):
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

def load_custom_option(config):
    raise ValueError("boom")
""",
    )

    option_spec = {
        "option_name": "A_2",
        "option_type": "custom",
        "option_subconfig_file": str(subconfig_path),
        "plugin": {
            "loader_module": str(loader_path),
            "loader_function": "load_custom_option",
        },
    }
    config_path = tmp_path / "library.yaml"
    _write_yaml(path=config_path, data=_build_library_config(option_specs=[option_spec]))

    env = create_mock_environment(antibiotic_names=["A"])
    with pytest.raises(RuntimeError, match="Plugin loader function raised an exception"):
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
        return np.full(shape=num_patients, fill_value='no_treatment', dtype=object)

    def get_referenced_antibiotics(self):
        return ["A"]


def load_block_option(name, config):
    return DummyOption(name=name, k=config.get("duration", 1))


def load_custom_option(config):
    return DummyOption(name=config.get("option_name", "unknown"), k=config.get("duration", 1))
""",
    )

    option_spec = {
        "option_name": "A_3",
        "option_type": "custom",
        "option_subconfig_file": subconfig_path.name,
        "plugin": {
            "loader_module": loader_path.name,
            "loader_function": "load_custom_option",
        },
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

    option_specs = [
        {
            "option_name": "A_1",
            "option_type": "block",
            "option_subconfig_file": subconfig_path.name,
            "config_params_override": {"duration": 2},
        },
        {
            "option_name": "A_3",
            "option_type": "block",
            "option_subconfig_file": subconfig_path.name,
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
