"""Tests for block and alternation option loaders."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import sys

from abx_amr_simulator.hrl import OptionLibrary, OptionLibraryLoader


UTILS_PATH = Path(__file__).resolve().parent.parent / "utils"
sys.path = [f"{UTILS_PATH}"] + sys.path

from test_reference_helpers import create_mock_environment


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(
        name=module_name,
        location=str(module_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec=spec)
    spec.loader.exec_module(module=module)
    return module


def _build_env_state(option_library: OptionLibrary, current_step: int) -> dict:
    return {
        "patients": [],
        "num_patients": 2,
        "current_amr_levels": {},
        "current_step": current_step,
        "max_steps": 10,
        "option_library": option_library,
    }


def test_block_option_decide_returns_single_action():
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=2,
    )
    option_library = OptionLibrary(env=env)

    module_path = (
        Path(__file__).resolve().parents[5]
        / "workspace"
        / "experiments"
        / "options"
        / "option_types"
        / "block"
        / "block_option_loader.py"
    )
    module = _load_module(module_name="block_option_loader", module_path=module_path)

    option = module.BlockOption(name="A_5", antibiotic="A", duration=5)
    env_state = _build_env_state(option_library=option_library, current_step=0)

    actions = option.decide(env_state=env_state)

    expected = option_library.abx_name_to_index["A"]
    assert actions.tolist() == [expected, expected]


def test_block_option_supports_no_rx_alias():
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=2,
    )
    option_library = OptionLibrary(env=env)

    module_path = (
        Path(__file__).resolve().parents[5]
        / "workspace"
        / "experiments"
        / "options"
        / "option_types"
        / "block"
        / "block_option_loader.py"
    )
    module = _load_module(module_name="block_option_loader", module_path=module_path)

    option = module.BlockOption(name="NO_RX_5", antibiotic="NO_RX", duration=5)
    env_state = _build_env_state(option_library=option_library, current_step=0)

    actions = option.decide(env_state=env_state)

    expected = option_library.abx_name_to_index["no_treatment"]
    assert actions.tolist() == [expected, expected]


def test_block_loader_validates_required_keys():
    module_path = (
        Path(__file__).resolve().parents[5]
        / "workspace"
        / "experiments"
        / "options"
        / "option_types"
        / "block"
        / "block_option_loader.py"
    )
    module = _load_module(module_name="block_option_loader", module_path=module_path)

    with pytest.raises(ValueError):
        module.load_block_option(name="bad", config={})

    with pytest.raises(ValueError):
        module.load_block_option(name="bad", config={"antibiotic": "A"})

    with pytest.raises(ValueError):
        module.load_block_option(name="bad", config={"duration": 5})


def test_alternation_option_sequences_actions():
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=2,
    )
    option_library = OptionLibrary(env=env)

    module_path = (
        Path(__file__).resolve().parents[5]
        / "workspace"
        / "experiments"
        / "options"
        / "option_types"
        / "alternation"
        / "alternation_option_loader.py"
    )
    module = _load_module(module_name="alternation_option_loader", module_path=module_path)

    option = module.AlternationOption(
        name="ALT_A_NO_RX_B",
        sequence=["A", "NO_RX", "B"],
    )

    env_state = _build_env_state(option_library=option_library, current_step=0)
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == [option_library.abx_name_to_index["A"]] * 2

    env_state = _build_env_state(option_library=option_library, current_step=1)
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == [option_library.abx_name_to_index["no_treatment"]] * 2

    env_state = _build_env_state(option_library=option_library, current_step=2)
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == [option_library.abx_name_to_index["B"]] * 2


def test_alternation_resets_on_step_gap():
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=2,
    )
    option_library = OptionLibrary(env=env)

    module_path = (
        Path(__file__).resolve().parents[5]
        / "workspace"
        / "experiments"
        / "options"
        / "option_types"
        / "alternation"
        / "alternation_option_loader.py"
    )
    module = _load_module(module_name="alternation_option_loader", module_path=module_path)

    option = module.AlternationOption(
        name="ALT_A_B",
        sequence=["A", "B"],
    )

    env_state = _build_env_state(option_library=option_library, current_step=0)
    _ = option.decide(env_state=env_state)

    env_state = _build_env_state(option_library=option_library, current_step=2)
    actions = option.decide(env_state=env_state)

    assert actions.tolist() == [option_library.abx_name_to_index["A"]] * 2


def test_alternation_loader_validates_sequence():
    module_path = (
        Path(__file__).resolve().parents[5]
        / "workspace"
        / "experiments"
        / "options"
        / "option_types"
        / "alternation"
        / "alternation_option_loader.py"
    )
    module = _load_module(module_name="alternation_option_loader", module_path=module_path)

    with pytest.raises(ValueError):
        module.load_alternation_option(name="bad", config={})

    with pytest.raises(ValueError):
        module.load_alternation_option(name="bad", config={"sequence": []})

    with pytest.raises(ValueError):
        module.load_alternation_option(name="bad", config={"sequence": [1, 2]})


def test_default_deterministic_library_loads():
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=1,
    )
    library_path = (
        Path(__file__).resolve().parents[5]
        / "workspace"
        / "experiments"
        / "options"
        / "option_libraries"
        / "default_deterministic.yaml"
    )

    library, resolved = OptionLibraryLoader.load_library(
        library_config_path=str(library_path),
        env=env,
    )

    assert len(library) == 12
    assert resolved["num_options"] == 12
    assert "options" in resolved
