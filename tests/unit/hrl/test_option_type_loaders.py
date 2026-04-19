"""Tests for block and alternation option loaders.

IMPORTANT: These tests load option loaders from the bundled package
(external/abx_amr_simulator/src/abx_amr_simulator/options/defaults/),
NOT from the workspace folder. This ensures complete independence from
any user workspace configuration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from abx_amr_simulator.hrl import OptionLibrary, OptionLibraryLoader

# Import test helpers from centralized location
# (sys.path is configured in tests/conftest.py)
from test_reference_helpers import create_mock_environment  # type: ignore[import-not-found]

def _get_bundled_option_loader_path(loader_type: str) -> Path:
    """Get path to bundled option loader from installed package.
    
    Args:
        loader_type: 'block' or 'alternation'
    
    Returns:
        Path to the loader module in the package.
    """
    # Path from test: external/abx_amr_simulator/tests/unit/hrl/test_option_type_loaders.py
    # parents[3] = external/abx_amr_simulator
    package_root = Path(__file__).resolve().parents[3]  # external/abx_amr_simulator
    loader_path = (
        package_root
        / "src"
        / "abx_amr_simulator"
        / "options"
        / "defaults"
        / "option_types"
        / loader_type
        / f"{loader_type}_option_loader.py"
    )
    if not loader_path.exists():
        raise FileNotFoundError(f"Bundled {loader_type} option loader not found: {loader_path}")
    return loader_path


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


def _build_env_state(option_library: OptionLibrary, current_step: int = None) -> dict:
    """Build a minimal env_state for testing options.
    
    Args:
        option_library: The option library.
        current_step: Deprecated; included for backward compatibility but not used by options.
    
    Returns:
        Dict with required env_state fields (без timestep fields).
    """
    return {
        "patients": [],
        "num_patients": 2,
        "current_amr_levels": {},
        "option_library": option_library,
    }


def test_block_option_decide_returns_single_action():
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=2,
    )
    option_library = OptionLibrary(env=env)

    module_path = _get_bundled_option_loader_path(loader_type="block")
    module = _load_module(module_name="block_option_loader", module_path=module_path)

    option = module.BlockOption(name="A_5", antibiotic="A", duration=5)
    env_state = _build_env_state(option_library=option_library, current_step=0)

    actions = option.decide(env_state=env_state)

    # decide() returns antibiotic name strings, not indices
    assert actions.tolist() == ["A", "A"]


def test_block_option_supports_no_rx_alias():
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=2,
    )
    option_library = OptionLibrary(env=env)

    module_path = _get_bundled_option_loader_path(loader_type="block")
    module = _load_module(module_name="block_option_loader", module_path=module_path)

    option = module.BlockOption(name="no_treatment_5", antibiotic="no_treatment", duration=5)
    env_state = _build_env_state(option_library=option_library, current_step=0)

    actions = option.decide(env_state=env_state)

    # decide() returns antibiotic name strings, not indices
    assert actions.tolist() == ["no_treatment", "no_treatment"]


def test_block_loader_validates_required_keys():
    module_path = _get_bundled_option_loader_path(loader_type="block")
    module = _load_module(module_name="block_option_loader", module_path=module_path)

    with pytest.raises(ValueError):
        module.load_block_option(config={})

    with pytest.raises(ValueError):
        module.load_block_option(config={"option_name": "bad", "antibiotic": "A"})

    with pytest.raises(ValueError):
        module.load_block_option(config={"option_name": "bad", "duration": 5})


def test_alternation_option_sequences_actions():
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=2,
    )
    option_library = OptionLibrary(env=env)

    module_path = _get_bundled_option_loader_path(loader_type="alternation")
    module = _load_module(module_name="alternation_option_loader", module_path=module_path)

    option = module.AlternationOption(
        name="ALT_A_no_treatment_B",
        sequence=["A", "no_treatment", "B"],
    )

    # decide() returns antibiotic name strings, not indices
    env_state = _build_env_state(option_library=option_library, current_step=0)
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == ["A", "A"]

    env_state = _build_env_state(option_library=option_library, current_step=1)
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == ["no_treatment", "no_treatment"]

    env_state = _build_env_state(option_library=option_library, current_step=2)
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == ["B", "B"]


def test_alternation_resets_via_explicit_reset_call():
    """Test that AlternationOption resets via explicit reset() call, not step gaps.
    
    With timestep awareness removed, alternation options cycle through their sequence
    on each decide() call. They only reset when reset() is explicitly called.
    """
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=2,
    )
    option_library = OptionLibrary(env=env)

    module_path = _get_bundled_option_loader_path(loader_type="alternation")
    module = _load_module(module_name="alternation_option_loader", module_path=module_path)

    option = module.AlternationOption(
        name="ALT_A_B",
        sequence=["A", "B"],
    )

    env_state = _build_env_state(option_library=option_library)
    
    # First call: get A
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == ["A", "A"]
    
    # Second call: get B (continues sequence)
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == ["B", "B"]
    
    # Third call: get A (wraps around)
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == ["A", "A"]
    
    # After explicit reset: back to beginning
    option.reset()
    actions = option.decide(env_state=env_state)
    assert actions.tolist() == ["A", "A"]


def test_alternation_loader_validates_sequence():
    module_path = _get_bundled_option_loader_path(loader_type="alternation")
    module = _load_module(module_name="alternation_option_loader", module_path=module_path)

    with pytest.raises(ValueError):
        module.load_alternation_option(config={})

    with pytest.raises(ValueError):
        module.load_alternation_option(config={"option_name": "bad", "sequence": []})

    with pytest.raises(ValueError):
        module.load_alternation_option(config={"option_name": "bad", "sequence": [1, 2]})


def test_default_deterministic_library_loads():
    env = create_mock_environment(
        antibiotic_names=["A", "B"],
        num_patients_per_time_step=1,
    )
    # Path from test: external/abx_amr_simulator/tests/unit/hrl/test_option_type_loaders.py
    # parents[3] = external/abx_amr_simulator
    package_root = Path(__file__).resolve().parents[3]  # external/abx_amr_simulator
    library_path = (
        package_root
        / "src"
        / "abx_amr_simulator"
        / "options"
        / "defaults"
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
