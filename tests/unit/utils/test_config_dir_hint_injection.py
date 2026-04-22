"""Tests verifying that factory functions inject _config_dir_hint into plugin configs.

Every component factory (create_reward_calculator, create_patient_generator,
create_amr_dynamics in factories.py; build_patient_generator_from_config and
build_reward_calculator_from_config in marl_factories.py) must forward the
resolved config directory as _config_dir_hint into the component subconfig
before calling the plugin loader.  Without this injection, plugin subclasses
that need to resolve relative file paths have no path context and must fall
back to fragile Path(__file__) workarounds.

Design note on capture mechanism:
    load_plugin_component imports plugin modules via importlib.util.spec_from_file_location,
    creating a fresh module object separate from the one pytest imported.  Class-level
    attributes on types defined in this file therefore cannot be shared across that
    boundary.  Instead, all plugins here write captured values to a file path passed
    via a _capture_file key in their config dict.  Tests read the file after the call.
    "NONE" is written when _config_dir_hint is absent; the actual path string otherwise.

All tests use real plugin fixtures — no mocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from abx_amr_simulator.utils import (
    create_amr_dynamics,
    create_patient_generator,
    create_reward_calculator,
)
from abx_amr_simulator.utils.marl_factories import (
    build_patient_generator_from_config,
    build_reward_calculator_from_config,
)


# ---------------------------------------------------------------------------
# Plugin loader functions — defined here so _THIS_FILE can be the loader_module.
# Each function writes the received _config_dir_hint to _capture_file (if set),
# then returns a real, functional component instance.
# ---------------------------------------------------------------------------

def load_patient_generator_component(config: Dict[str, Any]):
    from pathlib import Path as _Path
    from abx_amr_simulator.core.patient_generator import PatientGenerator

    hint = config.get("_config_dir_hint")
    capture_path = config.get("_capture_file")
    if capture_path is not None:
        _Path(capture_path).write_text(
            "NONE" if hint is None else str(hint), encoding="utf-8"
        )
    return PatientGenerator(config=config)


def load_reward_calculator_component(config: Dict[str, Any]):
    from pathlib import Path as _Path
    from abx_amr_simulator.core.reward_calculator import RewardCalculator

    hint = config.get("_config_dir_hint")
    capture_path = config.get("_capture_file")
    if capture_path is not None:
        _Path(capture_path).write_text(
            "NONE" if hint is None else str(hint), encoding="utf-8"
        )
    return RewardCalculator(config=config)


def load_amr_dynamics_component(config: Dict[str, Any]):
    from pathlib import Path as _Path
    from abx_amr_simulator.core.leaky_balloon import AMR_LeakyBalloon

    hint = config.get("_config_dir_hint")
    capture_path = config.get("_capture_file")
    if capture_path is not None:
        _Path(capture_path).write_text(
            "NONE" if hint is None else str(hint), encoding="utf-8"
        )

    antibiotics_amr_dict = config["antibiotics_AMR_dict"]
    return {
        name: AMR_LeakyBalloon(
            leak=params["leak"],
            flatness_parameter=params["flatness_parameter"],
            permanent_residual_volume=params["permanent_residual_volume"],
            initial_amr_level=params["initial_amr_level"],
        )
        for name, params in antibiotics_amr_dict.items()
    }


# Absolute path to this file — used as loader_module in plugin configs.
_THIS_FILE = str(Path(__file__).resolve())


# ---------------------------------------------------------------------------
# Minimal inline configs
# ---------------------------------------------------------------------------

def _minimal_pg_config() -> Dict[str, Any]:
    attr = {
        "prob_dist": {"type": "constant", "value": 0.7, "mu": None, "sigma": None},
        "obs_bias_multiplier": 1.0,
        "obs_noise_one_std_dev": 0.0,
        "obs_noise_std_dev_fraction": 0.0,
        "clipping_bounds": [0.0, 1.0],
    }
    return {
        "prob_infected": attr,
        "benefit_value_multiplier": {**attr, "clipping_bounds": [0.0, None]},
        "failure_value_multiplier": {**attr, "clipping_bounds": [0.0, None]},
        "benefit_probability_multiplier": {**attr, "clipping_bounds": [0.0, None]},
        "failure_probability_multiplier": {**attr, "clipping_bounds": [0.0, None]},
        "recovery_without_treatment_prob": attr,
        "visible_patient_attributes": ["prob_infected"],
    }


def _minimal_rc_config() -> Dict[str, Any]:
    return {
        "abx_clinical_reward_penalties_info_dict": {
            "clinical_benefit_reward": 10.0,
            "clinical_benefit_probability": 1.0,
            "clinical_failure_penalty": -5.0,
            "clinical_failure_probability": 1.0,
            "abx_adverse_effects_info": {
                "A": {"adverse_effect_penalty": -1.0, "adverse_effect_probability": 0.1}
            },
        },
        "lambda_weight": 0.0,
    }


def _minimal_amr_dict() -> Dict[str, Any]:
    return {
        "A": {
            "leak": 0.05,
            "flatness_parameter": 1.0,
            "permanent_residual_volume": 0.0,
            "initial_amr_level": 0.0,
        }
    }


# ---------------------------------------------------------------------------
# Helpers: build plugin config dicts and write them to YAML files
# ---------------------------------------------------------------------------

def _pg_plugin_dict(capture_file: str) -> Dict[str, Any]:
    return {
        **_minimal_pg_config(),
        "_capture_file": capture_file,
        "plugin": {
            "loader_module": _THIS_FILE,
            "loader_function": "load_patient_generator_component",
        },
    }


def _rc_plugin_dict(capture_file: str) -> Dict[str, Any]:
    return {
        **_minimal_rc_config(),
        "_capture_file": capture_file,
        "plugin": {
            "loader_module": _THIS_FILE,
            "loader_function": "load_reward_calculator_component",
        },
    }


def _amr_plugin_dict(capture_file: str) -> Dict[str, Any]:
    return {
        "antibiotics_AMR_dict": _minimal_amr_dict(),
        "_capture_file": capture_file,
        "plugin": {
            "loader_module": _THIS_FILE,
            "loader_function": "load_amr_dynamics_component",
        },
    }


def _read_captured(capture_file: Path) -> str | None:
    """Read back the value written by a capturing plugin. Returns None if 'NONE'."""
    text = capture_file.read_text(encoding="utf-8").strip()
    return None if text == "NONE" else text


# ---------------------------------------------------------------------------
# Tests: factories.py — create_patient_generator
# ---------------------------------------------------------------------------

class TestCreatePatientGeneratorInjectsConfigDirHint:

    def test_plugin_receives_config_dir_hint(self, tmp_path):
        """_config_dir_hint in plugin config must equal the umbrella config directory."""
        capture_file = tmp_path / "pg_hint.txt"

        config = {
            "patient_generator": _pg_plugin_dict(str(capture_file)),
            "_umbrella_config_dir": str(tmp_path),
        }

        create_patient_generator(config)

        assert _read_captured(capture_file) == str(tmp_path)

    def test_patient_generator_config_dir_takes_precedence_over_umbrella(self, tmp_path):
        """_patient_generator_config_dir beats _umbrella_config_dir when both are set."""
        capture_file = tmp_path / "pg_hint.txt"
        specific_dir = str(tmp_path / "pg_subdir")

        config = {
            "patient_generator": _pg_plugin_dict(str(capture_file)),
            "_patient_generator_config_dir": specific_dir,
            "_umbrella_config_dir": str(tmp_path),
        }

        create_patient_generator(config)

        assert _read_captured(capture_file) == specific_dir

    def test_no_hint_when_neither_dir_key_present(self, tmp_path):
        """_config_dir_hint must not be injected when neither dir key exists in config."""
        capture_file = tmp_path / "pg_hint.txt"

        config = {
            "patient_generator": _pg_plugin_dict(str(capture_file)),
        }

        create_patient_generator(config)

        assert _read_captured(capture_file) is None

    def test_original_config_dict_not_mutated(self, tmp_path):
        """The caller's patient_generator dict must not be modified in place."""
        capture_file = tmp_path / "pg_hint.txt"
        pg_dict = _pg_plugin_dict(str(capture_file))
        original_keys = set(pg_dict.keys())

        config = {
            "patient_generator": pg_dict,
            "_umbrella_config_dir": str(tmp_path),
        }

        create_patient_generator(config)

        assert set(pg_dict.keys()) == original_keys
        assert "_config_dir_hint" not in pg_dict


# ---------------------------------------------------------------------------
# Tests: factories.py — create_reward_calculator
# ---------------------------------------------------------------------------

class TestCreateRewardCalculatorInjectsConfigDirHint:

    def test_plugin_receives_config_dir_hint(self, tmp_path):
        capture_file = tmp_path / "rc_hint.txt"

        config = {
            "reward_calculator": _rc_plugin_dict(str(capture_file)),
            "_umbrella_config_dir": str(tmp_path),
        }

        create_reward_calculator(config)

        assert _read_captured(capture_file) == str(tmp_path)

    def test_reward_calculator_config_dir_takes_precedence(self, tmp_path):
        capture_file = tmp_path / "rc_hint.txt"
        specific_dir = str(tmp_path / "rc_subdir")

        config = {
            "reward_calculator": _rc_plugin_dict(str(capture_file)),
            "_reward_calculator_config_dir": specific_dir,
            "_umbrella_config_dir": str(tmp_path),
        }

        create_reward_calculator(config)

        assert _read_captured(capture_file) == specific_dir

    def test_original_config_dict_not_mutated(self, tmp_path):
        capture_file = tmp_path / "rc_hint.txt"
        rc_dict = _rc_plugin_dict(str(capture_file))
        original_keys = set(rc_dict.keys())

        config = {
            "reward_calculator": rc_dict,
            "_umbrella_config_dir": str(tmp_path),
        }

        create_reward_calculator(config)

        assert set(rc_dict.keys()) == original_keys
        assert "_config_dir_hint" not in rc_dict


# ---------------------------------------------------------------------------
# Tests: factories.py — create_amr_dynamics
# ---------------------------------------------------------------------------

class TestCreateAMRDynamicsInjectsConfigDirHint:

    def test_plugin_receives_config_dir_hint(self, tmp_path):
        capture_file = tmp_path / "amr_hint.txt"

        config = {
            "amr_dynamics": _amr_plugin_dict(str(capture_file)),
            "_umbrella_config_dir": str(tmp_path),
        }

        create_amr_dynamics(config)

        assert _read_captured(capture_file) == str(tmp_path)

    def test_environment_config_dir_takes_precedence(self, tmp_path):
        capture_file = tmp_path / "amr_hint.txt"
        specific_dir = str(tmp_path / "env_subdir")

        config = {
            "amr_dynamics": _amr_plugin_dict(str(capture_file)),
            "_environment_config_dir": specific_dir,
            "_umbrella_config_dir": str(tmp_path),
        }

        create_amr_dynamics(config)

        assert _read_captured(capture_file) == specific_dir

    def test_original_config_dict_not_mutated(self, tmp_path):
        capture_file = tmp_path / "amr_hint.txt"
        amr_dict = _amr_plugin_dict(str(capture_file))
        original_keys = set(amr_dict.keys())

        config = {
            "amr_dynamics": amr_dict,
            "_umbrella_config_dir": str(tmp_path),
        }

        create_amr_dynamics(config)

        assert set(amr_dict.keys()) == original_keys
        assert "_config_dir_hint" not in amr_dict


# ---------------------------------------------------------------------------
# Tests: marl_factories.py — build_patient_generator_from_config
# ---------------------------------------------------------------------------

class TestMarlBuildPatientGeneratorInjectsConfigDirHint:

    def test_file_based_pg_receives_hint_equal_to_its_own_directory(self, tmp_path):
        """When pg_value is a file path, _config_dir_hint must be the file's directory."""
        capture_file = tmp_path / "pg_hint.txt"
        pg_config_path = tmp_path / "pg_plugin.yaml"
        pg_config_path.write_text(
            yaml.safe_dump(_pg_plugin_dict(str(capture_file))), encoding="utf-8"
        )

        build_patient_generator_from_config(
            pg_value=str(pg_config_path),
            config_dir=str(tmp_path),
            seed=None,
        )

        assert _read_captured(capture_file) == str(tmp_path.resolve())

    def test_inline_pg_config_receives_config_dir_hint(self, tmp_path):
        """Inline (dict) pg_value also receives _config_dir_hint set to config_dir."""
        capture_file = tmp_path / "pg_hint.txt"

        build_patient_generator_from_config(
            pg_value=_pg_plugin_dict(str(capture_file)),
            config_dir=str(tmp_path),
            seed=None,
        )

        assert _read_captured(capture_file) == str(tmp_path.resolve())

    def test_original_inline_dict_not_mutated(self, tmp_path):
        """Inline pg_value dict passed by caller must not be modified in place."""
        capture_file = tmp_path / "pg_hint.txt"
        inline_pg = _pg_plugin_dict(str(capture_file))
        original_keys = set(inline_pg.keys())

        build_patient_generator_from_config(
            pg_value=inline_pg,
            config_dir=str(tmp_path),
            seed=None,
        )

        assert set(inline_pg.keys()) == original_keys
        assert "_config_dir_hint" not in inline_pg


# ---------------------------------------------------------------------------
# Tests: marl_factories.py — build_reward_calculator_from_config
# ---------------------------------------------------------------------------

class TestMarlBuildRewardCalculatorInjectsConfigDirHint:

    def test_file_based_rc_receives_hint_equal_to_its_own_directory(self, tmp_path):
        capture_file = tmp_path / "rc_hint.txt"
        rc_config_path = tmp_path / "rc_plugin.yaml"
        rc_config_path.write_text(
            yaml.safe_dump(_rc_plugin_dict(str(capture_file))), encoding="utf-8"
        )

        build_reward_calculator_from_config(
            rc_value=str(rc_config_path),
            config_dir=str(tmp_path),
            seed=None,
        )

        assert _read_captured(capture_file) == str(tmp_path.resolve())

    def test_inline_rc_config_receives_config_dir_hint(self, tmp_path):
        capture_file = tmp_path / "rc_hint.txt"

        build_reward_calculator_from_config(
            rc_value=_rc_plugin_dict(str(capture_file)),
            config_dir=str(tmp_path),
            seed=None,
        )

        assert _read_captured(capture_file) == str(tmp_path.resolve())

    def test_original_inline_dict_not_mutated(self, tmp_path):
        capture_file = tmp_path / "rc_hint.txt"
        inline_rc = _rc_plugin_dict(str(capture_file))
        original_keys = set(inline_rc.keys())

        build_reward_calculator_from_config(
            rc_value=inline_rc,
            config_dir=str(tmp_path),
            seed=None,
        )

        assert set(inline_rc.keys()) == original_keys
        assert "_config_dir_hint" not in inline_rc
