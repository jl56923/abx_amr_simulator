"""Tests for build_patient_generator_from_spec (Task 13.5).

Covers:
  1. Plain PatientGenerator from inline spec.
  2. PatientGeneratorMixer with inline child specs.
  3. PatientGeneratorMixer with absolute config_file paths.
  4. PatientGeneratorMixer with relative config_file paths + base_dir.
  5. Relative config_file with no base_dir raises ValueError.
  6. Missing config_file raises ValueError.
  7. Missing 'proportion' raises ValueError.
  8. Missing 'visible_patient_attributes' on inline plain spec raises ValueError.
  9. create_patient_generator uses the utility and produces the same result (regression).

All tests use real PatientGenerator/PatientGeneratorMixer instances — no mocks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from abx_amr_simulator.core.patient_generator import PatientGenerator, PatientGeneratorMixer
from abx_amr_simulator.utils import build_patient_generator_from_spec, create_patient_generator
from abx_amr_simulator.utils.factories import resolve_config_path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ATTR = {
    "prob_dist": {"type": "constant", "value": 0.7, "mu": None, "sigma": None},
    "obs_bias_multiplier": 1.0,
    "obs_noise_one_std_dev": 0.0,
    "obs_noise_std_dev_fraction": 0.0,
    "clipping_bounds": [0.0, 1.0],
}

_ATTR_UNBOUNDED = {**_ATTR, "clipping_bounds": [0.0, None]}

_ALL_SIX = {
    "prob_infected": _ATTR,
    "benefit_value_multiplier": _ATTR_UNBOUNDED,
    "failure_value_multiplier": _ATTR_UNBOUNDED,
    "benefit_probability_multiplier": _ATTR_UNBOUNDED,
    "failure_probability_multiplier": _ATTR_UNBOUNDED,
    "recovery_without_treatment_prob": _ATTR,
}

_TWO_ATTRS = ["prob_infected", "recovery_without_treatment_prob"]


def _plain_spec(visible: list | None = None) -> Dict[str, Any]:
    return {**_ALL_SIX, "visible_patient_attributes": visible or list(_ALL_SIX.keys())}


def _high_risk_spec(proportion: float = 0.5) -> Dict[str, Any]:
    return {
        "proportion": proportion,
        **_ALL_SIX,
        "visible_patient_attributes": list(_ALL_SIX.keys()),
    }


def _low_risk_spec(proportion: float = 0.5) -> Dict[str, Any]:
    return {
        "proportion": proportion,
        **_ALL_SIX,
        "visible_patient_attributes": _TWO_ATTRS,
    }


def _inline_mixer_spec() -> Dict[str, Any]:
    return {
        "type": "mixer",
        "generators": [_high_risk_spec(0.6), _low_risk_spec(0.4)],
    }


_RNG = np.random.default_rng(0)
_AMR = {"A": 0.1, "B": 0.05}


# ---------------------------------------------------------------------------
# 1. Plain PatientGenerator from inline spec
# ---------------------------------------------------------------------------

class TestPlainSpec:

    def test_returns_patient_generator(self):
        pg = build_patient_generator_from_spec(_plain_spec())
        assert isinstance(pg, PatientGenerator)
        assert not isinstance(pg, PatientGeneratorMixer)

    def test_visible_attrs_preserved(self):
        pg = build_patient_generator_from_spec(_plain_spec(visible=_TWO_ATTRS))
        assert pg.visible_patient_attributes == _TWO_ATTRS

    def test_sample_works(self):
        pg = build_patient_generator_from_spec(_plain_spec())
        patients = pg.sample(n_patients=3, true_amr_levels=_AMR, rng=_RNG)
        assert len(patients) == 3

    def test_missing_visible_attrs_raises(self):
        spec = dict(_ALL_SIX)  # no visible_patient_attributes
        with pytest.raises(ValueError, match="visible_patient_attributes"):
            build_patient_generator_from_spec(spec)


# ---------------------------------------------------------------------------
# 2. PatientGeneratorMixer with inline child specs
# ---------------------------------------------------------------------------

class TestInlineMixerSpec:

    def test_returns_patient_generator_mixer(self):
        mixer = build_patient_generator_from_spec(_inline_mixer_spec())
        assert isinstance(mixer, PatientGeneratorMixer)

    def test_heterogeneous_visibility_detected(self):
        mixer = build_patient_generator_from_spec(_inline_mixer_spec())
        assert mixer._uses_heterogeneous_visibility is True

    def test_visible_attrs_is_union(self):
        mixer = build_patient_generator_from_spec(_inline_mixer_spec())
        for attr in _ALL_SIX:
            assert attr in mixer.visible_patient_attributes

    def test_proportions_respected(self):
        spec = {
            "type": "mixer",
            "generators": [_high_risk_spec(0.7), _low_risk_spec(0.3)],
        }
        mixer = build_patient_generator_from_spec(spec)
        assert np.isclose(mixer.proportions[0], 0.7)
        assert np.isclose(mixer.proportions[1], 0.3)

    def test_seed_forwarded_to_children(self):
        mixer = build_patient_generator_from_spec(_inline_mixer_spec(), seed=7)
        for gen in mixer.generators:
            assert gen.seed == 7

    def test_sample_works(self):
        mixer = build_patient_generator_from_spec(_inline_mixer_spec())
        patients = mixer.sample(n_patients=10, true_amr_levels=_AMR, rng=_RNG)
        assert len(patients) == 10

    def test_empty_generators_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_patient_generator_from_spec({"type": "mixer", "generators": []})

    def test_missing_proportion_raises(self):
        spec = {
            "type": "mixer",
            "generators": [
                {**_ALL_SIX, "visible_patient_attributes": _TWO_ATTRS},
            ],
        }
        with pytest.raises(ValueError, match="proportion"):
            build_patient_generator_from_spec(spec)

    def test_inline_child_missing_visible_attrs_raises(self):
        spec = {
            "type": "mixer",
            "generators": [
                {"proportion": 1.0, **_ALL_SIX},  # no visible_patient_attributes
            ],
        }
        with pytest.raises(ValueError, match="visible_patient_attributes"):
            build_patient_generator_from_spec(spec)


# ---------------------------------------------------------------------------
# 3. PatientGeneratorMixer with absolute config_file paths
# ---------------------------------------------------------------------------

class TestAbsoluteConfigFilePaths:

    def test_absolute_config_file_resolves_without_base_dir(self, tmp_path):
        child_yaml = tmp_path / "gen.yaml"
        child_yaml.write_text(
            yaml.safe_dump({**_ALL_SIX, "visible_patient_attributes": list(_ALL_SIX.keys())}),
            encoding="utf-8",
        )
        spec = {
            "type": "mixer",
            "generators": [
                {"config_file": str(child_yaml), "proportion": 1.0},
            ],
        }
        mixer = build_patient_generator_from_spec(spec, base_dir=None)
        assert isinstance(mixer, PatientGeneratorMixer)

    def test_absolute_config_file_with_base_dir_also_works(self, tmp_path):
        child_yaml = tmp_path / "gen.yaml"
        child_yaml.write_text(
            yaml.safe_dump({**_ALL_SIX, "visible_patient_attributes": list(_ALL_SIX.keys())}),
            encoding="utf-8",
        )
        spec = {
            "type": "mixer",
            "generators": [
                {"config_file": str(child_yaml), "proportion": 1.0},
            ],
        }
        # base_dir is irrelevant when config_file is absolute
        mixer = build_patient_generator_from_spec(spec, base_dir=tmp_path / "unrelated")
        assert isinstance(mixer, PatientGeneratorMixer)


# ---------------------------------------------------------------------------
# 4. PatientGeneratorMixer with relative config_file paths + base_dir
# ---------------------------------------------------------------------------

class TestRelativeConfigFilePaths:

    def test_relative_path_resolves_against_base_dir(self, tmp_path):
        child_yaml = tmp_path / "low_risk.yaml"
        child_yaml.write_text(
            yaml.safe_dump({**_ALL_SIX, "visible_patient_attributes": _TWO_ATTRS}),
            encoding="utf-8",
        )
        spec = {
            "type": "mixer",
            "generators": [
                {"config_file": "low_risk.yaml", "proportion": 1.0},
            ],
        }
        mixer = build_patient_generator_from_spec(spec, base_dir=tmp_path)
        assert isinstance(mixer, PatientGeneratorMixer)
        assert mixer.visible_patient_attributes == _TWO_ATTRS

    def test_two_children_different_visibility(self, tmp_path):
        high = tmp_path / "high.yaml"
        high.write_text(
            yaml.safe_dump({**_ALL_SIX, "visible_patient_attributes": list(_ALL_SIX.keys())}),
            encoding="utf-8",
        )
        low = tmp_path / "low.yaml"
        low.write_text(
            yaml.safe_dump({**_ALL_SIX, "visible_patient_attributes": _TWO_ATTRS}),
            encoding="utf-8",
        )
        spec = {
            "type": "mixer",
            "generators": [
                {"config_file": "high.yaml", "proportion": 0.5},
                {"config_file": "low.yaml", "proportion": 0.5},
            ],
        }
        mixer = build_patient_generator_from_spec(spec, base_dir=tmp_path)
        assert mixer._uses_heterogeneous_visibility is True
        for attr in _ALL_SIX:
            assert attr in mixer.visible_patient_attributes


# ---------------------------------------------------------------------------
# 5. Relative config_file with no base_dir raises ValueError
# ---------------------------------------------------------------------------

class TestRelativePathNoBaseDirRaises:

    def test_raises_value_error(self):
        spec = {
            "type": "mixer",
            "generators": [
                {"config_file": "relative/path.yaml", "proportion": 1.0},
            ],
        }
        with pytest.raises(ValueError, match="base_dir"):
            build_patient_generator_from_spec(spec, base_dir=None)


# ---------------------------------------------------------------------------
# 6. Missing config_file raises ValueError
# ---------------------------------------------------------------------------

class TestMissingConfigFileRaises:

    def test_nonexistent_absolute_path_raises(self, tmp_path):
        spec = {
            "type": "mixer",
            "generators": [
                {"config_file": str(tmp_path / "does_not_exist.yaml"), "proportion": 1.0},
            ],
        }
        with pytest.raises(ValueError, match="config_file not found"):
            build_patient_generator_from_spec(spec)

    def test_nonexistent_relative_path_raises(self, tmp_path):
        spec = {
            "type": "mixer",
            "generators": [
                {"config_file": "does_not_exist.yaml", "proportion": 1.0},
            ],
        }
        with pytest.raises(ValueError, match="config_file not found"):
            build_patient_generator_from_spec(spec, base_dir=tmp_path)


# ---------------------------------------------------------------------------
# 9. Regression: create_patient_generator still works via the utility
# ---------------------------------------------------------------------------

class TestCreatePatientGeneratorRegression:

    def test_plain_generator_via_create_patient_generator(self):
        config = {
            "patient_generator": _plain_spec(),
        }
        pg = create_patient_generator(config)
        assert isinstance(pg, PatientGenerator)
        assert not isinstance(pg, PatientGeneratorMixer)

    def test_mixer_via_create_patient_generator_inline_children(self, tmp_path):
        # Inline children via create_patient_generator — exercises the
        # _config_dir_hint path through build_patient_generator_from_spec.
        spec = {
            "patient_generator": {
                "type": "mixer",
                "generators": [_high_risk_spec(0.5), _low_risk_spec(0.5)],
            },
            "_umbrella_config_dir": str(tmp_path),
        }
        mixer = create_patient_generator(spec)
        assert isinstance(mixer, PatientGeneratorMixer)
        assert mixer._uses_heterogeneous_visibility is True

    def test_mixer_via_create_patient_generator_file_children(self, tmp_path):
        child_yaml = tmp_path / "child.yaml"
        child_yaml.write_text(
            yaml.safe_dump({**_ALL_SIX, "visible_patient_attributes": list(_ALL_SIX.keys())}),
            encoding="utf-8",
        )
        spec = {
            "patient_generator": {
                "type": "mixer",
                "generators": [
                    {"config_file": "child.yaml", "proportion": 1.0},
                ],
            },
            "_umbrella_config_dir": str(tmp_path),
        }
        mixer = create_patient_generator(spec)
        assert isinstance(mixer, PatientGeneratorMixer)


# ---------------------------------------------------------------------------
# resolve_config_path
# ---------------------------------------------------------------------------

class TestResolveConfigPath:
    """Tests for the resolve_config_path utility (Task 5 / $CONFIG_BASE_FOLDER support)."""

    def test_absolute_path_returned_as_is(self, tmp_path):
        target = tmp_path / "some" / "file.yaml"
        result = resolve_config_path(str(target), base_dir=None)
        assert result == target

    def test_relative_path_resolved_against_base_dir(self, tmp_path):
        result = resolve_config_path("subdir/file.yaml", base_dir=tmp_path)
        assert result == (tmp_path / "subdir" / "file.yaml").resolve()

    def test_relative_path_without_base_dir_raises(self):
        with pytest.raises(ValueError, match="base_dir"):
            resolve_config_path("relative/path.yaml", base_dir=None)

    def test_config_base_folder_prefix_expands_from_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ABX_AMR_CONFIG_BASE_FOLDER", str(tmp_path))
        result = resolve_config_path("$CONFIG_BASE_FOLDER/configs/pg/foo.yaml")
        assert result == tmp_path / "configs" / "pg" / "foo.yaml"

    def test_config_base_folder_prefix_without_env_var_raises(self, monkeypatch):
        monkeypatch.delenv("ABX_AMR_CONFIG_BASE_FOLDER", raising=False)
        with pytest.raises(RuntimeError, match="ABX_AMR_CONFIG_BASE_FOLDER"):
            resolve_config_path("$CONFIG_BASE_FOLDER/configs/pg/foo.yaml")

    def test_config_base_folder_prefix_ignores_base_dir(self, tmp_path, monkeypatch):
        """$CONFIG_BASE_FOLDER/ prefix is absolute; base_dir is ignored."""
        config_base = tmp_path / "workspace" / "experiments"
        monkeypatch.setenv("ABX_AMR_CONFIG_BASE_FOLDER", str(config_base))
        other_dir = tmp_path / "unrelated"
        result = resolve_config_path("$CONFIG_BASE_FOLDER/configs/pg/foo.yaml", base_dir=other_dir)
        assert result == config_base / "configs" / "pg" / "foo.yaml"


class TestBuildPatientGeneratorFromSpecWithConfigBaseFolder:
    """Tests that $CONFIG_BASE_FOLDER/ prefixed config_file paths work in mixer specs."""

    def test_config_base_folder_path_resolves_correctly(self, tmp_path, monkeypatch):
        """Mixer spec with $CONFIG_BASE_FOLDER/... config_file resolves when env var is set."""
        child_yaml = tmp_path / "child.yaml"
        child_yaml.write_text(
            yaml.safe_dump({**_ALL_SIX, "visible_patient_attributes": list(_ALL_SIX.keys())}),
            encoding="utf-8",
        )
        monkeypatch.setenv("ABX_AMR_CONFIG_BASE_FOLDER", str(tmp_path))

        spec = {
            "type": "mixer",
            "generators": [
                {"config_file": "$CONFIG_BASE_FOLDER/child.yaml", "proportion": 1.0},
            ],
        }
        mixer = build_patient_generator_from_spec(spec)
        assert isinstance(mixer, PatientGeneratorMixer)

    def test_config_base_folder_path_without_env_raises_runtime_error(self, monkeypatch):
        """Mixer spec with $CONFIG_BASE_FOLDER/... raises RuntimeError if env var unset."""
        monkeypatch.delenv("ABX_AMR_CONFIG_BASE_FOLDER", raising=False)
        spec = {
            "type": "mixer",
            "generators": [
                {"config_file": "$CONFIG_BASE_FOLDER/configs/pg/foo.yaml", "proportion": 1.0},
            ],
        }
        with pytest.raises(RuntimeError, match="ABX_AMR_CONFIG_BASE_FOLDER"):
            build_patient_generator_from_spec(spec)
