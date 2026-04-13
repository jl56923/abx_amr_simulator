from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

def _load_module():
    module_name = "abx_amr_simulator.analysis.run_granular_eval_best_models_marl"

    if "stable_baselines3" not in sys.modules:
        sb3_module = types.ModuleType("stable_baselines3")

        class _DummyPPO:
            @staticmethod
            def load(path: str):  # pragma: no cover - import shim only
                return path

        sb3_module.PPO = _DummyPPO
        sys.modules["stable_baselines3"] = sb3_module

    if "abx_amr_simulator.hrl.marl_wrapper" not in sys.modules:
        marl_wrapper_module = types.ModuleType("abx_amr_simulator.hrl.marl_wrapper")

        class _DummyMARLOptionsWrapper:
            pass

        marl_wrapper_module.MARLOptionsWrapper = _DummyMARLOptionsWrapper
        sys.modules["abx_amr_simulator.hrl.marl_wrapper"] = marl_wrapper_module

    if "abx_amr_simulator.utils.marl_factories" not in sys.modules:
        factories_module = types.ModuleType("abx_amr_simulator.utils.marl_factories")

        def _unused(*args, **kwargs):  # pragma: no cover - import shim only
            raise RuntimeError("Factory shim should not be called in this unit test")

        factories_module.build_marl_env_from_config = _unused
        factories_module.build_marl_managers_from_config = _unused
        factories_module.build_marl_wrapper_from_config = _unused
        factories_module.load_marl_config = _unused
        sys.modules["abx_amr_simulator.utils.marl_factories"] = factories_module

    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_add_episode_arrays_logs_amr_series() -> None:
    module = _load_module()

    macro_steps = [
        {
            "primitive_infos": [
                {
                    "patient_full_data": {
                        "true": {
                            "prob_infected": [0.7, 0.2],
                            "benefit_value_multiplier": [1.0, 1.1],
                        },
                        "observed": {
                            "prob_infected": [0.68, 0.22],
                            "benefit_value_multiplier": [1.0, 1.1],
                        },
                    },
                    "individual_rewards": [1.0, 0.5],
                    "patients_actually_infected": [1.0, 0.0],
                    "actual_amr_levels": {"A": 0.2, "B": 0.4},
                    "visible_amr_levels": {"A": 0.25, "B": 0.45},
                }
            ],
            "primitive_actions": [np.array([0, 1], dtype=int)],
        }
    ]

    save_dict = {}
    module._add_episode_arrays(
        save_dict=save_dict,
        ep_prefix="episode_0",
        macro_steps=macro_steps,
        antibiotic_names=["A", "B"],
    )

    assert "episode_0/primitive_actual_amr_levels" in save_dict
    assert "episode_0/primitive_visible_amr_levels" in save_dict

    actual_amr = save_dict["episode_0/primitive_actual_amr_levels"]
    visible_amr = save_dict["episode_0/primitive_visible_amr_levels"]

    assert actual_amr.shape == (1, 1, 2)
    assert visible_amr.shape == (1, 1, 2)

    np.testing.assert_allclose(actual_amr[0, 0, :], np.array([0.2, 0.4]))
    np.testing.assert_allclose(visible_amr[0, 0, :], np.array([0.25, 0.45]))


def test_add_episode_arrays_fails_when_actual_amr_missing() -> None:
    module = _load_module()

    macro_steps = [
        {
            "primitive_infos": [
                {
                    "patient_full_data": {
                        "true": {
                            "prob_infected": [0.7],
                        },
                        "observed": {
                            "prob_infected": [0.68],
                        },
                    },
                    "individual_rewards": [1.0],
                    "patients_actually_infected": [1.0],
                    "visible_amr_levels": {"A": 0.25},
                }
            ],
            "primitive_actions": [np.array([0], dtype=int)],
        }
    ]

    with pytest.raises(ValueError, match="actual AMR"):
        module._add_episode_arrays(
            save_dict={},
            ep_prefix="episode_0",
            macro_steps=macro_steps,
            antibiotic_names=["A"],
        )


def test_add_episode_arrays_accepts_amr_alias_keys() -> None:
    module = _load_module()

    macro_steps = [
        {
            "primitive_infos": [
                {
                    "patient_full_data": {
                        "true": {
                            "prob_infected": [0.9],
                        },
                        "observed": {
                            "prob_infected": [0.85],
                        },
                    },
                    "individual_rewards": [0.1],
                    "patients_actually_infected": [1.0],
                    "true_amr_levels": {"A": 0.33},
                    "observed_amr_levels": {"A": 0.31},
                }
            ],
            "primitive_actions": [np.array([0], dtype=int)],
        }
    ]

    save_dict = {}
    module._add_episode_arrays(
        save_dict=save_dict,
        ep_prefix="episode_0",
        macro_steps=macro_steps,
        antibiotic_names=["A"],
    )

    np.testing.assert_allclose(
        save_dict["episode_0/primitive_actual_amr_levels"][0, 0, :],
        np.array([0.33]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_visible_amr_levels"][0, 0, :],
        np.array([0.31]),
    )
