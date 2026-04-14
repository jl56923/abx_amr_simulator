from __future__ import annotations

import importlib
import sys
import types
from typing import Dict, List, Optional

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


def _make_outcomes_breakdown(
    *,
    antibiotic_names: List[str],
    not_infected_no_treatment: int = 0,
    not_infected_treated: int = 0,
    infected_no_treatment: int = 0,
    sensitive_per_abx: Optional[Dict[str, int]] = None,
    resistant_per_abx: Optional[Dict[str, int]] = None,
) -> Dict:
    sensitive_per_abx = sensitive_per_abx or {}
    resistant_per_abx = resistant_per_abx or {}
    return {
        "not_infected_no_treatment": not_infected_no_treatment,
        "not_infected_treated": not_infected_treated,
        "infected_no_treatment": infected_no_treatment,
        "infected_treated": {
            abx: {
                "sensitive_infection_treated": sensitive_per_abx.get(abx, 0),
                "resistant_infection_treated": resistant_per_abx.get(abx, 0),
            }
            for abx in antibiotic_names
        },
    }


def _reward_info_fields(
    *,
    antibiotic_names: List[str],
    total_reward: float = 0.5,
    overall_individual: float = 1.5,
    normalized_individual: float = 0.75,
    overall_community: float = -0.2,
    normalized_community: float = -0.1,
    count_clinical_benefits: int = 1,
    count_clinical_failures: int = 0,
    count_adverse_events: int = 0,
    outcomes_breakdown: Optional[Dict] = None,
) -> Dict:
    if outcomes_breakdown is None:
        outcomes_breakdown = _make_outcomes_breakdown(
            antibiotic_names=antibiotic_names
        )
    return {
        "total_reward": total_reward,
        "overall_individual_reward_component": overall_individual,
        "normalized_individual_reward": normalized_individual,
        "overall_community_reward_component": overall_community,
        "normalized_community_reward": normalized_community,
        "count_clinical_benefits": count_clinical_benefits,
        "count_clinical_failures": count_clinical_failures,
        "count_adverse_events": count_adverse_events,
        "outcomes_breakdown": outcomes_breakdown,
    }


def test_add_episode_arrays_logs_amr_and_outcome_series() -> None:
    module = _load_module()
    antibiotic_names = ["A", "B"]

    outcomes = _make_outcomes_breakdown(
        antibiotic_names=antibiotic_names,
        not_infected_no_treatment=1,
        not_infected_treated=0,
        infected_no_treatment=0,
        sensitive_per_abx={"A": 1, "B": 0},
        resistant_per_abx={"A": 0, "B": 0},
    )
    info = {
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
        **_reward_info_fields(
            antibiotic_names=antibiotic_names,
            total_reward=0.75,
            overall_individual=1.5,
            normalized_individual=0.75,
            overall_community=-0.3,
            normalized_community=-0.15,
            count_clinical_benefits=1,
            count_clinical_failures=0,
            count_adverse_events=0,
            outcomes_breakdown=outcomes,
        ),
    }
    macro_steps = [
        {
            "primitive_infos": [info],
            "primitive_actions": [np.array([0, 1], dtype=int)],
        }
    ]

    save_dict: Dict = {}
    module._add_episode_arrays(
        save_dict=save_dict,
        ep_prefix="episode_0",
        macro_steps=macro_steps,
        antibiotic_names=antibiotic_names,
    )

    assert "episode_0/primitive_actual_amr_levels" in save_dict
    assert "episode_0/primitive_visible_amr_levels" in save_dict

    actual_amr = save_dict["episode_0/primitive_actual_amr_levels"]
    visible_amr = save_dict["episode_0/primitive_visible_amr_levels"]
    assert actual_amr.shape == (1, 1, 2)
    assert visible_amr.shape == (1, 1, 2)
    np.testing.assert_allclose(actual_amr[0, 0, :], np.array([0.2, 0.4]))
    np.testing.assert_allclose(visible_amr[0, 0, :], np.array([0.25, 0.45]))

    # Scalar reward-calculator fields are persisted and non-zero where expected.
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_total_reward"], np.array([[0.75]])
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_overall_individual_reward_component"],
        np.array([[1.5]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_normalized_individual_reward"],
        np.array([[0.75]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_overall_community_reward_component"],
        np.array([[-0.3]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_normalized_community_reward"],
        np.array([[-0.15]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_count_clinical_benefits"],
        np.array([[1.0]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_count_clinical_failures"],
        np.array([[0.0]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_count_adverse_events"],
        np.array([[0.0]]),
    )

    # Outcome breakdown fields.
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_not_infected_no_treatment"],
        np.array([[1.0]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_not_infected_treated"],
        np.array([[0.0]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_infected_no_treatment"],
        np.array([[0.0]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_sensitive_infection_treated/A"],
        np.array([[1.0]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_sensitive_infection_treated/B"],
        np.array([[0.0]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_resistant_infection_treated/A"],
        np.array([[0.0]]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_resistant_infection_treated/B"],
        np.array([[0.0]]),
    )


def test_add_episode_arrays_fails_when_actual_amr_missing() -> None:
    module = _load_module()
    antibiotic_names = ["A"]

    info = {
        "patient_full_data": {
            "true": {"prob_infected": [0.7]},
            "observed": {"prob_infected": [0.68]},
        },
        "individual_rewards": [1.0],
        "patients_actually_infected": [1.0],
        "visible_amr_levels": {"A": 0.25},
        **_reward_info_fields(antibiotic_names=antibiotic_names),
    }
    macro_steps = [
        {
            "primitive_infos": [info],
            "primitive_actions": [np.array([0], dtype=int)],
        }
    ]

    with pytest.raises(ValueError, match="actual AMR"):
        module._add_episode_arrays(
            save_dict={},
            ep_prefix="episode_0",
            macro_steps=macro_steps,
            antibiotic_names=antibiotic_names,
        )


def test_add_episode_arrays_accepts_amr_alias_keys() -> None:
    module = _load_module()
    antibiotic_names = ["A"]

    info = {
        "patient_full_data": {
            "true": {"prob_infected": [0.9]},
            "observed": {"prob_infected": [0.85]},
        },
        "individual_rewards": [0.1],
        "patients_actually_infected": [1.0],
        "true_amr_levels": {"A": 0.33},
        "observed_amr_levels": {"A": 0.31},
        **_reward_info_fields(antibiotic_names=antibiotic_names),
    }
    macro_steps = [
        {
            "primitive_infos": [info],
            "primitive_actions": [np.array([0], dtype=int)],
        }
    ]

    save_dict: Dict = {}
    module._add_episode_arrays(
        save_dict=save_dict,
        ep_prefix="episode_0",
        macro_steps=macro_steps,
        antibiotic_names=antibiotic_names,
    )

    np.testing.assert_allclose(
        save_dict["episode_0/primitive_actual_amr_levels"][0, 0, :],
        np.array([0.33]),
    )
    np.testing.assert_allclose(
        save_dict["episode_0/primitive_visible_amr_levels"][0, 0, :],
        np.array([0.31]),
    )


def test_add_episode_arrays_fails_when_reward_calculator_fields_missing() -> None:
    module = _load_module()
    antibiotic_names = ["A"]

    # Reward-calculator scalar fields (e.g. count_clinical_benefits) are absent.
    info = {
        "patient_full_data": {
            "true": {"prob_infected": [0.9]},
            "observed": {"prob_infected": [0.85]},
        },
        "individual_rewards": [0.1],
        "patients_actually_infected": [1.0],
        "actual_amr_levels": {"A": 0.33},
        "visible_amr_levels": {"A": 0.31},
    }
    macro_steps = [
        {
            "primitive_infos": [info],
            "primitive_actions": [np.array([0], dtype=int)],
        }
    ]

    with pytest.raises(ValueError, match="reward-calculator fields"):
        module._add_episode_arrays(
            save_dict={},
            ep_prefix="episode_0",
            macro_steps=macro_steps,
            antibiotic_names=antibiotic_names,
        )


def test_add_episode_arrays_fails_when_outcomes_breakdown_missing() -> None:
    module = _load_module()
    antibiotic_names = ["A"]

    reward_fields = _reward_info_fields(antibiotic_names=antibiotic_names)
    reward_fields.pop("outcomes_breakdown")

    info = {
        "patient_full_data": {
            "true": {"prob_infected": [0.9]},
            "observed": {"prob_infected": [0.85]},
        },
        "individual_rewards": [0.1],
        "patients_actually_infected": [1.0],
        "actual_amr_levels": {"A": 0.33},
        "visible_amr_levels": {"A": 0.31},
        **reward_fields,
    }
    macro_steps = [
        {
            "primitive_infos": [info],
            "primitive_actions": [np.array([0], dtype=int)],
        }
    ]

    with pytest.raises(ValueError, match="outcomes_breakdown"):
        module._add_episode_arrays(
            save_dict={},
            ep_prefix="episode_0",
            macro_steps=macro_steps,
            antibiotic_names=antibiotic_names,
        )
