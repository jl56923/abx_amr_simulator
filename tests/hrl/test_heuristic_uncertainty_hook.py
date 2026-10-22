"""Tests for HeuristicWorker's ``_adjust_expected_rewards_for_uncertainty`` extension point.

The hook exists so a subclass can make the prescribing decision depend on uncertainty more finely
than the hard ``uncertainty_threshold`` gate (the workspace's uncertainty-modulated worker uses it).
These tests pin the three properties subclasses rely on:

  1. the default is the identity, so the base worker's decisions are unchanged;
  2. the hook sits ON the decision path -- what it returns is what selection acts on;
  3. it runs after the hard gate, and receives the same uncertainty score the gate used.

Every object here is real: real ``RewardCalculator``, real ``HeuristicWorker`` (and real subclasses of
it). No mocks.
"""

from __future__ import annotations

from typing import Dict, List

from abx_amr_simulator.core.patient_generator import PatientGenerator
from abx_amr_simulator.core.reward_calculator import RewardCalculator
from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker,
)


ANTIBIOTIC_NAMES = ["A"]
AMR_LEVELS = {"A": 0.2}
ALL_ATTRIBUTES = list(PatientGenerator.KNOWN_ATTRIBUTE_TYPES)

# A patient the base worker prescribes for: likely infected, unlikely to recover unaided.
TREATABLE_PATIENT = {"prob_infected": 0.9, "recovery_without_treatment_prob": 0.05}


def _make_reward_calculator() -> RewardCalculator:
    return RewardCalculator(
        config={
            "abx_clinical_reward_penalties_info_dict": {
                "clinical_benefit_reward": 10.0,
                "clinical_benefit_probability": 0.6,
                "clinical_failure_penalty": -5.0,
                "clinical_failure_probability": 0.2,
                "abx_adverse_effects_info": {
                    "A": {"adverse_effect_penalty": -2.0, "adverse_effect_probability": 0.1},
                },
            },
            "lambda_weight": 0.0,
            "seed": 123,
        }
    )


def _make_patient_generator() -> PatientGenerator:
    """A real PatientGenerator; decide() reads its attribute catalogue for absolute uncertainty."""
    def constant(value, clipping_bounds=(0.0, 1.0)):
        return {
            "prob_dist": {"type": "constant", "value": value},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": list(clipping_bounds),
        }

    return PatientGenerator(
        config={
            "prob_infected": constant(0.5),
            "recovery_without_treatment_prob": constant(0.1),
            "benefit_value_multiplier": constant(1.0, (0.0, None)),
            "failure_value_multiplier": constant(1.0, (0.0, None)),
            "benefit_probability_multiplier": constant(1.0, (0.0, None)),
            "failure_probability_multiplier": constant(1.0, (0.0, None)),
            "visible_patient_attributes": ["prob_infected", "recovery_without_treatment_prob"],
        }
    )


def _worker_kwargs(uncertainty_threshold: float = 99.0) -> dict:
    return {
        "name": "HEURISTIC_hook_test",
        "duration": 10,
        "action_thresholds": {"prescribe_A": 0.5, "no_treatment": 0.0},
        "uncertainty_threshold": uncertainty_threshold,
        "default_recovery_without_treatment_prob": 0.1,
        "compare_against_no_treatment": False,
    }


def _select(worker: HeuristicWorker, patient: dict) -> str:
    """Run the real per-patient selection with relative uncertainty over all six attributes."""
    worker.set_observable_attributes(ALL_ATTRIBUTES)
    return worker._select_action_for_patient(
        patient=dict(patient),
        antibiotic_names=list(ANTIBIOTIC_NAMES),
        current_amr_levels=dict(AMR_LEVELS),
        reward_calculator=_make_reward_calculator(),
        use_relative_uncertainty=True,
        total_observable_attrs=len(ALL_ATTRIBUTES),
        option_library=None,
    )


class ZeroingWorker(HeuristicWorker):
    """A real subclass that wipes every reward, so no positive threshold can be met."""

    def _adjust_expected_rewards_for_uncertainty(self, expected_rewards, uncertainty):
        return {action: 0.0 for action in expected_rewards}


class RecordingWorker(HeuristicWorker):
    """A real subclass that leaves rewards alone and records the scores it was called with."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen_uncertainties: List[int] = []

    def _adjust_expected_rewards_for_uncertainty(self, expected_rewards, uncertainty):
        self.seen_uncertainties.append(uncertainty)
        return expected_rewards


def test_default_hook_is_the_identity():
    worker = HeuristicWorker(**_worker_kwargs())
    rewards: Dict[str, float] = {"prescribe_A": 3.0, "no_treatment": -1.0}
    assert worker._adjust_expected_rewards_for_uncertainty(expected_rewards=rewards, uncertainty=4) == rewards


def test_base_worker_still_prescribes_the_treatable_patient():
    """Premise for the next test: without an override, this patient clears the 0.5 threshold."""
    assert _select(HeuristicWorker(**_worker_kwargs()), TREATABLE_PATIENT) == "A"


def test_hook_output_is_what_selection_acts_on():
    """An override that zeroes rewards turns the same prescription into no_treatment."""
    assert _select(ZeroingWorker(**_worker_kwargs()), TREATABLE_PATIENT) == "no_treatment"


def test_hook_receives_the_gates_uncertainty_score():
    """Two of six attributes visible -> the relative score the gate computes is 4."""
    worker = RecordingWorker(**_worker_kwargs())
    _select(worker, TREATABLE_PATIENT)
    assert worker.seen_uncertainties == [len(ALL_ATTRIBUTES) - len(TREATABLE_PATIENT)]


def test_hook_is_not_called_for_a_patient_the_gate_refuses():
    """uncertainty_threshold 0 refuses on any missing attribute, before the hook is reached."""
    worker = RecordingWorker(**_worker_kwargs(uncertainty_threshold=0.0))
    assert _select(worker, TREATABLE_PATIENT) == "no_treatment"
    assert worker.seen_uncertainties == []


def test_hook_is_called_once_per_patient_through_decide():
    """Through the public decide(): one hook call per patient the gate lets through."""

    class SimpleOptionLibrary:
        """Carries the one mapping the worker reads. Real logic, no behaviour stubbed."""

        def __init__(self, antibiotic_names):
            self.abx_name_to_index = {name: index for index, name in enumerate(antibiotic_names)}

    worker = RecordingWorker(**_worker_kwargs())
    worker.set_observable_attributes(ALL_ATTRIBUTES)
    patients = [dict(TREATABLE_PATIENT), dict(TREATABLE_PATIENT, prob_infected=0.1)]
    env_state = {
        "patients": patients,
        "num_patients": len(patients),
        "current_amr_levels": dict(AMR_LEVELS),
        "reward_calculator": _make_reward_calculator(),
        "patient_generator": _make_patient_generator(),
        "use_relative_uncertainty": True,
        "option_library": SimpleOptionLibrary(ANTIBIOTIC_NAMES),
        "current_step": 0,
        "max_steps": 100,
    }
    actions = worker.decide(env_state=env_state)
    assert len(worker.seen_uncertainties) == 2
    assert actions.dtype == object and actions[0] == "A"
