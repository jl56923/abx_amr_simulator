"""Tests for HeuristicWorker's ``compare_against_no_treatment`` flag.

The flag exists because ``HeuristicWorker`` and the ``expected_reward_greedy`` fixed-prescribing
comparator differ STRUCTURALLY, not by degree:

    expected_reward_greedy : prescribe argmax abx iff E[R]_abx > 0
                             ('no_treatment' is a fallback, never a competitor)
    HeuristicWorker        : prescribe argmax abx iff E[R]_abx >= threshold
                             AND E[R]_abx > E[R]_no_treatment

So for a patient likely to recover unaided, the comparator prescribes and the worker does not --
at ANY threshold. An HRL option library therefore could not contain an option as permissive as the
comparator it is measured against, which means "HRL agents show restraint relative to the clinician
proxy" could not be distinguished from "HRL agents were never able to be as permissive."

``compare_against_no_treatment=False`` gives the worker the comparator's locked semantics, closing
that gap. These tests pin the three claims that reasoning rests on:

  1. the default (True) preserves the historical behaviour;
  2. no threshold can recover the comparator's permissiveness while True -- the claim that makes
     the flag necessary rather than merely convenient;
  3. with False, a threshold-0.0 option reproduces the real comparator's decisions.

Every object here is real: real ``RewardCalculator``, real ``PatientGenerator``, real
``HeuristicWorker``, real ``ExpectedRewardGreedyPolicy``. No mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from abx_amr_simulator.core.patient_generator import PatientGenerator
from abx_amr_simulator.core.reward_calculator import RewardCalculator
from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker,
    load_heuristic_option,
)
from abx_amr_simulator.policies.fixed_prescribing_rules import ExpectedRewardGreedyPolicy


ANTIBIOTIC_NAMES = ["A"]
AMR_LEVELS = {"A": 0.5}

# Only the two attributes the decision actually turns on are visible, so the observation layout
# stays small enough to assemble by hand in `_build_observation` below.
VISIBLE_ATTRIBUTES = ["prob_infected", "recovery_without_treatment_prob"]

# The motivating patient, verified against the real RewardCalculator by
# `test_motivating_patient_has_the_reward_ordering_the_flag_is_about` below:
# likely infected, but also likely to recover unaided.
# Both E[R] values are positive and no_treatment is the larger, so the comparator treats this
# patient and the default worker does not -- which is the whole point of the flag.
RECOVERS_UNAIDED_PATIENT = {
    "prob_infected": 0.8,
    "recovery_without_treatment_prob": 0.8,
}


def _make_reward_calculator() -> RewardCalculator:
    """A real RewardCalculator with a single antibiotic.

    The adverse-effect terms are set so that, at AMR 0.5, E[R]_prescribe_A crosses zero partway
    through the sampled population. That is what makes the comparator decline anyone at all --
    without it greedy treats essentially everybody and the agreement test below would be
    checking a constant.
    """
    return RewardCalculator(
        config={
            "abx_clinical_reward_penalties_info_dict": {
                "clinical_benefit_reward": 10.0,
                "clinical_benefit_probability": 0.6,
                "clinical_failure_penalty": -5.0,
                "clinical_failure_probability": 0.2,
                "abx_adverse_effects_info": {
                    "A": {
                        "adverse_effect_penalty": -2.0,
                        "adverse_effect_probability": 0.5,
                    },
                },
            },
            "lambda_weight": 0.0,
            "seed": 123,
        }
    )


def _make_worker(
    compare_against_no_treatment: bool,
    prescribe_a_threshold: float = 0.0,
) -> HeuristicWorker:
    """A real HeuristicWorker whose uncertainty gate never fires.

    `uncertainty_threshold` is set above the number of checked attributes deliberately: the
    uncertainty score is a COUNT of missing attributes, so a high value never refuses. That keeps
    these tests about `compare_against_no_treatment` alone.
    """
    return HeuristicWorker(
        name=f"HEURISTIC_cmp_{compare_against_no_treatment}",
        duration=10,
        action_thresholds={"prescribe_A": prescribe_a_threshold},
        uncertainty_threshold=99,
        default_recovery_without_treatment_prob=0.1,
        compare_against_no_treatment=compare_against_no_treatment,
    )


def _make_patient_generator() -> PatientGenerator:
    """A real PatientGenerator spanning both sides of the decision boundary.

    Both attributes are drawn wide (clipped gaussians -- the generator supports 'constant' and
    'gaussian' only) so the sampled population contains patients both rules treat, patients both
    decline, and patients they disagree about. Observation noise and bias are off, because this
    test is about the decision rule and not about the observation model.
    """
    def attribute(dist, clipping_bounds=(0.0, 1.0)):
        return {
            "prob_dist": dist,
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": list(clipping_bounds),
        }

    # The generator samples every attribute it knows about, so all six must be configured even
    # though only two are visible. The four multipliers are held at their neutral value of 1.0 so
    # they cannot influence which rule prescribes.
    neutral_multiplier = attribute(
        {"type": "constant", "value": 1.0}, clipping_bounds=(0.0, None)
    )

    return PatientGenerator(
        config={
            "prob_infected": attribute(
                {"type": "gaussian", "mu": 0.6, "sigma": 0.25}
            ),
            "recovery_without_treatment_prob": attribute(
                {"type": "gaussian", "mu": 0.5, "sigma": 0.30}
            ),
            "benefit_value_multiplier": dict(neutral_multiplier),
            "failure_value_multiplier": dict(neutral_multiplier),
            "benefit_probability_multiplier": dict(neutral_multiplier),
            "failure_probability_multiplier": dict(neutral_multiplier),
            "visible_patient_attributes": list(VISIBLE_ATTRIBUTES),
        }
    )


def _make_env_state(patients, reward_calculator) -> dict:
    """The env_state dict HeuristicWorker.decide expects."""

    class SimpleOptionLibrary:
        """Carries the one mapping the worker reads. Real logic, no behaviour stubbed."""

        def __init__(self, antibiotic_names):
            self.abx_name_to_index = {
                name: index for index, name in enumerate(antibiotic_names)
            }

    return {
        "patients": patients,
        "num_patients": len(patients),
        "current_amr_levels": dict(AMR_LEVELS),
        "reward_calculator": reward_calculator,
        "patient_generator": _make_patient_generator(),
        "use_relative_uncertainty": True,
        "option_library": SimpleOptionLibrary(ANTIBIOTIC_NAMES),
        "current_step": 0,
        "max_steps": 100,
    }


def _build_observation(patient_dicts) -> np.ndarray:
    """Assemble the flat observation the FP policies parse.

    Layout, per `FixedPrescribingRules._parse_observation_all_patients`:
        [p0_attr0, p0_attr1, ..., p1_attr0, ..., amr_A, amr_B, ...]
    """
    values = []
    for patient in patient_dicts:
        values.extend(patient[attribute] for attribute in VISIBLE_ATTRIBUTES)
    values.extend(AMR_LEVELS[name] for name in ANTIBIOTIC_NAMES)
    return np.array(values, dtype=float)


def _sample_patient_dicts(n_patients: int, seed: int) -> list:
    """Draw a real population and reduce it to the visible-attribute dicts both rules consume."""
    generator = _make_patient_generator()
    patients = generator.sample(
        n_patients=n_patients,
        rng=np.random.default_rng(seed),
        true_amr_levels=dict(AMR_LEVELS),
    )
    observed = generator.observe(patients)
    n_attributes = len(VISIBLE_ATTRIBUTES)
    return [
        dict(zip(VISIBLE_ATTRIBUTES, observed[i * n_attributes:(i + 1) * n_attributes]))
        for i in range(n_patients)
    ]


class TestTheMotivatingCase:
    """The single patient the flag exists for."""

    def test_motivating_patient_has_the_reward_ordering_the_flag_is_about(self):
        """Pin the premise: 0 < E[R]_prescribe_A < E[R]_no_treatment.

        If this ordering ever stops holding, the two tests below would pass vacuously.
        """
        worker = _make_worker(compare_against_no_treatment=True)
        expected_rewards = worker.compute_expected_reward(
            patient=dict(RECOVERS_UNAIDED_PATIENT),
            antibiotic_names=ANTIBIOTIC_NAMES,
            current_amr_levels=dict(AMR_LEVELS),
            reward_calculator=_make_reward_calculator(),
        )
        assert expected_rewards["prescribe_A"] > 0.0
        assert expected_rewards["no_treatment"] > expected_rewards["prescribe_A"]

    def test_default_worker_refuses_the_patient(self):
        """True (the historical default): no_treatment competes, and wins."""
        worker = _make_worker(compare_against_no_treatment=True)
        env_state = _make_env_state(
            patients=[dict(RECOVERS_UNAIDED_PATIENT)],
            reward_calculator=_make_reward_calculator(),
        )
        assert worker.decide(env_state=env_state)[0] == "no_treatment"

    def test_flag_false_prescribes_the_patient(self):
        """False: no_treatment is a fallback only, so a positive-E[R] antibiotic wins."""
        worker = _make_worker(compare_against_no_treatment=False)
        env_state = _make_env_state(
            patients=[dict(RECOVERS_UNAIDED_PATIENT)],
            reward_calculator=_make_reward_calculator(),
        )
        assert worker.decide(env_state=env_state)[0] == "A"

    @pytest.mark.parametrize(
        "threshold", [0.0, -1.0, -100.0, -np.inf]
    )
    def test_no_threshold_recovers_permissiveness_while_flag_is_true(self, threshold):
        """The claim that makes the flag necessary rather than convenient.

        Lowering `prescribe_A`'s threshold cannot help, because the threshold and the
        value-to-beat are two independent gates and only the second one is binding here. This is
        why the EoID option-library critique could not be closed by editing YAML.
        """
        worker = _make_worker(
            compare_against_no_treatment=True, prescribe_a_threshold=threshold
        )
        env_state = _make_env_state(
            patients=[dict(RECOVERS_UNAIDED_PATIENT)],
            reward_calculator=_make_reward_calculator(),
        )
        assert worker.decide(env_state=env_state)[0] == "no_treatment"


class TestAgreementWithTheRealComparator:
    """Population-level behaviour against the real `expected_reward_greedy` policy."""

    def _greedy_actions(self, patient_dicts, reward_calculator) -> np.ndarray:
        policy = ExpectedRewardGreedyPolicy(
            config={},
            reward_calculator=reward_calculator,
            visible_patient_attributes=list(VISIBLE_ATTRIBUTES),
            antibiotic_names=list(ANTIBIOTIC_NAMES),
            num_patients_per_time_step=len(patient_dicts),
        )
        actions, _ = policy.predict(_build_observation(patient_dicts))
        return actions

    def _worker_actions(
        self, patient_dicts, reward_calculator, compare_against_no_treatment
    ) -> np.ndarray:
        worker = _make_worker(compare_against_no_treatment=compare_against_no_treatment)
        return worker.decide(
            env_state=_make_env_state(
                patients=[dict(p) for p in patient_dicts],
                reward_calculator=reward_calculator,
            )
        )

    def test_greedy_equivalent_worker_reproduces_the_comparator_decisions(self):
        """A threshold-0.0 option with the flag off matches greedy patient for patient.

        This is the property EoID's `greedy_equivalent` option tier depends on.
        """
        reward_calculator = _make_reward_calculator()
        patient_dicts = _sample_patient_dicts(n_patients=200, seed=7)

        greedy = self._greedy_actions(patient_dicts, reward_calculator)
        worker = self._worker_actions(
            patient_dicts, reward_calculator, compare_against_no_treatment=False
        )

        assert list(worker) == list(greedy)

    def test_the_population_actually_exercises_both_outcomes(self):
        """Guard against the agreement test passing on a degenerate population."""
        reward_calculator = _make_reward_calculator()
        patient_dicts = _sample_patient_dicts(n_patients=200, seed=7)

        greedy = list(self._greedy_actions(patient_dicts, reward_calculator))
        assert greedy.count("A") > 10
        assert greedy.count("no_treatment") > 10

    def test_default_worker_treats_strictly_fewer_patients_than_the_comparator(self):
        """The option-library critique, stated mechanically.

        The default worker's treated set is a strict subset of greedy's: it never treats someone
        greedy declines, and it declines someone greedy treats. That gap is exactly what an HRL
        agent restricted to default-semantics options could not close.
        """
        reward_calculator = _make_reward_calculator()
        patient_dicts = _sample_patient_dicts(n_patients=200, seed=7)

        greedy = self._greedy_actions(patient_dicts, reward_calculator)
        default_worker = self._worker_actions(
            patient_dicts, reward_calculator, compare_against_no_treatment=True
        )

        greedy_treated = {i for i, a in enumerate(greedy) if a != "no_treatment"}
        worker_treated = {
            i for i, a in enumerate(default_worker) if a != "no_treatment"
        }

        assert worker_treated < greedy_treated


class TestLoader:
    """`load_heuristic_option` must carry the flag through, and stay backward compatible."""

    def _config(self, **extra) -> dict:
        config = {
            "option_name": "HEURISTIC_loader_test",
            "duration": 10,
            "action_thresholds": {"prescribe_A": 0.0},
            "default_recovery_without_treatment_prob": 0.1,
        }
        config.update(extra)
        return config

    def test_defaults_to_true_so_existing_libraries_are_unaffected(self):
        """Non-breaking by construction: LPP and VOI keep today's behaviour until they opt in."""
        option = load_heuristic_option(self._config())
        assert option.compare_against_no_treatment is True

    def test_reads_false_from_config(self):
        option = load_heuristic_option(
            self._config(compare_against_no_treatment=False)
        )
        assert option.compare_against_no_treatment is False

    @pytest.mark.parametrize("bad_value", ["false", 0, None, 1.0])
    def test_rejects_non_bool(self, bad_value):
        """A YAML `compare_against_no_treatment: "false"` must not silently read as truthy."""
        with pytest.raises(ValueError, match="compare_against_no_treatment"):
            load_heuristic_option(
                self._config(compare_against_no_treatment=bad_value)
            )
