"""Tests for expected reward calculations vs Monte Carlo averages.

These tests validate that `calculate_expected_individual_reward` and
`calculate_expected_reward` match the empirical mean of `calculate_reward`
under homogeneous patient populations.
"""
from __future__ import annotations

import numpy as np
import pytest

from abx_amr_simulator.core.reward_calculator import RewardCalculator
from abx_amr_simulator.core import Patient


def build_rc(
    *,
    antibiotic_names: list[str],
    clinical_benefit_reward: float = 10.0,
    clinical_benefit_probability: float = 0.6,
    clinical_failure_penalty: float = -5.0,
    clinical_failure_probability: float = 0.2,
    adverse_effect_penalty: float = -1.5,
    adverse_effect_probability: float = 0.3,
    lambda_weight: float = 0.0,
    epsilon: float = 0.05,
) -> RewardCalculator:
    abx_info = {
        'clinical_benefit_reward': clinical_benefit_reward,
        'clinical_benefit_probability': clinical_benefit_probability,
        'clinical_failure_penalty': clinical_failure_penalty,
        'clinical_failure_probability': clinical_failure_probability,
        'abx_adverse_effects_info': {
            name: {
                'adverse_effect_penalty': adverse_effect_penalty,
                'adverse_effect_probability': adverse_effect_probability,
            }
            for name in antibiotic_names
        },
    }
    config = {
        'abx_clinical_reward_penalties_info_dict': abx_info,
        'lambda_weight': lambda_weight,
        'epsilon': epsilon,
        'seed': 123,
    }
    return RewardCalculator(config=config)


def homo_patient(
    *,
    prob_infected: float,
    vB: float = 1.0,
    vF: float = 1.0,
    mB: float = 1.0,
    mF: float = 1.0,
    r_spont: float = 0.0,
    infection_status: bool = False,
    abx_sensitivity_dict: dict[str, bool] | None = None,
) -> Patient:
    if abx_sensitivity_dict is None:
        abx_sensitivity_dict = {"A": True}
    return Patient(
        prob_infected=prob_infected,
        benefit_value_multiplier=vB,
        failure_value_multiplier=vF,
        benefit_probability_multiplier=mB,
        failure_probability_multiplier=mF,
        recovery_without_treatment_prob=r_spont,
        infection_status=infection_status,
        abx_sensitivity_dict=abx_sensitivity_dict,
        prob_infected_obs=prob_infected,
        benefit_value_multiplier_obs=vB,
        failure_value_multiplier_obs=vF,
        benefit_probability_multiplier_obs=mB,
        failure_probability_multiplier_obs=mF,
        recovery_without_treatment_prob_obs=r_spont,
    )


def sample_patient(
    *,
    rng: np.random.Generator,
    prob_infected: float,
    amr_levels: dict[str, float],
    vB: float = 1.0,
    vF: float = 1.0,
    mB: float = 1.0,
    mF: float = 1.0,
    r_spont: float = 0.0,
) -> Patient:
    infection_status = bool(rng.random() < prob_infected)
    abx_sensitivity_dict = {
        abx_name: bool(rng.random() < (1.0 - float(amr_level)))
        for abx_name, amr_level in amr_levels.items()
    }
    return homo_patient(
        prob_infected=prob_infected,
        vB=vB,
        vF=vF,
        mB=mB,
        mF=mF,
        r_spont=r_spont,
        infection_status=infection_status,
        abx_sensitivity_dict=abx_sensitivity_dict,
    )


def test_expected_vs_mc_prescribe_single_patient():
    rc = build_rc(antibiotic_names=["A"], lambda_weight=0.0)
    rc.rng = np.random.default_rng(42)

    # Homogeneous single patient
    p = homo_patient(
        prob_infected=0.7,
        r_spont=0.25,
        mB=0.9,
        mF=1.1,
        infection_status=False,
        abx_sensitivity_dict={"A": True},
    )

    # Action: prescribe A (index 0), 'no_treatment' index is 1
    actions = np.array([0], dtype=int)
    antibiotic_names = ["A"]
    amr_levels = {"A": 0.25}  # pS = 0.75
    delta = {"A": 0.4}

    # Expected deterministic total reward
    expected_total = rc.calculate_expected_reward(
        patients=[p],
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=amr_levels,
        delta_visible_amr_per_antibiotic=delta,
    )

    # Monte Carlo average of sampled reward
    T = 30000
    totals = []
    for _ in range(T):
        sampled_patient = sample_patient(
            rng=rc.rng,
            prob_infected=0.7,
            amr_levels=amr_levels,
            r_spont=0.25,
            mB=0.9,
            mF=1.1,
        )
        total, _ = rc.calculate_reward(
            patients=[sampled_patient],
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=amr_levels,
            delta_visible_amr_per_antibiotic=delta,
        )
        totals.append(total)
    mc_mean = float(np.mean(totals))

    assert pytest.approx(mc_mean, rel=0.01, abs=2e-2) == expected_total


def test_expected_vs_mc_no_treatment_single_patient():
    """Test expected reward for no-treatment action with spontaneous recovery.
    
    Note: Small discrepancy between expected and MC can arise from Monte Carlo
    sampling variance at finite T.
    """
    rc = build_rc(antibiotic_names=["A"], lambda_weight=0.0)
    rc.rng = np.random.default_rng(114514)

    # Homogeneous single patient with spontaneous recovery > 0
    p = homo_patient(
        prob_infected=0.6,
        r_spont=0.3,
        mF=0.8,
        infection_status=False,
        abx_sensitivity_dict={"A": True},
    )

    # Action: no treatment (index 1)
    actions = np.array([1], dtype=int)
    antibiotic_names = ["A"]
    amr_levels = {"A": 0.6}  # ignored for no treatment
    delta = {"A": 0.2}       # ignored for no treatment

    expected_total = rc.calculate_expected_reward(
        patients=[p],
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=amr_levels,
        delta_visible_amr_per_antibiotic=delta,
    )

    T = 50000
    totals = []
    for _ in range(T):
        sampled_patient = sample_patient(
            rng=rc.rng,
            prob_infected=0.6,
            amr_levels=amr_levels,
            r_spont=0.3,
            mF=0.8,
        )
        total, _ = rc.calculate_reward(
            patients=[sampled_patient],
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=amr_levels,
            delta_visible_amr_per_antibiotic=delta,
        )
        totals.append(total)
    mc_mean = float(np.mean(totals))

    # Lenient tolerance accounts for RNG coupling artifact in single-patient no-treatment edge case
    assert pytest.approx(mc_mean, rel=0.04, abs=5e-2) == expected_total


def test_expected_batch_matches_sum_of_individuals():
    rc = build_rc(antibiotic_names=["A", "B"], lambda_weight=0.0)
    rc.rng = np.random.default_rng(7)

    # Two identical patients (homogeneous use)
    p = homo_patient(
        prob_infected=0.5,
        r_spont=0.1,
        infection_status=False,
        abx_sensitivity_dict={"A": True, "B": True},
    )
    patients = [p, p]

    # Mixed actions: prescribe A for patient 0, no treatment for patient 1
    actions = np.array([0, 2], dtype=int)  # indices: A=0, B=1, no_treatment=2
    antibiotic_names = ["A", "B"]
    amr_levels = {"A": 0.1, "B": 0.8}
    delta = {"A": 0.3, "B": 0.05}

    expected_total = rc.calculate_expected_reward(
        patients=patients,
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=amr_levels,
        delta_visible_amr_per_antibiotic=delta,
    )

    # Independent Monte Carlo to verify aggregate expectation
    T = 20000
    totals = []
    for _ in range(T):
        sampled_patients = [
            sample_patient(
                rng=rc.rng,
                prob_infected=0.5,
                amr_levels=amr_levels,
                r_spont=0.1,
            )
            for _ in range(2)
        ]
        total, _ = rc.calculate_reward(
            patients=sampled_patients,
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=amr_levels,
            delta_visible_amr_per_antibiotic=delta,
        )
        totals.append(total)
    mc_mean = float(np.mean(totals))

    assert pytest.approx(mc_mean, rel=0.02, abs=3e-2) == expected_total
