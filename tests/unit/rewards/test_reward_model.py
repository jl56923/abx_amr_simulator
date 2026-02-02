"""Unit tests for RewardCalculator and convenience subclasses."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abx_amr_simulator.core.reward_calculator import (
    BalancedReward,
    CommunityOnlyReward,
    IndividualOnlyReward,
    RewardCalculator,
)
from abx_amr_simulator.core import Patient

def create_test_reward_calculator(
    antibiotic_names=None,
    clinical_benefit_reward=10.0,
    clinical_benefit_probability=1.0,
    adverse_effect_penalty=-2.0,
    adverse_effect_probability=0.0,
    lambda_weight=0.5,
    epsilon=0.05,
    clinical_failure_penalty=-1.0,
    clinical_failure_probability=0.0,
):
    """Helper to create a RewardCalculator for testing.

    Updated to match RewardCalculator API which expects top-level clinical benefit/failure
    parameters and per-antibiotic adverse effects under 'abx_adverse_effects_info'.
    """
    if antibiotic_names is None:
        antibiotic_names = ["A"]

    abx_clinical_reward_penalties_info_dict = {
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
        'abx_clinical_reward_penalties_info_dict': abx_clinical_reward_penalties_info_dict,
        'lambda_weight': lambda_weight,
        'epsilon': epsilon,
    }
    return RewardCalculator(config=config)


def test_calculate_individual_reward_penalizes_amr_and_counts_adverse():
    """Test that individual reward penalizes AMR accumulation and counts adverse events.
    
    Purpose: Verifies that when an antibiotic is prescribed with a high delta_amr value
    and adverse effects are guaranteed to occur (probability=1.0), the reward includes
    both the clinical benefit, the adverse effect penalty, and the epsilon-weighted AMR cost.
    """
    model = create_test_reward_calculator(
        antibiotic_names=["A"],
        clinical_benefit_reward=10.0,
        clinical_benefit_probability=1.0,
        adverse_effect_penalty=-2.0,
        adverse_effect_probability=1.0,
        epsilon=0.05,
    )
    model.rng = np.random.default_rng(seed=0)

    reward, clinical_benefit, clinical_failure, adverse = model.calculate_individual_reward(
        patient_infected=True,
        antibiotic_name="A",
        infection_sensitive_to_prescribed_abx=True,
        delta_visible_amr=1.0,
        return_clinical_benefit_adverse_event_occurrence=True,
    )

    assert adverse is True
    # With normalization: max(|10.0|, |-2.0|) = 10.0
    # normalized_benefit = 10.0 / 10.0 = 1.0
    # normalized_adverse = -2.0 / 10.0 = -0.2
    # epsilon_amr = 0.05 * 1.0 = 0.05
    # Total: 1.0 - 0.2 - 0.05 = 0.75
    assert pytest.approx(reward, rel=1e-6) == 0.75


def test_calculate_individual_reward_no_prescription_no_adverse():
    """Test that when antibiotic_name is 'no_treatment' string, reward is zero with no adverse events.
    
    Purpose: Verifies that when 'no_treatment' is explicitly passed as the antibiotic name,
    the calculation returns zero reward and no adverse effects, demonstrating proper handling
    of non-treatment actions.
    """
    model = create_test_reward_calculator(
        antibiotic_names=["A"],
        clinical_benefit_reward=10.0,
        clinical_benefit_probability=1.0,
        adverse_effect_penalty=-2.0,
        adverse_effect_probability=1.0,
        epsilon=0.05,
    )
    model.rng = np.random.default_rng(seed=1)

    # Test with None as antibiotic_name (no treatment)
    # This should result in zero reward and no adverse effects since no antibiotic is prescribed
    reward, clinical_benefit, clinical_failure, adverse = model.calculate_individual_reward(
        patient_infected=True,
        antibiotic_name='no_treatment',
        infection_sensitive_to_prescribed_abx=None,
        delta_visible_amr=None,
        return_clinical_benefit_adverse_event_occurrence=True,
    )

    assert adverse is False
    assert reward == 0.0  # No treatment should produce zero reward


def test_calculate_reward_aggregation_and_counts():
    """Test that calculate_reward aggregates individual rewards and counts outcomes correctly.
    
    Purpose: Verifies that the environment-level reward calculation properly aggregates
    individual patient rewards, counts successful treatments and adverse events, and combines
    individual and community components using the lambda weight.
    """
    model = create_test_reward_calculator(
        antibiotic_names=["A"],
        clinical_benefit_reward=10.0,
        clinical_benefit_probability=1.0,
        adverse_effect_penalty=-2.0,
        adverse_effect_probability=0.0,
        lambda_weight=0.5,
        epsilon=0.1,
    )
    model.rng = np.random.default_rng(seed=123)

    # Create Patient objects instead of raw arrays
    patients = [
        Patient(
            prob_infected=1.0,
            benefit_value_multiplier=1.0,
            failure_value_multiplier=1.0,
            benefit_probability_multiplier=1.0,
            failure_probability_multiplier=1.0,
            recovery_without_treatment_prob=0.0,
            infection_status=True,
            abx_sensitivity_dict={"A": True},
            prob_infected_obs=1.0,
            benefit_value_multiplier_obs=1.0,
            failure_value_multiplier_obs=1.0,
            benefit_probability_multiplier_obs=1.0,
            failure_probability_multiplier_obs=1.0,
            recovery_without_treatment_prob_obs=0.0,
        ),
        Patient(
            prob_infected=0.0,
            benefit_value_multiplier=1.0,
            failure_value_multiplier=1.0,
            benefit_probability_multiplier=1.0,
            failure_probability_multiplier=1.0,
            recovery_without_treatment_prob=0.0,
            infection_status=False,
            abx_sensitivity_dict={"A": False},
            prob_infected_obs=0.0,
            benefit_value_multiplier_obs=1.0,
            failure_value_multiplier_obs=1.0,
            benefit_probability_multiplier_obs=1.0,
            failure_probability_multiplier_obs=1.0,
            recovery_without_treatment_prob_obs=0.0,
        ),
    ]
    actions = np.array([0, 1], dtype=int)
    antibiotic_names = ["A"]
    amr_levels = {"A": 0.0}
    delta_amr_per_antibiotic = {"A": 0.5}

    total_reward, info = model.calculate_reward(
        patients=patients,
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=amr_levels,
        delta_visible_amr_per_antibiotic=delta_amr_per_antibiotic,
    )

    assert info["count_clinical_benefits"] == 1
    assert info["count_adverse_events"] == 0
    # With always-on normalization:
    # Patient 1: benefit (1.0 normalized) - epsilon_amr (0.1 * 0.5 = 0.05) = 0.95
    # Patient 2: 0.0 (not infected, no treatment)
    # Total individual component: 0.95
    # Normalized individual: 0.95 / 2 = 0.475
    # Community reward: -0.0 (AMR level is 0), normalized: 0
    # Total: 0.5 * 0 + 0.5 * 0.475 = 0.2375
    assert pytest.approx(info["overall_individual_reward_component"], rel=1e-6) == 0.95
    assert pytest.approx(info["normalized_individual_reward"], rel=1e-6) == 0.475
    assert pytest.approx(total_reward, rel=1e-6) == 0.2375


def test_clone_and_convenience_constructors_preserve_params():
    """Test that cloning and convenience constructors preserve model parameters correctly.
    
    Purpose: Ensures that clone_with_lambda creates a new model with updated lambda while
    preserving other parameters, and that convenience constructors (IndividualOnlyReward,
    CommunityOnlyReward, BalancedReward) correctly set their lambda values to 0.0, 1.0,
    and 0.5 respectively while inheriting other parameters from the source model.
    """
    model = create_test_reward_calculator(
        antibiotic_names=["A"],
        clinical_benefit_reward=8.0,
        clinical_benefit_probability=0.7,
        adverse_effect_penalty=-1.5,
        adverse_effect_probability=0.25,
        lambda_weight=0.3,
        epsilon=0.02,
    )
    model.rng = np.random.default_rng(seed=999)

    clone = model.clone_with_lambda(0.8)
    val_clone = clone.rng.random()
    val_model = model.rng.random()

    assert clone.lambda_weight == 0.8
    assert clone.epsilon == model.epsilon
    assert val_clone == val_model

    indiv = IndividualOnlyReward.from_existing(model)
    comm = CommunityOnlyReward.from_existing(model)
    bal = BalancedReward.from_existing(model)

    assert indiv.lambda_weight == 0.0
    assert comm.lambda_weight == 1.0
    assert bal.lambda_weight == 0.5
    for derived in (indiv, comm, bal):
        assert derived.epsilon == model.epsilon


def test_missing_antibiotic_in_delta_raises():
    """Test that missing an antibiotic in delta_amr_per_antibiotic raises KeyError.
    
    Purpose: Ensures that the reward calculation fails with a clear error if a prescribed 
    antibiotic's marginal AMR contribution is not provided in delta_amr_per_antibiotic,
    preventing incomplete reward calculations.
    """
    model = create_test_reward_calculator(antibiotic_names=["A"])
    model.rng = np.random.default_rng(seed=0)
    
    # Create Patient object instead of raw array
    patients = [
        Patient(
            prob_infected=0.5,
            benefit_value_multiplier=1.0,
            failure_value_multiplier=1.0,
            benefit_probability_multiplier=1.0,
            failure_probability_multiplier=1.0,
            recovery_without_treatment_prob=0.0,
            infection_status=True,
            abx_sensitivity_dict={'A': True},
            prob_infected_obs=0.5,
            benefit_value_multiplier_obs=1.0,
            failure_value_multiplier_obs=1.0,
            benefit_probability_multiplier_obs=1.0,
            failure_probability_multiplier_obs=1.0,
            recovery_without_treatment_prob_obs=0.0,
        )
    ]
    actions = np.array([0], dtype=int)
    antibiotic_names = ["A"]
    amr_levels = {"A": 0.1}
    delta_amr_per_antibiotic = {"B": 0.0}  # incorrect key; should be "A"
    
    with pytest.raises(KeyError):
        model.calculate_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=amr_levels,
            delta_visible_amr_per_antibiotic=delta_amr_per_antibiotic,
        )
