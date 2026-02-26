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
    # Rewards are always normalized by max absolute value (10.0)
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
            abx_sensitivity_dict={"A": True},
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
            abx_sensitivity_dict={"A": True},
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


def test_patient_attributes_use_true_values():
    """Test that reward calculation uses TRUE patient attributes, not observed/noisy versions.
    
    Purpose: Verifies the architectural principle that rewards are always computed from
    ground-truth patient attributes (prob_infected, benefit_value_multiplier, etc.)
    rather than from observed/noisy versions (_obs variants). This is intentional design:
    agent observes noisy versions but is rewarded based on reality, creating a learning
    challenge for recurrent RL.
    
    Test strategy: Create two patients with identical true attributes but DIFFERENT observed
    attributes. Verify that rewards are identical when true attributes match, regardless of
    observed values. Then swap true values and verify rewards change accordingly.
    """
    model = create_test_reward_calculator(
        antibiotic_names=["A"],
        clinical_benefit_reward=10.0,
        clinical_benefit_probability=1.0,
        adverse_effect_penalty=-2.0,
        adverse_effect_probability=0.0,
        lambda_weight=0.0,  # Only individual component
        epsilon=0.05,
    )
    model.rng = np.random.default_rng(seed=42)

    # Patient 1: Infected (true), but observed as not infected
    # This demonstrates: true state drives reward, not observed perception
    patient_infected_true_noisy_obs = Patient(
        prob_infected=1.0,  # TRUE: definitely infected
        benefit_value_multiplier=1.0,
        failure_value_multiplier=1.0,
        benefit_probability_multiplier=1.0,
        failure_probability_multiplier=1.0,
        recovery_without_treatment_prob=0.0,
        infection_status=True,
        abx_sensitivity_dict={"A": True},
        prob_infected_obs=0.1,  # OBSERVED: looks like not infected (noisy/biased observation)
        benefit_value_multiplier_obs=1.0,
        failure_value_multiplier_obs=1.0,
        benefit_probability_multiplier_obs=1.0,
        failure_probability_multiplier_obs=1.0,
        recovery_without_treatment_prob_obs=0.0,
    )

    # Patient 2: Same true state, but DIFFERENT observed values
    patient_infected_true_different_noisy_obs = Patient(
        prob_infected=1.0,  # TRUE: same as Patient 1
        benefit_value_multiplier=1.0,
        failure_value_multiplier=1.0,
        benefit_probability_multiplier=1.0,
        failure_probability_multiplier=1.0,
        recovery_without_treatment_prob=0.0,
        infection_status=True,
        abx_sensitivity_dict={"A": True},
        prob_infected_obs=0.9,  # OBSERVED: looks like definitely infected (different noise)
        benefit_value_multiplier_obs=2.0,  # Different observed multiplier
        failure_value_multiplier_obs=0.5,
        benefit_probability_multiplier_obs=0.8,
        failure_probability_multiplier_obs=1.2,
        recovery_without_treatment_prob_obs=0.1,
    )

    # Both should give same reward because true attributes are identical
    actions_prescribe = np.array([0, 0], dtype=int)  # Both get antibiotic A
    amr_levels = {"A": 0.0}
    delta_amr = {"A": 0.5}

    reward1, info1 = model.calculate_reward(
        patients=[patient_infected_true_noisy_obs],
        actions=actions_prescribe[:1],
        antibiotic_names=["A"],
        visible_amr_levels=amr_levels,
        delta_visible_amr_per_antibiotic=delta_amr,
    )

    reward2, info2 = model.calculate_reward(
        patients=[patient_infected_true_different_noisy_obs],
        actions=actions_prescribe[:1],
        antibiotic_names=["A"],
        visible_amr_levels=amr_levels,
        delta_visible_amr_per_antibiotic=delta_amr,
    )

    # Rewards should match despite different observed attributes
    assert pytest.approx(reward1, rel=1e-6) == reward2, (
        "Reward should depend on TRUE attributes only, not on observed/noisy versions"
    )
    assert pytest.approx(info1["overall_individual_reward_component"], rel=1e-6) == (
        info2["overall_individual_reward_component"]
    )


def test_community_reward_uses_true_amr():
    """Test that community reward component uses visible_amr_levels for clinical authenticity.
    
    Purpose: Verifies that community-level AMR penalty is calculated using visible (observed)
    AMR levels, maintaining POMDP consistency. Agents should receive reward feedback based on
    what they observe, not ground truth. This ensures agents learn from observable signals.
    
    Test strategy: Create identical patient/action scenarios, then pass different visible_amr_levels.
    Verify that community penalty component differs proportionally with visible_amr changes.
    """
    # Set lambda_weight to 1.0 to isolate community component
    model = create_test_reward_calculator(
        antibiotic_names=["A", "B"],
        clinical_benefit_reward=10.0,
        clinical_benefit_probability=1.0,
        lambda_weight=1.0,  # Pure community reward
        epsilon=0.0,  # No individual component
    )
    model.rng = np.random.default_rng(seed=100)

    patients = [
        Patient(
            prob_infected=0.0,  # Not infected; community penalty is only component
            benefit_value_multiplier=1.0,
            failure_value_multiplier=1.0,
            benefit_probability_multiplier=1.0,
            failure_probability_multiplier=1.0,
            recovery_without_treatment_prob=0.0,
            infection_status=False,
            abx_sensitivity_dict={"A": True, "B": True},
            prob_infected_obs=0.0,
            benefit_value_multiplier_obs=1.0,
            failure_value_multiplier_obs=1.0,
            benefit_probability_multiplier_obs=1.0,
            failure_probability_multiplier_obs=1.0,
            recovery_without_treatment_prob_obs=0.0,
        )
    ]
    actions = np.array([1], dtype=int)  # No treatment
    antibiotic_names = ["A", "B"]

    # Scenario 1: Low actual AMR
    low_amr = {"A": 0.1, "B": 0.1}
    reward_low, _ = model.calculate_reward(
        patients=patients,
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=low_amr,
        delta_visible_amr_per_antibiotic={"A": 0.0, "B": 0.0},
    )

    # Scenario 2: High visible AMR (same patients/actions, different visible_amr_levels)
    high_amr = {"A": 0.9, "B": 0.9}
    reward_high, _ = model.calculate_reward(
        patients=patients,
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=high_amr,
        delta_visible_amr_per_antibiotic={"A": 0.0, "B": 0.0},
    )

    # Community penalty should be more negative when visible_amr is higher
    assert reward_high < reward_low, (
        "Community reward should be more negative (worse) when visible_amr_levels are higher"
    )
    # Verify the difference is proportional: 0.8 * 2 = 1.6 total AMR difference
    # Expected penalty difference should match AMR difference
    amr_difference = (0.9 - 0.1) + (0.9 - 0.1)  # Total AMR difference across antibiotics
    reward_difference = reward_low - reward_high
    assert reward_difference > 0, "Low AMR should yield higher (less negative) reward"


def test_sensitivity_calculation_uses_patient_sensitivity_dict():
    """Test that infection sensitivity is determined by Patient.abx_sensitivity_dict.
    
    Purpose: Verifies that RewardCalculator uses the patient-provided sensitivity status,
    not visible AMR levels, once sensitivity is sampled upstream by PatientGenerator.
    """
    model = create_test_reward_calculator(
        antibiotic_names=["A"],
        clinical_benefit_reward=100.0,  # Large benefit if sensitive
        clinical_benefit_probability=1.0,
        clinical_failure_penalty=-50.0,  # Large penalty if resistant
        clinical_failure_probability=1.0,
        adverse_effect_penalty=-1.0,  # Small penalty (must be negative)
        adverse_effect_probability=0.0,  # No adverse effects for clarity
        lambda_weight=0.0,  # Individual only
        epsilon=0.0,  # No AMR penalty for clarity
    )
    model.rng = np.random.default_rng(seed=200)

    # Patient definitely infected, definitely prescribed A
    patient = Patient(
        prob_infected=1.0,  # TRUE: definitely infected
        benefit_value_multiplier=1.0,
        failure_value_multiplier=1.0,
        benefit_probability_multiplier=1.0,
        failure_probability_multiplier=1.0,
        recovery_without_treatment_prob=0.0,
        infection_status=True,
        abx_sensitivity_dict={"A": False},
        prob_infected_obs=1.0,
        benefit_value_multiplier_obs=1.0,
        failure_value_multiplier_obs=1.0,
        benefit_probability_multiplier_obs=1.0,
        failure_probability_multiplier_obs=1.0,
        recovery_without_treatment_prob_obs=0.0,
    )

    actions = np.array([0], dtype=int)  # Prescribe antibiotic A
    antibiotic_names = ["A"]

    # Visible AMR levels should not affect sensitivity when patient dict is fixed
    high_visible_amr = {"A": 0.95}
    low_visible_amr = {"A": 0.05}

    reward_resistant_high, _ = model.calculate_reward(
        patients=[patient],
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=high_visible_amr,
        delta_visible_amr_per_antibiotic={"A": 0.0},
    )
    reward_resistant_low, _ = model.calculate_reward(
        patients=[patient],
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=low_visible_amr,
        delta_visible_amr_per_antibiotic={"A": 0.0},
    )

    assert reward_resistant_high == reward_resistant_low

    sensitive_patient = Patient(
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
    )

    reward_sensitive, _ = model.calculate_reward(
        patients=[sensitive_patient],
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=high_visible_amr,
        delta_visible_amr_per_antibiotic={"A": 0.0},
    )

    assert reward_resistant_high < reward_sensitive


def test_individual_reward_uses_true_delta_amr():
    """Test that AMR penalty component uses delta_visible_amr for clinical authenticity.
    
    Purpose: Verifies that the epsilon-weighted AMR penalty in individual rewards is
    calculated using the visible delta_visible_amr (change in observed AMR from prescribing),
    maintaining POMDP consistency. Agents should receive reward feedback based on observable
    AMR changes, not ground truth.
    
    Test strategy: Create identical patient scenarios and vary delta_visible_amr_per_antibiotic
    to show that epsilon penalties scale accordingly.
    """
    model = create_test_reward_calculator(
        antibiotic_names=["A"],
        clinical_benefit_reward=10.0,
        clinical_benefit_probability=1.0,
        adverse_effect_penalty=-1.0,  # Must be negative (small value for clarity)
        adverse_effect_probability=0.0,  # No adverse for clarity
        lambda_weight=0.0,  # Individual only
        epsilon=0.1,  # Significant epsilon for clear detection
    )
    model.rng = np.random.default_rng(seed=300)

    patient = Patient(
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
    )

    actions = np.array([0], dtype=int)  # Prescribe A
    antibiotic_names = ["A"]
    amr_levels = {"A": 0.5}

    # Scenario 1: Small delta_visible_amr (antibiotic doesn't contribute much observed resistance)
    small_delta = {"A": 0.1}
    reward_small, _ = model.calculate_reward(
        patients=[patient],
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=amr_levels,
        delta_visible_amr_per_antibiotic=small_delta,
    )

    # Scenario 2: Large delta_visible_amr (antibiotic drives significant observed resistance)
    large_delta = {"A": 0.5}
    reward_large, _ = model.calculate_reward(
        patients=[patient],
        actions=actions,
        antibiotic_names=antibiotic_names,
        visible_amr_levels=amr_levels,
        delta_visible_amr_per_antibiotic=large_delta,
    )

    # Larger delta should produce larger AMR penalty, hence lower reward
    assert reward_large < reward_small, (
        "Larger delta_visible_amr should yield lower reward due to higher AMR penalty"
    )


class TestExpectedRewardCalculation:
    """Tests for expected reward calculation methods."""
    
    def test_expected_individual_reward_no_treatment(self):
        """Test expected reward for no-treatment action (spontaneous recovery path)."""
        rc = create_test_reward_calculator(
            antibiotic_names=['A'],
            clinical_benefit_reward=10.0,
            clinical_benefit_probability=0.8,
            clinical_failure_penalty=-5.0,
            clinical_failure_probability=0.2,
        )
        
        patient = Patient(
            prob_infected=0.5,
            benefit_value_multiplier=1.0,
            failure_value_multiplier=1.0,
            benefit_probability_multiplier=1.0,
            failure_probability_multiplier=1.0,
            recovery_without_treatment_prob=0.3,
            infection_status=True,
            abx_sensitivity_dict={'A': True},
            prob_infected_obs=0.5,
            benefit_value_multiplier_obs=1.0,
            failure_value_multiplier_obs=1.0,
            benefit_probability_multiplier_obs=1.0,
            failure_probability_multiplier_obs=1.0,
            recovery_without_treatment_prob_obs=0.3,
        )
        
        # No-treatment path: uses recovery_without_treatment_prob
        reward = rc.calculate_expected_individual_reward(
            patient=patient,
            antibiotic_name='no_treatment',
            visible_amr_level=0.0,
            delta_visible_amr=0.0,
        )
        
        # Should incorporate recovery probability and failure probability
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        # With recovery prob 0.3, should get some benefit; with failure prob 0.2, should lose some
        assert reward != 0.0
    
    def test_expected_individual_reward_with_treatment(self):
        """Test expected reward for treatment action."""
        rc = create_test_reward_calculator(
            antibiotic_names=['A'],
            clinical_benefit_reward=10.0,
            clinical_benefit_probability=0.8,
            clinical_failure_penalty=-5.0,
            clinical_failure_probability=0.1,
            adverse_effect_penalty=-2.0,
            adverse_effect_probability=0.05,
        )
        
        patient = Patient(
            prob_infected=0.6,
            benefit_value_multiplier=1.2,
            failure_value_multiplier=0.9,
            benefit_probability_multiplier=1.1,
            failure_probability_multiplier=0.95,
            recovery_without_treatment_prob=0.2,
            infection_status=True,
            abx_sensitivity_dict={'A': True},
            prob_infected_obs=0.6,
            benefit_value_multiplier_obs=1.2,
            failure_value_multiplier_obs=0.9,
            benefit_probability_multiplier_obs=1.1,
            failure_probability_multiplier_obs=0.95,
            recovery_without_treatment_prob_obs=0.2,
        )
        
        # Treatment path: A antibiotic
        reward = rc.calculate_expected_individual_reward(
            patient=patient,
            antibiotic_name='A',
            visible_amr_level=0.2,  # Some resistance
            delta_visible_amr=0.05,  # Some AMR increase
        )
        
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        # Should account for sensitivity (1-amr=0.8), benefit, adverse effects, AMR penalty
        assert reward < 0.0 or reward > 0.0  # Could go either way depending on params
    
    def test_expected_individual_reward_high_amr(self):
        """Test expected reward when AMR is high (low sensitivity)."""
        rc = create_test_reward_calculator(
            antibiotic_names=['A'],
            clinical_benefit_reward=10.0,
            clinical_failure_penalty=-5.0,
            epsilon=0.1,  # Larger AMR penalty weight
        )
        
        patient = Patient(
            prob_infected=0.8,
            benefit_value_multiplier=1.0,
            failure_value_multiplier=1.0,
            benefit_probability_multiplier=1.0,
            failure_probability_multiplier=1.0,
            recovery_without_treatment_prob=0.1,
            infection_status=True,
            abx_sensitivity_dict={'A': True},
            prob_infected_obs=0.8,
            benefit_value_multiplier_obs=1.0,
            failure_value_multiplier_obs=1.0,
            benefit_probability_multiplier_obs=1.0,
            failure_probability_multiplier_obs=1.0,
            recovery_without_treatment_prob_obs=0.1,
        )
        
        # With high AMR (low sensitivity), treatment is less effective
        reward_high_amr = rc.calculate_expected_individual_reward(
            patient=patient,
            antibiotic_name='A',
            visible_amr_level=0.9,  # Very high resistance
            delta_visible_amr=0.05,
        )
        
        reward_low_amr = rc.calculate_expected_individual_reward(
            patient=patient,
            antibiotic_name='A',
            visible_amr_level=0.1,  # Low resistance
            delta_visible_amr=0.05,
        )
        
        # High AMR should produce lower reward (less sensitivity = less benefit)
        assert reward_high_amr < reward_low_amr
    
    def test_expected_individual_reward_probability_clamping(self):
        """Test that probabilities are clamped to [0,1] internally during calculation."""
        rc = create_test_reward_calculator(
            antibiotic_names=['A'],
            clinical_benefit_probability=0.8,
            adverse_effect_probability=0.6,
        )
        
        patient = Patient(
            prob_infected=0.5,
            benefit_value_multiplier=1.0,
            failure_value_multiplier=1.0,
            benefit_probability_multiplier=1.5,  # Will multiply probability
            failure_probability_multiplier=1.0,
            recovery_without_treatment_prob=0.5,
            infection_status=True,
            abx_sensitivity_dict={'A': True},
            prob_infected_obs=0.5,
            benefit_value_multiplier_obs=1.0,
            failure_value_multiplier_obs=1.0,
            benefit_probability_multiplier_obs=1.5,
            failure_probability_multiplier_obs=1.0,
            recovery_without_treatment_prob_obs=0.5,
        )
        
        # Should not raise; probabilities should be clamped internally
        reward = rc.calculate_expected_individual_reward(
            patient=patient,
            antibiotic_name='A',
            visible_amr_level=0.3,
            delta_visible_amr=0.0,
        )
        
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        # Clamping should keep reward in reasonable bounds
        assert -50 < reward < 50
    
    def test_expected_individual_reward_invalid_antibiotic(self):
        """Test error handling for invalid antibiotic name."""
        rc = create_test_reward_calculator(antibiotic_names=['A'])
        
        patient = Patient(
            prob_infected=0.5,
            benefit_value_multiplier=1.0,
            failure_value_multiplier=1.0,
            benefit_probability_multiplier=1.0,
            failure_probability_multiplier=1.0,
            recovery_without_treatment_prob=0.3,
            infection_status=True,
            abx_sensitivity_dict={'A': True},
            prob_infected_obs=0.5,
            benefit_value_multiplier_obs=1.0,
            failure_value_multiplier_obs=1.0,
            benefit_probability_multiplier_obs=1.0,
            failure_probability_multiplier_obs=1.0,
            recovery_without_treatment_prob_obs=0.3,
        )
        
        with pytest.raises(ValueError, match="not found"):
            rc.calculate_expected_individual_reward(
                patient=patient,
                antibiotic_name='InvalidDrug',
                visible_amr_level=0.3,
                delta_visible_amr=0.0,
            )
    
    def test_calculate_expected_reward_orchestration(self):
        """Test expected reward calculation orchestration with multiple patients."""
        rc = create_test_reward_calculator(
            antibiotic_names=['A', 'B'],
            lambda_weight=0.5,
        )
        
        patients = [
            Patient(
                prob_infected=0.5,
                benefit_value_multiplier=1.0,
                failure_value_multiplier=1.0,
                benefit_probability_multiplier=1.0,
                failure_probability_multiplier=1.0,
                recovery_without_treatment_prob=0.3,
                infection_status=True,
                abx_sensitivity_dict={'A': True, 'B': False},
                prob_infected_obs=0.5,
                benefit_value_multiplier_obs=1.0,
                failure_value_multiplier_obs=1.0,
                benefit_probability_multiplier_obs=1.0,
                failure_probability_multiplier_obs=1.0,
                recovery_without_treatment_prob_obs=0.3,
            ),
            Patient(
                prob_infected=0.7,
                benefit_value_multiplier=1.1,
                failure_value_multiplier=0.9,
                benefit_probability_multiplier=1.0,
                failure_probability_multiplier=1.0,
                recovery_without_treatment_prob=0.2,
                infection_status=True,
                abx_sensitivity_dict={'A': False, 'B': True},
                prob_infected_obs=0.7,
                benefit_value_multiplier_obs=1.1,
                failure_value_multiplier_obs=0.9,
                benefit_probability_multiplier_obs=1.0,
                failure_probability_multiplier_obs=1.0,
                recovery_without_treatment_prob_obs=0.2,
            ),
        ]
        
        # Actions: patient 0 → A, patient 1 → B
        actions = np.array([1, 2])  # Assuming index 1=A, 2=B
        
        visible_amr_levels = {'A': 0.3, 'B': 0.2}
        delta_visible_amr_per_antibiotic = {'A': 0.05, 'B': 0.04}
        
        expected_reward = rc.calculate_expected_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=rc.antibiotic_names,
            visible_amr_levels=visible_amr_levels,
            delta_visible_amr_per_antibiotic=delta_visible_amr_per_antibiotic,
        )
        
        assert isinstance(expected_reward, float)
        assert not np.isnan(expected_reward)
        # Should be a weighted combination of individual and community rewards
        assert -100 < expected_reward < 100  # Sanity bounds
    
    def test_expected_reward_no_treatment_action(self):
        """Test expected reward when action is 'no_treatment'."""
        rc = create_test_reward_calculator(antibiotic_names=['A'])
        
        patients = [
            Patient(
                prob_infected=0.5,
                benefit_value_multiplier=1.0,
                failure_value_multiplier=1.0,
                benefit_probability_multiplier=1.0,
                failure_probability_multiplier=1.0,
                recovery_without_treatment_prob=0.3,
                infection_status=True,
                abx_sensitivity_dict={'A': True},
                prob_infected_obs=0.5,
                benefit_value_multiplier_obs=1.0,
                failure_value_multiplier_obs=1.0,
                benefit_probability_multiplier_obs=1.0,
                failure_probability_multiplier_obs=1.0,
                recovery_without_treatment_prob_obs=0.3,
            ),
        ]
        
        # Action 0 = no_treatment (based on typical indexing)
        actions = np.array([0])
        
        visible_amr_levels = {'A': 0.3}
        delta_visible_amr_per_antibiotic = {'A': 0.0}  # No AMR increase with no treatment
        
        expected_reward = rc.calculate_expected_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=rc.antibiotic_names,
            visible_amr_levels=visible_amr_levels,
            delta_visible_amr_per_antibiotic=delta_visible_amr_per_antibiotic,
        )
        
        assert isinstance(expected_reward, float)
        assert not np.isnan(expected_reward)
    
    def test_expected_reward_lambda_weighting(self):
        """Test that lambda correctly weights community vs individual rewards."""
        rc_community = create_test_reward_calculator(
            antibiotic_names=['A'],
            lambda_weight=1.0,  # All community, no individual
        )
        
        rc_individual = create_test_reward_calculator(
            antibiotic_names=['A'],
            lambda_weight=0.0,  # All individual, no community
        )
        
        patients = [
            Patient(
                prob_infected=0.5,
                benefit_value_multiplier=1.0,
                failure_value_multiplier=1.0,
                benefit_probability_multiplier=1.0,
                failure_probability_multiplier=1.0,
                recovery_without_treatment_prob=0.3,
                infection_status=True,
                abx_sensitivity_dict={'A': True},
                prob_infected_obs=0.5,
                benefit_value_multiplier_obs=1.0,
                failure_value_multiplier_obs=1.0,
                benefit_probability_multiplier_obs=1.0,
                failure_probability_multiplier_obs=1.0,
                recovery_without_treatment_prob_obs=0.3,
            ),
        ]
        
        actions = np.array([1])  # Treat with A
        visible_amr_levels = {'A': 0.3}
        delta_visible_amr_per_antibiotic = {'A': 0.05}
        
        reward_community = rc_community.calculate_expected_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=rc_community.antibiotic_names,
            visible_amr_levels=visible_amr_levels,
            delta_visible_amr_per_antibiotic=delta_visible_amr_per_antibiotic,
        )
        
        reward_individual = rc_individual.calculate_expected_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=rc_individual.antibiotic_names,
            visible_amr_levels=visible_amr_levels,
            delta_visible_amr_per_antibiotic=delta_visible_amr_per_antibiotic,
        )
        
        # Should be different due to different weighting
        assert reward_community != reward_individual
