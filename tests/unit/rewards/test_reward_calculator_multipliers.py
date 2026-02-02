"""
Test suite for RewardCalculator patient-specific multipliers.

This module tests the new heterogeneous patient functionality where each patient
can have different multipliers for probability and value of clinical outcomes.
"""

import numpy as np
import pytest
from abx_amr_simulator.core import RewardCalculator


def create_test_reward_calculator(
    antibiotic_names=None,
    clinical_benefit_reward=10.0,
    clinical_benefit_probability=1.0,
    clinical_failure_penalty=-5.0,
    clinical_failure_probability=1.0,
    adverse_effect_penalty=-2.0,
    adverse_effect_probability=1.0,
    lambda_weight=0.5,
    epsilon=0.01,  # Small but valid (must be > 0)
    seed=0,
):
    """Helper to create a RewardCalculator with simple test parameters."""
    if antibiotic_names is None:
        antibiotic_names = ["A", "B"]
    
    abx_adverse_effects_info = {
        abx: {
            "adverse_effect_penalty": adverse_effect_penalty,
            "adverse_effect_probability": adverse_effect_probability,
        }
        for abx in antibiotic_names
    }
    
    abx_clinical_reward_penalties_info_dict = {
        "clinical_benefit_reward": clinical_benefit_reward,
        "clinical_benefit_probability": clinical_benefit_probability,
        "clinical_failure_penalty": clinical_failure_penalty,
        "clinical_failure_probability": clinical_failure_probability,
        "abx_adverse_effects_info": abx_adverse_effects_info,
    }
    
    config = {
        'abx_clinical_reward_penalties_info_dict': abx_clinical_reward_penalties_info_dict,
        'lambda_weight': lambda_weight,
        'epsilon': epsilon,
        'seed': seed,
    }
    return RewardCalculator(config=config)


class TestBenefitMultipliers:
    """Test benefit_value_multiplier and benefit_probability_multiplier."""
    
    def test_benefit_value_multiplier_scales_reward(self):
        """Verify that benefit_value_multiplier scales the clinical benefit reward."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
            clinical_benefit_probability=1.0,  # Guaranteed benefit
            adverse_effect_probability=0.0,
            epsilon=0.01,  # Small epsilon
        )
        model.rng = np.random.default_rng(seed=42)
        
        # With normalization: max(|10.0|, |-2.0|) = 10.0
        # normalized_benefit = 10.0 / 10.0 = 1.0
        reward_default = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="A",
            infection_sensitive_to_prescribed_abx=True,
            delta_visible_amr=0.0,  # No AMR change means no AMR penalty
            benefit_value_multiplier=1.0,
        )
        assert reward_default == 1.0
        
        # 2x multiplier should double the reward
        reward_doubled = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="A",
            infection_sensitive_to_prescribed_abx=True,
            delta_visible_amr=0.0,
            benefit_value_multiplier=2.0,
        )
        assert reward_doubled == 2.0
        
        # 0.5x multiplier should halve the reward
        reward_halved = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="A",
            infection_sensitive_to_prescribed_abx=True,
            delta_visible_amr=0.0,
            benefit_value_multiplier=0.5,
        )
        assert reward_halved == 0.5
    
    def test_benefit_probability_multiplier_affects_occurrence(self):
        """Verify that benefit_probability_multiplier scales the probability of benefit occurring."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
            clinical_benefit_probability=0.5,  # 50% base chance
            adverse_effect_probability=0.0,
        )
        
        # Run many trials to estimate probability
        n_trials = 1000
        
        # With 1.0 multiplier, should get ~50% success rate
        model.rng = np.random.default_rng(seed=42)
        successes_default = 0
        for _ in range(n_trials):
            _, benefit, _, _ = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="A",
                infection_sensitive_to_prescribed_abx=True,
                delta_visible_amr=0.0,
                benefit_probability_multiplier=1.0,
                return_clinical_benefit_adverse_event_occurrence=True,
            )
            if benefit:
                successes_default += 1
        
        rate_default = successes_default / n_trials
        assert 0.45 < rate_default < 0.55  # Should be ~0.5
        
        # With 2.0 multiplier (probability becomes min(1.0, 0.5 * 2.0) = 1.0), should get 100% success
        model.rng = np.random.default_rng(seed=42)
        successes_doubled = 0
        for _ in range(n_trials):
            _, benefit, _, _ = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="A",
                infection_sensitive_to_prescribed_abx=True,
                delta_visible_amr=0.0,
                benefit_probability_multiplier=2.0,
                return_clinical_benefit_adverse_event_occurrence=True,
            )
            if benefit:
                successes_doubled += 1
        
        rate_doubled = successes_doubled / n_trials
        assert rate_doubled == 1.0  # Clamped to 1.0
        
        # With 0.5 multiplier (probability becomes 0.5 * 0.5 = 0.25), should get ~25% success
        model.rng = np.random.default_rng(seed=42)
        successes_halved = 0
        for _ in range(n_trials):
            _, benefit, _, _ = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="A",
                infection_sensitive_to_prescribed_abx=True,
                delta_visible_amr=0.0,
                benefit_probability_multiplier=0.5,
                return_clinical_benefit_adverse_event_occurrence=True,
            )
            if benefit:
                successes_halved += 1
        
        rate_halved = successes_halved / n_trials
        assert 0.20 < rate_halved < 0.30  # Should be ~0.25


class TestFailureMultipliers:
    """Test failure_value_multiplier and failure_probability_multiplier."""
    
    def test_failure_value_multiplier_scales_penalty(self):
        """Verify that failure_value_multiplier scales the clinical failure penalty."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_failure_penalty=-10.0,
            clinical_failure_probability=1.0,  # Guaranteed failure
            adverse_effect_probability=0.0,
        )
        model.rng = np.random.default_rng(seed=42)
        
        # With normalization: max(|10.0|, |-10.0|) = 10.0
        # normalized_failure = -10.0 / 10.0 = -1.0
        reward_default = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="A",
            infection_sensitive_to_prescribed_abx=False,  # Resistant -> failure
            delta_visible_amr=0.0,
            failure_value_multiplier=1.0,
        )
        assert reward_default == -1.0
        
        # 2x multiplier should double the penalty (more negative)
        reward_doubled = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="A",
            infection_sensitive_to_prescribed_abx=False,
            delta_visible_amr=0.0,
            failure_value_multiplier=2.0,
        )
        assert reward_doubled == -2.0
        
        # 0.5x multiplier should halve the penalty (less negative)
        reward_halved = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="A",
            infection_sensitive_to_prescribed_abx=False,
            delta_visible_amr=0.0,
            failure_value_multiplier=0.5,
        )
        assert reward_halved == -0.5
    
    def test_failure_probability_multiplier_affects_occurrence(self):
        """Verify that failure_probability_multiplier scales the probability of failure occurring."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_failure_penalty=-10.0,
            clinical_failure_probability=0.5,  # 50% base chance
            adverse_effect_probability=0.0,
        )
        
        n_trials = 1000
        
        # With 1.0 multiplier, should get ~50% failure rate
        model.rng = np.random.default_rng(seed=42)
        failures_default = 0
        for _ in range(n_trials):
            _, _, failure, _ = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="A",
                infection_sensitive_to_prescribed_abx=False,  # Resistant -> check failure
                delta_visible_amr=0.0,
                failure_probability_multiplier=1.0,
                return_clinical_benefit_adverse_event_occurrence=True,
            )
            if failure:
                failures_default += 1
        
        rate_default = failures_default / n_trials
        assert 0.45 < rate_default < 0.55  # Should be ~0.5
        
        # With 0.5 multiplier (probability becomes 0.5 * 0.5 = 0.25), should get ~25% failure
        model.rng = np.random.default_rng(seed=42)
        failures_halved = 0
        for _ in range(n_trials):
            _, _, failure, _ = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="A",
                infection_sensitive_to_prescribed_abx=False,
                delta_visible_amr=0.0,
                failure_probability_multiplier=0.5,
                return_clinical_benefit_adverse_event_occurrence=True,
            )
            if failure:
                failures_halved += 1
        
        rate_halved = failures_halved / n_trials
        assert 0.20 < rate_halved < 0.30  # Should be ~0.25


class TestSpontaneousRecovery:
    """Test recovery_without_treatment_prob for no-treatment cases."""
    
    def test_spontaneous_recovery_grants_benefit(self):
        """Verify that spontaneous recovery when no treatment prescribed grants clinical benefit."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
            clinical_failure_penalty=-5.0,
            clinical_failure_probability=1.0,  # If no recovery, failure is certain
        )
        
        # With recovery_prob=1.0, should always get benefit reward
        # With normalization: max(|10.0|, |-5.0|) = 10.0
        # normalized_benefit = 10.0 / 10.0 = 1.0
        model.rng = np.random.default_rng(seed=42)
        reward, benefit, failure, adverse = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="no_treatment",
            recovery_without_treatment_prob=1.0,
            benefit_value_multiplier=1.0,
            return_clinical_benefit_adverse_event_occurrence=True,
        )
        assert reward == 1.0
        assert benefit is True
        assert failure is False
        assert adverse is False
    
    def test_spontaneous_recovery_scaled_by_benefit_multiplier(self):
        """Verify that spontaneous recovery reward is scaled by benefit_value_multiplier."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
            clinical_failure_probability=0.0,  # No failure if no recovery
        )
        model.rng = np.random.default_rng(seed=42)
        
        # 2x benefit multiplier should double the recovery reward
        # normalized_benefit = 10.0 / 10.0 = 1.0, scaled by 2x = 2.0
        reward = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="no_treatment",
            recovery_without_treatment_prob=1.0,
            benefit_value_multiplier=2.0,
        )
        assert reward == 2.0
    
    def test_no_recovery_leads_to_failure_check(self):
        """Verify that when no spontaneous recovery, failure check is performed."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
            clinical_failure_penalty=-5.0,
            clinical_failure_probability=1.0,  # Guaranteed failure if no recovery
        )
        
        # With recovery_prob=0.0, should get failure penalty
        # normalized_failure = -5.0 / 10.0 = -0.5
        model.rng = np.random.default_rng(seed=42)
        reward, benefit, failure, adverse = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="no_treatment",
            recovery_without_treatment_prob=0.0,
            failure_value_multiplier=1.0,
            return_clinical_benefit_adverse_event_occurrence=True,
        )
        assert reward == -0.5
        assert benefit is False
        assert failure is True
        assert adverse is False
    
    def test_spontaneous_recovery_probability_distribution(self):
        """Verify that spontaneous recovery probability is respected."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
            clinical_failure_penalty=-5.0,
            clinical_failure_probability=1.0,
        )
        
        n_trials = 1000
        recovery_prob = 0.3
        
        model.rng = np.random.default_rng(seed=42)
        recoveries = 0
        for _ in range(n_trials):
            _, benefit, _, _ = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="no_treatment",
                recovery_without_treatment_prob=recovery_prob,
                return_clinical_benefit_adverse_event_occurrence=True,
            )
            if benefit:  # Benefit means recovery occurred
                recoveries += 1
        
        observed_rate = recoveries / n_trials
        assert 0.25 < observed_rate < 0.35  # Should be ~0.3
    
    def test_no_spontaneous_recovery_when_not_infected(self):
        """Verify that spontaneous recovery logic doesn't apply when patient not infected."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
        )
        model.rng = np.random.default_rng(seed=42)
        
        # Not infected + no treatment -> reward should be 0, no benefit
        reward, benefit, failure, adverse = model.calculate_individual_reward(
            patient_infected=False,
            antibiotic_name="no_treatment",
            recovery_without_treatment_prob=1.0,  # High recovery prob, but not infected
            return_clinical_benefit_adverse_event_occurrence=True,
        )
        assert reward == 0.0
        assert benefit is False
        assert failure is False
        assert adverse is False


class TestMultiplierCombinations:
    """Test combinations of multiple multipliers."""
    
    def test_combined_benefit_multipliers(self):
        """Verify that benefit_probability_multiplier and benefit_value_multiplier work together."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
            clinical_benefit_probability=1.0,  # Guaranteed occurrence
            adverse_effect_probability=0.0,
        )
        model.rng = np.random.default_rng(seed=42)
        
        # Both at 2.0: probability stays 1.0 (clamped), value doubles
        # normalized_benefit = 10.0 / 10.0 = 1.0, scaled by 2.0 = 2.0
        reward = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="A",
            infection_sensitive_to_prescribed_abx=True,
            delta_visible_amr=0.0,
            benefit_value_multiplier=2.0,
            benefit_probability_multiplier=2.0,
        )
        assert reward == 2.0
    
    def test_combined_failure_multipliers(self):
        """Verify that failure_probability_multiplier and failure_value_multiplier work together."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_failure_penalty=-10.0,
            clinical_failure_probability=1.0,  # Guaranteed occurrence
            adverse_effect_probability=0.0,
        )
        model.rng = np.random.default_rng(seed=42)
        
        # Both at 2.0: probability stays 1.0 (clamped), penalty doubles
        # normalized_failure = -10.0 / 10.0 = -1.0, scaled by 2.0 = -2.0
        reward = model.calculate_individual_reward(
            patient_infected=True,
            antibiotic_name="A",
            infection_sensitive_to_prescribed_abx=False,
            delta_visible_amr=0.0,
            failure_value_multiplier=2.0,
            failure_probability_multiplier=2.0,
        )
        assert reward == -2.0
    
    def test_heterogeneous_patient_profiles(self):
        """Simulate different patient risk profiles with different multiplier combinations."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
            clinical_benefit_probability=0.8,
            clinical_failure_penalty=-5.0,
            clinical_failure_probability=0.5,
            adverse_effect_probability=0.0,
            seed=42,
        )
        
        # Young healthy patient: high benefit, low failure risk
        model.rng = np.random.default_rng(seed=100)
        rewards_healthy = []
        for _ in range(100):
            reward = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="A",
                infection_sensitive_to_prescribed_abx=True,
                delta_visible_amr=0.0,
                benefit_value_multiplier=1.2,
                benefit_probability_multiplier=1.1,
                failure_value_multiplier=0.5,
                failure_probability_multiplier=0.5,
            )
            rewards_healthy.append(reward)
        
        # Elderly frail patient: lower benefit, higher failure risk
        model.rng = np.random.default_rng(seed=100)
        rewards_frail = []
        for _ in range(100):
            reward = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="A",
                infection_sensitive_to_prescribed_abx=True,
                delta_visible_amr=0.0,
                benefit_value_multiplier=0.8,
                benefit_probability_multiplier=0.9,
                failure_value_multiplier=1.5,
                failure_probability_multiplier=1.2,
            )
            rewards_frail.append(reward)
        
        # Healthy patients should have higher average rewards
        assert np.mean(rewards_healthy) > np.mean(rewards_frail)


class TestProbabilityClamping:
    """Test that scaled probabilities are properly clamped to [0, 1]."""
    
    def test_benefit_probability_clamped_at_one(self):
        """Verify that benefit probability never exceeds 1.0."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_reward=10.0,
            clinical_benefit_probability=0.9,
            adverse_effect_probability=0.0,
        )
        
        # 10x multiplier would give 9.0, but should clamp to 1.0 (100% success rate)
        model.rng = np.random.default_rng(seed=42)
        successes = 0
        n_trials = 100
        for _ in range(n_trials):
            _, benefit, _, _ = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="A",
                infection_sensitive_to_prescribed_abx=True,
                delta_visible_amr=0.0,
                benefit_probability_multiplier=10.0,
                return_clinical_benefit_adverse_event_occurrence=True,
            )
            if benefit:
                successes += 1
        
        assert successes == n_trials  # All should succeed due to clamping
    
    def test_failure_probability_clamped_at_zero(self):
        """Verify that failure probability never goes below 0.0."""
        model = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_failure_penalty=-10.0,
            clinical_failure_probability=0.1,
            adverse_effect_probability=0.0,
        )
        
        # 0.01 multiplier would give 0.001, very low failure rate
        model.rng = np.random.default_rng(seed=42)
        failures = 0
        n_trials = 1000
        for _ in range(n_trials):
            _, _, failure, _ = model.calculate_individual_reward(
                patient_infected=True,
                antibiotic_name="A",
                infection_sensitive_to_prescribed_abx=False,
                delta_visible_amr=0.0,
                failure_probability_multiplier=0.01,
                return_clinical_benefit_adverse_event_occurrence=True,
            )
            if failure:
                failures += 1
        
        failure_rate = failures / n_trials
        assert failure_rate < 0.01  # Should be very rare
