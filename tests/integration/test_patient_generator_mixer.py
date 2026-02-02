"""Tests for PatientGeneratorMixer class."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abx_amr_simulator.core import PatientGenerator, PatientGeneratorMixer

# Default AMR levels for tests (single antibiotic 'A')
TRUE_AMR_LEVELS = {'A': 0.0}


def create_low_risk_generator():
    """Create a PatientGenerator with low risk parameters."""
    config = {
        'prob_infected': {
            'prob_dist': {'type': 'gaussian', 'mu': 0.3, 'sigma': 0.05},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'benefit_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 0.6, 'sigma': 0.05},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 0.6, 'sigma': 0.05},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'benefit_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.1},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.1},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'recovery_without_treatment_prob': {
            'prob_dist': {'type': 'constant', 'value': 0.2},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'visible_patient_attributes': ['prob_infected'],
    }
    return PatientGenerator(config=config)


def create_high_risk_generator():
    """Create a PatientGenerator with high risk parameters."""
    config = {
        'prob_infected': {
            'prob_dist': {'type': 'gaussian', 'mu': 0.8, 'sigma': 0.05},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'benefit_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.6, 'sigma': 0.05},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.6, 'sigma': 0.05},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'benefit_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.1},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.1},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'recovery_without_treatment_prob': {
            'prob_dist': {'type': 'constant', 'value': 0.01},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'visible_patient_attributes': ['prob_infected'],
    }
    return PatientGenerator(config=config)


class TestPatientGeneratorMixerInitialization:
    """Test PatientGeneratorMixer initialization and validation."""
    
    def test_basic_initialization(self):
        """Test basic initialization with two generators."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.7, 0.3],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        assert len(mixer.generators) == 2
        assert np.allclose(mixer.proportions, [0.7, 0.3])
        assert mixer.seed == 42
        assert mixer.rng is not None

    def test_child_seeds_are_synchronized(self):
        """Mixer should synchronize child generator seeds/rng to its seed."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        # Set different seeds to verify they get overwritten
        gen1.seed = 1
        gen2.seed = 2
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 99,
            }
        )
        assert getattr(gen1, 'seed', None) == 99
        assert getattr(gen2, 'seed', None) == 99
        assert hasattr(gen1, 'rng') and hasattr(gen2, 'rng')
    
    def test_proportions_must_sum_to_one(self):
        """Test that proportions must sum to 1.0."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        with pytest.raises(ValueError, match="Proportions must sum to 1.0"):
            PatientGeneratorMixer(
                config={
                    'generators': [gen1, gen2],
                    'proportions': [0.5, 0.4],  # Sum to 0.9
                    'visible_patient_attributes': ['prob_infected'],
                    'seed': 42,
                }
            )
    
    def test_proportions_tolerates_small_rounding_errors(self):
        """Test that proportions tolerate small rounding errors."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        # This should work (within tolerance)
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.7, 0.30000001],  # Tiny rounding error
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        assert mixer is not None
    
    def test_negative_proportions_raise_error(self):
        """Test that negative proportions raise ValueError."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        with pytest.raises(ValueError, match="non-negative"):
            PatientGeneratorMixer(
                config={
                    'generators': [gen1, gen2],
                    'proportions': [1.2, -0.2],
                    'visible_patient_attributes': ['prob_infected'],
                    'seed': 42,
                }
            )
    
    def test_mismatched_lengths_raise_error(self):
        """Test that mismatched generator/proportion lengths raise ValueError."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        with pytest.raises(ValueError, match="must match"):
            PatientGeneratorMixer(
                config={
                    'generators': [gen1, gen2],
                    'proportions': [0.5, 0.3, 0.2],  # 3 proportions, 2 generators
                    'visible_patient_attributes': ['prob_infected'],
                    'seed': 42,
                }
            )
    
    def test_empty_generators_raise_error(self):
        """Test that empty generator list raises ValueError."""
        with pytest.raises(ValueError, match="at least one generator"):
            PatientGeneratorMixer(
                config={
                    'generators': [],
                    'proportions': [],
                    'visible_patient_attributes': ['prob_infected'],
                    'seed': 42,
                }
            )


class TestPatientGeneratorMixerSampling:
    """Test PatientGeneratorMixer sampling functionality."""
    
    def test_sample_returns_correct_total_count(self):
        """Test that sample returns the requested number of patients."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.8, 0.2],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=100, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        assert len(patients) == 100
    
    def test_sample_approximate_proportions(self):
        """Test that sampled patients approximately follow specified proportions."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.7, 0.3],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        # Sample large cohort to check proportions
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=1000, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        # Count patients by infection probability (low-risk ~0.3, high-risk ~0.8)
        low_risk_count = sum(1 for p in patients if p.prob_infected < 0.5)
        high_risk_count = sum(1 for p in patients if p.prob_infected >= 0.5)
        
        # Should be approximately 700 low-risk, 300 high-risk
        assert 650 <= low_risk_count <= 750, f"Expected ~700 low-risk, got {low_risk_count}"
        assert 250 <= high_risk_count <= 350, f"Expected ~300 high-risk, got {high_risk_count}"
    
    def test_sample_handles_rounding_correctly(self):
        """Test that sample handles rounding to ensure exact total count."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.333, 0.667],  # Will require rounding
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        # Try various sample sizes
        for n in [10, 50, 99, 101]:
            rng = np.random.default_rng(42)
            patients = mixer.sample(n_patients=n, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
            assert len(patients) == n, f"Expected {n} patients, got {len(patients)}"
    
    def test_sample_shuffles_results(self):
        """Test that patients are shuffled (not all low-risk first, then high-risk)."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=100, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        # Check that patients are mixed (not all low-risk first)
        # Low-risk should have prob_infected around 0.3, high-risk around 0.8
        first_half_avg = np.mean([p.prob_infected for p in patients[:50]])
        second_half_avg = np.mean([p.prob_infected for p in patients[50:]])
        
        # If not shuffled, one half would be all ~0.3, other all ~0.8
        # With shuffling, both halves should be around 0.55 (average of 0.3 and 0.8)
        assert 0.4 < first_half_avg < 0.7, f"First half avg: {first_half_avg} (should be mixed)"
        assert 0.4 < second_half_avg < 0.7, f"Second half avg: {second_half_avg} (should be mixed)"
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces reproducible results."""
        gen1_a = create_low_risk_generator()
        gen2_a = create_high_risk_generator()
        mixer_a = PatientGeneratorMixer(
            config={
                'generators': [gen1_a, gen2_a],
                'proportions': [0.6, 0.4],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 123,
            }
        )
        
        gen1_b = create_low_risk_generator()
        gen2_b = create_high_risk_generator()
        mixer_b = PatientGeneratorMixer(
            config={
                'generators': [gen1_b, gen2_b],
                'proportions': [0.6, 0.4],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 123,
            }
        )
        
        rng_a = np.random.default_rng(123)
        patients_a = mixer_a.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng_a)
        rng_b = np.random.default_rng(123)
        patients_b = mixer_b.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng_b)
        
        # Check that both cohorts are identical
        for pa, pb in zip(patients_a, patients_b):
            assert np.isclose(pa.prob_infected, pb.prob_infected)
            assert np.isclose(pa.benefit_value_multiplier, pb.benefit_value_multiplier)
            assert np.isclose(pa.failure_value_multiplier, pb.failure_value_multiplier)
    
    def test_sample_with_three_generators(self):
        """Test mixing with three generators."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        gen3 = create_low_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2, gen3],
                'proportions': [0.5, 0.3, 0.2],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=100, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        assert len(patients) == 100
    
    def test_sample_with_extreme_proportions(self):
        """Test with extreme proportions (99% / 1%)."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.99, 0.01],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=100, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        assert len(patients) == 100
        low_risk_count = sum(1 for p in patients if p.prob_infected < 0.5)
        assert 95 <= low_risk_count <= 100  # Allow some variance
    
    def test_sample_zero_proportion_handled_correctly(self):
        """Test that zero proportion generators are not sampled."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [1.0, 0.0],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        assert len(patients) == 50
        
        # All patients should be low-risk (prob_infected around 0.3)
        avg_prob_infected = np.mean([p.prob_infected for p in patients])
        assert 0.25 < avg_prob_infected < 0.35


class TestPatientGeneratorMixerIntegration:
    """Test integration with ABXAMREnv (verify duck-typing compatibility)."""
    
    def test_mixer_has_sample_method(self):
        """Test that mixer has the required sample() method."""
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        assert hasattr(mixer, 'sample')
        assert callable(mixer.sample)
    
    def test_mixer_can_be_used_like_generator(self):
        """Test that mixer can be used interchangeably with PatientGenerator."""
        # This test verifies duck-typing compatibility
        gen1 = create_low_risk_generator()
        gen2 = create_high_risk_generator()
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'visible_patient_attributes': ['prob_infected'],
                'seed': 42,
            }
        )
        
        # Should be able to call sample() just like PatientGenerator
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=20, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        # Should return list of Patient instances
        assert isinstance(patients, list)
        assert len(patients) == 20
        assert all(hasattr(p, 'prob_infected') for p in patients)
        assert all(hasattr(p, 'benefit_value_multiplier') for p in patients)


class TestPatientGeneratorMixerHeterogeneousVisibility:
    """Test PatientGeneratorMixer with heterogeneous visibility configurations.
    
    These tests verify the padding behavior when subordinate generators have
    different sets of visible_patient_attributes.
    """
    
    def test_automatic_detection_of_uniform_visibility(self):
        """Test that mixer correctly detects when all generators have same visibility."""
        gen1 = create_low_risk_generator()  # visibility: ['prob_infected']
        gen2 = create_high_risk_generator()  # visibility: ['prob_infected']
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        # Should detect uniform visibility and use simple mode
        assert mixer.visible_patient_attributes == ['prob_infected']
        assert not hasattr(mixer, '_heterogeneous_visibility') or not mixer._heterogeneous_visibility
    
    def test_automatic_detection_of_heterogeneous_visibility(self):
        """Test that mixer correctly detects when generators have different visibility."""
        # Generator with minimal visibility
        gen1_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
        gen1 = PatientGenerator(config=gen1_config)
        
        # Generator with full visibility
        gen2_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.8},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.05},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': [
                'prob_infected',
                'benefit_value_multiplier',
                'failure_value_multiplier',
                'benefit_probability_multiplier',
            ],
        }
        gen2 = PatientGenerator(config=gen2_config)
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        # Should compute union of visible attributes
        expected_union = sorted([
            'benefit_probability_multiplier',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'prob_infected',
        ])
        assert mixer.visible_patient_attributes == expected_union
    
    def test_union_computation_with_two_generators(self):
        """Test that visibility union is correctly computed with 2 generators."""
        gen1_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected', 'benefit_value_multiplier'],
        }
        gen1 = PatientGenerator(config=gen1_config)
        
        gen2_config = gen1_config.copy()
        gen2_config['visible_patient_attributes'] = ['prob_infected', 'failure_value_multiplier']
        gen2 = PatientGenerator(config=gen2_config)
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        # Union should be sorted alphabetically
        expected_union = ['benefit_value_multiplier', 'failure_value_multiplier', 'prob_infected']
        assert mixer.visible_patient_attributes == expected_union
    
    def test_union_computation_with_three_generators(self):
        """Test that visibility union is correctly computed with 3+ generators."""
        base_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
        }
        
        gen1_config = base_config.copy()
        gen1_config['visible_patient_attributes'] = ['prob_infected']
        gen1 = PatientGenerator(config=gen1_config)
        
        gen2_config = base_config.copy()
        gen2_config['visible_patient_attributes'] = ['benefit_value_multiplier', 'failure_value_multiplier']
        gen2 = PatientGenerator(config=gen2_config)
        
        gen3_config = base_config.copy()
        gen3_config['visible_patient_attributes'] = ['benefit_probability_multiplier', 'prob_infected']
        gen3 = PatientGenerator(config=gen3_config)
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2, gen3],
                'proportions': [0.33, 0.33, 0.34],
                'seed': 42,
            }
        )
        
        # Union of all three sets
        expected_union = [
            'benefit_probability_multiplier',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'prob_infected',
        ]
        assert mixer.visible_patient_attributes == expected_union
    
    def test_observation_shape_consistency_heterogeneous(self):
        """Test that all patients produce same observation shape with heterogeneous visibility."""
        gen1_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],  # 1 attribute
        }
        gen1 = PatientGenerator(config=gen1_config)
        
        gen2_config = gen1_config.copy()
        gen2_config['prob_infected'] = {
            'prob_dist': {'type': 'constant', 'value': 0.8},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        }
        gen2_config['visible_patient_attributes'] = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
        ]  # 3 attributes
        gen2 = PatientGenerator(config=gen2_config)
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=20, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        # Get observations for all patients
        obs = mixer.observe(patients=patients)
        
        # All observations should be flattened: (num_patients * num_attrs,) = (20 * 3,) = (60,)
        expected_shape = (20 * 3,)
        assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    
    def test_padding_values_are_correct(self):
        """Test that padding values are correctly placed for non-visible attributes."""
        gen1_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],  # Only prob_infected visible
        }
        gen1 = PatientGenerator(config=gen1_config)
        
        gen2_config = gen1_config.copy()
        gen2_config['prob_infected'] = {
            'prob_dist': {'type': 'constant', 'value': 0.8},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        }
        gen2_config['benefit_value_multiplier'] = {
            'prob_dist': {'type': 'constant', 'value': 1.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        }
        gen2_config['visible_patient_attributes'] = ['prob_infected', 'benefit_value_multiplier']
        gen2 = PatientGenerator(config=gen2_config)
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        # Visibility union: ['benefit_value_multiplier', 'prob_infected']
        assert mixer.visible_patient_attributes == ['benefit_value_multiplier', 'prob_infected']
        
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=20, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        obs = mixer.observe(patients=patients)
        
        # Observation is flattened: [p0_attr0, p0_attr1, p1_attr0, p1_attr1, ...]
        # With 2 attributes: [p0_benefit_value, p0_prob_infected, p1_benefit_value, p1_prob_infected, ...]
        num_attrs = 2
        
        # Check patients from gen1 (low prob_infected ~0.3)
        # These should have PADDING_VALUE for benefit_value_multiplier
        for i, patient in enumerate(patients):
            base_idx = i * num_attrs
            if np.isclose(patient.prob_infected, 0.3, atol=0.05):
                # This is from gen1: should have padding for benefit_value_multiplier
                assert np.isclose(obs[base_idx], PatientGeneratorMixer.PADDING_VALUE), \
                    f"Patient {i} from gen1 should have PADDING_VALUE for benefit_value_multiplier, got {obs[base_idx]}"
                assert np.isclose(obs[base_idx + 1], 0.3, atol=0.05), \
                    f"Patient {i} from gen1 should have prob_infected ~0.3, got {obs[base_idx + 1]}"
            
            # Check patients from gen2 (high prob_infected ~0.8)
            if np.isclose(patient.prob_infected, 0.8, atol=0.05):
                # This is from gen2: should have actual benefit_value_multiplier
                assert np.isclose(obs[base_idx], 1.5, atol=0.05), \
                    f"Patient {i} from gen2 should have benefit_value_multiplier ~1.5, got {obs[base_idx]}"
                assert np.isclose(obs[base_idx + 1], 0.8, atol=0.05), \
                    f"Patient {i} from gen2 should have prob_infected ~0.8, got {obs[base_idx + 1]}"
    
    def test_obs_dim_returns_union_dimension(self):
        """Test that obs_dim() returns dimension based on visibility union."""
        gen1_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],  # 1 attribute
        }
        gen1 = PatientGenerator(config=gen1_config)
        
        gen2_config = gen1_config.copy()
        gen2_config['visible_patient_attributes'] = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
        ]  # 4 attributes
        gen2 = PatientGenerator(config=gen2_config)
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        # Union has 4 unique attributes
        assert len(mixer.visible_patient_attributes) == 4
        
        # obs_dim should return 4 * num_patients
        assert mixer.obs_dim(num_patients=10) == 40
        assert mixer.obs_dim(num_patients=1) == 4
        assert mixer.obs_dim(num_patients=100) == 400
    
    def test_backward_compatibility_uniform_visibility(self):
        """Test that uniform visibility produces same results as before (no padding)."""
        gen1 = create_low_risk_generator()  # visibility: ['prob_infected']
        gen2 = create_high_risk_generator()  # visibility: ['prob_infected']
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=10, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        obs = mixer.observe(patients=patients)
        
        # Should have shape (10,) - flattened, one attribute per patient
        assert obs.shape == (10,)
        
        # No padding should be present (all values in valid range [0, 1])
        assert np.all(obs >= 0) and np.all(obs <= 1)
        assert not np.any(np.isclose(obs, PatientGeneratorMixer.PADDING_VALUE))
    
    def test_integration_with_environment(self):
        """Test that heterogeneous mixer can be used with ABXAMREnv without errors."""
        from abx_amr_simulator.core import ABXAMREnv, RewardCalculator
        
        # Create generators with different visibility
        gen1_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
        gen1 = PatientGenerator(config=gen1_config)
        
        gen2_config = gen1_config.copy()
        gen2_config['prob_infected'] = {
            'prob_dist': {'type': 'constant', 'value': 0.8},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        }
        gen2_config['visible_patient_attributes'] = ['prob_infected', 'benefit_value_multiplier']
        gen2 = PatientGenerator(config=gen2_config)
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        # Create reward calculator
        reward_calc = RewardCalculator(
            config={
                'abx_clinical_reward_penalties_info_dict': {
                    'clinical_benefit_reward': 10.0,
                    'clinical_benefit_probability': 0.9,
                    'clinical_failure_penalty': -5.0,
                    'clinical_failure_probability': 0.1,
                    'abx_adverse_effects_info': {
                        'A': {
                            'adverse_effect_penalty': -2.0,
                            'adverse_effect_probability': 0.05,
                        }
                    },
                },
                'lambda_weight': 0.5,
                'epsilon': 0.05,
                'seed': 42,
            }
        )
        
        # Create environment with heterogeneous mixer (explicit kwargs API)
        antibiotics_AMR_dict = {
            'A': {
                'leak': 0.05,
                'flatness_parameter': 1.0,
                'permanent_residual_volume': 0.0,
                'initial_amr_level': 0.0,
            }
        }
        env = ABXAMREnv(
            reward_calculator=reward_calc,
            patient_generator=mixer,
            antibiotics_AMR_dict=antibiotics_AMR_dict,
            num_patients_per_time_step=5,
            max_time_steps=10,
            include_steps_since_amr_update_in_obs=False,
        )
        
        # Environment should initialize without errors
        obs, info = env.reset()
        
        # Observation shape should be correct: (union_size * num_patients + num_antibiotics,)
        # Union size = 2 (benefit_value_multiplier, prob_infected - alphabetically sorted)
        # num_patients = 5, num_antibiotics = 1
        expected_obs_dim = 2 * 5 + 1  # 11 dimensions total
        assert obs.shape == (expected_obs_dim,), f"Expected obs shape ({expected_obs_dim},), got {obs.shape}"
        
        # Should be able to step without errors
        action = np.array([0, 0, 0, 0, 0])  # Don't prescribe to anyone
        obs, reward, terminated, truncated, info = env.step(action=action)
        
        assert obs.shape == (expected_obs_dim,)
        assert isinstance(reward, (float, np.floating))
    
    def test_padding_with_all_six_attributes(self):
        """Test padding behavior when using all 6 patient attributes."""
        # Generator with minimal visibility
        gen1_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected', 'recovery_without_treatment_prob'],
        }
        gen1 = PatientGenerator(config=gen1_config)
        
        # Generator with full visibility (all 6 attributes)
        gen2_config = gen1_config.copy()
        gen2_config['prob_infected'] = {
            'prob_dist': {'type': 'constant', 'value': 0.8},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        }
        gen2_config['visible_patient_attributes'] = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob',
        ]
        gen2 = PatientGenerator(config=gen2_config)
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen1, gen2],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        # Union should have all 6 attributes
        assert len(mixer.visible_patient_attributes) == 6
        
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=20, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        obs = mixer.observe(patients=patients)
        
        # Check shape: flattened (20 patients * 6 attributes,) = (120,)
        assert obs.shape == (120,)
        
        # Check that gen1 patients have padding for 4 attributes
        num_attrs = 6
        for i, patient in enumerate(patients):
            if np.isclose(patient.prob_infected, 0.3, atol=0.05):
                # From gen1: should have 4 padded values and 2 real values
                base_idx = i * num_attrs
                patient_obs = obs[base_idx:base_idx + num_attrs]
                padding_count = np.sum(np.isclose(patient_obs, PatientGeneratorMixer.PADDING_VALUE))
                assert padding_count == 4, \
                    f"Patient {i} from gen1 should have 4 padded attributes, got {padding_count}"
