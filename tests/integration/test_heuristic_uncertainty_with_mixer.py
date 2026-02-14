"""Integration tests for HeuristicWorker uncertainty scoring with PatientGeneratorMixer.

These tests validate the end-to-end workflow of:
1. PatientGeneratorMixer with heterogeneous visibility
2. Sampling patients from mixed populations
3. Computing uncertainty scores via HeuristicWorker

This ensures that uncertainty-based heuristic strategies can correctly distinguish
between patient subpopulations with different levels of attribute visibility.
"""

import numpy as np
import pytest

from abx_amr_simulator.core.patient_generator import PatientGenerator, PatientGeneratorMixer
from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker,
)


def _create_full_visibility_generator(risk_level: str) -> PatientGenerator:
    """Create a PatientGenerator with all 6 attributes visible.
    
    Args:
        risk_level: 'low' or 'high' to set infection probability
    
    Returns:
        PatientGenerator with full visibility (6 attributes)
    """
    prob_infected_value = 0.3 if risk_level == 'low' else 0.8
    
    config = {
        'prob_infected': {
            'prob_dist': {'type': 'constant', 'value': prob_infected_value},
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
        'visible_patient_attributes': [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob',
        ],
    }
    return PatientGenerator(config=config)


def _create_limited_visibility_generator(risk_level: str) -> PatientGenerator:
    """Create a PatientGenerator with only 2 attributes visible.
    
    Args:
        risk_level: 'low' or 'high' to set infection probability
    
    Returns:
        PatientGenerator with limited visibility (2 attributes)
    """
    prob_infected_value = 0.3 if risk_level == 'low' else 0.8
    
    config = {
        'prob_infected': {
            'prob_dist': {'type': 'constant', 'value': prob_infected_value},
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
        'visible_patient_attributes': [
            'prob_infected',
            'recovery_without_treatment_prob',
        ],
    }
    return PatientGenerator(config=config)


def _extract_patient_dicts_from_observation(obs_array: np.ndarray, union_attrs: list, num_patients: int) -> list:
    """Extract per-patient dictionaries from flattened observation array.
    
    The mixer's observe() method returns a flattened array with padding already applied.
    This function reshapes it back into per-patient dictionaries.
    
    Args:
        obs_array: Flattened observation array from mixer.observe()
        union_attrs: List of attribute names in order (visibility union)
        num_patients: Number of patients
    
    Returns:
        List of dictionaries, one per patient, mapping attribute names to observed values
    """
    num_attrs = len(union_attrs)
    patient_dicts = []
    
    for i in range(num_patients):
        patient_dict = {}
        for j, attr in enumerate(union_attrs):
            idx = i * num_attrs + j
            patient_dict[attr] = obs_array[idx]
        patient_dicts.append(patient_dict)
    
    return patient_dicts


class TestUncertaintyWithUniformVisibility:
    """Test uncertainty scoring when all generators have same visibility (like Experiment Set #3)."""
    
    def test_all_patients_have_zero_uncertainty_with_full_visibility(self):
        """When all subpopulations have full visibility, all patients should have uncertainty = 0."""
        # Create two generators with same full visibility (6 attributes)
        gen_low = _create_full_visibility_generator(risk_level='low')
        gen_high = _create_full_visibility_generator(risk_level='high')
        
        # Mix them
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen_low, gen_high],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        # Sample patients
        rng = np.random.default_rng(seed=42)
        patients = mixer.sample(
            n_patients=20,
            true_amr_levels={'A': 0.0},
            rng=rng,
        )
        
        # Create HeuristicWorker
        worker = HeuristicWorker(
            name='test_worker',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        # Inject the full observable attributes list
        worker.set_observable_attributes(attributes=mixer.visible_patient_attributes)
        
        # Get padded observations from mixer
        obs_array = mixer.observe(patients=patients)
        patient_dicts = _extract_patient_dicts_from_observation(
            obs_array=obs_array,
            union_attrs=mixer.visible_patient_attributes,
            num_patients=len(patients),
        )
        
        # Check uncertainty for all patients
        for patient_dict in patient_dicts:
            # Both relative and absolute uncertainty should be 0
            rel_uncertainty = worker.compute_relative_uncertainty_score(patient=patient_dict)
            abs_uncertainty = worker.compute_absolute_uncertainty_score(
                patient=patient_dict,
                total_observable_attrs=len(mixer.visible_patient_attributes),
            )
            
            assert rel_uncertainty == 0, \
                f"Expected relative uncertainty 0, got {rel_uncertainty} for patient {patient_dict}"
            assert abs_uncertainty == 0, \
                f"Expected absolute uncertainty 0, got {abs_uncertainty} for patient {patient_dict}"


class TestUncertaintyWithHeterogeneousVisibility:
    """Test uncertainty scoring when generators have different visibility (like Experiment Set #4)."""
    
    def test_patients_distinguished_by_visibility(self):
        """Patients from different visibility subpopulations should have different uncertainty scores."""
        # Create generators with different visibility
        gen_full = _create_full_visibility_generator(risk_level='high')  # 6 attributes
        gen_limited = _create_limited_visibility_generator(risk_level='low')  # 2 attributes
        
        # Mix them
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen_full, gen_limited],
                'proportions': [0.5, 0.5],
                'seed': 42,
            }
        )
        
        # Visibility union should have all 6 attributes
        assert len(mixer.visible_patient_attributes) == 6
        
        # Sample patients
        rng = np.random.default_rng(seed=42)
        patients = mixer.sample(
            n_patients=20,
            true_amr_levels={'A': 0.0},
            rng=rng,
        )
        
        # Create HeuristicWorker
        worker = HeuristicWorker(
            name='test_worker',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        
        # Inject the full observable attributes list (union of both generators)
        worker.set_observable_attributes(attributes=mixer.visible_patient_attributes)
        
        # Get padded observations from mixer
        obs_array = mixer.observe(patients=patients)
        patient_dicts = _extract_patient_dicts_from_observation(
            obs_array=obs_array,
            union_attrs=mixer.visible_patient_attributes,
            num_patients=len(patients),
        )
        
        # Track uncertainty by subpopulation
        full_vis_uncertainties_rel = []
        full_vis_uncertainties_abs = []
        limited_vis_uncertainties_rel = []
        limited_vis_uncertainties_abs = []
        
        for i, patient_dict in enumerate(patient_dicts):
            rel_uncertainty = worker.compute_relative_uncertainty_score(patient=patient_dict)
            abs_uncertainty = worker.compute_absolute_uncertainty_score(
                patient=patient_dict,
                total_observable_attrs=len(mixer.visible_patient_attributes),
            )
            
            # Classify by infection probability (proxy for generator source)
            patient_obj = patients[i]
            if patient_obj.prob_infected > 0.5:
                # High-risk, full visibility generator
                full_vis_uncertainties_rel.append(rel_uncertainty)
                full_vis_uncertainties_abs.append(abs_uncertainty)
            else:
                # Low-risk, limited visibility generator
                limited_vis_uncertainties_rel.append(rel_uncertainty)
                limited_vis_uncertainties_abs.append(abs_uncertainty)
        
        # Verify both subpopulations are represented
        assert len(full_vis_uncertainties_rel) > 0, "No patients from full-visibility generator"
        assert len(limited_vis_uncertainties_rel) > 0, "No patients from limited-visibility generator"
        
        # Full visibility patients should have uncertainty = 0
        assert all(u == 0 for u in full_vis_uncertainties_rel), \
            f"Full visibility patients should have relative uncertainty 0, got {full_vis_uncertainties_rel}"
        assert all(u == 0 for u in full_vis_uncertainties_abs), \
            f"Full visibility patients should have absolute uncertainty 0, got {full_vis_uncertainties_abs}"
        
        # Limited visibility patients should have uncertainty = 4 (missing 4 of 6 attributes)
        assert all(u == 4 for u in limited_vis_uncertainties_rel), \
            f"Limited visibility patients should have relative uncertainty 4, got {limited_vis_uncertainties_rel}"
        assert all(u == 4 for u in limited_vis_uncertainties_abs), \
            f"Limited visibility patients should have absolute uncertainty 4, got {limited_vis_uncertainties_abs}"
    
    def test_relative_and_absolute_uncertainty_equivalent_with_padding(self):
        """With mixer padding, relative and absolute uncertainty should give same results."""
        gen_full = _create_full_visibility_generator(risk_level='high')  # 6 attributes
        gen_limited = _create_limited_visibility_generator(risk_level='low')  # 2 attributes
        
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen_full, gen_limited],
                'proportions': [0.5, 0.5],
                'seed': 123,
            }
        )
        
        rng = np.random.default_rng(seed=123)
        patients = mixer.sample(
            n_patients=50,
            true_amr_levels={'A': 0.0},
            rng=rng,
        )
        
        worker = HeuristicWorker(
            name='test_worker',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        worker.set_observable_attributes(attributes=mixer.visible_patient_attributes)
        
        # Get padded observations from mixer
        obs_array = mixer.observe(patients=patients)
        patient_dicts = _extract_patient_dicts_from_observation(
            obs_array=obs_array,
            union_attrs=mixer.visible_patient_attributes,
            num_patients=len(patients),
        )
        
        for patient_dict in patient_dicts:
            rel_uncertainty = worker.compute_relative_uncertainty_score(patient=patient_dict)
            abs_uncertainty = worker.compute_absolute_uncertainty_score(
                patient=patient_dict,
                total_observable_attrs=len(mixer.visible_patient_attributes),
            )
            
            # With mixer padding, both metrics should give identical results
            assert rel_uncertainty == abs_uncertainty, \
                f"Relative ({rel_uncertainty}) and absolute ({abs_uncertainty}) " \
                f"uncertainty should match with mixer padding for patient {patient_dict}"
    
    def test_uncertainty_with_three_visibility_levels(self):
        """Test uncertainty scoring with three different visibility levels."""
        # Create generators with 6, 4, and 2 visible attributes
        gen_full = _create_full_visibility_generator(risk_level='high')  # 6 attributes
        
        # Medium visibility: 4 attributes
        gen_medium_config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.55},
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
            'visible_patient_attributes': [
                'prob_infected',
                'benefit_value_multiplier',
                'failure_value_multiplier',
                'recovery_without_treatment_prob',
            ],
        }
        gen_medium = PatientGenerator(config=gen_medium_config)
        
        gen_limited = _create_limited_visibility_generator(risk_level='low')  # 2 attributes
        
        # Mix all three
        mixer = PatientGeneratorMixer(
            config={
                'generators': [gen_full, gen_medium, gen_limited],
                'proportions': [0.33, 0.34, 0.33],
                'seed': 99,
            }
        )
        
        assert len(mixer.visible_patient_attributes) == 6
        
        rng = np.random.default_rng(seed=99)
        patients = mixer.sample(
            n_patients=30,
            true_amr_levels={'A': 0.0},
            rng=rng,
        )
        
        worker = HeuristicWorker(
            name='test_worker',
            duration=10,
            action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
            uncertainty_threshold=2.0,
        )
        worker.set_observable_attributes(attributes=mixer.visible_patient_attributes)
        
        # Get padded observations from mixer
        obs_array = mixer.observe(patients=patients)
        patient_dicts = _extract_patient_dicts_from_observation(
            obs_array=obs_array,
            union_attrs=mixer.visible_patient_attributes,
            num_patients=len(patients),
        )
        
        # Classify patients by uncertainty levels
        uncertainty_0_count = 0  # Full visibility (6 attrs)
        uncertainty_2_count = 0  # Medium visibility (4 attrs → 2 missing)
        uncertainty_4_count = 0  # Limited visibility (2 attrs → 4 missing)
        
        for patient_dict in patient_dicts:
            uncertainty = worker.compute_relative_uncertainty_score(patient=patient_dict)
            
            if uncertainty == 0:
                uncertainty_0_count += 1
            elif uncertainty == 2:
                uncertainty_2_count += 1
            elif uncertainty == 4:
                uncertainty_4_count += 1
            else:
                pytest.fail(f"Unexpected uncertainty value: {uncertainty}")
        
        # All three levels should be present
        assert uncertainty_0_count > 0, "Should have some full-visibility patients"
        assert uncertainty_2_count > 0, "Should have some medium-visibility patients"
        assert uncertainty_4_count > 0, "Should have some limited-visibility patients"
        
        # Roughly equal proportions (allowing for randomness)
        total = uncertainty_0_count + uncertainty_2_count + uncertainty_4_count
        assert total == 30
        assert 5 <= uncertainty_0_count <= 15, f"Expected ~10 full-vis patients, got {uncertainty_0_count}"
        assert 5 <= uncertainty_2_count <= 15, f"Expected ~10 medium-vis patients, got {uncertainty_2_count}"
        assert 5 <= uncertainty_4_count <= 15, f"Expected ~10 limited-vis patients, got {uncertainty_4_count}"
