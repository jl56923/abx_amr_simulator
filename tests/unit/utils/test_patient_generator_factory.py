"""
Tests for create_patient_generator() factory function.

Tests mixer behavior where visible_patient_attributes is automatically
derived from sub-generators to prevent configuration mismatches.
"""

import tempfile
import yaml
from pathlib import Path
import pytest

from abx_amr_simulator.utils import create_patient_generator
from abx_amr_simulator.core import PatientGeneratorMixer


class TestPatientGeneratorMixerVisibility:
    """Tests for automatic visible_patient_attributes derivation in mixers."""
    
    def test_mixer_derives_union_of_visible_attributes(self):
        """Test that mixer automatically computes union of sub-generator visible attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create first sub-generator config (6 attributes - full visibility)
            sub_gen1_config = {
                'prob_infected': {
                    'prob_dist': {'type': 'constant', 'value': 0.8},
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
                    'failure_probability_multiplier',
                    'recovery_without_treatment_prob',
                ]
            }
            sub_gen1_path = tmpdir_path / "sub_gen1.yaml"
            with open(sub_gen1_path, 'w') as f:
                yaml.dump(sub_gen1_config, f)
            
            # Create second sub-generator config (2 attributes - limited visibility)
            sub_gen2_config = {
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
                    'prob_dist': {'type': 'constant', 'value': 0.05},
                    'obs_bias_multiplier': 1.0,
                    'obs_noise_one_std_dev': 0.0,
                    'obs_noise_std_dev_fraction': 0.0,
                    'clipping_bounds': [0.0, 1.0],
                },
                'visible_patient_attributes': [
                    'prob_infected',
                    'recovery_without_treatment_prob',
                ]
            }
            sub_gen2_path = tmpdir_path / "sub_gen2.yaml"
            with open(sub_gen2_path, 'w') as f:
                yaml.dump(sub_gen2_config, f)
            
            # Create mixer config (no visible_patient_attributes - should be auto-derived)
            config = {
                'patient_generator': {
                    'type': 'mixer',
                    'generators': [
                        {
                            'config_file': str(sub_gen1_path),
                            'proportion': 0.5,
                        },
                        {
                            'config_file': str(sub_gen2_path),
                            'proportion': 0.5,
                        },
                    ],
                },
                'training': {
                    'seed': 42,
                },
            }
            
            # Create patient generator via factory
            pg = create_patient_generator(config=config)
            
            # Verify it's a mixer
            assert isinstance(pg, PatientGeneratorMixer)
            
            # Verify visible_patient_attributes is the union of both sub-generators
            expected_attrs = {
                'prob_infected',
                'benefit_value_multiplier',
                'failure_value_multiplier',
                'benefit_probability_multiplier',
                'failure_probability_multiplier',
                'recovery_without_treatment_prob',
            }
            assert set(pg.visible_patient_attributes) == expected_attrs
    
    def test_mixer_validates_sub_generators_have_visibility(self):
        """Test that mixer raises error if sub-generator missing visible_patient_attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create sub-generator config WITHOUT visible_patient_attributes
            sub_gen_config = {
                'prob_infected': {
                    'prob_dist': {'type': 'constant', 'value': 0.8},
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
                    'prob_dist': {'type': 'constant', 'value': 0.05},
                    'obs_bias_multiplier': 1.0,
                    'obs_noise_one_std_dev': 0.0,
                    'obs_noise_std_dev_fraction': 0.0,
                    'clipping_bounds': [0.0, 1.0],
                },
                # Missing visible_patient_attributes!
            }
            sub_gen_path = tmpdir_path / "sub_gen.yaml"
            with open(sub_gen_path, 'w') as f:
                yaml.dump(sub_gen_config, f)
            
            # Create mixer config
            config = {
                'patient_generator': {
                    'type': 'mixer',
                    'generators': [
                        {
                            'config_file': str(sub_gen_path),
                            'proportion': 1.0,
                        },
                    ],
                },
                'training': {
                    'seed': 42,
                },
            }
            
            # Should raise ValueError about missing visible_patient_attributes
            with pytest.raises(ValueError, match="missing 'visible_patient_attributes'"):
                create_patient_generator(config=config)
    
    def test_mixer_removes_duplicate_attributes(self):
        """Test that mixer removes duplicate visible attributes while preserving order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Both sub-generators have overlapping attributes
            sub_gen1_config = {
                'prob_infected': {
                    'prob_dist': {'type': 'constant', 'value': 0.8},
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
                    'prob_dist': {'type': 'constant', 'value': 0.05},
                    'obs_bias_multiplier': 1.0,
                    'obs_noise_one_std_dev': 0.0,
                    'obs_noise_std_dev_fraction': 0.0,
                    'clipping_bounds': [0.0, 1.0],
                },
                'visible_patient_attributes': [
                    'prob_infected',
                    'benefit_value_multiplier',
                    'recovery_without_treatment_prob',
                ]
            }
            sub_gen1_path = tmpdir_path / "sub_gen1.yaml"
            with open(sub_gen1_path, 'w') as f:
                yaml.dump(sub_gen1_config, f)
            
            sub_gen2_config = {
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
                    'prob_dist': {'type': 'constant', 'value': 0.05},
                    'obs_bias_multiplier': 1.0,
                    'obs_noise_one_std_dev': 0.0,
                    'obs_noise_std_dev_fraction': 0.0,
                    'clipping_bounds': [0.0, 1.0],
                },
                'visible_patient_attributes': [
                    'prob_infected',  # duplicate
                    'failure_value_multiplier',  # new
                    'recovery_without_treatment_prob',  # duplicate
                ]
            }
            sub_gen2_path = tmpdir_path / "sub_gen2.yaml"
            with open(sub_gen2_path, 'w') as f:
                yaml.dump(sub_gen2_config, f)
            
            # Create mixer config
            config = {
                'patient_generator': {
                    'type': 'mixer',
                    'generators': [
                        {
                            'config_file': str(sub_gen1_path),
                            'proportion': 0.5,
                        },
                        {
                            'config_file': str(sub_gen2_path),
                            'proportion': 0.5,
                        },
                    ],
                },
                'training': {
                    'seed': 42,
                },
            }
            
            # Create patient generator via factory
            pg = create_patient_generator(config=config)
            
            # Verify duplicates removed
            expected_attrs = {
                'prob_infected',
                'benefit_value_multiplier',
                'recovery_without_treatment_prob',
                'failure_value_multiplier',  # Added from second generator
            }
            assert set(pg.visible_patient_attributes) == expected_attrs
    
    def test_non_mixer_requires_explicit_visibility(self):
        """Test that non-mixer generators still require explicit visible_patient_attributes."""
        config = {
            'patient_generator': {
                'prob_infected': {
                    'prob_dist': {'type': 'constant', 'value': 0.8},
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
                    'prob_dist': {'type': 'constant', 'value': 0.05},
                    'obs_bias_multiplier': 1.0,
                    'obs_noise_one_std_dev': 0.0,
                    'obs_noise_std_dev_fraction': 0.0,
                    'clipping_bounds': [0.0, 1.0],
                },
                # Missing visible_patient_attributes!
            },
            'training': {
                'seed': 42,
            },
        }
        
        # Should raise ValueError about missing visible_patient_attributes
        with pytest.raises(ValueError, match="'visible_patient_attributes' must be specified"):
            create_patient_generator(config=config)
