"""
Unit tests for GUI Phase 1 changes: Patient Generator config format migration and validation.
"""

import pytest
from abx_amr_simulator.gui.patient_gen_ui_helper import migrate_old_config_to_new


class TestConfigMigration:
    """Tests for old flat config → new nested config migration."""
    
    def test_new_format_passthrough(self):
        """Already-new configs should pass through unchanged."""
        new_cfg = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_std_dev_fraction': 0.1,
                'obs_noise_one_std_dev': 0.2,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
        
        migrated = migrate_old_config_to_new(new_cfg)
        assert migrated == new_cfg
    
    def test_old_flat_config_to_nested(self):
        """Old flat format should be converted to nested per-attribute."""
        old_cfg = {
            'prob_infected_dist': {'type': 'constant', 'value': 0.5},
            'prob_infected_observation_bias': 1.2,
            'prob_infected_observation_noise': 0.05,
            'benefit_value_multiplier_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
            'benefit_value_multiplier_observation_bias': 1.0,
            'benefit_value_multiplier_observation_noise': 0.0,
            'failure_value_multiplier_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
            'failure_value_multiplier_observation_bias': 1.0,
            'failure_value_multiplier_observation_noise': 0.0,
            'benefit_probability_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'benefit_probability_multiplier_observation_bias': 1.0,
            'benefit_probability_multiplier_observation_noise': 0.0,
            'failure_probability_multiplier_dist': {'type': 'gaussian', 'mu': 0.5, 'sigma': 0.1},
            'failure_probability_multiplier_observation_bias': 1.0,
            'failure_probability_multiplier_observation_noise': 0.0,
            'recovery_without_treatment_prob_dist': {'type': 'constant', 'value': 0.01},
            'recovery_without_treatment_prob_observation_bias': 1.0,
            'recovery_without_treatment_prob_observation_noise': 0.0,
        }
        
        migrated = migrate_old_config_to_new(old_cfg)
        
        # Check structure
        assert 'prob_infected' in migrated
        assert 'benefit_value_multiplier' in migrated
        assert 'failure_value_multiplier' in migrated
        assert 'visible_patient_attributes' in migrated
        
        # Check prob_infected config
        prob_inf_cfg = migrated['prob_infected']
        assert prob_inf_cfg['prob_dist']['type'] == 'constant'
        assert prob_inf_cfg['prob_dist']['value'] == 0.5
        assert prob_inf_cfg['obs_bias_multiplier'] == 1.2
        assert 'obs_noise_std_dev_fraction' in prob_inf_cfg
        assert prob_inf_cfg['clipping_bounds'] == [0.0, 1.0]
    
    def test_noise_conversion_old_to_new(self):
        """Observation noise should be converted from absolute to fraction of range."""
        old_cfg = {
            'prob_infected_dist': {'type': 'constant', 'value': 0.5},
            'prob_infected_observation_noise': 0.1,  # Absolute std
            # ... other fields
            'benefit_value_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'benefit_value_multiplier_observation_noise': 0.5,
            'failure_value_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'failure_value_multiplier_observation_noise': 0.5,
            'benefit_probability_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'benefit_probability_multiplier_observation_noise': 0.5,
            'failure_probability_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'failure_probability_multiplier_observation_noise': 0.5,
            'recovery_without_treatment_prob_dist': {'type': 'constant', 'value': 0.01},
            'recovery_without_treatment_prob_observation_noise': 0.1,
            'prob_infected_observation_bias': 1.0,
            'benefit_value_multiplier_observation_bias': 1.0,
            'failure_value_multiplier_observation_bias': 1.0,
            'benefit_probability_multiplier_observation_bias': 1.0,
            'failure_probability_multiplier_observation_bias': 1.0,
            'recovery_without_treatment_prob_observation_bias': 1.0,
        }
        
        migrated = migrate_old_config_to_new(old_cfg)
        
        # prob_infected: old noise=0.1, reference std=0.2, so fraction = 0.1/0.2 = 0.5
        prob_inf_noise_frac = migrated['prob_infected']['obs_noise_std_dev_fraction']
        assert pytest.approx(prob_inf_noise_frac, rel=0.01) == 0.5
        
        # benefit_value_multiplier: old noise=0.5, reference std=1.0, so fraction = 0.5/1.0 = 0.5
        benefit_noise_frac = migrated['benefit_value_multiplier']['obs_noise_std_dev_fraction']
        assert pytest.approx(benefit_noise_frac, rel=0.01) == 0.5
    
    def test_preserves_visible_attributes(self):
        """visible_patient_attributes should be preserved if present."""
        old_cfg = {
            'prob_infected_dist': {'type': 'constant', 'value': 0.5},
            'visible_patient_attributes': ['prob_infected', 'benefit_value_multiplier', 'failure_value_multiplier'],
            'prob_infected_observation_bias': 1.0,
            'prob_infected_observation_noise': 0.0,
            'benefit_value_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'benefit_value_multiplier_observation_bias': 1.0,
            'benefit_value_multiplier_observation_noise': 0.0,
            'failure_value_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'failure_value_multiplier_observation_bias': 1.0,
            'failure_value_multiplier_observation_noise': 0.0,
            'benefit_probability_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'benefit_probability_multiplier_observation_bias': 1.0,
            'benefit_probability_multiplier_observation_noise': 0.0,
            'failure_probability_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'failure_probability_multiplier_observation_bias': 1.0,
            'failure_probability_multiplier_observation_noise': 0.0,
            'recovery_without_treatment_prob_dist': {'type': 'constant', 'value': 0.01},
            'recovery_without_treatment_prob_observation_bias': 1.0,
            'recovery_without_treatment_prob_observation_noise': 0.0,
        }
        
        migrated = migrate_old_config_to_new(old_cfg)
        
        assert migrated['visible_patient_attributes'] == ['prob_infected', 'benefit_value_multiplier', 'failure_value_multiplier']
    
    def test_all_six_attributes_present(self):
        """All 6 patient attributes should be present after migration."""
        old_cfg = {
            'prob_infected_dist': {'type': 'constant', 'value': 0.5},
            'prob_infected_observation_bias': 1.0,
            'prob_infected_observation_noise': 0.0,
            'benefit_value_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'benefit_value_multiplier_observation_bias': 1.0,
            'benefit_value_multiplier_observation_noise': 0.0,
            'failure_value_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'failure_value_multiplier_observation_bias': 1.0,
            'failure_value_multiplier_observation_noise': 0.0,
            'benefit_probability_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'benefit_probability_multiplier_observation_bias': 1.0,
            'benefit_probability_multiplier_observation_noise': 0.0,
            'failure_probability_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'failure_probability_multiplier_observation_bias': 1.0,
            'failure_probability_multiplier_observation_noise': 0.0,
            'recovery_without_treatment_prob_dist': {'type': 'constant', 'value': 0.01},
            'recovery_without_treatment_prob_observation_bias': 1.0,
            'recovery_without_treatment_prob_observation_noise': 0.0,
        }
        
        migrated = migrate_old_config_to_new(old_cfg)
        
        expected_attributes = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob',
        ]
        
        for attr in expected_attributes:
            assert attr in migrated
            assert 'prob_dist' in migrated[attr]
            assert 'obs_bias_multiplier' in migrated[attr]
            assert 'obs_noise_std_dev_fraction' in migrated[attr]
            assert 'clipping_bounds' in migrated[attr]
    
    def test_clipping_bounds_sensible(self):
        """Clipping bounds should be sensible for each attribute type."""
        old_cfg = {
            'prob_infected_dist': {'type': 'constant', 'value': 0.5},
            'prob_infected_observation_bias': 1.0,
            'prob_infected_observation_noise': 0.0,
            'benefit_value_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'benefit_value_multiplier_observation_bias': 1.0,
            'benefit_value_multiplier_observation_noise': 0.0,
            'failure_value_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'failure_value_multiplier_observation_bias': 1.0,
            'failure_value_multiplier_observation_noise': 0.0,
            'benefit_probability_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'benefit_probability_multiplier_observation_bias': 1.0,
            'benefit_probability_multiplier_observation_noise': 0.0,
            'failure_probability_multiplier_dist': {'type': 'constant', 'value': 1.0},
            'failure_probability_multiplier_observation_bias': 1.0,
            'failure_probability_multiplier_observation_noise': 0.0,
            'recovery_without_treatment_prob_dist': {'type': 'constant', 'value': 0.01},
            'recovery_without_treatment_prob_observation_bias': 1.0,
            'recovery_without_treatment_prob_observation_noise': 0.0,
        }
        
        migrated = migrate_old_config_to_new(old_cfg)
        
        # Probabilities should have [0.0, 1.0]
        assert migrated['prob_infected']['clipping_bounds'] == [0.0, 1.0]
        assert migrated['recovery_without_treatment_prob']['clipping_bounds'] == [0.0, 1.0]
        
        # Multipliers should have [0.0, None] (unbounded upper)
        assert migrated['benefit_value_multiplier']['clipping_bounds'][0] == 0.0
        assert migrated['benefit_value_multiplier']['clipping_bounds'][1] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
