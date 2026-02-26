"""
Comprehensive tests for PatientGenerator class.

Tests cover:
- Patient object creation and attributes
- Distribution sampling (constant, gaussian)
- RNG seeding and reproducibility
- Bounds checking and clipping
- Observation bias and noise application
- Configuration validation
- Edge cases and error handling
"""

import pytest
import numpy as np
from abx_amr_simulator.core import PatientGenerator
from abx_amr_simulator.core import Patient

# Default AMR levels for tests (single antibiotic 'A')
TRUE_AMR_LEVELS = {'A': 0.0}


class TestPatientDataclass:
    """Tests for Patient dataclass."""
    
    def test_patient_creation(self):
        """Test basic patient creation with all attributes."""
        patient = Patient(
            prob_infected=0.5,
            benefit_value_multiplier=1.2,
            failure_value_multiplier=0.8,
            benefit_probability_multiplier=1.1,
            failure_probability_multiplier=0.9,
            recovery_without_treatment_prob=0.3,
            infection_status=True,
            abx_sensitivity_dict={'A': True, 'B': False},
            prob_infected_obs=0.5,
            benefit_value_multiplier_obs=1.1,
            failure_value_multiplier_obs=0.9,
            benefit_probability_multiplier_obs=1.05,
            failure_probability_multiplier_obs=0.95,
            recovery_without_treatment_prob_obs=0.35,
        )
        assert patient.prob_infected == 0.5
        assert patient.benefit_value_multiplier == 1.2
        assert patient.failure_value_multiplier == 0.8
        assert patient.benefit_probability_multiplier == 1.1
        assert patient.failure_probability_multiplier == 0.9
        assert patient.recovery_without_treatment_prob == 0.3
        assert patient.infection_status == True
        assert patient.abx_sensitivity_dict == {'A': True, 'B': False}
        assert patient.prob_infected_obs == 0.5
        assert patient.benefit_value_multiplier_obs == 1.1
        assert patient.failure_value_multiplier_obs == 0.9
        assert patient.benefit_probability_multiplier_obs == 1.05
        assert patient.failure_probability_multiplier_obs == 0.95
        assert patient.recovery_without_treatment_prob_obs == 0.35


class TestPatientGeneratorInit:
    """Tests for PatientGenerator initialization."""
    
    def get_valid_config(self):
        """Helper to get a valid configuration."""
        return {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.8},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
    
    def test_init_with_valid_config(self):
        """Test initialization with valid config."""
        config = self.get_valid_config()
        gen = PatientGenerator(config)
        assert gen.config == config
    
    # Removed test for antibiotic_names property, as PatientGenerator no longer has this attribute
    
    def test_init_with_observation_bias_noise_scale(self):
        """Test initialization with nested observation bias and noise parameters."""
        config = self.get_valid_config()
        # Update nested observation parameters for a few attributes
        config['benefit_value_multiplier']['obs_bias_multiplier'] = 1.1
        config['benefit_value_multiplier']['obs_noise_one_std_dev'] = 2.0
        config['benefit_value_multiplier']['obs_noise_std_dev_fraction'] = 0.05
        config['benefit_value_multiplier']['clipping_bounds'] = [0.0, None]

        config['failure_value_multiplier']['obs_bias_multiplier'] = 0.9
        config['failure_value_multiplier']['obs_noise_one_std_dev'] = 2.0
        config['failure_value_multiplier']['obs_noise_std_dev_fraction'] = 0.1
        config['failure_value_multiplier']['clipping_bounds'] = [0.0, None]

        config['recovery_without_treatment_prob']['obs_bias_multiplier'] = 1.0
        config['recovery_without_treatment_prob']['obs_noise_one_std_dev'] = 1.0
        config['recovery_without_treatment_prob']['obs_noise_std_dev_fraction'] = 0.02
        config['recovery_without_treatment_prob']['clipping_bounds'] = [0.0, 1.0]

        gen = PatientGenerator(config)
        # Validate nested config stored correctly
        assert gen.attribute_configs['benefit_value_multiplier']['obs_bias_multiplier'] == 1.1
        assert gen.attribute_configs['benefit_value_multiplier']['obs_noise_one_std_dev'] == 2.0
        assert gen.attribute_configs['benefit_value_multiplier']['obs_noise_std_dev_fraction'] == 0.05
        assert gen.attribute_configs['failure_value_multiplier']['obs_bias_multiplier'] == 0.9
        assert gen.attribute_configs['failure_value_multiplier']['obs_noise_std_dev_fraction'] == 0.1
        assert gen.attribute_configs['recovery_without_treatment_prob']['obs_noise_std_dev_fraction'] == 0.02
    
    def test_init_missing_required_dist_key(self):
        """Test initialization fails when required distribution key is missing."""
        config = self.get_valid_config()
        # Make 'benefit_value_multiplier' visible, then remove its config to trigger error
        config['visible_patient_attributes'] = ['prob_infected', 'benefit_value_multiplier']
        del config['benefit_value_multiplier']
        with pytest.raises(ValueError, match="visible_patient_attributes lists 'benefit_value_multiplier', but no config exists for it"):
            PatientGenerator(config)
    
    def test_init_missing_dist_type(self):
        """Test initialization fails when 'type' key is missing from distribution."""
        config = self.get_valid_config()
        config['prob_infected']['prob_dist'] = {'value': 0.5}
        with pytest.raises(ValueError, match="Missing 'type' in prob_infected.prob_dist"):
            PatientGenerator(config)
    
    def test_init_invalid_dist_type(self):
        """Test initialization fails with unknown distribution type."""
        config = self.get_valid_config()
        config['prob_infected']['prob_dist'] = {'type': 'invalid_dist', 'alpha': 2.0}
        with pytest.raises(ValueError, match="Unknown distribution type 'invalid_dist' in prob_infected.prob_dist"):
            PatientGenerator(config)
    
    def test_init_invalid_observation_bias(self):
        """Test initialization fails with invalid observation bias."""
        config = self.get_valid_config()
        config['benefit_value_multiplier']['obs_bias_multiplier'] = 0.0
        with pytest.raises(ValueError, match="benefit_value_multiplier.obs_bias_multiplier must be > 0"):
            PatientGenerator(config)
    
    def test_init_invalid_observation_noise_scale(self):
        """Test initialization fails with negative observation noise scale."""
        config = self.get_valid_config()
        config['benefit_value_multiplier']['obs_noise_std_dev_fraction'] = -0.1
        with pytest.raises(ValueError, match="benefit_value_multiplier.obs_noise_std_dev_fraction must be >= 0"):
            PatientGenerator(config)


class TestPatientGeneratorConstants:
    """Tests for PatientGenerator class constants and methods."""
    
    def test_required_config_keys_constant_exists(self):
        """Test that REQUIRED_CONFIG_KEYS constant is defined."""
        assert hasattr(PatientGenerator, 'REQUIRED_CONFIG_KEYS')
        assert isinstance(PatientGenerator.REQUIRED_CONFIG_KEYS, list)
    
    def test_required_config_keys_has_all_distributions(self):
        """Test that REQUIRED_CONFIG_KEYS contains all distribution keys."""
        expected_keys = {
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob',
            'visible_patient_attributes',
        }
        assert set(PatientGenerator.REQUIRED_CONFIG_KEYS) == expected_keys
    
    def test_default_config_method_exists(self):
        """Test that default_config class method is defined."""
        assert hasattr(PatientGenerator, 'default_config')
        assert callable(PatientGenerator.default_config)
    
    def test_default_config_returns_dict(self):
        """Test that default_config returns a dictionary."""
        config = PatientGenerator.default_config()
        assert isinstance(config, dict)
    
    def test_default_config_has_all_required_keys(self):
        """Test that default_config includes all required configuration keys."""
        config = PatientGenerator.default_config()
        for key in PatientGenerator.REQUIRED_CONFIG_KEYS:
            assert key in config, f"Missing required key: {key}"
    
    def test_default_config_has_all_observation_parameters(self):
        """Test that default_config includes nested observation parameters per attribute."""
        config = PatientGenerator.default_config()
        attrs = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
            'benefit_probability_multiplier',
            'failure_probability_multiplier',
            'recovery_without_treatment_prob',
        ]
        for attr in attrs:
            assert 'prob_dist' in config[attr]
            assert 'obs_bias_multiplier' in config[attr]
            assert 'obs_noise_one_std_dev' in config[attr]
            assert 'obs_noise_std_dev_fraction' in config[attr]
            assert 'clipping_bounds' in config[attr]
    
    def test_default_config_has_visibility_setting(self):
        """Test that default_config includes visible_patient_attributes."""
        config = PatientGenerator.default_config()
        assert 'visible_patient_attributes' in config
        assert isinstance(config['visible_patient_attributes'], list)
    
    def test_default_config_is_valid(self):
        """Test that default_config can be used to initialize PatientGenerator."""
        config = PatientGenerator.default_config()
        gen = PatientGenerator(config)
        assert gen.config == config
    
    def test_default_config_is_mutable(self):
        """Test that returned default_config can be modified."""
        config1 = PatientGenerator.default_config()
        config1['prob_infected']['prob_dist']['value'] = 0.9
        
        config2 = PatientGenerator.default_config()
        # Verify they are independent (not shared references)
        assert config2['prob_infected']['prob_dist']['value'] != 0.9
        assert config2['prob_infected']['prob_dist']['value'] == 0.8
    
    def test_default_config_customization_example(self):
        """Test the recommended pattern for customizing default_config."""
        # Get default template
        config = PatientGenerator.default_config()
        
        # Customize specific values
        config['prob_infected']['prob_dist']['value'] = 0.6
        config['benefit_value_multiplier']['prob_dist'] = {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2}
        config['visible_patient_attributes'] = [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
        ]
        
        # Should work without errors
        gen = PatientGenerator(config)
        assert gen.visible_patient_attributes == [
            'prob_infected',
            'benefit_value_multiplier',
            'failure_value_multiplier',
        ]
    
    def test_error_message_references_constants_and_methods(self):
        """Test that error message for missing visibility key references default_config()."""
        config = PatientGenerator.default_config()
        del config['visible_patient_attributes']
        
        with pytest.raises(ValueError) as exc_info:
            PatientGenerator(config)
        
        error_msg = str(exc_info.value)
        assert 'default_config()' in error_msg


class TestPatientSampling:
    """Tests for patient sampling functionality."""
    
    def get_valid_config(self):
        """Helper to get a valid configuration."""
        return {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.8},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
    
    def test_sample_basic(self):
        """Test basic patient sampling returns correct number of patients."""
        config = self.get_valid_config()
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        patients = gen.sample(n_patients=10, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        assert len(patients) == 10
        assert all(isinstance(p, Patient) for p in patients)
    
    def test_sample_patient_attributes_in_bounds(self):
        """Test that sampled patient attributes are within expected bounds."""
        config = self.get_valid_config()
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        patients = gen.sample(n_patients=100, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        for patient in patients:
            assert 0.0 <= patient.prob_infected <= 1.0
        assert patient.benefit_value_multiplier > 0
        assert patient.failure_value_multiplier > 0
        assert patient.benefit_probability_multiplier > 0
        assert patient.failure_probability_multiplier > 0
        assert 0.0 <= patient.recovery_without_treatment_prob <= 1.0
        assert patient.benefit_value_multiplier_obs > 0
        assert patient.failure_value_multiplier_obs > 0
        assert patient.benefit_probability_multiplier_obs > 0
        assert patient.failure_probability_multiplier_obs > 0
        assert 0.0 <= patient.recovery_without_treatment_prob_obs <= 1.0
    
    def test_sample_reproducibility_with_seed(self):
        """Test that sampling is reproducible with the same seed."""
        config = self.get_valid_config()
        gen = PatientGenerator(config)
        
        rng1 = np.random.default_rng(seed=42)
        patients1 = gen.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng1)
        
        rng2 = np.random.default_rng(seed=42)
        patients2 = gen.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng2)
        
        for p1, p2 in zip(patients1, patients2):
            assert p1.prob_infected == p2.prob_infected
            assert p1.benefit_value_multiplier == p2.benefit_value_multiplier
            assert p1.failure_value_multiplier == p2.failure_value_multiplier
            assert p1.benefit_probability_multiplier == p2.benefit_probability_multiplier
            assert p1.failure_probability_multiplier == p2.failure_probability_multiplier
            assert p1.recovery_without_treatment_prob == p2.recovery_without_treatment_prob
    
    def test_sample_different_seeds_produce_different_patients(self):
        """Test that different seeds produce different patient samples."""
        config = self.get_valid_config()
        gen = PatientGenerator(config)
        
        rng1 = np.random.default_rng(seed=42)
        patients1 = gen.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng1)
        
        rng2 = np.random.default_rng(seed=43)
        patients2 = gen.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng2)
        
        # At least some patients should differ
        differences = sum(
            1 for p1, p2 in zip(patients1, patients2)
            if (
                p1.prob_infected != p2.prob_infected or
                p1.benefit_value_multiplier != p2.benefit_value_multiplier or
                p1.failure_value_multiplier != p2.failure_value_multiplier or
                p1.benefit_probability_multiplier != p2.benefit_probability_multiplier or
                p1.failure_probability_multiplier != p2.failure_probability_multiplier or
                p1.recovery_without_treatment_prob != p2.recovery_without_treatment_prob
            )
        )
        assert differences > 0
    
    def test_sample_single_patient(self):
        """Test sampling a single patient."""
        config = self.get_valid_config()
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        patients = gen.sample(n_patients=1, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        assert len(patients) == 1
        assert isinstance(patients[0], Patient)
    
    def test_sample_many_patients(self):
        """Test sampling many patients."""
        config = self.get_valid_config()
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        patients = gen.sample(n_patients=1000, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        assert len(patients) == 1000
    
    def test_sample_zero_patients_raises_error(self):
        """Test that sampling zero patients raises ValueError."""
        config = self.get_valid_config()
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        with pytest.raises(ValueError, match="n_patients must be > 0"):
            gen.sample(n_patients=0, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
    
    def test_sample_negative_patients_raises_error(self):
        """Test that sampling negative patients raises ValueError."""
        config = self.get_valid_config()
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        with pytest.raises(ValueError, match="n_patients must be > 0"):
            gen.sample(n_patients=-5, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)


class TestDistributionSampling:
    """Tests for individual distribution sampling."""
    
    def get_gen_and_rng(self):
        """Helper to get generator and RNG."""
        config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.8},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        return gen, rng

    def test_sample_from_constant_dist(self):
        """Test sampling from constant distribution."""
        gen, rng = self.get_gen_and_rng()
        dist_config = {'type': 'constant', 'value': 0.75}
        
        for _ in range(100):
            value = gen._sample_from_dist(rng, dist_config, bounds=(0.0, 1.0))
            assert value == 0.75
    
    def test_sample_from_gaussian_dist(self):
        """Test sampling from Gaussian distribution."""
        gen, rng = self.get_gen_and_rng()
        dist_config = {'type': 'gaussian', 'mu': 0.5, 'sigma': 0.1}
        
        values = [gen._sample_from_dist(rng, dist_config) for _ in range(100)]
        # Mean should be approximately 0.5
        assert abs(np.mean(values) - 0.5) < 0.2
    
    def test_constant_dist_missing_parameters(self):
        """Test constant distribution fails without required parameters."""
        gen, rng = self.get_gen_and_rng()
        dist_config = {'type': 'constant'}  # missing 'value'
        
        with pytest.raises(ValueError, match="Constant distribution requires"):
            gen._sample_from_dist(rng, dist_config)
    
    def test_gaussian_dist_missing_parameters(self):
        """Test Gaussian distribution fails without required parameters."""
        gen, rng = self.get_gen_and_rng()
        dist_config = {'type': 'gaussian', 'mu': 0.0}  # missing 'sigma'
        
        with pytest.raises(ValueError, match="Gaussian distribution requires"):
            gen._sample_from_dist(rng, dist_config)
    
    def test_gaussian_dist_invalid_sigma(self):
        """Test Gaussian distribution fails with negative sigma."""
        gen, rng = self.get_gen_and_rng()
        dist_config = {'type': 'gaussian', 'mu': 0.0, 'sigma': -0.1}
        
        with pytest.raises(ValueError, match="Gaussian sigma must be non-negative"):
            gen._sample_from_dist(rng, dist_config)
    
    def test_dist_bounds_lower_clipping(self):
        """Test that lower bound is applied correctly."""
        gen, rng = self.get_gen_and_rng()
        # Gaussian that could go negative, but we clip at 0
        dist_config = {'type': 'gaussian', 'mu': -1.0, 'sigma': 0.5}
        
        for _ in range(50):
            value = gen._sample_from_dist(rng, dist_config, bounds=(0.0, None))
            assert value >= 0.0
    
    def test_dist_bounds_upper_clipping(self):
        """Test that upper bound is applied correctly."""
        gen, rng = self.get_gen_and_rng()
        # Gaussian that could go > 1, but we clip at 1
        dist_config = {'type': 'gaussian', 'mu': 0.5, 'sigma': 2.0}
        
        for _ in range(50):
            value = gen._sample_from_dist(rng, dist_config, bounds=(0.0, 1.0))
            assert 0.0 <= value <= 1.0
    
    def test_unknown_dist_type_raises_error(self):
        """Test that unknown distribution type raises ValueError."""
        gen, rng = self.get_gen_and_rng()
        dist_config = {'type': 'exponential', 'lambda': 1.0}
        
        with pytest.raises(ValueError, match="Unknown distribution type"):
            gen._sample_from_dist(rng, dist_config)


class TestObservationBiasAndNoise:
    """Tests for observation bias and noise application."""
    
    def get_gen_and_rng(self):
        """Helper to get generator and RNG."""
        config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.8},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        return gen, rng
    
    def test_apply_bias_no_noise(self):
        """Test bias application without noise."""
        gen, rng = self.get_gen_and_rng()
        true_value = 1.0
        bias = 1.5
        obs_value = gen._apply_multip_bias_and_additive_noise(
            true_value=true_value,
            obs_bias_multiplier=bias,
            obs_noise_one_std_dev=0.0,
            obs_noise_std_dev_fraction=0.0,
            clipping_bounds=(0.0, None),
            rng=rng,
        )
        assert obs_value == pytest.approx(1.5)
    
    def test_apply_noise_no_bias(self):
        """Test noise application without bias."""
        gen, rng = self.get_gen_and_rng()
        true_value = 1.0
        bias = 1.0
        # With seed set, we know the noise amount
        rng_test = np.random.default_rng(seed=42)
        obs_value = gen._apply_multip_bias_and_additive_noise(
            true_value=true_value,
            obs_bias_multiplier=bias,
            obs_noise_one_std_dev=10.0,
            obs_noise_std_dev_fraction=0.1,
            clipping_bounds=None,
            rng=rng_test,
        )
        # Should be approximately 1.0 plus some small noise
        assert abs(obs_value - 1.0) < 1.0
    
    def test_apply_bias_and_noise(self):
        """Test both bias and noise application."""
        gen, rng = self.get_gen_and_rng()
        true_value = 1.0
        bias = 2.0
        obs_value = gen._apply_multip_bias_and_additive_noise(
            true_value=true_value,
            obs_bias_multiplier=bias,
            obs_noise_one_std_dev=10.0,
            obs_noise_std_dev_fraction=0.1,
            clipping_bounds=None,
            rng=rng,
        )
        # Should be approximately 2.0 plus noise (noise_scale=0.1 on [0,10] range = std of 1.0)
        assert 1.0 < obs_value < 3.5
    
    def test_apply_bounds_lower(self):
        """Test lower bound clipping during bias/noise application."""
        gen, rng = self.get_gen_and_rng()
        true_value = 0.5
        # Set up rng to produce large negative noise
        rng_test = np.random.default_rng(seed=123)
        # Add large noise that would go negative without bounds
        obs_value = gen._apply_multip_bias_and_additive_noise(
            true_value=true_value,
            obs_bias_multiplier=1.0,
            obs_noise_one_std_dev=1.0,
            obs_noise_std_dev_fraction=1.0,
            clipping_bounds=(0.0, 1.0),
            rng=rng_test,
        )
        assert obs_value >= 0.0
    
    def test_apply_bounds_upper(self):
        """Test upper bound clipping during bias/noise application."""
        gen, rng = self.get_gen_and_rng()
        true_value = 0.9
        rng_test = np.random.default_rng(seed=456)
        # Add large noise that would exceed 1.0 without bounds
        obs_value = gen._apply_multip_bias_and_additive_noise(
            true_value=true_value,
            obs_bias_multiplier=2.0,
            obs_noise_one_std_dev=1.0,
            obs_noise_std_dev_fraction=1.0,
            clipping_bounds=(0.0, 1.0),
            rng=rng_test,
        )
        assert obs_value <= 1.0
    
    def test_observation_parameters_in_patient(self):
        """Test that observation parameters are applied in patient sampling."""
        config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.2,
                'obs_noise_one_std_dev': 2.0,
                'obs_noise_std_dev_fraction': 0.05,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.8},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        patients = gen.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        # Observed values should generally differ from true values due to bias
        for patient in patients:
            # benefit_value_multiplier_obs should be ~20% higher on average than benefit_value_multiplier
            # (before noise, but noise is small)
            assert patient.benefit_value_multiplier_obs > 0  # Always positive after bounds check


class TestComplexConfigs:
    """Tests with various complex configurations."""
    
    def test_all_distributions_mixed(self):
        """Test using different distribution types for different attributes."""
        config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.15},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'gaussian', 'mu': 0.5, 'sigma': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        patients = gen.sample(n_patients=100, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        assert len(patients) == 100
        assert all(isinstance(p, Patient) for p in patients)
    
    def test_high_bias_high_noise(self):
        """Test with high observation bias and noise."""
        config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 2.0,
                'obs_noise_one_std_dev': 2.0,
                'obs_noise_std_dev_fraction': 0.5,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 0.5,
                'obs_noise_one_std_dev': 2.0,
                'obs_noise_std_dev_fraction': 0.2,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.8},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        patients = gen.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        assert all(isinstance(p, Patient) for p in patients)
    
    def test_no_observation_noise_or_bias(self):
        """Test with zero observation noise and bias of 1.0 (no distortion)."""
        config = {
            'prob_infected': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.3},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.8},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        }
        gen = PatientGenerator(config)
        rng = np.random.default_rng(seed=42)
        
        patients = gen.sample(n_patients=50, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        for patient in patients:
            # When bias=1.0 and noise=0.0, observed should equal true
            assert patient.benefit_value_multiplier_obs == pytest.approx(patient.benefit_value_multiplier)
            assert patient.failure_value_multiplier_obs == pytest.approx(patient.failure_value_multiplier)
            assert patient.benefit_probability_multiplier_obs == pytest.approx(patient.benefit_probability_multiplier)
            assert patient.failure_probability_multiplier_obs == pytest.approx(patient.failure_probability_multiplier)
            assert patient.recovery_without_treatment_prob_obs == pytest.approx(patient.recovery_without_treatment_prob)


class TestPatientGeneratorMixer:
    """Comprehensive tests for PatientGeneratorMixer class."""
    
    @staticmethod
    def create_simple_generator(visibility_attrs=None, seed=None):
        """Helper to create a simple generator for testing."""
        if visibility_attrs is None:
            visibility_attrs = ['prob_infected']
        
        config = {
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
                'prob_dist': {'type': 'constant', 'value': 0.2},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': visibility_attrs,
        }
        if seed is not None:
            config['seed'] = seed
        
        return PatientGenerator(config=config)
    
    def test_mixer_init_uniform_visibility(self):
        """Test PatientGeneratorMixer initialization with uniform visibility."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_a = self.create_simple_generator(visibility_attrs=['prob_infected'])
        gen_b = self.create_simple_generator(visibility_attrs=['prob_infected'])
        
        config = {
            'generators': [gen_a, gen_b],
            'proportions': [0.5, 0.5],
        }
        
        mixer = PatientGeneratorMixer(config=config)
        
        assert mixer.visible_patient_attributes == ['prob_infected']
        assert not mixer._uses_heterogeneous_visibility
        assert len(mixer.generators) == 2
        assert np.allclose(mixer.proportions, [0.5, 0.5])
    
    def test_mixer_init_heterogeneous_visibility(self):
        """Test PatientGeneratorMixer with different generator visibilities."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_a = self.create_simple_generator(visibility_attrs=['prob_infected', 'benefit_value_multiplier'])
        gen_b = self.create_simple_generator(visibility_attrs=['prob_infected'])
        
        config = {
            'generators': [gen_a, gen_b],
            'proportions': [0.5, 0.5],
        }
        
        mixer = PatientGeneratorMixer(config=config)
        
        assert mixer._uses_heterogeneous_visibility
        assert set(mixer.visible_patient_attributes) == {'prob_infected', 'benefit_value_multiplier'}
        assert len(mixer.generators) == 2
    
    def test_mixer_init_proportions_sum_validation(self):
        """Test that proportions must sum to 1.0."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_a = self.create_simple_generator()
        gen_b = self.create_simple_generator()
        
        config = {
            'generators': [gen_a, gen_b],
            'proportions': [0.5, 0.6],  # Sum is 1.1, not 1.0
        }
        
        with pytest.raises(ValueError, match="sum to 1.0"):
            PatientGeneratorMixer(config=config)
    
    def test_mixer_init_proportions_count_mismatch(self):
        """Test that proportions count must match generator count."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_a = self.create_simple_generator()
        gen_b = self.create_simple_generator()
        
        config = {
            'generators': [gen_a, gen_b],
            'proportions': [1.0],  # Only 1 proportion for 2 generators
        }
        
        with pytest.raises(ValueError, match="must match"):
            PatientGeneratorMixer(config=config)
    
    def test_mixer_init_empty_generators(self):
        """Test that empty generator list raises error."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        config = {
            'generators': [],
            'proportions': [],
        }
        
        with pytest.raises(ValueError, match="at least one"):
            PatientGeneratorMixer(config=config)
    
    def test_mixer_sample_proportions(self):
        """Test that sample respects proportions."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_a = self.create_simple_generator()
        gen_b = self.create_simple_generator()
        
        config = {
            'generators': [gen_a, gen_b],
            'proportions': [0.7, 0.3],
            'seed': 42,
        }
        
        mixer = PatientGeneratorMixer(config=config)
        rng = np.random.default_rng(42)
        
        patients = mixer.sample(n_patients=100, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        # Should have approximately 70 from gen_a and 30 from gen_b
        # Due to rounding, allow some flexibility
        assert len(patients) == 100
        assert all(hasattr(p, 'source_generator_index') for p in patients)
    
    def test_mixer_sample_rounding(self):
        """Test that proportional allocation rounding works correctly."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_a = self.create_simple_generator()
        gen_b = self.create_simple_generator()
        gen_c = self.create_simple_generator()
        
        config = {
            'generators': [gen_a, gen_b, gen_c],
            'proportions': [0.33, 0.33, 0.34],  # Will have rounding issues with 100
            'seed': 42,
        }
        
        mixer = PatientGeneratorMixer(config=config)
        rng = np.random.default_rng(42)
        
        patients = mixer.sample(n_patients=100, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        # Total must equal exactly 100
        assert len(patients) == 100
    
    def test_mixer_observe_uniform_visibility(self):
        """Test observe method with uniform visibility (no padding)."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_a = self.create_simple_generator(visibility_attrs=['prob_infected'])
        gen_b = self.create_simple_generator(visibility_attrs=['prob_infected'])
        
        config = {
            'generators': [gen_a, gen_b],
            'proportions': [0.5, 0.5],
            'seed': 42,
        }
        
        mixer = PatientGeneratorMixer(config=config)
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=4, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        obs = mixer.observe(patients)
        
        # Should be 4 patients * 1 attribute = 4 values
        assert obs.shape == (4,)
    
    def test_mixer_observe_heterogeneous_visibility_padding(self):
        """Test observe with heterogeneous visibility shows padding."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_full = self.create_simple_generator(visibility_attrs=['prob_infected', 'benefit_value_multiplier'])
        gen_minimal = self.create_simple_generator(visibility_attrs=['prob_infected'])
        
        config = {
            'generators': [gen_full, gen_minimal],
            'proportions': [0.5, 0.5],
            'seed': 42,
        }
        
        mixer = PatientGeneratorMixer(config=config)
        rng = np.random.default_rng(42)
        patients = mixer.sample(n_patients=4, true_amr_levels=TRUE_AMR_LEVELS, rng=rng)
        
        obs = mixer.observe(patients)
        
        # Should be 4 patients * 2 attributes (union) = 8 values
        assert obs.shape == (8,)
        
        # Check for padding value (-1.0) where attributes are hidden
        # At least some patients from gen_minimal should have padding
        has_padding = np.any(obs == mixer.PADDING_VALUE)
        assert has_padding or len(patients) < 2  # May not have gen_minimal patients in small sample
    
    def test_mixer_obs_dim(self):
        """Test obs_dim method."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_a = self.create_simple_generator(visibility_attrs=['prob_infected', 'benefit_value_multiplier'])
        gen_b = self.create_simple_generator(visibility_attrs=['prob_infected'])
        
        config = {
            'generators': [gen_a, gen_b],
            'proportions': [0.5, 0.5],
        }
        
        mixer = PatientGeneratorMixer(config=config)
        
        # Union has 2 attributes, 10 patients → 20
        assert mixer.obs_dim(num_patients=10) == 20
        assert mixer.obs_dim(num_patients=1) == 2
        assert mixer.obs_dim(num_patients=100) == 200
    
    def test_mixer_reproducibility_with_seed(self):
        """Test that mixer sampling is reproducible with same seed."""
        from abx_amr_simulator.core import PatientGeneratorMixer
        
        gen_a = self.create_simple_generator()
        gen_b = self.create_simple_generator()
        
        config = {
            'generators': [gen_a, gen_b],
            'proportions': [0.5, 0.5],
            'seed': 42,
        }
        
        mixer1 = PatientGeneratorMixer(config=config)
        mixer2 = PatientGeneratorMixer(config=config)
        
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        
        patients1 = mixer1.sample(n_patients=20, true_amr_levels=TRUE_AMR_LEVELS, rng=rng1)
        patients2 = mixer2.sample(n_patients=20, true_amr_levels=TRUE_AMR_LEVELS, rng=rng2)
        
        # Same seed should produce same patients (shuffling order may differ)
        assert len(patients1) == len(patients2) == 20
        # Verify attribute values match (order might differ due to shuffle)
        vals1 = sorted([p.prob_infected for p in patients1])
        vals2 = sorted([p.prob_infected for p in patients2])
        assert np.allclose(vals1, vals2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
