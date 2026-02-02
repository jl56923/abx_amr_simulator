"""
Unit tests for RNG ownership semantics across RewardCalculator, PatientGenerator, and ABXAMREnv.

Tests the lifecycle: standalone → environment-owned, runtime validation, and cascading ownership.
"""

import pytest
import numpy as np
from abx_amr_simulator.core.reward_calculator import RewardCalculator
from abx_amr_simulator.core.patient_generator import PatientGenerator, PatientGeneratorMixer
from abx_amr_simulator.core.abx_amr_env import ABXAMREnv


class TestRewardCalculatorOwnership:
    """Test ownership lifecycle and validation for RewardCalculator."""
    
    def test_starts_in_standalone_mode(self):
        """RewardCalculator should initialize with _standalone=True."""
        config = RewardCalculator.default_config()
        rc = RewardCalculator(config=config)
        assert rc._standalone is True
    
    def test_ownership_transfer(self):
        """_set_environment_owned should flip _standalone to False."""
        config = RewardCalculator.default_config()
        rc = RewardCalculator(config=config)
        assert rc._standalone is True
        
        rc._set_environment_owned()
        assert rc._standalone is False
    
    def test_standalone_allows_missing_rng(self):
        """Standalone RC should use self.rng when rng argument not provided."""
        config = RewardCalculator.default_config()
        rc = RewardCalculator(config=config)
        
        # Should not raise - uses self.rng
        patients = [self._make_dummy_patient()]
        actions = np.array([1])  # prescribe antibiotic
        antibiotic_names = ['A']
        amr_levels = {'A': 0.0}
        delta_amr = {'A': 0.01}
        
        reward, info = rc.calculate_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=amr_levels,
            delta_visible_amr_per_antibiotic=delta_amr,
            # No rng argument - should use self.rng
        )
        assert isinstance(reward, float)
    
    def test_owned_requires_explicit_rng(self):
        """Environment-owned RC should fail loudly when rng not provided."""
        config = RewardCalculator.default_config()
        rc = RewardCalculator(config=config)
        rc._set_environment_owned()
        
        patients = [self._make_dummy_patient()]
        actions = np.array([1])
        antibiotic_names = ['A']
        amr_levels = {'A': 0.0}
        delta_amr = {'A': 0.01}
        
        with pytest.raises(ValueError, match="environment-owned.*requires explicit rng"):
            rc.calculate_reward(
                patients=patients,
                actions=actions,
                antibiotic_names=antibiotic_names,
                visible_amr_levels=amr_levels,
                delta_visible_amr_per_antibiotic=delta_amr,
                # Missing rng argument
            )
    
    def test_owned_works_with_explicit_rng(self):
        """Environment-owned RC should work when explicit rng provided."""
        config = RewardCalculator.default_config()
        rc = RewardCalculator(config=config)
        rc._set_environment_owned()
        
        patients = [self._make_dummy_patient()]
        actions = np.array([1])
        antibiotic_names = ['A']
        amr_levels = {'A': 0.0}
        delta_amr = {'A': 0.01}
        rng = np.random.default_rng(42)
        
        # Should not raise
        reward, info = rc.calculate_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=amr_levels,
            delta_visible_amr_per_antibiotic=delta_amr,
            rng=rng,
        )
        assert isinstance(reward, float)
    
    @staticmethod
    def _make_dummy_patient():
        """Helper to create a minimal Patient object for testing."""
        from abx_amr_simulator.core.types import Patient
        return Patient(
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


class TestPatientGeneratorOwnership:
    """Test ownership lifecycle and validation for PatientGenerator."""
    
    def test_starts_in_standalone_mode(self):
        """PatientGenerator should initialize with _standalone=True."""
        config = PatientGenerator.default_config()
        pg = PatientGenerator(config=config)
        assert pg._standalone is True
    
    def test_ownership_transfer(self):
        """_set_environment_owned should flip _standalone to False."""
        config = PatientGenerator.default_config()
        pg = PatientGenerator(config=config)
        assert pg._standalone is True
        
        pg._set_environment_owned()
        assert pg._standalone is False
    
    def test_standalone_allows_missing_rng(self):
        """Standalone PG should use self.rng when rng argument not provided."""
        config = PatientGenerator.default_config()
        pg = PatientGenerator(config=config)
        pg.rng = np.random.default_rng(42)  # Initialize RNG
        
        # Should not raise - uses self.rng
        patients = pg.sample(n_patients=5, true_amr_levels={"A": 0.0})
        assert len(patients) == 5
    
    def test_owned_requires_explicit_rng(self):
        """Environment-owned PG should fail loudly when rng not provided."""
        config = PatientGenerator.default_config()
        pg = PatientGenerator(config=config)
        pg._set_environment_owned()
        
        with pytest.raises(ValueError, match="environment-owned.*requires explicit rng"):
            pg.sample(n_patients=5, true_amr_levels={"A": 0.0})  # Missing rng argument
    
    def test_owned_works_with_explicit_rng(self):
        """Environment-owned PG should work when explicit rng provided."""
        config = PatientGenerator.default_config()
        pg = PatientGenerator(config=config)
        pg._set_environment_owned()
        
        rng = np.random.default_rng(42)
        patients = pg.sample(n_patients=5, true_amr_levels={"A": 0.0}, rng=rng)
        assert len(patients) == 5


class TestPatientGeneratorMixerOwnership:
    """Test cascading ownership for PatientGeneratorMixer."""
    
    def test_mixer_starts_standalone(self):
        """PatientGeneratorMixer should initialize with _standalone=True."""
        pg1 = PatientGenerator(config=PatientGenerator.default_config())
        pg2 = PatientGenerator(config=PatientGenerator.default_config())
        mixer = PatientGeneratorMixer(config={
            'generators': [pg1, pg2],
            'proportions': [0.5, 0.5],
        })
        assert mixer._standalone is True
    
    def test_ownership_cascades_to_children(self):
        """_set_environment_owned should cascade to all subordinate generators."""
        pg1 = PatientGenerator(config=PatientGenerator.default_config())
        pg2 = PatientGenerator(config=PatientGenerator.default_config())
        mixer = PatientGeneratorMixer(config={
            'generators': [pg1, pg2],
            'proportions': [0.5, 0.5],
        })
        
        # All start standalone
        assert mixer._standalone is True
        assert pg1._standalone is True
        assert pg2._standalone is True
        
        # Transfer ownership
        mixer._set_environment_owned()
        
        # Should cascade to all children
        assert mixer._standalone is False
        assert pg1._standalone is False
        assert pg2._standalone is False
    
    def test_owned_mixer_requires_explicit_rng(self):
        """Environment-owned mixer should fail when rng not provided."""
        pg1 = PatientGenerator(config=PatientGenerator.default_config())
        pg2 = PatientGenerator(config=PatientGenerator.default_config())
        mixer = PatientGeneratorMixer(config={
            'generators': [pg1, pg2],
            'proportions': [0.5, 0.5],
        })
        mixer._set_environment_owned()
        
        with pytest.raises(ValueError, match="environment-owned.*requires explicit rng"):
            mixer.sample(n_patients=10, true_amr_levels={"A": 0.0})


class TestEnvironmentOwnershipIntegration:
    """Test that ABXAMREnv correctly transfers ownership."""
    
    def test_env_transfers_ownership(self):
        """Environment should call _set_environment_owned on PG and RC."""
        rc = RewardCalculator(config=RewardCalculator.default_config())
        pg = PatientGenerator(config=PatientGenerator.default_config())
        
        # Before env creation
        assert rc._standalone is True
        assert pg._standalone is True
        
        # Create environment
        antibiotics_AMR_dict = {
            'A': {'leak': 0.95, 'flatness_parameter': 10.0, 
                  'permanent_residual_volume': 0.0, 'initial_amr_level': 0.0}
        }
        env = ABXAMREnv(
            reward_calculator=rc,
            patient_generator=pg,
            antibiotics_AMR_dict=antibiotics_AMR_dict,
            num_patients_per_time_step=5,
            max_time_steps=100,
        )
        
        # After env creation
        assert rc._standalone is False
        assert pg._standalone is False
    
    def test_env_passes_shared_rng(self):
        """Environment should pass shared RNG to PG and RC calls."""
        rc = RewardCalculator(config=RewardCalculator.default_config())
        pg = PatientGenerator(config=PatientGenerator.default_config())
        
        antibiotics_AMR_dict = {
            'A': {'leak': 0.95, 'flatness_parameter': 10.0,
                  'permanent_residual_volume': 0.0, 'initial_amr_level': 0.0}
        }
        env = ABXAMREnv(
            reward_calculator=rc,
            patient_generator=pg,
            antibiotics_AMR_dict=antibiotics_AMR_dict,
            num_patients_per_time_step=5,
            max_time_steps=100,
        )
        
        # Reset and step should work (components receive shared RNG)
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should not raise any ownership errors
        assert isinstance(reward, float)


class TestNoTreatmentRNGSkipping:
    """Test that no_treatment actions don't consume RNG draws for sensitivity."""
    
    def test_no_treatment_skips_sensitivity_draws(self):
        """No treatment actions should not consume RNG for sensitivity sampling."""
        config = RewardCalculator.default_config()
        rc = RewardCalculator(config=config)
        
        # Create homogeneous patients
        from abx_amr_simulator.core.types import Patient
        patients = [
            Patient(
                prob_infected=0.0,  # Not infected - simplifies test
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
            )
            for _ in range(5)
        ]
        
        # All no_treatment actions (action index = num_antibiotics)
        no_treatment_idx = rc.abx_name_to_index['no_treatment']
        actions = np.array([no_treatment_idx] * 5)
        
        antibiotic_names = ['A']
        amr_levels = {'A': 0.0}
        delta_amr = {'A': 0.0}
        
        # Run twice with same seed - should get identical results
        rng1 = np.random.default_rng(42)
        reward1, info1 = rc.calculate_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=amr_levels,
            delta_visible_amr_per_antibiotic=delta_amr,
            rng=rng1,
        )
        
        rng2 = np.random.default_rng(42)
        reward2, info2 = rc.calculate_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=amr_levels,
            delta_visible_amr_per_antibiotic=delta_amr,
            rng=rng2,
        )
        
        # Results should be identical (deterministic with same seed)
        assert reward1 == reward2
        
        # Verify RNG states match (no extra draws consumed)
        assert rng1.bit_generator.state == rng2.bit_generator.state
    
    def test_prescribe_consumes_rng_draws(self):
        """Prescribe actions should consume RNG for clinical/adverse sampling."""
        config = RewardCalculator.default_config()
        rc = RewardCalculator(config=config)
        
        from abx_amr_simulator.core.types import Patient
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
            for _ in range(5)
        ]
        
        # All prescribe actions
        actions = np.array([0] * 5)  # Index 0 = antibiotic A
        
        antibiotic_names = ['A']
        amr_levels = {'A': 0.0}
        delta_amr = {'A': 0.01}
        
        rng1 = np.random.default_rng(42)
        state_before = rng1.bit_generator.state
        
        reward, info = rc.calculate_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=amr_levels,
            delta_visible_amr_per_antibiotic=delta_amr,
            rng=rng1,
        )
        
        state_after = rng1.bit_generator.state
        
        # RNG state should have changed (draws consumed)
        assert state_before != state_after
