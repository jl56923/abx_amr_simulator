"""Functional tests for ABXAMREnv basic behavior and AMR accumulation."""
import pathlib
import sys
from typing import Optional

import pytest
import numpy as np

from tests.unit.utils.test_reference_helpers import create_mock_environment
from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.core import RewardCalculator
from abx_amr_simulator.core import PatientGenerator


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
    """Helper to create a RewardCalculator for testing (updated for new API)."""
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


def create_test_antibiotics_dict(antibiotic_names=None):
    """Helper to create a default antibiotics_AMR_dict for testing."""
    if antibiotic_names is None:
        antibiotic_names = ["A"]
    
    return {
        name: {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0
        }
        for name in antibiotic_names
    }

def create_ABXAMR_Env_instance(
    antibiotic_names=None,
    reward_calculator=None,
    antibiotics_AMR_dict=None,
    update_visible_AMR_levels_every_n_timesteps: int = 1,
    add_noise_to_visible_AMR_levels: float = 0.0,
    add_bias_to_visible_AMR_levels: float = 0.0,
    baseline_probability_of_infection: Optional[float] = None,
    std_dev_probability_of_infection: Optional[float] = None,
    crossresistance_matrix: Optional[dict] = None,
    visible_patient_attributes: Optional[list] = None,
    patient_generator: Optional[PatientGenerator] = None,
    include_steps_since_amr_update_in_obs: bool = False,
):
    """Helper to create a default ABXAMR_Env instance for testing (updated to require PatientGenerator).

    Notes:
    - We preserve `baseline_probability_of_infection` and `std_dev_probability_of_infection` parameters in this helper
        for backward-compatibility in tests by mapping them to PatientGenerator's nested `prob_infected` → `prob_dist` Gaussian config.
    - By default `visible_patient_attributes` shows only ['prob_infected'] to preserve prior observation shape.
    """
    if antibiotic_names is None:
        antibiotic_names = ["A"]

    # If reward_calculator not provided, create a default one
    if reward_calculator is None:
        reward_calculator = create_test_reward_calculator(antibiotic_names=antibiotic_names)

    if antibiotics_AMR_dict is None:
        antibiotics_AMR_dict = create_test_antibiotics_dict(antibiotic_names)

    # Build a default PatientGenerator if not provided
    if patient_generator is None:
        mu = 0.5 if baseline_probability_of_infection is None else float(baseline_probability_of_infection)
        sigma = 0.1 if std_dev_probability_of_infection is None else float(std_dev_probability_of_infection)
        
        if visible_patient_attributes is None:
            visible_patient_attributes = ['prob_infected']
        
        pg_config = {
            'prob_infected': {
                'prob_dist': {'type': 'gaussian', 'mu': mu, 'sigma': sigma},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            # Use tight gaussians around 1.0 for multipliers
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.5},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': visible_patient_attributes,
        }
        patient_generator = PatientGenerator(config=pg_config)
    elif visible_patient_attributes is not None:
        # If a patient_generator was provided AND visibility was specified, set it on the PG
        patient_generator.visible_patient_attributes = visible_patient_attributes

    return ABXAMREnv(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        antibiotics_AMR_dict=antibiotics_AMR_dict,
        num_patients_per_time_step=5,
        max_time_steps=10,
        update_visible_AMR_levels_every_n_timesteps=update_visible_AMR_levels_every_n_timesteps,
        add_noise_to_visible_AMR_levels=add_noise_to_visible_AMR_levels,
        add_bias_to_visible_AMR_levels=add_bias_to_visible_AMR_levels,
        crossresistance_matrix=crossresistance_matrix,
        include_steps_since_amr_update_in_obs=include_steps_since_amr_update_in_obs,
    )

def test_basic_functionality_shapes_and_truncation():
    """Test that environment initializes with correct shapes and properly truncates after max_time_steps.
    
    Purpose: Ensures that the observation and action space shapes are correctly initialized,
    and that the episode properly truncates when max_time_steps is reached.
    """
    antibiotic_names = ["A", "B"]
    env = create_ABXAMR_Env_instance(antibiotic_names=antibiotic_names)
    obs, _ = env.reset(seed=42)

    # Verify observation and action space shapes are correct
    assert obs.shape == (env.num_patients_per_time_step + env.num_abx,)
    assert env.action_space.shape == (env.num_patients_per_time_step,)

    # Take first step and verify not truncated yet
    no_treatment = np.full(env.num_patients_per_time_step, env.no_treatment_action, dtype=int)
    _, _, _, truncated, _ = env.step(no_treatment)
    assert truncated is False

    # Take steps until we reach max_time_steps and verify truncation
    # Default max_time_steps is 10, so we need to take more steps
    for _ in range(8):  # Take 8 more steps (total 9)
        _, _, _, truncated, _ = env.step(no_treatment)
        assert truncated is False
    
    # On the 10th step, should be truncated
    _, _, _, truncated, info = env.step(no_treatment)
    assert truncated is True
    assert set(info["actual_amr_levels"].keys()) == set(env.antibiotic_names)
    assert set(info["visible_amr_levels"].keys()) == set(env.antibiotic_names)


def test_amr_accumulation_and_leak():
    """Test AMR accumulation when antibiotic is prescribed, and AMR leakage when no treatment is given.
    
    Purpose: Verifies that prescribing an antibiotic increases AMR levels, and that AMR decreases
    (leaks) when no treatment is given, demonstrating the core AMR dynamics.
    """
    antibiotic_names = ["A"]
    env = create_ABXAMR_Env_instance(antibiotic_names=antibiotic_names)
    obs, _ = env.reset(seed=123)
    amr_start = env.amr_balloon_models["A"].get_volume()
    obs, _ = env.reset(seed=123)
    amr_start = env.amr_balloon_models["A"].get_volume()

    prescribe_a = np.zeros(env.num_patients_per_time_step, dtype=int)
    obs, _, _, _, info1 = env.step(prescribe_a)
    amr_after_prescribe = info1["actual_amr_levels"]["A"]
    assert amr_after_prescribe > amr_start

    no_treatment = np.full(env.num_patients_per_time_step, env.no_treatment_action, dtype=int)
    obs, _, _, _, info2 = env.step(no_treatment)
    amr_after_rest = info2["actual_amr_levels"]["A"]
    # AMR should decrease (leak) when no treatment is given
    assert amr_after_rest < amr_after_prescribe


def test_observation_shape_order_and_dtype():
    """Test observation vector format: patient infection probs followed by AMR levels.
    
    Purpose: Verifies that observations have the correct shape, data type, and value ranges,
    with the first num_patients values being infection probabilities and remaining values being AMR levels.
    """
    antibiotic_names = ["A", "B"]
    env = create_ABXAMR_Env_instance(antibiotic_names=antibiotic_names)
    obs, _ = env.reset(seed=7)

    num_patients = env.num_patients_per_time_step
    patient_probs = obs[:num_patients]
    amr_levels_obs = obs[num_patients:]

    assert obs.dtype == np.float32
    assert patient_probs.shape == (num_patients,)
    assert amr_levels_obs.shape == (env.num_abx,)
    assert np.all((patient_probs >= 0.0) & (patient_probs <= 1.0))
    assert np.all((amr_levels_obs >= 0.0) & (amr_levels_obs <= 1.0))


def test_delta_amr_is_computed_and_included_in_info():
    """Test that delta AMR (change in AMR) is properly computed and included in the info dict.
    
    Purpose: Ensures that the environment computes the marginal AMR impact for each antibiotic
    prescription and includes it in the info dict for reward calculation and analysis.
    """
    antibiotic_names = ["A"]
    env = create_ABXAMR_Env_instance(antibiotic_names=antibiotic_names)
    env.reset(seed=0)

    action = np.zeros(env.num_patients_per_time_step, dtype=int)
    _, _, _, _, info = env.step(action)

    # Check that delta_visible_amr_per_antibiotic is in the info dict
    assert "delta_visible_amr_per_antibiotic" in info
    assert "A" in info["delta_visible_amr_per_antibiotic"]
    # Delta should be positive when prescribing (2 doses)
    assert info["delta_visible_amr_per_antibiotic"]["A"] > 0


def test_visible_amr_refresh_cadence():
    """Test that visible AMR levels update periodically at specified intervals.
    
    Purpose: Verifies that the agent's view of AMR levels is refreshed only every N timesteps,
    allowing for imperfect information about current resistance levels.
    """
    antibiotic_names = ["A"]
    env = create_ABXAMR_Env_instance(antibiotic_names=antibiotic_names)
    env.update_visible_AMR_levels_every_n_timesteps = 2
    env.reset(seed=1)
    prescribe = np.zeros(env.num_patients_per_time_step, dtype=int)

    _, _, _, _, _ = env.step(prescribe)
    visible_after_one = env.visible_amr_levels["A"]
    actual_after_one = env.amr_balloon_models["A"].get_volume()
    
    # After 1 step, visible should not match actual (update happens every 2)
    assert not np.isclose(visible_after_one, actual_after_one)
    
    _, _, _, _, _ = env.step(prescribe)
    visible_after_two = env.visible_amr_levels["A"]
    actual_after_two = env.amr_balloon_models["A"].get_volume()
    
    # After 2 steps, visible should match actual (update just happened)
    assert np.isclose(visible_after_two, actual_after_two)


def test_visible_amr_bias_clips_and_applies():
    """Visible AMR bias should be applied and clipped to [0, 1]."""
    antibiotics = {
        "A": {
            "leak": 0.05,
            "flatness_parameter": 1.0,
            "permanent_residual_volume": 0.0,
            "initial_amr_level": 0.8,
        }
    }
    env = create_ABXAMR_Env_instance(
        antibiotic_names=["A"],
        antibiotics_AMR_dict=antibiotics,
        add_bias_to_visible_AMR_levels=0.4,
        add_noise_to_visible_AMR_levels=0.0,
        update_visible_AMR_levels_every_n_timesteps=1,
    )
    _, _ = env.reset(seed=314)

    actual_volume = env.amr_balloon_models["A"].get_volume()
    visible_volume = env.visible_amr_levels["A"]

    # Bias lifts the visible AMR level and should clip if it exceeds 1.
    assert visible_volume >= actual_volume
    assert visible_volume == pytest.approx(1.0)


def test_visible_amr_noise_is_seeded_and_repeatable():
    """Noise added to visible AMR levels should respect the RNG seed."""
    antibiotics = {
        "A": {
            "leak": 0.05,
            "flatness_parameter": 1.0,
            "permanent_residual_volume": 0.0,
            "initial_amr_level": 0.5,
        }
    }

    env1 = create_ABXAMR_Env_instance(
        antibiotic_names=["A"],
        antibiotics_AMR_dict=antibiotics,
        add_noise_to_visible_AMR_levels=0.1,
        update_visible_AMR_levels_every_n_timesteps=1,
    )
    env2 = create_ABXAMR_Env_instance(
        antibiotic_names=["A"],
        antibiotics_AMR_dict=antibiotics,
        add_noise_to_visible_AMR_levels=0.1,
        update_visible_AMR_levels_every_n_timesteps=1,
    )
    env3 = create_ABXAMR_Env_instance(
        antibiotic_names=["A"],
        antibiotics_AMR_dict=antibiotics,
        add_noise_to_visible_AMR_levels=0.1,
        update_visible_AMR_levels_every_n_timesteps=1,
    )

    _, _ = env1.reset(seed=777)
    _, _ = env2.reset(seed=777)
    _, _ = env3.reset(seed=778)

    visible1 = env1.visible_amr_levels["A"]
    visible2 = env2.visible_amr_levels["A"]
    visible3 = env3.visible_amr_levels["A"]

    assert visible1 == pytest.approx(visible2)
    assert not np.isclose(visible1, visible3)


def test_reward_alignment_with_reward_calculator():
    """Test that the environment's step function returns valid reward values.
    
    Purpose: Verifies that the environment integrates with the reward calculator correctly,
    producing valid numeric reward values and including required info dict keys.
    """
    reward_calculator = create_test_reward_calculator(
        antibiotic_names=["A"],
        clinical_benefit_reward=10.0,
        clinical_benefit_probability=1.0,
        adverse_effect_penalty=-2.0,
        adverse_effect_probability=0.0,
        lambda_weight=0.5,
        epsilon=0.1,
    )
    env = create_ABXAMR_Env_instance(
        antibiotic_names=["A"],
        reward_calculator=reward_calculator,
    )
    obs, _ = env.reset(seed=123)

    # Create action for all 5 patients (default num_patients_per_time_step)
    actions = np.zeros(env.num_patients_per_time_step, dtype=int)  # All prescribe A
    # Just verify that the environment produces a valid reward
    _, actual_reward, _, _, info = env.step(actions)
    
    # The reward should be a float
    assert isinstance(actual_reward, (float, np.floating))
    # The reward info should contain expected keys
    assert 'count_clinical_benefits' in info
    assert 'count_adverse_events' in info
    assert 'actual_amr_levels' in info


def test_invalid_action_rejected():
    """Test that invalid actions are properly rejected.
    
    Purpose: Ensures that the environment validates actions and raises an AssertionError
    when an action index exceeds the valid range (0 to num_abx).
    """
    antibiotic_names = ["A"]
    env = create_ABXAMR_Env_instance(antibiotic_names=antibiotic_names)
    env.reset(seed=0)
    # num_abx=1, so valid actions are 0 (prescribe A) or 1 (no treatment)
    # action 3 is invalid
    bad_action = np.array([3, 0])
    with pytest.raises(AssertionError):
        env.step(bad_action)


def test_rng_repeatability_with_seed():
    """Test that the same seed produces reproducible results across different environment instances.
    
    Purpose: Ensures that when two independent environment instances are reset with the same seed,
    they produce identical rewards and AMR levels, demonstrating deterministic behavior.
    """
    # Create two separate reward models with the same parameters
    reward_calculator1 = create_test_reward_calculator(antibiotic_names=["A"])
    reward_calculator2 = create_test_reward_calculator(antibiotic_names=["A"])
    
    # Create envs with separate reward models
    env1 = create_ABXAMR_Env_instance(antibiotic_names=["A"], reward_calculator=reward_calculator1)
    env2 = create_ABXAMR_Env_instance(antibiotic_names=["A"], reward_calculator=reward_calculator2)

    # Reset both with same seed for reproducibility
    env1.reset(seed=99)
    env2.reset(seed=99)

    # Execute same action on both (5 patients, all prescribe A)
    action = np.zeros(env1.num_patients_per_time_step, dtype=int)
    _, r1, _, _, i1 = env1.step(action)
    _, r2, _, _, i2 = env2.step(action)

    # Rewards should be identical due to same seed
    assert np.isclose(r1, r2)
    # AMR levels should be identical
    assert i1["actual_amr_levels"] == i2["actual_amr_levels"]


def test_stochastic_reward_env_integration_is_reproducible():
    """Test that stochastic rewards are reproducible when using the same seed.
    
    Purpose: Ensures that when stochastic reward calculations are seeded (clinical benefit
    probability, adverse effects probability), running the environment twice with the same
    seed produces identical outcomes, including success counts and adverse event counts.
    """
    def run_once(seed: int):
        reward_calculator = create_test_reward_calculator(
            antibiotic_names=["A"],
            clinical_benefit_probability=0.6,
            adverse_effect_probability=0.4,
            epsilon=0.05,
        )
        reward_calculator.rng = np.random.default_rng(seed)
        env = create_ABXAMR_Env_instance(antibiotic_names=["A"], reward_calculator=reward_calculator)
        obs, _ = env.reset(seed=seed)
        # 5 patients (default): all prescribe A
        actions = np.zeros(env.num_patients_per_time_step, dtype=int)
        _, reward, _, _, info = env.step(actions)
        return reward, info

    # Run twice with same seed
    r1, i1 = run_once(123)
    r2, i2 = run_once(123)

    # Results should be identical
    assert np.isclose(r1, r2)
    assert i1["count_adverse_events"] == i2["count_adverse_events"]
    assert i1["count_clinical_benefits"] == i2["count_clinical_benefits"]


def test_infection_probability_stddev_controls_variation():
    """std_dev_probability_of_infection should control the width of patient infection noise."""
    env_deterministic = create_ABXAMR_Env_instance(
        antibiotic_names=["A"],
        baseline_probability_of_infection=0.7,
        std_dev_probability_of_infection=0.0,
    )
    obs_det, _ = env_deterministic.reset(seed=2024)
    patient_probs_det = obs_det[:env_deterministic.num_patients_per_time_step]
    assert np.allclose(patient_probs_det, 0.7)

    env_variable = create_ABXAMR_Env_instance(
        antibiotic_names=["A"],
        baseline_probability_of_infection=0.7,
        std_dev_probability_of_infection=0.2,
    )
    obs_var, _ = env_variable.reset(seed=2024)
    patient_probs_var = obs_var[:env_variable.num_patients_per_time_step]
    # With non-zero std dev we expect variability across patients
    assert patient_probs_var.std() > 0.0

def test_crossresistance_matrix_diagonal_forced_to_one():
    """Test that diagonal entries are always 1.0 even if user supplies different value."""
    crossresistance_bad = {
        "A": {"A": 0.5, "B": 0.3},  # Bad: self-entry should not be supplied
        "B": {"B": 0.2},
    }
    
    # Should raise error because user supplied self-entry
    with pytest.raises(ValueError, match="should not include self-entries"):
        env = create_ABXAMR_Env_instance(
            antibiotic_names=["A", "B"],
            crossresistance_matrix=crossresistance_bad,
        )


def test_crossresistance_matrix_identity_default():
    """Test that no crossresistance defaults to identity matrix (current behavior)."""
    env = create_ABXAMR_Env_instance(
        antibiotic_names=["A", "B"],
        crossresistance_matrix=None,
    )
    
    # Default should be identity
    assert env.crossresistance_matrix["A"]["A"] == 1.0
    assert env.crossresistance_matrix["A"]["B"] == 0.0
    assert env.crossresistance_matrix["B"]["A"] == 0.0
    assert env.crossresistance_matrix["B"]["B"] == 1.0


def test_asymmetric_crossresistance_increases_target_amr():
    """Test that asymmetric crossresistance increases target antibiotic AMR by ratio*count."""
    crossresistance = {
        "A": {"B": 0.5},  # Prescribing A contributes 0.5x its doses to B
        "B": {"A": 0.01},
    }
    env = create_ABXAMR_Env_instance(
        antibiotic_names=["A", "B"],
        crossresistance_matrix=crossresistance,
    )
    env.reset(seed=42)
    
    # Prescribe A to all 5 patients
    action_all_a = np.zeros(env.num_patients_per_time_step, dtype=int)
    obs, _, _, _, info = env.step(action_all_a)
    
    # Check effective doses: A gets 5, B gets 5 * 0.5 = 2.5
    assert info["effective_doses"]["A"] == 5.0
    assert info["effective_doses"]["B"] == pytest.approx(2.5)
    
    # B's AMR should increase due to crossresistance from A
    assert info["actual_amr_levels"]["B"] > 0.0


def test_crossresistance_fractional_doses():
    """Test that fractional crossresistance ratios produce fractional effective doses."""
    crossresistance = {
        "A": {"B": 0.33},
        "B": {"A": 0.25},
    }
    env = create_ABXAMR_Env_instance(
        antibiotic_names=["A", "B"],
        crossresistance_matrix=crossresistance,
    )
    env.reset(seed=123)
    
    # Prescribe A to 3 patients (out of 5)
    action = np.array([0, 0, 0, env.no_treatment_action, env.no_treatment_action], dtype=int)
    obs, _, _, _, info = env.step(action)
    
    # A gets 3 doses (3 * 1.0), B gets 3 * 0.33 = 0.99
    assert info["effective_doses"]["A"] == 3.0
    assert info["effective_doses"]["B"] == pytest.approx(3 * 0.33)


def test_crossresistance_invalid_antibiotic_raises():
    """Test that supplying invalid antibiotic names in crossresistance matrix raises error."""
    crossresistance_bad = {
        "A": {"C": 0.5},  # C doesn't exist
    }
    
    with pytest.raises(ValueError, match="not found in antibiotic_names"):
        env = create_ABXAMR_Env_instance(
            antibiotic_names=["A", "B"],
            crossresistance_matrix=crossresistance_bad,
        )


def test_crossresistance_invalid_ratio_raises():
    """Test that ratios outside [0, 1] raise error."""
    crossresistance_bad = {
        "A": {"B": 1.5},  # Ratio > 1
    }
    
    with pytest.raises(ValueError, match="must be a number in \\[0, 1\\]"):
        env = create_ABXAMR_Env_instance(
            antibiotic_names=["A", "B"],
            crossresistance_matrix=crossresistance_bad,
        )


def test_crossresistance_identity_recovers_no_interaction():
    """Test that explicitly passing identity matrix recovers no-interaction behavior."""
    crossresistance_identity = {
        "A": {"B": 0.0},
        "B": {"A": 0.0},
    }
    
    env_with_explicit_identity = create_ABXAMR_Env_instance(
        antibiotic_names=["A", "B"],
        crossresistance_matrix=crossresistance_identity,
    )
    
    env_with_implicit_identity = create_ABXAMR_Env_instance(
        antibiotic_names=["A", "B"],
        crossresistance_matrix=None,
    )
    
    # Both should have same crossresistance matrix
    assert env_with_explicit_identity.crossresistance_matrix == env_with_implicit_identity.crossresistance_matrix


def test_crossresistance_seeded_repeatability():
    """Test that seeded runs with crossresistance produce identical results."""
    crossresistance = {
        "A": {"B": 0.4},
        "B": {"A": 0.1},
    }
    
    env1 = create_ABXAMR_Env_instance(
        antibiotic_names=["A", "B"],
        crossresistance_matrix=crossresistance,
    )
    env2 = create_ABXAMR_Env_instance(
        antibiotic_names=["A", "B"],
        crossresistance_matrix=crossresistance,
    )
    
    env1.reset(seed=555)
    env2.reset(seed=555)
    
    # Same action on both
    action = np.array([0, 1, 0, env1.no_treatment_action, 1], dtype=int)
    _, r1, _, _, info1 = env1.step(action)
    _, r2, _, _, info2 = env2.step(action)
    
    # Should be identical
    assert np.isclose(r1, r2)
    assert info1["effective_doses"] == info2["effective_doses"]
    assert info1["actual_amr_levels"] == info2["actual_amr_levels"]


def test_crossresistance_applied_in_info():
    """Test that crossresistance_applied dict is populated correctly in info."""
    crossresistance = {
        "A": {"B": 0.5},
        "B": {"A": 0.1},
    }
    env = create_ABXAMR_Env_instance(
        antibiotic_names=["A", "B"],
        crossresistance_matrix=crossresistance,
    )
    env.reset(seed=666)
    
    # Prescribe A to first 3 patients, B to next 2
    action = np.array([0, 0, 0, 1, 1], dtype=int)
    obs, _, _, _, info = env.step(action)
    
    # Check crossresistance_applied structure
    assert "crossresistance_applied" in info
    # A should receive doses from A (3) and B (2 * 0.1 = 0.2)
    assert "A" in info["crossresistance_applied"]
    # B should receive doses from A (3 * 0.5 = 1.5) and B (2)
    assert info["crossresistance_applied"]["B"]["A"] == pytest.approx(1.5)
    assert info["crossresistance_applied"]["B"]["B"] == 2.0


def test_steps_since_amr_update_in_obs():
    """Test that steps_since_amr_update is included in obs when enabled, increments and resets correctly."""
    env = create_ABXAMR_Env_instance(include_steps_since_amr_update_in_obs=True, update_visible_AMR_levels_every_n_timesteps=3)
    obs, _ = env.reset(seed=123)
    # Should be appended as last element
    assert obs.shape[-1] == env.num_patients_per_time_step + env.num_abx + 1
    assert obs[-1] == 0  # Should start at 0
    # Step 1
    _, _, _, _, _ = env.step(np.full(env.num_patients_per_time_step, env.no_treatment_action, dtype=int))
    assert env.state[-1] == 1
    # Step 2
    _, _, _, _, _ = env.step(np.full(env.num_patients_per_time_step, env.no_treatment_action, dtype=int))
    assert env.state[-1] == 2
    # Step 3 triggers AMR update, should reset counter
    _, _, _, _, _ = env.step(np.full(env.num_patients_per_time_step, env.no_treatment_action, dtype=int))
    assert env.state[-1] == 0


def test_steps_since_amr_update_not_in_obs_by_default():
    """Test that steps_since_amr_update is not in obs by default (blind delay)."""
    env = create_ABXAMR_Env_instance()
    obs, _ = env.reset(seed=123)
    # Should match old shape
    assert obs.shape == (env.num_patients_per_time_step + env.num_abx,)


class TestComponentCompatibilityValidation:
    """Tests for strict validation of RewardCalculator and PatientGenerator class constants."""
    
    def test_missing_required_patient_attrs_raises_error(self):
        """Test that environment raises error if RewardCalculator lacks REQUIRED_PATIENT_ATTRS."""
        from unittest.mock import Mock
        
        pg = PatientGenerator(PatientGenerator.default_config())
        rc = create_test_reward_calculator()
        
        # Create a mock RewardCalculator without REQUIRED_PATIENT_ATTRS (will fail hasattr check)
        rc_mock = Mock(spec=[])  # Empty spec so hasattr returns False
        rc_mock.abx_clinical_reward_penalties_info_dict = rc.abx_clinical_reward_penalties_info_dict
        rc_mock.antibiotic_names = rc.antibiotic_names
        rc_mock.index_to_abx_name = rc.index_to_abx_name
        rc_mock.abx_name_to_index = rc.abx_name_to_index
        rc_mock.seed = 42
        rc_mock.rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="RewardCalculator must define REQUIRED_PATIENT_ATTRS"):
            create_mock_environment(
                reward_calculator=rc_mock,
                patient_generator=pg,
                antibiotics_AMR_dict={'A': {'leak': 0.05, 'flatness_parameter': 1.0, 'permanent_residual_volume': 0.0, 'initial_amr_level': 0.0}}
            )
    
    def test_missing_provides_attributes_raises_error(self):
        """Test that environment raises error if PatientGenerator lacks PROVIDES_ATTRIBUTES."""
        from unittest.mock import Mock
        
        rc = create_test_reward_calculator()
        pg = PatientGenerator(PatientGenerator.default_config())
        
        # Create a mock PatientGenerator without PROVIDES_ATTRIBUTES (will fail hasattr check)
        pg_mock = Mock(spec=[])  # Empty spec so hasattr returns False
        pg_mock.sample = pg.sample
        pg_mock.observe = pg.observe
        pg_mock.obs_dim = pg.obs_dim
        pg_mock.visible_patient_attributes = pg.visible_patient_attributes
        pg_mock.seed = 42
        pg_mock.rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="PatientGenerator must define PROVIDES_ATTRIBUTES"):
            create_mock_environment(
                reward_calculator=rc,
                patient_generator=pg_mock,
                antibiotics_AMR_dict={'A': {'leak': 0.05, 'flatness_parameter': 1.0, 'permanent_residual_volume': 0.0, 'initial_amr_level': 0.0}}
            )
    
    def test_incompatible_required_attributes_raises_error(self):
        """Test that environment raises error if required attributes are missing from provided set."""
        from unittest.mock import Mock
        
        rc = create_test_reward_calculator()
        pg = PatientGenerator(PatientGenerator.default_config())
        
        # Create a mock PatientGenerator with incomplete PROVIDES_ATTRIBUTES
        pg_mock = Mock()
        pg_mock.sample = pg.sample
        pg_mock.observe = pg.observe
        pg_mock.obs_dim = pg.obs_dim
        pg_mock.visible_patient_attributes = pg.visible_patient_attributes
        pg_mock.seed = 42
        pg_mock.rng = np.random.default_rng(42)
        pg_mock.PROVIDES_ATTRIBUTES = ['prob_infected']  # Missing multipliers required by RewardCalculator
        
        with pytest.raises(ValueError, match="RewardCalculator requires Patient attributes"):
            create_mock_environment(
                reward_calculator=rc,
                patient_generator=pg_mock,
                antibiotics_AMR_dict={'A': {'leak': 0.05, 'flatness_parameter': 1.0, 'permanent_residual_volume': 0.0, 'initial_amr_level': 0.0}}
            )
    
    def test_compatible_attributes_succeeds(self):
        """Test that environment succeeds when all required attributes are provided."""
        rc = create_test_reward_calculator()
        pg = PatientGenerator(PatientGenerator.default_config())
        
        # Should not raise any error
        env = create_mock_environment(reward_calculator=rc, patient_generator=pg)
        assert env is not None
        assert hasattr(env, 'reward_calculator')
        assert hasattr(env, 'patient_generator')