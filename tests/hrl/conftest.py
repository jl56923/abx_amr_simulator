"""Shared fixtures for HRL tests."""

import pytest
import sys
import os

# Import test helpers from the unit tests directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../unit/utils')))
from test_reference_helpers import (
    create_mock_environment,
    create_mock_patient_generator,
    create_mock_reward_calculator,
    create_mock_antibiotics_dict
)


@pytest.fixture
def simple_env():
    """Create a simple ABXAMREnv for HRL testing.
    
    Uses standard test helpers to create a real environment instance with:
    - 2 antibiotics: A, B
    - 1 patient per timestep
    - RewardCalculator with antibiotic name-to-index mapping
    - PatientGenerator with visible_patient_attributes
    - LeakyBalloon AMR dynamics
    
    Returns:
        ABXAMREnv: Real environment instance (not a mock)
    """
    return create_mock_environment(
        antibiotic_names=['A', 'B'],
        num_patients_per_time_step=1,
        max_time_steps=100,
        visible_patient_attributes=['prob_infected'],
    )


@pytest.fixture
def multi_patient_env():
    """Create environment with multiple patients per timestep.
    
    Returns:
        ABXAMREnv: Environment with 3 patients per timestep
    """
    return create_mock_environment(
        antibiotic_names=['A', 'B'],
        num_patients_per_time_step=3,
        max_time_steps=100,
        visible_patient_attributes=['prob_infected'],
    )

