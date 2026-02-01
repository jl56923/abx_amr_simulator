"""ABX AMR Simulator: reusable library for training and analyzing RL agents on antibiotic prescribing."""

__version__ = "0.1.0"
# Re-export core components for backward compatibility and convenience
from .core import (
    ABXAMREnv,
    RewardCalculator,
    AMR_LeakyBalloon,
    Patient,
    PatientGenerator,
    PatientGeneratorMixer,
    PatientGeneratorBase,
    RewardCalculatorBase,
    validate_compatibility,
)

__all__ = [
    'ABXAMREnv',
    'RewardCalculator',
    'AMR_LeakyBalloon',
    'Patient',
    'PatientGenerator',
    'PatientGeneratorMixer',
    'PatientGeneratorBase',
    'RewardCalculatorBase',
    'validate_compatibility',
]