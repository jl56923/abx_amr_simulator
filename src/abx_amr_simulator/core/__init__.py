"""ABX AMR Environment and related utilities."""

from .abx_amr_env import ABXAMREnv
from .reward_calculator import RewardCalculator
from .leaky_balloon import AMR_LeakyBalloon
from .types import Patient
from .patient_generator import PatientGenerator, PatientGeneratorMixer
from .base_patient_generator import PatientGeneratorBase
from .base_reward_calculator import RewardCalculatorBase
from .base_amr_dynamics import AMRDynamicsBase


def validate_compatibility(pg: PatientGeneratorBase, rc: RewardCalculatorBase) -> None:
    """
    Validate that a PatientGenerator and RewardCalculator are compatible.
    
    Checks that all attributes required by the RewardCalculator are provided
    by the PatientGenerator.
    
    Args:
        pg: PatientGenerator instance to validate
        rc: RewardCalculator instance to validate
        
    Raises:
        ValueError: If RewardCalculator requires attributes that PatientGenerator doesn't provide
    """
    if not hasattr(rc, 'REQUIRED_PATIENT_ATTRS') or not hasattr(pg, 'PROVIDES_ATTRIBUTES'):
        # Skip validation if either class doesn't declare its contract
        return
    
    required = set(rc.REQUIRED_PATIENT_ATTRS)
    provided = set(pg.PROVIDES_ATTRIBUTES)
    missing = required - provided
    
    if missing:
        raise ValueError(
            f"RewardCalculator requires Patient attributes {sorted(missing)} "
            f"but PatientGenerator doesn't provide them. "
            f"PatientGenerator provides: {sorted(provided)}, "
            f"RewardCalculator requires: {sorted(required)}"
        )


__all__ = [
    'ABXAMREnv',
    'RewardCalculator',
    'AMR_LeakyBalloon',
    'PatientGenerator',
    'PatientGeneratorMixer',
    'Patient',
    'PatientGeneratorBase',
    'RewardCalculatorBase',
    'AMRDynamicsBase',
    'validate_compatibility',
]
