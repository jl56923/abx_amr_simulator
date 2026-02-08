"""
Abstract base class for patient generators.

Defines the contract that all patient generator implementations must follow,
including required methods and attribute declarations.
"""

from abc import ABC, abstractmethod
from typing import List, ClassVar, Any, Dict
import numpy as np


class PatientGeneratorBase(ABC):
    """
    Abstract base class for patient generation.
    
    Subclasses must:
    1. Define PROVIDES_ATTRIBUTES class constant listing all Patient attributes
    2. Implement sample() method to generate patient populations
    """
    
    # Class constant declaring which attributes generated Patients will have
    PROVIDES_ATTRIBUTES: ClassVar[List[str]] = []

    # Instance attribute declaring which patient attributes are visible in observations
    visible_patient_attributes: List[str]
    
    @abstractmethod
    def sample(self, n: int, true_amr_levels: Dict[str, float], rng: np.random.Generator) -> List[Any]:
        """
        Generate n patient objects with heterogeneous attributes.
        
        Args:
            n: Number of patients to generate
            true_amr_levels: Ground-truth AMR levels per antibiotic (used for sensitivity sampling)
            rng: NumPy random number generator for reproducibility
            
        Returns:
            List of patient objects with attributes declared in PROVIDES_ATTRIBUTES
        """
        pass

    @abstractmethod
    def observe(self, patients: List[Any]) -> np.ndarray:
        """
        Project full Patient objects to observed features for the agent.
        Extracts *_obs variants of configured visible attributes in a fixed order.

        Args:
            patients: List of Patient objects (with true and observed attributes)

        Returns:
            1D numpy array of length (num_patients * num_visible_attributes)
        """
        pass

    @abstractmethod
    def obs_dim(self, num_patients: int) -> int:
        """
        Compute the observation dimension contributed by patient attributes.

        Args:
            num_patients: Number of patients in the cohort

        Returns:
            Integer dimension equal to num_patients * num_visible_attributes
        """
        pass
