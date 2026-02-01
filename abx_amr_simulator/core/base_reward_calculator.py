"""
Abstract base class for reward calculators.

Defines the contract that all reward calculator implementations must follow,
including required patient attributes and calculation methods.
"""

from abc import ABC, abstractmethod
from typing import List, ClassVar, Any
import numpy as np


class RewardCalculatorBase(ABC):
    """
    Abstract base class for reward calculation.
    
    Subclasses must:
    1. Define REQUIRED_PATIENT_ATTRS class constant listing required Patient attributes
    2. Implement calculate_reward() method accepting List[Patient]
    """
    
    # Class constant declaring which Patient attributes are required for reward calculation
    REQUIRED_PATIENT_ATTRS: ClassVar[List[str]] = []
    
    @abstractmethod
    def calculate_reward(
        self,
        patients: List[Any],
        actions: np.ndarray,
        current_AMR_levels: np.ndarray,
        **kwargs
    ) -> float:
        """
        Calculate reward based on patient states, actions, and AMR levels.
        
        Args:
            patients: List of patient objects with attributes in REQUIRED_PATIENT_ATTRS
            actions: Binary action array indicating which antibiotics were prescribed
            current_AMR_levels: Current antimicrobial resistance levels for each antibiotic
            **kwargs: Additional context (e.g., infection_outcomes, effective_treatments)
            
        Returns:
            Scalar reward value for the step
        """
        pass
