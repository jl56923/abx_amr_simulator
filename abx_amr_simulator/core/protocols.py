"""
Protocol definitions for type hints and duck typing.

Defines structural types for patient generation and reward calculation
to enable flexible composition without tight coupling.
"""

from typing import Protocol, runtime_checkable, List, Any, Dict
import numpy as np


@runtime_checkable
class PatientProvider(Protocol):
    """
    Protocol for objects that can generate patient populations.
    
    Implementations should provide a sample() method that returns
    a list of patient objects with standardized attributes.
    """
    
    def sample(self, n: int, true_amr_levels: Dict[str, float], rng: np.random.Generator) -> List[Any]:
        """
        Generate n patient objects with heterogeneous attributes.
        
        Args:
            n: Number of patients to generate
            true_amr_levels: Ground-truth AMR levels per antibiotic (used for sensitivity sampling)
            rng: NumPy random number generator for reproducibility
            
        Returns:
            List of patient objects (typically Patient dataclass instances)
        """
        ...


@runtime_checkable
class PatientConsumer(Protocol):
    """
    Protocol for objects that consume patient data for reward calculation.
    
    Implementations should provide a calculate_reward() method that accepts
    a list of patient objects along with action and environment state data.
    """
    
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
            patients: List of patient objects with attributes (prob_infected, multipliers, etc.)
            actions: Binary action array indicating which antibiotics were prescribed
            current_AMR_levels: Current antimicrobial resistance levels for each antibiotic
            **kwargs: Additional context (e.g., infection_outcomes, effective_treatments)
            
        Returns:
            Scalar reward value for the step
        """
        ...
