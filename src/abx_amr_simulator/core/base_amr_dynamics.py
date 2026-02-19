"""Abstract base class for AMR dynamics models.

Defines the contract that all AMR dynamics implementations must follow.
AMR dynamics models translate antibiotic exposure (doses) into antimicrobial
resistance levels, ensuring bounded output in [0, 1].
"""

from abc import ABC, abstractmethod
from typing import ClassVar


class AMRDynamicsBase(ABC):
    """
    Abstract base class for antimicrobial resistance dynamics.
    
    Subclasses must implement methods to:
    1. Update dynamics state based on prescribing pressure (step)
    2. Reset internal state to a specified AMR level
    
    All implementations must ensure that:
    - Output (visible AMR level) is always in [0, 1]
    - The model is deterministic given the same sequence of inputs
    - Reset properly reinitializes all internal state
    
    Class Variables (must be set by subclasses):
        NAME: Unique identifier for this dynamics model (e.g., "leaky_balloon")
    
    Examples:
        >>> class CustomAMRModel(AMRDynamicsBase):
        ...     NAME = "custom_exponential"
        ...     def __init__(self):
        ...         self.level = 0.0
        ...     def step(self, doses):
        ...         self.level = min(1.0, self.level + 0.01 * doses)
        ...         return self.level
        ...     def reset(self, initial_level):
        ...         if not (0.0 <= initial_level <= 1.0):
        ...             raise ValueError("initial_level must be in [0, 1]")
        ...         self.level = initial_level
        >>> model = CustomAMRModel()
        >>> model.step(10)  # Returns resistance level
        0.1
    """
    
    # Class constant that subclasses MUST override
    NAME: ClassVar[str] = ""
    
    @abstractmethod
    def step(self, doses: float) -> float:
        """
        Execute one timestep of AMR dynamics.
        
        Updates internal state based on antibiotic exposure (doses) and returns
        the current observable AMR level (resistance fraction).
        
        Args:
            doses (float): Non-negative antibiotic exposure this timestep.
                Represents the total amount of antibiotic prescribing or equivalent
                driving force. Must be >= 0.
        
        Returns:
            float: Observable AMR level (resistance fraction) after applying dynamics.
                Must be in [0, 1].
        
        Raises:
            ValueError: If doses < 0.
        
        Example:
            >>> dynamics = AMR_LeakyBalloon(leak=0.1)
            >>> dynamics.step(5.0)  # Add 5 antibiotic doses
            0.42  # Returns current resistance level
        """
        pass
    
    @abstractmethod
    def reset(self, initial_level: float) -> None:
        """
        Reset AMR dynamics state to a specified level.
        
        Reinitializes all internal state (latent pressure, counters, etc.) such that
        the next call to step() will start from the specified initial_level.
        
        Args:
            initial_level (float): AMR level to reset to, in [0, 1].
                Represents the observable resistance fraction before any new doses.
        
        Raises:
            ValueError: If initial_level is not in [0, 1] (implementations should validate).
        
        Implementation note:
            Subclasses should validate that initial_level is within valid bounds before
            updating internal state. This ensures early error detection and prevents
            silent invalid state initialization.
        
        Example:
            >>> dynamics = AMR_LeakyBalloon()
            >>> dynamics.step(10)  # Increases resistance
            >>> dynamics.reset(0.2)  # Reset to 20% resistance
            >>> dynamics.step(0)  # Next step starts from 0.2
        """
        pass
