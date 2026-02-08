"""Base abstract class for hierarchical RL options.

All options must inherit from OptionBase and implement the required abstract methods.
Options represent macro-actions—templates for multi-step behavior that the manager policy selects.

Design philosophy:
    - Options receive env_state (decoded, human-readable dict) instead of raw observation arrays.
    - Why env_state?
        - **Readability**: env_state['patients'] immediately tells you what it is; flattened arrays do not.
        - **Debugging**: Print env_state and see exactly what the option sees; no index mapping needed.
        - **Testing**: Construct simple dicts in tests; no need to build flattened arrays with index mappings.
        - **Performance**: Dict access is O(1) and negligible vs. NN inference in PPO/DQN; clarity > premature optimization.
        - **Alignment**: The codebase emphasizes explicit orchestration and clarity-first design.
    - Options declare their requirements via class variables (REQUIRES_OBSERVATION_ATTRIBUTES, etc.).
    - Validation occurs in three layers: Option creation, OptionLibrary compatibility, OptionsWrapper runtime.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, ClassVar
import numpy as np


class OptionBase(ABC):
    """Abstract base class for hierarchical RL options.
    
    All custom options must inherit from this class and implement the abstract methods.
    Options are deterministic (for MVP) or adaptive (conditional termination) templates
    that execute for up to k steps, collecting discounted rewards.
    
    Attributes:
        name: Unique machine-readable identifier for this option instance (e.g., 'A_5', 'ALT_AABBa').
        k: Maximum duration in steps. Either positive int or float('inf') for indefinite options.
        
    Class Variables (must be set by subclasses):
        REQUIRES_OBSERVATION_ATTRIBUTES: List of patient attribute names this option needs from env_state.
            Example: ['prob_infected'] means option only needs infection probability.
            Empty list [] means option is deterministic and doesn't need patient data.
        REQUIRES_AMR_LEVELS: Whether option needs current AMR levels in env_state.
            Use True if option adapts behavior to resistance levels; False if deterministic.
        REQUIRES_STEP_NUMBER: Whether option needs current episode step in env_state.
            Use True if option behavior changes over time; False otherwise.
        PROVIDES_TERMINATION_CONDITION: Whether this option can terminate early (before k steps).
            Use False for MVP (deterministic fixed-duration options).
    """

    # Class variables that subclasses MUST override
    REQUIRES_OBSERVATION_ATTRIBUTES: ClassVar[List[str]] = []
    REQUIRES_AMR_LEVELS: ClassVar[bool] = False
    REQUIRES_STEP_NUMBER: ClassVar[bool] = False
    PROVIDES_TERMINATION_CONDITION: ClassVar[bool] = False

    def __init__(self, name: str, k: int | float = None):
        """Initialize option instance.
        
        Args:
            name: Unique identifier for this option (e.g., 'A_5', 'HEURISTIC_001').
                  Must be unique within the option library.
            k: Maximum duration in steps. Can be:
               - Positive integer: fixed-duration option (runs exactly k steps unless episode ends)
               - float('inf'): indefinite option (runs until should_terminate() returns True or episode ends)
               Defaults to 1 if not specified.
        
        Raises:
            ValueError: If k is not positive int or float('inf').
        """
        if k is None:
            k = 1
        if not (isinstance(k, int) and k > 0) and k != float('inf'):
            raise ValueError(
                f"Option '{name}': k must be positive int or float('inf'), got {k} ({type(k).__name__})"
            )
        self.name = name
        self.k = k

    @abstractmethod
    def decide(self, env_state: Dict[str, Any], antibiotic_names: List[str]) -> np.ndarray:
        """Decide which actions to take for each patient at current step.
        
        This is the core decision-making method. It receives decoded environment state
        (not raw observation arrays) for clarity and maintainability.
        
        Args:
            env_state: Decoded environment context dict containing:
                - 'patients': List of dicts, one per patient, with observed attributes.
                  Keys in each dict correspond to the PatientGenerator's visible_patient_attributes.
                  Example: [{'prob_infected': 0.8, 'benefit_value_multiplier': 1.2}, ...]
                - 'num_patients': Number of patients per timestep.
                - 'current_amr_levels': Dict mapping antibiotic name -> resistance (float in [0, 1]).
                  Only present if REQUIRES_AMR_LEVELS=True; empty dict otherwise.
                - 'current_step': Current timestep in episode (int).
                  Only present if REQUIRES_STEP_NUMBER=True.
                - 'max_steps': Total episode length limit.
                  Only present if REQUIRES_STEP_NUMBER=True.
            antibiotic_names: Ordered list of antibiotic names from environment (e.g., ['A', 'B']).
                Used to convert antibiotic names to action indices.
        
        Returns:
            np.ndarray: Action indices, shape (num_patients,).
                Values: integers in [0, len(antibiotic_names)] where:
                    - 0 to len(antibiotic_names)-1: prescribe that antibiotic
                    - len(antibiotic_names): no treatment (NO_RX)
                dtype: np.int32
        
        Raises:
            ValueError: If configuration is invalid (e.g., antibiotic not in antibiotic_names).
            TypeError: If return type is not np.ndarray or if action values wrong type.
        
        Example (BlockOption - prescribe antibiotic 'A' to all patients):
            def decide(self, env_state, antibiotic_names):
                num_patients = env_state['num_patients']
                try:
                    action_idx = antibiotic_names.index(self.target_antibiotic)
                except ValueError:
                    raise ValueError(
                        f"Option '{self.name}': antibiotic '{self.target_antibiotic}' "
                        f"not in environment. Available: {antibiotic_names}"
                    )
                return np.full(shape=num_patients, fill_value=action_idx, dtype=np.int32)
        """
        pass

    def reset(self) -> None:
        """Reset internal state at episode start (optional override).
        
        Called by OptionsWrapper at episode reset. Subclasses should override
        if they maintain internal state (e.g., step counter for stateful options).
        
        Default implementation does nothing.
        
        Example (AlternationOption tracks step in sequence):
            def reset(self):
                self._step_index = 0
        """
        pass

    def should_terminate(self, env_state: Dict[str, Any]) -> bool:
        """Check if option should terminate early before k steps (optional override).
        
        Called at each substep if PROVIDES_TERMINATION_CONDITION=True.
        Allows adaptive termination (e.g., "exit if AMR too high" or "exit if oscillation detected").
        
        For MVP, return False (deterministic fixed-duration options only).
        Will be used in future stages for adaptive termination.
        
        Args:
            env_state: Same structure as decide()'s env_state parameter.
        
        Returns:
            bool: True if option should terminate this step, False otherwise.
        
        Default implementation returns False (no early termination).
        
        Example (future - terminate if AMR for target antibiotic exceeds threshold):
            def should_terminate(self, env_state):
                current_amr = env_state['current_amr_levels'].get(self.target_antibiotic, 0)
                return current_amr > 0.8
        """
        return False
