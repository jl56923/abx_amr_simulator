"""
Deterministic option library for HRL.

Each option is a fixed macro-action sequence that repeats for k steps.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Option:
    """Represents a deterministic macro-action."""
    
    option_id: int
    name: str
    description: str
    action_sequence: List[int]  # List of base-level actions (0=no rx, 1=drug A, 2=drug B, etc.)
    duration: int  # k: number of steps to repeat
    
    def __post_init__(self):
        """Validate option definition."""
        if len(self.action_sequence) != self.duration:
            raise ValueError(
                f"Option {self.name}: action_sequence length ({len(self.action_sequence)}) "
                f"must equal duration ({self.duration})"
            )
        if not all(isinstance(a, (int, np.integer)) for a in self.action_sequence):
            raise ValueError(f"Option {self.name}: all actions must be integers")
        if min(self.action_sequence) < 0 or max(self.action_sequence) > 2:
            raise ValueError(
                f"Option {self.name}: actions must be in [0, 1, 2] (no_rx, drug_A, drug_B)"
            )
    
    def get_action(self, step_within_option: int) -> int:
        """Get action at a specific step within the option execution."""
        if step_within_option < 0 or step_within_option >= self.duration:
            raise IndexError(
                f"Step {step_within_option} out of bounds for option {self.name} (duration {self.duration})"
            )
        return self.action_sequence[step_within_option]


class OptionLibrary:
    """Container and manager for deterministic options."""
    
    def __init__(self, options: List[Option]):
        """Initialize option library."""
        self.options = options
        self.option_by_id = {opt.option_id: opt for opt in options}
        self.num_options = len(options)
        
        # Validate uniqueness
        if len(self.option_by_id) != len(options):
            raise ValueError("Option IDs must be unique")
    
    def get_option(self, option_id: int) -> Option:
        """Retrieve option by id."""
        if option_id not in self.option_by_id:
            raise KeyError(f"Option ID {option_id} not found in library")
        return self.option_by_id[option_id]
    
    def list_options(self) -> List[Option]:
        """Return all options."""
        return self.options
    
    def __len__(self):
        return self.num_options
    
    def __repr__(self):
        return f"OptionLibrary(num_options={self.num_options})"


def get_default_option_library() -> OptionLibrary:
    """
    Create default option library for Stage 1.
    
    Includes:
    - Block options: A×{5,10,15}, B×{5,10,15}, NO_RX×{5,10,15} (9 options)
    - Alternation options: A,A,B,B,A; B,B,A,A,B; A,NO_RX,B,A,NO_RX (3 options)
    
    Total: 12 deterministic options.
    """
    options = []
    option_id = 0
    
    # Block options: NO_RX
    for k in [5, 10, 15]:
        options.append(Option(
            option_id=option_id,
            name=f"NO_RX×{k}",
            description=f"Prescribe nothing for {k} steps",
            action_sequence=[0] * k,
            duration=k,
        ))
        option_id += 1
    
    # Block options: Drug A
    for k in [5, 10, 15]:
        options.append(Option(
            option_id=option_id,
            name=f"A×{k}",
            description=f"Prescribe drug A for {k} steps",
            action_sequence=[1] * k,
            duration=k,
        ))
        option_id += 1
    
    # Block options: Drug B
    for k in [5, 10, 15]:
        options.append(Option(
            option_id=option_id,
            name=f"B×{k}",
            description=f"Prescribe drug B for {k} steps",
            action_sequence=[2] * k,
            duration=k,
        ))
        option_id += 1
    
    # Alternation options
    alternations = [
        ([1, 1, 2, 2, 1], "A,A,B,B,A"),
        ([2, 2, 1, 1, 2], "B,B,A,A,B"),
        ([1, 0, 2, 1, 0], "A,NO_RX,B,A,NO_RX"),
    ]
    
    for action_seq, name in alternations:
        options.append(Option(
            option_id=option_id,
            name=name,
            description=f"Deterministic cycle: {name}",
            action_sequence=action_seq,
            duration=len(action_seq),
        ))
        option_id += 1
    
    return OptionLibrary(options)
