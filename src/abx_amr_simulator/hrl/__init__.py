"""
Hierarchical Reinforcement Learning (HRL) for ABX-AMR optimization.

Implements options-based temporal abstraction via deterministic macro-actions.
"""

from .options import Option, OptionLibrary, get_default_option_library
from .manager_obs import ManagerObsBuilder
from .wrapper import OptionsWrapper

__all__ = [
    "Option",
    "OptionLibrary",
    "get_default_option_library",
    "ManagerObsBuilder",
    "OptionsWrapper",
]
