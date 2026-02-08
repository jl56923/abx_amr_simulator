"""Hierarchical RL module for abx_amr_simulator.

This module provides the infrastructure for hierarchical reinforcement learning,
including the option framework for temporal abstraction.

Key components:
    - OptionBase: Abstract base class for all options
    - OptionLibrary: Container for options with validation
    - OptionLibraryLoader: Loads option libraries from YAML configs
    - OptionsWrapper: Gymnasium wrapper that executes options
"""

from abx_amr_simulator.hrl.base_option import OptionBase
from abx_amr_simulator.hrl.options import OptionLibrary
from abx_amr_simulator.hrl.option_loaders import OptionLibraryLoader
from abx_amr_simulator.hrl.wrapper import OptionsWrapper

__all__ = [
    'OptionBase',
    'OptionLibrary',
    'OptionLibraryLoader',
    'OptionsWrapper',
]
