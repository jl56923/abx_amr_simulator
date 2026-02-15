"""
Short alias module for alternation option loader.

This module re-exports the alternation option loader from the defaults package,
providing a convenient short module path: abx_amr_simulator.options.alternation_loader

Usage in option libraries:
    loader_module: "abx_amr_simulator.options.alternation_loader"
"""

from abx_amr_simulator.options.defaults.option_types.alternation.alternation_option_loader import (
    AlternationOption,
    load_alternation_option,
)

__all__ = ["AlternationOption", "load_alternation_option"]
