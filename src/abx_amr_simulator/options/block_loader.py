"""
Short alias module for block option loader.

This module re-exports the block option loader from the defaults package,
providing a convenient short module path: abx_amr_simulator.options.block_loader

Usage in option libraries:
    loader_module: "abx_amr_simulator.options.block_loader"
"""

from abx_amr_simulator.options.defaults.option_types.block.block_option_loader import (
    BlockOption,
    load_block_option,
)

__all__ = ["BlockOption", "load_block_option"]
