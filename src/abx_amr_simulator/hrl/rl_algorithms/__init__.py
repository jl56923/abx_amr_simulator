"""HRL RL algorithms with masking support for clipped manager transitions.

This package provides masked variants of standard RL algorithms for use in
hierarchical reinforcement learning (HRL) settings where manager transitions
can be clipped at episode boundaries.

Modules:
    - ppo_masked: PPO with support for masking clipped manager transitions
    - recurrent_ppo_masked: RecurrentPPO with support for masking clipped manager transitions
"""

from abx_amr_simulator.hrl.rl_algorithms.ppo_masked import PPO_Masked
from abx_amr_simulator.hrl.rl_algorithms.recurrent_ppo_masked import RecurrentPPO_Masked

__all__ = ['PPO_Masked', 'RecurrentPPO_Masked']
