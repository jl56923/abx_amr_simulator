"""End-to-end test for HRL training with real components.

Tests that:
1. Option library loads from real config files
2. OptionLibrary instantiates correctly with real options
3. OptionsWrapper wraps ABXAMREnv successfully
4. HRL_PPO agent trains without errors
5. Resolved option library config is generated

This test uses real instances (not mocks) of:
- ABXAMREnv
- RewardCalculator
- PatientGenerator
- OptionLibrary
- OptionsWrapper
- HRL_PPO (via stable-baselines3)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import numpy as np
from stable_baselines3 import PPO

from abx_amr_simulator.core import ABXAMREnv, RewardCalculator, PatientGenerator
from abx_amr_simulator.hrl import (
    OptionLibrary,
    OptionLibraryLoader,
    OptionsWrapper,
    setup_options_folders_with_defaults,
)
from abx_amr_simulator.utils import create_agent


# Reference test helper for creating real mock environments
def create_mock_patient_generator(
    baseline_probability_of_infection: float = 0.5,
    std_dev_probability_of_infection: float = 0.1,
) -> PatientGenerator:
    """Create a real PatientGenerator instance with test defaults."""
    config = {
        'prob_infected': {
            'prob_dist': {'type': 'gaussian', 'mu': baseline_probability_of_infection, 'sigma': std_dev_probability_of_infection},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'benefit_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_value_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'benefit_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'failure_probability_multiplier': {
            'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, None],
        },
        'recovery_without_treatment_prob': {
            'prob_dist': {'type': 'constant', 'value': 0.01},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.0,
            'obs_noise_std_dev_fraction': 0.0,
            'clipping_bounds': [0.0, 1.0],
        },
        'visible_patient_attributes': ['prob_infected'],
    }
    return PatientGenerator(config=config)


def create_mock_reward_calculator(antibiotic_names: list | None = None) -> RewardCalculator:
    """Create a real RewardCalculator instance with test defaults."""
    if antibiotic_names is None:
        antibiotic_names = ["A", "B"]
    config = {
        'abx_clinical_reward_penalties_info_dict': {
            'clinical_benefit_reward': 10.0,
            'clinical_benefit_probability': 1.0,
            'clinical_failure_penalty': -1.0,
            'clinical_failure_probability': 0.0,
            'abx_adverse_effects_info': {
                name: {
                    'adverse_effect_penalty': -2.0,
                    'adverse_effect_probability': 0.0,
                } for name in antibiotic_names
            },
        },
        'lambda_weight': 0.5,
        'epsilon': 0.05,
        'seed': 42,
    }
    return RewardCalculator(config=config)


def create_mock_environment(
    antibiotic_names: list | None = None,
    num_patients_per_time_step: int = 1,
    max_time_steps: int = 50,
) -> ABXAMREnv:
    """Create a real ABXAMREnv instance with test defaults."""
    if antibiotic_names is None:
        antibiotic_names = ["A", "B"]
    
    patient_generator = create_mock_patient_generator()
    reward_calculator = create_mock_reward_calculator(antibiotic_names)
    
    antibiotics_AMR_dict = {
        name: {
            'leak': 0.05,
            'flatness_parameter': 1.0,
            'permanent_residual_volume': 0.0,
            'initial_amr_level': 0.0
        } for name in antibiotic_names
    }
    
    return ABXAMREnv(
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        antibiotics_AMR_dict=antibiotics_AMR_dict,
        num_patients_per_time_step=num_patients_per_time_step,
        max_time_steps=max_time_steps,
    )


def _write_block_option_loader(target_dir: Path) -> None:
    """Write block_option_loader.py to target directory (with strict validation)."""
    code = '''"""Block option loader for deterministic antibiotic prescribing."""

from typing import Any, Dict, List

import numpy as np

from abx_amr_simulator.hrl import OptionBase


class BlockOption(OptionBase):
    """Deterministic option that repeats a single antibiotic for k steps."""

    REQUIRES_OBSERVATION_ATTRIBUTES: List[str] = []
    REQUIRES_AMR_LEVELS: bool = False
    REQUIRES_STEP_NUMBER: bool = False
    PROVIDES_TERMINATION_CONDITION: bool = False

    def __init__(self, name: str, antibiotic: str, duration: int) -> None:
        super().__init__(name=name, k=duration)
        self.antibiotic = antibiotic

    def decide(self, env_state: Dict[str, Any]) -> np.ndarray:
        num_patients = env_state["num_patients"]
        option_library = env_state["option_library"]
        
        # No normalization - antibiotic name must match exactly
        try:
            action_idx = option_library.abx_name_to_index[self.antibiotic]
        except KeyError as exc:
            available = list(option_library.abx_name_to_index.keys())
            raise ValueError(
                f"Option '{self.name}': antibiotic '{self.antibiotic}' not in environment. "
                f"Available: {available}. "
                f"Note: Use exactly 'no_treatment' (no variations like 'NO_RX', 'no_treat')."
            ) from exc

        return np.full(shape=num_patients, fill_value=action_idx, dtype=np.int32)

    def get_referenced_antibiotics(self) -> List[str]:
        """Return the single antibiotic this block option prescribes."""
        return [self.antibiotic]


def load_block_option(name: str, config: Dict[str, Any]) -> OptionBase:
    """Instantiate a BlockOption from config.

    Expected config keys:
        - antibiotic (str)
        - duration (int)
        - allowed_antibiotics (optional list[str])
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    if "antibiotic" not in config:
        raise ValueError("BlockOption config missing required key 'antibiotic'")
    if "duration" not in config:
        raise ValueError("BlockOption config missing required key 'duration'")

    antibiotic = config["antibiotic"]
    duration = config["duration"]

    if not isinstance(antibiotic, str) or not antibiotic:
        raise ValueError("BlockOption 'antibiotic' must be a non-empty string")
    if not isinstance(duration, int) or duration < 1:
        raise ValueError("BlockOption 'duration' must be an int >= 1")

    _validate_allowed_antibiotics(
        antibiotic=antibiotic,
        allowed_antibiotics=config.get("allowed_antibiotics"),
    )

    return BlockOption(name=name, antibiotic=antibiotic, duration=duration)


def _validate_allowed_antibiotics(antibiotic: str, allowed_antibiotics: Any) -> None:
    if allowed_antibiotics is None:
        return
    if not isinstance(allowed_antibiotics, list) or not all(
        isinstance(entry, str) for entry in allowed_antibiotics
    ):
        raise ValueError("allowed_antibiotics must be a list of strings")
    if antibiotic not in allowed_antibiotics:
        raise ValueError(
            f"antibiotic '{antibiotic}' not in allowed_antibiotics {allowed_antibiotics}"
        )
'''
    (target_dir / "block_option_loader.py").write_text(code, encoding="utf-8")


def _write_alternation_option_loader(target_dir: Path) -> None:
    """Write alternation_option_loader.py to target directory (with strict validation)."""
    code = '''"""Alternation option loader for cycling antibiotic sequences."""

from typing import Any, Dict, List

import numpy as np

from abx_amr_simulator.hrl import OptionBase


class AlternationOption(OptionBase):
    """Deterministic option that cycles through a sequence of antibiotics."""

    REQUIRES_OBSERVATION_ATTRIBUTES: List[str] = []
    REQUIRES_AMR_LEVELS: bool = False
    REQUIRES_STEP_NUMBER: bool = True
    PROVIDES_TERMINATION_CONDITION: bool = False

    def __init__(self, name: str, sequence: List[str]) -> None:
        super().__init__(name=name, k=len(sequence))
        self.sequence = sequence
        self._sequence_index = 0

    def decide(self, env_state: Dict[str, Any]) -> np.ndarray:
        num_patients = env_state["num_patients"]
        option_library = env_state["option_library"]
        
        # No normalization - get antibiotic name directly from sequence
        antibiotic_name = self.sequence[self._sequence_index]
        
        try:
            action_idx = option_library.abx_name_to_index[antibiotic_name]
        except KeyError as exc:
            available = list(option_library.abx_name_to_index.keys())
            raise ValueError(
                f"Option '{self.name}': antibiotic '{antibiotic_name}' not in environment. "
                f"Available: {available}. "
                f"Note: Use exactly 'no_treatment' (no variations like 'NO_RX', 'no_treat')."
            ) from exc

        # Advance to next in sequence for next call
        self._sequence_index = (self._sequence_index + 1) % len(self.sequence)
        
        return np.full(shape=num_patients, fill_value=action_idx, dtype=np.int32)

    def get_referenced_antibiotics(self) -> List[str]:
        """Return all antibiotics in this alternation sequence."""
        return list(self.sequence)


def load_alternation_option(name: str, config: Dict[str, Any]) -> OptionBase:
    """Instantiate an AlternationOption from config.

    Expected config keys:
        - sequence (list[str])
        - allowed_antibiotics (optional list[str])
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    if "sequence" not in config:
        raise ValueError("AlternationOption config missing required key 'sequence'")

    sequence = config["sequence"]

    if not isinstance(sequence, list) or len(sequence) == 0:
        raise ValueError("AlternationOption 'sequence' must be a non-empty list")
    if not all(isinstance(item, str) for item in sequence):
        raise ValueError("AlternationOption 'sequence' items must all be strings")

    for antibiotic in sequence:
        _validate_allowed_antibiotics(
            antibiotic=antibiotic,
            allowed_antibiotics=config.get("allowed_antibiotics"),
        )

    return AlternationOption(name=name, sequence=sequence)


def _validate_allowed_antibiotics(antibiotic: str, allowed_antibiotics: Any) -> None:
    if allowed_antibiotics is None:
        return
    if not isinstance(allowed_antibiotics, list) or not all(
        isinstance(entry, str) for entry in allowed_antibiotics
    ):
        raise ValueError("allowed_antibiotics must be a list of strings")
    if antibiotic not in allowed_antibiotics:
        raise ValueError(
            f"antibiotic '{antibiotic}' not in allowed_antibiotics {allowed_antibiotics}"
        )
'''
    (target_dir / "alternation_option_loader.py").write_text(code, encoding="utf-8")


@pytest.fixture
def temp_options_dir():
    """Create a real (persistent) options directory with scaffolding and loaders for testing."""
    # Use a real folder in the test workspace instead of tempfile for debugging bundled copies
    test_workspace = Path(__file__).parent.parent.parent.parent.parent.parent / ".test_hrl_workspace"
    test_workspace.mkdir(parents=True, exist_ok=True)
    
    # Clean up before creating fresh fixtures
    if test_workspace.exists():
        shutil.rmtree(test_workspace)
    test_workspace.mkdir(parents=True, exist_ok=True)
    
    # Create scaffolding
    setup_options_folders_with_defaults(target_path=test_workspace)
    
    # Write option type loaders (with UPDATED implementations that include get_referenced_antibiotics)
    block_dir = test_workspace / "option_types" / "block"
    alternation_dir = test_workspace / "option_types" / "alternation"
    _write_block_option_loader(block_dir)
    _write_alternation_option_loader(alternation_dir)
    
    # Write block option default config
    (block_dir / "block_option_default_config.yaml").write_text(
        "# Default block option config\n"
        "antibiotic: 'A'\n"
        "duration: 5\n",
        encoding="utf-8"
    )
    
    # Write alternation option default config
    (alternation_dir / "alternation_option_default_config.yaml").write_text(
        "# Default alternation option config\n"
        "sequence:\n"
        "  - 'A'\n"
        "  - 'B'\n",
        encoding="utf-8"
    )
    
    try:
        yield test_workspace
    finally:
        # Clean up after test completes
        if test_workspace.exists():
            shutil.rmtree(test_workspace)


def test_option_library_loads_from_real_configs(temp_options_dir):
    """Test that OptionLibrary loads from real option config files."""
    env = create_mock_environment(antibiotic_names=["A", "B"], num_patients_per_time_step=1)
    
    # Create a minimal library config
    library_config_path = temp_options_dir / "option_libraries" / "test_library.yaml"
    library_config_path.write_text(
        "library_name: 'test_library'\n"
        "description: 'Test library with 2 simple options'\n"
        "version: '1.0'\n"
        "options:\n"
        "  - option_name: 'A_5'\n"
        "    option_type: 'block'\n"
        "    option_subconfig_file: '../option_types/block/block_option_default_config.yaml'\n"
        "    loader_module: '../option_types/block/block_option_loader.py'\n"
        "    config_params_override:\n"
        "      antibiotic: 'A'\n"
        "      duration: 5\n"
        "  - option_name: 'ALT_A_B'\n"
        "    option_type: 'alternation'\n"
        "    option_subconfig_file: '../option_types/alternation/alternation_option_default_config.yaml'\n"
        "    loader_module: '../option_types/alternation/alternation_option_loader.py'\n"
        "    config_params_override:\n"
        "      sequence:\n"
        "        - 'A'\n"
        "        - 'B'\n",
        encoding="utf-8"
    )
    
    # Load library
    library, resolved_config = OptionLibraryLoader.load_library(
        library_config_path=str(library_config_path),
        env=env,
    )
    
    # Verify library loaded
    assert len(library) == 2
    assert "A_5" in library.list_options()
    assert "ALT_A_B" in library.list_options()
    
    # Verify resolved config
    assert resolved_config is not None
    assert "options" in resolved_config


def test_options_wrapper_wraps_environment(temp_options_dir):
    """Test that OptionsWrapper correctly wraps ABXAMREnv."""
    env = create_mock_environment(antibiotic_names=["A", "B"], num_patients_per_time_step=1, max_time_steps=50)
    
    # Create a minimal library config
    library_config_path = temp_options_dir / "option_libraries" / "test_library.yaml"
    library_config_path.write_text(
        "library_name: 'test_library'\n"
        "options:\n"
        "  - option_name: 'A_5'\n"
        "    option_type: 'block'\n"
        "    option_subconfig_file: '../option_types/block/block_option_default_config.yaml'\n"
        "    loader_module: '../option_types/block/block_option_loader.py'\n"
        "    config_params_override:\n"
        "      antibiotic: 'A'\n"
        "      duration: 5\n",
        encoding="utf-8"
    )
    
    # Load library
    library, _ = OptionLibraryLoader.load_library(
        library_config_path=str(library_config_path),
        env=env,
    )
    
    # Wrap environment
    wrapped_env = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    
    # Verify wrapper properties
    assert wrapped_env is not None
    assert wrapped_env.observation_space is not None
    assert wrapped_env.action_space is not None
    assert wrapped_env.action_space.n == 1  # Only 1 option
    
    # Test reset
    obs, info = wrapped_env.reset(seed=42)
    assert obs is not None
    assert isinstance(obs, np.ndarray)


def test_hrl_ppo_agent_trains_without_error(temp_options_dir):
    """Test that HRL_PPO agent can train on wrapped environment."""
    env = create_mock_environment(antibiotic_names=["A", "B"], num_patients_per_time_step=1, max_time_steps=50)
    
    # Create a minimal library config
    library_config_path = temp_options_dir / "option_libraries" / "test_library.yaml"
    library_config_path.write_text(
        "library_name: 'test_library'\n"
        "options:\n"
        "  - option_name: 'A_5'\n"
        "    option_type: 'block'\n"
        "    option_subconfig_file: '../option_types/block/block_option_default_config.yaml'\n"
        "    loader_module: '../option_types/block/block_option_loader.py'\n"
        "    config_params_override:\n"
        "      antibiotic: 'A'\n"
        "      duration: 5\n"
        "  - option_name: 'no_treatment_5'\n"
        "    option_type: 'block'\n"
        "    option_subconfig_file: '../option_types/block/block_option_default_config.yaml'\n"
        "    loader_module: '../option_types/block/block_option_loader.py'\n"
        "    config_params_override:\n"
        "      antibiotic: 'no_treatment'\n"
        "      duration: 5\n",
        encoding="utf-8"
    )
    
    # Load library
    library, _ = OptionLibraryLoader.load_library(
        library_config_path=str(library_config_path),
        env=env,
    )
    
    # Wrap environment
    wrapped_env = OptionsWrapper(env=env, option_library=library, gamma=0.99)
    
    # CRITICAL: Call reset() before passing to PPO to ensure observation_space is properly sized
    wrapped_env.reset(seed=42)
    
    # Create agent
    agent = PPO(
        policy="MlpPolicy",
        env=wrapped_env,
        learning_rate=3.0e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        verbose=0,
        seed=42,
    )
    
    # Verify agent created
    assert agent is not None
    
    # Train for a short period
    agent.learn(total_timesteps=500)
    
    # Verify training completed
    assert agent.num_timesteps > 0


def test_end_to_end_with_real_factory_functions(temp_options_dir):
    """Test full end-to-end with config dict and create_agent factory."""
    from abx_amr_simulator.utils import (
        create_reward_calculator,
        create_patient_generator,
        create_environment,
        wrap_environment_for_hrl,
    )
    
    # Create a minimal library config
    library_config_path = temp_options_dir / "option_libraries" / "test_library.yaml"
    library_config_path.write_text(
        "library_name: 'test_library'\n"
        "options:\n"
        "  - option_name: 'A_5'\n"
        "    option_type: 'block'\n"
        "    option_subconfig_file: '../option_types/block/block_option_default_config.yaml'\n"
        "    loader_module: '../option_types/block/block_option_loader.py'\n"
        "    config_params_override:\n"
        "      antibiotic: 'A'\n"
        "      duration: 5\n"
        "  - option_name: 'no_treatment_5'\n"
        "    loader_module: '../option_types/block/block_option_loader.py'\n"
        "    option_type: 'block'\n"
        "    option_subconfig_file: '../option_types/block/block_option_default_config.yaml'\n"
        "    config_params_override:\n"
        "      antibiotic: 'no_treatment'\n"
        "      duration: 5\n",
        encoding="utf-8"
    )
    
    # Build config dict for factory functions
    config = {
        'algorithm': 'HRL_PPO',
        'training': {'seed': 42},
        'reward_calculator': {
            'abx_clinical_reward_penalties_info_dict': {
                'clinical_benefit_reward': 10.0,
                'clinical_benefit_probability': 1.0,
                'clinical_failure_penalty': -1.0,
                'clinical_failure_probability': 0.0,
                'abx_adverse_effects_info': {
                    'A': {
                        'adverse_effect_penalty': -2.0,
                        'adverse_effect_probability': 0.0,
                    },
                    'B': {
                        'adverse_effect_penalty': -2.0,
                        'adverse_effect_probability': 0.0,
                    }
                }
            },
            'lambda_weight': 0.5,
            'epsilon': 0.05,
        },
        'patient_generator': {
            'prob_infected': {
                'prob_dist': {'type': 'gaussian', 'mu': 0.5, 'sigma': 0.1},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'benefit_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_value_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'benefit_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'failure_probability_multiplier': {
                'prob_dist': {'type': 'gaussian', 'mu': 1.0, 'sigma': 1e-6},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, None],
            },
            'recovery_without_treatment_prob': {
                'prob_dist': {'type': 'constant', 'value': 0.01},
                'obs_bias_multiplier': 1.0,
                'obs_noise_one_std_dev': 0.0,
                'obs_noise_std_dev_fraction': 0.0,
                'clipping_bounds': [0.0, 1.0],
            },
            'visible_patient_attributes': ['prob_infected'],
        },
        'environment': {
            'antibiotics_AMR_dict': {
                'A': {
                    'leak': 0.05,
                    'flatness_parameter': 1.0,
                    'permanent_residual_volume': 0.0,
                    'initial_amr_level': 0.0
                },
                'B': {
                    'leak': 0.05,
                    'flatness_parameter': 1.0,
                    'permanent_residual_volume': 0.0,
                    'initial_amr_level': 0.0
                }
            },
            'num_patients_per_time_step': 1,
            'max_time_steps': 50,
        },
        'hrl': {
            'option_library': str(library_config_path),
            'option_gamma': 0.99,
        },
        'ppo': {
            'learning_rate': 3.0e-4,
            'n_steps': 128,
            'batch_size': 32,
            'n_epochs': 5,
            'gamma': 0.99,
            'verbose': 0,
        }
    }
    
    # Create components using factory functions
    rc = create_reward_calculator(config)
    assert rc is not None
    
    pg = create_patient_generator(config)
    assert pg is not None
    
    env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)
    assert env is not None
    
    # Wrap with HRL
    wrapped_env = wrap_environment_for_hrl(env=env, config=config)
    assert wrapped_env is not None
    assert wrapped_env.observation_space is not None
    assert wrapped_env.action_space is not None
    
    # CRITICAL: Call reset() before passing to create_agent to ensure observation_space is properly sized
    wrapped_env.reset(seed=42)
    
    # Create agent using factory
    agent = create_agent(config=config, env=wrapped_env, verbose=0)
    assert isinstance(agent, PPO)
    
    # Train briefly
    agent.learn(total_timesteps=500)
    assert agent.num_timesteps > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
