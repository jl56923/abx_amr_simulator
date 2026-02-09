"""Tests for HRL config parameter overrides.

Validates that HRL-specific config parameters (hrl.option_gamma, hrl.option_library, etc.)
can be overridden using apply_param_overrides() the same way as standard experiment parameters.
"""

from pathlib import Path
import tempfile

import pytest

from abx_amr_simulator.hrl import setup_options_folders_with_defaults
from abx_amr_simulator.utils import (
    apply_param_overrides,
    create_reward_calculator,
    create_patient_generator,
    create_environment,
    wrap_environment_for_hrl,
)

# Import test helpers (sys.path configured in tests/conftest.py)
from test_reference_helpers import create_mock_environment  # type: ignore[import-not-found]


@pytest.fixture
def temp_options_dir():
    """Create temporary directory with bundled default option configurations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        setup_options_folders_with_defaults(target_path=tmpdir)
        yield tmpdir


class TestHRLConfigParameterOverrides:
    """Test that HRL config parameters support apply_param_overrides()."""

    def test_override_option_gamma_parameter(self):
        """Test overriding hrl.option_gamma via apply_param_overrides()."""
        config = {
            "hrl": {
                "option_library": "default",
                "option_gamma": 0.99,
                "front_edge_use_full_vector": False,
            },
            "ppo": {
                "learning_rate": 3.0e-4,
                "n_steps": 256,
            },
        }

        # Override option_gamma
        overrides = {"hrl.option_gamma": 0.95}
        result = apply_param_overrides(config=config, overrides=overrides)

        assert result["hrl"]["option_gamma"] == 0.95
        # Other HRL and PPO params unchanged
        assert result["hrl"]["option_library"] == "default"
        assert result["hrl"]["front_edge_use_full_vector"] is False
        assert result["ppo"]["learning_rate"] == 3.0e-4

    def test_override_front_edge_use_full_vector_parameter(self):
        """Test overriding hrl.front_edge_use_full_vector via apply_param_overrides()."""
        config = {
            "hrl": {
                "option_library": "default",
                "option_gamma": 0.99,
                "front_edge_use_full_vector": False,
            },
        }

        # Override front_edge_use_full_vector
        overrides = {"hrl.front_edge_use_full_vector": True}
        result = apply_param_overrides(config=config, overrides=overrides)

        assert result["hrl"]["front_edge_use_full_vector"] is True
        assert result["hrl"]["option_gamma"] == 0.99

    def test_override_multiple_hrl_parameters(self):
        """Test overriding multiple HRL parameters at once."""
        config = {
            "hrl": {
                "option_library": "default",
                "option_gamma": 0.99,
                "front_edge_use_full_vector": False,
            },
            "ppo": {
                "learning_rate": 3.0e-4,
                "ent_coef": 0.02,
            },
        }

        # Override multiple HRL and PPO params
        overrides = {
            "hrl.option_gamma": 0.95,
            "hrl.front_edge_use_full_vector": True,
            "ppo.learning_rate": 1.0e-4,
            "ppo.ent_coef": 0.05,
        }

        result = apply_param_overrides(config=config, overrides=overrides)

        assert result["hrl"]["option_gamma"] == 0.95
        assert result["hrl"]["front_edge_use_full_vector"] is True
        assert result["ppo"]["learning_rate"] == 1.0e-4
        assert result["ppo"]["ent_coef"] == 0.05

    def test_override_creates_missing_hrl_section(self):
        """Test that overriding hrl params creates hrl section if missing."""
        config = {
            "algorithm": "HRL_PPO",
            "ppo": {"learning_rate": 3.0e-4},
        }

        # Override creates hrl section
        overrides = {
            "hrl.option_gamma": 0.95,
            "hrl.option_library": "default",
        }

        result = apply_param_overrides(config=config, overrides=overrides)

        assert "hrl" in result
        assert result["hrl"]["option_gamma"] == 0.95
        assert result["hrl"]["option_library"] == "default"
        assert result["ppo"]["learning_rate"] == 3.0e-4

    def test_wrap_environment_respects_overridden_gamma(self, temp_options_dir):
        """Test that wrap_environment_for_hrl() respects overridden option_gamma."""
        # Create base environment
        env = create_mock_environment(
            antibiotic_names=["A", "B"],
            num_patients_per_time_step=1,
            max_time_steps=50,
        )

        # Create config with original gamma
        library_config_path = temp_options_dir / "options" / "option_libraries" / "default_deterministic.yaml"
        config = {
            "hrl": {
                "option_library": str(library_config_path),
                "option_gamma": 0.99,
                "front_edge_use_full_vector": False,
            },
        }

        # Override gamma
        overrides = {"hrl.option_gamma": 0.95}
        config = apply_param_overrides(config=config, overrides=overrides)

        # Wrap and verify the gamma is actually used
        wrapped_env = wrap_environment_for_hrl(env=env, config=config)

        # Check that the wrapper has the overridden gamma value
        assert wrapped_env.gamma == 0.95
        assert config["hrl"]["option_gamma"] == 0.95

    def test_wrap_environment_respects_overridden_front_edge_setting(self, temp_options_dir):
        """Test that wrap_environment_for_hrl() respects front_edge_use_full_vector override."""
        # Create base environment
        env = create_mock_environment(
            antibiotic_names=["A", "B"],
            num_patients_per_time_step=1,
            max_time_steps=50,
        )

        library_config_path = temp_options_dir / "options" / "option_libraries" / "default_deterministic.yaml"
        config = {
            "hrl": {
                "option_library": str(library_config_path),
                "option_gamma": 0.99,
                "front_edge_use_full_vector": False,
            },
        }

        # Override front_edge_use_full_vector
        overrides = {"hrl.front_edge_use_full_vector": True}
        config = apply_param_overrides(config=config, overrides=overrides)

        # Wrap and verify the setting is actually used
        wrapped_env = wrap_environment_for_hrl(env=env, config=config)

        # Check that the wrapper has the overridden setting
        assert wrapped_env.front_edge_use_full_vector is True
        assert config["hrl"]["front_edge_use_full_vector"] is True

    def test_hrl_overrides_alongside_standard_parameters(self, temp_options_dir):
        """Test that HRL overrides work alongside environment/reward_calculator overrides."""
        # Create base components
        library_config_path = (
            temp_options_dir / "options" / "option_libraries" / "default_deterministic.yaml"
        )

        # Full config with all component sections
        config = {
            "algorithm": "HRL_PPO",
            "environment": {
                "num_patients_per_time_step": 1,
                "max_time_steps": 50,
                "reward_calculator": {
                    "abx_clinical_reward_penalties_info_dict": {
                        "clinical_benefit_reward": 10.0,
                        "clinical_benefit_probability": 1.0,
                        "clinical_failure_penalty": -1.0,
                        "clinical_failure_probability": 0.0,
                        "abx_adverse_effects_info": {
                            "adverse_effect_penalty": -0.1,
                            "adverse_effect_probability": 0.01,
                        },
                        "allow_treatment_after_recovery": False,
                    }
                },
                "amr_dynamics": {
                    "leak": 0.9,
                    "flatness_parameter": 1.0,
                    "permanent_residual_volume": 0.0,
                    "initial_amr_level": 0.0,
                },
            },
            "reward_calculator": {
                "lambda_weight": 0.5,
                "epsilon": 0.0,
            },
            "patient_generator": {
                "prob_infected": 0.5,
                "visible_patient_attributes": ["prob_infected"],
            },
            "hrl": {
                "option_library": str(library_config_path),
                "option_gamma": 0.99,
                "front_edge_use_full_vector": False,
            },
            "ppo": {
                "learning_rate": 3.0e-4,
                "n_steps": 256,
                "ent_coef": 0.02,
            },
        }

        # Apply multiple overrides across all sections
        overrides = {
            "environment.num_patients_per_time_step": 2,
            "environment.max_time_steps": 100,
            "reward_calculator.lambda_weight": 0.8,
            "hrl.option_gamma": 0.95,
            "hrl.front_edge_use_full_vector": True,
            "ppo.learning_rate": 1.0e-4,
            "ppo.ent_coef": 0.05,
        }

        result = apply_param_overrides(config=config, overrides=overrides)

        # Verify environment overrides
        assert result["environment"]["num_patients_per_time_step"] == 2
        assert result["environment"]["max_time_steps"] == 100

        # Verify reward_calculator overrides
        assert result["reward_calculator"]["lambda_weight"] == 0.8

        # Verify HRL overrides
        assert result["hrl"]["option_gamma"] == 0.95
        assert result["hrl"]["front_edge_use_full_vector"] is True

        # Verify PPO overrides
        assert result["ppo"]["learning_rate"] == 1.0e-4
        assert result["ppo"]["ent_coef"] == 0.05

        # Verify unchanged values
        assert result["patient_generator"]["prob_infected"] == 0.5
        assert result["hrl"]["option_library"] == str(library_config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
