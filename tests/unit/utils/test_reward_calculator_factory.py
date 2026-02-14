"""
Tests for create_reward_calculator() factory function.

Tests reward calculator instantiation with various configurations including
per-antibiotic parameters, lambda_weight, and seeding.
"""

import tempfile
import yaml
from pathlib import Path
import pytest

from abx_amr_simulator.utils import create_reward_calculator, load_config
from abx_amr_simulator.core import RewardCalculator


class TestRewardCalculatorFactory:
    """Tests for create_reward_calculator() factory function."""
    
    def test_creates_reward_calculator_with_config(self):
        """Test basic reward calculator instantiation with umbrella config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create reward calculator config
            rc_config = {
                'abx_clinical_reward_penalties_info_dict': {
                    'clinical_benefit_reward': 10.0,
                    'clinical_benefit_probability': 1.0,
                    'clinical_failure_penalty': -5.0,
                    'clinical_failure_probability': 1.0,
                    'abx_adverse_effects_info': {
                        'A': {
                            'adverse_effect_penalty': -1.0,
                            'adverse_effect_probability': 0.1,
                        }
                    }
                },
                'lambda_weight': 0.5,
                'epsilon': 0.05,
            }
            rc_config_path = tmpdir_path / "reward_calc.yaml"
            with open(rc_config_path, 'w') as f:
                yaml.dump(rc_config, f)
            
            # Create umbrella config
            umbrella_config = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'reward_calc.yaml',
                'training': {'seed': 42}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella_config, f)
            
            # Load and create
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            
            assert isinstance(rc, RewardCalculator)
            assert rc.lambda_weight == 0.5
            assert rc.seed == 42
    
    def test_lambda_weight_controls_amr_tradeoff(self):
        """Test that lambda_weight parameter is correctly set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create reward calculator config with low lambda
            rc_config_low = {
                'abx_clinical_reward_penalties_info_dict': {
                    'clinical_benefit_reward': 10.0,
                    'clinical_benefit_probability': 1.0,
                    'clinical_failure_penalty': -5.0,
                    'clinical_failure_probability': 1.0,
                    'abx_adverse_effects_info': {
                        'A': {
                            'adverse_effect_penalty': -1.0,
                            'adverse_effect_probability': 0.1,
                        }
                    }
                },
                'lambda_weight': 0.1,  # Clinical-focused
                'epsilon': 0.05,
            }
            rc_config_low_path = tmpdir_path / "rc_low.yaml"
            with open(rc_config_low_path, 'w') as f:
                yaml.dump(rc_config_low, f)
            
            # Create reward calculator config with high lambda
            rc_config_high = rc_config_low.copy()
            rc_config_high['lambda_weight'] = 0.9  # AMR-focused
            rc_config_high_path = tmpdir_path / "rc_high.yaml"
            with open(rc_config_high_path, 'w') as f:
                yaml.dump(rc_config_high, f)
            
            # Create umbrella configs
            umbrella_low = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc_low.yaml',
                'training': {'seed': 42}
            }
            umbrella_low_path = tmpdir_path / "umbrella_low.yaml"
            with open(umbrella_low_path, 'w') as f:
                yaml.dump(umbrella_low, f)
            
            umbrella_high = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'rc_high.yaml',
                'training': {'seed': 42}
            }
            umbrella_high_path = tmpdir_path / "umbrella_high.yaml"
            with open(umbrella_high_path, 'w') as f:
                yaml.dump(umbrella_high, f)
            
            # Load and create
            config_low = load_config(config_path=str(umbrella_low_path))
            config_high = load_config(config_path=str(umbrella_high_path))
            
            rc_low = create_reward_calculator(config=config_low)
            rc_high = create_reward_calculator(config=config_high)
            
            assert rc_low.lambda_weight == 0.1  # Clinical-focused
            assert rc_high.lambda_weight == 0.9  # AMR-penalty-focused
    
    def test_seed_applied_for_reproducibility(self):
        """Test that seed is correctly passed to reward calculator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create reward calculator config
            rc_config = {
                'abx_clinical_reward_penalties_info_dict': {
                    'clinical_benefit_reward': 10.0,
                    'clinical_benefit_probability': 1.0,
                    'clinical_failure_penalty': -5.0,
                    'clinical_failure_probability': 1.0,
                    'abx_adverse_effects_info': {
                        'A': {
                            'adverse_effect_penalty': -1.0,
                            'adverse_effect_probability': 0.1,
                        }
                    }
                },
                'lambda_weight': 0.5,
                'epsilon': 0.05,
            }
            rc_config_path = tmpdir_path / "reward_calc.yaml"
            with open(rc_config_path, 'w') as f:
                yaml.dump(rc_config, f)
            
            # Create umbrella config with seed
            umbrella_config = {
                'config_folder_location': str(tmpdir_path),
                'reward_calculator': 'reward_calc.yaml',
                'training': {'seed': 12345}
            }
            umbrella_path = tmpdir_path / "umbrella.yaml"
            with open(umbrella_path, 'w') as f:
                yaml.dump(umbrella_config, f)
            
            # Load and create
            config = load_config(config_path=str(umbrella_path))
            rc = create_reward_calculator(config=config)
            
            # With seed from training config, should be injected
            assert rc.seed == 12345
