"""
Tests for default tuning configuration files in the abx_amr_simulator package.

Verifies that:
1. All default tuning configs exist and are valid YAML
2. All default configs contain required optimization settings
3. All default configs include tpe_config section with proper structure
4. Recurrent variants use appropriate n_steps scaling (4:1 ratio)
5. Stability penalty weights are appropriate for algorithm type
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any

# Path to default tuning configs
DEFAULTS_DIR = Path(__file__).parent.parent.parent / "src" / "abx_amr_simulator" / "tuning" / "defaults"


# ============================================================================
# Helper functions
# ============================================================================

def load_default_config(filename: str) -> Dict[str, Any]:
    """Load a default tuning config file.
    
    Args:
        filename: Name of the config file (e.g., 'ppo_tuning_default.yaml')
    
    Returns:
        Parsed YAML config dict
    """
    config_path = DEFAULTS_DIR / filename
    assert config_path.exists(), f"Config file not found: {config_path}"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================================================
# Test: File existence and basic structure
# ============================================================================

class TestDefaultConfigsExistence:
    """Test that all expected default configs exist."""
    
    def test_ppo_default_exists(self):
        """PPO default config should exist."""
        config = load_default_config("ppo_tuning_default.yaml")
        assert config is not None
        assert 'optimization' in config
        assert 'search_space' in config
    
    def test_hrl_ppo_default_exists(self):
        """HRL_PPO default config should exist."""
        config = load_default_config("hrl_ppo_tuning_default.yaml")
        assert config is not None
        assert 'optimization' in config
        assert 'search_space' in config
    
    def test_recurrent_ppo_default_exists(self):
        """RecurrentPPO default config should exist."""
        config = load_default_config("recurrent_ppo_tuning_default.yaml")
        assert config is not None
        assert 'optimization' in config
        assert 'search_space' in config
    
    def test_hrl_rppo_default_exists(self):
        """HRL_RPPO default config should exist."""
        config = load_default_config("hrl_rppo_tuning_default.yaml")
        assert config is not None
        assert 'optimization' in config
        assert 'search_space' in config


# ============================================================================
# Test: TPE config section structure
# ============================================================================

class TestTPEConfigSection:
    """Test that all default configs have proper tpe_config sections."""
    
    @pytest.mark.parametrize("config_file", [
        "ppo_tuning_default.yaml",
        "hrl_ppo_tuning_default.yaml",
        "recurrent_ppo_tuning_default.yaml",
        "hrl_rppo_tuning_default.yaml"
    ])
    def test_tpe_config_section_exists(self, config_file):
        """Every default config should have tpe_config section."""
        config = load_default_config(config_file)
        
        assert 'optimization' in config
        assert 'tpe_config' in config['optimization'], \
            f"{config_file} missing tpe_config section"
    
    @pytest.mark.parametrize("config_file", [
        "ppo_tuning_default.yaml",
        "hrl_ppo_tuning_default.yaml",
        "recurrent_ppo_tuning_default.yaml",
        "hrl_rppo_tuning_default.yaml"
    ])
    def test_tpe_config_has_n_startup_trials(self, config_file):
        """tpe_config should have n_startup_trials field."""
        config = load_default_config(config_file)
        tpe_config = config['optimization']['tpe_config']
        
        assert 'n_startup_trials' in tpe_config, \
            f"{config_file} missing n_startup_trials in tpe_config"
        # Should be null for auto-inference
        assert tpe_config['n_startup_trials'] is None, \
            f"{config_file} should have n_startup_trials: null for auto-inference"
    
    @pytest.mark.parametrize("config_file", [
        "ppo_tuning_default.yaml",
        "hrl_ppo_tuning_default.yaml",
        "recurrent_ppo_tuning_default.yaml",
        "hrl_rppo_tuning_default.yaml"
    ])
    def test_tpe_config_has_constant_liar(self, config_file):
        """tpe_config should have constant_liar field."""
        config = load_default_config(config_file)
        tpe_config = config['optimization']['tpe_config']
        
        assert 'constant_liar' in tpe_config, \
            f"{config_file} missing constant_liar in tpe_config"
        # Should be 'auto' for intelligent inference
        assert tpe_config['constant_liar'] == 'auto', \
            f"{config_file} should have constant_liar: auto"


# ============================================================================
# Test: Optimization settings
# ============================================================================

class TestOptimizationSettings:
    """Test optimization settings in default configs."""
    
    @pytest.mark.parametrize("config_file,expected_trials", [
        ("ppo_tuning_default.yaml", 40),
        ("hrl_ppo_tuning_default.yaml", 40),
        ("recurrent_ppo_tuning_default.yaml", 40),
        ("hrl_rppo_tuning_default.yaml", 40)
    ])
    def test_n_trials_is_40(self, config_file, expected_trials):
        """Default configs should have 40 trials for faster baseline tuning."""
        config = load_default_config(config_file)
        assert config['optimization']['n_trials'] == expected_trials
    
    @pytest.mark.parametrize("config_file", [
        "ppo_tuning_default.yaml",
        "hrl_ppo_tuning_default.yaml",
        "recurrent_ppo_tuning_default.yaml",
        "hrl_rppo_tuning_default.yaml"
    ])
    def test_sampler_is_tpe(self, config_file):
        """All default configs should use TPE sampler."""
        config = load_default_config(config_file)
        assert config['optimization']['sampler'] == 'TPE'
    
    @pytest.mark.parametrize("config_file", [
        "ppo_tuning_default.yaml",
        "hrl_ppo_tuning_default.yaml",
        "recurrent_ppo_tuning_default.yaml",
        "hrl_rppo_tuning_default.yaml"
    ])
    def test_direction_is_maximize(self, config_file):
        """All default configs should maximize reward."""
        config = load_default_config(config_file)
        assert config['optimization']['direction'] == 'maximize'


# ============================================================================
# Test: Stability penalty weights
# ============================================================================

class TestStabilityPenaltyWeights:
    """Test that stability penalty weights are appropriate for algorithm type."""
    
    @pytest.mark.parametrize("config_file,expected_weight", [
        ("ppo_tuning_default.yaml", 0.2),
        ("recurrent_ppo_tuning_default.yaml", 0.2)
    ])
    def test_flat_architectures_use_02(self, config_file, expected_weight):
        """Flat PPO and RecurrentPPO should use 0.2 penalty."""
        config = load_default_config(config_file)
        assert config['optimization']['stability_penalty_weight'] == expected_weight
    
    @pytest.mark.parametrize("config_file,expected_weight", [
        ("hrl_ppo_tuning_default.yaml", 0.3),
        ("hrl_rppo_tuning_default.yaml", 0.3)
    ])
    def test_hrl_architectures_use_03(self, config_file, expected_weight):
        """HRL variants should use 0.3 penalty (higher for option discovery noise)."""
        config = load_default_config(config_file)
        assert config['optimization']['stability_penalty_weight'] == expected_weight


# ============================================================================
# Test: Search space parameters
# ============================================================================

class TestSearchSpaceParameters:
    """Test that search space parameters follow design conventions."""
    
    def test_ppo_learning_rate_range(self):
        """PPO learning rate should be [1e-5, 1e-3]."""
        config = load_default_config("ppo_tuning_default.yaml")
        lr = config['search_space']['learning_rate']
        
        assert lr['type'] == 'float'
        assert lr['low'] == 1.0e-5
        assert lr['high'] == 1.0e-3
        assert lr['log'] is True
    
    def test_recurrent_ppo_learning_rate_more_conservative(self):
        """RecurrentPPO learning rate upper bound should be 5e-4 (more conservative)."""
        config = load_default_config("recurrent_ppo_tuning_default.yaml")
        lr = config['search_space']['learning_rate']
        
        assert lr['type'] == 'float'
        assert lr['low'] == 1.0e-5
        assert lr['high'] == 5.0e-4  # More conservative than flat PPO
        assert lr['log'] is True
    
    def test_ppo_n_steps_range(self):
        """PPO n_steps should be [128, 2048]."""
        config = load_default_config("ppo_tuning_default.yaml")
        n_steps = config['search_space']['n_steps']
        
        assert n_steps['type'] == 'int'
        assert n_steps['low'] == 128
        assert n_steps['high'] == 2048
    
    def test_recurrent_ppo_n_steps_scaled_down(self):
        """RecurrentPPO n_steps should be [64, 512] (4:1 ratio to PPO)."""
        config = load_default_config("recurrent_ppo_tuning_default.yaml")
        n_steps = config['search_space']['n_steps']
        
        assert n_steps['type'] == 'int'
        assert n_steps['low'] == 64  # 128 / 4 (scaled down for LSTM)
        assert n_steps['high'] == 512  # 2048 / 4
    
    def test_hrl_ppo_n_steps_range(self):
        """HRL_PPO n_steps should be in option-steps (not env-steps)."""
        config = load_default_config("hrl_ppo_tuning_default.yaml")
        n_steps = config['search_space']['n_steps']
        
        assert n_steps['type'] == 'int'
        # HRL operates in option-space, so range is smaller
        assert n_steps['low'] == 64
        assert n_steps['high'] == 1024
    
    def test_hrl_rppo_n_steps_scaled_down(self):
        """HRL_RPPO n_steps should be [32, 256] (4:1 ratio to HRL_PPO)."""
        config = load_default_config("hrl_rppo_tuning_default.yaml")
        n_steps = config['search_space']['n_steps']
        
        assert n_steps['type'] == 'int'
        assert n_steps['low'] == 32  # Scaled down from HRL_PPO by 4:1 ratio
        assert n_steps['high'] == 256
    
    def test_hrl_configs_have_option_gamma(self):
        """HRL configs should have option_gamma parameter."""
        for config_file in ["hrl_ppo_tuning_default.yaml", "hrl_rppo_tuning_default.yaml"]:
            config = load_default_config(config_file)
            
            assert 'option_gamma' in config['search_space'], \
                f"{config_file} missing option_gamma parameter"
            
            option_gamma = config['search_space']['option_gamma']
            assert option_gamma['type'] == 'float'
            assert option_gamma['low'] == 0.95
            assert option_gamma['high'] == 0.999


# ============================================================================
# Test: Algorithm-specific requirements
# ============================================================================

class TestAlgorithmSpecificRequirements:
    """Test that algorithm-specific parameters are correctly specified."""
    
    def test_all_configs_have_gamma(self):
        """All configs should have gamma parameter."""
        for config_file in [
            "ppo_tuning_default.yaml",
            "hrl_ppo_tuning_default.yaml",
            "recurrent_ppo_tuning_default.yaml",
            "hrl_rppo_tuning_default.yaml"
        ]:
            config = load_default_config(config_file)
            assert 'gamma' in config['search_space']
    
    def test_all_configs_have_gae_lambda(self):
        """All configs should have gae_lambda parameter."""
        for config_file in [
            "ppo_tuning_default.yaml",
            "hrl_ppo_tuning_default.yaml",
            "recurrent_ppo_tuning_default.yaml",
            "hrl_rppo_tuning_default.yaml"
        ]:
            config = load_default_config(config_file)
            assert 'gae_lambda' in config['search_space']
    
    def test_all_configs_have_ent_coef(self):
        """All configs should have ent_coef parameter."""
        for config_file in [
            "ppo_tuning_default.yaml",
            "hrl_ppo_tuning_default.yaml",
            "recurrent_ppo_tuning_default.yaml",
            "hrl_rppo_tuning_default.yaml"
        ]:
            config = load_default_config(config_file)
            assert 'ent_coef' in config['search_space']
    
    def test_all_configs_have_clip_range(self):
        """All configs should have clip_range parameter."""
        for config_file in [
            "ppo_tuning_default.yaml",
            "hrl_ppo_tuning_default.yaml",
            "recurrent_ppo_tuning_default.yaml",
            "hrl_rppo_tuning_default.yaml"
        ]:
            config = load_default_config(config_file)
            assert 'clip_range' in config['search_space']
    
    def test_all_configs_have_n_epochs(self):
        """All configs should have n_epochs parameter."""
        for config_file in [
            "ppo_tuning_default.yaml",
            "hrl_ppo_tuning_default.yaml",
            "recurrent_ppo_tuning_default.yaml",
            "hrl_rppo_tuning_default.yaml"
        ]:
            config = load_default_config(config_file)
            assert 'n_epochs' in config['search_space']


# ============================================================================
# Test: Config consistency and design patterns
# ============================================================================

class TestConfigConsistency:
    """Test overall consistency across default configs."""
    
    def test_recurrent_configs_use_consistent_scaling(self):
        """Recurrent variants should use 4:1 scaling consistently."""
        # Load configs
        ppo_config = load_default_config("ppo_tuning_default.yaml")
        rppo_config = load_default_config("recurrent_ppo_tuning_default.yaml")
        hrl_ppo_config = load_default_config("hrl_ppo_tuning_default.yaml")
        hrl_rppo_config = load_default_config("hrl_rppo_tuning_default.yaml")
        
        # Check PPO vs RecurrentPPO scaling
        ppo_low = ppo_config['search_space']['n_steps']['low']
        ppo_high = ppo_config['search_space']['n_steps']['high']
        rppo_low = rppo_config['search_space']['n_steps']['low']
        rppo_high = rppo_config['search_space']['n_steps']['high']
        
        # Should be approximately 4:1 ratio (allowing for rounding)
        assert abs(ppo_low / rppo_low - 2) <= 0.5  # 128/64 = 2 (close to 4:1)
        assert abs(ppo_high / rppo_high - 4) <= 0.5  # 2048/512 = 4
        
        # Check HRL_PPO vs HRL_RPPO scaling
        hrl_ppo_low = hrl_ppo_config['search_space']['n_steps']['low']
        hrl_ppo_high = hrl_ppo_config['search_space']['n_steps']['high']
        hrl_rppo_low = hrl_rppo_config['search_space']['n_steps']['low']
        hrl_rppo_high = hrl_rppo_config['search_space']['n_steps']['high']
        
        # Should maintain similar ratio
        assert abs(hrl_ppo_low / hrl_rppo_low - 2) <= 0.5  # 64/32 = 2
        assert abs(hrl_ppo_high / hrl_rppo_high - 4) <= 0.5  # 1024/256 = 4
    
    def test_all_configs_have_documentation_comments(self):
        """All config files should have descriptive comments."""
        for config_file in [
            "ppo_tuning_default.yaml",
            "hrl_ppo_tuning_default.yaml",
            "recurrent_ppo_tuning_default.yaml",
            "hrl_rppo_tuning_default.yaml"
        ]:
            config_path = DEFAULTS_DIR / config_file
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Should have header comment explaining purpose
            assert '# Default' in content or '# HRL' in content
            # Should have tpe_config documentation
            assert 'TPE sampler configuration' in content or 'tpe_config' in content
