"""
Tests for TPE sampler configuration inference in tune.py.

Verifies that:
1. Three-tier configuration precedence works correctly (CLI > YAML > defaults)
2. Intelligent defaults are inferred correctly from parallelism level
3. num_workers is resolved from CLI > WORKERS env var > default 1
4. constant_liar strategy is enabled/disabled based on parallelism ratio
5. Edge cases (zero trials, very high parallelism) are handled correctly
"""

import pytest
import os
import argparse
from typing import Dict, Any

# infer_tpe_config is defined in tune.py
from abx_amr_simulator.training.tune import infer_tpe_config


# ============================================================================
# Helper function to create mock argparse.Namespace
# ============================================================================

def create_cli_args(
    num_workers=None,
    n_startup_trials=None,
    constant_liar='auto'
) -> argparse.Namespace:
    """Create a mock argparse.Namespace for testing.
    
    Args:
        num_workers: Number of workers (None for env var fallback)
        n_startup_trials: Explicit startup trials (None for auto-infer)
        constant_liar: 'auto', 'on', or 'off'
    
    Returns:
        argparse.Namespace with the specified values
    """
    return argparse.Namespace(
        num_workers=num_workers,
        n_startup_trials=n_startup_trials,
        constant_liar=constant_liar
    )


def create_tuning_config(
    n_trials: int = 48,
    n_startup_trials=None,
    constant_liar='auto'
) -> Dict[str, Any]:
    """Create a mock tuning config dict for testing.
    
    Args:
        n_trials: Total number of trials
        n_startup_trials: Explicit startup trials in YAML (None for omit)
        constant_liar: 'auto', 'on', or 'off'
    
    Returns:
        Dict mimicking loaded tuning config YAML
    """
    config = {
        'optimization': {
            'n_trials': n_trials,
            'sampler': 'TPE'
        }
    }
    
    # Only add tpe_config if explicitly provided
    if n_startup_trials is not None or constant_liar != 'auto':
        config['optimization']['tpe_config'] = {}
        if n_startup_trials is not None:
            config['optimization']['tpe_config']['n_startup_trials'] = n_startup_trials
        if constant_liar != 'auto':
            config['optimization']['tpe_config']['constant_liar'] = constant_liar
    
    return config


# ============================================================================
# Test: num_workers inference (CLI > WORKERS env var > default 1)
# ============================================================================

class TestNumWorkersInference:
    """Test how num_workers is resolved from different sources."""
    
    def test_cli_arg_takes_precedence(self, monkeypatch):
        """CLI arg for num_workers should override env var."""
        monkeypatch.setenv('WORKERS', '8')
        
        cli_args = create_cli_args(num_workers=12)
        tuning_config = create_tuning_config(n_trials=48)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['num_workers'] == 12  # CLI wins
    
    def test_workers_env_var_fallback(self, monkeypatch):
        """If no CLI arg, should fall back to WORKERS env var."""
        monkeypatch.setenv('WORKERS', '16')
        
        cli_args = create_cli_args(num_workers=None)
        tuning_config = create_tuning_config(n_trials=48)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['num_workers'] == 16
    
    def test_default_to_one_worker(self, monkeypatch):
        """If no CLI arg and no env var, default to 1."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(num_workers=None)
        tuning_config = create_tuning_config(n_trials=48)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['num_workers'] == 1
    
    def test_invalid_workers_env_var(self, monkeypatch, capsys):
        """Invalid WORKERS env var should print warning and default to 1."""
        monkeypatch.setenv('WORKERS', 'not_a_number')
        
        cli_args = create_cli_args(num_workers=None)
        tuning_config = create_tuning_config(n_trials=48)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['num_workers'] == 1
        captured = capsys.readouterr()
        assert 'Warning' in captured.out
        assert 'WORKERS' in captured.out


# ============================================================================
# Test: n_startup_trials inference (CLI > YAML > intelligent default)
# ============================================================================

class TestStartupTrialsInference:
    """Test how n_startup_trials is resolved from different sources."""
    
    def test_cli_overrides_yaml(self, monkeypatch):
        """CLI arg should override YAML config value."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=12,
            n_startup_trials=20  # Explicit CLI override
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            n_startup_trials=10  # YAML value should be ignored
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['n_startup_trials'] == 20  # CLI wins
    
    def test_yaml_overrides_default(self, monkeypatch):
        """YAML value should override intelligent default."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=12,
            n_startup_trials=None  # No CLI override
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            n_startup_trials=15  # Explicit YAML value
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['n_startup_trials'] == 15  # YAML wins
    
    def test_intelligent_default_standard_case(self, monkeypatch):
        """Test default inference: min(num_workers, max(5, n_trials // 4))."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=12,
            n_startup_trials=None
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            n_startup_trials=None  # No YAML override
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        # Expected: min(12, max(5, 48 // 4)) = min(12, max(5, 12)) = min(12, 12) = 12
        assert result['n_startup_trials'] == 12
    
    def test_intelligent_default_many_workers(self, monkeypatch):
        """With many workers, should cap at n_trials // 4."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=50,  # More than n_trials // 4
            n_startup_trials=None
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            n_startup_trials=None
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        # Expected: min(50, max(5, 48 // 4)) = min(50, 12) = 12
        assert result['n_startup_trials'] == 12
    
    def test_intelligent_default_few_trials(self, monkeypatch):
        """With very few trials, should ensure at least 5 startup trials."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=10,
            n_startup_trials=None
        )
        tuning_config = create_tuning_config(
            n_trials=12,  # n_trials // 4 = 3, but min is 5
            n_startup_trials=None
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=12
        )
        
        # Expected: min(10, max(5, 12 // 4)) = min(10, max(5, 3)) = min(10, 5) = 5
        assert result['n_startup_trials'] == 5


# ============================================================================
# Test: constant_liar inference (CLI > YAML > parallelism-based)
# ============================================================================

class TestConstantLiarInference:
    """Test how constant_liar is resolved from different sources."""
    
    def test_cli_overrides_yaml_on(self, monkeypatch):
        """CLI 'on' should override YAML 'off'."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=12,
            constant_liar='on'  # CLI says on
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            constant_liar='off'  # YAML says off (should be ignored)
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['constant_liar'] is True  # CLI wins
    
    def test_cli_overrides_yaml_off(self, monkeypatch):
        """CLI 'off' should override YAML 'on'."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=12,
            constant_liar='off'  # CLI says off
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            constant_liar='on'  # YAML says on (should be ignored)
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['constant_liar'] is False  # CLI wins
    
    def test_yaml_overrides_auto(self, monkeypatch):
        """YAML explicit 'on' should override parallelism-based auto."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=2,  # Low parallelism (2/48 = 4%)
            constant_liar='auto'  # CLI is auto (defers to YAML)
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            constant_liar='on'  # YAML forces on
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        # Despite low parallelism, YAML says 'on'
        assert result['constant_liar'] is True
    
    def test_auto_low_parallelism(self, monkeypatch):
        """With low parallelism (<15%), constant_liar should be False."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=4,  # 4/48 = 8.3% parallelism
            constant_liar='auto'
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            constant_liar='auto'
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['constant_liar'] is False  # 8.3% < 15%
    
    def test_auto_moderate_parallelism(self, monkeypatch):
        """With moderate parallelism (>15%), constant_liar should be True."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=12,  # 12/48 = 25% parallelism
            constant_liar='auto'
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            constant_liar='auto'
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['constant_liar'] is True  # 25% > 15%
    
    def test_auto_high_parallelism(self, monkeypatch):
        """With high parallelism (>30%), constant_liar should definitely be True."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=24,  # 24/48 = 50% parallelism
            constant_liar='auto'
        )
        tuning_config = create_tuning_config(
            n_trials=48,
            constant_liar='auto'
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['constant_liar'] is True  # 50% > 15%
    
    def test_auto_exactly_15_percent(self, monkeypatch):
        """At exactly 15% parallelism, should be False (threshold is >15%)."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=15,  # 15/100 = 15% parallelism
            constant_liar='auto'
        )
        tuning_config = create_tuning_config(
            n_trials=100,
            constant_liar='auto'
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=100
        )
        
        assert result['constant_liar'] is False  # 15% is NOT > 15%
    
    def test_auto_just_above_15_percent(self, monkeypatch):
        """Just above 15% parallelism should enable constant_liar."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(
            num_workers=16,  # 16/100 = 16% parallelism
            constant_liar='auto'
        )
        tuning_config = create_tuning_config(
            n_trials=100,
            constant_liar='auto'
        )
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=100
        )
        
        assert result['constant_liar'] is True  # 16% > 15%


# ============================================================================
# Test: Edge cases and realistic scenarios
# ============================================================================

class TestEdgeCasesAndRealisticScenarios:
    """Test edge cases and realistic HPC scenarios."""
    
    def test_single_worker_no_parallelism(self, monkeypatch):
        """With 1 worker, constant_liar should be False."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(num_workers=1)
        tuning_config = create_tuning_config(n_trials=48)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['num_workers'] == 1
        assert result['constant_liar'] is False  # 1/48 < 15%
        # n_startup_trials = min(1, max(5, 12)) = min(1, 12) = 1
        assert result['n_startup_trials'] == 1
    
    def test_exp_2f_scenario_12_workers(self, monkeypatch):
        """Realistic scenario: exp_2f with 12 workers, 48 trials."""
        monkeypatch.setenv('WORKERS', '12')
        
        cli_args = create_cli_args(num_workers=None)  # Should read from env
        tuning_config = create_tuning_config(n_trials=48)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['num_workers'] == 12
        assert result['n_startup_trials'] == 12  # min(12, max(5, 12)) = 12
        assert result['constant_liar'] is True  # 12/48 = 25% > 15%
    
    def test_exp_2f_scenario_24_workers(self, monkeypatch):
        """Realistic scenario: exp_2f with 24 workers, 48 trials."""
        monkeypatch.setenv('WORKERS', '24')
        
        cli_args = create_cli_args(num_workers=None)
        tuning_config = create_tuning_config(n_trials=48)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['num_workers'] == 24
        assert result['n_startup_trials'] == 12  # min(24, max(5, 12)) = 12
        assert result['constant_liar'] is True  # 24/48 = 50% > 15%
    
    def test_debug_config_4_trials(self, monkeypatch):
        """Debug scenario: hrl_rppo_tuning_debug.yaml with 4 trials."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(num_workers=2)
        tuning_config = create_tuning_config(n_trials=4)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=4
        )
        
        assert result['num_workers'] == 2
        # n_startup_trials = min(2, max(5, 4 // 4)) = min(2, max(5, 1)) = min(2, 5) = 2
        assert result['n_startup_trials'] == 2
        assert result['constant_liar'] is True  # 2/4 = 50% > 15%
    
    def test_very_large_study(self, monkeypatch):
        """Large-scale scenario: 200 trials with 40 workers."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(num_workers=40)
        tuning_config = create_tuning_config(n_trials=200)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=200
        )
        
        assert result['num_workers'] == 40
        # n_startup_trials = min(40, max(5, 200 // 4)) = min(40, 50) = 40
        assert result['n_startup_trials'] == 40
        assert result['constant_liar'] is True  # 40/200 = 20% > 15%
    
    def test_zero_trials_edge_case(self, monkeypatch):
        """Edge case: n_trials = 0 (should not crash)."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(num_workers=12)
        tuning_config = create_tuning_config(n_trials=0)
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=0
        )
        
        assert result['num_workers'] == 12
        # n_startup_trials = min(12, max(5, 0 // 4)) = min(12, max(5, 0)) = min(12, 5) = 5
        assert result['n_startup_trials'] == 5
        # Parallelism ratio = 12/0 would be inf, but we check n_trials > 0
        assert result['constant_liar'] is False  # Handles division safely


# ============================================================================
# Test: Integration with real YAML config structure
# ============================================================================

class TestRealConfigStructures:
    """Test with realistic config structures from actual tuning configs."""
    
    def test_config_with_no_tpe_section(self, monkeypatch):
        """Legacy config without tpe_config section should work."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(num_workers=12)
        tuning_config = {
            'optimization': {
                'n_trials': 48,
                'sampler': 'TPE',
                'stability_penalty_weight': 0.2
                # No tpe_config section
            }
        }
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        # Should use intelligent defaults
        assert result['num_workers'] == 12
        assert result['n_startup_trials'] == 12
        assert result['constant_liar'] is True
    
    def test_config_with_empty_tpe_section(self, monkeypatch):
        """Config with empty tpe_config section should use defaults."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(num_workers=8)
        tuning_config = {
            'optimization': {
                'n_trials': 40,
                'sampler': 'TPE',
                'tpe_config': {}  # Empty section
            }
        }
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=40
        )
        
        # Should use intelligent defaults
        assert result['num_workers'] == 8
        assert result['n_startup_trials'] == 8  # min(8, max(5, 10)) = 8
        assert result['constant_liar'] is True  # 8/40 = 20% > 15%
    
    def test_config_with_null_values(self, monkeypatch):
        """Config with null/None values should trigger auto-inference."""
        monkeypatch.delenv('WORKERS', raising=False)
        
        cli_args = create_cli_args(num_workers=12)
        tuning_config = {
            'optimization': {
                'n_trials': 48,
                'sampler': 'TPE',
                'tpe_config': {
                    'n_startup_trials': None,  # Explicitly null
                    'constant_liar': 'auto'
                }
            }
        }
        
        result = infer_tpe_config(
            tuning_config=tuning_config,
            cli_args=cli_args,
            n_trials=48
        )
        
        assert result['n_startup_trials'] == 12  # Auto-inferred
        assert result['constant_liar'] is True  # Auto-inferred from 25% parallelism
