"""Test the EarlyStoppingCallback."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from abx_amr_simulator.callbacks.early_stopping import EarlyStoppingCallback


class TestEarlyStoppingCallback:
    """Test early stopping callback behavior."""
    
    def test_initialization(self):
        """Test callback initializes with correct defaults."""
        callback = EarlyStoppingCallback(patience=5, min_delta=0.01, verbose=0)
        assert callback.patience == 5
        assert callback.min_delta == 0.01
        assert callback.best_metric is None
        assert callback.evaluations_since_improvement == 0
    
    def test_first_evaluation_sets_best_metric(self):
        """Test that first metric value is stored as best_metric."""
        callback = EarlyStoppingCallback(patience=3, min_delta=0.0, verbose=0)
        callback.model = Mock()
        
        # Simulate logger with metric value
        callback.model.logger = Mock()
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        
        # First step should set best_metric
        result = callback._on_step()
        assert result is True
        assert callback.best_metric == 100.0
        assert callback.evaluations_since_improvement == 0
    
    def test_improvement_resets_counter(self):
        """Test that improvement resets evaluations_since_improvement counter."""
        callback = EarlyStoppingCallback(patience=3, min_delta=0.01, verbose=0)
        callback.model = Mock()
        callback.model.logger = Mock()
        
        # First evaluation
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        callback._on_step()
        
        # No improvement
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.005}
        callback._on_step()
        assert callback.evaluations_since_improvement == 1
        
        # Significant improvement
        callback.model.logger.name_to_value = {'eval/mean_reward': 101.0}
        callback._on_step()
        assert callback.evaluations_since_improvement == 0
        assert callback.best_metric == 101.0
    
    def test_stops_after_patience_exceeded(self):
        """Test that training stops when patience is exceeded."""
        callback = EarlyStoppingCallback(patience=2, min_delta=0.01, verbose=0)
        callback.model = Mock()
        callback.model.logger = Mock()
        
        # Initial good metric
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        callback._on_step()
        
        # First plateau
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.005}
        result = callback._on_step()
        assert result is True
        
        # Second plateau
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.005}
        result = callback._on_step()
        assert result is False  # Should stop
    
    def test_respects_min_delta(self):
        """Test that improvements smaller than min_delta are ignored."""
        callback = EarlyStoppingCallback(patience=2, min_delta=1.0, verbose=0)
        callback.model = Mock()
        callback.model.logger = Mock()
        
        # Initial metric
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        callback._on_step()
        
        # Small improvement (less than min_delta)
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.5}
        callback._on_step()
        assert callback.evaluations_since_improvement == 1
        
        # Large improvement (greater than min_delta)
        callback.model.logger.name_to_value = {'eval/mean_reward': 101.5}
        callback._on_step()
        assert callback.evaluations_since_improvement == 0
    
    def test_handles_missing_logger(self):
        """Test graceful handling when logger is not attached."""
        callback = EarlyStoppingCallback(patience=3, verbose=0)
        callback.model = Mock()
        callback.model.logger = None
        
        # Should continue training if no logger
        result = callback._on_step()
        assert result is True
    
    def test_handles_missing_metric(self):
        """Test graceful handling when metric is not yet available."""
        callback = EarlyStoppingCallback(patience=3, verbose=0)
        callback.model = Mock()
        callback.model.logger = Mock()
        callback.model.logger.name_to_value = {}  # No metric yet
        
        # Should continue training if metric not available
        result = callback._on_step()
        assert result is True
        assert callback.best_metric is None
    
    def test_respects_min_timesteps(self):
        """Test that early stopping doesn't trigger before min_timesteps is reached."""
        callback = EarlyStoppingCallback(
            patience=2,
            min_delta=0.01,
            min_timesteps=1000,  # Require at least 1000 timesteps
            verbose=0
        )
        callback.model = Mock()
        callback.model.logger = Mock()
        
        # Below minimum timesteps - callback should just continue without processing metrics
        callback.num_timesteps = 500
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        result = callback._on_step()
        assert result is True
        assert callback.best_metric is None  # Metrics not processed yet
        
        # Now at min_timesteps - should start processing metrics
        callback.num_timesteps = 1000
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        result = callback._on_step()
        assert result is True
        assert callback.best_metric == 100.0  # First metric recorded
        
        # No improvement - first time
        callback.num_timesteps = 1100
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.005}
        result = callback._on_step()
        assert result is True
        assert callback.evaluations_since_improvement == 1
        
        # No improvement - second time (patience=2, should stop now)
        callback.num_timesteps = 1200
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.003}
        result = callback._on_step()
        assert result is False  # Should stop (patience=2 exceeded)
        assert callback.evaluations_since_improvement == 2
    
    def test_min_timesteps_default_zero(self):
        """Test that min_timesteps defaults to 0 (no minimum)."""
        callback = EarlyStoppingCallback(patience=2, verbose=0)
        assert callback.min_timesteps == 0
        
        callback.model = Mock()
        callback.model.logger = Mock()
        callback.num_timesteps = 10  # Very early in training
        
        # Set metric and trigger early stopping immediately
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        callback._on_step()
        
        # No improvement
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        callback._on_step()
        
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        callback._on_step()
        
        # Should stop after patience=2 with no minimum
        callback.model.logger.name_to_value = {'eval/mean_reward': 100.0}
        result = callback._on_step()
        assert result is False  # Early stopping can trigger immediately


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
