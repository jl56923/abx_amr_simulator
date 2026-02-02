"""Unit tests for convergence plotting fallback mechanism in diagnostic_analysis.py

Tests that the fallback from TensorBoard logs to eval_logs .npz files works correctly.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from abx_amr_simulator.analysis.diagnostic_analysis import (
    extract_eval_metrics,
    extract_eval_metrics_from_eval_logs,
)


def test_extract_eval_metrics_from_eval_logs():
    """Test that extract_eval_metrics_from_eval_logs correctly reads .npz files."""
    # Create temporary directory structure
    with tempfile.TemporaryDirectory(prefix="test_diagnostic_fallback_") as tmpdir:
        tmpdir = Path(tmpdir)
        run_dir = tmpdir / "test_run"
        eval_logs_dir = run_dir / "eval_logs"
        eval_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock .npz files with expected structure
        test_data = [
            (0, 1000, [1.5, 2.0, 1.8]),  # (eval_count, timestep, episode_rewards)
            (1, 2000, [2.1, 2.3, 2.0]),
            (2, 3000, [2.8, 3.0, 2.9]),
        ]
        
        for eval_count, timestep, rewards in test_data:
            filename = eval_logs_dir / f"eval_{eval_count:04d}_step_{timestep}.npz"
            np.savez_compressed(
                filename,
                episode_rewards=np.array(rewards).reshape(-1, 1),
                episode_lengths=np.array([100] * len(rewards)),
                num_episodes=len(rewards),
                timestep=timestep,
                antibiotic_names=np.array(["A", "B"]),
            )
        
        # Extract metrics
        metrics = extract_eval_metrics_from_eval_logs(run_dir, eval_logs_subdir="eval_logs")
        
        # Verify correct extraction
        assert len(metrics) == 3, "Should extract 3 datapoints"
        
        # Check values (should be mean of episode rewards for each timestep)
        assert metrics[0] == (1000, pytest.approx(np.mean([1.5, 2.0, 1.8]), abs=0.01))
        assert metrics[1] == (2000, pytest.approx(np.mean([2.1, 2.3, 2.0]), abs=0.01))
        assert metrics[2] == (3000, pytest.approx(np.mean([2.8, 3.0, 2.9]), abs=0.01))
        
        # Check sorted by timestep
        timesteps = [t for t, _ in metrics]
        assert timesteps == sorted(timesteps)


def test_extract_eval_metrics_fallback_on_missing_logs():
    """Test that extract_eval_metrics returns empty list when logs directory missing."""
    with tempfile.TemporaryDirectory(prefix="test_diagnostic_fallback_") as tmpdir:
        tmpdir = Path(tmpdir)
        run_dir = tmpdir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # No logs directory created - should return empty list gracefully
        metrics = extract_eval_metrics(run_dir, metric_name="eval/mean_reward")
        assert metrics == [], "Should return empty list when logs directory missing"


def test_extract_eval_metrics_from_eval_logs_missing_keys():
    """Test that function handles .npz files with missing keys gracefully."""
    with tempfile.TemporaryDirectory(prefix="test_diagnostic_fallback_") as tmpdir:
        tmpdir = Path(tmpdir)
        run_dir = tmpdir / "test_run"
        eval_logs_dir = run_dir / "eval_logs"
        eval_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create .npz with missing required keys
        filename = eval_logs_dir / "eval_0000_step_1000.npz"
        np.savez_compressed(
            filename,
            # Missing episode_rewards and timestep
            num_episodes=3,
        )
        
        # Should handle gracefully and return empty list
        metrics = extract_eval_metrics_from_eval_logs(run_dir, eval_logs_subdir="eval_logs")
        assert metrics == [], "Should handle missing keys gracefully"


def test_extract_eval_metrics_from_eval_logs_empty_directory():
    """Test that function handles empty eval_logs directory gracefully."""
    with tempfile.TemporaryDirectory(prefix="test_diagnostic_fallback_") as tmpdir:
        tmpdir = Path(tmpdir)
        run_dir = tmpdir / "test_run"
        eval_logs_dir = run_dir / "eval_logs"
        eval_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Empty directory - should return empty list
        metrics = extract_eval_metrics_from_eval_logs(run_dir, eval_logs_subdir="eval_logs")
        assert metrics == [], "Should return empty list for empty directory"


def test_extract_eval_metrics_from_eval_logs_no_directory():
    """Test that function handles missing eval_logs directory gracefully."""
    with tempfile.TemporaryDirectory(prefix="test_diagnostic_fallback_") as tmpdir:
        tmpdir = Path(tmpdir)
        run_dir = tmpdir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # No eval_logs directory - should return empty list
        metrics = extract_eval_metrics_from_eval_logs(run_dir, eval_logs_subdir="eval_logs")
        assert metrics == [], "Should return empty list when eval_logs directory missing"


def test_extract_eval_metrics_from_eval_logs_sorting():
    """Test that metrics are returned sorted by timestep even if files are out of order."""
    with tempfile.TemporaryDirectory(prefix="test_diagnostic_fallback_") as tmpdir:
        tmpdir = Path(tmpdir)
        run_dir = tmpdir / "test_run"
        eval_logs_dir = run_dir / "eval_logs"
        eval_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create files with timesteps out of order
        test_data = [
            (2, 3000, [3.0]),
            (0, 1000, [1.0]),
            (1, 2000, [2.0]),
        ]
        
        for eval_count, timestep, rewards in test_data:
            filename = eval_logs_dir / f"eval_{eval_count:04d}_step_{timestep}.npz"
            np.savez_compressed(
                filename,
                episode_rewards=np.array(rewards).reshape(-1, 1),
                timestep=timestep,
            )
        
        metrics = extract_eval_metrics_from_eval_logs(run_dir, eval_logs_subdir="eval_logs")
        
        # Should be sorted by timestep
        timesteps = [t for t, _ in metrics]
        assert timesteps == [1000, 2000, 3000], "Should be sorted by timestep"


def test_extract_eval_metrics_from_eval_logs_skips_macos_hidden_files():
    """Test that macOS hidden files (._*) are filtered out and don't cause errors."""
    with tempfile.TemporaryDirectory(prefix="test_diagnostic_fallback_") as tmpdir:
        tmpdir = Path(tmpdir)
        run_dir = tmpdir / "test_run"
        eval_logs_dir = run_dir / "eval_logs"
        eval_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create valid .npz file
        valid_file = eval_logs_dir / "eval_0000_step_1000.npz"
        np.savez_compressed(
            valid_file,
            episode_rewards=np.array([1.5]).reshape(-1, 1),
            timestep=1000,
        )
        
        # Create macOS hidden file (resource fork)
        hidden_file = eval_logs_dir / "._eval_0001_step_2000.npz"
        hidden_file.write_bytes(b'\x00\x00\x00\x00')  # Corrupted data
        
        # Create another hidden file
        dot_hidden = eval_logs_dir / ".eval_0002_step_3000.npz"
        dot_hidden.write_bytes(b'\x00\x00\x00\x00')
        
        # Should only extract from valid file, ignoring hidden files
        metrics = extract_eval_metrics_from_eval_logs(run_dir, eval_logs_subdir="eval_logs")
        
        assert len(metrics) == 1, "Should extract only from valid file"
        assert metrics[0] == (1000, pytest.approx(1.5, abs=0.01)), "Should have correct value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
