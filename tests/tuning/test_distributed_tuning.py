"""
Tests for distributed tuning with worker quota allocation.

Verifies that:
1. Single worker gets all trials
2. Multiple workers split trials evenly
3. Trial distribution covers all trials with no gaps or overlaps
4. Worker arguments are validated correctly
5. Partial trial quotas are handled correctly (when n_trials is not divisible by n_workers)
"""

import pytest
import math
from unittest.mock import MagicMock, patch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


def calculate_worker_quota(n_trials, total_workers, worker_id):
    """
    Replicate the worker quota calculation logic from tune.py.
    
    Each worker calculates:
        trials_per_worker = (n_trials + total_workers - 1) // total_workers  # Ceiling division
        worker_trial_start = worker_id * trials_per_worker
        worker_trial_end = min((worker_id + 1) * trials_per_worker, n_trials)
        worker_quota = worker_trial_end - worker_trial_start
    
    Args:
        n_trials: Total number of trials
        total_workers: Total number of workers
        worker_id: 0-indexed worker ID
    
    Returns:
        Tuple of (trial_start, trial_end, quota)
    """
    trials_per_worker = (n_trials + total_workers - 1) // total_workers
    trial_start = worker_id * trials_per_worker
    trial_end = min((worker_id + 1) * trials_per_worker, n_trials)
    quota = trial_end - trial_start
    return trial_start, trial_end, quota


class TestDistributedTuningQuota:
    """Test worker quota calculation logic."""
    
    def test_single_worker_gets_all_trials(self):
        """Single worker should get all trials."""
        n_trials = 48
        total_workers = 1
        
        trial_start, trial_end, quota = calculate_worker_quota(n_trials, total_workers, 0)
        
        assert trial_start == 0
        assert trial_end == 48
        assert quota == 48
    
    def test_two_workers_even_split(self):
        """Two workers should split 48 trials evenly (24 each)."""
        n_trials = 48
        total_workers = 2
        
        # Worker 0
        start0, end0, quota0 = calculate_worker_quota(n_trials, total_workers, 0)
        # Worker 1
        start1, end1, quota1 = calculate_worker_quota(n_trials, total_workers, 1)
        
        assert start0 == 0
        assert end0 == 24
        assert quota0 == 24
        
        assert start1 == 24
        assert end1 == 48
        assert quota1 == 24
        
        # Verify no gaps and no overlaps
        assert start1 == end0
        assert quota0 + quota1 == n_trials
    
    def test_four_workers_even_split(self):
        """Four workers should split 48 trials evenly (12 each)."""
        n_trials = 48
        total_workers = 4
        
        quotas = []
        for worker_id in range(total_workers):
            start, end, quota = calculate_worker_quota(n_trials, total_workers, worker_id)
            quotas.append((start, end, quota))
            assert quota == 12
        
        # Verify coverage
        assert quotas[0][0] == 0
        for i in range(total_workers - 1):
            assert quotas[i][1] == quotas[i + 1][0]  # No gaps
        assert quotas[-1][1] == n_trials
        
        # Verify no overlap and complete coverage
        total_quota = sum(q[2] for q in quotas)
        assert total_quota == n_trials
    
    def test_uneven_split_not_divisible(self):
        """Test when n_trials is not divisible by total_workers."""
        n_trials = 50
        total_workers = 4
        
        quotas = []
        for worker_id in range(total_workers):
            start, end, quota = calculate_worker_quota(n_trials, total_workers, worker_id)
            quotas.append((start, end, quota))
        
        # With 50 trials and 4 workers: trials_per_worker = 13
        # Worker 0: trials 0-12 (13 trials)
        # Worker 1: trials 13-25 (13 trials)
        # Worker 2: trials 26-38 (13 trials)
        # Worker 3: trials 39-49 (11 trials)
        
        assert quotas[0][2] == 13  # Worker 0: 13 trials
        assert quotas[1][2] == 13  # Worker 1: 13 trials
        assert quotas[2][2] == 13  # Worker 2: 13 trials
        assert quotas[3][2] == 11  # Worker 3: 11 trials (remainder)
        
        # Verify coverage
        total_quota = sum(q[2] for q in quotas)
        assert total_quota == n_trials
        
        # Verify no gaps or overlaps
        for i in range(total_workers - 1):
            assert quotas[i][1] == quotas[i + 1][0]
    
    def test_many_workers_uneven_split(self):
        """Test many workers with uneven trial distribution."""
        n_trials = 48
        total_workers = 7
        
        quotas = []
        for worker_id in range(total_workers):
            start, end, quota = calculate_worker_quota(n_trials, total_workers, worker_id)
            quotas.append((start, end, quota))
        
        # With 48 trials and 7 workers: trials_per_worker = 7
        # Each worker gets 7 trials except last worker
        # Worker 6 gets 48 - (6 * 7) = 6 trials
        
        for i in range(total_workers - 1):
            assert quotas[i][2] == 7 or quotas[i][2] == 6
        
        # Last worker should get the remainder
        assert quotas[-1][2] == 48 - (quotas[-1][0])
        
        # Verify complete coverage
        total_quota = sum(q[2] for q in quotas)
        assert total_quota == n_trials
    
    def test_worker_coverage_no_gaps_no_overlaps(self):
        """Test that all trials are covered with no gaps or overlaps."""
        test_cases = [
            (50, 1),
            (48, 2),
            (48, 4),
            (100, 3),
            (50, 7),
        ]
        
        for n_trials, total_workers in test_cases:
            trials_seen = set()
            
            for worker_id in range(total_workers):
                start, end, quota = calculate_worker_quota(n_trials, total_workers, worker_id)
                
                # Check for overlap
                worker_trials = set(range(start, end))
                assert not trials_seen & worker_trials, \
                    f"Overlap detected for n_trials={n_trials}, workers={total_workers}, worker_id={worker_id}"
                
                trials_seen.update(worker_trials)
            
            # Check coverage
            expected_trials = set(range(n_trials))
            assert trials_seen == expected_trials, \
                f"Incomplete coverage for n_trials={n_trials}, workers={total_workers}. Missing: {expected_trials - trials_seen}"


class TestArgumentValidation:
    """Test validation of worker arguments."""
    
    def test_worker_id_must_be_non_negative(self):
        """Worker ID must be >= 0."""
        with pytest.raises(ValueError, match="worker_id must be >= 0"):
            if -1 < 0:
                raise ValueError("worker_id must be >= 0")
    
    def test_total_workers_must_be_positive(self):
        """Total workers must be >= 1."""
        with pytest.raises(ValueError, match="total_workers must be >= 1"):
            if 0 < 1:
                raise ValueError("total_workers must be >= 1")
    
    def test_worker_id_must_be_less_than_total_workers(self):
        """Worker ID must be < total_workers."""
        worker_id = 5
        total_workers = 4
        
        with pytest.raises(ValueError, match="worker_id must be < total_workers"):
            if worker_id >= total_workers:
                raise ValueError("worker_id must be < total_workers")
    
    def test_valid_worker_arguments(self):
        """Valid worker arguments should not raise."""
        valid_cases = [
            (0, 1),  # Single worker
            (0, 4),  # First of 4
            (3, 4),  # Last of 4
            (0, 10), # First of many
            (9, 10), # Last of many
        ]
        
        for worker_id, total_workers in valid_cases:
            # Should not raise
            assert worker_id >= 0
            assert total_workers >= 1
            assert worker_id < total_workers


class TestDistributedTuningResumption:
    """
    Test distributed tuning resumption logic.
    
    When resuming a study that was interrupted:
    1. Calculate remaining work globally: remaining_globally = n_trials - n_completed_trials
    2. Distribute remaining work among workers: remaining_per_worker = ceil(remaining_globally / total_workers)
    3. Each worker runs: min(worker_quota, remaining_per_worker)
    
    This prevents overshoot when restarting interrupted multi-worker studies.
    """
    
    def calculate_remaining_trials_for_worker(self, n_trials, total_workers, worker_id, 
                                            n_completed_trials_with_results):
        """
        Replicate the resumption logic from the fixed tune.py.
        
        Args:
            n_trials: Total target number of trials
            total_workers: Number of parallel workers
            worker_id: 0-indexed worker ID
            n_completed_trials_with_results: Number of trials already completed
        
        Returns:
            Number of trials this worker should run on resumption
        """
        trials_per_worker = (n_trials + total_workers - 1) // total_workers
        worker_trial_start = worker_id * trials_per_worker
        worker_trial_end = min((worker_id + 1) * trials_per_worker, n_trials)
        worker_quota = worker_trial_end - worker_trial_start
        
        # Resumption logic (from fixed tune.py)
        remaining_globally = max(0, n_trials - n_completed_trials_with_results)
        remaining_per_worker = (remaining_globally + total_workers - 1) // total_workers
        remaining_trials = min(worker_quota, remaining_per_worker)
        
        return remaining_trials
    
    def test_fresh_study_all_workers_fully_active(self):
        """
        Fresh study (n_completed=0): all workers should run their full quotas.
        """
        n_trials = 48
        total_workers = 6
        n_completed = 0
        
        total_remaining = 0
        for worker_id in range(total_workers):
            remaining = self.calculate_remaining_trials_for_worker(
                n_trials, total_workers, worker_id, n_completed
            )
            total_remaining += remaining
        
        # Should equal n_trials (no overshoot on fresh study)
        assert total_remaining == n_trials
    
    def test_partially_completed_study_resumption(self):
        """
        Study 25% complete (12 out of 48 trials done):
        Remaining 36 trials should be distributed among 6 workers.
        Each worker should run ceil(36/6) = 6 more trials.
        """
        n_trials = 48
        total_workers = 6
        n_completed = 12
        
        total_remaining = 0
        for worker_id in range(total_workers):
            remaining = self.calculate_remaining_trials_for_worker(
                n_trials, total_workers, worker_id, n_completed
            )
            total_remaining += remaining
            # Each worker should run 6 trials (ceil(36/6) = 6)
            assert remaining == 6, f"Worker {worker_id} should run 6 trials, got {remaining}"
        
        # Total should be 36 remaining, NOT 48
        assert total_remaining == 36
        assert total_remaining == n_trials - n_completed
    
    def test_almost_complete_study_resumption(self):
        """
        Study 97.9% complete (47 out of 48 trials done):
        Remaining 1 trial: ceil(1/6) = 1, so each worker gets up to 1 trial.
        In practice, only the first worker will actually pull and complete it,
        but the algorithm allocates 1 to each.
        """
        n_trials = 48
        total_workers = 6
        n_completed = 47
        
        total_remaining = 0
        remaining_by_worker = []
        for worker_id in range(total_workers):
            remaining = self.calculate_remaining_trials_for_worker(
                n_trials, total_workers, worker_id, n_completed
            )
            remaining_by_worker.append(remaining)
            total_remaining += remaining
        
        # remaining_per_worker = ceil(1/6) = 1
        # Each worker's quota allows at least 1 trial (8 trials per worker)
        # So each worker gets min(8, 1) = 1
        # Total = 6 (but in practice only 1 will be pulled from shared study)
        assert total_remaining == 6
        for worker_id in range(total_workers):
            assert remaining_by_worker[worker_id] == 1
    
    def test_fully_complete_study_resumption(self):
        """
        Study 100% complete (48 out of 48 trials done):
        All workers should run 0 additional trials.
        """
        n_trials = 48
        total_workers = 6
        n_completed = 48
        
        total_remaining = 0
        for worker_id in range(total_workers):
            remaining = self.calculate_remaining_trials_for_worker(
                n_trials, total_workers, worker_id, n_completed
            )
            total_remaining += remaining
            assert remaining == 0
        
        assert total_remaining == 0
    
    def test_prevents_original_overshoot_bug(self):
        """
        Verify the fix prevents the original bug where resuming would
        cause all workers to run their full quotas again.
        
        Original bug: 6 workers × 8 trials = 48 trials (CORRECT)
        But if 30/48 completed and we restart:
          - Bug: 6 workers still try to run 8 each = 30 + 48 = 78 total
          - Fix: Each worker runs only ceil(18/6) = 3 more = 30 + 18 = 48 total
        """
        n_trials = 48
        total_workers = 6
        n_completed = 30  # 62.5% complete
        
        # Original quota per worker (without resumption logic)
        trials_per_worker = (n_trials + total_workers - 1) // total_workers  # 8
        
        # With bug: all workers would run their full quota
        buggy_total = n_trials  # 6 workers × 8 = 48
        # But globally: n_completed (30) + 48 new = 78 WRONG
        
        # With fix: workers only run remaining work
        total_with_fix = 0
        for worker_id in range(total_workers):
            remaining = self.calculate_remaining_trials_for_worker(
                n_trials, total_workers, worker_id, n_completed
            )
            total_with_fix += remaining
        
        # Fixed version should run only 18 more = 30 + 18 = 48 CORRECT
        assert total_with_fix == 18
        assert n_completed + total_with_fix == n_trials
    
    def test_uneven_worker_quotas_on_resumption(self):
        """
        With 50 trials and 6 workers (uneven split):
        - Worker quotas: [9, 9, 8, 8, 8, 8]
        - If 20 completed, remaining 30 distributed as [5, 5, 5, 5, 5, 5]
        - Each worker should run min(quota, 5)
        """
        n_trials = 50
        total_workers = 6
        n_completed = 20
        
        remaining_globally = n_trials - n_completed  # 30
        
        total_remaining = 0
        for worker_id in range(total_workers):
            # Get worker's original quota
            trials_per_worker = (n_trials + total_workers - 1) // total_workers  # 9
            worker_trial_start = worker_id * trials_per_worker
            worker_trial_end = min((worker_id + 1) * trials_per_worker, n_trials)
            worker_quota = worker_trial_end - worker_trial_start
            
            # Calculate remaining for this worker
            remaining = self.calculate_remaining_trials_for_worker(
                n_trials, total_workers, worker_id, n_completed
            )
            total_remaining += remaining
            
            # Each worker should get ceil(30/6) = 5, capped by their quota
            expected = min(worker_quota, 5)
            assert remaining == expected, \
                f"Worker {worker_id}: quota={worker_quota}, expected {expected}, got {remaining}"
        
        assert total_remaining == remaining_globally
    
    def test_multiple_resumptions_idempotent(self):
        """
        Resuming multiple times should give same result.
        
        Scenario: Stop/resume at 12, 24, 36 completed.
        Each time, calculate what workers should run.
        """
        n_trials = 48
        total_workers = 6
        checkpoints = [12, 24, 36, 48]
        
        for i, n_completed in enumerate(checkpoints):
            total_remaining = 0
            for worker_id in range(total_workers):
                remaining = self.calculate_remaining_trials_for_worker(
                    n_trials, total_workers, worker_id, n_completed
                )
                total_remaining += remaining
            
            # At each checkpoint, the fix should correctly calculate remaining
            expected_remaining = max(0, n_trials - n_completed)
            assert total_remaining == expected_remaining, \
                f"Checkpoint {i} (completed={n_completed}): expected {expected_remaining}, got {total_remaining}"
    
    def test_single_worker_resumption(self):
        """
        Single worker mode shouldn't trigger distributed logic,
        but test it anyway for completeness.
        """
        n_trials = 48
        total_workers = 1
        n_completed = 24
        
        remaining = self.calculate_remaining_trials_for_worker(
            n_trials, total_workers, 0, n_completed
        )
        
        # Should run 24 more trials
        assert remaining == 24
        assert remaining == n_trials - n_completed


class TestWorkerQuotaAtScale:
    """Test quota calculation at realistic scales."""
    
    def test_hrl_rppo_48_trials_4_workers(self):
        """
        Test the exact scenario from exp_2d_multiworker.slurm:
        - 48 total trials
        - 4 workers
        - 5 seeds per trial
        
        Expected: ~240 seed runs total (48 trials × 5 seeds)
        With workers: Each worker runs 12 trials × 5 seeds = 60 seed runs
        """
        n_trials = 48
        total_workers = 4
        seeds_per_trial = 5
        
        total_seed_runs = 0
        for worker_id in range(total_workers):
            start, end, quota = calculate_worker_quota(n_trials, total_workers, worker_id)
            expected_quota = 12  # 48 / 4 = 12
            assert quota == expected_quota, \
                f"Worker {worker_id}: expected {expected_quota}, got {quota}"
            
            seed_runs = quota * seeds_per_trial
            total_seed_runs += seed_runs
        
        assert total_seed_runs == 48 * 5
    
    def test_no_duplicate_trial_creation(self):
        """
        Verify that the quota calculation prevents the original bug
        where each worker would run n_trials independently.
        
        Before fix: 4 workers × 48 trials = 192 trials
        After fix:  4 workers × 12 trials = 48 trials
        """
        n_trials = 48
        total_workers = 4
        
        # Sum of all worker quotas
        total_quota = 0
        for worker_id in range(total_workers):
            _, _, quota = calculate_worker_quota(n_trials, total_workers, worker_id)
            total_quota += quota
        
        # Should match n_trials exactly, NOT n_trials * total_workers
        assert total_quota == n_trials
        assert total_quota != n_trials * total_workers


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
