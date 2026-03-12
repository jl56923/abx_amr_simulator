"""Regression tests for run_evaluation_episodes() in evaluative_plots.py.

Verifies:
- Each episode starts with an independent env.reset() call
- Episodes can have genuinely different lengths (variable-length support)
- Return structure contains all expected keys with correct shapes
- No crashes or silent failures on very short (1-step) episodes
"""

import numpy as np
import pytest

from abx_amr_simulator.analysis.evaluative_plots import run_evaluation_episodes


# ---------------------------------------------------------------------------
# Minimal stubs — real interface, no SB3 or abx_amr_simulator env required
# ---------------------------------------------------------------------------

class _CountingEnv:
    """Deterministic environment stub that tracks reset/step call counts.

    Each episode terminates after `episode_length` steps. The obs returned
    by reset() is a running counter so tests can verify fresh resets.
    """

    def __init__(self, episode_length: int = 3):
        self._episode_length = episode_length
        self._steps_taken = 0
        self.reset_count = 0
        self.obs_at_reset: list = []

    def reset(self, seed=None):
        self.reset_count += 1
        self._steps_taken = 0
        obs = np.array([float(self.reset_count)], dtype=np.float32)
        self.obs_at_reset.append(float(obs[0]))
        return obs, {}

    def step(self, action):
        self._steps_taken += 1
        obs = np.array([float(self.reset_count)], dtype=np.float32)
        reward = 1.0
        terminated = self._steps_taken >= self._episode_length
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class _VariableLengthEnv:
    """Env stub where each successive episode is 1 step longer.

    Episode i (0-indexed) terminates after (i + 1) steps.
    """

    def __init__(self):
        self._episode_index = -1
        self._steps_taken = 0
        self._episode_length = 0
        self.reset_count = 0

    def reset(self, seed=None):
        self.reset_count += 1
        self._episode_index += 1
        self._episode_length = self._episode_index + 1
        self._steps_taken = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self._steps_taken += 1
        terminated = self._steps_taken >= self._episode_length
        return np.zeros(1, dtype=np.float32), 1.0, terminated, False, {}


class _ConstantModel:
    """Model stub that always predicts action=0."""

    def predict(self, obs, deterministic=True):
        return np.array([0]), None


# ---------------------------------------------------------------------------
# Tests: episode independence
# ---------------------------------------------------------------------------

class TestRunEvaluationEpisodesIndependence:
    def test_reset_called_once_per_episode(self):
        """env.reset() must be called exactly num_episodes times."""
        env = _CountingEnv(episode_length=3)
        model = _ConstantModel()

        run_evaluation_episodes(model=model, env=env, num_episodes=5, is_hrl=False)

        assert env.reset_count == 5, (
            f"Expected reset_count=5, got {env.reset_count}. "
            "env.reset() must be called at the start of every episode."
        )

    def test_each_episode_starts_from_fresh_reset(self):
        """obs at the start of each episode should reflect a fresh reset state.

        _CountingEnv increments reset_count on each reset(), so obs[0] at the
        start of episode i should equal i+1 (1-indexed reset counter).
        """
        env = _CountingEnv(episode_length=2)
        model = _ConstantModel()

        run_evaluation_episodes(model=model, env=env, num_episodes=4, is_hrl=False)

        # Each reset should have yielded a distinct obs value
        assert env.obs_at_reset == [1.0, 2.0, 3.0, 4.0], (
            f"Expected fresh reset obs [1,2,3,4], got {env.obs_at_reset}. "
            "Episodes are not starting from an independent reset."
        )


# ---------------------------------------------------------------------------
# Tests: return structure
# ---------------------------------------------------------------------------

class TestRunEvaluationEpisodesReturnStructure:
    def test_returns_dict_with_expected_keys(self):
        env = _CountingEnv(episode_length=2)
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=3, is_hrl=False)

        assert isinstance(result, dict), "run_evaluation_episodes must return a dict"
        for key in ("episode_rewards", "episode_lengths", "episodes"):
            assert key in result, f"Expected key '{key}' missing from result"

    def test_episode_count_matches_num_episodes(self):
        env = _CountingEnv(episode_length=2)
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=7, is_hrl=False)

        assert len(result["episode_rewards"]) == 7
        assert len(result["episode_lengths"]) == 7
        assert len(result["episodes"]) == 7

    def test_episode_rewards_correct_value(self):
        # 3-step episode with reward=1.0 each step → total reward = 3.0
        env = _CountingEnv(episode_length=3)
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=2, is_hrl=False)

        for ep_reward in result["episode_rewards"]:
            assert ep_reward == pytest.approx(3.0), (
                f"Expected episode reward 3.0 (3 steps × 1.0), got {ep_reward}"
            )

    def test_episode_lengths_correct_value(self):
        env = _CountingEnv(episode_length=4)
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=3, is_hrl=False)

        for ep_len in result["episode_lengths"]:
            assert ep_len == 4, f"Expected episode length 4, got {ep_len}"

    def test_episode_trajectory_keys_present(self):
        env = _CountingEnv(episode_length=2)
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=2, is_hrl=False)

        for ep_traj in result["episodes"]:
            for key in ("obs", "actions", "rewards"):
                assert key in ep_traj, f"Trajectory dict missing key '{key}'"


# ---------------------------------------------------------------------------
# Tests: variable-length episodes
# ---------------------------------------------------------------------------

class TestRunEvaluationEpisodesVariableLength:
    def test_episodes_can_have_different_lengths(self):
        """_VariableLengthEnv terminates episode i after (i+1) steps.

        Lengths should be [1, 2, 3, 4] for 4 episodes.
        """
        env = _VariableLengthEnv()
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=4, is_hrl=False)

        expected_lengths = [1, 2, 3, 4]
        assert result["episode_lengths"] == expected_lengths, (
            f"Expected episode lengths {expected_lengths}, got {result['episode_lengths']}. "
            "run_evaluation_episodes must preserve true realized episode lengths."
        )

    def test_trajectory_length_matches_episode_length(self):
        """The number of recorded steps in each trajectory must equal episode_lengths."""
        env = _VariableLengthEnv()
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=4, is_hrl=False)

        for i, (ep_len, ep_traj) in enumerate(zip(result["episode_lengths"], result["episodes"])):
            traj_len = len(ep_traj["rewards"])
            assert traj_len == ep_len, (
                f"Episode {i}: episode_length={ep_len} but trajectory has {traj_len} steps"
            )

    def test_variable_length_rewards_match_lengths(self):
        """Each episode's total reward = number of steps (reward=1 per step)."""
        env = _VariableLengthEnv()
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=4, is_hrl=False)

        for i, (ep_len, ep_reward) in enumerate(zip(result["episode_lengths"], result["episode_rewards"])):
            assert ep_reward == pytest.approx(float(ep_len)), (
                f"Episode {i}: expected reward {ep_len}, got {ep_reward}"
            )


# ---------------------------------------------------------------------------
# Tests: robustness on short episodes
# ---------------------------------------------------------------------------

class TestRunEvaluationEpisodesShortEpisodes:
    def test_single_step_episode_does_not_crash(self):
        """A 1-step episode should complete without errors."""
        env = _CountingEnv(episode_length=1)
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=3, is_hrl=False)

        assert all(length == 1 for length in result["episode_lengths"]), (
            f"Expected all episodes of length 1, got {result['episode_lengths']}"
        )

    def test_single_step_episode_trajectory_has_one_entry(self):
        env = _CountingEnv(episode_length=1)
        model = _ConstantModel()
        result = run_evaluation_episodes(model=model, env=env, num_episodes=2, is_hrl=False)

        for ep_traj in result["episodes"]:
            assert len(ep_traj["rewards"]) == 1, (
                f"Expected 1 reward entry for 1-step episode, got {len(ep_traj['rewards'])}"
            )
            assert len(ep_traj["actions"]) == 1
            assert len(ep_traj["obs"]) == 1
