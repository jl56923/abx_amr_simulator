import numpy as np

from abx_amr_simulator.utils.metrics import run_episode_and_get_trajectory
from tests.unit.utils.test_reference_helpers import create_mock_environment


class _DeterministicModel:
    """Minimal SB3-like model that always returns the same action."""

    def __init__(self, action):
        self._action = action

    def predict(self, obs, deterministic=True):
        del obs, deterministic
        return self._action, None


def test_run_episode_preserves_multidiscrete_array_action():
    """Ensure MultiDiscrete actions stay array-typed for single-patient envs."""
    env = create_mock_environment(num_patients_per_time_step=1, max_time_steps=1)
    action = np.array([env.no_treatment_action], dtype=int)
    model = _DeterministicModel(action=action)

    trajectory = run_episode_and_get_trajectory(model=model, env=env, deterministic=True)

    assert trajectory["actions"], "Expected at least one action in trajectory"
    for action_taken in trajectory["actions"]:
        assert isinstance(action_taken, np.ndarray)
        assert action_taken.shape == (1,)
        assert env.action_space.contains(action_taken)

    env.close()
