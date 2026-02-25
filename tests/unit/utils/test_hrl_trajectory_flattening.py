"""Tests for HRL trajectory flattening in metric collection."""

import numpy as np
from abx_amr_simulator.utils.metrics import _flatten_hrl_trajectory


def test_flatten_hrl_trajectory_simple_two_steps():
    """Test flattening with simple 2-option trajectory (5 + 3 primitive steps)."""
    # Create a simple HRL trajectory at manager level
    trajectory = {
        "obs": [
            np.array([1.0]),      # Initial obs
            np.array([2.0]),      # After option 1 (5 primitive steps)
            np.array([3.0]),      # After option 2 (3 primitive steps)
        ],
        "actions": [
            "option_1",
            "option_2",
        ],
        "rewards": [10.0, 6.0],  # Rewards at manager level
        "infos": [
            {},  # Initial info
            {
                "macro_action": 0,
                "primitive_actions": [0, 1, 0, 1, 0],  # 5 primitive actions
                "primitive_infos": [
                    {"step": 0},
                    {"step": 1},
                    {"step": 2},
                    {"step": 3},
                    {"step": 4},
                ],
            },
            {
                "macro_action": 1,
                "primitive_actions": [1, 1, 0],  # 3 primitive actions
                "primitive_infos": [
                    {"step": 5},
                    {"step": 6},
                    {"step": 7},
                ],
            },
        ],
    }

    flattened = _flatten_hrl_trajectory(trajectory=trajectory)

    # Should have 5 + 3 = 8 primitive steps
    assert len(flattened["actions"]) == 8
    assert len(flattened["rewards"]) == 8
    assert len(flattened["infos"]) == 8

    # First 5 actions should match option 1's primitive actions
    assert flattened["actions"][:5] == [0, 1, 0, 1, 0]

    # Next 3 actions should match option 2's primitive actions
    assert flattened["actions"][5:] == [1, 1, 0]

    # Rewards should be distributed: 10.0 / 5 = 2.0 per step for option 1
    expected_rewards = [2.0] * 5 + [2.0] * 3  # 6.0 / 3 = 2.0 per step for option 2
    np.testing.assert_array_almost_equal(flattened["rewards"], expected_rewards)

    # Check macro-action boundaries: [0, 5, 8]
    # 0 = start, 5 = start of option 2, 8 = end
    assert flattened["macro_action_boundaries"] == [0, 5, 8]

    # Observations: initial + 8 primitive steps = 9 total
    assert len(flattened["obs"]) == 9


def test_flatten_hrl_trajectory_single_option():
    """Test flattening with single option (10 primitive steps)."""
    trajectory = {
        "obs": [
            np.array([1.0]),
            np.array([2.0]),
        ],
        "actions": ["option_0"],
        "rewards": [50.0],
        "infos": [
            {},
            {
                "macro_action": 0,
                "primitive_actions": list(range(10)),  # 10 primitive actions
                "primitive_infos": [{"step": i} for i in range(10)],
            },
        ],
    }

    flattened = _flatten_hrl_trajectory(trajectory=trajectory)

    # Should have 10 primitive steps
    assert len(flattened["actions"]) == 10
    assert len(flattened["rewards"]) == 10
    assert len(flattened["infos"]) == 10

    # Rewards should be distributed: 50.0 / 10 = 5.0 per step
    expected_rewards = [5.0] * 10
    np.testing.assert_array_almost_equal(flattened["rewards"], expected_rewards)

    # Macro-action boundaries: [0, 10]
    assert flattened["macro_action_boundaries"] == [0, 10]

    # Observations: initial + 10 steps = 11 total
    assert len(flattened["obs"]) == 11


def test_flatten_hrl_trajectory_multiple_options():
    """Test flattening with >2 options to verify generalization."""
    trajectory = {
        "obs": [
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
        ],
        "actions": ["option_0", "option_1", "option_2"],
        "rewards": [10.0, 20.0, 30.0],
        "infos": [
            {},
            {
                "macro_action": 0,
                "primitive_actions": [0, 1],
                "primitive_infos": [{"step": 0}, {"step": 1}],
            },
            {
                "macro_action": 1,
                "primitive_actions": [1, 0, 1, 0],
                "primitive_infos": [{"step": 2}, {"step": 3}, {"step": 4}, {"step": 5}],
            },
            {
                "macro_action": 2,
                "primitive_actions": [0],
                "primitive_infos": [{"step": 6}],
            },
        ],
    }

    flattened = _flatten_hrl_trajectory(trajectory=trajectory)

    # Total: 2 + 4 + 1 = 7 primitive steps
    assert len(flattened["actions"]) == 7
    assert len(flattened["rewards"]) == 7
    assert len(flattened["infos"]) == 7

    # Verify rewards distributed correctly
    expected_rewards = [5.0, 5.0] + [5.0] * 4 + [30.0]  # 10/2, 20/4, 30/1
    np.testing.assert_array_almost_equal(flattened["rewards"], expected_rewards)

    # Macro-action boundaries: [0, 2, 6, 7]
    assert flattened["macro_action_boundaries"] == [0, 2, 6, 7]


def test_flatten_hrl_trajectory_preserves_initial_and_final_obs():
    """Ensure initial and final observations are preserved after flattening."""
    initial_obs = np.array([1.0, 2.0, 3.0])
    final_obs = np.array([9.0, 8.0, 7.0])

    trajectory = {
        "obs": [initial_obs, final_obs],
        "actions": ["option_0"],
        "rewards": [15.0],
        "infos": [
            {},
            {
                "macro_action": 0,
                "primitive_actions": [0, 1, 2],
                "primitive_infos": [{"step": 0}, {"step": 1}, {"step": 2}],
            },
        ],
    }

    flattened = _flatten_hrl_trajectory(trajectory=trajectory)

    # Check initial obs preserved
    np.testing.assert_array_equal(flattened["obs"][0], initial_obs)

    # Check final obs preserved
    np.testing.assert_array_equal(flattened["obs"][-1], final_obs)

    # Total obs: initial + 3 primitive steps = 4
    assert len(flattened["obs"]) == 4


def test_flatten_hrl_trajectory_uneven_reward_distribution():
    """Test that uneven reward distribution is handled correctly."""
    trajectory = {
        "obs": [np.array([1.0]), np.array([2.0])],
        "actions": ["option_0"],
        "rewards": [10.0],
        "infos": [
            {},
            {
                "macro_action": 0,
                "primitive_actions": [0, 1, 2],  # 3 steps
                "primitive_infos": [{"step": 0}, {"step": 1}, {"step": 2}],
            },
        ],
    }

    flattened = _flatten_hrl_trajectory(trajectory=trajectory)

    # 10.0 / 3 = 3.333...
    expected_rewards = [10.0 / 3.0] * 3
    np.testing.assert_array_almost_equal(flattened["rewards"], expected_rewards)


def test_flatten_hrl_trajectory_infos_preserved():
    """Test that primitive info dicts are correctly preserved in flattened trajectory."""
    primitive_infos = [
        {"reward": 1.0, "patient_id": 0},
        {"reward": 2.0, "patient_id": 1},
        {"reward": 3.0, "patient_id": 0},
    ]

    trajectory = {
        "obs": [np.array([1.0]), np.array([2.0])],
        "actions": ["option_0"],
        "rewards": [6.0],
        "infos": [
            {},
            {
                "macro_action": 0,
                "primitive_actions": [0, 1, 2],
                "primitive_infos": primitive_infos,
            },
        ],
    }

    flattened = _flatten_hrl_trajectory(trajectory=trajectory)

    # Should have exactly 3 flattened infos from the primitive steps
    assert len(flattened["infos"]) == 3

    # Check that each primitive info is preserved
    for i, expected_info in enumerate(primitive_infos):
        assert flattened["infos"][i] == expected_info
