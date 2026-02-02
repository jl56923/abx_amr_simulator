import json

import numpy as np

from abx_amr_simulator.utils.metrics import plot_metrics_ensemble_agents
from abx_amr_simulator.core import AMR_LeakyBalloon


class _DummyEnv:
    """Minimal env stub to drive plot_metrics_ensemble_agents without SB3."""

    def __init__(self):
        self.unwrapped = self
        self.antibiotic_names = ["A", "B"]
        self.antibiotics_AMR_dict = {
            "A": {
                "leak": 0.5,
                "flatness_parameter": 1.0,
                "permanent_residual_volume": 0.0,
                "initial_amr_level": 0.0,
            },
            "B": {
                "leak": 0.5,
                "flatness_parameter": 1.0,
                "permanent_residual_volume": 0.0,
                "initial_amr_level": 0.0,
            },
        }
        self.max_time_steps = 2

    def reset(self, seed=None):
        del seed
        return np.array([0.0], dtype=np.float32), {}


def _fake_trajectory():
    """Return a minimal trajectory with one populated step."""
    info_step = {
        "actual_amr_levels": {"A": 0.1, "B": 0.2},
        "visible_amr_levels": {"A": 0.1, "B": 0.2},
        "total_reward": 1.2,
        "overall_individual_reward_component": 0.8,
        "normalized_individual_reward": 0.8,
        "overall_community_reward_component": 0.4,
        "normalized_community_reward": 0.4,
        "count_clinical_benefits": 1,
        "count_clinical_failures": 0,
        "count_adverse_events": 0,
        "outcomes_breakdown": {
            "not_infected_no_treatment": 1,
            "not_infected_treated": 0,
            "infected_no_treatment": 0,
            "infected_treated": {
                "A": {"sensitive_infection_treated": 1, "resistant_infection_treated": 0},
                "B": {"sensitive_infection_treated": 0, "resistant_infection_treated": 0},
            },
        },
    }
    return {"obs": [np.array([0.0])], "actions": [], "rewards": [], "infos": [{}, info_step]}


def test_plot_metrics_ensemble_agents_writes_outcomes_json(tmp_path, monkeypatch):
    """Ensure ensemble plotting writes raw and summary outcome JSON with total_reward present."""

    dummy_env = _DummyEnv()

    # Avoid actual plotting side effects
    monkeypatch.setattr(AMR_LeakyBalloon, "plot_leaky_balloon_response_curve", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(AMR_LeakyBalloon, "plot_leaky_balloon_response_to_puff_sequence", lambda self, *args, **kwargs: None)

    # Force deterministic single-step trajectory
    monkeypatch.setattr(
        "abx_amr_simulator.utils.metrics.run_episode_and_get_trajectory",
        lambda model, env, deterministic=True: _fake_trajectory(),
    )

    experiment_folder = tmp_path
    plot_metrics_ensemble_agents(
        models=[object()],
        env=dummy_env,
        experiment_folder=str(experiment_folder),
        n_episodes_per_agent=1,
        deterministic=True,
        figures_folder_name="ensemble_figures",
        per_seed_figures=False,
        episode_seed_start=0,
    )

    ensemble_dir = experiment_folder / "ensemble_figures"
    raw_path = ensemble_dir / "overall_outcomes_summary_raw_vals.json"
    summary_path = ensemble_dir / "overall_outcomes_summary_summary_stats.json"

    assert raw_path.exists(), "Raw outcomes JSON was not written"
    assert summary_path.exists(), "Summary outcomes JSON was not written"

    raw = json.loads(raw_path.read_text())
    assert "overall_total_reward" in raw, "overall_total_reward missing from raw outcomes"
    assert raw["overall_total_reward"] == [1.2], "Unexpected total reward value"

    summary = json.loads(summary_path.read_text())
    assert "overall_total_reward" in summary, "overall_total_reward missing from summary stats"
    assert summary["overall_total_reward"]["p50"] == 1.2, "Median total reward incorrect"
