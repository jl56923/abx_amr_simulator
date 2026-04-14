"""Unit tests for tune_marl_agents.py.

Tests use the minimal_two_agent.yaml fixture and real ABXAMREnv instances —
no mocks of internal logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from abx_amr_simulator.training.tune_marl_agents import (
    tune_marl_agents_sequentially,
)

# --------------------------------------------------------------------------- #
# Fixtures / shared helpers
# --------------------------------------------------------------------------- #

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "marl_configs"
_FIXTURE_CONFIG = _FIXTURE_DIR / "minimal_two_agent.yaml"


def _minimal_tuning_cfg() -> dict:
    return {
        "optimization": {
            "n_trials": 2,
            "n_seeds_per_trial": 1,
            "truncated_primitive_steps": 40,
            "direction": "maximize",
            "sampler": "TPE",
            "stability_penalty_weight": 0.0,
            "n_eval_episodes": 1,
            "early_stopping": {"enabled": False},
        },
        "search_space": {
            "learning_rate": {
                "type": "float", "low": 1e-4, "high": 3e-4, "log": True
            },
        },
    }


# --------------------------------------------------------------------------- #
# Happy-path: all agents tuned
# --------------------------------------------------------------------------- #

class TestTuneMarlAgentsSequentially:

    def test_all_agents_tuned_returns_best_params_paths(self, tmp_path):
        """Both agents are tuned and best_params.json written in unified layout."""
        result = tune_marl_agents_sequentially(
            marl_config_path=_FIXTURE_CONFIG,
            tuning_config=_minimal_tuning_cfg(),
            optimization_dir=tmp_path / "optimization",
            experiment_folder="exp_1a__tuning",
            agent_ids=None,
            skip_if_exists=False,
            seed=0,
            n_workers=1,
        )

        assert set(result.keys()) == {"agent_0", "agent_1"}
        for agent_id, path in result.items():
            assert path.exists(), f"best_params.json missing for {agent_id}"
            assert path.name == "best_params.json"
            # Verify unified layout: optimization / exp_1a__tuning / {agent_id} /
            expected = (
                tmp_path / "optimization" / "exp_1a__tuning" / agent_id / "best_params.json"
            ).resolve()
            assert path == expected

    def test_best_params_json_is_valid(self, tmp_path):
        """best_params.json written for each agent is valid JSON with at least
        one hyperparameter key."""
        result = tune_marl_agents_sequentially(
            marl_config_path=_FIXTURE_CONFIG,
            tuning_config=_minimal_tuning_cfg(),
            optimization_dir=tmp_path / "optimization",
            experiment_folder="exp_valid",
            seed=0,
            n_workers=1,
        )

        for agent_id, path in result.items():
            data = json.loads(path.read_text())
            assert isinstance(data, dict), f"best_params.json not a dict for {agent_id}"
            assert len(data) > 0, f"best_params.json is empty for {agent_id}"

    def test_agents_processed_sequentially_in_config_order(self, tmp_path):
        """Both agents produce artefacts; order of keys in result matches
        the config's agent list order."""
        from abx_amr_simulator.utils.marl_factories import load_marl_config
        config = load_marl_config(_FIXTURE_CONFIG)
        config_order = [str(e["agent_id"]) for e in config["environment"]["agents"]]

        result = tune_marl_agents_sequentially(
            marl_config_path=_FIXTURE_CONFIG,
            tuning_config=_minimal_tuning_cfg(),
            optimization_dir=tmp_path / "optimization",
            experiment_folder="exp_order",
            seed=0,
            n_workers=1,
        )

        assert list(result.keys()) == config_order


# --------------------------------------------------------------------------- #
# agent_ids subset
# --------------------------------------------------------------------------- #

class TestAgentIdsSubset:

    def test_subset_only_tunes_specified_agents(self, tmp_path):
        """When agent_ids=['agent_0'] only agent_0 is tuned."""
        result = tune_marl_agents_sequentially(
            marl_config_path=_FIXTURE_CONFIG,
            tuning_config=_minimal_tuning_cfg(),
            optimization_dir=tmp_path / "optimization",
            experiment_folder="exp_subset",
            agent_ids=["agent_0"],
            seed=0,
            n_workers=1,
        )

        assert set(result.keys()) == {"agent_0"}
        assert result["agent_0"].exists()
        # agent_1 folder should not exist at all
        agent_1_dir = tmp_path / "optimization" / "exp_subset" / "agent_1"
        assert not agent_1_dir.exists()

    def test_unknown_agent_id_raises_value_error(self, tmp_path):
        """Requesting an agent_id not in the config raises ValueError."""
        with pytest.raises(ValueError, match="nonexistent_agent"):
            tune_marl_agents_sequentially(
                marl_config_path=_FIXTURE_CONFIG,
                tuning_config=_minimal_tuning_cfg(),
                optimization_dir=tmp_path / "optimization",
                experiment_folder="exp_bad",
                agent_ids=["nonexistent_agent"],
                seed=0,
                n_workers=1,
            )


# --------------------------------------------------------------------------- #
# skip_if_exists
# --------------------------------------------------------------------------- #

class TestSkipIfExists:

    def test_skip_if_exists_skips_agent_with_existing_best_params(self, tmp_path):
        """When best_params.json already exists for an agent and skip_if_exists
        is True, the study is skipped and the existing path is returned."""
        opt_dir = tmp_path / "optimization"
        agent_dir = opt_dir / "exp_skip" / "agent_0"
        agent_dir.mkdir(parents=True)
        existing = agent_dir / "best_params.json"
        existing.write_text('{"learning_rate": 9.9e-5}')

        result = tune_marl_agents_sequentially(
            marl_config_path=_FIXTURE_CONFIG,
            tuning_config=_minimal_tuning_cfg(),
            optimization_dir=opt_dir,
            experiment_folder="exp_skip",
            agent_ids=["agent_0"],
            skip_if_exists=True,
            seed=0,
            n_workers=1,
        )

        # The returned path should be the pre-existing file (not re-run).
        assert result["agent_0"] == existing.resolve()
        # Content must be unchanged (study was not rerun).
        assert json.loads(existing.read_text()) == {"learning_rate": 9.9e-5}


# --------------------------------------------------------------------------- #
# Missing config
# --------------------------------------------------------------------------- #

class TestMissingConfig:

    def test_nonexistent_marl_config_raises_file_not_found(self, tmp_path):
        """Providing a path to a non-existent MARL config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="MARL config not found"):
            tune_marl_agents_sequentially(
                marl_config_path=tmp_path / "does_not_exist.yaml",
                tuning_config=_minimal_tuning_cfg(),
                optimization_dir=tmp_path / "optimization",
                experiment_folder="exp_missing",
                seed=0,
                n_workers=1,
            )


# --------------------------------------------------------------------------- #
# Multi-worker distributed tuning
# --------------------------------------------------------------------------- #

class TestMultiWorker:

    def test_two_workers_complete_and_write_best_params(self, tmp_path):
        """n_workers=2 runs distributed subprocess workers and writes
        best_params.json for all agents."""
        cfg = _minimal_tuning_cfg()
        cfg["optimization"]["n_trials"] = 2

        result = tune_marl_agents_sequentially(
            marl_config_path=_FIXTURE_CONFIG,
            tuning_config=cfg,
            optimization_dir=tmp_path / "optimization",
            experiment_folder="exp_multiworker",
            agent_ids=None,
            skip_if_exists=False,
            seed=0,
            n_workers=2,
        )

        assert set(result.keys()) == {"agent_0", "agent_1"}
        for agent_id, path in result.items():
            assert path.exists()
            summary_path = path.parent / "study_summary.json"
            assert summary_path.exists()
            summary = json.loads(summary_path.read_text())
            assert summary["n_trials_completed"] == 2
