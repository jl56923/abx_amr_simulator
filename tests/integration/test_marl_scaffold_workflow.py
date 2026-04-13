"""Integration smoke test for scaffolded MARL workflow.

Validates the package-owned user path:
1. Scaffold default config folders
2. Scaffold default option folders
3. Run MARL training from configs/marl/minimal_two_agent.yaml
4. Verify expected run artifacts
"""

from __future__ import annotations

from pathlib import Path

import yaml

from abx_amr_simulator.hrl import setup_options_folders_with_defaults
from abx_amr_simulator.training.train_marl import run_marl_training
from abx_amr_simulator.utils import setup_config_folders_with_defaults


def test_scaffolded_marl_config_runs_end_to_end(tmp_path: Path) -> None:
    """Scaffolded MARL defaults should execute a short training run successfully."""
    experiments_dir = tmp_path / "experiments"
    results_dir = tmp_path / "results"
    run_name = "marl_scaffold_integration"

    setup_config_folders_with_defaults(target_path=experiments_dir)
    setup_options_folders_with_defaults(target_path=experiments_dir)

    marl_config_path = (
        experiments_dir
        / "configs"
        / "marl"
        / "minimal_two_agent.yaml"
    )
    assert marl_config_path.exists()

    run_marl_training(
        marl_config_path=marl_config_path,
        results_dir=results_dir,
        run_name=run_name,
        seed=42,
        overrides=[
            "training.total_primitive_steps=64",
            "training.n_steps=8",
            "training.batch_size=4",
            "training.n_epochs=1",
            "training.eval_freq_episodes=1",
            "training.save_freq_episodes=1",
            "training.n_eval_episodes=1",
        ],
    )

    run_candidates = sorted(results_dir.glob(f"{run_name}_????????_??????"))
    assert len(run_candidates) == 1
    run_dir = run_candidates[0]
    checkpoint_dir = run_dir / "checkpoints"
    saved_config_path = run_dir / "marl_full_agents_env_config.yaml"

    assert run_dir.exists()
    assert checkpoint_dir.exists()
    assert saved_config_path.exists()

    assert (checkpoint_dir / "final_model_agent_0.zip").exists()
    assert (checkpoint_dir / "final_model_agent_1.zip").exists()

    saved_config = yaml.safe_load(saved_config_path.read_text(encoding="utf-8"))
    agent_entries = saved_config["environment"]["agents"]

    for entry in agent_entries:
        option_library_path = Path(entry["option_library"])
        assert option_library_path.is_absolute()
        assert option_library_path.exists()
