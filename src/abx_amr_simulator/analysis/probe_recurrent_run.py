#!/usr/bin/env python3
"""Probe a trained recurrent run directory end-to-end.

This entrypoint loads a trained recurrent policy from a run directory,
reconstructs the environment, runs evaluation with recurrent logging,
and computes probe metrics from logged hidden states.

Usage:
    python -m abx_amr_simulator.analysis.probe_recurrent_run \
        --run-dir /path/to/run_dir \
        --num-episodes 5
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import yaml

from stable_baselines3 import A2C, DQN, PPO

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from abx_amr_simulator.analysis.evaluative_plots import (
    compute_lstm_probe_stats,
    run_evaluation_episodes,
)
from abx_amr_simulator.core import ABXAMREnv
from abx_amr_simulator.utils.factories import (
    create_environment,
    create_patient_generator,
    create_reward_calculator,
    wrap_environment_for_hrl,
)


def extract_algorithm_name(*, config: Dict[str, Any]) -> str:
    """Extract algorithm name from either flat or nested config."""
    if "algorithm" in config and config.get("algorithm") is not None:
        return str(config.get("algorithm", "")).strip()

    nested_agent = config.get("agent_algorithm", {})
    if isinstance(nested_agent, dict):
        return str(nested_agent.get("algorithm", "")).strip()

    return ""


def is_recurrent_algorithm(*, algorithm: str) -> bool:
    """Return True if algorithm corresponds to recurrent policy variants."""
    algo_upper = algorithm.upper()
    return ("RECURRENT" in algo_upper) or ("RPPO" in algo_upper)


def resolve_run_artifacts(*, run_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    """Validate run artifacts and load config."""
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    model_path = run_dir / "checkpoints" / "best_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_path}")

    config_path = run_dir / "full_agent_env_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config artifact missing: {config_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(stream=config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must parse to dict: {config_path}")

    return model_path, config_path, config


def load_agent_and_environment(*, run_dir: Path) -> Tuple[Any, Any, Dict[str, Any], str]:
    """Load trained agent + reconstructed environment from run directory."""
    model_path, _config_path, config = resolve_run_artifacts(run_dir=run_dir)

    algorithm = extract_algorithm_name(config=config)
    if not algorithm:
        raise ValueError("Unable to determine algorithm from saved config")
    if not is_recurrent_algorithm(algorithm=algorithm):
        raise ValueError(
            "Probe entrypoint requires recurrent algorithm; "
            f"got '{algorithm}'"
        )

    reward_calculator = create_reward_calculator(config=config)
    patient_generator = create_patient_generator(config=config)
    environment = create_environment(
        config=config,
        reward_calculator=reward_calculator,
        patient_generator=patient_generator,
        wrap_monitor=False,
    )

    algo_upper = algorithm.upper()
    if algo_upper in ("HRL_PPO", "HRL_RPPO"):
        base_env = cast(ABXAMREnv, environment)
        environment = wrap_environment_for_hrl(env=base_env, config=config)

    if algo_upper in ("RECURRENTPPO", "HRL_RPPO"):
        if RecurrentPPO is None:
            raise ImportError(
                "sb3_contrib.RecurrentPPO is required but not importable in active environment"
            )
        agent = RecurrentPPO.load(path=model_path, env=environment, verbose=0)
    elif algo_upper in ("PPO", "HRL_PPO"):
        agent = PPO.load(path=model_path, env=environment, verbose=0)
    elif algo_upper == "A2C":
        agent = A2C.load(path=model_path, env=environment, verbose=0)
    elif algo_upper == "DQN":
        agent = DQN.load(path=model_path, env=environment, verbose=0)
    else:
        raise ValueError(f"Unsupported algorithm for probing: {algorithm}")

    return agent, environment, config, algorithm


def write_legacy_aggregate_probe_metrics(
    *,
    probe_dir: Path,
    seed_stats: Dict[str, Any],
    probe_results: Dict[str, Any],
) -> Path:
    """Write compatibility metrics JSON matching legacy workspace schema."""
    per_abx = seed_stats.get("per_antibiotic", [])
    hidden_shape = seed_stats.get("hidden_state_shape", [])
    hidden_dim = int(hidden_shape[1]) if len(hidden_shape) >= 2 else 0

    x_train = probe_results.get("X_train")
    x_test = probe_results.get("X_test")
    num_train_samples = int(x_train.shape[0]) if x_train is not None else 0
    num_test_samples = int(x_test.shape[0]) if x_test is not None else 0

    compat_metrics: Dict[str, Any] = {}
    for abx_stat in per_abx:
        abx_idx = int(abx_stat.get("antibiotic_idx", -1))
        if abx_idx < 0:
            continue

        key = f"ABX_{abx_idx}"
        test_r2 = abx_stat.get("test_r2")
        compat_metrics[key] = {
            "r2_score": float(test_r2) if test_r2 is not None else None,
            "r2_train": float(abx_stat.get("train_r2", 0.0)),
            "mae": float(abx_stat.get("test_mae", 0.0)),
            "mae_train": float(abx_stat.get("train_mae", 0.0)),
            "hidden_dim": hidden_dim,
            "num_train_samples": num_train_samples,
            "num_test_samples": num_test_samples,
        }

    output_path = probe_dir / "aggregate_probe_metrics.json"
    with open(output_path, "w", encoding="utf-8") as metrics_file:
        json.dump(obj=compat_metrics, fp=metrics_file, indent=2)

    return output_path


def main() -> int:
    """Run recurrent probing from a trained run directory."""
    parser = argparse.ArgumentParser(
        description="Probe a trained recurrent run directory end-to-end"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing checkpoints/best_model.zip and full_agent_env_config.yaml",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run",
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=None,
        help="Output directory for probe artifacts (default: <run_dir>/lstm_probe)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Probe test split fraction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for probe train/test split",
    )

    args = parser.parse_args()

    probe_dir = args.probe_dir if args.probe_dir is not None else args.run_dir / "lstm_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    agent = None
    environment = None
    try:
        agent, environment, _config, algorithm = load_agent_and_environment(run_dir=args.run_dir)

        _evaluation_data = run_evaluation_episodes(
            model=agent,
            env=environment,
            num_episodes=args.num_episodes,
            is_hrl=("HRL" in algorithm.upper()),
            is_recurrent=True,
            recurrent_log_dir=probe_dir,
        )

        seed_stats, probe_results = compute_lstm_probe_stats(
            log_dir=probe_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        seed_stats_path = probe_dir / "lstm_probe_seed_stats.json"
        with open(seed_stats_path, "w", encoding="utf-8") as seed_stats_file:
            json.dump(obj=seed_stats, fp=seed_stats_file, indent=2)

        compat_path = write_legacy_aggregate_probe_metrics(
            probe_dir=probe_dir,
            seed_stats=seed_stats,
            probe_results=probe_results,
        )

        print(f"[probe_recurrent_run] complete: run_dir={args.run_dir}")
        print(f"[probe_recurrent_run] seed stats: {seed_stats_path}")
        print(f"[probe_recurrent_run] compatibility metrics: {compat_path}")
        return 0
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
