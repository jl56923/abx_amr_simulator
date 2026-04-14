"""Sequential isolation tuning for all agents in a MARL config.

This module provides ``tune_marl_agents_sequentially``, the canonical
high-level entry point for per-agent isolation tuning of MARL experiments.

Design rationale
----------------
MARL hyperparameter tuning in this package uses *isolation tuning*: each agent
is tuned independently in its own single-agent ``ABXAMREnv + OptionsWrapper``,
not in the full joint ``MARLOptionsWrapper`` environment.  Joint tuning suffers
from non-stationarity — each trial's reward signal depends on the other agent's
current HP configuration, inflating variance and degrading the Optuna surrogate.
For the cooperative, non-adversarial agents in this simulator (whose only
coupling is through shared AMR dynamics already present in the single-agent env)
isolation tuning is both stable and a good approximation to joint optimality.

Artefact layout
---------------
All per-agent tuning outputs for one experiment live under a single
experiment-level directory::

    optimization_dir/
      <experiment_folder>/
        <agent_id>/
          best_params.json
          optuna_study.db
          study_summary.json

The ``experiment_folder`` is determined by the caller — this module does not
impose a naming convention.  The workspace uses
``exp_{id}__marl_{cfg_abbrev}__tuning`` by convention; other callers may use
any string.

Multi-worker distributed tuning
---------------------------------
When ``n_workers > 1``, each agent study is run as a set of coordinated
subprocesses sharing a single SQLite Optuna database.  Worker 0 is launched
first and allowed to initialise the database before the remaining workers are
started (the staggered-start pattern).  This module owns the subprocess
management so callers do not have to reimplement it.

Typical usage
-------------
::

    from pathlib import Path
    import yaml
    from abx_amr_simulator.training.tune_marl_agents import (
        tune_marl_agents_sequentially,
    )

    tuning_cfg = yaml.safe_load(open("hrl_ppo_marl_tuning.yaml"))

    best_params = tune_marl_agents_sequentially(
        marl_config_path=Path("marl_configs/marl_lpp_cov20.yaml"),
        tuning_config=tuning_cfg,
        optimization_dir=Path("workspace/optimization"),
        experiment_folder="exp_1a__marl_mlc20__tuning",
        agent_ids=None,          # tune all agents in the config
        skip_if_exists=True,
        seed=42,
        n_workers=8,
    )
    # best_params == {"agent_p": Path(".../agent_p/best_params.json"),
    #                 "agent_n": Path(".../agent_n/best_params.json")}
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from abx_amr_simulator.training.tune_marl_agent import run_marl_agent_tuning
from abx_amr_simulator.utils.marl_factories import load_marl_config


def tune_marl_agents_sequentially(
    marl_config_path: Path,
    tuning_config: Dict[str, Any],
    optimization_dir: Path,
    experiment_folder: str,
    agent_ids: Optional[List[str]] = None,
    skip_if_exists: bool = True,
    seed: int = 42,
    n_workers: int = 1,
) -> Dict[str, Path]:
    """Tune all (or a specified subset of) agents in a MARL config, one at a time.

    Each agent is tuned in isolation using a single-agent ``ABXAMREnv +
    OptionsWrapper`` extracted from the MARL config.  Agents are processed
    sequentially in the order they appear in the config (or in the order of
    ``agent_ids`` when supplied).

    Artefacts for each agent land at::

        optimization_dir / experiment_folder / {agent_id} /

    When ``n_workers > 1``, the study for each agent is run as a set of
    coordinated subprocesses sharing a SQLite Optuna database.  Worker 0 is
    launched first so it can initialise the database; the remaining workers
    start once ``optuna_study.db`` appears on disk.

    Args:
        marl_config_path: Path to the MARL template YAML file.  The path (not
            a pre-loaded dict) is used so it can be forwarded to distributed
            worker subprocesses as ``--marl-config``.
        tuning_config: Loaded tuning configuration dict (e.g. from
            ``hrl_ppo_marl_tuning.yaml``).  All agents share the same tuning
            configuration.
        optimization_dir: Root directory for Optuna study outputs.
        experiment_folder: Folder name under ``optimization_dir`` that groups
            all per-agent artefacts for this experiment.  The naming convention
            is determined by the caller.
        agent_ids: Agent IDs to tune.  ``None`` (default) tunes every agent
            found in the MARL config in config order.
        skip_if_exists: If ``True`` and ``best_params.json`` already exists for
            an agent, skip that agent's study.  Defaults to ``True``.
        seed: Base random seed forwarded to each agent's Optuna study.
        n_workers: Number of parallel SQLite workers per agent study.
            ``1`` runs in-process (no subprocesses).

    Returns:
        Dict mapping ``agent_id`` → absolute ``Path`` to that agent's
        ``best_params.json``.

    Raises:
        FileNotFoundError: If ``marl_config_path`` does not exist, or if
            ``best_params.json`` is missing after a study completes.
        RuntimeError: If any distributed worker subprocess exits with a
            non-zero status.
        ValueError: If a requested ``agent_id`` is not present in the config.
    """
    marl_config_path = Path(marl_config_path).resolve()
    if not marl_config_path.exists():
        raise FileNotFoundError(
            f"MARL config not found: {marl_config_path}"
        )

    config = load_marl_config(marl_config_path)

    # Resolve agent IDs from config if not supplied.
    config_agent_ids = [
        str(e["agent_id"]) for e in config["environment"]["agents"]
    ]
    if agent_ids is None:
        agents_to_tune = config_agent_ids
    else:
        unknown = [aid for aid in agent_ids if aid not in config_agent_ids]
        if unknown:
            raise ValueError(
                f"Requested agent IDs not found in MARL config: {unknown}. "
                f"Available: {config_agent_ids}"
            )
        agents_to_tune = list(agent_ids)

    # Artefacts land under optimization_dir / experiment_folder / agent_id /
    agent_optimization_dir = optimization_dir / experiment_folder
    worker_count = max(1, int(n_workers))

    results: Dict[str, Path] = {}

    for agent_id in agents_to_tune:
        agent_dir = agent_optimization_dir / agent_id

        if worker_count == 1:
            run_marl_agent_tuning(
                config=config,
                agent_id=agent_id,
                tuning_config=tuning_config,
                optimization_dir=agent_optimization_dir,
                run_name=agent_id,
                seed=seed,
                skip_if_exists=skip_if_exists,
            )
        else:
            _run_distributed_agent_study(
                marl_config_path=marl_config_path,
                agent_id=agent_id,
                tuning_config=tuning_config,
                agent_optimization_dir=agent_optimization_dir,
                agent_dir=agent_dir,
                seed=seed,
                worker_count=worker_count,
                skip_if_exists=skip_if_exists,
            )

        best_params_path = agent_dir / "best_params.json"
        if not best_params_path.exists():
            raise FileNotFoundError(
                f"Tuning completed for agent '{agent_id}' but "
                f"best_params.json not found at: {best_params_path}"
            )
        results[agent_id] = best_params_path.resolve()

    return results


def _run_distributed_agent_study(
    *,
    marl_config_path: Path,
    agent_id: str,
    tuning_config: Dict[str, Any],
    agent_optimization_dir: Path,
    agent_dir: Path,
    seed: int,
    worker_count: int,
    skip_if_exists: bool,
) -> None:
    """Launch ``worker_count`` coordinated subprocesses for one agent study.

    Worker 0 is started first and given up to 30 seconds to initialise the
    SQLite database before the remaining workers are launched.  All workers are
    awaited; any non-zero exit code raises ``RuntimeError``.

    This is a private helper; callers should use
    ``tune_marl_agents_sequentially`` instead.

    Args:
        marl_config_path: Resolved path to the MARL config YAML.
        agent_id: The agent whose study is being run.
        tuning_config: Tuning configuration dict (written to a temp file
            forwarded via ``--tuning-config``).
        agent_optimization_dir: Parent directory for the agent's artefact
            folder (``agent_optimization_dir / agent_id /``).
        agent_dir: Full path to the agent's artefact directory.
        seed: Base random seed.
        worker_count: Number of subprocesses to launch.
        skip_if_exists: Forwarded as ``--skip-if-exists`` to workers.

    Raises:
        RuntimeError: If any worker exits with a non-zero status.
    """
    import json
    import tempfile

    # Write the tuning config to a temporary file so workers can read it.
    # We cannot pass the dict directly on the command line.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        prefix="marl_tuning_cfg_",
    ) as tf:
        import yaml
        yaml.dump(tuning_config, tf)
        tuning_cfg_tmp = Path(tf.name)

    try:
        worker_cmds: List[List[str]] = []
        for worker_id in range(worker_count):
            cmd = [
                sys.executable,
                "-m",
                "abx_amr_simulator.training.tune_marl_agent",
                "--marl-config", str(marl_config_path),
                "--agent-id", str(agent_id),
                "--tuning-config", str(tuning_cfg_tmp),
                "--optimization-dir", str(agent_optimization_dir),
                "--run-name", agent_id,
                "--seed", str(seed),
                "--worker-id", str(worker_id),
                "--total-workers", str(worker_count),
            ]
            if skip_if_exists:
                cmd.append("--skip-if-exists")
            worker_cmds.append(cmd)

        procs: List[subprocess.Popen] = []

        # Launch worker 0 first so it initialises the SQLite database.
        procs.append(subprocess.Popen(worker_cmds[0]))

        db_path = agent_dir / "optuna_study.db"
        for _ in range(30):
            if db_path.exists():
                break
            if procs[0].poll() is not None:
                break
            time.sleep(1)

        for cmd in worker_cmds[1:]:
            procs.append(subprocess.Popen(cmd))

        exit_codes = [proc.wait() for proc in procs]
        failed = [code for code in exit_codes if code != 0]
        if failed:
            raise RuntimeError(
                f"Distributed MARL tuning failed for agent '{agent_id}' "
                f"(folder: {agent_dir}) with worker exit codes: {exit_codes}"
            )
    finally:
        tuning_cfg_tmp.unlink(missing_ok=True)
