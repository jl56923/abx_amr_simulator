# Tutorial 14: MARL Isolation Tuning

**Goal**: Learn how to tune PPO/RPPO hyperparameters for MARL experiments using
per-agent isolation tuning.

**Prerequisites**: Completed Tutorial 6 (Optimization with Optuna) and Tutorial 13
(MARL Training Quick Start).

---

## Overview

This tutorial covers:

- Why MARL tuning is done per-agent in isolation (not jointly)
- The tuning config format for MARL studies
- Running `tune_marl_agents_sequentially` to tune all agents in one call
- Artifact layout and what gets written
- Subset tuning and `skip_if_exists`
- Multi-worker distributed mode
- Tuning a single agent via the CLI (for SLURM scripts)
- Connecting tuning results to training

---

## 1. Why Per-Agent Isolation Tuning?

Joint MARL hyperparameter tuning — where all agents' hyperparameters are
optimized simultaneously in the full multi-agent environment — suffers from
**non-stationarity**:

> Each trial's reward signal depends on which hyperparameter configuration the
> *other* agent happens to be using in that trial. As the other agent's
> hyperparameters shift across trials, the reward landscape seen by the current
> agent shifts too. The Optuna surrogate model is fitting a moving target.

This inflates trial-to-trial variance and degrades the quality of the surrogate,
causing the study to explore less effectively.

**Isolation tuning** avoids this entirely: each agent is tuned independently in
its own single-agent `ABXAMREnv + OptionsWrapper`. The environment and reward
function are extracted from the MARL config for that agent, but the other agents
are not present. This gives a stationary, well-posed optimization problem per
agent.

For the cooperative agents in this simulator — where inter-agent coupling flows
only through the shared AMR dynamics already present in each agent's single-agent
environment — isolation tuning is a good approximation to joint optimality and
substantially more stable.

> **Note on `option_gamma`**: This hyperparameter is shared across all agents by
> `MARLOptionsWrapper` and is set at the MARL config level, not per-agent. Do not
> include `option_gamma` in the MARL tuning search space.

---

## 2. The Tuning Config Format

MARL tuning uses the same config format as single-agent HRL tuning (see
Tutorial 6), with one MARL-specific difference: the `truncated_primitive_steps`
key replaces `truncated_episodes` because MARL training is budgeted in primitive
environment steps, not episodes.

```yaml
optimization:
  n_trials: 50
  n_seeds_per_trial: 2
  truncated_primitive_steps: 50000   # primitive steps per trial (not episodes)
  direction: maximize
  sampler: TPE
  stability_penalty_weight: 0.1
  n_eval_episodes: 5
  early_stopping:
    enabled: true
    n_trials_no_improve: 15

search_space:
  learning_rate:
    type: float
    low: 1.0e-5
    high: 1.0e-3
    log: true
  n_steps:
    type: int
    low: 128
    high: 512
    step: 128
  batch_size:
    type: int
    low: 64
    high: 256
    step: 64
```

All agents in a MARL config share the same tuning config. If agents are
meaningfully different (e.g. very different patient cohort sizes), you can
override `tuning_n_patients` at the call site — but the search space is always
shared.

> **Note on HRL_RPPO agents:** The isolation tuning infrastructure respects each
> agent's `algorithm` field. An agent with `algorithm: HRL_RPPO` will be tuned
> using a `RecurrentPPO_Masked` policy. The standard PPO hyperparameter search
> space (learning_rate, n_steps, batch_size) applies to both PPO and RPPO agents.
> LSTM-specific parameters (`lstm_hidden_size`, `n_lstm_layers`,
> `enable_critic_lstm`) are set via `lstm_kwargs` in the MARL config and are not
> included in the tuning search space by default.

---

## 3. `tune_marl_agents_sequentially`

The canonical entry point for MARL isolation tuning is
`tune_marl_agents_sequentially` from `abx_amr_simulator.training`.

```python
from pathlib import Path
import yaml
from abx_amr_simulator.training import tune_marl_agents_sequentially

tuning_cfg = yaml.safe_load(open("tuning_configs/hrl_ppo_marl_tuning.yaml"))

best_params = tune_marl_agents_sequentially(
    marl_config_path=Path("configs/marl/my_experiment.yaml"),
    tuning_config=tuning_cfg,
    optimization_dir=Path("workspace/optimization"),
    experiment_folder="exp_1a__marl_mls1n__tuning",
    agent_ids=None,          # None = tune every agent in the config
    skip_if_exists=True,
    seed=42,
    n_workers=1,
)
# Returns:
# {
#   "agent_0": Path(".../exp_1a__marl_mls1n__tuning/agent_0/best_params.json"),
#   "agent_1": Path(".../exp_1a__marl_mls1n__tuning/agent_1/best_params.json"),
# }
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `marl_config_path` | `Path` | Path to the MARL template YAML |
| `tuning_config` | `dict` | Loaded tuning config (the YAML dict, not a path) |
| `optimization_dir` | `Path` | Root directory for all Optuna outputs |
| `experiment_folder` | `str` | Sub-folder under `optimization_dir` grouping all agents for this experiment |
| `agent_ids` | `list[str]` or `None` | Agents to tune; `None` tunes all in config order |
| `skip_if_exists` | `bool` | Skip if `best_params.json` already exists for an agent |
| `seed` | `int` | Base random seed forwarded to each agent's Optuna study |
| `n_workers` | `int` | Workers per agent study (1 = in-process; >1 = distributed SQLite) |

**Return value:** A dict mapping `agent_id → Path` to each agent's
`best_params.json`.

---

## 4. Artifact Layout

All tuning outputs for one experiment land under a single experiment-level folder:

```
optimization/
  exp_1a__marl_mls1n__tuning/
    agent_0/
      best_params.json
      optuna_study.db
      study_summary.json
    agent_1/
      best_params.json
      optuna_study.db
      study_summary.json
```

The naming convention for `experiment_folder` used in the workspace is
`exp_{id}__marl_{cfg_abbrev}__tuning`, but you can pass any string. The function
does not enforce a naming convention.

Each agent's sub-folder contains:

- `best_params.json` — the hyperparameter dict selected by Optuna
- `optuna_study.db` — the SQLite database holding all trial records
- `study_summary.json` — metadata: number of trials completed, best value,
  best trial number

---

## 5. Subset Tuning and `skip_if_exists`

### Tuning only some agents

Pass a list of agent IDs to tune a subset:

```python
best_params = tune_marl_agents_sequentially(
    marl_config_path=Path("configs/marl/my_experiment.yaml"),
    tuning_config=tuning_cfg,
    optimization_dir=Path("workspace/optimization"),
    experiment_folder="exp_2a__marl_mlc20__tuning",
    agent_ids=["agent_p"],   # only tune agent_p; agent_n will not be created
    skip_if_exists=True,
    seed=42,
    n_workers=1,
)
```

Requesting an agent ID that does not appear in the MARL config raises
`ValueError`.

### Resuming partial runs with `skip_if_exists`

If a study was interrupted (cluster preemption, power loss, etc.), re-run the
same call with `skip_if_exists=True`. Any agent whose `best_params.json` already
exists is skipped entirely and its existing path is returned. Any agent without
`best_params.json` is re-tuned from scratch (Optuna will resume from the existing
SQLite database if one is present).

```python
# Re-run after interruption — already-completed agents are skipped.
best_params = tune_marl_agents_sequentially(
    ...,
    skip_if_exists=True,
)
```

---

## 6. Multi-Worker Distributed Mode

When `n_workers > 1`, each agent study is run as a set of coordinated
subprocesses sharing a single SQLite database. This follows the same staggered-
start pattern used for single-agent tuning:

1. Worker 0 is launched first and given time to initialise the SQLite database.
2. Workers 1–N are launched once `optuna_study.db` appears on disk.
3. All workers are awaited; any non-zero exit code raises `RuntimeError`.

```python
best_params = tune_marl_agents_sequentially(
    marl_config_path=Path("configs/marl/my_experiment.yaml"),
    tuning_config=tuning_cfg,
    optimization_dir=Path("workspace/optimization"),
    experiment_folder="exp_1a__marl_mls1n__tuning",
    seed=42,
    n_workers=4,   # 4 subprocess workers share one SQLite study per agent
)
```

**Important**: SQLite distributed tuning works well on local NFS-free
filesystems. For cluster environments with shared filesystems, prefer 1 worker
per experiment with SLURM array parallelism (see Section 7) rather than
relying on SQLite concurrent writes over NFS.

---

## 7. Tuning a Single Agent via the CLI

For SLURM scripts where each agent's study is submitted as a separate job,
the `tune_marl_agent` module provides a CLI for tuning exactly one agent from
a MARL config:

```bash
python -m abx_amr_simulator.training.tune_marl_agent \
  --marl-config   $(pwd)/configs/marl/my_experiment.yaml \
  --agent-id      agent_0 \
  --tuning-config $(pwd)/tuning_configs/hrl_ppo_marl_tuning.yaml \
  --optimization-dir $(pwd)/workspace/optimization/exp_1a__marl_mls1n__tuning \
  --run-name      agent_0 \
  --seed          42 \
  --skip-if-exists
```

The artifact layout mirrors `tune_marl_agents_sequentially`:
`optimization_dir / run_name / best_params.json`. When the SLURM script passes
the experiment-level folder as `--optimization-dir` and the agent ID as
`--run-name`, artefacts land at:

```
optimization/
  exp_1a__marl_mls1n__tuning/
    agent_0/
      best_params.json
      optuna_study.db
      study_summary.json
```

For distributed multi-worker studies from SLURM, add `--worker-id` and
`--total-workers`:

```bash
# Worker 0
python -m abx_amr_simulator.training.tune_marl_agent \
  --marl-config   ... \
  --agent-id      agent_0 \
  --tuning-config ... \
  --optimization-dir ... \
  --run-name      agent_0 \
  --seed          42 \
  --worker-id     0 \
  --total-workers 4

# Workers 1–3 (same command, different --worker-id)
```

All workers for the same agent must share the same `--optimization-dir`,
`--run-name`, and database.

---

## 8. Connecting Tuning Results to Training

Once tuning is complete, load `best_params.json` for each agent and pass them
to `train_marl` via an `agent_init_params.json` file.

### `agent_init_params.json` format

```json
{
  "experiment_id": "exp_1a",
  "agents": {
    "agent_0": {
      "best_params_path": "/abs/path/to/exp_1a__marl_mls1n__tuning/agent_0/best_params.json"
    },
    "agent_1": {
      "best_params_path": "/abs/path/to/exp_1a__marl_mls1n__tuning/agent_1/best_params.json"
    }
  }
}
```

The `best_params_path` for each agent points to the `best_params.json` written
by the tuning study. Leave `best_params_path` as `""` for any agent that was not
tuned (default PPO hyperparameters will be used).

### Passing it to the training entrypoint

```bash
python -m abx_amr_simulator.training.train_marl \
  --marl-config      $(pwd)/configs/marl/my_experiment.yaml \
  --results-dir      $(pwd)/results \
  --run-name         exp_1a_seed42 \
  --seed             42 \
  --agent-init-params $(pwd)/workspace/optimization/exp_1a__marl_mls1n__tuning/agent_init_params.json
```

At startup the trainer reads `best_params_path` for each agent, loads the JSON,
and initialises that agent's PPO policy with the tuned hyperparameters. Any agent
without a path uses the values from the MARL config's `training` section.

### Building `agent_init_params.json` programmatically

```python
import json
from pathlib import Path
from abx_amr_simulator.training import tune_marl_agents_sequentially

opt_dir = Path("workspace/optimization")
exp_folder = "exp_1a__marl_mls1n__tuning"

best_params = tune_marl_agents_sequentially(
    marl_config_path=Path("configs/marl/my_experiment.yaml"),
    tuning_config=tuning_cfg,
    optimization_dir=opt_dir,
    experiment_folder=exp_folder,
    skip_if_exists=True,
    seed=42,
    n_workers=1,
)

agent_init = {
    "experiment_id": "exp_1a",
    "agents": {
        agent_id: {"best_params_path": str(path)}
        for agent_id, path in best_params.items()
    },
}

out = opt_dir / exp_folder / "agent_init_params.json"
out.write_text(json.dumps(agent_init, indent=2))
```

---

## 9. Troubleshooting

### `FileNotFoundError: MARL config not found`

The `marl_config_path` must be a path to a file that exists on disk. Check
that the path is absolute or correctly resolved relative to your working
directory.

### `ValueError: Requested agent IDs not found in MARL config`

The `agent_ids` you passed does not match the `agent_id` fields in the MARL
config. Print `config["environment"]["agents"]` to see what IDs are defined.

### `FileNotFoundError: Tuning completed but best_params.json not found`

The study ran but Optuna did not write `best_params.json`. This usually means
`n_trials` was zero or all trials failed. Check the study summary and Optuna
logs in `optuna_study.db`.

### `RuntimeError: Distributed MARL tuning failed ... exit codes: [1, 0]`

One or more subprocess workers exited with a non-zero code. The worker's stderr
output contains the actual exception. Re-run the same command without
`n_workers > 1` to see the error in-process.

### SQLite locking errors in distributed mode

If you are running on a shared NFS filesystem (e.g. on a cluster with
`$SCRATCH` or shared project directories), SQLite concurrent writes can
deadlock. Use `n_workers=1` and submit parallel SLURM array jobs that each
tune one agent sequentially instead.

---

## 10. Key Takeaways

1. Tune each agent in isolation in its own single-agent environment — joint MARL
   tuning inflates variance due to non-stationarity.
2. `tune_marl_agents_sequentially` is the canonical Python API; use
   `tune_marl_agent` CLI for SLURM-based scheduling.
3. All per-agent artefacts for one experiment land under one experiment-level
   folder (`optimization_dir / experiment_folder / {agent_id} /`).
4. `agent_ids=None` tunes all agents; pass a list to tune a subset.
5. `skip_if_exists=True` makes tuning runs safely re-entrant.
6. Do not include `option_gamma` in the MARL search space — it is shared and
   set at the MARL config level.
7. Pass tuning results to training via `--agent-init-params` with an
   `agent_init_params.json` file.

---

## Next Tutorials

- [06_optimization.md](06_optimization.md) — single-agent optimization background
- [13_marl_training_quickstart.md](13_marl_training_quickstart.md) — MARL training reference
- [12_callbacks_and_logging.md](12_callbacks_and_logging.md) — custom callbacks and logging
