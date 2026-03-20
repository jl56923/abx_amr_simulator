# Tutorial 13: Callbacks and Logging

Goal: Learn how callback orchestration works in training, how to configure logging behavior from YAML, and how to use built-in callbacks for evaluation, trajectories, early stopping, and recurrent belief analysis.

Prerequisites: Completed Tutorial 1 (Basic Training) and Tutorial 2 (Config Scaffolding)

---

## Overview

In this project, callbacks are the main mechanism for:

- Logging patient-level aggregate metrics during training
- Running periodic evaluation and saving best models
- Saving full evaluation trajectories for downstream analysis
- Stopping training automatically (early stopping or target episode count)
- Capturing recurrent hidden states for belief probing

The training entrypoint wires these callbacks through:

- `abx_amr_simulator.utils.setup_callbacks`
- `abx_amr_simulator.training.train`

Core callback implementations live in:

- `src/abx_amr_simulator/callbacks/__init__.py`
- `src/abx_amr_simulator/callbacks/early_stopping.py`
- `src/abx_amr_simulator/callbacks/lstm_state_logger.py`

---

## 1. Callback Stack Used by Training

When you run:

```bash
python -m abx_amr_simulator.training.train \
  --umbrella-config /absolute/path/to/base_experiment.yaml
```

`setup_callbacks(...)` assembles a callback list in this order:

1. `PatientStatsLoggingCallback` (always on)
2. `EpisodeCounterCallback` (always on)
3. Evaluation callback if `eval_env` exists:
   - `DetailedEvalCallback` when `training.log_patient_trajectories: true`
   - otherwise standard SB3 `EvalCallback`
4. `CheckpointCallback` (always on)
5. `EarlyStoppingCallback` (optional, if enabled)

Why this matters:

- Episode counting and patient stats are always collected
- Evaluation callback determines whether full trajectory artifacts are saved
- Early stopping depends on evaluation metrics being present in logger state

---

## 2. Output Locations and What They Contain

For a run directory:

```text
results/<run_name>_<timestamp>/
```

you should expect:

- `logs/`
  - TensorBoard scalars from SB3 and custom callbacks
- `checkpoints/`
  - periodic model checkpoints from `CheckpointCallback`
  - best model from evaluation callback (`best_model.zip`)
  - final model saved at training end
- `eval_logs/`
  - evaluation artifacts; when using `DetailedEvalCallback`, trajectory `.npz` files are stored here
- `full_agent_env_config.yaml`
  - resolved config snapshot used for the run

Open TensorBoard:

```bash
tensorboard --logdir results/<run_name>_<timestamp>/logs
```

---

## 3. Built-in Callback Reference

## PatientStatsLoggingCallback

Purpose:

- Logs `info['patient_stats']` to TensorBoard every training step.
- Supports vectorized environments by averaging metrics across envs.

Key metric namespace:

- `patient_stats/...`

Typical uses:

- Track observed-vs-true attribute error trends
- Detect distribution drift across training
- Verify patient heterogeneity behavior is stable

## EpisodeCounterCallback

Purpose:

- Counts completed episodes robustly (important when episode lengths vary).
- Logs `training/completed_episodes` to TensorBoard.
- Optionally stops training after exact completed episode target.

Why it exists:

- `model.learn(total_timesteps=...)` controls timesteps, not actual episodes.
- Under clipping/variable-length episodes, timestep-only control can overshoot intended episode budget.

## DetailedEvalCallback

Purpose:

- Extends SB3 `EvalCallback` with richer evaluation logging.
- Can save full patient trajectories for each eval episode.

Enabled when:

- `training.log_patient_trajectories: true`

Useful for:

- Post-hoc analysis scripts in `abx_amr_simulator.analysis`
- Reward decomposition and policy interpretation

## EarlyStoppingCallback

Purpose:

- Stops training when a monitored metric plateaus.

Main parameters:

- `patience`
- `min_delta`
- `metric_name` (default `eval/mean_reward`)

Important note:

- It relies on evaluation metrics being logged, so keep evaluation enabled and frequent enough.

## LSTMStateLogger

Purpose:

- Logs recurrent hidden states, observations, actions, rewards, and true AMR into `.npz` episodes.

Designed for:

- Recurrent policy belief probing workflows (for example with `analysis/probe_hidden_belief.py`)

Compatibility:

- Intended for recurrent algorithms with `_last_lstm_states` (for example `RecurrentPPO`)

---

## 4. YAML Configuration Patterns

Most callback behavior is configured under `training:` in umbrella config.

Example:

```yaml
training:
  run_name: callbacks_demo
  total_num_training_episodes: 100
  eval_freq_every_n_episodes: 5
  save_freq_every_n_episodes: 5
  num_eval_episodes: 10
  log_patient_trajectories: true

  early_stopping:
    enabled: true
    patience: 8
    min_delta: 0.0
    metric_name: "eval/mean_reward"
```

CLI override examples:

```bash
python -m abx_amr_simulator.training.train \
  --umbrella-config /absolute/path/to/base_experiment.yaml \
  -o "training.log_patient_trajectories=true" \
  -o "training.early_stopping.enabled=true" \
  -o "training.early_stopping.patience=8"
```

Notes:

- Frequency keys are episode-based in config; conversion to timesteps happens internally.
- If early stopping is enabled but evaluation is too sparse, stopping decisions may be delayed.

---

## 5. Programmatic Usage Examples

## Example A: Use built-in setup_callbacks

```python
from abx_amr_simulator.utils import setup_callbacks

callbacks = setup_callbacks(
    config=config,
    run_dir=run_dir,
    eval_env=eval_env,
    stop_after_n_episodes=config["training"]["total_num_training_episodes"],
)

agent.learn(
    total_timesteps=config["training"]["_converted_total_timesteps"],
    callback=callbacks,
)
```

## Example B: Add LSTMStateLogger manually for recurrent runs

```python
from stable_baselines3.common.callbacks import CallbackList
from abx_amr_simulator.callbacks import LSTMStateLogger
from abx_amr_simulator.utils import setup_callbacks

base_callbacks = setup_callbacks(config=config, run_dir=run_dir, eval_env=eval_env)
lstm_logger = LSTMStateLogger(save_dir=f"{run_dir}/lstm_logs", log_freq=100)

callback = CallbackList(base_callbacks + [lstm_logger])

agent.learn(
    total_timesteps=config["training"]["_converted_total_timesteps"],
    callback=callback,
)
```

---

## 6. Troubleshooting

## Early stopping never triggers

Checks:

- Confirm `training.early_stopping.enabled: true`
- Confirm evaluation callback is active (`eval_env` exists)
- Confirm `metric_name` matches a logged metric (default `eval/mean_reward`)
- Confirm enough evaluations occur before training ends

## No trajectory files found

Checks:

- `training.log_patient_trajectories: true`
- Evaluation is actually being run
- Inspect `results/<run>/eval_logs/`

## LSTM logs are empty

Checks:

- Ensure algorithm is recurrent (for example `RecurrentPPO`/`HRL_RPPO` path)
- `LSTMStateLogger` warns once if model has no `_last_lstm_states`

## Episode count seems inconsistent with timesteps

Expected behavior:

- `EpisodeCounterCallback` tracks real completed episodes.
- Under variable-length episodes, completed episodes and fixed timestep budgets will not be a 1:1 mapping.

---

## 7. Recommended Defaults

For most experiments:

- Keep `log_patient_trajectories: true` for post-training analysis
- Keep evaluation every few episodes (`eval_freq_every_n_episodes` small enough for feedback)
- Enable early stopping only after baseline behavior is stable
- Use `EpisodeCounterCallback` target stopping (already wired via training pipeline)

For recurrent belief-analysis experiments:

- Add `LSTMStateLogger` programmatically
- Save logs in a dedicated subfolder (`<run_dir>/lstm_logs`)
- Pair with offline probe analysis tools

---

## Related Reading

- `docs/tutorials/01_basic_training.md`
- `docs/tutorials/07_hrl_diagnostics.md`
- `src/abx_amr_simulator/analysis/README.md`
- `src/abx_amr_simulator/callbacks/__init__.py`
- `src/abx_amr_simulator/callbacks/early_stopping.py`
- `src/abx_amr_simulator/callbacks/lstm_state_logger.py`
