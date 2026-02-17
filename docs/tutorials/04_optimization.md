# Tutorial 4: Optimization With Optuna

**Goal**: Master hyperparameter tuning with Optuna, including study management, storage backends, parallel optimization, and troubleshooting common issues.

**Prerequisites**: Completed Tutorial 2 (Config Scaffolding) and Tutorial 3 (Custom Experiments)

---

## Overview

This tutorial covers:
- Setting up tuning configs and running basic optimization
- Study naming and management strategies
- Storage backends (SQLite vs PostgreSQL)
- Parallel tuning with multiple workers
- Resuming interrupted studies
- Training with optimized hyperparameters
- Common failures and troubleshooting

---

## 1. Scaffold Tuning Configs

If you do not already have tuning configs, scaffold the defaults (from `my_project/experiments/` directory):

```bash
python -m abx_amr_simulator.training.setup_tuning --target-path .
```

This creates:
- `tuning_configs/ppo_tuning_default.yaml` — PPO search space
- `tuning_configs/hrl_ppo_tuning_default.yaml` — HRL PPO search space

**Tuning config structure:**
```yaml
tuning_config:
  optimization:
    n_trials: 50                      # Number of parameter configurations to try
    n_seeds_per_trial: 3              # Seeds to average per configuration
    truncated_episodes: 50            # Reduced episodes for fast evaluation
    direction: maximize               # maximize or minimize
    sampler: TPE                      # TPE (Bayesian), Random, or CMAES
    stability_penalty_weight: 0.2     # Penalize inconsistent learning (see below)
  
  search_space:
    learning_rate:
      type: float
      low: 1.0e-5
      high: 1.0e-3
      log: true                       # Sample on log scale
    
    n_steps:
      type: int
      low: 128
      high: 2048
      step: 128                       # Discrete steps (128, 256, 384, ...)
    
    gamma:
      type: float
      low: 0.95
      high: 0.999
```

---

## 2. Understanding Study Naming

**Two names control optimization runs:**

### `--run-name` (Required)
- **Purpose**: Names the optimization outputs directory
- **Scope**: Local filesystem organization
- **Created**: `optimization/<run_name>/`
- **Contains**: `optuna_study.db`, `best_params.json`, `study_summary.json`, configs
- **Registry**: Tracked in `optimization/.optimization_completed.txt`
- **Use for**: Organizing different experiments ("ppo_baseline_tune", "lambda_sweep_tune", etc.)

### `--study-name` (Optional)
- **Purpose**: Names the Optuna study object in the database
- **Scope**: Database-level identifier (critical for parallel workers)
- **Default**: Auto-generated from run_name if not specified
- **Use for**: Shared studies across multiple workers (parallel tuning)

**When to use each:**

| Scenario | --run-name | --study-name | Notes |
|----------|-----------|--------------|-------|
| **Single-worker SQLite** | `exp1_tune` | *(auto)* | Study name auto-set to run_name |
| **Parallel PostgreSQL** | `exp1_tune` | `exp1_tune` | All workers use same study_name |
| **Multiple independent runs** | `exp1_v1`, `exp1_v2` | *(auto)* | Different run_names create separate studies |
| **Resume interrupted run** | `exp1_tune` *(same)* | *(auto)* | Re-use exact same run_name |

**Naming conventions:**
- Use descriptive, filesystem-safe names (no spaces, special chars)
- Include variant info: `ppo_lambda_0.5_tune`, `hrl_high_risk_patients_tune`
- Add version suffixes for iterations: `exp1_tune_v1`, `exp1_tune_v2`
- For parallel runs, **--study-name must match across all workers**

---

## 3. Storage Backends: SQLite vs PostgreSQL

### SQLite (Default)

**When to use:**
- Single-worker tuning
- Local development/testing
- Simple workflows without concurrency

**Advantages:**
- No setup required (automatic)
- Portable (single file database)
- Simple debugging

**Limitations:**
- **No concurrent writes** — only one worker at a time
- File locking issues on network filesystems (NFS)
- Not suitable for parallel tuning

**Example:**
```bash
# From my_project/ directory
# Verify your location first:
pwd  # Should show: .../my_project

python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_baseline_tune
```

**Output:**
```
optimization/
└── ppo_baseline_tune/
    ├── optuna_study.db              # SQLite database
    ├── full_agent_env_config.yaml   # Resolved config snapshot
    ├── tuning_config.yaml
    ├── best_params.json             # After completion
    └── study_summary.json
```

### PostgreSQL (Parallel-Safe)

**When to use:**
- Parallel tuning with multiple workers
- Distributed optimization
- HPC/cluster environments

**Advantages:**
- **Concurrent writes** — multiple workers share one study
- Network-safe (no file locking issues)
- Real-time study visibility across workers

**Requirements:**
- PostgreSQL server running (use `spinup_postgres.py`)
- Environment variables: `PG_USERNAME`, `PG_PORT`, `DB_NAME`

**Setup workflow:**

1. **Start PostgreSQL server** (once, before launching workers):
```bash
export PG_USERNAME="$USER"
export PG_PORT="5432"
export DB_NAME="optuna_tuning"

python -m abx_amr_simulator.training.spinup_postgres
```

2. **Pre-create study** (once, before launching workers):
```bash
python -c "import optuna; optuna.create_study(
    study_name='ppo_parallel_tune',
    storage='postgresql://${PG_USERNAME}@localhost:${PG_PORT}/${DB_NAME}',
    direction='maximize',
    load_if_exists=True)"
```

3. **Launch workers** (multiple terminals/nodes):
```bash
# Worker 1
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_parallel_tune \
  --study-name ppo_parallel_tune \
  --use-postgres

# Worker 2 (same command, different seed)
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_parallel_tune \
  --study-name ppo_parallel_tune \
  --use-postgres \
  -p training.seed=1001

# Worker 3, Worker 4, ...
```

4. **Shutdown PostgreSQL** (after all workers complete):
```bash
python -m abx_amr_simulator.training.shutdown_postgres
```

**Parallel templates:**

The package includes ready-to-use scripts:
- [src/abx_amr_simulator/scripts/run_optuna_parallel_local.sh](../../src/abx_amr_simulator/scripts/run_optuna_parallel_local.sh) — Local multi-worker tuning with task-spooler
- [src/abx_amr_simulator/scripts/run_optuna_parallel.slurm](../../src/abx_amr_simulator/scripts/run_optuna_parallel.slurm) — SLURM cluster tuning

**Key points:**
- All workers must use **same `--study-name`**
- All workers must use **same `--use-postgres`** flag
- Each worker can have different `--run-name` (but usually same for shared outputs)
- `n_trials` in tuning config is **per worker** (total trials = n_workers × n_trials)

### Comparison Table

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| **Setup complexity** | None | Requires server setup |
| **Concurrent workers** | ❌ No | ✅ Yes |
| **Network filesystems** | ⚠️ Risky (file locking) | ✅ Safe |
| **Use case** | Single-worker, local | Parallel, distributed |
| **Database location** | `optimization/<run_name>/optuna_study.db` | PostgreSQL server |
| **Study visibility** | Per-directory | Shared across workers |
| **Recommended for** | Development, testing | Production, HPC |

---

## 4. Running Optimization Studies

### Basic Single-Worker Tuning

```bash
# From my_project/ directory
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_baseline_tune
```

### With Config Overrides

Customize the tuning experiment with subconfig/parameter overrides:

```bash
# From my_project/ directory
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_high_risk_patients_tune \
  -s patient_generator-subconfig=$(pwd)/experiments/configs/patient_generator/high_risk.yaml \
  -p reward_calculator.lambda_weight=0.5
```

---

## 5. Resuming and Managing Studies

### Default Behavior: Resume Existing Studies

By default, `tune.py` **continues** existing studies if the database is found:

```bash
# First run: creates study, runs 50 trials
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_tune_exp1

# [Interrupted after 30 trials]

# Second run: resumes from trial 31 (same command)
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_tune_exp1  # Same run_name = resume
```

**Study persistence:**
- SQLite: `optimization/<run_name>/optuna_study.db` stores all trials
- PostgreSQL: Study persists in database across all workers
- Resuming works seamlessly — Optuna picks up where it left off

### Starting Fresh: Overwrite Existing Studies

To discard old trials and start a new study with the same name:

```bash
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_tune_exp1 \
  --overwrite-existing-study  # Deletes old database, starts fresh
```

**⚠️ WARNING**: `--overwrite-existing-study` is **unsafe with PostgreSQL parallel workers**. If multiple workers race to delete the study, data loss and crashes can occur. For PostgreSQL studies:
1. Delete the study manually **once** before launching workers:
   ```bash
   python -c "import optuna; optuna.delete_study(
       study_name='ppo_parallel_tune',
       storage='postgresql://${PG_USERNAME}@localhost:${PG_PORT}/${DB_NAME}')"
   ```
2. Then launch workers **without** `--overwrite-existing-study`

### Skipping Completed Studies

The optimization registry (`.optimization_completed.txt`) tracks finished runs. Use `--skip-if-exists` to avoid re-running:

```bash
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_tune_exp1 \
  --skip-if-exists  # Skips if ppo_tune_exp1 in registry
```

**Use case:** Running many tuning experiments in a loop/script — completed experiments are skipped automatically.

### Config Validation on Resume

When resuming an existing study, `tune.py` validates that your current configs match the saved configs:

```bash
# First run
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_tune_exp1

# [Later: you modify reward_calculator.lambda_weight in base_experiment.yaml]

# Resume attempt with changed config → FAILS with error:
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name ppo_tune_exp1  # ERROR: Config mismatch detected!
```

**Error message:**
```
❌ ERROR: Configuration Mismatch Detected

The current umbrella config differs from the saved config used in the existing study.
This prevents comparing trials fairly.

Options:
  1) Use a different --run-name to create a new study
  2) Use --overwrite-existing-study to discard old trials and start fresh
```

**Why this matters:** Mixing trials from different configs corrupts the study — results aren't comparable.

---

## 6. Objective Function: Mean-Variance Penalty

Optuna optimizes: **mean(rewards) - λ × std(rewards)**

This balances reward maximization with consistency across seeds. The `stability_penalty_weight` parameter (λ) controls the trade-off:

- **λ = 0.0**: Pure mean optimization (maximize average reward, ignore variance)
- **λ = 0.1-0.3**: Balanced (prefer consistent learning with slight reward trade-off)
- **λ = 0.5+**: Conservative (strongly penalize inconsistent seeds)

**Default values**:
- PPO: `stability_penalty_weight: 0.2` (moderate penalty)
- HRL: `stability_penalty_weight: 0.3` (slightly higher due to option discovery noise)

Each trial logs `mean_reward`, `std_reward`, and `stability_penalty` for debugging.

**Why variance matters:** High variance means unreliable training — small random changes cause large performance swings. Penalizing variance finds hyperparameters that learn consistently.

---

## 7. Train With Best Params

Use the most recent optimization run by experiment name:

```bash
# From my_project/ directory
python -m abx_amr_simulator.training.train \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --load-best-params-by-experiment-name ppo_baseline_tune
```

The best parameters from `optimization/ppo_baseline_tune/best_params.json` are applied as `agent_algorithm.*` overrides during training.

**Output location:**
```
results/
└── ppo_baseline_tune_optimized_<timestamp>/  # Note: "_optimized" suffix
```

---

## 8. Tips and Best Practices

- **Start small**: Use `n_trials: 20-30` and `truncated_episodes: 50` for initial exploration
- **Increase seeds for robustness**: `n_seeds_per_trial: 5-10` gives more stable estimates (but slower)
- **Use `--skip-if-exists`** in scripts/loops to avoid re-running completed studies
- **PostgreSQL for parallel tuning**: Essential for multi-worker optimization (avoid SQLite locking)
- **Match `--study-name` across workers**: All parallel workers must use identical study name
- **Monitor progress**: Check `optimization/<run_name>/optuna_study.db` size — grows with trials
- **Separate tuning and training seeds**: Use different seed ranges (tuning: 1000-1010, training: 1-100)
- **Name studies descriptively**: Include config variants ("ppo_lambda_0.5_tune")

---

## 9. Common Failures and Troubleshooting

### "File not found" errors

**Symptom:**
```
Error: Umbrella config file not found: base_experiment.yaml
Error: --umbrella-config must be an absolute path
```

**Cause:** `--umbrella-config` and `--tuning-config` require absolute paths.

**Solution:**
```bash
# Use $(pwd) to construct absolute paths
python -m abx_amr_simulator.training.tune \
  --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
  --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
  --run-name my_tune
```

### Configuration mismatch on resume

**Symptom:**
```
❌ ERROR: Configuration Mismatch Detected

The current umbrella config differs from the saved config.
```

**Cause:** You modified a config file after starting a study, then tried to resume.

**Solution:**
- **Option 1**: Use a new `--run-name` to start a separate study
- **Option 2**: Use `--overwrite-existing-study` to discard old trials (SQLite only!)
- **Option 3**: Revert your config changes and resume existing study

### SQLite database locked

**Symptom:**
```
sqlite3.OperationalError: database is locked
```

**Cause:** Multiple workers trying to write to SQLite simultaneously, or NFS file locking issues.

**Solution:**
- Use PostgreSQL instead of SQLite for parallel tuning:
  ```bash
  python -m abx_amr_simulator.training.tune \
    --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
    --tuning-config $(pwd)/experiments/tuning_configs/ppo_tuning_default.yaml \
    --run-name my_tune \
    --study-name my_tune \
    --use-postgres
  ```

### PostgreSQL connection refused

**Symptom:**
```
psycopg2.OperationalError: could not connect to server: Connection refused
```

**Cause:** PostgreSQL server not running, or wrong connection details.

**Solution:**
1. Ensure PostgreSQL is running:
   ```bash
   python -m abx_amr_simulator.training.spinup_postgres
   ```
2. Verify environment variables:
   ```bash
   export PG_USERNAME="$USER"
   export PG_PORT="5432"
   export DB_NAME="optuna_tuning"
   ```
3. Check that PostgreSQL is listening on the correct port:
   ```bash
   ps aux | grep postgres  # Should show postgres process with correct port
   ```

### "No optimization runs found"

**Symptom:**
```
Error: No optimization runs found for experiment name 'ppo_tune_exp1'
```

**Cause:** When using `--load-best-params-by-experiment-name` in `train.py`, the optimization directory doesn't contain completed runs.

**Solution:**
- Check `optimization/.optimization_completed.txt` to see registered runs
- Verify `--optimization-dir` matches between `tune.py` and `train.py`
- Ensure tuning completed successfully (look for `best_params.json` in `optimization/<run_name>/`)

### Tuning crashes mid-trial

**Symptom:**
```
WARNING: Training failed for seed 42
Trial returned -inf objective
```

**Cause:** Individual training run within a trial failed (OOM, numerical instability, etc.).

**Impact:** Optuna marks the trial as failed and continues. If all seeds fail, trial gets `-inf` objective.

**Solution:**
- Check trial outputs in temporary results directory
- Reduce `truncated_episodes` to lower memory usage
- Adjust search space bounds (e.g., constrain `learning_rate` to avoid extreme values)
- Increase `n_seeds_per_trial` for robustness (failed seeds won't dominate)

### HRL tuning: "Option library file not found"

**Symptom:**
```
Warning: Option library file not found: /path/to/option_library.yaml
```

**Cause:** HRL tuning requires an option library, but path is incorrect or file missing.

**Solution:**
- Verify option library exists: `cat experiments/options/option_libraries/my_library.yaml`
- Check `hrl.option_library` path in umbrella config is correct
- Scaffold options if needed: `python -m abx_amr_simulator.hrl.setup_options --target-path experiments`

### Parallel workers not sharing trials

**Symptom:** Each worker runs full `n_trials`, not contributing to shared study.

**Cause:** Workers using different `--study-name` or missing `--use-postgres` flag.

**Solution:**
- Ensure all workers use **identical `--study-name`**:
  ```bash
  # Every worker must have this
  --study-name ppo_parallel_tune  # MUST MATCH
  --use-postgres                   # MUST BE PRESENT
  ```
- Verify all workers connect to same PostgreSQL instance (same `PG_PORT`, `DB_NAME`)

### "--overwrite-existing-study is unsafe with --use-postgres"

**Symptom:**
```
ERROR: --overwrite-existing-study is unsafe with --use-postgres
```

**Cause:** Attempting to use `--overwrite-existing-study` with PostgreSQL parallel workers.

**Why it fails:** Multiple workers racing to delete the study causes data corruption.

**Solution:**
Delete the study **once** before launching workers:
```bash
# In wrapper script, before launching workers:
python -c "import optuna; optuna.delete_study(
    study_name='my_study',
    storage='postgresql://${PG_USERNAME}@localhost:${PG_PORT}/${DB_NAME}')"

# Then launch workers WITHOUT --overwrite-existing-study
```

### Best params not loading during training

**Symptom:**
```
Warning: best_params.json not found for experiment 'ppo_tune_exp1'
```

**Cause:** Optimization didn't complete (no `best_params.json` written).

**Solution:**
- Check that tuning ran to completion (look for "Optimization complete" in logs)
- Verify `optimization/<run_name>/best_params.json` exists
- If using custom `--optimization-dir`, ensure `train.py` uses same directory:
  ```bash
  python -m abx_amr_simulator.training.train \
    --umbrella-config $(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml \
    --load-best-params-by-experiment-name ppo_tune_exp1 \
    --optimization-dir $(pwd)/custom_optimization  # Must match tune.py
  ```

### Unknown sampler warning

**Symptom:**
```
Warning: Unknown sampler 'CMAES', using TPE
```

**Cause:** Typo in tuning config `sampler` field, or unsupported sampler name.

**Solution:**
Use one of the supported samplers in `tuning_config.optimization.sampler`:
- `TPE` (Tree-structured Parzen Estimator, default, recommended)
- `Random` (random search)
- `CmaEs` (Covariance Matrix Adaptation Evolution Strategy)

### Missing module errors

**Symptom:**
```
ModuleNotFoundError: No module named 'abx_amr_simulator'
```

**Cause:** Package not installed or wrong Python environment.

**Solution:**
```bash
# Install in editable mode from package root
pip install -e .

# Or verify you're in the correct conda environment
conda activate rlenv
```

---

## 10. What's Next?

✅ You've mastered hyperparameter tuning with Optuna!

**Next steps**:
- **Tutorial 5**: Use the GUI experiment runner and viewer for visual tuning management
- **Tutorial 6**: Apply tuned hyperparameters to HRL experiments
- **Tutorial 10**: Run large-scale experiment sweeps with `experiment_set_runner`
