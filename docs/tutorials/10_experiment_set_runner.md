# Experiment Set Runner

The **experiment set runner** is a universal shell script that orchestrates multiple experiments defined in a JSON configuration file. It handles:
- Running tuning (Optuna hyperparameter optimization) per experiment
- Executing training runs across all specified seeds
- Optionally expanding a training grid (cartesian product of parameter sweeps)
- Reusing tuned hyperparameters across multiple training configurations

This tutorial shows how to structure your experiment sets and run them.

---

## JSON Structure

All experiment set runners use the same JSON schema with three top-level keys:

```json
{
  "schema": "1.1",
  "experiment_set_info": {
    "description": "...",
    "defaults": { ... }
  },
  "experiments": [ ... ]
}
```

### `experiment_set_info` (Metadata & Defaults)

The `experiment_set_info` object contains a description and a `defaults` object. Defaults provide fallback values for all experiments:

```json
"experiment_set_info": {
  "description": "Experiment Set 1: perfect observability.",
  "defaults": {
    "umbrella_config": "base_experiment.yaml",
    "training_seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "parameter_overrides": {
      "environment.num_patients_per_time_step": 1,
      "reward_calculator.epsilon": 0.0
    },
    "tuning": {
      "optimization_subdir": "optimization",
      "skip_if_exists": true,
      "mode": "per_experiment"
    },
    "training": {
      "total_num_training_episodes": 200,
      "skip_if_exists": true
    }
  }
}
```

**Key defaults:**
- **`umbrella_config`**: Base configuration file (e.g., `base_experiment.yaml` or `hrl_base_experiment.yaml`)
- **`training_seeds`**: List of seeds for training run reproducibility
- **`parameter_overrides`**: Global parameter overrides (applied to all experiments, overridable per-experiment)
- **`tuning`**: Tuning configuration with:
  - **`mode`**: `"per_experiment"` (tune once per experiment) or `"grid_reference"` (tune once, reuse across grid)
  - **`skip_if_exists`**: Skip if optimization already completed
- **`training`**: Training settings (total episodes, skip behavior)

### `experiments` Array

Each experiment has:

```json
{
  "id": "1a",
  "notes": "Single antibiotic baseline with flat PPO agent",
  "umbrella_config": "base_experiment.yaml",
  "environment_subconfig": "single_abx_environment.yaml",
  "reward_calculator_subconfig": "single_abx_reward_calculator.yaml",
  "patient_generator_subconfig": "baseline_homogeneous_patient_generator.yaml",
  "agent_algorithm_subconfig": "ppo.yaml",
  "options_library_config": null,
  "tuning": {
    "enabled": true,
    "tuning_config": "ppo_tuning.yaml",
    "overrides": {}
  },
  "training": {
    "overrides": {
      "total_num_training_episodes": 150
    }
  }
}
```

**Fields:**
- **`id`**: Unique experiment identifier (e.g., `1a`, `2c`). Used for selective runs.
- **`notes`**: Human-readable description of the experiment.
- **`*_subconfig`**: Configuration file names for environment, reward calculator, patient generator, and agent algorithm.
- **`options_library_config`**: Path to HRL option library (if applicable); `null` for non-HRL agents.
- **`parameter_overrides`**: Per-experiment parameter overrides (extend/override defaults). Any config key can be overridden using dot notation (e.g., `"environment.num_patients_per_time_step"`, `"environment.update_visible_AMR_levels_every_n_timesteps"`).
- **`tuning.enabled`**: Whether to run Optuna tuning for this experiment.
- **`tuning.tuning_config`**: Tuning hyperparameter config (e.g., `ppo_tuning.yaml`).
- **`tuning.overrides`**: Per-tuning-run overrides (e.g., `{"agent_algorithm.ppo.ent_coef": 0.01}`).
- **`training.overrides`**: Per-training-run overrides (e.g., different episode counts).

### Parameter Overrides Pattern

This runner uses a **unified parameter override system** where all config parameters are set via `parameter_overrides`. This ensures consistency and transparency:

**Default overrides** (applied to all experiments):

```json
"defaults": {
  "parameter_overrides": {
    "environment.num_patients_per_time_step": 1,
    "reward_calculator.epsilon": 0.0
  }
}
```

**Per-experiment overrides** (extend/override defaults):

```json
{
  "id": "1b",
  "notes": "Single antibiotic with different patient count",
  "parameter_overrides": {
    "environment.num_patients_per_time_step": 5
  }
}
```

The runner merges them: experiment overrides take precedence. The merged result is passed to both tuning and training via `-p key=value` flags.

**Benefits:**
- All parameters use the same mechanism (no special-case keys)
- Full config paths are explicit and discoverable
- Easy to override any config parameter
- Consistent with CLI override patterns

---

## Optional: Training Grid (Parameter Sweep)

When you want to sweep over multiple environment configurations (e.g., different AMR delay levels, noise, bias), add a `training_grid` to defaults:

```json
"defaults": {
  "training_grid": {
    "name_joiner": "_",
    "params": [
      {
        "key": "environment.update_visible_AMR_levels_every_n_timesteps",
        "values": [7, 30, 90],
        "name_token": "d",
        "format": "int"
      },
      {
        "key": "environment.add_noise_to_visible_AMR_levels",
        "values": [0.0, 0.2],
        "name_token": "n",
        "format": "float2"
      },
      {
        "key": "environment.add_bias_to_visible_AMR_levels",
        "values": [-0.2, 0.0, 0.2],
        "name_token": "b",
        "format": "float2"
      }
    ]
  }
}
```

The runner will:
1. Expand the grid into all combinations (e.g., 3 × 2 × 3 = 18 combinations)
2. For each experiment and each grid combination, run all seeds
3. Append a suffix to run names using the `name_token` and formatted values (e.g., `d7_n0.00_b-0.20`)

**Grid parameter fields:**
- **`key`**: Full config key to override (e.g., `environment.update_visible_AMR_levels_every_n_timesteps`)
- **`values`**: List of values to sweep
- **`name_token`**: Short identifier for the run name suffix (e.g., `d` for delay)
- **`format`**: Value formatter: `int`, `float2`, `float3`, `str`, `bool`

### Tuning Reuse with Grid

When using a training grid with `tuning.mode = "grid_reference"`, the runner:
1. Selects one representative grid combination for tuning (e.g., the "middle" config with delay=30, noise=0.0, bias=0.0)
2. Runs Optuna tuning once on that configuration
3. Reuses the best hyperparameters for **all** training runs across the entire grid

This avoids expensive per-configuration tuning while ensuring hyperparameters are optimized for a representative scenario.

**Configure the selector:**

```json
"tuning": {
  "mode": "grid_reference",
  "grid_selector": {
    "strategy": "match",
    "values": {
      "environment.update_visible_AMR_levels_every_n_timesteps": 30,
      "environment.add_noise_to_visible_AMR_levels": 0.0,
      "environment.add_bias_to_visible_AMR_levels": 0.0
    }
  }
}
```

Alternative strategies:
- **`"first"`** (default): Tune on the first grid combination
- **`"index"`**: Tune on a specific combo by index: `{"strategy": "index", "index": 5}`
- **`"match"`**: Tune on a specific combo by value match (as shown above)

---

## Tuning Strategies

The experiment runner supports **three distinct tuning strategies** that offer different trade-offs between optimization cost and experimental flexibility.

### Strategy 1: Per-Experiment Tuning (1:1 Pairing)

**When to use:** 
- Each experiment has unique hyperparameter needs
- You want fine-grained control over each experiment's tuning
- Configuration variations (environment, reward, agent) are significant

**How it works:**
- Each experiment runs its own Optuna tuning study
- Best hyperparameters from that study are used for all training runs
- Complete independence between experiments

**Configuration:**

```json
"defaults": {
  "tuning": {
    "mode": "per_experiment",
    "skip_if_exists": true
  }
}
```

Each experiment specifies its own tuning config:

```json
{
  "id": "1a",
  "notes": "Single antibiotic baseline",
  "tuning": {
    "enabled": true,
    "tuning_config": "ppo_tuning.yaml",
    "overrides": {}
  }
}
```

**Cost:** If you have 12 experiments, you run 12 Optuna studies.

**Example:** Set 1 (perfect observability) where each variant needs its own tuning.

---

### Strategy 2: Grid Reference Tuning (One Tuning + Many Training Configs)

**When to use:**
- You're sweeping over multiple training parameter combinations (via `training_grid`)
- A single representative configuration captures the essence of the hyperparameter space
- You want to avoid running Optuna for every grid combination

**How it works:**
1. Identifies a representative grid combination (e.g., middle config for delay/noise/bias)
2. Runs Optuna tuning once on that representative configuration
3. Reuses the best hyperparameters for all training runs across the entire grid
4. Each grid combo still gets its own training runs, but they all use shared hyperparameters

**Configuration:**

```json
"defaults": {
  "training_grid": {
    "params": [
      {"key": "environment.update_visible_AMR_levels_every_n_timesteps", "values": [7, 30, 90], "name_token": "d", "format": "int"},
      {"key": "environment.add_noise_to_visible_AMR_levels", "values": [0.0, 0.2], "name_token": "n", "format": "float2"}
    ]
  },
  "tuning": {
    "mode": "grid_reference",
    "grid_selector": {
      "strategy": "match",
      "values": {
        "environment.update_visible_AMR_levels_every_n_timesteps": 30,
        "environment.add_noise_to_visible_AMR_levels": 0.0
      }
    },
    "skip_if_exists": true
  }
}
```

Each experiment still declares tuning:

```json
{
  "id": "2a",
  "notes": "Two antibiotics with AMR observation delay/noise grid",
  "tuning": {
    "enabled": true,
    "tuning_config": "ppo_tuning.yaml"
  }
}
```

**Grid selector strategies:**
- **`"first"`** (default): Tune on the first grid combination
- **`"index"`**: Tune on combo by index: `{"strategy": "index", "index": 5}`
- **`"match"`**: Tune on combo by value match (shown above)

**Cost:** With a 3×2 grid (6 combos) and 6 experiments: 1 tuning + (6 experiments × 6 combos × 20 seeds) training runs.

**Example:** Set 2 where you sweep AMR feedback delay/noise but only optimize hyperparameters for a representative (delay=30, noise=0.0) config.

---

### Strategy 3: Explicit Experiment ID Reuse (Tuning Sharing Pool)

**When to use:**
- Multiple experiments share the same agent algorithm and tuning config (but differ in environment/reward/patient populations)
- You want to group experiments by "antibiotic scenario" and tune once per group
- You need clarity about which experiments reuse which tuning results

**How it works:**
1. Designate representative experiments (e.g., 3a, 3e, 3i) as the "tuning leaders" for each group
2. These leaders run Optuna tuning
3. Other experiments (e.g., 3b–3d) explicitly reference their group's leader via `reuse_from_experiment_id`
4. The runner loads tuning results from the leader, skipping Optuna for followers

**Configuration:**

Designate tuning leaders (enable tuning):

```json
{
  "id": "3a",
  "notes": "Single ABX baseline - tuning leader for single ABX group",
  "environment_subconfig": "single_abx_environment.yaml",
  "agent_algorithm_subconfig": "hrl_ppo.yaml",
  "tuning": {
    "enabled": true,
    "tuning_config": "hrl_ppo_tuning.yaml"
  }
},
{
  "id": "3e",
  "notes": "Two ABX no CR illusory - tuning leader for two ABX no CR group",
  "environment_subconfig": "two_abx_no_crossresistance_environment.yaml",
  "agent_algorithm_subconfig": "hrl_ppo.yaml",
  "tuning": {
    "enabled": true,
    "tuning_config": "hrl_ppo_tuning.yaml"
  }
}
```

Followers reference their leader:

```json
{
  "id": "3b",
  "notes": "Single ABX true risk - reuses tuning from 3a",
  "environment_subconfig": "single_abx_environment.yaml",
  "agent_algorithm_subconfig": "hrl_ppo.yaml",
  "tuning": {
    "reuse_from_experiment_id": "3a"
  }
},
{
  "id": "3c",
  "notes": "Single ABX exaggerated risk - reuses tuning from 3a",
  "environment_subconfig": "single_abx_environment.yaml",
  "agent_algorithm_subconfig": "hrl_ppo.yaml",
  "tuning": {
    "reuse_from_experiment_id": "3a"
  }
}
```

**Cost:** With 12 experiments in 3 groups (3a/3e/3i as leaders, rest as followers): 3 tuning studies + (12 experiments × 20 seeds) training.

**Example:** Set 3 where you have 12 HRL experiments grouped by antibiotic scenario (single ABX, two ABX no CR, two ABX with CR). Tune once per scenario, reuse across all variants within that scenario.

---

## Comparison Table

| Strategy | Best For | Tuning Runs | Flexibility | Clarity |
|----------|----------|------------|-------------|---------|
| **Per-Experiment** | Diverse configs per experiment | High (one per experiment) | Very high | Very clear |
| **Grid Reference** | Parameter sweeps with fixed agent | Low (one per experiment) | Medium (grid only) | Clear but needs selector |
| **Experiment ID Reuse** | Grouped experiments with shared agent | Low (one per group) | High (any grouping) | Highest (explicit mapping) |

---

## Running Experiments

### Basic Usage

Run all experiments from the default JSON:

```bash
bash workspace/experiments/shell_scripts/experiment_set_runner.sh
```

### Specify Experiment JSON

```bash
EXP_JSON_PATH=workspace/experiments/shell_scripts/revised_set1_experiments.json \
  bash workspace/experiments/shell_scripts/experiment_set_runner.sh
```

### Select Specific Experiments

Run only experiments with IDs `1a` and `1c`:

```bash
EXPERIMENT_IDS=1a,1c bash workspace/experiments/shell_scripts/experiment_set_runner.sh
```

Run a single experiment:

```bash
EXPERIMENT_ID=2c bash workspace/experiments/shell_scripts/experiment_set_runner.sh
```

### Select Specific Grid Combinations

When a training grid is defined, you can target specific grid combinations:

```bash
GRID_SELECTOR='{"strategy":"match","values":{"environment.update_visible_AMR_levels_every_n_timesteps":30,"environment.add_noise_to_visible_AMR_levels":0.0,"environment.add_bias_to_visible_AMR_levels":0.0}}' \
  bash workspace/experiments/shell_scripts/experiment_set_runner.sh
```

---

## How It Works (High Level)

The runner:

1. **Parses the JSON** to extract defaults, experiments, and optional training grid
2. **For each selected experiment:**
   - If tuning is enabled, runs Optuna: `training/tune.py` with the tuning config
   - Stores best hyperparameters in an optimization registry
3. **Expands the training grid** (if present) into all combinations
4. **For each grid combination and seed:**
   - Runs training: `training/train.py`
   - Loads best hyperparameters from the tuning run (if applicable)
   - Applies grid parameter overrides
   - Generates a unique run name with grid suffix and seed

**Run names** follow the pattern:
```
exp_{experiment_id}__{env_abbrev}_{rc_abbrev}_{pg_abbrev}_{ag_abbrev}[_{grid_suffix}]_seed{seed}
```

Example: `exp_2c__SA_SA_BH_PPO_d30_n0.00_b0.00_seed1`

---

## Cluster Execution (SLURM)

To run a single experiment on a cluster node with a 18-hour wall-clock limit:

```bash
#!/bin/bash
#SBATCH --job-name=exp_1a
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /path/to/workspace
EXPERIMENT_ID=1a bash workspace/experiments/shell_scripts/experiment_set_runner.sh
```

Submit one job per experiment:

```bash
for exp_id in 1a 1b 1c 1d 1e; do
  sbatch --job-name="exp_$exp_id" \
         --wrap "EXPERIMENT_ID=$exp_id bash workspace/experiments/shell_scripts/experiment_set_runner.sh"
done
```

---

## Heuristic Baselines Runner

The **heuristic baselines runner** (`heuristic_baselines_runner.sh`) is a companion script that evaluates heuristic (rule-based) policies using the same experiment configurations as the RL training runner.

### Purpose

Heuristic policies provide interpretable baselines that do not learn from data. Running them alongside RL agents helps:
- **Validate environment**: Confirm that simple strategies achieve expected results
- **Benchmark learning**: Measure how much RL improves over interpretable rules
- **Diagnose issues**: If RL underperforms heuristics, inspect reward/observation design

### Heuristic Policies Available

1. **`threshold`**: Prescribes an antibiotic if patient infection probability exceeds a fixed threshold
2. **`escalation`**: Starts with narrow-spectrum antibiotics, escalates if patient remains infected
3. **`risk_stratified`**: Uses different thresholds per patient based on observed risk multipliers

### JSON Configuration

Enable heuristic evaluation by adding a `heuristics` section to the experiment set JSON:

```json
"experiment_set_info": {
  "defaults": {
    "heuristics": {
      "enabled": true,
      "policies": ["threshold", "escalation", "risk_stratified"],
      "eval_seeds": [1, 2, 3],
      "skip_if_exists": true
    }
  }
}
```

**Fields:**
- **`enabled`**: Set to `true` to enable heuristic evaluation for all experiments (can override per-experiment)
- **`policies`**: List of heuristic policy names to evaluate
- **`eval_seeds`**: Seeds for evaluation reproducibility (separate from training seeds)
- **`skip_if_exists`**: Skip evaluation if results already exist

Per-experiment override example:

```json
{
  "id": "4a",
  "heuristics": {
    "enabled": false
  }
}
```

### Running Heuristic Baselines

Run all experiments with heuristics enabled:

```bash
bash workspace/experiments/shell_scripts/heuristic_baselines_runner.sh
```

Specify experiment JSON:

```bash
EXP_JSON_PATH=workspace/experiments/shell_scripts/set4_experiments.json \
  bash workspace/experiments/shell_scripts/heuristic_baselines_runner.sh
```

Run specific experiments:

```bash
EXPERIMENT_IDS=4a,4b bash workspace/experiments/shell_scripts/heuristic_baselines_runner.sh
```

### How It Works

The heuristic baselines runner:
1. **Reads the same experiment JSON** as the RL training runner
2. **Reuses all environment, reward, and patient generator configs**
3. **Extracts `num_eval_episodes`** from the umbrella config YAML file (no hardcoded values)
4. **Merges parameter overrides** identically to the RL runner (defaults + per-experiment)
5. **Calls `train_w_heuristic_policies.py`** for each enabled heuristic policy
6. **Saves results** to the same directory structure as RL training runs

**Key Design Principle:** Heuristics use the same evaluation episode count as RL agents (extracted from `num_eval_episodes` in the umbrella YAML). This ensures fair comparison—both learned and rule-based policies are evaluated under identical conditions.

### Output Structure

Results saved to:
```
results/{experiment_id}/heuristics/{policy_name}_seed{seed}/
```

Example:
```
results/4a/heuristics/threshold_seed1/
results/4a/heuristics/escalation_seed2/
```

---

## Error Handling

The runner **fails loudly** with clear error messages if:
- **No experiments matched** the selection criteria (e.g., invalid `EXPERIMENT_ID`)
- **No grid combinations matched** the selector (e.g., typo in `GRID_SELECTOR`)
- **Required config files are missing** (e.g., missing subconfig or tuning config)
- **Abbreviations not found** in `abbreviations.json`

Example error:
```
Error: No experiments matched selection criteria.
```

If you see this, check:
1. Is the `EXPERIMENT_ID` correct? (Use `EXPERIMENT_IDS=invalid_id` to see available IDs)
2. Is the JSON path correct? (Check `EXP_JSON_PATH`)
3. Are all config files present? (Check `experiments/configs/` subdirectories)

---

## Best Practices

1. **Use meaningful experiment IDs** (e.g., `1a` for Set 1 experiment A)
2. **Add descriptive `notes`** for each experiment (helps with reproducibility)
3. **Test with a single seed first** before running all 20 seeds:
   ```bash
   # Modify defaults in JSON: "training_seeds": [1]
   bash workspace/experiments/shell_scripts/experiment_set_runner.sh
   ```
4. **Use `skip_if_exists: true` in defaults** to avoid re-running completed jobs (especially tuning)
5. **Validate your JSON** before submitting large runs:
   ```bash
   jq . workspace/experiments/shell_scripts/revised_set1_experiments.json
   ```