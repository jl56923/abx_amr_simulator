# Tutorial 5: GUI Experiment Runner and Viewer

**Goal**: Learn to use the Streamlit GUI applications for lightweight experiment configuration, launching training runs, and browsing results.

**Prerequisites**: Completed Tutorial 1 (Basic Training Workflow) and Tutorial 2 (Config Scaffolding)

**Scope**: The GUI is designed for **lightweight workflows only**. For HRL, optimization, or advanced customization, use the CLI as documented in other tutorials.

---

## Overview

The `abx_amr_simulator` package includes two Streamlit GUI applications:

1. **Experiment Runner** — Interactive interface for configuring basic training runs
2. **Experiment Viewer** — Browse completed experiments and view configs

Both apps treat your current working directory as the project root and look for
configs in `./experiments/configs/`.

**Important**: The GUI does **not** support HRL or Optuna workflows. Read through the tutorials on using the CLI for those workflows.

---

## Step 1: Set Up Your Project

The GUI expects you to run it from your project root (e.g., `myproject/`) where you have:

```
myproject/
├── experiments/
│   ├── configs/
│   │   ├── umbrella_configs/
│   │   │   └── base_experiment.yaml
│   │   ├── environment/
│   │   ├── patient_generator/
│   │   ├── reward_calculator/
│   │   └── agent_algorithm/
│   └── options/     # Optional (HRL only, not GUI-supported)
├── results/         # Created automatically if missing
```

If you haven't set up configs yet, run from `myproject/`:

```bash
python -c "from abx_amr_simulator.utils import setup_config_folders_with_defaults; setup_config_folders_with_defaults('experiments')"
```

This creates `experiments/configs/` with default subconfig files and umbrella configs.

---

## Step 2: Launch the Experiment Runner

### From your project directory:

```bash
cd myproject
abx-amr-simulator-experiment-runner
```

This launches the app on `http://localhost:8501` and uses your current directory as the project root:
- Results saved to `./results/`
- Configs loaded from `./experiments/configs/umbrella_configs/`
- Working directory printed on startup for verification

**Custom results directory** (optional):

```bash
abx-amr-simulator-experiment-runner --results-dir /path/to/custom/results
```

---

## Step 3: Using the Experiment Runner

The Experiment Runner provides a simplified interface for basic training workflows:

### 1. Config Selection (Sidebar)
- **Base config**: Select umbrella config from `experiments/configs/umbrella_configs/` (defaults to `base_experiment.yaml`)
- **Results directory**: Shown for reference (defaults to `./results/`)
- **Run name prefix**: Give your experiment a descriptive name

### 2. Environment Settings
- **Core environment**: Number of patients, max timesteps, AMR observation settings
- **Observable patient attributes**: Choose which patient features the agent can observe (minimum: `prob_infected`)
- **Antibiotics & Crossresistance**: Configure antibiotic parameters (leak rate, AMR dynamics, crossresistance)

### 3. Reward Calculator
- **Lambda weight**: Balance between AMR penalty (community) and clinical benefit (individual patients)
- **Per-antibiotic rewards**: Clinical benefit, failure penalty, adverse effect penalty

### 4. Patient Generator
- **Observable attributes**: Select which patient attributes to include in observations
- **Per-attribute configuration**: Set distribution (constant/gaussian), observation noise/bias, clipping bounds

### 5. Agent Algorithm
- **Algorithm**: PPO, A2C
- **Network architecture**: MLP policy with configurable hidden layers
- **Hyperparameters**: Learning rate, batch size, GAE lambda, etc.

### 6. Training Settings (Sidebar)
- **Total episodes**: How many episodes to train
- **Save/eval frequency**: How often to checkpoint and evaluate
- **Seed**: Random seed for reproducibility

### 7. Launch Training
Click **"Start Training Run"** to:
- Generate umbrella config with GUI settings
- Launch training via `python -m abx_amr_simulator.training.train`
- Stream terminal output in the GUI
- Save results to `results/<run_name>_<timestamp>/`

**Continue Training**: Check the sidebar option to resume from a prior experiment (adds more episodes to an existing run).

### What the GUI Does NOT Support
- **HRL workflows**: Use [Tutorial 06](06_hrl_quickstart.md) for option-based training
- **Optuna optimization**: Use [Tutorial 04](04_optimization.md) for hyperparameter tuning
- **Advanced parameter overrides**: Use CLI with `-p` flags for complex customization
- **Batch sweeps**: Use CLI workflows for scripted parameter sweeps

---

## Step 4: Launch the Experiment Viewer

### From your project directory:

```bash
cd myproject
abx-amr-simulator-experiment-viewer
```

This launches the app on `http://localhost:8502` (different port than the runner, so you can run both simultaneously).

**Custom results directory** (optional):

```bash
abx-amr-simulator-experiment-viewer --results-dir /path/to/custom/results
```

---

## Step 5: Using the Experiment Viewer

The Experiment Viewer provides a read-only interface to browse completed experiments:

### Interface Overview
1. **Results directory**: Displayed at top (defaults to `./results/`)
2. **Experiments list**: All experiment folders with timestamps
3. **Config inspector**: View the `full_agent_env_config.yaml` used for each run
4. **Basic metadata**: Run name, timestamp, completion status

### Workflow
1. Launch the viewer from your project directory
2. Select an experiment from the dropdown
3. View its configuration in YAML format
4. Compare configs across multiple runs

**Note**: The viewer shows basic experiment information only. For advanced analysis (diagnostic plots, evaluation metrics), use the CLI analysis tools documented in other tutorials.

---

## Step 6: Running Both Apps Simultaneously

Launch both apps in separate terminals from your project directory:

```bash
# Terminal 1 (Experiment Runner)
cd myproject
abx-amr-simulator-experiment-runner

# Terminal 2 (Experiment Viewer)
cd myproject
abx-amr-simulator-experiment-viewer
```

**Or use the dual launcher**:

```bash
cd myproject
abx-amr-simulator-launch-gui
```

This starts both apps and opens them in separate browser tabs (runner on port 8501, viewer on port 8502).

---

## GUI vs CLI: When to Use Each

### Use the GUI when:
- You want interactive exploration of basic config options
- You're new to the package and learning the config structure
- You need quick visual feedback on parameter choices
- You want to browse and compare completed experiments quickly

### Use the CLI when:
- **HRL workflows**: Option libraries, PPO/RPPO training, diagnostics
- **Hyperparameter optimization**: Optuna sweeps with PostgreSQL backend
- **Batch parameter sweeps**: Experiment sets with JSON-defined grids
- **Remote execution**: SSH/cluster jobs without GUI access
- **Reproducible workflows**: Version-controlled config files
- **Advanced customization**: Complex parameter overrides, subclassing

**Recommendation**: Use the GUI for initial learning and lightweight experiments, then transition to CLI for production workflows.

---

## Troubleshooting

### "Address already in use" error

**Cause**: Another Streamlit app is running on the same port.

**Solution**:
1. Stop the other app (Ctrl+C in its terminal)
2. Or use a different port: `abx-amr-simulator-experiment-runner --results-dir . & sleep 2 && streamlit run ... --server.port 8503`

### "Working directory: /wrong/path"

**Cause**: You launched the GUI from the wrong directory.

**Solution**: Always launch from your project root (e.g., `cd myproject && abx-amr-simulator-experiment-runner`)

### GUI doesn't show my configs/experiments

**Cause**: Missing `experiments/configs/umbrella_configs/` or `results/` directories.

**Solution**: Run `setup_config_folders_with_defaults('experiments')` from your project root (see Step 1).

### Training command fails with "Config not found"

**Cause**: GUI-generated config uses relative paths, but your project structure doesn't match expectations.

**Solution**: Ensure you have `experiments/configs/` with component subfolders, and `results/` at project root. The GUI reads umbrella configs from `experiments/configs/umbrella_configs/`.

---

## What's Next?

✅ You've learned to use the GUI for lightweight workflows!

**Next tutorials**:
- **Tutorial 06**: [HRL Quick Start](06_hrl_quickstart.md) — Train hierarchical RL agents with option libraries (CLI-only)
- **Tutorial 04**: [Optimization](04_optimization.md) — Hyperparameter tuning with Optuna (CLI-only)
- **Tutorial 10**: [Advanced Heuristic Worker Subclassing](10_advanced_heuristic_worker_subclassing.md) — Extend heuristic options (CLI-only)

---

## Key Takeaways

1. **Lightweight scope**: GUI supports basic PPO/A2C training only—no HRL, no optimization
2. **Working directory**: Always launch from project root (`myproject/`) where `experiments/configs/` and `results/` live
3. **Two apps**: Experiment Runner (configure/launch) and Experiment Viewer (browse)
4. **Dual launcher**: `abx-amr-simulator-launch-gui` starts both apps simultaneously
5. **Different ports**: Runner on 8501, Viewer on 8502
6. **CLI for advanced features**: HRL, Optuna, experiment sets, complex overrides
