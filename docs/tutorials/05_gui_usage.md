# Tutorial 5: GUI Experiment Runner and Viewer

**Goal**: Learn to use the Streamlit GUI applications for configuring experiments, launching training runs, and analyzing results.

**Prerequisites**: Completed Tutorial 1 (Basic Training Workflow) and Tutorial 2 (Config Scaffolding)

---

## Overview

The `abx_amr_simulator` package includes two Streamlit GUI applications:

1. **Experiment Runner** — Interactive interface for configuring and launching training runs
2. **Experiment Viewer** — Browse completed experiments, view configs, and explore diagnostic plots

Both apps are accessible via console entry points or direct Streamlit commands.

---

## Step 1: Launch the Experiment Runner

### Option A: Console Entry Point (Recommended)

```bash
abx-amr-simulator-experiment-runner
```

This launches the app on `http://localhost:8501`.

### Option B: Direct Streamlit Command

```bash
cd /path/to/your/workspace
streamlit run /path/to/abx_amr_simulator/src/abx_amr_simulator/gui/experiment_runner.py
```

**Note**: The console entry point is more convenient and doesn't require knowing the package installation path.

---

## Step 2: Using the Experiment Runner

### Interface Overview

The Experiment Runner has several sections:

1. **Project Setup** — Select your workspace directory (containing `experiments/`, `results/`, etc.)
2. **Config Selection** — Choose umbrella config and component subconfigs
3. **Parameter Overrides** — Modify config values via interactive widgets
4. **Training Settings** — Set run name, seed, total episodes, eval frequency
5. **Launch Button** — Start training with a single click

### Workflow Example

1. **Set workspace directory**:
   - Click "Browse" in the Project Setup section
   - Select your `my_project/` directory

2. **Choose configs**:
   - **Umbrella Config**: Select from `experiments/configs/umbrella_configs/`
   - **Component Subconfigs**: Optionally override default subconfigs

3. **Override parameters** (optional):
   - Click "Add Parameter Override"
   - Enter key (e.g., `reward_calculator.lambda_weight`) and value (e.g., `0.8`)

4. **Configure training**:
   - **Run Name**: Enter a descriptive name (e.g., `lambda_sweep_clinical`)
   - **Seed**: Set random seed for reproducibility
   - **Total Episodes**: Set training length
   - **Eval Frequency**: How often to evaluate the agent

5. **Launch training**:
   - Click "Launch Training"
   - The app runs training in a background process
   - Monitor progress in the terminal output section

### Advanced Features

- **Load Best Params**: Load tuned hyperparameters from prior Optuna runs
- **Skip If Exists**: Avoid overwriting completed experiments
- **Train from Prior Results**: Continue training from a checkpoint

---

## Step 3: Launch the Experiment Viewer

### Option A: Console Entry Point (Recommended)

```bash
abx-amr-simulator-experiment-viewer
```

This launches the app on `http://localhost:8502` (different port than the runner).

### Option B: Direct Streamlit Command

```bash
streamlit run /path/to/abx_amr_simulator/src/abx_amr_simulator/gui/experiment_viewer.py --server.port 8502
```

**Why port 8502?** This lets you run both apps simultaneously—configure in the runner while viewing results in the viewer.

---

## Step 4: Using the Experiment Viewer

### Interface Overview

The Experiment Viewer has several tabs:

1. **Experiments List** — Browse all completed experiments
2. **Config Inspector** — View full configuration used for each run
3. **Diagnostic Plots** — Explore training metrics and environment behavior
4. **Evaluation Plots** — View policy performance and action analysis

### Workflow Example

1. **Select results directory**:
   - Click "Browse" and select `my_project/results/`

2. **Browse experiments**:
   - The app lists all experiment folders (with timestamps)
   - Click on an experiment to view details

3. **Inspect configuration**:
   - Switch to "Config Inspector" tab
   - View the full `config.yaml` used for that run
   - See all parameter values in structured format

4. **View diagnostic plots**:
   - Switch to "Diagnostic Plots" tab
   - Explore:
     - Observation error distributions
     - Reward-error correlations
     - AMR dynamics over time

5. **View evaluation plots**:
   - Switch to "Evaluation Plots" tab
   - Explore:
     - Performance curves (rewards, AMR levels, prescribing rates)
     - Action-attribute associations (which patient features drive actions)

---

## Step 5: Running Both Apps Simultaneously

You can run both the Experiment Runner and Viewer at the same time:

```bash
# Terminal 1
abx-amr-simulator-experiment-runner

# Terminal 2
abx-amr-simulator-experiment-viewer
```

This lets you:
- Launch new training runs in the Runner (port 8501)
- Monitor completed runs in the Viewer (port 8502)
- Switch between tabs in your browser

---

## GUI vs CLI: When to Use Each

### Use the GUI when:
- You want interactive exploration of configuration options
- You're new to the package and want guided setup
- You prefer visual feedback and point-and-click workflows
- You want to quickly browse and compare completed experiments

### Use the CLI when:
- You need to run large parameter sweeps (shell scripts)
- You want reproducible workflows (version-controlled configs)
- You're running experiments on remote servers (no GUI)
- You need fine-grained control over all parameters

**Recommendation**: Use the GUI for learning and exploration, then transition to CLI for production workflows.

---

## Troubleshooting

### "Address already in use" error

**Solution**: Another Streamlit app is running on the same port. Either:
1. Stop the other app
2. Use a different port: `streamlit run ... --server.port 8503`

### GUI doesn't show my experiments

**Solution**: Make sure you've selected the correct results directory. The Viewer looks for folders inside `results/` with timestamps in their names.

### Parameter override doesn't work

**Solution**: Check that you're using the correct dot notation for nested config keys. Example:
- ✓ Correct: `reward_calculator.lambda_weight`
- ✗ Incorrect: `lambda_weight` or `reward_calculator/lambda_weight`

---

## What's Next?

✅ You've learned to use the GUI!

**Next tutorials**:
- **Tutorial 6**: Train hierarchical RL agents with option libraries
- **Tutorial 10**: Run experiment sets with JSON-defined parameter sweeps

---

## Key Takeaways

1. **Two GUI apps**: Experiment Runner (config/launch) and Experiment Viewer (browse/analyze)
2. **Different ports**: Runner on 8501, Viewer on 8502 (run simultaneously)
3. **Console entry points**: `abx-amr-simulator-experiment-runner` and `abx-amr-simulator-experiment-viewer`
4. **Complementary to CLI**: GUI for exploration, CLI for production workflows
