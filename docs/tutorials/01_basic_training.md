# Tutorial 1: Basic Training Workflow

**Goal**: Train your first RL agent from scratch using the `abx_amr_simulator` package. This tutorial takes you from installation → configuration → training → monitoring → analyzing results in ~30 minutes.

**Prerequisites**: Python 3.10+, pip, basic familiarity with command line

## Important: CLI-First Workflow

The `abx_amr_simulator` package is built in Python, but the primary way you'll interact with it is through **YAML configuration files and command-line commands**. Most experiments are managed by:
- Creating or modifying YAML configs
- Running training via `python -m abx_amr_simulator.training.train` (or running this command via shell scripts)
- Analyzing results via CLI analysis tools

This design prioritizes **reproducibility and modularity**: you can swap out and modify the parameters of different components of your experiment (i.e. changing the environment, reward function, patient population) by pointing to different config files, without touching any of the Python code.

*When do I write Python code?* See Tutorial 4.

---

## Step 1: Install the Package

Install the `abx_amr_simulator` package in editable mode (for development) or standard mode (for usage).

### Development Installation (if modifying code)

```bash
git clone https://github.com/your-repo/abx_amr_capacitor_rl.git
cd abx_amr_capacitor_rl
pip install -e .
```

### Standard Installation

NOTE: This is only going to be true if this package is published to Pypi

```bash
pip install abx-amr-simulator
```

### Verify Installation

```bash
python -c "from abx_amr_simulator import ABXAMREnv; print('✓ Installation successful')"
```

---

## Step 2: Create Your First Experiment Directory

Set up a working directory for your experiments:

```bash
mkdir my_first_project
cd my_first_project
```

Copy the default configuration files using Python:

```bash
python -c "from abx_amr_simulator.utils import setup_config_folders_with_defaults; from pathlib import Path; setup_config_folders_with_defaults(Path('.'))"
```

**Note**: This is one of the few Python commands you'll use directly. For most experiments, your workflow will consist of editing YAML config/subconfig files + CLI commands.

This creates a `configs/` folder with:
- `configs/umbrella_configs/base_experiment.yaml` — Main config that coordinates all component subconfigs
- `configs/environment/default.yaml` — Environment parameters
- `configs/reward_calculator/default.yaml` — Reward function settings
- `configs/patient_generator/default.yaml` — Patient distribution settings
- `configs/agent_algorithm/default.yaml` — PPO hyperparameters

---

## Step 3: Review and Understand the Configuration

Examine the base experiment config (the umbrella config that ties everything together):

```bash
cat configs/umbrella_configs/base_experiment.yaml
```

You should see something like:

```yaml
# Base experiment configuration
environment: ../environment/default.yaml
reward_calculator: ../reward_calculator/default.yaml
patient_generator: ../patient_generator/default.yaml
agent_algorithm: ../agent_algorithm/default.yaml

training:
  run_name: example_run
  total_num_training_episodes: 25
  save_freq_every_n_episodes: 5
  eval_freq_every_n_episodes: 5
  num_eval_episodes: 10
  seed: 42
  log_patient_trajectories: true
```

**Key concepts**:
- The first four lines point to **subconfigs** (component configs) stored in sister folders (`../environment/`, `../reward_calculator/`, etc.)
- All training parameters are **episode-based**, not timestep-based
- `total_num_training_episodes: 25` means 25 complete training episodes (each episode has a fixed length determined by `max_time_steps` in the environment config)
- `eval_freq_every_n_episodes: 5` means evaluate the agent every 5 training episodes
- `num_eval_episodes: 10` means run 10 evaluation episodes each time we evaluate
- `log_patient_trajectories: true` saves full patient-level data during evaluation (useful for analysis)

**Why episodes instead of timesteps?** The environment has no built-in episode termination criteria—episodes only end when `max_time_steps` is reached. Therefore, all episodes are the same length. Using episodes as the unit makes training progress clearer and analysis more straightforward.

**Changing the run name**: The `run_name: example_run` field controls where results are saved. When you run training, results go to `results/example_run_<timestamp>/`. You can override the run name on the command line (see Step 4).

---

## Step 4: Run Training

Train a PPO agent on the default environment:

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml
```

**What happens**:
1. Config is loaded and component configs are merged
2. Environment is created with default parameters
3. PPO agent is initialized
4. Training starts; progress appears on terminal
5. Every 5 training episodes, the agent is evaluated on 10 evaluation episodes
6. Results are saved to `results/example_run_<timestamp>/`

**Expected output** (first few lines):
```
Loading config from configs/umbrella_configs/base_experiment.yaml...
Config loaded successfully.
Creating environment...
Creating reward calculator...
Creating patient generator...
Creating agent (PPO)...
Starting training for 25 episodes...
Episode  1/25 | Eval every 5 episodes...
Episode  5/25 | Evaluation: Mean reward: -12.34
Episode 10/25 | Evaluation: Mean reward: -10.21
Episode 15/25 | Evaluation: Mean reward:   -8.05
```

**Total runtime**: ~2-5 minutes for 25 episodes (varies by CPU)

---

## Step 5: Evaluate Training Progress

After training completes, there are two recommended ways to evaluate your agent's performance:

### Option A: Use Diagnostic and Evaluative Analysis (Recommended)

The package provides specialized analysis tools that are more informative than TensorBoard:

```bash
# Run diagnostic analysis on your completed experiment
python -m abx_amr_simulator.analysis.diagnostic_analysis --prefix example_run_20250115_143614

# Generate evaluative plots (reward decomposition, action analysis)
python -m abx_amr_simulator.analysis.evaluative_plots --prefix example_run_20250115_143614
```

**Note on prefixes**: By default, both tools find the exact run you specify. If you ran multiple seeds and want to aggregate results across them (create ensemble plots with statistics), use the `--aggregate-by-seed` flag:

```bash
# Aggregate results from multiple seeds: example_run_seed1_*, example_run_seed2_*, etc.
python -m abx_amr_simulator.analysis.evaluative_plots --prefix example_run --aggregate-by-seed
```

These tools will generate:
- **Diagnostic plots**: Observation error metrics, reward-observation error correlations
- **Evaluative plots**: Ensemble reward curves with percentile bands, action-attribute associations, policy interpretability metrics

Results are saved to `analysis_output/` with organized subfolders.

**Note on experiment naming**: For analyzing single runs, use the full run name (including timestamp) as the `--prefix` argument:
```bash
python -m abx_amr_simulator.analysis.evaluative_plots --prefix example_run_20250115_143614
```

If you ran **multiple training seeds** and want to aggregate them together, use the naming pattern `<run_name>_seed1`, `<run_name>_seed2`, etc. Then use the base name with the `--aggregate-by-seed` flag:
```bash
# Find and aggregate all: example_run_seed1_<timestamp>, example_run_seed2_<timestamp>, etc.
python -m abx_amr_simulator.analysis.evaluative_plots --prefix example_run --aggregate-by-seed
```

Without the `--aggregate-by-seed` flag set, the tools find single runs matching your prefix exactly.

### Option B: Monitor with TensorBoard (Optional)

If you prefer real-time monitoring during training, open TensorBoard:

```bash
tensorboard --logdir results/example_run_<timestamp>/logs
```

Open your browser to `http://localhost:6006`

**Note**: While TensorBoard provides real-time feedback, the diagnostic and evaluative analysis tools give much deeper insights into agent behavior and are the recommended way to evaluate results.

---

## Step 6: Inspect Results

After training completes, navigate to the results folder:

```bash
ls results/example_run_20250115_143614/
```

You'll see:

```
checkpoints/             # Saved model checkpoints during training
config.yaml              # Full configuration used (for reproducibility)
eval_logs/               # Full patient trajectories from evaluation episodes
figures_best_agent/      # Diagnostic plots for the best model
figures_final_agent/     # Diagnostic plots for the final model
logs/                    # TensorBoard event files
summary.json             # Run metadata (algorithm, episodes, timestamp)
```

The `figures_best_agent/` and `figures_final_agent/` folders contain diagnostic visualizations generated automatically during training.

### Load and Evaluate Your Trained Agent (Optional, Python API)

If you want to programmatically load and evaluate your trained agent using Python, you can write a script. **Most users won't need this—the CLI analysis tools (Step 7) provide everything you need.** If you're curious about the Python API, see Tutorial 4 for complete examples.

For now, proceed to Step 7 to use the recommended CLI analysis tools.

---

## Step 7: Analyze Results with Diagnostic Tools (Recommended)

The results folder already contains diagnostic plots in `figures_best_agent/` and `figures_final_agent/`, which are generated for the single agent trained during that experiment run. For further analysis across experiments and deeper insights (and to aggregate results across seeds, as described in the note above), use the CLI tools:

```bash
# Generate diagnostic analysis (observation error metrics, correlations)
python -m abx_amr_simulator.analysis.diagnostic_analysis --experiment-prefix example_run

# Generate evaluative plots (ensemble analysis, action interpretability)
python -m abx_amr_simulator.analysis.evaluative_plots --experiment-prefix example_run
```

### What These Tools Do

**Diagnostic Analysis** validates your experimental setup:
- Confirms that configured observation noise/bias is actually being applied to patient data
- Measures whether imperfect observations correlate with lower rewards (does the agent suffer from seeing noisy data?)
- Produces two key metrics: observation error statistics and reward-error correlations

**Evaluative Plots** aggregates learning outcomes across all training seeds:
- **Ensemble performance curves** showing mean trajectories with 10-90 percentile bands across seeds
  - AMR evolution, reward components, clinical outcomes, antibiotic prescriptions
- **Action-Attribute Associations** revealing which patient features the agent actually uses when making prescribing decisions
  - High mutual information with `prob_infected` + rising prescribe_rate = sensible behavior
  - Low mutual information everywhere = agent failed to learn meaningful policy

**Why both?** Diagnostic analysis is a "sanity check" that your experiment is set up correctly. Evaluative plots show whether the agent learned a good policy given that setup.

**For complete interpretation guidance**, see [Analysis Tools Documentation](../../src/abx_amr_simulator/analysis/README.md), which explains every plot, metric, and interpretation scenario.

**Note**: These are the developer-recommended evaluation tools and provide significantly more useful insights than TensorBoard for this domain.

---

## Troubleshooting

### "Config file not found"

```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/umbrella_configs/base_experiment.yaml'
```

**Solution**: Make sure you're in the correct directory and ran `setup_config_folders_with_defaults`.

```bash
pwd  # Should be in my_first_project/
ls configs/umbrella_configs/base_experiment.yaml  # Should exist
```

### "No module named 'abx_amr_simulator'"

```
ModuleNotFoundError: No module named 'abx_amr_simulator'
```

**Solution**: Reinstall the package and make sure your Python environment is activated:

```bash
pip install -e .
```

### Training is too slow

**Solution**: Reduce `total_num_training_episodes` in config for quick testing:

```yaml
training:
  total_num_training_episodes: 5  # Instead of 25
```

Note: All training parameters are episode-based (not timestep-based), so reducing episodes directly reduces training time.

### Out of memory

**Solution**: Reduce `num_patients_per_time_step` in environment config:

```yaml
environment:
  num_patients_per_time_step: 5  # Instead of 5; note that the default is set to 1 patient per time step
```

---

## What's Next?

✅ You've trained your first RL agent!

**Next tutorials**:
- **Tutorial 2**: Customize your experiments (change patient distributions, tune reward lambda, configure AMR dynamics) — all via CLI
- **Tutorial 3**: Analyze results with diagnostic and evaluative plots — CLI tools
- **Tutorial 4**: Use the Python API for advanced customization (programmatic analysis, custom components) — ~5% of users need this
- **Tutorial 5**: Use the Streamlit GUI for experiment management

---

## Key Takeaways

1. **Configurations are modular**: Environment, rewards, patients, and algorithm settings are separate YAML files
2. **CLI is the primary interface**: Edit configs, run `python -m abx_amr_simulator.training.train`, analyze with CLI tools
3. **Reproducibility matters**: Seeds are automatically synchronized across components
4. **Python API is optional**: ~95% of workflows use CLI; Python code is for advanced customization (see Tutorial 4)
