# ABX AMR Simulator

A reinforcement learning simulator for optimizing antibiotic prescribing under antimicrobial resistance (AMR) constraints.

## Installation

### From source (editable install)

```bash
git clone <repo-url>
cd abx_amr_simulator
pip install -e .
```

This installs the `abx_amr_simulator` package with all dependencies (numpy, pyyaml, gymnasium, stable-baselines3, tensorboard).

### Optional development dependencies

```bash
pip install -e ".[dev]"
```

This adds pytest, pytest-cov, black, flake8, and mypy for development.

## Quick Start

### CLI-First Workflow

The `abx_amr_simulator` package is built in Python, but the primary way you'll interact with it is through **YAML configuration files and command-line commands**. Most experiments are managed by:
- Creating or modifying YAML configs
- Running training via `python -m abx_amr_simulator.training.train`
- Analyzing results via CLI analysis tools

This design prioritizes **reproducibility and modularity**: swap out components (environment, reward function, patient population) by pointing to different config files, without touching code.

*When do I write Python code?* See Tutorial 4.

### Setup and Training

Create a local config structure with bundled defaults:

```python
from pathlib import Path
from abx_amr_simulator.utils.config import setup_config_folders_with_defaults

# Creates configs/ folder in current directory with defaults
setup_config_folders_with_defaults(target_path=Path('.'))
```

This generates:
```
configs/
  ├── umbrella_configs/base_experiment.yaml
  ├── agent_algorithm/default.yaml (+ ppo.yaml)
  ├── environment/default.yaml
  ├── patient_generator/default.yaml
  └── reward_calculator/default.yaml
```

### 2. Run a Basic Experiment (CLI)

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -o "training.total_num_training_episodes=10" \
  -o "training.seed=42"
```

**Key command-line options:**
- `--config`: Path to umbrella experiment config YAML
- `-o` or `--override`: 
- Override specific subconfigs with '{class_type}_subconfig=relative_path_to_config.yaml'
  - Example: `-o "environment_subconfig=custom_env.yaml"`
- Override config values with dot notation
  - Example: `-o "reward_calculator.lambda_weight=0.5"`
  - Example: `-o "environment.num_patients_per_time_step=20"`

**Output:**
- Run saved to `results/<run_name>/`
- Trained model at `results/<run_name>/model.zip`
- TensorBoard logs at `results/<run_name>/logs/`

View results with:
```bash
tensorboard --logdir results/<run_name>/logs
```

### 3. Launch GUI Apps

#### Experiment Runner
Configure and launch training runs interactively:

```bash
streamlit run src/abx_amr_simulator/gui/experiment_runner.py
```

Opens at `http://localhost:8501` with:
- Base config selection
- Environment/reward/algorithm parameter tuning
- Continue training from prior experiments
- Real-time run monitoring

#### Experiment Viewer
Analyze completed runs:

```bash
streamlit run src/abx_amr_simulator/gui/experiment_viewer.py --server.port 8502
```

Opens at `http://localhost:8502` with:
- Run browser and config inspection
- Diagnostic plots (Phase 2)
- Evaluation metrics (Phase 3)

### 4. Post-Training Analysis (CLI)

After training completes, run analysis scripts to generate diagnostic and evaluation plots:

#### Phase 2: Diagnostic Analysis
Analyze training behavior and environment dynamics:

```bash
python -m abx_amr_simulator.analysis.diagnostic_analysis \
  --results-dir ./workspace/results
```

**Options:**
- `--results-dir`: Path to results directory (default: `results` folder in current working directory). Use this when the script is installed as a pip package and results folder is not located in the current working directory
- `--prefix`: Analyze specific experiment (e.g., `exp_1.a_single_abx_lambda0.0`)
- `--force`: Re-analyze all exp_* experiments

**Example with absolute path:**
```bash
python -m abx_amr_simulator.analysis.diagnostic_analysis \
  --results-dir /absolute/path/to/workspace/results \
  --prefix exp_base_test
```

#### Phase 3: Evaluative Plots
Generate evaluation plots from trained policies:

```bash
python -m abx_amr_simulator.analysis.evaluative_plots \
  --results-dir ./workspace/results
```

**Options:**
- `--results-dir`: Path to results directory (default: `results`)
- `--prefix`: Generate plots for specific experiment
- `--num-eval-episodes`: Number of evaluation episodes per seed (default: 50)
- `--force`: Force regeneration of all plots

**When to use `--results-dir`:**
When you install `abx_amr_simulator` via pip, the analysis scripts no longer run from your workspace root. You must specify where results are stored:

```bash
# If results are in a sibling directory
python -m abx_amr_simulator.analysis.evaluative_plots \
  --results-dir ../workspace/results

# Or use absolute path
python -m abx_amr_simulator.analysis.evaluative_plots \
  --results-dir /Users/joycelee/Work/Code/abx_amr_simulator/workspace/results
```

---

## Core Concepts

### Imports

**Core domain classes:**
```python
from abx_amr_simulator.core import (
    ABXAMREnv,                  # Main environment
    RewardCalculator,           # Reward function
    PatientGenerator,           # Patient distribution
    AMR_LeakyBalloon,          # AMR dynamics model
    Patient,                    # Patient dataclass
)
```

**Training utilities:**
```python
from abx_amr_simulator.utils import (
    load_config,                          # Load YAML configs
    create_environment,                   # Instantiate env
    create_reward_calculator,             # Instantiate reward calc
    create_patient_generator,             # Instantiate patient gen
    create_agent,                         # Create PPO/A2C/DQN
    setup_callbacks,                      # Training callbacks
)
```

**Analysis tools:**
```python
from abx_amr_simulator.analysis import (
    diagnostic_analysis,                  # Diagnostic metrics
    evaluative_plots,                     # Evaluation plots
    plot_ensemble_results,                # Multi-seed aggregates
    analyze_patient_data,                 # Patient trajectory analysis
)
```

### Configuration

Configs use nested YAML format:

**`base_experiment.yaml`** (umbrella config):
```yaml
environment: ../environment/default.yaml
reward_calculator: ../reward_calculator/default.yaml
patient_generator: ../patient_generator/default.yaml
agent_algorithm: ../agent_algorithm/default.yaml

training:
  total_num_training_episodes: 100
  seed: 42
```

**`environment/default.yaml`** (component config):
```yaml
num_patients_per_time_step: 50
max_time_steps: 500
antibiotics: [penicillin, amoxicillin]
leaky_balloon:
  leak: 0.1
  flatness_parameter: 0.5
```

Override via CLI:
```bash
python -m abx_amr_simulator.training.train \
  --config base_experiment.yaml \
  -o "environment.num_patients_per_time_step=20" \
  -o "reward_calculator.lambda_weight=0.8"
```

---

## Architecture Overview

```
src/abx_amr_simulator/
├── core/                       # Domain classes
│   ├── abx_amr_env.py         # Main environment
│   ├── reward_calculator.py    # Reward logic
│   ├── patient_generator.py    # Patient sampling
│   ├── leaky_balloon.py        # AMR dynamics
│   └── types.py                # Dataclasses
│
├── training/
│   └── train.py               # Main training entrypoint
│
├── analysis/                   # Analysis tools
│   ├── diagnostic_analysis.py
│   ├── evaluative_plots.py
│   └── ...
│
├── callbacks/                  # SB3 callbacks
│   └── __init__.py
│
├── utils/                      # Utilities
│   ├── config.py              # Config loading
│   ├── factories.py           # Object creation
│   ├── registry.py            # Experiment tracking
│   └── metrics.py             # Evaluation plots
│
├── gui/                        # Streamlit apps
│   ├── experiment_runner.py
│   ├── experiment_viewer.py
│   └── launch_apps.py
│
├── wrappers.py                # SB3 environment wrappers
├── formatters.py              # Debug output formatting
└── README.md                  # This file
```

---

## Troubleshooting

**Config file not found:**
- Ensure paths in umbrella config are relative to config file location
- Or use absolute paths

**"PatientGenerator seed differs from RewardCalculator seed":**
- Expected warning; environment auto-syncs seeds for reproducibility
- Suppress by setting both to same value in config

**Streamlit app won't launch:**
- Install streamlit: `pip install streamlit`
- Ensure port is available (default 8501, 8502)

**Tests failing:**
```bash
pytest tests/
```
- Expected: **226 tests pass**. See [tests/README.md](../../tests/README.md) for suite details.

---

## Further Reading

- [ENVIRONMENT_SPEC.md](../../docs/ENVIRONMENT_SPEC.md) — Environment specification
- [CONFIG_SYSTEM.md](../../docs/CONFIG_SYSTEM.md) — Configuration system details
- [Diagnostic Analysis](../../workspace/docs/diagnostic_analysis.md) — Phase 2 outputs and interpretation
- [Evaluative Plots](../../workspace/docs/evaluative_plots.md) — Phase 3 outputs and interpretation
- [CHANGELOG.md](../../docs/CHANGELOG.md) — Release notes and architecture evolution

---

## Citation

```bibtex
@software{abx_amr_simulator_2026,
  title = {ABX AMR Simulator: Reinforcement Learning for Antibiotic Stewardship},
  author = {Joyce Lee},
  year = {2026},
  url = {https://github.com/jl56923/abx_simulator}
}
```