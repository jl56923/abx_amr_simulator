# Tutorial 2: Config and Options Scaffolding

**Goal**: Learn how to set up your experiment workspace with configuration folders, option libraries, and tuning configs using the package's scaffolding utilities.

**Prerequisites**: Completed Tutorial 1 (Basic Training Workflow)

---

## Overview

Before running experiments, you need to scaffold your workspace with the proper directory structure and default configuration files. The `abx_amr_simulator` package provides three utilities to help you set up:

1. **Config scaffolding** — Creates `configs/` folder with default component configs (environment, reward_calculator, patient_generator, agent_algorithm, umbrella configs)
2. **Options scaffolding** — Creates `options/` folder with default HRL option libraries and templates
3. **Tuning scaffolding** — Creates `tuning_configs/` folder with default Optuna hyperparameter search spaces

This tutorial shows how to use each utility and explains the resulting folder structure.

---

## Step 1: Create Your Project Directory

Start by creating a project directory (consistent with Tutorial 1):

```bash
mkdir my_first_project
cd my_first_project
mkdir experiments
cd experiments

# Verify you're in the correct directory
pwd  # Should show: .../my_first_project/experiments
```

Your workspace should look like:
```
my_first_project/
└── experiments/
```

**Path validation tip**: Many commands in this tutorial refer to "From my_first_project/ directory" or "From my_first_project/experiments/ directory". Always verify your location with `pwd` before running commands to avoid "file not found" errors.

---

## Step 2: Scaffold Configuration Folders

### Using Python API

The config scaffolding utility is available as a Python function:

**NOTE:** revise the bash snippet to tell the user to run 'setup_config_folders_with_defaults' from the 'experiments' subfolde,r not from 'my_first_project'.

```bash
python -c "from abx_amr_simulator.utils import setup_config_folders_with_defaults; setup_config_folders_with_defaults('.')"
```

This creates:

```
experiments/
└── configs/
    ├── umbrella_configs/
    │   ├── base_experiment.yaml
    │   └── hrl_ppo_default.yaml
    ├── agent_algorithm/
    │   ├── default.yaml
    │   ├── ppo.yaml
    │   ├── a2c.yaml
    │   ├── hrl_ppo.yaml
    │   ├── hrl_rppo.yaml
    │   ├── mbpo.yaml
    │   └── recurrent_ppo.yaml
    ├── environment/
    │   └── default.yaml
    ├── patient_generator/
    │   ├── default.yaml
    │   └── default_mixer.yaml
    └── reward_calculator/
        └── default.yaml
```

**What each folder contains:**
- **umbrella_configs/** — Top-level configs that stitch together all component subconfigs. These use `config_folder_location` and `options_folder_location` to specify where components live.
- **agent_algorithm/** — Agent-specific hyperparameters (PPO, A2C, HRL variants)
- **environment/** — Environment parameters (patients per timestep, max timesteps, AMR dynamics)
- **patient_generator/** — Patient population distributions and observation configurations
- **reward_calculator/** — Reward function parameters (lambda weight, clinical benefits, penalties)

**Path resolution:**
The umbrella configs use a modern format that explicitly declares where to find component configs:
```yaml
config_folder_location: ../  # Relative to umbrella config's directory
options_folder_location: ../../options

# Component paths are then relative to config_folder_location (no ../ prefix needed)
environment: environment/default.yaml
reward_calculator: reward_calculator/default.yaml
patient_generator: patient_generator/default.yaml
agent_algorithm: agent_algorithm/default.yaml
```

This design means:
- `config_folder_location: ../` resolves from `experiments/configs/umbrella_configs/` → `experiments/configs/`
- `environment: environment/default.yaml` then resolves to `experiments/configs/environment/default.yaml`
- If you move your configs folder, you only need to update `config_folder_location` in the umbrella config

---

## Step 3: Scaffold Options Folders (for HRL)

If you plan to use hierarchical RL (HRL) with option libraries, scaffold the options folder:

**NOTE:** Again, need to tell the user to run this from the 'experiments' subfolder

```bash
python -m abx_amr_simulator.hrl.setup_options --target-path .
```

This creates:

```
experiments/
└── options/
    ├── option_libraries/
    │   └── example_library.yaml
    └── option_types/
        ├── block/
        │   └── block_option_default_config.yaml
        ├── alternation/
        │   └── alternation_option_default_config.yaml
        └── heuristic/
            ├── heuristic_option_default_config.yaml
            └── heuristic_option_loader.py
```

**What each folder contains:**
- **option_libraries/** — YAML files defining collections of options (the "library" the manager chooses from)
- **option_types/** — Default configs and loaders for each option type (block, alternation, heuristic)

You'll learn how to customize option libraries in Tutorial 7. **NOTE** provide link to Tutorial 7, and also the title of it.

---

## Step 4: Scaffold Tuning Configs (for Optuna)

If you plan to run hyperparameter tuning with Optuna, scaffold the tuning configs:

**NOTE:** Again, need to tell the user to run this from the 'experiments' subfolder

```bash
python -m abx_amr_simulator.training.setup_tuning --target-path .
```

This creates:

```
experiments/
└── tuning_configs/
    ├── ppo_tuning_default.yaml
    └── hrl_ppo_tuning_default.yaml
```

**What each file contains:**
- **ppo_tuning_default.yaml** — Optuna search space for PPO hyperparameters
- **hrl_ppo_tuning_default.yaml** — Optuna search space for HRL-PPO hyperparameters

You'll learn how to use these in Tutorial 4 (Optimization). **NOTE** provide link to Tutorial 4, and also the title of it.

---

## Step 5: Complete Project Structure

After running all three scaffolding utilities, your `my_first_project/` directory should look like:

```
my_first_project/
├── experiments/
│   ├── configs/
│   │   ├── umbrella_configs/
│   │   ├── agent_algorithm/
│   │   ├── environment/
│   │   ├── patient_generator/
│   │   └── reward_calculator/
│   ├── options/
│   │   ├── option_libraries/
│   │   └── option_types/
│   └── tuning_configs/
├── results/          (created automatically during training)
└── optimization/     (created automatically during tuning)
```

**Note**: `results/` and `optimization/` folders are created automatically when you run training or tuning commands. You don't need to create them manually.

---

## Step 6: Verify Your Setup

Check that all folders exist:

```bash
cd my_first_project
ls -R experiments/
```

You should see all the scaffolded folders and default config files.

---

## Quick Reference: Scaffolding Commands

```bash
# From within my_first_project/experiments/, run:

# 1. Config scaffolding (Python API)
python -c "from abx_amr_simulator.utils import setup_config_folders_with_defaults; setup_config_folders_with_defaults('.')"

# 2. Options scaffolding (CLI)
python -m abx_amr_simulator.hrl.setup_options --target-path .

# 3. Tuning scaffolding (CLI)
python -m abx_amr_simulator.training.setup_tuning --target-path .
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'abx_amr_simulator'"

**Solution**: Install the package:

```bash
pip install -e .
```

### "FileExistsError: configs/ already exists"

**Solution**: The scaffolding utilities are idempotent—they won't overwrite existing files. If you want to reset, delete the folder first:

```bash
rm -rf experiments/configs/
python -c "from abx_amr_simulator.utils import setup_config_folders_with_defaults; setup_config_folders_with_defaults('experiments')"
```

---

## What's Next?

✅ You've set up your workspace!

**Next tutorials**:
- **Tutorial 3**: Customize your experiments (modify configs, tune reward lambda, run parameter sweeps)
- **Tutorial 4**: Optimize hyperparameters with Optuna
- **Tutorial 6**: Train hierarchical RL agents with option libraries

---

## Key Takeaways

1. **Three scaffolding utilities**: Config, options, and tuning scaffolding create default folder structures
2. **experiments/ is your workspace**: All configs, options, and tuning configs live here
3. **results/ and optimization/ are auto-created**: Don't create them manually
4. **Idempotent utilities**: Safe to re-run without overwriting existing files
