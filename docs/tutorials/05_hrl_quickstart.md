# Tutorial 6: HRL Quick Start

**Goal**: Learn to train hierarchical reinforcement learning (HRL) agents using deterministic option libraries.

**Prerequisites**: Completed Tutorial 1 (Basic Training) and Tutorial 2 (Config Scaffolding)

**What is HRL?**: Instead of the agent choosing raw actions (which antibiotic to prescribe), the agent chooses from a library of **options** (high-level strategies like "prescribe A for 10 steps" or "alternate A and B"). This enables learning compositional policies and comparing learned strategies against expert heuristics.

---

## Overview

HRL training workflow:
1. **Set up configs**: Standard environment/reward/patient configs + HRL algorithm config
2. **Set up option library**: Define menu of high-level strategies (options)
3. **Train manager policy**: Agent learns to select options from the library
4. **Analyze outputs**: Compare learned option preferences against baselines

**Algorithms supported**:
- **HRL_PPO**: Manager uses PPO to select options
- **HRL_RPPO**: Manager uses R-PPO (recurrent PPO with LSTM for hidden state tracking)

---

## Step 1: Set Up Project Structure

From your project root (e.g., `myproject/`):

```bash
# Verify you're in project root
pwd  # Should show /path/to/myproject

# Create configs folder (if not done already)
python -c "from abx_amr_simulator.utils import setup_config_folders_with_defaults; setup_config_folders_with_defaults('.')"

# Create options folder with default option library
python -c "from abx_amr_simulator.hrl import setup_options_folders_with_defaults; setup_options_folders_with_defaults('.')"
```

**Expected structure after setup:**

```
myproject/
├── configs/
│   ├── umbrella_configs/
│   │   ├── base_experiment.yaml
│   │   └── hrl_ppo_default.yaml  ← Use this for HRL
│   ├── environment/
│   ├── patient_generator/
│   ├── reward_calculator/
│   └── agent_algorithm/
│       ├── ppo.yaml
│       ├── hrl_ppo.yaml
│       └── hrl_rppo.yaml
├── options/                        ← New! HRL-specific
│   ├── option_libraries/
│   │   └── default_deterministic.yaml  ← Menu of 12 options
│   └── option_types/
│       ├── block/
│       ├── alternation/
│       └── heuristic/
└── results/                        ← Training outputs saved here
```

---

## Step 2: Inspect the Default Option Library

View the default option library:

```bash
cat options/option_libraries/default_deterministic.yaml
```

**Contents** (12 deterministic options):
- **9 block options**: Prescribe a single antibiotic for fixed duration
  - Examples: `no_treatment_5`, `A_10`, `B_15` (antibiotic + duration)
- **3 alternation options**: Prescribe sequence of antibiotics
  - Examples: `ALT_AABBA` (prescribe A, A, B, B, A in sequence)

**Key fields in each option:**
- `option_name`: Unique identifier (e.g., `A_10`)
- `option_type`: Type of option (e.g., `block`, `alternation`, `heuristic`)
- `loader_module`: Path to Python module that implements the option
- `option_subconfig_file`: Path to config file with option parameters
- `config_params_override`: Custom parameters for this option instance

---

## Step 3: Inspect the HRL Umbrella Config

View the default HRL PPO umbrella config:

```bash
cat configs/umbrella_configs/hrl_ppo_default.yaml
```

**Key HRL-specific fields:**

```yaml
# Folder locations (relative paths from umbrella config directory)
config_folder_location: ../
options_folder_location: ../../options

# Component configs
agent_algorithm: agent_algorithm/hrl_ppo.yaml  # ← Uses HRL algorithm

# HRL configuration
hrl:
  option_library: option_libraries/default_deterministic.yaml  # ← Points to option library

# Training configuration
training:
  run_name: hrl_ppo_default
  total_num_training_episodes: 25  # Shorter than flat RL (faster episodes with options)
  # ... other training params
```

**What's different from flat RL?**
- `agent_algorithm` points to `hrl_ppo.yaml` (not `ppo.yaml`)
- New `hrl:` section with `option_library` path
- `options_folder_location` tells config loader where to find option libraries

---

## Step 4: Train HRL Agent

From your project root:

```bash
python -m abx_amr_simulator.training.train \
  --umbrella-config $(pwd)/configs/umbrella_configs/hrl_ppo_default.yaml \
  --seed 42
```

**What happens during training:**
1. Manager policy receives environment observations (patient attributes, AMR levels)
2. Manager selects an option index (0-11 for default library with 12 options)
3. Selected option executes its strategy for its duration (5-15 steps)
4. Manager receives reward and observes next state
5. Repeat until episode ends

**Training outputs:**

```
results/
└── hrl_ppo_default_20260217_143027/
    ├── best_model.zip
    ├── final_model.zip
    ├── full_agent_env_config.yaml
    ├── logs/                     ← TensorBoard logs
    │   └── PPO_1/
    ├── analysis_output/          ← Evaluation metrics
    │   └── eval_episode_*.json
    └── diagnostics/              ← HRL-specific diagnostics
        └── option_selection_counts.json
```

---

## Step 5: Monitor Training

### TensorBoard (same as flat RL):

```bash
tensorboard --logdir results/hrl_ppo_default_20260217_143027/logs
```

Open `http://localhost:6006` to view:
- **Reward curves**: Manager policy learning progress
- **Option selection distribution**: Which options the manager prefers over time

### Terminal output summary:

```
Episode 5/25: mean_reward=120.5, mean_amr=0.35
Option selection counts: {0: 45, 3: 32, 7: 18, ...}  ← Which options were used
```

---

## Step 6: Understanding Manager Observations and Actions

### Manager observation space:
- **Same observations as flat RL**: Patient attributes + AMR levels + (optionally) steps since AMR update
- Example shape: `(num_patients × num_visible_attrs + num_antibiotics + 1,)`

### Manager action space:
- **NOT antibiotic prescriptions**: Manager chooses option indices
- Example: `Discrete(12)` for 12-option library
- Action 0 = select first option, action 1 = select second option, etc.

### Option execution:
- Once manager selects an option, that option runs for its full duration
- Manager doesn't intervene until option completes
- Option observes environment and takes low-level actions (antibiotic prescriptions)

**Analogy**: Manager is the quarterback calling plays (options), options are the plays that execute on the field (antibiotic prescriptions).

---

## Step 7: Compare HRL vs Flat RL

### When to use HRL:
- ✅ You want to inject expert heuristics as options (benchmark learned policy against domain knowledge)
- ✅ You want compositional policies (combine simple strategies)
- ✅ You want temporal abstraction (operate at coarser timescales than individual actions)
- ✅ You want interpretable policies (option names are human-readable)

### When to use flat RL:
- ✅ You want maximum flexibility (no constraints from option library)
- ✅ You want simplest possible setup (no option library management)
- ✅ You have domain-agnostic problem (no obvious high-level strategies)

### Training time comparison:
- **HRL**: Faster episodes (fewer manager decisions), but overhead from option execution
- **Flat RL**: More granular control, but more decisions per episode
- **Rule of thumb**: If your problem has natural temporal structure (e.g., treatment courses lasting multiple steps), HRL can be more sample-efficient

---

## Step 8: Customize Your Option Library

To create custom options, see [Tutorial 07: Options Library Setup](07_options_library_setup.md).

**Quick customization examples:**

### Add a new block option:

Edit `options/option_libraries/default_deterministic.yaml`:

```yaml
options:
  # ... existing options ...
  
  - option_name: "A_20"  # New longer duration
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    loader_module: "../option_types/block/block_option_loader.py"
    config_params_override:
      antibiotic: "A"
      duration: 20
```

### Add a new alternation option:

```yaml
  - option_name: "ALT_ABAB"  # New alternation pattern
    option_type: "alternation"
    option_subconfig_file: "../option_types/alternation/alternation_option_default_config.yaml"
    loader_module: "../option_types/alternation/alternation_option_loader.py"
    config_params_override:
      sequence:
        - "A"
        - "B"
        - "A"
        - "B"
```

**After editing the library**, rerun training—the manager will now have access to the new options.

---

## Step 9: Switch to HRL RPPO (Recurrent Manager)

For environments with partial observability or hidden dynamics, use HRL_RPPO (manager with LSTM):

```bash
# Copy and edit the umbrella config
cp configs/umbrella_configs/hrl_ppo_default.yaml configs/umbrella_configs/hrl_rppo_custom.yaml

# Edit: change agent_algorithm line
# agent_algorithm: agent_algorithm/hrl_rppo.yaml  # ← Changed from hrl_ppo.yaml

# Train with RPPO manager
python -m abx_amr_simulator.training.train \
  --umbrella-config $(pwd)/configs/umbrella_configs/hrl_rppo_custom.yaml \
  --seed 42
```

**Differences from HRL_PPO:**
- Manager policy uses LSTM to maintain hidden state
- Better for partially observable environments (e.g., noisy AMR observations)
- Longer training time due to recurrent network overhead

See [Tutorial 09: HRL RPPO Manager](09_hrl_rppo_manager.md) for RPPO-specific configuration and troubleshooting.

---

## Troubleshooting

### "Option library not found"

**Cause**: `hrl.option_library` path is incorrect or options folder missing.

**Solution**: Verify `options_folder_location` in umbrella config and check that `options/option_libraries/default_deterministic.yaml` exists.

### "Option loader module not found"

**Cause**: `loader_module` path in option library is incorrect.

**Solution**: Verify paths in option library YAML are relative to the library file itself. Example:
```yaml
loader_module: "../option_types/block/block_option_loader.py"  # Correct
loader_module: "option_types/block/block_option_loader.py"     # Incorrect (missing ../)
```

### Manager selects same option repeatedly

**Possible causes**:
1. **Option too rewarding**: One option dominates reward signal
2. **Insufficient exploration**: Manager needs higher entropy coefficient

**Solution**: Check `ent_coef` in `agent_algorithm/hrl_ppo.yaml` (increase for more exploration, e.g., `0.05` → `0.1`).

### Training much slower than expected

**Cause**: Option library too large (>20 options) or options have long durations.

**Solution**:
- Reduce number of options (fewer than 15 is typical)
- Reduce option durations (5-15 steps is typical)
- Use HRL_PPO instead of HRL_RPPO (LSTM adds overhead)

---

## What's Next?

✅ You've trained your first HRL agent!

**Next tutorials**:
- **Tutorial 07**: [Options Library Setup](07_options_library_setup.md) — Create custom option libraries with heuristic options
- **Tutorial 08**: [HRL Diagnostics](08_hrl_diagnostics.md) — Analyze option selection patterns and visualize manager behavior
- **Tutorial 09**: [HRL RPPO Manager](09_hrl_rppo_manager.md) — Configure recurrent manager policies for partial observability

---

## Key Takeaways

1. **HRL = hierarchical decision making**: Manager selects options, options execute low-level actions
2. **Option library is the menu**: Defines what high-level strategies the manager can choose from
3. **Two setup utilities**: `setup_config_folders_with_defaults()` for configs, `setup_options_folders_with_defaults()` for options
4. **Manager observation space**: Same as flat RL (patient + AMR observations)
5. **Manager action space**: Discrete option indices (not antibiotic prescriptions)
6. **Two manager algorithms**: HRL_PPO (feedforward) and HRL_RPPO (recurrent with LSTM)
7. **Customization**: Edit option library YAML to add/remove/modify options
8. **Diagnostics**: Check `results/.../diagnostics/option_selection_counts.json` to see which options the manager prefers
