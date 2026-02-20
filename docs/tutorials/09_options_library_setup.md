# Tutorial 7: Options Library Setup

**Goal**: Learn to create and customize option libraries for HRL training.

**Prerequisites**: Completed Tutorial 6 (HRL Quick Start)

**What is an option library?** A YAML file that defines a menu of high-level strategies (options) the manager policy can select from. Each option specifies:
- **What it does**: Block, alternation, heuristic, or custom behavior
- **How long it runs**: Fixed duration in environment steps
- **How it's configured**: Custom parameters via config overrides

---

## Overview

Option library workflow:
1. **Define library structure**: Create YAML file with metadata and option list
2. **Add options**: Specify each option's type, loader, config, and parameters
3. **Point umbrella config at library**: Update `hrl.option_library` path
4. **Train and iterate**: Test manager with library, adjust options based on performance

---

## Step 1: Library YAML Structure

Minimal library structure:

```yaml
library_name: "my_custom_library"
description: "Custom option library for experiments"
version: "1.0"

options:
  - option_name: "no_treatment_10"
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    loader_module: "../option_types/block/block_option_loader.py"
    config_params_override:
      antibiotic: "no_treatment"
      duration: 10

  - option_name: "A_15"
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    loader_module: "../option_types/block/block_option_loader.py"
    config_params_override:
      antibiotic: "A"
      duration: 15
  
  # ... more options ...
```

**Key fields:**
- `library_name`: Unique identifier for this library
- `description`: Human-readable summary
- `version`: Version string (for tracking changes)
- `options`: List of option definitions

---

## Step 2: Standard Option Types

The package includes three built-in option types:

### 1. Block Options

**What they do**: Prescribe a single antibiotic for a fixed duration.

**Use cases**: Baseline strategies, treatment courses, no-treatment periods.

**Example:**

```yaml
- option_name: "A_10"
  option_type: "block"
  option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
  loader_module: "../option_types/block/block_option_loader.py"
  config_params_override:
    antibiotic: "A"          # Which antibiotic to prescribe
    duration: 10             # How many steps to prescribe it
```

**Parameters:**
- `antibiotic`: Action name (e.g., `"A"`, `"B"`, `"no_treatment"`)
- `duration`: Number of environment steps (positive integer)

### 2. Alternation Options

**What they do**: Prescribe a sequence of antibiotics in order, cycling if necessary.

**Use cases**: Cycling strategies, combination therapies, stewardship protocols.

**Example:**

```yaml
- option_name: "ALT_AABB"
  option_type: "alternation"
  option_subconfig_file: "../option_types/alternation/alternation_option_default_config.yaml"
  loader_module: "../option_types/alternation/alternation_option_loader.py"
  config_params_override:
    sequence:
      - "A"
      - "A"
      - "B"
      - "B"
```

**Parameters:**
- `sequence`: List of action names to prescribe in order
- Duration is implicit: length of sequence (loops if needed for longer episodes)

### 3. Heuristic Options

**What they do**: Choose actions based on patient attributes and AMR levels using thresholds.

**Use cases**: Expert heuristics, risk-stratified treatment, domain knowledge injection.

**Example:**

```yaml
- option_name: "HEURISTIC_conservative"
  option_type: "heuristic"
  option_subconfig_file: "../option_types/heuristic/heuristic_option_default_config.yaml"
  loader_module: "../option_types/heuristic/heuristic_option_loader.py"
  config_params_override:
    duration: 10
    action_thresholds:
      no_treatment: 0.0        # Prescribe no treatment if prob_infected < 0.3
      A: 0.3                   # Prescribe A if 0.3 ≤ prob_infected < 0.7
      B: 0.7                   # Prescribe B if prob_infected ≥ 0.7
    attribute_weights:
      prob_infected: 1.0       # Which patient attributes to use for thresholds
```

**Parameters:**
- `duration`: Number of steps to run
- `action_thresholds`: Dict mapping actions to threshold values (lower bound for that action)
- `attribute_weights`: Dict mapping patient attributes to weights (for scoring each patient)

**How heuristic options work:**
1. Calculate weighted score for each patient: `score = Σ(weight × attribute_value)`
2. Compare score against action thresholds
3. Select action with highest threshold that score exceeds
4. Prescribe selected action for current step

---

## Step 3: Loader Module Formats

`loader_module` supports two formats:

### Format 1: Relative file path (for bundled loaders)

```yaml
loader_module: "../option_types/block/block_option_loader.py"
```

- Path is relative to the option library YAML file
- Used for default loaders created by `setup_options_folders_with_defaults()`

### Format 2: Python module path (for custom loaders)

```yaml
loader_module: "abx_amr_simulator.options.block_loader"
```

- Standard Python import path
- Used for loaders in installed packages
- Avoids absolute file paths

**Both formats work identically**—choose based on your project structure.

---

## Step 4: End-to-End Example

Let's create a custom option library from scratch:

### 4.1: Create library file

```bash
cd myproject/options/option_libraries
touch my_experiment.yaml
```

### 4.2: Define library content

Edit `my_experiment.yaml`:

```yaml
library_name: "my_experiment"
description: "6 options: 3 blocks + 2 alternations + 1 heuristic"
version: "1.0"

options:
  # Conservative: wait and see
  - option_name: "no_treatment_5"
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    loader_module: "../option_types/block/block_option_loader.py"
    config_params_override:
      antibiotic: "no_treatment"
      duration: 5

  # Aggressive: treat immediately
  - option_name: "A_10"
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    loader_module: "../option_types/block/block_option_loader.py"
    config_params_override:
      antibiotic: "A"
      duration: 10

  - option_name: "B_10"
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    loader_module: "../option_types/block/block_option_loader.py"
    config_params_override:
      antibiotic: "B"
      duration: 10

  # Cycling strategies
  - option_name: "ALT_AB"
    option_type: "alternation"
    option_subconfig_file: "../option_types/alternation/alternation_option_default_config.yaml"
    loader_module: "../option_types/alternation/alternation_option_loader.py"
    config_params_override:
      sequence:
        - "A"
        - "B"

  - option_name: "ALT_BA"
    option_type: "alternation"
    option_subconfig_file: "../option_types/alternation/alternation_option_default_config.yaml"
    loader_module: "../option_types/alternation/alternation_option_loader.py"
    config_params_override:
      sequence:
        - "B"
        - "A"

  # Risk-stratified heuristic
  - option_name: "HEURISTIC_risk_stratified"
    option_type: "heuristic"
    option_subconfig_file: "../option_types/heuristic/heuristic_option_default_config.yaml"
    loader_module: "../option_types/heuristic/heuristic_option_loader.py"
    config_params_override:
      duration: 10
      action_thresholds:
        no_treatment: 0.0     # Low risk: no treatment
        A: 0.4               # Medium risk: use A
        B: 0.7               # High risk: use B
      attribute_weights:
        prob_infected: 1.0
```

### 4.3: Update umbrella config

Edit your umbrella config to point at the new library:

```bash
# Copy default HRL config
cp configs/umbrella_configs/hrl_ppo_default.yaml configs/umbrella_configs/my_experiment.yaml
```

Edit `configs/umbrella_configs/my_experiment.yaml`:

```yaml
# ... other config ...

# HRL configuration
hrl:
  option_library: option_libraries/my_experiment.yaml  # ← Changed from default_deterministic.yaml

# Training configuration
training:
  run_name: my_experiment  # ← Descriptive name
  # ... other training params ...
```

### 4.4: Train with custom library

```bash
python -m abx_amr_simulator.training.train \
  --umbrella-config $(pwd)/configs/umbrella_configs/my_experiment.yaml \
  --seed 42
```

**Expected output:**

```
Loading option library: options/option_libraries/my_experiment.yaml
Loaded 6 options: no_treatment_5, A_10, B_10, ALT_AB, ALT_BA, HEURISTIC_risk_stratified
Manager action space: Discrete(6)
Training...
```

### 4.5: Verify option loading

Check TensorBoard for option selection distribution:

```bash
tensorboard --logdir results/my_experiment_*/logs
```

Look for metrics like:
- `option_selection_counts`: How often each option was selected
- `option_rewards`: Average reward per option

---

## Step 5: Advanced Library Design Patterns

### Pattern 1: Duration Sweep

Create options with varying durations to let manager learn temporal strategy:

```yaml
options:
  - option_name: "A_5"
    option_type: "block"
    # ... config ...
    config_params_override:
      antibiotic: "A"
      duration: 5

  - option_name: "A_10"
    option_type: "block"
    # ... config ...
    config_params_override:
      antibiotic: "A"
      duration: 10

  - option_name: "A_15"
    option_type: "block"
    # ... config ...
    config_params_override:
      antibiotic: "A"
      duration: 15
```

**Use case**: Manager learns optimal treatment duration for each antibiotic.

### Pattern 2: Action Coverage

Ensure every primitive action is available as an option:

```yaml
options:
  - option_name: "no_treatment_10"
    # ... no_treatment block ...

  - option_name: "A_10"
    # ... A block ...

  - option_name: "B_10"
    # ... B block ...
```

**Use case**: Guarantee manager can always choose any action (avoid dead ends).

### Pattern 3: Heuristic Baseline

Include at least one expert heuristic for comparison:

```yaml
options:
  # ... learned options ...

  - option_name: "HEURISTIC_expert"
    option_type: "heuristic"
    # ... expert thresholds based on domain knowledge ...
```

**Use case**: Benchmark learned policy against expert strategy (did manager improve over expert?).

### Pattern 4: Small Library for Debugging

Start with 3-5 options for initial experiments:

```yaml
options:
  - option_name: "no_treatment_10"
    # ... conservative option ...

  - option_name: "A_10"
    # ... aggressive option A ...

  - option_name: "B_10"
    # ... aggressive option B ...
```

**Use case**: Faster training, easier debugging, clearer option selection patterns.

---

## Step 6: Custom Option Types (Advanced)

To create a new option type beyond block/alternation/heuristic:

### 6.1: Implement loader module

Create `options/option_types/my_custom_type/my_custom_loader.py`:

```python
"""Custom option loader."""
from abx_amr_simulator.options.base_option import OptionBase
import yaml

def load_my_custom_type_option(
    option_name: str,
    option_subconfig_file: str,
    config_params_override: dict,
    env,
    **kwargs
) -> OptionBase:
    """Load custom option type.
    
    Args:
        option_name: Unique name for this option instance
        option_subconfig_file: Path to config YAML
        config_params_override: Custom parameters from library
        env: ABXAMREnv instance
        **kwargs: Additional arguments
    
    Returns:
        OptionBase: Configured option instance
    """
    # Load base config
    with open(option_subconfig_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    config.update(config_params_override)
    
    # Instantiate your custom option class
    option = MyCustomOption(
        name=option_name,
        duration=config['duration'],
        # ... other params ...
    )
    
    return option
```

### 6.2: Define option class

Create `options/option_types/my_custom_type/my_custom_option.py`:

```python
"""Custom option implementation."""
from abx_amr_simulator.options.base_option import OptionBase
import numpy as np

class MyCustomOption(OptionBase):
    """Custom option that implements domain-specific logic."""
    
    def __init__(self, name: str, duration: int, **kwargs):
        super().__init__(name=name, duration=duration)
        # ... custom initialization ...
    
    def act(self, obs: np.ndarray, env) -> np.ndarray:
        """Select actions for current step.
        
        Args:
            obs: Observation from environment
            env: ABXAMREnv instance
        
        Returns:
            actions: Action array (shape: num_patients)
        """
        # Implement your custom logic here
        # Example: simple threshold-based policy
        actions = np.zeros(env.num_patients, dtype=int)
        # ... fill in actions based on obs ...
        return actions
```

### 6.3: Add to library

```yaml
- option_name: "MY_CUSTOM_OPTION"
  option_type: "my_custom_type"
  option_subconfig_file: "../option_types/my_custom_type/my_custom_config.yaml"
  loader_module: "../option_types/my_custom_type/my_custom_loader.py"
  config_params_override:
    duration: 10
    # ... custom params ...
```

---

## Troubleshooting

### "Option library file not found"

**Cause**: `hrl.option_library` path is incorrect or relative to wrong directory.

**Solution**: Paths in `hrl.option_library` are relative to `options_folder_location`. Example:
```yaml
options_folder_location: ../../options  # From umbrella config directory
hrl:
  option_library: option_libraries/my_experiment.yaml  # Relative to options/
```

### "Loader module not found"

**Cause**: `loader_module` path is incorrect or Python can't import it.

**Solution for file paths**: Verify path is relative to library YAML:
```yaml
loader_module: "../option_types/block/block_option_loader.py"  # From option_libraries/
```

**Solution for module paths**: Ensure module is in PYTHONPATH or is an installed package:
```yaml
loader_module: "abx_amr_simulator.options.block_loader"  # Must be importable
```

### "Duration mismatch" or options end prematurely

**Cause**: Option duration exceeds episode length or environment max_timesteps.

**Solution**: Ensure option durations are shorter than typical episode length:
- If `max_timesteps=100`, use durations ≤ 20 (allow multiple option executions per episode)
- If options should run until episode end, set `duration=-1` (if supported by option type)

### Manager doesn't explore all options

**Cause**: Insufficient exploration (low entropy coefficient) or reward signal favors one option.

**Solution**:
1. Increase `ent_coef` in HRL algorithm config (e.g., `0.01` → `0.05`)
2. Add reward shaping to encourage option diversity
3. Reduce library size (fewer options = more samples per option)

---

## What's Next?

✅ You've learned to create custom option libraries!

**Next tutorials**:
- **Tutorial 08**: [HRL Diagnostics](08_hrl_diagnostics.md) — Analyze option selection patterns and visualize manager behavior
- **Tutorial 11**: [Advanced Heuristic Worker Subclassing](11_advanced_heuristic_worker_subclassing.md) — Implement sophisticated heuristic options with attribute estimation

---

## Key Takeaways

1. **Option library is a YAML file**: Defines menu of high-level strategies
2. **Three standard option types**: Block, alternation, heuristic
3. **Two loader formats**: File paths (for bundled loaders) or module paths (for custom loaders)
4. **Library design patterns**: Duration sweeps, action coverage, heuristic baselines, small libraries for debugging
5. **Path resolution**: `option_subconfig_file` and `loader_module` are relative to library YAML file
6. **Custom options**: Implement loader module + option class, add to library
7. **End-to-end workflow**: Create library → update umbrella config → train → verify in TensorBoard
