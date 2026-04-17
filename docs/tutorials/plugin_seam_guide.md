# Plugin Seam Guide

**Goal**: Configure and load custom `PatientGenerator`, `RewardCalculator`, AMR dynamics, and HRL option components using the package-standard plugin seam.

**Prerequisites**: Familiarity with YAML experiment configs and basic subclassing in Python.

---

## 1) Overview

The plugin seam is a config-driven way to replace core simulator components without changing `train.py` or `tune.py` code paths. It exists so users can subclass core components and swap them in using YAML only.

Design philosophy:
- **Config-driven**: plugin selection happens in config, not in wrapper scripts.
- **Opt-in**: canonical behavior is unchanged unless plugin keys are provided.
- **Fail-loud**: invalid plugin config raises immediately before training/tuning starts.
- **Consistent across component families**: custom components use the same `plugin.loader_module` / `plugin.loader_function` seam, while canonical built-ins remain loader-free.

---

## 2) How It Works

### Config key structure

```yaml
patient_generator:
  plugin:
    loader_module: my.module.path
    loader_function: load_patient_generator_component  # optional (default)

reward_calculator:
  plugin:
    loader_module: my.module.path
    loader_function: load_reward_calculator_component  # optional (default)

amr_dynamics:
  plugin:
    loader_module: my.module.path
    loader_function: load_amr_dynamics_component  # optional (default)
```

For HRL options, the same seam is used inside each option entry when `option_type: custom` is selected:

```yaml
options:
    - option_name: "A_10"
        option_type: "block"
        option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
        config_params_override:
            antibiotic: "A"
            duration: 10

    - option_name: "MY_CUSTOM_OPTION"
        option_type: "custom"
        option_subconfig_file: "../option_types/heuristic/my_custom_option_config.yaml"
        plugin:
            loader_module: "../option_types/heuristic/my_custom_option.py"
            loader_function: load_my_custom_option
        config_params_override:
            duration: 10
```

### Path resolution

`plugin.loader_module` supports:
- Python import path (for example: `my_project.plugins.custom_pg_loader`)
- Filesystem path (absolute or relative)

Relative filesystem paths are resolved against `_umbrella_config_dir`.

### Loader signatures

```python
from typing import Any, Dict
from abx_amr_simulator.core import PatientGeneratorBase, RewardCalculatorBase, AMRDynamicsBase

def load_patient_generator_component(config: Dict[str, Any]) -> PatientGeneratorBase: ...
def load_reward_calculator_component(config: Dict[str, Any]) -> RewardCalculatorBase: ...
def load_amr_dynamics_component(config: Dict[str, Any]) -> Dict[str, AMRDynamicsBase]: ...
```

Each loader receives the merged component config and must return the correct type.

### What fail-loud means

If any of the following are invalid, execution stops immediately with a descriptive exception:
- Missing `plugin.loader_module`
- Non-importable loader module
- Missing/non-callable loader function
- Loader return type mismatch

For HRL options, the loader contract is stricter:
- canonical `option_type` values (`block`, `alternation`, `heuristic`) must not include plugin fields
- `option_type: custom` must include both `plugin.loader_module` and `plugin.loader_function`
- legacy flat top-level `loader_module` or `loader_function` keys on option specs are rejected with a migration error

---

## 3) Options

Options have the same plugin seam as the other extensibility points, but they support two explicit modes.

### Canonical option mode

Use canonical option types when the package already provides the loader:

```yaml
options:
    - option_name: "A_10"
        option_type: "block"
        option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
        config_params_override:
            antibiotic: "A"
            duration: 10

    - option_name: "ALT_AB"
        option_type: "alternation"
        option_subconfig_file: "../option_types/alternation/alternation_option_default_config.yaml"
        config_params_override:
            sequence:
                - "A"
                - "B"
```

Canonical option types are exactly:
- `block`
- `alternation`
- `heuristic`

Do not add `plugin`, `loader_module`, or `loader_function` to canonical option specs.

### Custom option mode

Use the plugin seam when the option loader is project-specific:

```yaml
options:
    - option_name: "CUSTOM_WEIGHTED_HEURISTIC"
        option_type: "custom"
        option_subconfig_file: "../option_types/heuristic/my_custom_option_config.yaml"
        plugin:
            loader_module: "../option_types/heuristic/my_custom_option.py"
            loader_function: load_my_custom_option
        config_params_override:
            duration: 10
            risk_weight: 0.75
```

Custom option loader shape:

```python
from typing import Any, Dict

from abx_amr_simulator.options.base_option import OptionBase


def load_my_custom_option(config: Dict[str, Any]) -> OptionBase:
    return MyCustomOption(
        name=config["option_name"],
        duration=config["duration"],
    )
```

The loader receives the merged option config and must return an `OptionBase` instance.

Relative option plugin paths are resolved relative to the option library YAML file, not the umbrella config directory.

---

## 4) Subclassing Guide — `PatientGenerator`

### Base class import

```python
from abx_amr_simulator.core import PatientGeneratorBase
```

### Required contract

Implement:
- `sample(...)`
- `observe(...)`
- `obs_dim(...)`

Define:
- `PROVIDES_ATTRIBUTES` class variable

### Runnable example

```python
from typing import Any, Dict

from abx_amr_simulator.core import PatientGeneratorBase, PatientGenerator


class MyPatientGeneratorPlugin(PatientGeneratorBase):
    PROVIDES_ATTRIBUTES = list(PatientGenerator.PROVIDES_ATTRIBUTES)

    def __init__(self, config: Dict[str, Any]) -> None:
        self._delegate = PatientGenerator(config=config)
        self.visible_patient_attributes = list(self._delegate.visible_patient_attributes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def sample(self, n_patients, true_amr_levels, rng=None, **kwargs):
        return self._delegate.sample(
            n_patients=n_patients,
            true_amr_levels=true_amr_levels,
            rng=rng,
            **kwargs,
        )

    def observe(self, patients):
        return self._delegate.observe(patients=patients)

    def obs_dim(self, num_patients: int) -> int:
        return self._delegate.obs_dim(num_patients=num_patients)


def load_patient_generator_component(config: Dict[str, Any]) -> PatientGeneratorBase:
    return MyPatientGeneratorPlugin(config=config)
```

### YAML snippet

```yaml
patient_generator:
  visible_patient_attributes: [prob_infected]
  plugin:
    loader_module: "my_project/plugins/my_patient_generator_plugin.py"
    loader_function: load_patient_generator_component
```

### Full reference fixture

See `tests/integration/fixtures/custom_patient_generator_plugin.py`.

---

## 5) Subclassing Guide — `RewardCalculator`

### Base class import

```python
from abx_amr_simulator.core import RewardCalculatorBase
```

### Required contract

Implement:
- `calculate_reward(...)`

Define:
- `REQUIRED_PATIENT_ATTRS` class variable (must match patient attributes your reward logic expects)

### Runnable example

```python
from typing import Any, Dict, List

import numpy as np

from abx_amr_simulator.core import RewardCalculatorBase, RewardCalculator


class MyRewardCalculatorPlugin(RewardCalculatorBase):
    REQUIRED_PATIENT_ATTRS = list(RewardCalculator.REQUIRED_PATIENT_ATTRS)

    def __init__(self, config: Dict[str, Any]) -> None:
        self._delegate = RewardCalculator(config=config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def calculate_reward(
        self,
        patients: List[Any],
        actions: np.ndarray,
        antibiotic_names,
        visible_amr_levels,
        delta_visible_amr_per_antibiotic,
        **kwargs,
    ):
        return self._delegate.calculate_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=visible_amr_levels,
            delta_visible_amr_per_antibiotic=delta_visible_amr_per_antibiotic,
            **kwargs,
        )


def load_reward_calculator_component(config: Dict[str, Any]) -> RewardCalculatorBase:
    return MyRewardCalculatorPlugin(config=config)
```

### YAML snippet

```yaml
reward_calculator:
  lambda_weight: 0.5
  plugin:
    loader_module: "my_project.plugins.my_reward_calculator_plugin"
    loader_function: load_reward_calculator_component
```

### Full reference fixture

See `tests/integration/fixtures/custom_reward_calculator_plugin.py`.

---

## 6) Subclassing Guide — AMR Dynamics

### Base class import

```python
from abx_amr_simulator.core import AMRDynamicsBase
```

### Required contract

Implement:
- `step(doses: float) -> float`
- `reset(initial_level: float) -> None`

For this family, the loader must return:
- `Dict[str, AMRDynamicsBase]`

The dict keys must be antibiotic names matching `environment.antibiotics_AMR_dict` keys.

### Runnable example

```python
from typing import Any, Dict

from abx_amr_simulator.core import AMRDynamicsBase, AMR_LeakyBalloon


class MyAMRDynamicsPlugin(AMRDynamicsBase):
    NAME = "my_amr_dynamics_plugin"

    def __init__(self, params: Dict[str, Any]) -> None:
        self._delegate = AMR_LeakyBalloon(
            leak=params["leak"],
            flatness_parameter=params["flatness_parameter"],
            permanent_residual_volume=params["permanent_residual_volume"],
            initial_amr_level=params["initial_amr_level"],
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def step(self, doses: float) -> float:
        return self._delegate.step(doses=doses)

    def reset(self, initial_level: float) -> None:
        self._delegate.reset(initial_amr_level=initial_level)


def load_amr_dynamics_component(config: Dict[str, Any]) -> Dict[str, AMRDynamicsBase]:
    antibiotics_amr_dict = config["antibiotics_AMR_dict"]
    return {
        antibiotic_name: MyAMRDynamicsPlugin(params=antibiotic_params)
        for antibiotic_name, antibiotic_params in antibiotics_amr_dict.items()
    }
```

### YAML snippet

```yaml
environment:
  antibiotics_AMR_dict:
    A:
      leak: 0.05
      flatness_parameter: 1.0
      permanent_residual_volume: 0.0
      initial_amr_level: 0.0

amr_dynamics:
  plugin:
    loader_module: "my_project/plugins/my_amr_dynamics_plugin.py"
    loader_function: load_amr_dynamics_component
```

### Full reference fixture

See `tests/integration/fixtures/custom_amr_dynamics_plugin.py`.

---

## 7) Design Pattern: Splitting YAML Config from Python Object Construction

### The problem

The plugin seam works well when every parameter your custom component needs can be expressed as a YAML scalar (numbers, strings, booleans, lists). It breaks down when your component requires something that cannot be serialized into YAML — for example:

- A **callable** (e.g., a trajectory function `f(timestep, current_params) -> updated_params`)
- A **nested custom class** (e.g., a mechanistic population model that is itself a user-defined Python object)
- Any object that requires non-trivial Python logic to construct

Attempting to solve this by embedding import paths in YAML and resolving them in the factory adds fragility (import paths break on refactoring) and couples the factory to arbitrary user code.

### The recommended split

**YAML carries scalar parameters only.** The loader function — which is already arbitrary Python — handles all class instantiation, callable construction, and object wiring.

```
YAML config:        scalar parameters (numbers, strings, flags)
Loader function:    Python construction logic (imports, class instantiation, wiring)
```

The loader function receives the merged YAML config as a plain dict. It is free to import any workspace module, instantiate any class, define or import any callable, and compose objects in whatever way is needed. It then returns the fully constructed component.

### Example: component that requires a callable

Suppose you want a custom `PatientGenerator` whose mean `prob_infected` drifts linearly over the course of an episode. The drift slope cannot go in YAML as a callable, but the scalar parameters (starting probability and slope) can.

**Custom component (`my_experiment/components/drifting_pg.py`):**
```python
import math
from typing import Any, Callable, Dict, List

import numpy as np

from abx_amr_simulator.core import PatientGenerator, PatientGeneratorBase
from abx_amr_simulator.core.types import Patient


class DriftingPatientGenerator(PatientGeneratorBase):
    """PatientGenerator whose mean prob_infected shifts each timestep via a callable."""

    PROVIDES_ATTRIBUTES = list(PatientGenerator.PROVIDES_ATTRIBUTES)

    def __init__(
        self,
        base_config: Dict[str, Any],
        prob_modifier: Callable[[int], float],
    ) -> None:
        # prob_modifier(timestep) -> new mean prob_infected for that step
        self._prob_modifier = prob_modifier
        self._current_timestep = 0
        self._base_config = base_config
        self._delegate = PatientGenerator(config=base_config)
        self.visible_patient_attributes = list(self._delegate.visible_patient_attributes)

    def advance_episode_step(self, timestep: int) -> None:
        self._current_timestep = timestep
        # Rebuild delegate config with updated prob_infected mean
        new_prob = float(np.clip(self._prob_modifier(timestep), 0.0, 1.0))
        updated_config = dict(self._base_config)
        updated_config["prob_infected"] = dict(self._base_config["prob_infected"])
        updated_config["prob_infected"]["prob_dist"] = {
            "type": "constant",
            "value": new_prob,
        }
        self._delegate = PatientGenerator(config=updated_config)

    def sample(self, n_patients, true_amr_levels, rng=None, **kwargs) -> List[Patient]:
        return self._delegate.sample(
            n_patients=n_patients,
            true_amr_levels=true_amr_levels,
            rng=rng,
            **kwargs,
        )

    def observe(self, patients):
        return self._delegate.observe(patients=patients)

    def obs_dim(self, num_patients: int) -> int:
        return self._delegate.obs_dim(num_patients=num_patients)
```

**YAML config:**
```yaml
patient_generator:
  visible_patient_attributes: [prob_infected]
  prob_infected:
    prob_dist:
      type: constant
      value: 0.3       # starting probability
    obs_bias_multiplier: 1.0
    obs_noise_one_std_dev: 0.1
    obs_noise_std_dev_fraction: 0.5
    clipping_bounds: [0.0, 1.0]
  drift_slope: 0.0002  # scalar parameter for the callable
  plugin:
    loader_module: "my_experiment/setup/pg_loader.py"
    loader_function: load_patient_generator_component
```

**Loader function (`my_experiment/setup/pg_loader.py`):**
```python
from typing import Any, Dict
from my_experiment.components.drifting_pg import DriftingPatientGenerator


def load_patient_generator_component(config: Dict[str, Any]) -> DriftingPatientGenerator:
    slope = config["drift_slope"]
    start_prob = config["prob_infected"]["prob_dist"]["value"]

    # Construct the callable from scalar YAML parameters — no callable in YAML needed
    def linear_drift(timestep: int) -> float:
        return start_prob + slope * timestep

    return DriftingPatientGenerator(base_config=config, prob_modifier=linear_drift)
```

The key point: `drift_slope` and `start_prob` are plain numbers in YAML. The loader function constructs the callable `linear_drift` from those numbers and passes it to `DriftingPatientGenerator`. A different experiment could swap in a sinusoidal or step-function callable by changing only the loader function, without touching the YAML schema.

### Example: component that requires a nested custom object

Suppose you want a custom `AMRDynamicsBase` implementation whose leak rate is modulated by a user-defined `LeakSchedule` helper object — a class with its own parameters that cannot be expressed as a single scalar in YAML.

**Custom component (`my_experiment/components/scheduled_amr.py`):**
```python
from typing import Any, Dict, List

from abx_amr_simulator.core import AMR_LeakyBalloon, AMRDynamicsBase


class LeakSchedule:
    """Modulates leak rate linearly until it reaches a floor value."""

    def __init__(self, initial_leak: float, floor_leak: float, decay_per_step: float) -> None:
        self.initial_leak = initial_leak
        self.floor_leak = floor_leak
        self.decay_per_step = decay_per_step

    def get_leak(self, timestep: int) -> float:
        return max(self.floor_leak, self.initial_leak - self.decay_per_step * timestep)


class ScheduledLeakAMR(AMRDynamicsBase):
    """AMR dynamics where the leak rate decreases over time per a LeakSchedule."""

    NAME = "scheduled_leak_amr"

    def __init__(self, params: Dict[str, Any], schedule: LeakSchedule) -> None:
        self._schedule = schedule
        self._timestep = 0
        self._balloon = AMR_LeakyBalloon(
            leak=schedule.get_leak(0),
            flatness_parameter=params["flatness_parameter"],
            permanent_residual_volume=params["permanent_residual_volume"],
            initial_amr_level=params["initial_amr_level"],
        )

    def advance_episode_step(self, timestep: int) -> None:
        self._timestep = timestep
        self._balloon.leak = self._schedule.get_leak(timestep)

    def step(self, doses: float) -> float:
        return self._balloon.step(doses=doses)

    def reset(self, initial_level: float) -> None:
        self._balloon.reset(initial_amr_level=initial_level)
        self._timestep = 0
        self._balloon.leak = self._schedule.get_leak(0)
```

**YAML config:**
```yaml
environment:
  antibiotics_AMR_dict:
    A:
      flatness_parameter: 1.0
      permanent_residual_volume: 0.0
      initial_amr_level: 0.0
      initial_leak: 0.15
      floor_leak: 0.03
      decay_per_step: 0.0001

amr_dynamics:
  plugin:
    loader_module: "my_experiment/setup/amr_loader.py"
    loader_function: load_amr_dynamics_component
```

**Loader function (`my_experiment/setup/amr_loader.py`):**
```python
from typing import Any, Dict

from abx_amr_simulator.core import AMRDynamicsBase
from my_experiment.components.scheduled_amr import LeakSchedule, ScheduledLeakAMR


def load_amr_dynamics_component(config: Dict[str, Any]) -> Dict[str, AMRDynamicsBase]:
    antibiotics_amr_dict = config["antibiotics_AMR_dict"]
    result = {}
    for abx_name, params in antibiotics_amr_dict.items():
        # Construct the nested helper object from scalar YAML parameters
        schedule = LeakSchedule(
            initial_leak=params["initial_leak"],
            floor_leak=params["floor_leak"],
            decay_per_step=params["decay_per_step"],
        )
        result[abx_name] = ScheduledLeakAMR(params=params, schedule=schedule)
    return result
```

Again, `LeakSchedule` is a Python object that cannot be serialized to YAML. Its constructor parameters (`initial_leak`, `floor_leak`, `decay_per_step`) are plain scalars that live in YAML. The loader function instantiates `LeakSchedule` from those scalars and passes it to `ScheduledLeakAMR`. The canonical factory never needs to know `LeakSchedule` exists.

### When to apply this pattern

Apply the split whenever you would otherwise need to put a Python import path, a class name, or any non-scalar value into a YAML config in order to drive object construction. The rule is: **YAML is for values, Python is for types and wiring.**

---

## 8) Filesystem vs Importable Module Paths

Use filesystem paths when your plugin module lives next to your experiment/config files and is not installed as a package.

Use importable module paths when your plugin code is in an installed Python package.

If you provide a relative filesystem path, it is resolved against `_umbrella_config_dir`.

For custom HRL options, relative filesystem paths are resolved against the option library YAML file.

---

## 8) Troubleshooting

Common failures and fixes:
- **`missing required key 'plugin.loader_module'`**: add `plugin.loader_module` under the correct component section.
- **Module import fails**: verify import path spelling or filesystem path existence.
- **Loader function not found/non-callable**: confirm the function name and ensure it is defined as a Python function.
- **Invalid return type**: make sure your loader returns the expected base type (`PatientGeneratorBase`, `RewardCalculatorBase`, or `Dict[str, AMRDynamicsBase]`).
- **AMR dynamics dict value type errors**: ensure every dict value is an `AMRDynamicsBase` instance.
- **Canonical option with `plugin` fields**: remove the plugin block and keep only the canonical option keys.
- **`custom` option missing `plugin.loader_function`**: add both required plugin fields under the nested `plugin:` block.
- **Legacy flat option `loader_module` key**: migrate the option to `option_type: custom` with nested `plugin.loader_module` and `plugin.loader_function`.

Quick diagnostic checklist:
1. Confirm plugin keys are nested under the right component or, for options, under the right option spec.
2. Confirm loader signature is `loader(config: Dict[str, Any]) -> expected_type`.
3. Confirm returned object(s) subclass the expected base class.
4. For AMR dynamics plugins, confirm dict keys match antibiotic names from `environment.antibiotics_AMR_dict`.
5. Re-run from the same config location if using relative filesystem paths.
