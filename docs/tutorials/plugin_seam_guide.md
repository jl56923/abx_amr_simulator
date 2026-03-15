# Plugin Seam Guide

**Goal**: Configure and load custom `PatientGenerator`, `RewardCalculator`, and AMR dynamics components using the package-standard plugin seam.

**Prerequisites**: Familiarity with YAML experiment configs and basic subclassing in Python.

---

## 1) Overview

The plugin seam is a config-driven way to replace core simulator components without changing `train.py` or `tune.py` code paths. It exists so users can subclass core components and swap them in using YAML only.

Design philosophy:
- **Config-driven**: plugin selection happens in config, not in wrapper scripts.
- **Opt-in**: canonical behavior is unchanged unless plugin keys are provided.
- **Fail-loud**: invalid plugin config raises immediately before training/tuning starts.
- **Symmetric with options**: this uses the same loader-module pattern originally introduced for HRL options.

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

---

## 3) Subclassing Guide — `PatientGenerator`

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

## 4) Subclassing Guide — `RewardCalculator`

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

## 5) Subclassing Guide — AMR Dynamics

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

## 6) Filesystem vs Importable Module Paths

Use filesystem paths when your plugin module lives next to your experiment/config files and is not installed as a package.

Use importable module paths when your plugin code is in an installed Python package.

If you provide a relative filesystem path, it is resolved against `_umbrella_config_dir`.

---

## 7) Troubleshooting

Common failures and fixes:
- **`missing required key 'plugin.loader_module'`**: add `plugin.loader_module` under the correct component section.
- **Module import fails**: verify import path spelling or filesystem path existence.
- **Loader function not found/non-callable**: confirm the function name and ensure it is defined as a Python function.
- **Invalid return type**: make sure your loader returns the expected base type (`PatientGeneratorBase`, `RewardCalculatorBase`, or `Dict[str, AMRDynamicsBase]`).
- **AMR dynamics dict value type errors**: ensure every dict value is an `AMRDynamicsBase` instance.

Quick diagnostic checklist:
1. Confirm plugin keys are nested under the right component (`patient_generator`, `reward_calculator`, or `amr_dynamics`).
2. Confirm loader signature is `loader(config: Dict[str, Any]) -> expected_type`.
3. Confirm returned object(s) subclass the expected base class.
4. For AMR dynamics plugins, confirm dict keys match antibiotic names from `environment.antibiotics_AMR_dict`.
5. Re-run from the same config location if using relative filesystem paths.
