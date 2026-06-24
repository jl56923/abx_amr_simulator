# Tutorial 11: Component Subclassing

**Goal**: Learn how to subclass core components (`RewardCalculator`, `PatientGenerator`, `AMRDynamicsBase`, and `ABXAMREnv`) to implement custom behavior beyond what's achievable via configuration alone.

**Prerequisites**: Completed Tutorials 1-3, strong Python skills, familiarity with the package architecture

**Note**: This is an advanced tutorial for users who need customization beyond what YAML configuration provides. Most users (~95%) won't need this.

---

## Overview

The `abx_amr_simulator` package is designed to be highly configurable via YAML files, but sometimes you need to implement truly custom behavior that can't be expressed in configuration. This tutorial shows:

1. **When to subclass** vs when to configure
2. **RewardCalculator + PatientGenerator coupling** — Why these must be subclassed together
3. **AMRDynamicsBase** — Replacing the per-antibiotic resistance dynamics model
4. **ABXAMREnv subclassing** — Extending the environment for structural changes
5. **Integration patterns** — Wiring custom components with the training pipeline

> **See also**: This tutorial focuses on writing the Python subclass. If you want to wire your custom component into the config-driven training pipeline (`train.py` / `tune.py`) without modifying any script files, the [Plugin Seam Guide](plugin_seam_guide.md) explains the YAML-driven loader mechanism, path resolution, and troubleshooting.

---

## When to Subclass vs Configure

### Use Configuration When:
- Changing numerical parameters (lambda weight, AMR leak rate, patient distributions)
- Selecting from existing options (agent algorithm, action mode)
- Toggling features (observation noise, AMR update frequency)

### Subclass When:
- Implementing a novel reward function (beyond weighted clinical + community AMR)
- Creating new patient attributes (beyond the 6 standard attributes)
- Replacing the AMR dynamics model (beyond the leaky balloon)
- Adding custom observation transformations or action spaces

**Rule of thumb**: If your change can be expressed as "use this number instead of that number", use configs. If it's "do this fundamentally different thing", subclass.

---

## Subclassing RewardCalculator and PatientGenerator Together

### Why These Are Coupled

The `RewardCalculator` expects specific patient attributes to compute rewards. If you add new attributes to `PatientGenerator`, the `RewardCalculator` must know about them. Similarly, if your reward function uses new patient features, `PatientGenerator` must provide them.

**Example coupling**:
- PatientGenerator defines `prob_infected`, `benefit_value_multiplier`, etc.
- RewardCalculator uses these attributes in `REQUIRED_PATIENT_ATTRS`
- Environment validates that PG provides exactly what RC requires

**Breaking this coupling causes runtime errors.**

### Example: Adding a "severity" Attribute

Let's add a new patient attribute `infection_severity` ∈ [0,1] that modulates treatment benefit.

#### Step 1: Subclass PatientGenerator

```python
# my_first_project/custom_components/severity_patient_generator.py

from dataclasses import dataclass
from typing import List
import numpy as np
from abx_amr_simulator.core.types import Patient
from abx_amr_simulator.core.base_patient_generator import BasePatientGenerator


@dataclass
class PatientWithSeverity(Patient):
    """Extended patient with infection severity attribute."""
    infection_severity: float = 0.5  # New attribute
    infection_severity_obs: float = 0.5  # Observed version


class SeverityPatientGenerator(BasePatientGenerator):
    """Patient generator that includes infection severity."""
    
    def __init__(
        self,
        num_patients: int,
        visible_patient_attributes: List[str],
        seed: int = None,
        # Standard patient attribute configs...
        prob_infected_dist: dict = None,
        benefit_value_multiplier_dist: dict = None,
        # ... (include all standard attributes)
        
        # New severity config
        severity_dist: dict = None,
        severity_obs_noise: float = 0.0,
        severity_obs_bias: float = 0.0,
    ):
        super().__init__(
            num_patients=num_patients,
            visible_patient_attributes=visible_patient_attributes,
            seed=seed,
        )
        
        # Store standard attribute configs
        # ... (same as base PatientGenerator)
        
        # New severity config
        self.severity_dist = severity_dist or {"type": "constant", "value": 0.5}
        self.severity_obs_noise = severity_obs_noise
        self.severity_obs_bias = severity_obs_bias
    
    def sample(self) -> List[PatientWithSeverity]:
        """Sample a batch of patients with severity attribute."""
        patients = []
        
        for _ in range(self.num_patients):
            # Sample standard attributes (prob_infected, benefit_value_multiplier, etc.)
            # ... (same as base PatientGenerator)
            
            # Sample new severity attribute
            if self.severity_dist["type"] == "constant":
                severity = self.severity_dist["value"]
            elif self.severity_dist["type"] == "gaussian":
                severity = self.rng.normal(
                    self.severity_dist["mean"],
                    self.severity_dist["std"]
                )
                severity = np.clip(severity, 0.0, 1.0)
            
            # Apply observation noise/bias
            severity_obs = severity + self.rng.normal(0, self.severity_obs_noise) + self.severity_obs_bias
            severity_obs = np.clip(severity_obs, 0.0, 1.0)
            
            # Create patient with new attribute
            patient = PatientWithSeverity(
                # Standard attributes...
                prob_infected=prob_infected,
                prob_infected_obs=prob_infected_obs,
                # ... (include all standard attributes)
                
                # New attribute
                infection_severity=severity,
                infection_severity_obs=severity_obs,
            )
            patients.append(patient)
        
        return patients
    
    def observe(self, patients: List[PatientWithSeverity]) -> np.ndarray:
        """Extract observed features including severity."""
        obs_list = []
        for patient in patients:
            patient_obs = []
            for attr in self.visible_patient_attributes:
                # Support standard attributes
                if attr in ['prob_infected', 'benefit_value_multiplier', ...]:
                    patient_obs.append(getattr(patient, f"{attr}_obs"))
                # Support new severity attribute
                elif attr == 'infection_severity':
                    patient_obs.append(patient.infection_severity_obs)
                else:
                    raise ValueError(f"Unknown patient attribute: {attr}")
            obs_list.extend(patient_obs)
        return np.array(obs_list, dtype=np.float32)
    
    def obs_dim(self, num_patients: int) -> int:
        """Compute observation dimension (supports new attribute)."""
        return num_patients * len(self.visible_patient_attributes)
```

#### Step 2: Subclass RewardCalculator

```python
# my_first_project/custom_components/severity_reward_calculator.py

from typing import List
import numpy as np
from abx_amr_simulator.core.base_reward_calculator import BaseRewardCalculator
from .severity_patient_generator import PatientWithSeverity


class SeverityRewardCalculator(BaseRewardCalculator):
    """Reward calculator that uses infection severity to modulate benefits."""
    
    # Declare required attributes (must match PatientGenerator output)
    REQUIRED_PATIENT_ATTRS = [
        'prob_infected',
        'is_currently_infected',
        'currently_infected_abx_sensitivity',
        'benefit_value_multiplier',
        'benefit_probability_multiplier',
        'failure_value_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob',
        'infection_severity',  # New required attribute
    ]
    
    def __init__(
        self,
        num_abx: int,
        lambda_weight: float,
        abx_clinical_reward_penalties_info_dict: dict,
        seed: int = None,
    ):
        super().__init__(
            num_abx=num_abx,
            lambda_weight=lambda_weight,
            abx_clinical_reward_penalties_info_dict=abx_clinical_reward_penalties_info_dict,
            seed=seed,
        )
    
    def calculate_individual_patient_rewards(
        self,
        patients: List[PatientWithSeverity],
        actions: np.ndarray,
        current_AMR_levels: np.ndarray,
        delta_AMR_levels: np.ndarray,
    ) -> np.ndarray:
        """Compute individual rewards with severity modulation."""
        num_patients = len(patients)
        individual_rewards = np.zeros(num_patients, dtype=np.float32)
        
        for i, patient in enumerate(patients):
            # Extract severity
            severity = patient.infection_severity
            
            # Get action (which antibiotic prescribed, or -1 for no treatment)
            prescribed_abx = np.where(actions[i] == 1)[0]
            
            if len(prescribed_abx) == 0:
                # No treatment
                # Natural recovery (less likely with higher severity)
                recovery_prob = patient.recovery_without_treatment_prob * (1.0 - severity)
                if patient.is_currently_infected and self.rng.random() < recovery_prob:
                    individual_rewards[i] += 5.0  # Small reward for natural recovery
                else:
                    # Clinical failure (worse with higher severity)
                    individual_rewards[i] -= 10.0 * (1.0 + severity)
            else:
                abx_idx = prescribed_abx[0]
                
                # Check if antibiotic is effective
                is_sensitive = patient.currently_infected_abx_sensitivity[abx_idx] > 0.5
                
                if is_sensitive:
                    # Treatment success (benefit increases with severity)
                    benefit = 10.0 * patient.benefit_value_multiplier * (1.0 + severity)
                    individual_rewards[i] += benefit
                else:
                    # Treatment failure (penalty increases with severity)
                    penalty = -10.0 * patient.failure_value_multiplier * (1.0 + severity)
                    individual_rewards[i] += penalty
                
                # Adverse effects (risk increases with severity)
                adverse_prob = 0.1 * (1.0 + 0.5 * severity)
                if self.rng.random() < adverse_prob:
                    individual_rewards[i] -= 2.0
            
        return individual_rewards
```

#### Step 3: Create Factory Functions

```python
# my_first_project/custom_components/factories.py

from pathlib import Path
from abx_amr_simulator.utils import load_config
from .severity_patient_generator import SeverityPatientGenerator
from .severity_reward_calculator import SeverityRewardCalculator


def create_severity_patient_generator(config: dict, seed: int = None) -> SeverityPatientGenerator:
    """Factory for custom patient generator."""
    pg_config = config.get('patient_generator', {})
    
    return SeverityPatientGenerator(
        num_patients=pg_config['num_patients'],
        visible_patient_attributes=pg_config['visible_patient_attributes'],
        seed=seed,
        # Standard attributes
        prob_infected_dist=pg_config.get('prob_infected_dist', {"type": "constant", "value": 0.3}),
        # ... (all standard attributes)
        
        # New severity attribute
        severity_dist=pg_config.get('severity_dist', {"type": "gaussian", "mean": 0.5, "std": 0.2}),
        severity_obs_noise=pg_config.get('severity_obs_noise', 0.0),
        severity_obs_bias=pg_config.get('severity_obs_bias', 0.0),
    )


def create_severity_reward_calculator(config: dict, seed: int = None) -> SeverityRewardCalculator:
    """Factory for custom reward calculator."""
    rc_config = config.get('reward_calculator', {})
    
    return SeverityRewardCalculator(
        num_abx=len(config['environment']['antibiotic_names']),
        lambda_weight=rc_config['lambda_weight'],
        abx_clinical_reward_penalties_info_dict=rc_config['abx_clinical_reward_penalties_info_dict'],
        seed=seed,
    )
```

#### Step 4: Integrate with Training Pipeline

For one-off scripts, instantiate your components directly and pass them to the environment:

```python
# my_first_project/train_with_custom_components.py

from abx_amr_simulator.utils import load_config
from abx_amr_simulator.core import ABXAMREnv
from custom_components.factories import create_severity_patient_generator, create_severity_reward_calculator

# Load config
config = load_config('experiments/configs/umbrella_configs/base_experiment.yaml')

# Create custom components
seed = config['training']['seed']
pg = create_severity_patient_generator(config, seed=seed)
rc = create_severity_reward_calculator(config, seed=seed)

# Create environment (standard)
env = ABXAMREnv(
    patient_generator=pg,
    reward_calculator=rc,
    # ... (standard environment params from config)
)

# Train agent (standard)
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

> For config-driven pipelines (`train.py` / `tune.py`), use the `patient_generator.plugin` and `reward_calculator.plugin` YAML keys instead of modifying scripts. See [Plugin Seam Guide §4–5](plugin_seam_guide.md).

---

## Subclassing AMRDynamicsBase

`AMRDynamicsBase` is the extension point for replacing the per-antibiotic resistance dynamics model. The environment holds one dynamics instance per antibiotic (a `Dict[str, AMRDynamicsBase]`); each instance encapsulates how resistance evolves under prescribing pressure for that antibiotic. The default implementation is `AMR_LeakyBalloon`.

Subclass `AMRDynamicsBase` when you want to replace the *mathematical model* of how resistance rises and falls. This is the right level of abstraction for most dynamics customizations — you get a clean, isolated unit that the rest of the environment doesn't need to know about.

### Required contract

```python
from abx_amr_simulator.core import AMRDynamicsBase
```

Implement:
- `step(doses: float) -> float` — advance resistance by one timestep given antibiotic exposure; return the new AMR level
- `reset(initial_level: float) -> None` — restore to a given resistance level at episode start

Set:
- `NAME: str` — a string identifier for this dynamics model

### Example: Exponential Decay Model

The leaky balloon uses a logistic pressure function. Here's a simpler alternative: resistance grows proportionally to antibiotic dose and decays exponentially when pressure is removed.

```python
# my_first_project/custom_components/exponential_decay_amr.py

import numpy as np
from typing import Any, Dict
from abx_amr_simulator.core import AMRDynamicsBase


class ExponentialDecayAMR(AMRDynamicsBase):
    """AMR dynamics where resistance decays exponentially without antibiotic pressure.

    AMR level evolves as:
        level_{t+1} = level_t * exp(-decay_rate) + pressure_factor * doses
    clamped to [0, 1].
    """

    NAME = "exponential_decay"

    def __init__(self, decay_rate: float, pressure_factor: float, initial_amr_level: float = 0.0):
        self._decay_rate = decay_rate
        self._pressure_factor = pressure_factor
        self._level = float(np.clip(initial_amr_level, 0.0, 1.0))

    def step(self, doses: float) -> float:
        """Advance AMR by one timestep under the given antibiotic dose."""
        self._level = self._level * np.exp(-self._decay_rate) + self._pressure_factor * doses
        self._level = float(np.clip(self._level, 0.0, 1.0))
        return self._level

    def reset(self, initial_level: float) -> None:
        """Reset resistance to the given level at episode start."""
        self._level = float(np.clip(initial_level, 0.0, 1.0))
```

### Instantiating one per antibiotic

The environment expects a `Dict[str, AMRDynamicsBase]`, one entry per antibiotic name. For direct Python usage:

```python
# my_first_project/train_with_custom_amr.py

from abx_amr_simulator.utils import load_config
from abx_amr_simulator.core import ABXAMREnv
from custom_components.exponential_decay_amr import ExponentialDecayAMR

config = load_config('experiments/configs/umbrella_configs/base_experiment.yaml')

amr_dynamics = {
    antibiotic_name: ExponentialDecayAMR(
        decay_rate=params["decay_rate"],
        pressure_factor=params["pressure_factor"],
        initial_amr_level=params.get("initial_amr_level", 0.0),
    )
    for antibiotic_name, params in config["environment"]["antibiotics_AMR_dict"].items()
}

env = ABXAMREnv(
    amr_dynamics=amr_dynamics,
    # ... other standard env params from config
)
```

> For config-driven pipelines, use the `amr_dynamics.plugin` YAML key. See [Plugin Seam Guide §6](plugin_seam_guide.md).

### Testing your dynamics

```python
# tests/custom/test_exponential_decay_amr.py

from custom_components.exponential_decay_amr import ExponentialDecayAMR


def test_exponential_decay_amr_rises_under_pressure():
    amr = ExponentialDecayAMR(decay_rate=0.1, pressure_factor=0.3, initial_amr_level=0.0)
    level = amr.step(doses=1.0)
    assert level > 0.0


def test_exponential_decay_amr_decays_without_pressure():
    amr = ExponentialDecayAMR(decay_rate=0.1, pressure_factor=0.3, initial_amr_level=0.8)
    level = amr.step(doses=0.0)
    assert level < 0.8


def test_exponential_decay_amr_reset():
    amr = ExponentialDecayAMR(decay_rate=0.1, pressure_factor=0.3, initial_amr_level=0.8)
    amr.step(doses=1.0)
    amr.reset(initial_level=0.3)
    assert abs(amr._level - 0.3) < 1e-6


def test_exponential_decay_amr_clamps_to_unit_interval():
    amr = ExponentialDecayAMR(decay_rate=0.0, pressure_factor=1.0, initial_amr_level=0.9)
    level = amr.step(doses=10.0)
    assert 0.0 <= level <= 1.0
```

---

## Subclassing ABXAMREnv

Subclass `ABXAMREnv` when you need **structural changes to the environment itself** — changes that cut across components and can't be handled by swapping a single component:

- Modifying the **observation space** (e.g., adding new observation dimensions beyond patients and AMR levels)
- Modifying the **action space** (e.g., combination therapy actions)
- Adding **env-level state** that persists across timesteps and doesn't belong to a single component (e.g., outbreak tracking, seasonal pressure effects)
- Changing the **episode step or reset logic** in ways that span multiple components

If your goal is custom AMR dynamics, use `AMRDynamicsBase` instead — it's the correct, isolated extension point for that and avoids coupling to private environment internals.

### Example: Adding Outbreak Tracking to the Observation

Suppose you want the agent to observe a normalised cumulative treatment failure count that resets each episode. This is env-level state — it aggregates information across all patients and timesteps — so it belongs in `ABXAMREnv`:

```python
# my_first_project/custom_components/outbreak_tracking_env.py

import numpy as np
from gymnasium import spaces
from abx_amr_simulator.core import ABXAMREnv


class OutbreakTrackingEnv(ABXAMREnv):
    """Environment that appends a normalised cumulative failure count to the observation."""

    def __init__(self, max_steps: int, **kwargs):
        super().__init__(**kwargs)
        self._max_steps = max_steps
        self._cumulative_failures = 0

        # Extend the observation space by one dimension
        base_low = self.observation_space.low
        base_high = self.observation_space.high
        self.observation_space = spaces.Box(
            low=np.append(base_low, 0.0),
            high=np.append(base_high, 1.0),
            dtype=np.float32,
        )

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        failure_rate = self._cumulative_failures / self._max_steps
        return np.append(obs, float(np.clip(failure_rate, 0.0, 1.0)))

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self._cumulative_failures += info.get("num_clinical_failures", 0)
        return self._augment_obs(obs), reward, done, truncated, info

    def reset(self, **kwargs):
        self._cumulative_failures = 0
        obs, info = super().reset(**kwargs)
        return self._augment_obs(obs), info
```

---

## Testing Custom Components

Always write unit tests for custom components. Test each component in isolation before testing the full environment:

```python
# tests/custom/test_severity_components.py

import pytest
from custom_components.severity_patient_generator import SeverityPatientGenerator
from custom_components.severity_reward_calculator import SeverityRewardCalculator


def test_severity_patient_generator():
    pg = SeverityPatientGenerator(
        num_patients=10,
        visible_patient_attributes=['prob_infected', 'infection_severity'],
        seed=42,
        severity_dist={"type": "constant", "value": 0.8},
    )
    
    patients = pg.sample()
    assert len(patients) == 10
    assert all(hasattr(p, 'infection_severity') for p in patients)
    assert all(0.0 <= p.infection_severity <= 1.0 for p in patients)


def test_severity_reward_calculator():
    rc = SeverityRewardCalculator(
        num_abx=2,
        lambda_weight=0.5,
        abx_clinical_reward_penalties_info_dict={...},
        seed=42,
    )
    
    from custom_components.severity_patient_generator import PatientWithSeverity
    patients = [
        PatientWithSeverity(
            prob_infected=0.9,
            is_currently_infected=True,
            infection_severity=0.8,  # High severity
            # ... (all required attributes)
        )
    ]
    
    actions = np.array([[1, 0]])  # Prescribe antibiotic A
    rewards = rc.calculate_individual_patient_rewards(
        patients, actions, current_AMR_levels=np.array([0.2, 0.3]), delta_AMR_levels=np.array([0.01, 0.0])
    )
    
    assert len(rewards) == 1
    # Verify reward magnitude increases with severity (compared to baseline)
```

---

## Best Practices

1. **Maintain consistency**: If you add a patient attribute, update both PG and RC
2. **Test thoroughly**: Unit tests + integration tests with full environment
3. **Document coupling**: Write clear docstrings explaining dependencies
4. **Avoid breaking interfaces**: Subclass, don't modify base classes
5. **Use factories**: Create factory functions for clean instantiation

---

## What's Next?

✅ You've learned to subclass core components!

For advanced topics:
- **Config-driven wiring**: See the [Plugin Seam Guide](plugin_seam_guide.md) to register custom components in YAML without modifying training scripts. It also covers the design pattern for components that require non-serializable constructor arguments (callables, nested objects).
- Study the source code in `src/abx_amr_simulator/core/`
- Review existing subclassing examples in `tests/unit/core/`
- Consult the architecture diagrams in `.github/copilot-instructions.md`

---

## Key Takeaways

1. **Config first**: Only subclass when configuration isn't sufficient
2. **RewardCalculator + PatientGenerator are coupled**: Must be subclassed together
3. **`AMRDynamicsBase` for dynamics**: One instance per antibiotic; implement `step()` and `reset()`
4. **`ABXAMREnv` for structural changes**: Observation/action space, env-level state, episode logic — not for per-antibiotic dynamics
5. **Declare required attributes**: Use `REQUIRED_PATIENT_ATTRS` for RC validation
6. **Plugin seam for config pipelines**: See [Plugin Seam Guide](plugin_seam_guide.md) to wire components via YAML
7. **Test rigorously**: Custom components need thorough unit tests
