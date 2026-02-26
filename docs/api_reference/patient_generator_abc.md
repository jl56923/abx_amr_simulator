# PatientGeneratorBase API Reference

## Overview

`PatientGeneratorBase` is the abstract base class for all patient population generation implementations. It defines the contract for creating heterogeneous patient cohorts and exposing their attributes to agents.

**Location**: `abx_amr_simulator.core.PatientGeneratorBase`

**Purpose**: Subclasses implement custom patient population sampling strategies and observation interfaces that agents use to make prescribing decisions.

## Class Definition

```python
from abc import ABC, abstractmethod
from typing import List, ClassVar, Any, Dict
import numpy as np

class PatientGeneratorBase(ABC):
    """
    Abstract base class for patient generation.
    
    Subclasses must:
    1. Define PROVIDES_ATTRIBUTES class constant listing all Patient attributes
    2. Implement sample() method to generate patient populations
    3. Implement observe() method to extract visible features
    4. Implement obs_dim() method to compute observation dimension
    """
    
    # Class constant declaring which attributes generated Patients will have
    PROVIDES_ATTRIBUTES: ClassVar[List[str]] = []

    # Instance attribute declaring which patient attributes are visible in observations
    visible_patient_attributes: List[str]
    
    @abstractmethod
    def sample(self, n: int, true_amr_levels: Dict[str, float], rng: np.random.Generator) -> List[Any]:
        """Generate n patient objects with heterogeneous attributes."""
        pass

    @abstractmethod
    def observe(self, patients: List[Any]) -> np.ndarray:
        """Project patients to observed features."""
        pass

    @abstractmethod
    def obs_dim(self, num_patients: int) -> int:
        """Compute observation dimension for this generator."""
        pass
```

## Required Class Constant

### `PROVIDES_ATTRIBUTES: ClassVar[List[str]]`

**Purpose**: Declare which Patient attributes this generator creates.

**Type**: `ClassVar[List[str]]` — a list of attribute name strings

**Validation**: At environment initialization (via `validate_compatibility()`), the system checks that all attributes required by `RewardCalculator.REQUIRED_PATIENT_ATTRS` are in `PROVIDES_ATTRIBUTES`.

**Examples**:

```python
# Minimal: constant population (no heterogeneity)
class HomogeneousPatientGenerator(PatientGeneratorBase):
    PROVIDES_ATTRIBUTES = []  # No heterogeneous attributes
    
    def sample(self, n, true_amr_levels, rng):
        return [Patient(...) for _ in range(n)]

# Comprehensive: full heterogeneity
class HeterogeneousPatientGenerator(PatientGeneratorBase):
    PROVIDES_ATTRIBUTES = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob'
    ]
    
    def sample(self, n, true_amr_levels, rng):
        patients = []
        for _ in range(n):
            # Sample each attribute
            patients.append(Patient(...))
        return patients
```

## Required Instance Attribute

### `visible_patient_attributes: List[str]`

**Purpose**: Multi-instance attribute specifying which attributes from `PROVIDES_ATTRIBUTES` are observable by the agent.

**Type**: `List[str]` — subset of `PROVIDES_ATTRIBUTES`

**Set during initialization**: This is a configuration choice—agents never see all attributes (some represent hidden true values).

**Usage**:
```python
class HeterogeneousPatientGenerator(PatientGeneratorBase):
    PROVIDES_ATTRIBUTES = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_probability_multiplier',
    ]
    
    def __init__(self, visible_attributes: List[str] = None):
        # Agent only sees infection probability, not multipliers
        self.visible_patient_attributes = visible_attributes or ['prob_infected']
        # The observe() method will only extract these
```

## Abstract Methods

### `sample(n, true_amr_levels, rng) -> List[Patient]`

**Purpose**: Generate n patient objects with attributes drawn from configured distributions.

**Signature**:
```python
@abstractmethod
def sample(
    self,
    n: int,                              # Number of patients to generate
    true_amr_levels: Dict[str, float],   # Current AMR levels per antibiotic
    rng: np.random.Generator             # RNG for reproducibility
) -> List[Any]:  # List of Patient objects
```

**Parameters**:

- **n** (`int`): Number of patients to generate for this timestep (e.g., 8 patients per episode)

- **true_amr_levels** (`Dict[str, float]`): Ground-truth antimicrobial resistance levels per antibiotic.
  - Keys: antibiotic names (e.g., 'A', 'B')
  - Values: floats in [0, 1] representing resistance fractions
  - Used to determine infection sensitivity: P(sensitive to antibiotic) = 1 - amr_level
  - Allows correlation between patient attributes and true resistance

- **rng** (`np.random.Generator`): NumPy random number generator seeded for reproducibility.
  - Use this for all randomness in sampling (don't use global random state)
  - Ensures deterministic behavior given same seed

**Returns**: `List[Patient]` — list of n patient objects with:
  - All attributes declared in `PROVIDES_ATTRIBUTES`
  - Both true and observed versions (e.g., `prob_infected` and `prob_infected_obs`)

**Design Notes**:
- Each call generates a new cohort (no memory between calls)
- Attributes can be deterministic (constant) or stochastic (sampled from distribution)
- Observed attributes may include noise/bias to simulate imperfect risk assessment

**Example**:
```python
def sample(self, n: int, true_amr_levels: Dict[str, float], rng: np.random.Generator) -> List[Patient]:
    """Sample n patients with heterogeneous infection risks."""
    patients = []
    for _ in range(n):
        # Sample true infection probability
        true_prob_infected = rng.uniform(0.3, 0.8)
        
        # Add observation noise (agent sees noisy version)
        obs_noise = rng.normal(0, 0.05)
        obs_prob_infected = np.clip(true_prob_infected + obs_noise, 0, 1)
        
        patient = Patient(
            prob_infected=true_prob_infected,
            prob_infected_obs=obs_prob_infected,
            # ... other attributes
        )
        patients.append(patient)
    return patients
```

### `observe(patients) -> np.ndarray`

**Purpose**: Extract observable patient attributes in fixed order to create agent observation.

**Signature**:
```python
@abstractmethod
def observe(self, patients: List[Any]) -> np.ndarray:
    """Extract observed features from patients."""
    pass
```

**Parameters**:

- **patients** (`List[Patient]`): List of patient objects from `sample()`

**Returns**: `np.ndarray` of shape `(num_patients * num_visible_attributes,)` containing:
  - Observed attribute values (e.g., `prob_infected_obs` not true `prob_infected`)
  - Flattened in consistent order matching `visible_patient_attributes`
  - dtype: typically float32 or float64

**Design Notes**:
- Extraction order must be deterministic (same subclass always maps attributes in same order)
- Used to construct part of the agent's state observation
- Returns flattened 1D array for concatenation with AMR levels, etc.

**Example**:
```python
def observe(self, patients: List[Patient]) -> np.ndarray:
    """Extract observed attributes in order."""
    observations = []
    for patient in patients:
        for attr_name in self.visible_patient_attributes:
            # Use _obs version (noisy observed value)
            attr_obs_name = f"{attr_name}_obs"
            value = getattr(patient, attr_obs_name, 0.0)
            observations.append(value)
    return np.array(observations, dtype=np.float32)
```

### `obs_dim(num_patients) -> int`

**Purpose**: Compute the observation vector dimension contributed by patients.

**Signature**:
```python
@abstractmethod
def obs_dim(self, num_patients: int) -> int:
    """Return observation dimension for num_patients."""
    pass
```

**Parameters**:

- **num_patients** (`int`): Number of patients per timestep (same as n in `sample()`)

**Returns**: `int` dimension equal to `num_patients * len(visible_patient_attributes)`

**Design Notes**:
- Used at environment initialization to size observation space
- Must match the length of arrays returned by `observe()`

**Example**:
```python
def obs_dim(self, num_patients: int) -> int:
    """Dimension = (# patients) × (# visible attributes)."""
    return num_patients * len(self.visible_patient_attributes)
```

## Concrete Example: Gaussian Heterogeneous Patients

```python
from dataclasses import dataclass
import numpy as np
from abx_amr_simulator.core import PatientGeneratorBase, Patient

@dataclass
class Patient:
    """Simple patient record."""
    prob_infected: float
    prob_infected_obs: float
    benefit_value_multiplier: float
    benefit_value_multiplier_obs: float

class GaussianPatientGenerator(PatientGeneratorBase):
    """
    Generates patients with attributes sampled from Gaussian distributions.
    Attributes can have observational noise (agent sees noisy version).
    """
    
    PROVIDES_ATTRIBUTES = [
        'prob_infected',
        'benefit_value_multiplier',
    ]
    
    def __init__(
        self,
        visible_attributes: list = None,
        infection_mean: float = 0.5,
        infection_std: float = 0.15,
        benefit_mean: float = 1.0,
        benefit_std: float = 0.3,
        observation_noise_std: float = 0.05,
    ):
        """Initialize distribution parameters."""
        self.visible_patient_attributes = visible_attributes or ['prob_infected']
        self.infection_mean = infection_mean
        self.infection_std = infection_std
        self.benefit_mean = benefit_mean
        self.benefit_std = benefit_std
        self.observation_noise_std = observation_noise_std
    
    def sample(
        self,
        n: int,
        true_amr_levels: dict,
        rng: np.random.Generator
    ) -> list:
        """Sample n patients from Gaussian distributions with noise."""
        patients = []
        for _ in range(n):
            # True values
            true_prob_infected = np.clip(
                rng.normal(self.infection_mean, self.infection_std),
                0, 1
            )
            true_benefit = np.clip(
                rng.normal(self.benefit_mean, self.benefit_std),
                0.1, 3.0
            )
            
            # Observed values (with noise)
            obs_prob_infected = np.clip(
                true_prob_infected + rng.normal(0, self.observation_noise_std),
                0, 1
            )
            obs_benefit = np.clip(
                true_benefit + rng.normal(0, self.observation_noise_std),
                0.1, 3.0
            )
            
            patient = Patient(
                prob_infected=true_prob_infected,
                prob_infected_obs=obs_prob_infected,
                benefit_value_multiplier=true_benefit,
                benefit_value_multiplier_obs=obs_benefit,
            )
            patients.append(patient)
        
        return patients
    
    def observe(self, patients: list) -> np.ndarray:
        """Extract observed attributes in order."""
        observations = []
        for patient in patients:
            for attr_name in self.visible_patient_attributes:
                obs_attr = f"{attr_name}_obs"
                value = getattr(patient, obs_attr, 0.0)
                observations.append(value)
        return np.array(observations, dtype=np.float32)
    
    def obs_dim(self, num_patients: int) -> int:
        """Compute total observation dimension."""
        return num_patients * len(self.visible_patient_attributes)
```

## Usage in Training

Patient generators are instantiated and passed to the environment:

```python
from abx_amr_simulator.core import ABXAMREnv, PatientGenerator
from abx_amr_simulator.core import validate_compatibility, RewardCalculator

# Create generator with visibility config
pg = PatientGenerator(
    visible_patient_attributes=['prob_infected'],
    infection_mean=0.5,
)

# Create reward calculator (requires certain attributes)
rc = RewardCalculator(REQUIRED_PATIENT_ATTRS=['prob_infected'])

# Validate compatibility
validate_compatibility(pg, rc)

# Create environment
env = ABXAMREnv(
    patient_generator=pg,
    reward_calculator=rc,
    num_patients_per_episode=8,
    max_steps=50
)

# Reset and use
obs, info = env.reset()
# obs includes flattened patient attributes from observe()
```

## Subclassing Pattern

Here's the recommended pattern for custom patient generators:

```python
class CustomPatientGenerator(PatientGeneratorBase):
    """
    Your custom patient population generator.
    """
    
    # Step 1: Declare all attributes your patients will have
    PROVIDES_ATTRIBUTES = [
        'prob_infected',
        'benefit_value_multiplier',
        # Add more as needed
    ]
    
    def __init__(self, visible_attributes: list = None, **config):
        """Initialize with configuration."""
        self.visible_patient_attributes = visible_attributes or ['prob_infected']
        # Store other config
    
    def sample(self, n: int, true_amr_levels: dict, rng: np.random.Generator) -> list:
        """Generate n patient objects."""
        patients = []
        for _ in range(n):
            # Sample or compute attributes
            true_value = rng.uniform(0, 1)
            obs_value = true_value + rng.normal(0, 0.05)  # Add noise
            
            patient = Patient(
                prob_infected=true_value,
                prob_infected_obs=np.clip(obs_value, 0, 1),
                # Add other attributes
            )
            patients.append(patient)
        return patients
    
    def observe(self, patients: list) -> np.ndarray:
        """Extract and flatten observed attributes."""
        obs_list = []
        for patient in patients:
            for attr in self.visible_patient_attributes:
                obs_attr = f"{attr}_obs"
                obs_list.append(getattr(patient, obs_attr, 0.0))
        return np.array(obs_list, dtype=np.float32)
    
    def obs_dim(self, num_patients: int) -> int:
        """Return observation dimension."""
        return num_patients * len(self.visible_patient_attributes)
```

## Patient Dataclass

The `Patient` type is typically a simple dataclass:

```python
from dataclasses import dataclass

@dataclass
class Patient:
    """Patient record with true and observed attributes."""
    prob_infected: float              # True infection probability
    prob_infected_obs: float          # Observed (noisy) infection probability
    benefit_value_multiplier: float   # True treatment benefit multiplier
    benefit_value_multiplier_obs: float  # Observed multiplier
    # ... other attributes
```

## Best Practices

1. **Deterministic seeding**: Always use the provided `rng` parameter, never `np.random` directly
2. **Respect bounds**: Keep attributes in valid ranges (probabilities in [0, 1], multipliers positive)
3. **Consistent ordering**: `observe()` must always extract attributes in same order as `visible_patient_attributes`
4. **Validate dimensions**: Ensure `len(observe(patients)) == obs_dim(len(patients))`
5. **Document distributions**: Clearly state how you sample attributes (mean, std, bounds)
6. **Use_obs variants**: Always expose observed attributes, not true ones

## Validation and Error Handling

The system validates patient generators at environment initialization:

```python
# Raises ValueError if 'benefit_value_multiplier' requires by rc but not provided by pg
validate_compatibility(pg, rc)
```

Error message example:
```
ValueError: RewardCalculator requires Patient attributes ['benefit_value_multiplier'] 
but PatientGenerator doesn't provide them.
PatientGenerator provides: ['prob_infected'],
RewardCalculator requires: ['prob_infected', 'benefit_value_multiplier']
```

## Related ABCs

- **RewardCalculatorBase**: Requires specific patient attributes from the generator
- **ABXAMREnv**: Orchestrates environment; calls `sample()` and `observe()` each timestep
- **Patient** (dataclass): Data structure used to pass information between components

## See Also

- Tutorial 2: Config Scaffolding (example patient generator setup)
- Tutorial 11: Component Subclassing (advanced patient generation patterns)
- docs/concepts/patient_heterogeneity.md (modeling decisions)
