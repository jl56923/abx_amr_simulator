# AMRDynamicsBase API Reference

## Overview

`AMRDynamicsBase` is the abstract base class for antimicrobial resistance (AMR) dynamics models. It defines the contract for implementing custom models that predict how resistance levels evolve in response to antibiotic prescribing.

**Location**: `abx_amr_simulator.core.AMRDynamicsBase`

**Purpose**: Subclasses implement different mathematical models of AMR accumulation and decay (e.g., leaky balloon, exponential, pharmacodynamic models).

## Class Definition

```python
from abc import ABC, abstractmethod
from typing import ClassVar

class AMRDynamicsBase(ABC):
    """
    Abstract base class for antimicrobial resistance dynamics.
    
    Subclasses must:
    1. Define NAME class constant identifying the model type
    2. Implement step(doses) to evolve AMR given antibiotic prescriptions
    3. Implement reset(initial_level) to initialize state
    4. Maintain output bounds: resistance ∈ [residual_floor, 1.0]
    
    Design: Models are per-antibiotic singletons. Environment instantiates
    one per antibiotic, calls step() each timestep, and reads current level.
    """
    
    # Model identifier string
    NAME: ClassVar[str] = ""
    
    @abstractmethod
    def step(self, doses: float) -> float:
        """Execute one dynamics timestep given antibiotic prescriptions."""
        pass
    
    @abstractmethod
    def reset(self, initial_amr_level: float) -> None:
        """Reset state to specified AMR level."""
        pass
```

## Required Class Constant

### `NAME: ClassVar[str]`

**Purpose**: Unique identifier for this AMR dynamics model.

**Type**: `ClassVar[str]` — a short string like "leaky_balloon" or "exponential_accumulation"

**Usage**: Helps distinguish models in logging, configuration, and analysis.

**Examples**:

```python
class LeakyBalloonDynamics(AMRDynamicsBase):
    NAME = "leaky_balloon"

class ExponentialAccumulation(AMRDynamicsBase):
    NAME = "exponential"

class LinearAccumulation(AMRDynamicsBase):
    NAME = "linear_naive"
```

## Abstract Methods

### `step(doses) -> float`

**Purpose**: Execute one dynamics timestep given antibiotic prescription intensity, return evolved resistance level.

**Signature**:
```python
@abstractmethod
def step(self, doses: float) -> float:
    """
    Evolve AMR state by one timestep given antibiotic use.
    
    Args:
        doses (float): Intensity of antibiotic prescribing (0 = no use, 1+ = prescriptions).
                       Most environments use binary (0 or 1), but models should accept any non-negative value.
    
    Returns:
        float: Current AMR level after evolution, bounded in [residual_floor, 1.0].
    """
    pass
```

**Parameters**:

- **doses** (`float >= 0`): Magnitude of antibiotic prescribing pressure driving up resistance.
  - Typical range: [0, 1] (0 = no prescriptions, 1 = fully prescribed)
  - Models should accept arbitrary non-negative floats for flexibility
  - Interpretation: "amount of antibiotic pressure" that increases internal state

**Returns**: `float` in range `[residual_floor, 1.0]`
  - Must be non-negative
  - Upper bound: 1.0 (maximum resistance)
  - Lower bound: `residual_floor` (baseline resistance without prescribing, typically 0.0)
  - Same model instance returns monotonically evolving series across calls

**Contract**:
- Output must always be bounded: `residual_floor <= output <= 1.0`
- If `doses=0` repeated, output should decay toward `residual_floor` (no prescribing → resistance decreases)
- If `doses>0` repeated, output should increase toward 1.0 (prescribing → resistance accumulates)
- Must be deterministic (same puff sequence → same resistance trajectory)

**Design Notes**:
- `step()` is called once per timestep for each antibiotic
- Called after all patients are processed, so doses represents aggregate prescribing
- Models maintain internal state (e.g., latent pressure in leaky balloon)
- No direct access to step count or episode progress

**Example**: Exponential accumulation model
```python
def step(self, doses: float) -> float:
    """Exponential growth with decay."""
    # Increase internal state proportional to doses
    self._pressure += doses * self._growth_rate
    # Exponential decay between prescriptions
    self._pressure *= self._decay_rate  # typically 0.9-0.99 per step
    # Map to [residual, 1.0]
    amr_level = self._residual + (1.0 - self._residual) * (1.0 - np.exp(-self._pressure))
    return np.clip(amr_level, self._residual, 1.0)
```

### `reset(initial_amr_level) -> None`

**Purpose**: Initialize/reset the dynamics model state to a specified AMR level.

**Signature**:
```python
@abstractmethod
def reset(self, initial_amr_level: float) -> None:
    """
    Reset state to specified AMR level.
    
    Args:
        initial_amr_level (float): Initial resistance level, typically in [0, 1].
                                   Must be >= residual_floor.
    
    Raises:
        ValueError: If initial_amr_level < residual_floor or > 1.0.
    """
    pass
```

**Parameters**:

- **initial_amr_level** (`float`): Target AMR level for reset.
  - Expected range: [0, 1] representing fraction of resistant organisms
  - Validated: Must satisfy `residual_floor <= initial_amr_level <= 1.0`

**Returns**: `None`

**Contract**:
- After reset, the next `step(0)` call should return approximately `initial_amr_level` (no prescribing → stable)
- Must validate bounds and raise `ValueError` if out of range
- Must not return a value (only update internal state)

**Design Notes**:
- Called at environment reset to initialize all antibiotics to same starting level
- Allows deterministic initialization for reproducible experiments
- Internal state may be different from output level (e.g., leaky balloon has latent pressure)

**Example**: Leaky balloon reset
```python
def reset(self, initial_amr_level: float) -> None:
    """Reset to specified AMR level."""
    # Validate bounds
    if initial_amr_level < self._residual or initial_amr_level > 1.0:
        raise ValueError(
            f"initial_amr_level {initial_amr_level} outside valid range "
            f"[{self._residual}, 1.0]"
        )
    
    # Determine internal pressure needed to produce this output
    # Output = residual + (1 - residual) * sigmoid(pressure)
    # Solve for pressure
    normalized = (initial_amr_level - self._residual) / (1.0 - self._residual)
    normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0) or log(1)
    self._pressure = np.log(normalized / (1 - normalized)) / self._flatness
    self._internal_state = self._pressure  # Store for step() to use
```

## Concrete Example 1: Leaky Balloon Model

The default AMR dynamics model in this simulator:

```python
import numpy as np
from abx_amr_simulator.core import AMRDynamicsBase

class AMR_LeakyBalloon(AMRDynamicsBase):
    """
    Soft-bounded AMR accumulator with leak (decay).
    
    The model tracks latent prescribing "pressure" that drives up resistance,
    while a constant leak lets pressure decay each timestep. A sigmoid maps
    pressure to visible AMR output, ensuring bounds [residual, 1.0].
    
    Physics: Like a balloon that slowly leaks air. Prescriptions pump air in,
    natural decay lets it out, and the volume determines resistance level.
    """
    
    NAME = "leaky_balloon"
    
    def __init__(
        self,
        leak: float = 0.95,              # Decay multiplier per step (0 < leak < 1)
        flatness_parameter: float = 1.0, # Sigmoid steepness (higher = sharper)
        permanent_residual_volume: float = 0.0,  # Floor resistance
        initial_amr_level: float = 0.1,  # Starting resistance
    ):
        """Initialize balloon parameters."""
        self.leak = leak
        self.flatness_parameter = flatness_parameter
        self.permanent_residual_volume = permanent_residual_volume
        self._pressure = 0.0  # Internal latent state
        
        # Initialize to starting level
        self.reset(initial_amr_level)
    
    def step(self, doses: float) -> float:
        """Execute one step: add doses, decay, return output."""
        # Add prescribing pressure
        self._pressure += doses
        
        # Leak (exponential decay of pressure)
        self._pressure *= self.leak
        
        # Map pressure to [residual, 1.0] via sigmoid
        sigmoid_output = 1.0 / (1.0 + np.exp(-self._pressure / self.flatness_parameter))
        amr_level = (
            self.permanent_residual_volume +
            (1.0 - self.permanent_residual_volume) * sigmoid_output
        )
        
        return np.clip(amr_level, self.permanent_residual_volume, 1.0)
    
    def reset(self, initial_amr_level: float) -> None:
        """Reset to specified AMR level."""
        if initial_amr_level < self.permanent_residual_volume or initial_amr_level > 1.0:
            raise ValueError(
                f"initial_amr_level {initial_amr_level} outside valid range "
                f"[{self.permanent_residual_volume}, 1.0]"
            )
        
        # Inverse sigmoid: solve for pressure needed to output initial_amr_level
        normalized = (
            (initial_amr_level - self.permanent_residual_volume) /
            (1.0 - self.permanent_residual_volume)
        )
        normalized = np.clip(normalized, 0.001, 0.999)
        self._pressure = (
            self.flatness_parameter *
            np.log(normalized / (1.0 - normalized))
        )
```

## Concrete Example 2: Minimal Linear Accumulation

A simpler model for testing or when prescribing dynamics are unclear:

```python
class LinearAccumulation(AMRDynamicsBase):
    """
    Simple linear accumulation with exponential decay.
    
    Resistance increases by 0.01 per unit puff, decays by 10% each step.
    No sigmoid—just limits to [0, 1].
    """
    
    NAME = "linear_accumulation"
    
    def __init__(self, growth_rate: float = 0.01, decay_rate: float = 0.9):
        """Initialize parameters."""
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self._accumulated = 0.0
    
    def step(self, doses: float) -> float:
        """Linear accumulation with exponential decay."""
        self._accumulated += doses * self.growth_rate
        self._accumulated *= self.decay_rate
        return np.clip(self._accumulated, 0.0, 1.0)
    
    def reset(self, initial_amr_level: float) -> None:
        """Reset accumulated resistance."""
        if initial_amr_level < 0.0 or initial_amr_level > 1.0:
            raise ValueError(f"initial_amr_level {initial_amr_level} outside [0, 1]")
        self._accumulated = initial_amr_level
```

## Usage in Environment

AMR dynamics are instantiated per antibiotic and stepped during environment `step()`:

```python
from abx_amr_simulator.core import ABXAMREnv, AMR_LeakyBalloon

# Create environment
env = ABXAMREnv(
    num_patients_per_episode=8,
    antibiotic_names=['A', 'B'],  # Two antibiotics
)

# Internally, env creates one dynamics model per antibiotic:
# env.amr_balloon_models = {
#     'A': AMR_LeakyBalloon(...),
#     'B': AMR_LeakyBalloon(...),
# }

obs, info = env.reset(seed=42)
# reset() calls: dynamics_model.reset(initial_amr_level=0.1) for each antibiotic

# During step
actions = [0, 1, 0, 1, ...]  # Prescribe A to patients 1,3; B to patients 0,2
obs, reward, done, truncated, info = env.step(actions)
# Internally:
# - Sums prescriptions: doses_A = 2, doses_B = 2
# - Steps all models:
#   - new_level_A = env.amr_balloon_models['A'].step(doses=2)
#   - new_level_B = env.amr_balloon_models['B'].step(doses=2)
# - Observation includes new AMR levels
```

## Subclassing Pattern

Here's the template for creating custom AMR dynamics:

```python
import numpy as np
from abx_amr_simulator.core import AMRDynamicsBase

class MyCustomAMRDynamics(AMRDynamicsBase):
    """
    Your custom AMR dynamics model.
    
    Describe the biological/pharmacodynamic assumptions here.
    """
    
    # Step 1: Set model name
    NAME = "my_custom_model"
    
    def __init__(
        self,
        param1: float = 1.0,
        param2: float = 0.9,
        residual_level: float = 0.0,
    ):
        """Initialize with tunable parameters."""
        self.param1 = param1
        self.param2 = param2
        self.residual_level = residual_level
        
        # Initialize internal state
        self._state = 0.0
    
    def step(self, doses: float) -> float:
        """
        Evolve state and return AMR level.
        
        Your dynamics logic here:
        - Update self._state based on doses
        - For high doses: state → higher values
        - For doses=0: state → decay naturally
        - Return value bounded in [residual_level, 1.0]
        """
        # Example: quadratic accumulation with linear decay
        self._state += doses * doses * self.param1
        self._state *= self.param2
        
        amr_level = self.residual_level + (1.0 - self.residual_level) * np.tanh(self._state)
        return np.clip(amr_level, self.residual_level, 1.0)
    
    def reset(self, initial_amr_level: float) -> None:
        """Reset to specified AMR level."""
        if initial_amr_level < self.residual_level or initial_amr_level > 1.0:
            raise ValueError(
                f"initial_amr_level {initial_amr_level} outside "
                f"[{self.residual_level}, 1.0]"
            )
        
        # Set internal state to match desired output
        # (Model-specific inversion logic)
        normalized = (initial_amr_level - self.residual_level) / (1.0 - self.residual_level)
        self._state = np.arctanh(np.clip(normalized, -0.999, 0.999))
```

## Design Principles

### 1. Output Bounds Guarantee

**Contract**: `step()` must always return values in `[residual_floor, 1.0]`

```python
# ✓ Good: Explicit clipping
return np.clip(amr_level, self.residual, 1.0)

# ✗ Bad: Unclipped sigmoid (could return 0.5 when residual=0.1)
return 1.0 / (1.0 + np.exp(-pressure))
```

### 2. Decay Under Zero Prescribing

**Intent**: Repeated `step(0)` should decrease AMR toward residual floor.

```python
# ✓ Good: Exponential leak ensures decay
self._pressure *= 0.95  # Leak multiplier < 1.0

# ✗ Bad: Pressure only increases, never decays
self._pressure += doses  # No decay term
```

### 3. Deterministic Behavior

**Intent**: Same puff sequence → same resistance trajectory.

```python
# ✓ Good: No randomness, repeatable
def step(self, doses):
    self._pressure += doses * self.growth_rate
    return sigmoid(self._pressure)

# ✗ Bad: Uses random noise (non-deterministic)
def step(self, doses):
    noise = np.random.normal(0, 0.01)
    return sigmoid(self._pressure + noise)
```

### 4. Physically Interpretable Parameters

**Intent**: Parameters should map to meaningful biological concepts.

```python
# ✓ Good: Leak = decay rate per step (understandable)
self._pressure *= self.leak  # leak ∈ (0, 1)

# ✗ Bad: Magic constant without meaning
self._pressure *= 0.95312786  # What does this represent?
```

## Pharmacodynamic Inspiration

Real AMR dynamics depend on:

1. **Selection pressure**: Antibiotic killing sensitive organisms → resistant ones survive
2. **Population growth**: Resistant populations expand when antibiotics present
3. **Fitness cost**: Resistance mutations often have fitness costs; absent antibiotics, wild-type dominant
4. **Ecological time scales**: Complex dynamics over months/years

Custom models should ideally reflect these principles:

```python
class PharmacodyanmicModel(AMRDynamicsBase):
    """
    Simplified pharmacodynamic model.
    
    Resistance increases when selective pressure present,
    decreases when antibiotics absent (fitness cost).
    """
    
    NAME = "pharmacodynamic"
    
    def __init__(self, selection_pressure: float = 0.05, fitness_cost: float = 0.02):
        self.selection_pressure = selection_pressure
        self.fitness_cost = fitness_cost
        self._resistance_fraction = 0.1
    
    def step(self, doses: float) -> float:
        if doses > 0:
            # Selection: sensitive organisms die, resistant ones survive
            self._resistance_fraction += doses * self.selection_pressure
        else:
            # No antibiotics: resistant organisms have fitness cost
            self._resistance_fraction *= (1.0 - self.fitness_cost)
        
        return np.clip(self._resistance_fraction, 0.0, 1.0)
    
    def reset(self, initial_amr_level: float) -> None:
        if initial_amr_level < 0.0 or initial_amr_level > 1.0:
            raise ValueError(f"invalid level {initial_amr_level}")
        self._resistance_fraction = initial_amr_level
```

## Validation and Error Handling

Models are validated at environment initialization:

```python
# Validation checks:
# 1. NAME constant defined and non-empty
if not hasattr(self, 'NAME') or not self.NAME:
    raise ValueError(f"{self.__class__.__name__} must define NAME constant")

# 2. step() output in valid range
output = model.step(1.0)
if not (residual <= output <= 1.0):
    raise ValueError(
        f"step() returned {output} outside [{residual}, 1.0]"
    )

# 3. reset() bounds checking (example in LeakyBalloon above)
try:
    model.reset(-0.5)  # Invalid
except ValueError:
    pass  # Correctly rejects invalid input
```

## Testing Custom Models

Recommended test cases when implementing AMRDynamicsBase:

```python
import pytest
from abx_amr_simulator.core import AMRDynamicsBase

def test_name_defined():
    model = MyDynamics()
    assert hasattr(model, 'NAME')
    assert isinstance(model.NAME, str) and len(model.NAME) > 0

def test_step_bounds():
    model = MyDynamics(residual=0.1)
    outputs = [model.step(doses) for doses in [0, 0.5, 1.0, 2.0]]
    assert all(0.1 <= o <= 1.0 for o in outputs)

def test_reset_initialization():
    model = MyDynamics()
    model.reset(0.5)
    output = model.step(0)  # No prescribing
    assert abs(output - 0.5) < 0.01  # Should be close

def test_reset_bounds_validation():
    model = MyDynamics(residual=0.1)
    with pytest.raises(ValueError):
        model.reset(-0.1)  # Below residual
    with pytest.raises(ValueError):
        model.reset(1.5)  # Above 1.0

def test_decay_monotonic():
    model = MyDynamics()
    model.reset(1.0)
    outputs = [model.step(0) for _ in range(10)]
    # Outputs should be non-increasing (decay)
    assert all(outputs[i] >= outputs[i+1] for i in range(len(outputs)-1))

def test_accumulation_monotonic():
    model = MyDynamics()
    model.reset(0.1)
    outputs = [model.step(1.0) for _ in range(10)]
    # Outputs should be non-decreasing (accumulation)
    assert all(outputs[i] <= outputs[i+1] for i in range(len(outputs)-1))
```

## Related ABCs & Components

- **RewardCalculatorBase**: Uses current AMR levels to penalize resistance accumulation
- **ABXAMREnv**: Orchestrates one dynamics model per antibiotic; calls `step()` and reads output
- **Tutorial 06**: Quick-start for using custom dynamics
- **Tutorial 11**: Advanced subclassing for domain-specific models

## See Also

- docs/concepts/amr_dynamics_modeling.md — Mathematical background
- docs/concepts/crossresistance.md — Multi-antibiotic resistance coupling
- CHANGELOG.md — AMRDynamicsBase implementation history
