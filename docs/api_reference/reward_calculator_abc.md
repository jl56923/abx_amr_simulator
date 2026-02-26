# RewardCalculatorBase API Reference

## Overview

`RewardCalculatorBase` is the abstract base class for all reward calculation implementations. It defines the contract that reward calculators must follow to be compatible with the environment and training pipeline.

**Location**: `abx_amr_simulator.core.RewardCalculatorBase`

**Purpose**: Subclasses implement custom reward functions that balance individual patient outcomes against community antimicrobial resistance burden.

## Class Definition

```python
from abc import ABC, abstractmethod
from typing import List, ClassVar, Any
import numpy as np

class RewardCalculatorBase(ABC):
    """
    Abstract base class for reward calculation.
    
    Subclasses must:
    1. Define REQUIRED_PATIENT_ATTRS class constant listing required Patient attributes
    2. Implement calculate_reward() method accepting List[Patient]
    """
    
    # Class constant declaring which Patient attributes are required
    REQUIRED_PATIENT_ATTRS: ClassVar[List[str]] = []
    
    @abstractmethod
    def calculate_reward(
        self,
        patients: List[Any],
        actions: np.ndarray,
        current_AMR_levels: np.ndarray,
        **kwargs
    ) -> float:
        """Calculate reward based on patient states, actions, and AMR levels."""
        pass
```

## Required Class Constant

### `REQUIRED_PATIENT_ATTRS: ClassVar[List[str]]`

**Purpose**: Declare which Patient attributes your reward calculator needs to assess patient outcomes.

**Type**: `ClassVar[List[str]]` — a list of attribute name strings

**Validation**: At environment initialization, the system checks that `PatientGenerator.PROVIDES_ATTRIBUTES` contains all attributes in `REQUIRED_PATIENT_ATTRS`. Raises `ValueError` if any required attributes are missing.

**Examples**:

```python
# Minimal: only assess infection status and benefits
class MinimalRewardCalculator(RewardCalculatorBase):
    REQUIRED_PATIENT_ATTRS = ['prob_infected', 'benefit_value_multiplier']
    
    def calculate_reward(self, patients, actions, current_AMR_levels, **kwargs):
        # Implement reward logic
        return reward_value

# Comprehensive: assess all patient heterogeneity
class ComprehensiveRewardCalculator(RewardCalculatorBase):
    REQUIRED_PATIENT_ATTRS = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob'
    ]
    
    def calculate_reward(self, patients, actions, current_AMR_levels, **kwargs):
        # Full access to patient heterogeneity
        return reward_value
```

## Abstract Method

### `calculate_reward(patients, actions, current_AMR_levels, **kwargs) -> float`

**Purpose**: Compute the scalar reward signal for the current timestep.

**Signature**:
```python
@abstractmethod
def calculate_reward(
    self,
    patients: List[Any],           # Patient instances
    actions: np.ndarray,           # Action indices per patient
    current_AMR_levels: np.ndarray, # Resistance levels per antibiotic
    **kwargs                        # Additional context
) -> float:
```

**Parameters**:

- **patients** (`List[Any]`): List of `Patient` dataclass instances. Each patient has:
  - True attributes (ground truth): e.g., `prob_infected`, `benefit_value_multiplier`
  - Observed attributes (with noise): e.g., `prob_infected_obs`
  - All attributes declared in `REQUIRED_PATIENT_ATTRS` will be present

- **actions** (`np.ndarray`): Integer action array, shape `(num_patients,)`.
  - Values: 0 to num_antibiotics (antibiotic index) or num_antibiotics (NO_RX/no treatment)
  - Indicates which antibiotic was prescribed to each patient

- **current_AMR_levels** (`np.ndarray`): Resistance levels, shape `(num_antibiotics,)`.
  - Values: floats in [0, 1] representing observable AMR fraction per antibiotic
  - Updated by AMRDynamicsBase.step() each timestep

- **kwargs** (dict): Additional context (implementation-specific). May include:
  - `infection_outcomes`: Whether each patient had infection this step
  - `effective_treatments`: Which patients recovered due to treatment
  - `adverse_effects_incurred`: Which patients experienced adverse effects
  - `amr_doses`: Prescribing counts that drove AMR model

**Returns**: Single `float` scalar representing the immediate reward for this timestep.

**Design Notes**:
- Reward is NOT episode return—it's the **immediate** signal for this timestep only
- Typically combines:
  - Individual patient outcomes (clinical benefit - costs) weighted by (1 - λ)
  - Community AMR burden (summed resistance levels) weighted by λ, negated
- Should be bounded and stable (avoid extreme values > 1000 that destabilize training)

## Concrete Example: Lambda-Weighted Reward

Here's the canonical example from the codebase:

```python
from abx_amr_simulator.core import RewardCalculatorBase, Patient
import numpy as np

class LambdaWeightedRewardCalculator(RewardCalculatorBase):
    """
    Balances individual patient benefit against community AMR burden.
    
    Reward = (1 - λ) × individual_reward + λ × community_amr_penalty
    
    - λ = 0.0: Clinical-only (ignore AMR)
    - λ = 0.5: Balanced
    - λ = 1.0: AMR-only (ignore patient health)
    """
    
    REQUIRED_PATIENT_ATTRS = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_value_multiplier',
        'benefit_probability_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob'
    ]
    
    def __init__(
        self,
        lambda_weight: float = 0.5,
        abx_clinical_reward_penalties_info_dict: dict = None,
        **kwargs
    ):
        """
        Initialize reward calculator.
        
        Args:
            lambda_weight: Trade-off parameter ∈ [0, 1]
            abx_clinical_reward_penalties_info_dict: Per-antibiotic benefit/cost parameters
                Example: {
                    'clinical_benefit_reward': 10.0,
                    'clinical_failure_penalty': -20.0,
                    'adverse_effect_penalty': -5.0,
                    ...
                }
        """
        self.lambda_weight = lambda_weight
        self.abx_params = abx_clinical_reward_penalties_info_dict or {}
    
    def calculate_reward(
        self,
        patients: list,
        actions: np.ndarray,
        current_AMR_levels: np.ndarray,
        **kwargs
    ) -> float:
        # Example: compute individual patient rewards
        individual_rewards = self._compute_individual_rewards(
            patients, actions, **kwargs
        )
        
        # Community AMR penalty: sum all resistance levels
        amr_penalty = -np.sum(current_AMR_levels)
        
        # Combine via lambda weighting
        reward = (
            (1 - self.lambda_weight) * individual_rewards +
            self.lambda_weight * amr_penalty
        )
        
        return float(reward)
    
    def _compute_individual_rewards(self, patients, actions, **kwargs):
        """Compute sum of individual patient benefits and costs."""
        # Implementation: sum benefits for treated infected patients,
        # subtract costs for adverse effects, etc.
        pass
```

## Usage in Training

Reward calculators are instantiated and passed to the environment:

```python
from abx_amr_simulator.core import ABXAMREnv, PatientGenerator, RewardCalculator

# Create generator and calculator
pg = PatientGenerator(config)
rc = RewardCalculator(lambda_weight=0.5)  # or your custom subclass

# Create environment
env = ABXAMREnv(
    patient_generator=pg,
    reward_calculator=rc,
    num_patients_per_episode=8,
    max_steps=50
)

# Training loop
obs, info = env.reset()
for step in range(num_steps):
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    # reward comes from rc.calculate_reward()
```

## Subclassing Pattern

Here's the recommended pattern for creating custom reward calculators:

```python
class CustomRewardCalculator(RewardCalculatorBase):
    """
    Your custom reward function implementation.
    """
    
    # Step 1: Declare required patient attributes
    REQUIRED_PATIENT_ATTRS = [
        'prob_infected',  # Add all attributes your logic needs
        'benefit_value_multiplier',
    ]
    
    def __init__(self, param1: float = 1.0, param2: str = "default", **kwargs):
        """Initialize with your custom parameters."""
        self.param1 = param1
        self.param2 = param2
        # Store any other configuration
    
    def calculate_reward(
        self,
        patients: list,
        actions: np.ndarray,
        current_AMR_levels: np.ndarray,
        **kwargs
    ) -> float:
        """
        Implement your reward logic here.
        
        Tips:
        - Extract patient attributes from the patient list
        - Combine with action and AMR information
        - Return a single scalar reward
        - Avoid NaN, inf, or extreme values
        """
        
        # Access patient data
        num_patients = len(patients)
        
        # Your custom reward calculation
        clinical_component = self._compute_clinical_reward(patients, actions, **kwargs)
        amr_component = self._compute_amr_cost(current_AMR_levels)
        
        return clinical_component + amr_component
    
    def _compute_clinical_reward(self, patients, actions, **kwargs) -> float:
        """Helper: compute clinical benefit component."""
        total_benefit = 0.0
        for i, patient in enumerate(patients):
            if actions[i] != -1:  # Treatment prescribed
                # Reward if patient is infected and treatment sensitive
                if patient.prob_infected > 0.5:
                    total_benefit += patient.benefit_value_multiplier
        return total_benefit
    
    def _compute_amr_cost(self, current_AMR_levels: np.ndarray) -> float:
        """Helper: compute AMR burden component."""
        return -np.sum(current_AMR_levels)
```

## Best Practices

1. **Validate inputs**: Check patient list length, action array shape, AMR levels bounds
2. **Handle edge cases**: Empty patient lists, all-NO_RX actions, zero AMR levels
3. **Document assumptions**: What does your reward function assume about the system?
4. **Test with synthetic data**: Create dummy patient lists and verify reward computation
5. **Keep it bounded**: Reward values in roughly [-100, 100] range for RL stability
6. **Use kwargs loosely**: Don't assume all context keys present; provide sensible defaults

## Validation and Error Handling

The system validates reward calculators at environment initialization:

```python
# This will raise ValueError if 'benefit_value_multiplier' not in PROVIDES_ATTRIBUTES
from abx_amr_simulator.core import validate_compatibility

rc = CustomRewardCalculator()
pg = PatientGenerator(config)

validate_compatibility(pg, rc)  # Raises if incompatible
```

If a required attribute is missing, you'll see:
```
ValueError: RewardCalculator requires Patient attributes ['benefit_value_multiplier'] 
but PatientGenerator doesn't provide them.
```

## Related ABCs

- **PatientGeneratorBase**: Provides the Patient objects that reward calculators evaluate
- **ABXAMREnv**: Orchestrates the environment; calls `calculate_reward()` each timestep
- **AMRDynamicsBase**: Generates current_AMR_levels passed to `calculate_reward()`

## See Also

- Tutorial 3: Custom Experiments (reward function tuning examples)
- docs/concepts/reward_function_design.md (mathematical formulation)
