# OptionBase API Reference

## Overview

`OptionBase` is the abstract base class for hierarchical reinforcement learning (HRL) options. Options are primitive action abstractions that encapsulate decision-making policies: they receive environment state, decide what action to take, and report termination conditions.

**Location**: `abx_amr_simulator.hrl.OptionBase`

**Purpose**: Subclasses implement specific decision strategies (e.g., "always prescribe antibiotic A", "prescribe if infection probable", or learned reward-seeking policies).

**Scope**: This project includes three concrete option types:
- **BlockOption**: Always returns the same action
- **AlternationOption**: Cycles through actions
- **HeuristicOption**: Uses rules/heuristics to decide based on environment state

## Class Definition

```python
from abc import ABC, abstractmethod
from typing import ClassVar, List, Dict, Tuple, Any

class OptionBase(ABC):
    """
    Abstract base class for HRL options.
    
    An option encapsulates a decision-making strategy:
    - decide(): Given environment state, choose an action
    - reset(): Prepare for new episode
    - Check requirements: environment must provide observation attributes needed by this option
    
    Subclasses declare what observation/AMR data they require, enabling
    the environment to validate compatibility before training.
    """
    
    # Class constants declaring what observations/AMR this option needs
    REQUIRES_OBSERVATION_ATTRIBUTES: ClassVar[List[str]] = []
    REQUIRES_AMR_LEVELS: ClassVar[bool] = False
    REQUIRES_STEP_NUMBER: ClassVar[bool] = False
    PROVIDES_TERMINATION_CONDITION: ClassVar[bool] = True
    
    @abstractmethod
    def decide(
        self,
        observation: dict,  # env_state dict with keys from REQUIRES_*
        is_training: bool = True,
    ) -> np.ndarray:
        """
        Decide action and termination condition.
        
        Returns:
            np.ndarray: Array of shape (num_patients,) with dtype=object.
                       Each element is an antibiotic name string (e.g., 'A', 'B', 'no_treatment')
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset option state for new episode."""
        pass
```

## Required Class Constants

### `REQUIRES_OBSERVATION_ATTRIBUTES: ClassVar[List[str]]`

**Purpose**: Declare which patient observation attributes this option reads.

**Type**: `ClassVar[List[str]]` — list of attribute names like `['prob_infected', 'benefit_value_multiplier']`

**Validation**: At environment initialization, the system checks that all declared observations are provided by the patient generator.

**Examples**:

```python
# Option that doesn't read patient attributes
class BlockOption(OptionBase):
    REQUIRES_OBSERVATION_ATTRIBUTES = []  # Only prescribes fixed action
    
    def decide(self, observation: dict, is_training: bool = True):
        # action doesn't depend on observation
        return self.action, "always_active"

# Option that reads infection probability
class AlwaysPrescribeIfInfected(OptionBase):
    REQUIRES_OBSERVATION_ATTRIBUTES = ['prob_infected']
    
    def decide(self, observation: dict, is_training: bool = True):
        patients = observation['patients']  # List of Patient objects
        for patient in patients:
            if patient.prob_infected > 0.5:
                return self.prescribe_action, "patient_is_infected"
        return self.no_treat_action, "patient_not_infected"

# Option that uses all available observations
class ComprehensiveHeuristic(OptionBase):
    REQUIRES_OBSERVATION_ATTRIBUTES = [
        'prob_infected',
        'benefit_value_multiplier',
        'failure_probability_multiplier',
        'recovery_without_treatment_prob',
    ]
    
    def decide(self, observation: dict, is_training: bool = True):
        # Complex logic using all patient attributes
        ...
```

### `REQUIRES_AMR_LEVELS: ClassVar[bool]`

**Purpose**: Declare if this option needs current AMR levels.

**Type**: `ClassVar[bool]` — `True` if option reads AMR levels, `False` otherwise

**Content when True**: `observation['amr_levels']` is a `dict` with keys = antibiotic names, values = floats in [0, 1]

**Examples**:

```python
# Option that doesn't care about resistance levels
class DummyBlockOption(OptionBase):
    REQUIRES_AMR_LEVELS = False
    
    def decide(self, observation: dict, is_training: bool = True):
        return 0, "fixed_action"  # Always prescribe A

# Option sensitive to resistance before prescribing
class PrescribeWhenLowResistance(OptionBase):
    REQUIRES_AMR_LEVELS = True
    
    def decide(self, observation: dict, is_training: bool = True):
        amr = observation['amr_levels']  # {'A': 0.1, 'B': 0.8, ...}
        target_abx = 'A'
        if amr[target_abx] < 0.5:
            return self.action_A, "resistance_acceptable"
        else:
            return self.no_treat, "resistance_too_high"
```

### `REQUIRES_STEP_NUMBER: ClassVar[bool]`

**Purpose**: Declare if this option needs current episode step count.

**Type**: `ClassVar[bool]` — `True` if option reads step number, `False` otherwise

**Content when True**: `observation['step_number']` is an `int` ≥ 0

**Examples**:

```python
# Option with fixed behavior (no step dependence)
class StaticOption(OptionBase):
    REQUIRES_STEP_NUMBER = False
    
    def decide(self, observation: dict, is_training: bool = True):
        return 0, "static_decision"

# Option that changes behavior over episode (e.g., "switch treatments every 10 steps")
class AlternateEvery10Steps(OptionBase):
    REQUIRES_STEP_NUMBER = True
    
    def decide(self, observation: dict, is_training: bool = True):
        step = observation['step_number']
        if (step // 10) % 2 == 0:
            return 0, f"prescribe_A_at_step_{step}"
        else:
            return 1, f"prescribe_B_at_step_{step}"
```

### `PROVIDES_TERMINATION_CONDITION: ClassVar[bool]`

**Purpose**: Indicate whether `decide()` returns meaningful termination info (usually `True`).

**Type**: `ClassVar[bool]` — typically `True` for all substantive options

**Standard Value**: `True` for normal options. Set to `False` only if option never terminates.

## Abstract Methods

### `decide(observation, is_training) -> Tuple[int, str]`

**Purpose**: Decide which action to take and provide termination condition.

**Signature**:
```python
@abstractmethod
def decide(
    self,
    observation: dict,          # Environment state dict with requested fields
    is_training: bool = True,   # True during training, False during eval
) -> np.ndarray:
    """
    Make decision based on observation.
    
    Args:
        observation (dict): Contains:
            - 'patients': List[Patient] (if REQUIRES_OBSERVATION_ATTRIBUTES non-empty)
            - 'amr_levels': Dict[str, float] (if REQUIRES_AMR_LEVELS=True)
            - 'step_number': int (if REQUIRES_STEP_NUMBER=True)
            - 'option_library': OptionLibrary instance (always available)
        is_training (bool): Flag for exploratory vs. deterministic behavior
    
    Returns:
        np.ndarray: Shape (num_patients,) with dtype=object. Each element is an antibiotic name
                   string (e.g., 'A', 'B', 'no_treatment') representing the action selected
                   for each patient.
    """
    pass
```

**Parameters**:

- **observation** (`dict`): Partial environment state with only fields the option declares needing:
  - **'patients'** (if `REQUIRES_OBSERVATION_ATTRIBUTES` non-empty): List of `Patient` objects with true and observed attributes
  - **'amr_levels'** (if `REQUIRES_AMR_LEVELS=True`): Dict mapping antibiotic name → current resistance level (float in [0, 1])
  - **'step_number'** (if `REQUIRES_STEP_NUMBER=True`): Current episode step (int ≥ 0)
  - **'option_library'** (always): Reference to the `OptionLibrary` for accessing option configuration and indices

- **is_training** (`bool`): Hints whether training (can explore) or evaluation (should be deterministic)

**Returns**: Tuple of:
  - **action** (`int`): Action index chosen by option. Must be in valid action set (0 ≤ action < num_actions)
  - **termination_info** (`str`): Human-readable description of termination condition (for logging/diagnosis)

**Contract**:
- Must inspect `observation` only for fields declared in `REQUIRES_*` constants
- Must return `np.ndarray` with shape (num_patients,) and dtype=object
- Each element must be a valid antibiotic name string (e.g., 'A', 'B', 'no_treatment')
- Must be deterministic if `is_training=False` (for reproducibility in evaluation)
- The OptionsWrapper will convert strings to action indices before passing to environment

### `reset() -> None`

**Purpose**: Reset option state for beginning of new episode.

**Signature**:
```python
@abstractmethod
def reset(self) -> None:
    """Reset option state (called at start of new episode)."""
    pass
```

**Parameters**: None

**Returns**: `None`

**Contract**:
- Clear any episode-specific state (counters, accumulators, etc.)
- Prepare for fresh `decide()` calls
- Called by environment.reset() before first decide()

**Design Notes**:
- Critical for reproducible multi-episode training
- Typically clears internal step counters or accumators
- Don't reset learned parameters (those persist across episodes)

## Concrete Example 1: Block Option (Constant Action)

```python
import numpy as np
from abx_amr_simulator.hrl import OptionBase

class BlockOption(OptionBase):
    """Option that always returns the same antibiotic name string."""
    
    REQUIRES_OBSERVATION_ATTRIBUTES = []  # No observation needed
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = True
    
    def __init__(self, antibiotic: str):
        """Initialize to always return this antibiotic name.
        
        Args:
            antibiotic: Name of antibiotic to prescribe (e.g., 'A', 'B', 'no_treatment')
        """
        self.antibiotic = antibiotic
        self.steps_since_start = 0
    
    def decide(self, observation: dict, is_training: bool = True) -> np.ndarray:
        """Always return the same antibiotic name.
        
        Returns:
            np.ndarray: Shape (num_patients,) with dtype=object, all elements = self.antibiotic
        """
        self.steps_since_start += 1
        num_patients = len(observation.get('patients', []))
        return np.full(num_patients, self.antibiotic, dtype=object)
    
    def reset(self) -> None:
        """Reset step counter."""
        self.steps_since_start = 0
```

Usage:
```python
option = BlockOption(antibiotic='A')  # Always prescribe antibiotic A
# With 3 patients in observation:
actions = option.decide({'patients': [p1, p2, p3]})  
# Returns: np.array(['A', 'A', 'A'], dtype=object)
```

## Concrete Example 2: Condition-Based Heuristic

```python
import numpy as np
from abx_amr_simulator.hrl import OptionBase

class PrescribeIfHighRisk(OptionBase):
    """Prescribe if infection probability > threshold."""
    
    REQUIRES_OBSERVATION_ATTRIBUTES = ['prob_infected']
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = True
    
    def __init__(
        self,
        prescribe_action: str,
        no_treat_action: str,
        infection_threshold: float = 0.5,
    ):
        """Initialize threshold and antibiotic names.
        
        Args:
            prescribe_action: Antibiotic name to prescribe (e.g., 'A')
            no_treat_action: Antibiotic name for no treatment (e.g., 'no_treatment')
            infection_threshold: Infection probability threshold above which to prescribe
        """
        self.prescribe_action = prescribe_action
        self.no_treat_action = no_treat_action
        self.infection_threshold = infection_threshold
    
    def decide(self, observation: dict, is_training: bool = True) -> np.ndarray:
        """Decide based on aggregated infection probability.
        
        Returns:
            np.ndarray: Shape (num_patients,) with dtype=object.
                       Returns self.prescribe_action or self.no_treat_action for each patient.
        """
        patients = observation['patients']
        num_patients = len(patients)
        
        avg_prob_infected = sum(
            p.prob_infected_obs for p in patients
        ) / num_patients
        
        if avg_prob_infected > self.infection_threshold:
            action_name = self.prescribe_action
        else:
            action_name = self.no_treat_action
        
        return np.full(num_patients, action_name, dtype=object)
    
    def reset(self) -> None:
        """No state to reset."""
        pass
```

Usage:
```python
option = PrescribeIfHighRisk(
    prescribe_action='A',
    no_treat_action='no_treatment',
    infection_threshold=0.6,
)
actions = option.decide({'patients': [Patient(...), Patient(...), Patient(...)]})
# If avg infection > 0.6:
#   Returns np.array(['A', 'A', 'A'], dtype=object)
# Otherwise:
#   Returns np.array(['no_treatment', 'no_treatment', 'no_treatment'], dtype=object)
```

## Concrete Example 3: Time-Varying Strategy

```python
import numpy as np
from abx_amr_simulator.hrl import OptionBase

class AlternatingStrategy(OptionBase):
    """
    Alternates between two antibiotic names every N steps.
    
    Useful for testing multi-step strategies during exploration.
    """
    
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = True  # Read step number
    PROVIDES_TERMINATION_CONDITION = True
    
    def __init__(
        self,
        action_even: str,
        action_odd: str,
        cycle_length: int = 1,
    ):
        """Initialize alternating scheme.
        
        Args:
            action_even: Antibiotic name to use on even cycles (e.g., 'A')
            action_odd: Antibiotic name to use on odd cycles (e.g., 'B')
            cycle_length: Number of steps per cycle
        """
        self.action_even = action_even
        self.action_odd = action_odd
        self.cycle_length = cycle_length
        self.internal_step = 0
    
    def decide(self, observation: dict, is_training: bool = True) -> np.ndarray:
        """Alternate based on step number.
        
        Returns:
            np.ndarray: Shape (num_patients,) with dtype=object.
                       Alternates between self.action_even and self.action_odd.
        """
        step = observation['step_number']
        num_patients = len(observation.get('patients', []))
        action_idx = (step // self.cycle_length) % 2
        action = self.action_even if action_idx == 0 else self.action_odd
        return np.full(num_patients, action, dtype=object)
    
    def reset(self) -> None:
        """Reset internal counter."""
        self.internal_step = 0
```

## Advanced Example: Heuristic Option with Custom Logic

```python
import numpy as np
from abx_amr_simulator.hrl import OptionBase

class ResistanceAwareHeuristic(OptionBase):
    """
    Complex heuristic: balance infection risk against resistance levels.
    
    Decision rule:
    - If infection probable AND resistance low: prescribe
    - If infection probable AND resistance high: wait
    - If infection unlikely: don't prescribe
    """
    
    REQUIRES_OBSERVATION_ATTRIBUTES = ['prob_infected', 'benefit_value_multiplier']
    REQUIRES_AMR_LEVELS = True  # Need current AMR
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = True
    
    def __init__(
        self,
        antibiotic: str,
        prescribe_action: str,
        no_treat_action: str,
        infection_threshold: float = 0.5,
        resistance_threshold: float = 0.6,
        benefit_multiplier_threshold: float = 1.0,
    ):
        """Initialize heuristic parameters.
        
        Args:
            antibiotic: Name of antibiotic to consider (e.g., 'A')
            prescribe_action: Antibiotic name to prescribe
            no_treat_action: Antibiotic name for no treatment
            infection_threshold: Minimum infection probability to consider prescribing
            resistance_threshold: Maximum acceptable resistance level
            benefit_multiplier_threshold: Minimum expected benefit to prescribe
        """
        self.antibiotic = antibiotic
        self.prescribe_action = prescribe_action
        self.no_treat_action = no_treat_action
        self.infection_threshold = infection_threshold
        self.resistance_threshold = resistance_threshold
        self.benefit_multiplier_threshold = benefit_multiplier_threshold
    
    def decide(self, observation: dict, is_training: bool = True) -> np.ndarray:
        """Complex decision logic combining multiple factors.
        
        Returns:
            np.ndarray: Shape (num_patients,) with dtype=object.
                       Returns self.prescribe_action or self.no_treat_action for each patient.
        """
        patients = observation['patients']
        num_patients = len(patients)
        amr_levels = observation['amr_levels']
        
        # Get current resistance for this antibiotic
        current_resistance = amr_levels[self.antibiotic]
        
        # Aggregate patient metrics
        avg_infection = np.mean([p.prob_infected_obs for p in patients])
        avg_benefit = np.mean([p.benefit_value_multiplier_obs for p in patients])
        
        # Decision tree
        if avg_infection < self.infection_threshold:
            return np.full(num_patients, self.no_treat_action, dtype=object)
        
        if current_resistance > self.resistance_threshold:
            return np.full(num_patients, self.no_treat_action, dtype=object)
        
        if avg_benefit < self.benefit_multiplier_threshold:
            return np.full(num_patients, self.no_treat_action, dtype=object)
        
        return np.full(num_patients, self.prescribe_action, dtype=object)
    
    def reset(self) -> None:
        """No episode-specific state."""
        pass
```

## Usage in Options Library

Options are typically created via loader functions and registered in the OptionsLibrary:

```python
from abx_amr_simulator.hrl import OptionLibrary, BlockOption

# Create library
option_library = OptionLibrary(
    abx_names=['A', 'B'],
    num_patient_types=1,
    num_patients_per_patient_type=8,
)

# Register block options
for abx_idx, abx_name in enumerate(['A', 'B']):
    option = BlockOption(action=abx_idx)
    option_library.register_option(
        name=f"always_prescribe_{abx_name}",
        option=option,
        option_type="block",
    )

# Use in training
obs = {"patients": [...], "option_library": option_library}
action, term_info = option_library.get_option("always_prescribe_A").decide(obs)
```

## Integration with OptionLibrary

Options interact with OptionLibrary primarily through the `observation['option_library']` field:

```python
def decide(self, observation: dict, is_training: bool = True) -> tuple:
    option_library = observation['option_library']
    
    # Access option metadata
    abx_names = option_library.abx_names
    abx_index = option_library.abx_name_to_index[abx_name]
    
    # Access configuration (if needed)
    original_config = option_library.option_configs.get(self.name)
    
    # Return decision
    return action, termination_info
```

## Subclassing Pattern

Template for implementing custom options:

```python
class MyCustomOption(OptionBase):
    """
    Your custom option description.
    
    Decision logic:
    - [Describe your decision rule]
    - [List factors considered]
    """
    
    # Step 1: Declare requirements
    REQUIRES_OBSERVATION_ATTRIBUTES = []  # Add if reading patient attributes
    REQUIRES_AMR_LEVELS = False           # Set True if reading amr_levels
    REQUIRES_STEP_NUMBER = False          # Set True if reading step_number
    PROVIDES_TERMINATION_CONDITION = True # Usually True
    
    def __init__(self, *args, **kwargs):
        """Initialize with parameters."""
        # Store configuration
        self.param1 = kwargs.get('param1', default_value)
        # Initialize state
        self.counter = 0
    
    def decide(self, observation: dict, is_training: bool = True) -> np.ndarray:
        """
        Make decision based on observation.
        
        Your logic:
        1. Extract fields from observation
        2. Compute decision (deterministic if is_training=False)
        3. Return np.ndarray with antibiotic name strings
        """
        option_library = observation.get('option_library')
        
        # Example: Read requested observations
        patients = observation.get('patients', [])
        amr = observation.get('amr_levels', {})
        step = observation.get('step_number', 0)
        
        # Decision logic here
        action_name = 'A'  # Your logic determines this
        num_patients = len(patients)
        
        return np.full(num_patients, action_name, dtype=object)
    
    def reset(self) -> None:
        """Reset state for new episode."""
        self.counter = 0
        # Reset other episode-varying state
```

## Loader Function Pattern

Options are typically instantiated via YAML configuration and loader functions:

```python
# In YAML config (e.g., always_prescribe_A.yaml):
# option_type: block
# antibiotic: "A"

def load_block_option(config: dict) -> BlockOption:
    """Loader function: dict → BlockOption."""
    return BlockOption(antibiotic=config['antibiotic'])

# Registry pattern
OPTION_LOADERS = {
    'block': load_block_option,
    'heuristic': load_heuristic_option,
    # ...
}

def create_option_from_config(option_type: str, config: dict) -> OptionBase:
    """Dispatch to appropriate loader."""
    loader = OPTION_LOADERS.get(option_type)
    if not loader:
        raise ValueError(f"Unknown option type: {option_type}")
    return loader(config)
```

## Best Practices

1. **Minimize observation requirements**: Specify only fields truly needed
   ```python
   # ✓ Good: Only requests what's needed
   REQUIRES_OBSERVATION_ATTRIBUTES = ['prob_infected']
   
   # ✗ Bad: Requests everything even if not used
   REQUIRES_OBSERVATION_ATTRIBUTES = [... all attributes ...]
   ```

2. **Deterministic behavior in eval**: When `is_training=False`, always return same action for same input
   ```python
   # ✓ Good: Deterministic
   if is_training and random() > epsilon:
       action_name = explore()
   else:
       action_name = best_known()
   
   # ✗ Bad: Non-deterministic in eval
   action_name = random_choice(['A', 'B', 'no_treatment'])  # Always random
   ```

3. **Return antibiotic name strings**: Always return np.ndarray with dtype=object containing antibiotic names
   ```python
   # ✓ Good: Correct string protocol
   num_patients = len(observation.get('patients', []))
   return np.full(num_patients, 'A', dtype=object)
   
   # ✗ Bad: Wrong return type
   return np.array([0, 0, 0], dtype=np.int32)  # Action indices, not names
   return ('A', 0)  # Tuple, not ndarray
   ```

4. **Validate inputs**: Check observation has required fields
   ```python
   # ✓ Good: Validate before use
   if 'patients' not in observation:
       raise ValueError("Option requires 'patients' in observation")
   
   # ✗ Bad: Crashes silently
   patients = observation['patients']  # KeyError if missing
   ```

5. **Document requirements clearly**: State what REQUIRES_* constants mean in docstring
   ```python
   class MyOption(OptionBase):
       """
       Uses infection probability to decide prescribing.
       
       REQUIRES_OBSERVATION_ATTRIBUTES: ['prob_infected']
           - Needs patient infection risk estimates
       REQUIRES_AMR_LEVELS: False
           - Doesn't consider resistance levels
       """
   ```

## Validation and Error Handling

Options are validated at environment initialization. The primary validation occurs in OptionsWrapper when options are called during step-time:

```python
# Options return antibiotic name strings, which are validated by wrapper:
option_library = OptionLibrary(abx_names=['A', 'B', 'no_treatment'], ...)
option = BlockOption(antibiotic='A')  # Valid antibiotic name

# At step-time, wrapper validates:
actions = option.decide(observation)  # Returns np.array(['A', 'A', 'A'], dtype=object)

# Wrapper checks:
# 1. actions is np.ndarray with dtype=object
if actions.dtype != object:
    raise TypeError(f"decide() must return dtype=object, got {actions.dtype}")

# 2. All action names are valid antibiotics
valid_abx = set(option_library.abx_names)
for action_name in np.unique(actions):
    if action_name not in valid_abx:
        raise ValueError(
            f"decide() returned invalid antibiotic '{action_name}'. "
            f"Valid antibiotics: {valid_abx}"
        )

# 3. reset() is callable
option.reset()  # Should not raise
```

**Earlier validation** (at library initialization):
```python
# All required REQUIRES_* fields are accessible
observed_attrs = patient_generator.visible_patient_attributes
for required in option.REQUIRES_OBSERVATION_ATTRIBUTES:
    if required not in observed_attrs:
        raise ValueError(
            f"Option {option.name} requires '{required}' "
            f"but PatientGenerator only provides {observed_attrs}"
        )
```

## Related ABCs & Components

- **RewardCalculatorBase**: Evaluates whether option's actions were beneficial
- **PatientGeneratorBase**: Provides attributes that options may require
- **OptionLibrary**: Registry and manager for all options; passed to decide()
- **Tutorial 08**: HRL diagnostics (monitoring option behavior)
- **Tutorial 09**: RPPO Manager (integrating options with RL agents)

## See Also

- docs/concepts/hierarchical_rl.md — HRL design philosophy
- docs/concepts/option_discovery.md — Automated option generation
- private_docs/OPTIONS_LIBRARY_IMPLEMENTATION.md — Detailed implementation reference
- private_docs/OPTION_PROTOCOL.md — Low-level protocol documentation
