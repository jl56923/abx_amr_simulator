# Advanced Heuristic Worker Subclassing

## Overview

This tutorial shows how to create custom `HeuristicWorker` subclasses to implement sophisticated decision-making strategies. Two main patterns are covered:

1. **Attribute Estimation**: Use clinical reasoning to infer unobserved patient attributes from observed ones
2. **Uncertainty Modulation**: Adjust decision confidence based on how much patient information is missing

**Why this is useful:**
- Models realistic clinical decision-making (doctors adapt to incomplete information)
- Enables sophisticated heuristic baselines for benchmarking learned policies
- Supports experimentation with different risk attitudes and reasoning strategies

**For example:**
- A clinician sees high infection probability → infers patient is frail → estimates lower treatment benefit
- A clinician sees many missing attributes → becomes more conservative → requires higher expected reward to prescribe

---

## Quick Start

### Step 1: Create Your Custom Subclass

Create a new Python file in `workspace/experiments/options/option_types/heuristic/`:

```python
# workspace/experiments/options/option_types/heuristic/my_custom_heuristic.py

from typing import Dict, Any
from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker, load_heuristic_option
)


class ClinicalReasoningHeuristicWorker(HeuristicWorker):
    """Heuristic worker with clinical reasoning for missing attributes."""
    
    def _estimate_unobserved_attribute_values_from_observed(
        self,
        patient: Dict[str, float],
    ) -> Dict[str, float]:
        """Estimate missing attributes using clinical reasoning."""
        # Make a copy to avoid mutating original
        patient = patient.copy()
        
        # Get observed infection probability (always visible)
        pI = patient.get('prob_infected', 0.5)
        
        # Estimate recovery_prob: sicker patients less likely to recover
        if ('recovery_without_treatment_prob' not in patient 
            or patient['recovery_without_treatment_prob'] == -1):
            # Linear model: pI=0.0 → r=0.20, pI=1.0 → r=0.05
            patient['recovery_without_treatment_prob'] = max(0.05, 0.2 - 0.15 * pI)
        
        # Estimate benefit multiplier: healthier patients benefit more
        if ('benefit_value_multiplier' not in patient
            or patient['benefit_value_multiplier'] == -1):
            # pI=0.0 → vB=1.3, pI=1.0 → vB=1.0
            patient['benefit_value_multiplier'] = 1.0 + (1.0 - pI) * 0.3
        
        # Estimate failure multiplier: sicker patients fail harder
        if ('failure_value_multiplier' not in patient
            or patient['failure_value_multiplier'] == -1):
            # pI=0.0 → vF=1.0, pI=1.0 → vF=1.2
            patient['failure_value_multiplier'] = 1.0 + pI * 0.2
        
        return patient


def load_clinical_reasoning_heuristic(name: str, config: Dict[str, Any]) -> HeuristicWorker:
    """Loader function for custom worker (delegates to base loader)."""
    # Validate and extract config using base loader validation logic
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dict, got {type(config).__name__}")
    
    if 'duration' not in config:
        raise ValueError(f"Worker '{name}' config missing required key 'duration'")
    if 'action_thresholds' not in config:
        raise ValueError(f"Worker '{name}' config missing required key 'action_thresholds'")
    
    # Instantiate custom subclass
    return ClinicalReasoningHeuristicWorker(
        name=name,
        duration=config['duration'],
        action_thresholds=config['action_thresholds'],
        uncertainty_threshold=config.get('uncertainty_threshold', 2.0),
        default_recovery_without_treatment_prob=config.get('default_recovery_without_treatment_prob', 0.1),
    )
```

### Step 2: Register Your Custom Loader

Update `workspace/experiments/options/option_types/__init__.py`:

```python
from .heuristic.my_custom_heuristic import load_clinical_reasoning_heuristic

OPTION_LOADERS = {
    'block': load_block_option,
    'alternation': load_alternation_option,
    'heuristic': load_heuristic_option,
    'clinical_reasoning_heuristic': load_clinical_reasoning_heuristic,  # Add your loader
}
```

### Step 3: Use in Option Library YAML

```yaml
# workspace/experiments/options/option_libraries/my_library.yaml
option_library_name: "clinical_reasoning_single_abx"
use_relative_uncertainty: false

options:
  CLINICAL_CONSERVATIVE:
    type: clinical_reasoning_heuristic  # Use your custom type
    duration: 10
    action_thresholds:
      prescribe_A: 0.8
      no_treatment: 0.0
    uncertainty_threshold: 2.0
    default_recovery_without_treatment_prob: 0.1
  
  CLINICAL_AGGRESSIVE:
    type: clinical_reasoning_heuristic
    duration: 10
    action_thresholds:
      prescribe_A: 0.5
      no_treatment: 0.0
    uncertainty_threshold: 3.0
```

### Step 4: Train Your Agent

```bash
python experiments/train.py \
    --config experiments/configs/umbrella_configs/hrl_experiment.yaml \
    -o "hrl.option_library_config=workspace/experiments/options/option_libraries/my_library.yaml"
```

---

## Advanced Examples

### Example 1: Nonlinear Estimation Model

Use a sigmoid function for smooth transitions:

```python
import numpy as np

def _estimate_unobserved_attribute_values_from_observed(
    self,
    patient: Dict[str, float],
) -> Dict[str, float]:
    patient = patient.copy()
    pI = patient.get('prob_infected', 0.5)
    
    if ('recovery_without_treatment_prob' not in patient 
        or patient['recovery_without_treatment_prob'] == -1):
        # Sigmoid: smooth transition from 0.3 (healthy) to 0.05 (sick)
        recovery_prob = 0.05 + 0.25 / (1 + np.exp(10 * (pI - 0.5)))
        patient['recovery_without_treatment_prob'] = recovery_prob
    
    return patient
```

### Example 2: Multi-Attribute Reasoning

Use multiple observed attributes together:

```python
def _estimate_unobserved_attribute_values_from_observed(
    self,
    patient: Dict[str, float],
) -> Dict[str, float]:
    patient = patient.copy()
    
    pI = patient.get('prob_infected', 0.5)
    benefit_prob_mult = patient.get('benefit_probability_multiplier', 1.0)
    
    # Combine infection risk + benefit probability to infer overall health
    if ('recovery_without_treatment_prob' not in patient 
        or patient['recovery_without_treatment_prob'] == -1):
        # High pI + low benefit_prob_mult → very sick → low recovery
        # Low pI + high benefit_prob_mult → healthy → high recovery
        health_score = (1 - pI) * benefit_prob_mult
        patient['recovery_without_treatment_prob'] = 0.05 + 0.15 * health_score
    
    return patient
```

### Example 3: Pessimistic vs. Optimistic Clinician

Create two subclasses with different philosophies:

```python
class PessimisticHeuristicWorker(HeuristicWorker):
    """Conservative clinician: assumes worst when data is missing."""
    
    def _estimate_unobserved_attribute_values_from_observed(
        self, patient: Dict[str, float]
    ) -> Dict[str, float]:
        patient = patient.copy()
        pI = patient.get('prob_infected', 0.5)
        
        # Assume low recovery, high failure risk
        if ('recovery_without_treatment_prob' not in patient 
            or patient['recovery_without_treatment_prob'] == -1):
            patient['recovery_without_treatment_prob'] = 0.05  # Always pessimistic
        
        if ('failure_value_multiplier' not in patient
            or patient['failure_value_multiplier'] == -1):
            patient['failure_value_multiplier'] = 1.3  # Assume high failure cost
        
        return patient


class OptimisticHeuristicWorker(HeuristicWorker):
    """Optimistic clinician: assumes best when data is missing."""
    
    def _estimate_unobserved_attribute_values_from_observed(
        self, patient: Dict[str, float]
    ) -> Dict[str, float]:
        patient = patient.copy()
        pI = patient.get('prob_infected', 0.5)
        
        # Assume higher recovery, lower failure risk
        if ('recovery_without_treatment_prob' not in patient 
            or patient['recovery_without_treatment_prob'] == -1):
            patient['recovery_without_treatment_prob'] = 0.15  # Optimistic baseline
        
        if ('benefit_value_multiplier' not in patient
            or patient['benefit_value_multiplier'] == -1):
            patient['benefit_value_multiplier'] = 1.2  # Assume high benefit
        
        return patient
```

---

## Advanced Subclassing: Uncertainty Modulation

### Overview

A fundamentally different approach to handling incomplete information is **uncertainty-modulated decision-making**, where the heuristic worker adjusts its confidence (expected reward) based on how many patient attributes are missing, rather than filling in defaults.

**Key differences from attribute estimation:**
- **Attribute estimation**: "I don't know recovery probability, so I'll guess 0.10"
- **Uncertainty modulation**: "I'm missing 4 out of 6 attributes, so I'll discount my expected reward by 60%"

**When to use each approach:**
- **Attribute estimation**: When you have domain knowledge about relationships between attributes
- **Uncertainty modulation**: When you want to encode risk attitudes (risk-averse vs risk-seeking) without making specific attribute guesses

### Implementation: UncertaintyModulatedHeuristicWorker

This subclass applies a **discount factor** to expected rewards based on uncertainty score (number of missing/padded attributes):

```python
# workspace/experiments/options/option_types/heuristic/uncertainty_modulated_heuristic_worker.py

from typing import Any, Dict
import numpy as np
from .heuristic_option_loader import HeuristicWorker


class UncertaintyModulatedHeuristicWorker(HeuristicWorker):
    """Adjusts expected rewards based on missing patient attributes.
    
    Instead of filling in defaults for missing attributes, this worker
    discounts expected rewards when many attributes are missing, encoding
    risk-averse behavior under uncertainty.
    """
    
    def __init__(
        self,
        name: str,
        duration: int,
        action_thresholds: Dict[str, float],
        uncertainty_threshold: float = 2.0,
        default_recovery_without_treatment_prob: float = 0.1,
        uncertainty_modulation_type: str = 'none',
        uncertainty_modulation_alpha: float = 0.5,
        max_uncertainty: int = 6,
        use_relative_uncertainty: bool = True,
    ):
        super().__init__(
            name=name,
            duration=duration,
            action_thresholds=action_thresholds,
            uncertainty_threshold=uncertainty_threshold,
            default_recovery_without_treatment_prob=default_recovery_without_treatment_prob,
        )
        self.uncertainty_modulation_type = uncertainty_modulation_type
        self.uncertainty_modulation_alpha = uncertainty_modulation_alpha
        self.max_uncertainty = max_uncertainty
        self.use_relative_uncertainty = use_relative_uncertainty
    
    def _adjust_expected_reward_for_uncertainty(
        self,
        expected_reward: float,
        uncertainty_score: float,
    ) -> float:
        """Apply modulation strategy to discount expected reward."""
        if self.uncertainty_modulation_type == 'none' or expected_reward <= 0:
            return expected_reward
        
        alpha = self.uncertainty_modulation_alpha
        
        if self.uncertainty_modulation_type == 'linear':
            # Linear penalty: adjusted = expected * (1 - α * uncertainty / max)
            normalized = min(1.0, uncertainty_score / self.max_uncertainty)
            discount = max(0.0, 1.0 - alpha * normalized)
            return expected_reward * discount
            
        elif self.uncertainty_modulation_type == 'exponential':
            # Exponential decay: adjusted = expected * exp(-α * uncertainty)
            discount = np.exp(-alpha * uncertainty_score)
            return expected_reward * discount
            
        elif self.uncertainty_modulation_type == 'conservative':
            # Hyperbolic: adjusted = expected / (1 + α * uncertainty)
            discount = 1.0 / (1.0 + alpha * uncertainty_score)
            return expected_reward * discount
        
        return expected_reward
    
    def decide(self, env_state: Dict[str, Any]) -> np.ndarray:
        """Override to apply uncertainty modulation before threshold comparison."""
        # [Standard decide() logic, but modulate expected_rewards before threshold check]
        # See full implementation in uncertainty_modulated_heuristic_worker.py
        pass  # Abbreviated for tutorial
```

### Modulation Strategies

Three strategies are available, each encoding a different risk attitude:

#### 1. Linear Modulation (`'linear'`)
```
adjusted_reward = expected_reward × (1 - α × uncertainty / max_uncertainty)
```
- **When to use**: Proportional discounting, intuitive scaling
- **Example**: α=0.5, uncertainty=4/6 → discount by 33% → adjusted = 0.67 × expected
- **Risk attitude**: Moderate risk aversion

#### 2. Exponential Modulation (`'exponential'`)
```
adjusted_reward = expected_reward × exp(-α × uncertainty)
```
- **When to use**: Strong risk aversion, aggressive discounting
- **Example**: α=1.0, uncertainty=4 → discount by 98% → adjusted = 0.018 × expected
- **Risk attitude**: Very risk-averse (small amounts of missing data heavily penalized)

#### 3. Conservative Modulation (`'conservative'`)
```
adjusted_reward = expected_reward / (1 + α × uncertainty)
```
- **When to use**: Cautious but not paralyzed, gradual discounting
- **Example**: α=1.0, uncertainty=4 → discount by 80% → adjusted = 0.2 × expected
- **Risk attitude**: Balanced risk aversion (never reaches zero)

### Configuration Example

```yaml
# workspace/experiments/options/option_libraries/uncertainty_modulated_library.yaml
option_library_name: "uncertainty_modulated_single_abx"
use_relative_uncertainty: true

options:
  RISK_AVERSE_CONSERVATIVE:
    type: uncertainty_modulated_heuristic  # Custom type
    duration: 10
    action_thresholds:
      prescribe_A: 0.5
      no_treatment: 0.0
    uncertainty_modulation_type: 'conservative'  # Hyperbolic
    uncertainty_modulation_alpha: 1.0  # Significant discounting
    max_uncertainty: 6  # All 6 attributes could be missing
    use_relative_uncertainty: true  # Count -1 values
  
  RISK_NEUTRAL:
    type: uncertainty_modulated_heuristic
    duration: 10
    action_thresholds:
      prescribe_A: 0.5
      no_treatment: 0.0
    uncertainty_modulation_type: 'none'  # No discounting
    uncertainty_modulation_alpha: 0.0
  
  ULTRA_CONSERVATIVE:
    type: uncertainty_modulated_heuristic
    duration: 10
    action_thresholds:
      prescribe_A: 0.7  # Higher threshold
      no_treatment: 0.0
    uncertainty_modulation_type: 'exponential'  # Aggressive
    uncertainty_modulation_alpha: 0.8
```

### Registering the Loader

Add to `workspace/experiments/options/option_types/__init__.py`:

```python
from .heuristic.uncertainty_modulated_heuristic_worker import (
    load_uncertainty_modulated_heuristic_option
)

OPTION_LOADERS = {
    'block': load_block_option,
    'alternation': load_alternation_option,
    'heuristic': load_heuristic_option,
    'uncertainty_modulated_heuristic': load_uncertainty_modulated_heuristic_option,
}
```

### Use Case: Heterogeneous Patient Visibility

This approach is particularly useful when patient populations have **heterogeneous attribute visibility**:

```python
# Experiment Setup with PatientGeneratorMixer
# High-risk patients: all 6 attributes visible → uncertainty = 0
# Low-risk patients: only 2 attributes visible → uncertainty = 4

# With uncertainty modulation:
# - High-risk patient (uncertainty=0): No discounting, full expected reward  
# - Low-risk patient (uncertainty=4): Significant discounting, conservative prescribing

# This encodes: "Be more cautious when data is incomplete"
```

### Comparison with Attribute Estimation

| **Approach** | **Attribute Estimation** | **Uncertainty Modulation** |
|-------------|-------------------------|---------------------------|
| **Philosophy** | "Fill in missing data with educated guesses" | "Discount confidence when data is missing" |
| **Domain knowledge needed** | High (need attribute relationships) | Low (just encode risk attitude) |
| **Risk encoding** | Implicit (via estimated values) | Explicit (via modulation parameters) |
| **Interpretability** | Less transparent (hidden in estimates) | More transparent (clear discount factors) |
| **Use case** | Known attribute correlations | Heterogeneous visibility experiments |

**General guideline:**
- Use **attribute estimation** when you have strong domain knowledge about how attributes relate
- Use **uncertainty modulation** when experimenting with different risk attitudes or when visibility varies across subpopulations

---

## Best Practices

### ✅ Do:
- **Always copy patient dict** before modifying: `patient = patient.copy()`
- **Check for missing values** before filling: `if attr not in patient or patient[attr] == -1`
- **Use clinically plausible ranges** (e.g., probabilities in [0, 1])
- **Document your reasoning** with comments explaining clinical logic
- **Test edge cases** (all attributes visible, all missing, partial visibility)

### ❌ Don't:
- **Mutate original dict** - always work on a copy
- **Overwrite observed values** - only fill in missing attributes
- **Access required attributes without checking** - `prob_infected` should always be present, but others may not be
- **Use complex/slow computations** - this runs every timestep for every patient
- **Assume specific attribute order** - check for presence, don't rely on dict iteration order

---

## Testing Your Custom Worker

Create a test file `tests/unit/options/test_my_custom_heuristic.py`:

```python
import pytest
from workspace.experiments.options.option_types.heuristic.my_custom_heuristic import (
    ClinicalReasoningHeuristicWorker
)


def test_estimation_fills_missing_recovery_prob():
    """Test that missing recovery_prob is estimated from prob_infected."""
    worker = ClinicalReasoningHeuristicWorker(
        name='test_worker',
        duration=10,
        action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
    )
    
    # Patient with only prob_infected visible
    patient = {'prob_infected': 0.8}
    
    # Estimate missing attributes
    result = worker._estimate_unobserved_attribute_values_from_observed(patient=patient)
    
    # Should have estimated recovery_prob
    assert 'recovery_without_treatment_prob' in result
    # High infection → low recovery (0.2 - 0.15 * 0.8 = 0.08)
    assert 0.05 <= result['recovery_without_treatment_prob'] <= 0.15


def test_estimation_preserves_observed_values():
    """Test that observed values are not overwritten."""
    worker = ClinicalReasoningHeuristicWorker(
        name='test_worker',
        duration=10,
        action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
    )
    
    # Patient with recovery_prob already observed
    patient = {
        'prob_infected': 0.8,
        'recovery_without_treatment_prob': 0.25  # Observed value
    }
    
    result = worker._estimate_unobserved_attribute_values_from_observed(patient=patient)
    
    # Should preserve observed value, not estimate
    assert result['recovery_without_treatment_prob'] == 0.25


def test_estimation_handles_padded_values():
    """Test that -1 padded values are treated as missing."""
    worker = ClinicalReasoningHeuristicWorker(
        name='test_worker',
        duration=10,
        action_thresholds={'prescribe_A': 0.5, 'no_treatment': 0.0},
    )
    
    # Patient with benefit_value_multiplier padded as -1
    patient = {
        'prob_infected': 0.3,
        'benefit_value_multiplier': -1.0  # Padded
    }
    
    result = worker._estimate_unobserved_attribute_values_from_observed(patient=patient)
    
    # Should estimate benefit multiplier (low infection → high benefit)
    assert result['benefit_value_multiplier'] > 1.0
    assert result['benefit_value_multiplier'] != -1.0
```

Run tests:
```bash
pytest tests/unit/options/test_my_custom_heuristic.py -v
```

---

## Experiment Design Tips

### Comparing Estimation Strategies

Create an experiment set to compare different estimation strategies:

```json
{
  "experiment_set_name": "heuristic_estimation_comparison",
  "experiments": [
    {
      "id": "baseline_constant",
      "name": "Baseline: Constant Defaults",
      "type": "hrl_training",
      "option_library": "option_libraries/heuristic_single_abx.yaml",
      "description": "Standard HeuristicWorker with fixed defaults"
    },
    {
      "id": "clinical_reasoning",
      "name": "Clinical Reasoning Estimation",
      "type": "hrl_training",
      "option_library": "option_libraries/clinical_reasoning_single_abx.yaml",
      "description": "Custom worker with prob_infected-based estimation"
    },
    {
      "id": "pessimistic",
      "name": "Pessimistic Clinician",
      "type": "hrl_training",
      "option_library": "option_libraries/pessimistic_clinician.yaml",
      "description": "Conservative estimation (always assume worst)"
    }
  ]
}
```

### Ablation Studies

Test impact of estimation on different visibility levels:

1. **Full visibility** (6 attributes): Estimation should have minimal impact
2. **Partial visibility** (3 attributes): Estimation should improve performance
3. **Minimal visibility** (1 attribute): Estimation is critical

Compare manager learning curves across these conditions.

---

## Troubleshooting

### Issue: "prob_infected not in patient" error

**Cause:** `prob_infected` is required and must be in `visible_patient_attributes`

**Solution:** Check your environment config:
```yaml
patient_generator:
  visible_patient_attributes:
    - prob_infected  # Must be here!
    - benefit_value_multiplier
```

### Issue: Estimation logic not being called

**Cause:** Using base `HeuristicWorker` instead of custom subclass

**Solution:** Check option library type matches registered loader:
```yaml
options:
  MY_WORKER:
    type: clinical_reasoning_heuristic  # Not 'heuristic'!
```

### Issue: Estimated values seem wrong

**Cause:** Logic error in estimation function

**Solution:** Add print statements for debugging:
```python
def _estimate_unobserved_attribute_values_from_observed(self, patient):
    patient = patient.copy()
    pI = patient.get('prob_infected', 0.5)
    
    if 'recovery_without_treatment_prob' not in patient:
        estimated = max(0.05, 0.2 - 0.15 * pI)
        print(f"[DEBUG] pI={pI:.2f} → estimated recovery={estimated:.3f}")
        patient['recovery_without_treatment_prob'] = estimated
    
    return patient
```

---

## Related Documentation

- [HEURISTIC_POLICY_WORKERS_IMPLEMENTATION.md](../HEURISTIC_POLICY_WORKERS_IMPLEMENTATION.md) - Architecture and design decisions
- [OPTIONS_LIBRARY_IMPLEMENTATION.md](../hrl/OPTIONS_LIBRARY_IMPLEMENTATION.md) - How to create option libraries
- [OPTION_PROTOCOL.md](../hrl/OPTION_PROTOCOL.md) - OptionBase protocol specification
- [uncertainty_modulated_heuristic_worker.py](../../workspace/experiments/options/option_types/heuristic/uncertainty_modulated_heuristic_worker.py) - Full implementation of uncertainty modulation

---

## Summary

**Key takeaways:**

**For Attribute Estimation:**
1. Subclass `HeuristicWorker` and override `_estimate_unobserved_attribute_values_from_observed()`
2. Use domain knowledge to infer missing attributes from observed ones
3. Test with edge cases (all visible, all missing, partial)
4. Compare against baseline with constant defaults

**For Uncertainty Modulation:**
1. Subclass `HeuristicWorker` and override `_adjust_expected_reward_for_uncertainty()` and `decide()`
2. Choose modulation strategy (linear/exponential/conservative) based on desired risk attitude
3. Tune `uncertainty_modulation_alpha` to control discounting strength
4. Useful for heterogeneous visibility experiments and risk attitude exploration

**Both approaches** enable sophisticated heuristic baselines that model realistic clinical reasoning patterns, providing stronger benchmarks for learned policies.
