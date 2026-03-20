# Environment API Reference

## Overview

`ABXAMREnv` is the core Gymnasium-compatible environment for antibiotic prescribing under evolving antimicrobial resistance (AMR).

Location:
- `abx_amr_simulator.core.ABXAMREnv`
- Source: `src/abx_amr_simulator/core/abx_amr_env.py`

Design role:
- Orchestrates episode lifecycle, action/observation spaces, and step transitions.
- Delegates patient sampling to `PatientGeneratorBase` implementations.
- Delegates reward computation to `RewardCalculatorBase` implementations.
- Delegates AMR state transitions to `AMRDynamicsBase` implementations (default: `AMR_LeakyBalloon`).

The environment enforces a clean composition pattern: instantiate reward calculator and patient generator first, then inject both into the environment.

## Constructor

```python
ABXAMREnv(
      reward_calculator,
      patient_generator,
      antibiotics_AMR_dict: dict,
      crossresistance_matrix: dict | None = None,
      num_patients_per_time_step: int = 10,
      update_visible_AMR_levels_every_n_timesteps: int = 1,
      add_noise_to_visible_AMR_levels: float = 0.0,
      add_bias_to_visible_AMR_levels: float = 0.0,
      max_time_steps: int = 1000,
      include_steps_since_amr_update_in_obs: bool = False,
      enable_temporal_features: bool = False,
      temporal_windows: list[int] | None = None,
      amr_dynamics_instances: dict[str, AMRDynamicsBase] | None = None,
)
```

### Required Inputs

- `reward_calculator`: instantiated reward component.
   - Must define `REQUIRED_PATIENT_ATTRS`.
   - Must provide antibiotic mapping fields used by action decoding.
- `patient_generator`: instantiated patient generator component.
   - Must implement `sample(...)`, `observe(...)`, and `obs_dim(...)`.
   - Must define `PROVIDES_ATTRIBUTES`.
- `antibiotics_AMR_dict`: non-empty dict keyed by antibiotic name, with per-antibiotic AMR parameters:
   - `leak`
   - `flatness_parameter`
   - `permanent_residual_volume`
   - `initial_amr_level`

### Key Optional Inputs

- `crossresistance_matrix`: off-diagonal coupling map between antibiotics.
- `update_visible_AMR_levels_every_n_timesteps`: delayed AMR observation frequency.
- `add_noise_to_visible_AMR_levels`: Gaussian noise SD applied to visible AMR.
- `add_bias_to_visible_AMR_levels`: constant additive AMR observation bias (clipped to `[0, 1]`).
- `include_steps_since_amr_update_in_obs`: appends a scalar counter feature.
- `enable_temporal_features`: appends temporal prescription and AMR-delta features.
- `temporal_windows`: window sizes used when temporal features are enabled.
- `amr_dynamics_instances`: inject custom AMR dynamics objects; bypasses fallback leaky-balloon creation.

## Validation and Fail-Loud Behavior

At initialization, the environment validates:

- At least one antibiotic exists in `antibiotics_AMR_dict`.
- `reward_calculator` has at least one antibiotic reward entry.
- Antibiotic names in environment and reward calculator match exactly.
- Patient generator implements required methods (`sample`, `observe`, `obs_dim`).
- `num_patients_per_time_step > 0`.
- `update_visible_AMR_levels_every_n_timesteps >= 1`.
- `add_noise_to_visible_AMR_levels >= 0`.
- `add_bias_to_visible_AMR_levels` is in `[-1, 1]`.
- Reward/patient compatibility:
   - `set(reward_calculator.REQUIRED_PATIENT_ATTRS) ⊆ set(patient_generator.PROVIDES_ATTRIBUTES)`.
- If AMR dynamics are injected:
   - keys must exactly match antibiotics,
   - all values must be `AMRDynamicsBase` instances.

If any validation fails, construction raises `ValueError` or `TypeError` immediately.

## Action Space

Type:

```python
spaces.MultiDiscrete([num_abx + 1] * num_patients_per_time_step)
```

Semantics per patient index `i`:
- `0 .. num_abx-1`: prescribe that antibiotic index.
- `num_abx`: no treatment.

This design enforces mutually exclusive prescribing: one antibiotic or no treatment per patient at each timestep.

Action mappings are exposed via:
- `get_action_to_antibiotic_mapping()`
- `get_antibiotic_to_action_mapping()`
- `no_treatment_action` property

## Observation Space

Type:

```python
spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
```

Base observation layout:
1. Patient features from `patient_generator.observe(patients)`
2. Visible AMR levels in antibiotic order

Optional appendices:
3. `steps_since_amr_update` scalar if `include_steps_since_amr_update_in_obs=True`
4. Temporal features if `enable_temporal_features=True`
    - normalized prescription counts for each `(antibiotic, window)`
    - visible AMR deltas per antibiotic

Dimension formula:

```text
obs_dim = patient_generator.obs_dim(num_patients_per_time_step) + num_abx
obs_dim += 1                                    # optional steps-since-update
obs_dim += num_abx * len(temporal_windows) + num_abx  # optional temporal features
```

## Episode and Step Semantics

### `reset(seed=None, options=None)`

Returns:
- `(observation, info)`

Behavior:
- Reseeds shared RNG stream (env, reward calculator, patient generator) when `seed` is provided.
- Resets each AMR dynamics model to configured `initial_amr_level`.
- Refreshes visible AMR immediately.
- Samples initial patient cohort.
- Constructs first observation.

Initial `info` includes:
- `visible_amr_levels`

### `step(action)`

Returns:
- `(observation, reward, terminated, truncated, info)`

Execution flow:
1. Validate action against `action_space`.
2. Count antibiotic prescriptions from per-patient actions.
3. Compute effective doses using crossresistance matrix.
4. Step each AMR dynamics model with effective dose.
5. Update visible AMR levels (respecting update frequency).
6. Compute marginal AMR deltas used in reward calculation.
7. Sample next patient cohort.
8. Build next observation.
9. Compute reward via `reward_calculator.calculate_reward(...)`.

Termination behavior (current implementation):
- `terminated` and `truncated` are both set to `current_time_step >= max_time_steps`.

## `info` Dictionary Fields

Common fields provided each step:
- `current_time_step`
- `actual_amr_levels`
- `visible_amr_levels`
- `prescriptions_per_abx`
- `effective_doses`
- `crossresistance_applied`
- `delta_visible_amr_per_antibiotic`
- `patient_stats`

Conditionally included:
- `patient_full_data` when full patient attribute logging is enabled.

Reward component fields are merged from reward calculator output and typically include:
- `total_reward`
- `individual_reward`
- `community_reward`
- additional outcome decomposition fields

## Crossresistance Matrix

Input format (off-diagonal only):

```yaml
crossresistance_matrix:
   A:
      B: 0.3
   B:
      A: 0.1
```

Rules:
- Missing matrix defaults to identity (self-only effect).
- Diagonal entries are auto-set to `1.0`.
- User-provided self entries are rejected.
- Ratios must be numeric in `[0, 1]`.
- Unknown antibiotic names raise `ValueError`.

Effective dose for target antibiotic `t` is:

```text
effective_dose[t] = Σ_p prescription_count[p] * crossresistance[p][t]
```

## AMR Dynamics Injection

By default, environment builds `AMR_LeakyBalloon` instances from `antibiotics_AMR_dict`.

Advanced use: inject custom AMR dynamics objects with:
- `amr_dynamics_instances: dict[str, AMRDynamicsBase]`

Use this when experimenting with non-leaky-balloon AMR models while keeping the same environment orchestration.

## Recommended Construction Path

Use factory functions to preserve configuration ownership boundaries and validation:

```python
from abx_amr_simulator.utils import (
      create_reward_calculator,
      create_patient_generator,
      create_environment,
)

rc = create_reward_calculator(config)
pg = create_patient_generator(config)
env = create_environment(config, reward_calculator=rc, patient_generator=pg)
```

Important ownership rule:
- `visible_patient_attributes` belongs in patient generator config, not environment config.

## Configuration Example

```yaml
antibiotics_AMR_dict:
   A:
      leak: 0.2
      flatness_parameter: 50
      permanent_residual_volume: 0.0
      initial_amr_level: 0.0
   B:
      leak: 0.1
      flatness_parameter: 25
      permanent_residual_volume: 0.0
      initial_amr_level: 0.0

crossresistance_matrix:
   A:
      B: 0.3
   B:
      A: 0.1

num_patients_per_time_step: 5
update_visible_AMR_levels_every_n_timesteps: 1
add_noise_to_visible_AMR_levels: 0.0
add_bias_to_visible_AMR_levels: 0.0
max_time_steps: 500
include_steps_since_amr_update_in_obs: false
```

## Public Helper Methods

- `render()`
   - Console debug print of timestep, AMR levels, and state summary.
- `get_action_to_antibiotic_mapping()`
   - Returns action-index to antibiotic-name mapping including no-treatment.
- `get_antibiotic_to_action_mapping()`
   - Returns reverse mapping.

## Common Error Cases

1. Antibiotic order mismatch between reward calculator and environment
- Symptom: constructor `ValueError` about antibiotic names/order.
- Fix: ensure environment `antibiotics_AMR_dict` keys match reward calculator antibiotic order exactly.

2. Missing patient generator observation API
- Symptom: constructor `ValueError` requiring `observe` or `obs_dim`.
- Fix: subclass `PatientGeneratorBase` and implement all required methods.

3. Missing required patient attributes for reward calculator
- Symptom: constructor `ValueError` listing missing attributes.
- Fix: add attributes to patient generator output and `PROVIDES_ATTRIBUTES`.

4. Invalid crossresistance configuration
- Symptom: `ValueError` for unknown antibiotics, self-entries, or ratios out of range.
- Fix: provide off-diagonal values only and keep all ratios in `[0, 1]`.

## Related References

- `docs/api_reference/patient_generator_abc.md`
- `docs/api_reference/reward_calculator_abc.md`
- `docs/api_reference/amr_dynamics_abc.md`
- `docs/tutorials/01_basic_training.md`
- `docs/tutorials/03_custom_experiments.md`
- `docs/tutorials/11_component_subclassing.md`
