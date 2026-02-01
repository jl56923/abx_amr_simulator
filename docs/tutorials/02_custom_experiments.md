# Tutorial 2: Custom Experiments

**Goal**: Learn to modify the config/subconfig files to run your own experiments. Learn how to tune the reward function, configure AMR dynamics, run parameter sweeps, and customize patient populations.

**Prerequisites**: Completed Tutorial 1 (Basic Training Workflow)

---

## Overview: The Configuration Hierarchy

Before diving into customization, understand how configurations are organized. This is what you should see immediately after running 

```
configs/
├── umbrella_configs/
│   └── base_experiment.yaml          # References subconfigs + training params
├── environment/
│   └── default.yaml                  # Default environment (1 patient per time step, 2 antibiotics)
├── reward_calculator/
│   └── default.yaml                  # Default reward calculator
├── patient_generator/
│   └── default.yaml                  # Homogeneous population (all constants/same values)
└── agent_algorithm/
    ├── default.yaml                  # PPO
    ├── ppo.yaml                      # PPO
    └── a2c.yaml                      # A2C
```

The default subconfig files contain values that are appropriate to run a simple experiment by using 'base_experiment.yaml' file. Each subconfig file contains a dictionary of parameters that are required to initialize the objects required to create the full ABXAMREnv instance and the agent that will explore the ABXAMREnv environment. The 'PatientGenerator' and 'RewardCalculator' objects are created separately, and then passed in to the ABXAMREnv. Please refer to the docstrings for the initializers for each of these classes for more information.

In order to run your own custom experiments, we suggest that you copy the default subconfig file that you are interested in modifying, then changing the values in that subconfig YAML file. You can then either modify the umbrella config 'base_experiment.yaml' file directly to point to your new subconfig file, or you can override which subconfig file is loaded directly from the CLI (console line interface) commands. If you decide to override both subconfig and parameter values, the subconfig override is executed first, and then the parameter overrides. See below for examples.

---

## Example Customization 1: Tune the Reward Function (Lambda Trade-off)

In the 'RewardCalculator' instance, the reward function balances **clinical benefit** (treating individual patients) against **community AMR burden** (resistance accumulation).

Formula:

$$\text{reward} = (1-\lambda) \times \text{individual\_reward} + \lambda \times \text{community\_reward}$$

- **λ = 0.0**: Selfish agent (maximize patient health, ignore AMR) → agents prescribe heavily
- **λ = 0.5**: Balanced (default)
- **λ = 1.0**: Altruistic agent (minimize community AMR, ignore patient health) → agents underprescribe

### Compare Different Lambda Values

COPILOT: Revise this example, I want to show the user that they can use the base_experiment.yaml and override it with either revised reward_calculator yaml files w/ a range of lambda values, or they can use a param override directly.

COPILOT: Also, the 'training' params are not correct, please go examine one of the real 'base_experiment.yaml' files to correct what the training params should look like in the umbrella configs.

You have two approaches: create separate reward calculator subconfig files, or override the parameter directly from the CLI. Here's both:

**Approach 1: Using parameter overrides (Quickest)**

Run three training jobs with different lambda values using direct parameter overrides:

```bash
# Clinical-focused agent (minimize patient failures)
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -p reward_calculator.lambda_weight=0.0 \
  -p training.run_name=lambda_sweep_clinical_only_seed42 \
  -p training.seed=42

# Balanced agent (λ = 0.5)
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -p reward_calculator.lambda_weight=0.5 \
  -p training.run_name=lambda_sweep_balanced_seed42 \
  -p training.seed=42

# AMR-focused agent (minimize community resistance)
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -p reward_calculator.lambda_weight=1.0 \
  -p training.run_name=lambda_sweep_amr_only_seed42 \
  -p training.seed=42
```

**Approach 2: Using separate subconfig files**

You may also choose to create separate reward calculator configs and modify them individually; this option allows you to full customize how the RewardCalculator is initialized:

```bash
# Create three lambda-specific reward calculator configs
cat > configs/reward_calculator/lambda_0.0.yaml << 'EOF'
lambda_weight: 0.0  # Only consider individual clinical benefit

abx_clinical_reward_penalties_info_dict:
  clinical_benefit_reward: 10.0
  clinical_benefit_probability: 1.0
  clinical_failure_penalty: -10.0
  clinical_failure_probability: 1.0
  abx_adverse_effects_info:
    Antibiotic_A:
      adverse_effect_penalty: -2.0
      adverse_effect_probability: 0.1

epsilon: 0.05  # Shaping parameter for delta AMR info
seed: 42
EOF

cat > configs/reward_calculator/lambda_0.5.yaml << 'EOF'
lambda_weight: 0.5  # Balance between individual and community reward

abx_clinical_reward_penalties_info_dict:
  clinical_benefit_reward: 10.0
  clinical_benefit_probability: 1.0
  clinical_failure_penalty: -10.0
  clinical_failure_probability: 1.0
  abx_adverse_effects_info:
    Antibiotic_A:
      adverse_effect_penalty: -2.0
      adverse_effect_probability: 0.1

epsilon: 0.05  # Shaping parameter for delta AMR info
seed: 42
EOF

cat > configs/reward_calculator/lambda_1.0.yaml << 'EOF'
lambda_weight: 1.0  # Only consider community AMR levels

abx_clinical_reward_penalties_info_dict:
  clinical_benefit_reward: 10.0
  clinical_benefit_probability: 1.0
  clinical_failure_penalty: -10.0
  clinical_failure_probability: 1.0
  abx_adverse_effects_info:
    Antibiotic_A:
      adverse_effect_penalty: -2.0
      adverse_effect_probability: 0.1

epsilon: 0.05  # Shaping parameter for delta AMR info
seed: 42
EOF
```

Now run with subconfig overrides:

```bash
# Clinical-focused
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s reward_calculator-subconfig=configs/reward_calculator/lambda_0.0.yaml \
  -p training.run_name=lambda_sweep_individual_clinical_only

# Balanced
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s reward_calculator-subconfig=configs/reward_calculator/lambda_0.5.yaml \
  -p training.run_name=lambda_sweep_balanced

# AMR-focused
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s reward_calculator-subconfig=configs/reward_calculator/lambda_1.0.yaml \
  -p training.run_name=lambda_sweep_community_amr_only
```

### Analyze the Trade-off

After all three runs complete, they will have unique timestamps appended to their names:
- `lambda_sweep_individual_clinical_only_<TIMESTAMP>`
- `lambda_sweep_balanced_<TIMESTAMP>`
- `lambda_sweep_community_amr_only_<TIMESTAMP>`

To analyze a single run, use the full name (including timestamp):

```bash
# Analyze one run (replace <TIMESTAMP> with the actual timestamp)
python -m abx_amr_simulator.analysis.evaluative_plots --experiment-name lambda_sweep_balanced_20260115_143614
```

This generates (for a single run without seed aggregation):
- Performance curves (rewards, AMR levels, prescribing rates)
- Action-attribute associations showing which patient features drive prescribing decisions
- Plots saved to `analysis_output/lambda_sweep_balanced_<TIMESTAMP>/evaluation/plots/`

**To compare multiple runs**: If you ran multiple seeds per lambda value (e.g., `lambda_sweep_individual_clinical_only_seed1_<TIMESTAMP>`, `lambda_sweep_individual_clinical_only_seed2_<TIMESTAMP>`, etc.), use the `--aggregate-by-seed` flag:

```bash
# Compare seed1, seed2, seed3, etc. together
python -m abx_amr_simulator.analysis.evaluative_plots \
  --experiment-name lambda_sweep_individual_clinical_only \
  --aggregate-by-seed
```

Results with aggregation by seed are saved to `analysis_output/lambda_sweep_individual_clinical_only/evaluation/ensemble/` with ensemble statistics across seeds. (Note that the timestamp is excluded; this differs from what happens if you run abx_amr_simulator.analysis.evaluative_plots on a single experiment run, where the output will be saved in `analysis_output/lambda_sweep_individual_clinical_only_<TIMESTAMP>/evaluation/plots/`)

**Want more detailed Python analysis?** See Tutorial 4 for examples of programmatically loading models and extracting custom metrics. For most users, the CLI analysis tools provide everything needed to understand the lambda trade-off.

---

## Customization 2: Configure AMR Dynamics (Leaky Balloon)

The ABXAMREnv instance 
The AMR dynamics are modeled as a **leaky balloon**: prescriptions add "puff" (resistance), time adds "leak" (recovery).

**View the default environment config**:

```bash
cat configs/environment/default.yaml
```

Look for the `antibiotics_AMR_dict` section. Each antibiotic has:
- **leak**: Decay rate per timestep (∈ 0-1). Higher leak = faster resistance decay
- **flatness_parameter**: Controls sigmoid steepness (how quickly resistance increases in response to this antibiotic being prescribed). Higher = steeper sigmoid
- **permanent_residual_volume**: Floor resistance (can't drop below this)
- **initial_amr_level**: Starting resistance level (∈ [permanent_residual_volume, 1.0])

The `antibiotics_AMR_dict` must contain one entry per antibiotic, where the parameters for each antibiotic's leaky balloon is set separately.

### Example 1: Single Antibiotic with Steep Resistance Response to Prescribing

This example shows how to adjust the **flatness_parameter** to make resistance more responsive. First, let's visualize what the response curve looks like with different parameters.

Create `configs/environment/steep_amr.yaml`:

```yaml
num_patients_per_time_step: 5
max_time_steps: 500

antibiotic_names:
  - Antibiotic_A

antibiotics_AMR_dict:
  Antibiotic_A:
    leak: 0.95
    flatness_parameter: 8.0         # Much steeper sigmoid (higher = steeper)
    permanent_residual_volume: 0.0
    initial_amr_level: 0.0

action_mode: multidiscrete
include_steps_since_amr_update_in_obs: false
```

**Before running an experiment**, visualize the leaky balloon's response to understand the dynamics:

```bash
python -m abx_amr_simulator.utils.visualization.visualize_environment_behavior \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s environment-subconfig=configs/environment/steep_amr.yaml
```

This generates plots showing:
- How AMR levels respond to different puff sequences
- Sample random trajectories with observations and rewards
- Saved to `visualization_outputs/`

After understanding the dynamics, run the training experiment:

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s environment-subconfig=configs/environment/steep_amr.yaml \
  -p training.run_name=experiment_steep_amr
```

**What changed**: The resistance increases much more quickly in response to prescriptions. The agent will see resistance building and may adjust its strategy accordingly.

### Example 2: Two Antibiotics Without Crossresistance

Create `configs/environment/two_abx_no_crossresistance.yaml`:

```yaml
num_patients_per_time_step: 50
max_time_steps: 500

antibiotic_names:
  - Antibiotic_A
  - Antibiotic_B

antibiotics_AMR_dict:
  Antibiotic_A:
    leak: 0.95
    flatness_parameter: 4.0
    permanent_residual_volume: 0.0
    initial_amr_level: 0.0
  Antibiotic_B:
    leak: 0.95
    flatness_parameter: 4.0
    permanent_residual_volume: 0.0
    initial_amr_level: 0.0

crossresistance_matrix:                # No crossresistance: each antibiotic independent
  Antibiotic_A:
    Antibiotic_A: 1.0
    Antibiotic_B: 0.0            # Prescribing A doesn't affect B resistance
  Antibiotic_B:
    Antibiotic_A: 0.0
    Antibiotic_B: 1.0

action_mode: multidiscrete
include_steps_since_amr_update_in_obs: false
```

COPILOT: does 'visualize_environment_behavior' plot the leaky ballons for an environment yaml if there are multiple antibiotics? If it doesn't, it should be revised so that it does.

Run it:

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s environment-subconfig=environment/two_abx_no_crossresistance.yaml \
  -p training.run_name=experiment_two_abx_independent
```

**What changed**: The agent now sees two independent resistance dynamics. It can develop different prescribing strategies for each antibiotic.

### Example 3: Two Antibiotics With Moderate Crossresistance

Create `configs/environment/two_abx_crossresistance.yaml`:

```yaml
num_patients_per_time_step: 50
max_time_steps: 500

antibiotic_names:
  - Antibiotic_A
  - Antibiotic_B

antibiotics_AMR_dict:
  Antibiotic_A:
    leak: 0.95
    flatness_parameter: 4.0
    permanent_residual_volume: 0.0
    initial_amr_level: 0.0
  Antibiotic_B:
    leak: 0.95
    flatness_parameter: 4.0
    permanent_residual_volume: 0.0
    initial_amr_level: 0.0

crossresistance_matrix:                # Moderate crossresistance between antibiotics
  Antibiotic_A:
    Antibiotic_A: 1.0
    Antibiotic_B: 0.3            # Prescribing A increases B resistance by 30%
  Antibiotic_B:
    Antibiotic_A: 0.2            # Asymmetric: B → A crossresistance is weaker
    Antibiotic_B: 1.0

action_mode: multidiscrete
observation_mode: partial
include_steps_since_amr_update_in_obs: false
```

Run it:

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s environment-subconfig=environment/two_abx_crossresistance.yaml \
  -p training.run_name=experiment_two_abx_crossresistance
```

**What changed**: Prescribing Antibiotic A now has side effects—it increases resistance to Antibiotic B. The agent must learn that conservative use of A might preserve B for later. This creates an interesting strategic trade-off.

---

## Customization 3: Run a Parameter Sweep

Compare multiple configurations systematically using shell scripts. **This is the recommended pattern for running experiments at scale.**

Create a shell script `run_sweep.sh`:

```bash
#!/bin/bash

# Define sweep parameters
LAMBDAS=(0.0 0.3 0.5 0.7 1.0)
PATIENT_TYPES=("default" "high_risk")
SEEDS=(1 2 3)

for lambda in "${LAMBDAS[@]}"; do
  for patient_type in "${PATIENT_TYPES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_name="sweep_lambda${lambda}_patients_${patient_type}_seed${seed}"
      
      echo "Starting: $run_name"
      
      python -m abx_amr_simulator.training.train \
        --config configs/umbrella_configs/base_experiment.yaml \
        -p reward_calculator.lambda_weight=$lambda \
        -s patient_generator-subconfig=patient_generator/${patient_type}.yaml \
        -p training.seed=$seed \
        -p training.run_name=$run_name
    done
  done
done

echo "All experiments complete!"
```

Run the sweep:

```bash
bash run_sweep.sh
```

This will create **5 × 2 × 3 = 30** training runs, each exploring a different configuration.

After all runs complete, use the analysis tools to compare results:

```bash
python -m abx_amr_simulator.analysis.evaluative_plots --experiment-prefix sweep_
```

---

## Customization 4: Modify Patient Populations

By default, patients are **homogeneous** (all have the same attributes). You can introduce heterogeneity in two ways: via distributions or by changing which attributes are visible.

### Understanding the Default Patient Population

Open `configs/patient_generator/default.yaml`:

```yaml
visible_patient_attributes:
  - prob_infected

seed: null

distributions:
  prob_infected:
    type: constant
    value: 0.3
  benefit_value_multiplier:
    type: constant
    value: 1.0
  failure_value_multiplier:
    type: constant
    value: 1.0
  benefit_probability_multiplier:
    type: constant
    value: 1.0
  failure_probability_multiplier:
    type: constant
    value: 1.0
  recovery_without_treatment_prob:
    type: constant
    value: 0.1
```

**Key insight**: This creates a **completely homogeneous population** where:
- All attributes are **constants** (not random distributions)
- Every patient has 30% infection probability
- Every patient has 100% treatment benefit (1.0 multiplier)
- Only `prob_infected` is visible to the agent

This is the simplest case. All patients are identical, so the agent's prescribing strategy should be uniform.

### Example 1: Gaussian Distributions (Heterogeneous Population)

To add patient heterogeneity, switch distributions to Gaussian. Create `configs/patient_generator/heterogeneous.yaml`:

```yaml
visible_patient_attributes:
  - prob_infected
  - benefit_value_multiplier

seed: null

distributions:
  prob_infected:
    type: gaussian
    mean: 0.3
    std: 0.1              # Some patients 20%, others 40%
  benefit_value_multiplier:
    type: gaussian
    mean: 1.0
    std: 0.2              # Some patients low benefit, others high benefit
  failure_value_multiplier:
    type: gaussian
    mean: 1.0
    std: 0.15
  benefit_probability_multiplier:
    type: gaussian
    mean: 1.0
    std: 0.1
  failure_probability_multiplier:
    type: gaussian
    mean: 1.0
    std: 0.1
  recovery_without_treatment_prob:
    type: constant
    value: 0.1
```

Run it:

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s patient_generator-subconfig=patient_generator/heterogeneous.yaml \
  -p training.run_name=experiment_heterogeneous_patients
```

**What changed**: Patients now have varying infection rates and treatment benefits. The agent sees `prob_infected` and `benefit_value_multiplier` and must learn to prescribe differently for different patient types.

### Example 2: Control Visibility (What Agent Can See)

The `visible_patient_attributes` list controls what the agent observes. This is a critical experimental knob!

Create `configs/patient_generator/limited_visibility.yaml`:

```yaml
visible_patient_attributes:
  - prob_infected
  # Note: benefit_value_multiplier is NOT visible

seed: null

distributions:
  prob_infected:
    type: gaussian
    mean: 0.3
    std: 0.1
  benefit_value_multiplier:
    type: gaussian
    mean: 1.0
    std: 0.2              # Varies but invisible to agent
  failure_value_multiplier:
    type: gaussian
    mean: 1.0
    std: 0.15
  benefit_probability_multiplier:
    type: gaussian
    mean: 1.0
    std: 0.1
  failure_probability_multiplier:
    type: gaussian
    mean: 1.0
    std: 0.1
  recovery_without_treatment_prob:
    type: constant
    value: 0.1
```

This is useful for studying **information asymmetry**: Does the agent struggle without knowing treatment benefit? Compare against the full-visibility experiment to see the impact.

### Example 3: Mix Multiple Patient Populations (PatientGeneratorMixer)

For advanced scenarios, combine multiple PatientGenerators to create distinct subpopulations. Create `configs/patient_generator/mixed_populations.yaml`:

```yaml
type: mixer                         # Special type for mixing

components:
  - fraction: 0.7                 # 70% of patients from first generator
    config: patient_generator/homogeneous.yaml
  
  - fraction: 0.3                 # 30% of patients from second generator
    config: patient_generator/high_risk.yaml
```

Run it:

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s patient_generator-subconfig=patient_generator/mixed_populations.yaml \
  -p training.run_name=experiment_mixed_populations
```

**What changed**: Now the patient cohort at each timestep is composed of two distinct subgroups:
- 70% from the homogeneous population (30% infection, 100% benefit)
- 30% from the high-risk population (60% infection, 80% benefit)

The agent must learn different strategies for different patient types. `PatientGeneratorMixer` can mix any number of subpopulations and works exactly like a regular `PatientGenerator`—no special handling needed.

---

## Customization 5: Create Your Own Custom Subconfig

### Example: Custom Environment with 5 Patients and 2 Antibiotics

Create `configs/environment/custom_small.yaml`:

```yaml
num_patients_per_time_step: 5         # Smaller cohort
max_time_steps: 200                   # Shorter episodes

antibiotic_names:
  - Antibiotic_A
  - Antibiotic_B

antibiotics_AMR_dict:
  Antibiotic_A:
    leak: 0.95
    flatness_parameter: 4.0
    permanent_residual_volume: 0.0
    initial_amr_level: 0.0
  Antibiotic_B:
    leak: 0.95
    flatness_parameter: 4.0
    permanent_residual_volume: 0.0
    initial_amr_level: 0.0

action_mode: multidiscrete
include_steps_since_amr_update_in_obs: false
```

### Example: Custom Reward Calculator

Create `configs/reward_calculator/conservative.yaml`:

```yaml
lambda_weight: 0.7                # 70% AMR penalty, 30% individual reward (conservative)

per_antibiotic_rewards:
  Antibiotic_A:
    clinical_benefit_reward: 1.0
    clinical_failure_penalty: -2.0
    adverse_effect_penalty: -0.3
  Antibiotic_B:
    clinical_benefit_reward: 1.0
    clinical_failure_penalty: -2.0
    adverse_effect_penalty: -0.3

marginal_amr_penalty: -0.05           # Higher penalty per unit AMR
```

### Run with Both Custom Subconfigs

You can customize any subconfig (environment, reward_calculator, patient_generator, agent_algorithm) and mix-and-match them:

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -o environment_subconfig=environment/custom_small.yaml \
  -o reward_calculator_subconfig=reward_calculator/conservative.yaml \
  -o training.run_name=experiment_small_conservative
```

**Note**: You can use shorthand flags: `-p` for `--param-override` and `-s` for `--subconfig-override`:

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -s environment-subconfig=environment/custom_small.yaml \
  -s reward_calculator-subconfig=reward_calculator/conservative.yaml \
  -p training.run_name=experiment_small_conservative
```

**Modularity**: You can customize any component independently. Mix conservative reward with high-risk patients, steep AMR dynamics with clinical-focused rewards, etc. All subconfigs work together seamlessly.

---

## Troubleshooting

### "YAML parsing error"

Check your YAML syntax:
- No tabs (only spaces)
- Proper indentation (2 spaces per level)
- Colons followed by space before value

```bash
python -c "import yaml; yaml.safe_load(open('configs/environment/custom_small.yaml'))" && echo "Valid YAML"
```

### "Unknown parameter"

If you override a parameter that doesn't exist in the config:

```
ValueError: Unknown parameter: 'foo_bar'
```

Double-check the parameter name in the original config file.

### Config path not found

If you get a "file not found" error when pointing to a subconfig, use the full path:

```bash
# Instead of:
python -m abx_amr_simulator.training.train --config configs/umbrella_configs/base_experiment.yaml \
  -s environment-subconfig=environment/custom_small.yaml

# Use full absolute path:
python -m abx_amr_simulator.training.train --config /absolute/path/to/configs/umbrella_configs/base_experiment.yaml \
  -s environment-subconfig=/absolute/path/to/configs/environment/custom_small.yaml
```

Or use relative paths from the directory where you're running the command. This is especially useful if you're running training from a different working directory.

### Command-line shorthand

You can use short flags instead of long ones:
- `-p` = `--param-override` (for parameter values using dot notation)
- `-s` = `--subconfig-override` (for replacing entire subconfig files with format `{key}-subconfig=path/to/file.yaml`)

Example:
```bash
# Instead of:
python -m abx_amr_simulator.training.train --config base_experiment.yaml \
  --param-override training.run_name=my_run \
  --subconfig-override environment-subconfig=environment/custom.yaml \
  --param-override reward_calculator.lambda_weight=0.8

# Type:
python -m abx_amr_simulator.training.train --config base_experiment.yaml \
  -p training.run_name=my_run \
  -s environment-subconfig=environment/custom.yaml \
  -p reward_calculator.lambda_weight=0.8
```

---

## What's Next?

✅ You've customized and experimented with different configurations!

**Next steps**:
- **Tutorial 3**: Analyze your results with diagnostic and evaluative plots
- Compare different configurations and identify optimal settings
- Run larger parameter sweeps to find Pareto frontiers (clinical benefit vs. AMR)
- **Tutorial 4**: Use the Python API for programmatic analysis

---

## Key Takeaways

1. **Modular configs**: Swap patient distributions, reward functions, and environment settings independently
2. **Command-line overrides**: Use `--param-override` to avoid creating dozens of config files
3. **Lambda trade-off**: Experiment systematically to find the right balance for your use case
4. **AMR dynamics**: Leak rates and crossresistance control resistance accumulation patterns
5. **Reproducibility**: Seeds make results replicable across different machines and times
