# Tutorial 13: MARL Training Quick Start

**Goal**: Learn how to configure and run multi-agent hierarchical reinforcement learning (MARL) experiments with `abx_amr_simulator`.

**Prerequisites**: Completed Tutorial 1 (Basic Training), Tutorial 5 (HRL Quick Start), and Tutorial 9 (Options Library Setup)

---

## Overview

The MARL path in `abx_amr_simulator` is for experiments where:

- multiple agents each control their own patient cohort,
- each agent uses its own HRL manager policy and option library,
- all agents interact through one shared AMR system.

This is not just "run two HRL agents side by side." The agents are coupled through shared resistance dynamics, so one agent's prescribing behavior can affect the future resistance seen by the others.

At a high level, MARL training works like this:

1. Build one shared parallel environment with global AMR dynamics
2. Build one option library per agent
3. Build one PPO manager per agent
4. Wrap the environment with `MARLOptionsWrapper` so each agent can execute options asynchronously
5. Train with `python -m abx_amr_simulator.training.train_marl`

Important differences from single-agent HRL:

- the config format is different,
- the training entrypoint is different,
- checkpoints are saved per agent,
- option execution is asynchronous across agents,
- only MARL HRL PPO is currently supported in this canonical training path.

If you are starting from an empty experiment directory, create the bundled config and option scaffolds first:

```bash
python -c "from abx_amr_simulator.utils import setup_config_folders_with_defaults; from pathlib import Path; setup_config_folders_with_defaults(Path('.'))"
python -c "from abx_amr_simulator.hrl import setup_options_folders_with_defaults; from pathlib import Path; setup_options_folders_with_defaults(Path('.'))"
```

That gives you a package-owned starter MARL config at `configs/marl/minimal_two_agent.yaml` plus the default option library under `options/option_libraries/default_deterministic.yaml`.

---

## 1. What MARL Means in This Package

In the single-agent HRL path:

- one manager selects options,
- one environment emits observations and rewards,
- one checkpoint stream is produced.

In the MARL path:

- each agent has its own manager,
- each agent has its own patient generator and reward calculator,
- all agents share the same AMR balloons and episode clock,
- each agent writes its own `best_model`, `final_model`, and periodic checkpoints.

The training loop advances primitive time until the next option-completion event. When an agent's option completes, that agent receives a manager-level transition and can choose a new option. Agents whose options are still running do not act again yet.

This means MARL training is:

- **agent-local at the policy level**,
- **shared at the AMR/environment level**,
- **event-driven at the rollout level**.

---

## 2. Minimal MARL Config Shape

The MARL trainer does not use the single-agent umbrella config format. Instead, it expects one dedicated MARL YAML with two top-level sections:

- `environment`
- `training`

Minimal runnable example:

```yaml
environment:
  shared:
    antibiotics_AMR_dict:
      A:
        leak: 0.05
        flatness_parameter: 1.0
        permanent_residual_volume: 0.0
        initial_amr_level: 0.0
      B:
        leak: 0.05
        flatness_parameter: 1.0
        permanent_residual_volume: 0.0
        initial_amr_level: 0.0
    max_time_steps: 100

  agents:
    - agent_id: agent_0
      n_patients: 3
      patient_generator:
        prob_infected:
          prob_dist: {type: constant, value: 0.8, mu: null, sigma: null}
          obs_bias_multiplier: 1.0
          obs_noise_one_std_dev: 0.0
          obs_noise_std_dev_fraction: 0.0
          clipping_bounds: [0.0, 1.0]
        benefit_value_multiplier:
          prob_dist: {type: constant, value: 1.0, mu: null, sigma: null}
          obs_bias_multiplier: 1.0
          obs_noise_one_std_dev: 0.0
          obs_noise_std_dev_fraction: 0.0
          clipping_bounds: [0.0, null]
        failure_value_multiplier:
          prob_dist: {type: constant, value: 1.0, mu: null, sigma: null}
          obs_bias_multiplier: 1.0
          obs_noise_one_std_dev: 0.0
          obs_noise_std_dev_fraction: 0.0
          clipping_bounds: [0.0, null]
        benefit_probability_multiplier:
          prob_dist: {type: constant, value: 1.0, mu: null, sigma: null}
          obs_bias_multiplier: 1.0
          obs_noise_one_std_dev: 0.0
          obs_noise_std_dev_fraction: 0.0
          clipping_bounds: [0.0, null]
        failure_probability_multiplier:
          prob_dist: {type: constant, value: 1.0, mu: null, sigma: null}
          obs_bias_multiplier: 1.0
          obs_noise_one_std_dev: 0.0
          obs_noise_std_dev_fraction: 0.0
          clipping_bounds: [0.0, null]
        recovery_without_treatment_prob:
          prob_dist: {type: constant, value: 0.0, mu: null, sigma: null}
          obs_bias_multiplier: 1.0
          obs_noise_one_std_dev: 0.0
          obs_noise_std_dev_fraction: 0.0
          clipping_bounds: [0.0, 1.0]
        visible_patient_attributes:
          - prob_infected
      reward_calculator:
        abx_clinical_reward_penalties_info_dict:
          clinical_benefit_reward: 10.0
          clinical_benefit_probability: 1.0
          clinical_failure_penalty: -10.0
          clinical_failure_probability: 1.0
          abx_adverse_effects_info:
            A:
              adverse_effect_penalty: -2.0
              adverse_effect_probability: 0.1
            B:
              adverse_effect_penalty: -3.0
              adverse_effect_probability: 0.15
        lambda_weight: 0.0
        seed: null
      option_library: default_deterministic.yaml

    - agent_id: agent_1
      n_patients: 4
      patient_generator: patient_generators/agent_1_pg.yaml
      reward_calculator: reward_calculators/agent_1_rc.yaml
      option_library: default_deterministic.yaml

training:
  n_steps: 256
  batch_size: 64
  n_epochs: 10
  learning_rate: 0.0003
  total_primitive_steps: 500000
  option_gamma: 0.99
  eval_freq_episodes: 100
  save_freq_episodes: 100
  n_eval_episodes: 5
  seed: 42
```

---

## 3. Required Sections and What They Mean

## `environment.shared`

This section defines what all agents share:

- `antibiotics_AMR_dict`
- `max_time_steps`
- optional cross-agent AMR settings such as `crossresistance_matrix`
- optional observation degradation settings such as delayed/noisy visible AMR

Think of this as the global environment state.

## `environment.agents`

This is a list of agent-specific entries. Every entry must define:

- `agent_id`
- `n_patients`
- `patient_generator`
- `reward_calculator`
- `option_library`

The important design point is that MARL lets each agent have its own:

- patient cohort definition,
- reward function,
- option menu.

## `training`

The canonical MARL training fields are:

- `n_steps`
- `batch_size`
- `n_epochs`
- `learning_rate`
- `total_primitive_steps`
- `option_gamma`
- `eval_freq_episodes`
- `save_freq_episodes`
- `n_eval_episodes`
- `seed`

Important cadence behavior:

- `eval_freq_episodes` controls how often evaluation runs
- `save_freq_episodes` controls how often periodic checkpoints are written
- `best_model_{agent_id}.zip` is updated only on eval cycles when reward improves

If `save_freq_episodes` is omitted, MARL defaults to using `eval_freq_episodes` for periodic checkpointing.

---

## 4. Component Wiring Patterns

MARL supports the same core component patterns as the single-agent codebase, but they are declared per agent.

## Inline component configs

Use inline dicts when you want a self-contained MARL YAML:

```yaml
patient_generator:
  prob_infected:
    prob_dist: {type: constant, value: 0.8, mu: null, sigma: null}
    obs_bias_multiplier: 1.0
    obs_noise_one_std_dev: 0.0
    obs_noise_std_dev_fraction: 0.0
    clipping_bounds: [0.0, 1.0]
  visible_patient_attributes:
    - prob_infected
```

This is the easiest pattern for a minimal tutorial or small prototype.

## File-backed component configs

Use file paths when you want reusable subconfigs:

```yaml
patient_generator: patient_generators/agent_1_pg.yaml
reward_calculator: reward_calculators/agent_1_rc.yaml
```

Relative paths are resolved relative to the MARL config file.

## Plugin-backed components

MARL also supports plugin-backed patient generators and reward calculators through the standard plugin seam:

```yaml
patient_generator:
  plugin:
    loader_module: ./plugins/custom_pg.py
    loader_function: load_patient_generator_component
  visible_patient_attributes:
    - prob_infected
  # ... other patient generator config fields ...
```

```yaml
reward_calculator:
  plugin:
    loader_module: ./plugins/custom_rc.py
    loader_function: load_reward_calculator_component
  # ... reward calculator config fields ...
```

Path rules:

- relative component file paths are resolved relative to the MARL YAML file,
- relative plugin loader paths are also resolved relative to the original config location before the resolved run config is saved,
- relative option library paths are resolved relative to the MARL YAML file.

If these paths are wrong, training fails loudly before proceeding.

For more background, see [plugin_seam_guide.md](plugin_seam_guide.md).

---

## 5. Option Libraries in MARL

Each agent must have an `option_library`.

Example:

```yaml
option_library: ../../options/option_libraries/default_deterministic.yaml
```

or:

```yaml
option_library: ../../options/option_libraries/my_agent_specific_library.yaml
```

Each option library is loaded independently for each agent, but all libraries must agree on the antibiotic action mapping. In practice, that means all agents must operate over the same antibiotic set.

The scaffolded `configs/marl/minimal_two_agent.yaml` uses this `../../options/...` pattern because MARL option-library paths are resolved relative to the MARL YAML file itself.

If you need to build custom option menus, see [09_options_library_setup.md](09_options_library_setup.md).

---

## 6. Run Training

From your project root:

```bash
python -m abx_amr_simulator.training.train_marl \
  --marl-config $(pwd)/configs/marl/minimal_two_agent.yaml \
  --results-dir $(pwd)/results \
  --run-name marl_demo_seed42 \
  --seed 42
```

What this does:

1. loads the MARL YAML,
2. applies seed and any CLI overrides,
3. resolves relative config/plugin/library paths,
4. writes a resolved `marl_full_agents_env_config.yaml` into the run folder,
5. rebuilds the training objects from that saved config,
6. trains one PPO manager per agent.

## CLI overrides

Use `-p` for dot-path overrides:

```bash
python -m abx_amr_simulator.training.train_marl \
  --marl-config $(pwd)/configs/marl/minimal_two_agent.yaml \
  --results-dir $(pwd)/results \
  --run-name marl_override_demo \
  --seed 7 \
  -p environment.agents.0.n_patients=5 \
  -p training.total_primitive_steps=100000
```

Note: override paths must already exist in the config. This override system is fail-loud and does not silently create new keys.

## Resume semantics with `--skip-if-exists`

If all `final_model_{agent_id}.zip` files already exist, training can be skipped:

```bash
python -m abx_amr_simulator.training.train_marl \
  --marl-config $(pwd)/configs/marl/minimal_two_agent.yaml \
  --results-dir $(pwd)/results \
  --run-name marl_demo_seed42 \
  --seed 42 \
  --skip-if-exists
```

## Optional per-agent hyperparameter initialization

You can also provide per-agent initialization metadata with `--agent-init-params`:

```bash
python -m abx_amr_simulator.training.train_marl \
  --marl-config $(pwd)/configs/marl/minimal_two_agent.yaml \
  --results-dir $(pwd)/results \
  --run-name marl_demo_seed42 \
  --seed 42 \
  --agent-init-params $(pwd)/agent_init_params.json
```

This is mainly useful when downstream orchestration scripts initialize agents from prior tuning results.

---

## 7. Understand the Output Folder

Expected run layout:

```text
results/
└── marl_demo_seed42/
    ├── marl_full_agents_env_config.yaml
    └── checkpoints/
        ├── agent_0_checkpoint_100000.zip
        ├── agent_1_checkpoint_100000.zip
        ├── best_model_agent_0.zip
        ├── best_model_agent_1.zip
        ├── final_model_agent_0.zip
        └── final_model_agent_1.zip
```

Key files:

- `marl_full_agents_env_config.yaml`
  - resolved run-local config snapshot used for replay and evaluation
- `best_model_{agent_id}.zip`
  - best eval checkpoint for that agent
- `final_model_{agent_id}.zip`
  - model at end of training for that agent
- `{agent_id}_checkpoint_{primitive_steps}.zip`
  - periodic checkpoint for that agent

Why there are more checkpoint files than in single-agent runs:

- periodic checkpoints are written per agent,
- best models are tracked per agent,
- checkpoint count depends on `save_freq_episodes`, not just total training budget.

If you want fewer periodic checkpoints, increase `save_freq_episodes`.

---

## 8. How MARL Evaluation Works

During training, MARL evaluation:

- runs deterministic eval episodes through the MARL wrapper,
- computes one mean reward per agent,
- updates `best_model_{agent_id}.zip` independently.

This means one agent can improve while another does not. There is no single global `best_model.zip` in the MARL path.

When interpreting outputs, always think in terms of per-agent artifacts.

---

## 9. Troubleshooting

## "Option library not found"

Checks:

- confirm `option_library` path exists,
- confirm relative paths are written relative to the MARL YAML,
- confirm the saved run config contains an absolute or correctly resolved path.

## "Plugin loader module is not importable and filesystem path does not exist"

Checks:

- confirm `plugin.loader_module` points to a real module or file,
- confirm relative plugin paths are authored relative to the config file that contains them,
- confirm the loader function name matches the function defined in the plugin file.

## "Missing n_steps entry" or rollout buffer mismatch errors

This usually means the PPO construction and trainer expectations disagree about `n_steps`. Use one consistent `training.n_steps` value unless you are intentionally doing per-agent customization through programmatic entrypoints.

## Too many checkpoint files

This is controlled by `save_freq_episodes`.

Examples:

- `eval_freq_episodes: 100`, `save_freq_episodes: 100` gives sparse checkpointing similar to single-agent HRL runs,
- `eval_freq_episodes: 5`, `save_freq_episodes: 5` gives very frequent per-agent checkpointing.

## Override fails with `KeyError`

The MARL override system only updates existing keys. If you need a new key such as `save_freq_episodes`, add it to the YAML first rather than relying on `-p` to create it.

---

## 10. Key Takeaways

1. MARL uses a dedicated config schema, not the single-agent umbrella config schema.
2. Each agent gets its own patient generator, reward calculator, option library, and PPO manager.
3. All agents still share one AMR system and one episode clock.
4. Training is run through `python -m abx_amr_simulator.training.train_marl`.
5. Checkpoints and best models are written per agent.
6. `eval_freq_episodes` and `save_freq_episodes` are separate controls.
7. Relative component/plugin/library paths are resolved relative to the MARL config location and fail loudly when invalid.

---

## Next Tutorials

- [14_marl_isolation_tuning.md](14_marl_isolation_tuning.md) — how to tune PPO
  hyperparameters for MARL experiments using per-agent isolation tuning
- [05_hrl_quickstart.md](05_hrl_quickstart.md)
- [09_options_library_setup.md](09_options_library_setup.md)
- [12_callbacks_and_logging.md](12_callbacks_and_logging.md)
- [plugin_seam_guide.md](plugin_seam_guide.md)