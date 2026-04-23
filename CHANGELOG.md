# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`resolve_config_path` utility and `$CONFIG_BASE_FOLDER/` prefix support (Task 5, April 2026)**:
  - Added `resolve_config_path(path_str, base_dir)` to `abx_amr_simulator.utils.factories`.
    Handles three cases: `$CONFIG_BASE_FOLDER/`-prefixed paths (expanded from
    `ABX_AMR_CONFIG_BASE_FOLDER` env var), absolute paths (returned as-is), and plain
    relative paths (resolved against `base_dir`).  Raises `RuntimeError` with clear
    instructions when the env var is unset and the prefix is encountered.
  - `build_patient_generator_from_spec` now delegates `config_file` resolution to
    `resolve_config_path`, enabling mixer specs to use the portable
    `$CONFIG_BASE_FOLDER/configs/patient_generator/...` prefix instead of fragile
    deep relative paths.
  - Added tutorial `docs/tutorials/11_config_path_conventions.md` explaining the
    mechanism for both config authors and factory contributors.

### Fixed
- **`build_patient_generator_from_config` now handles `type: mixer` configs**:
  Previously, `build_patient_generator_from_config` in `marl_factories.py` only
  checked for a `plugin` key and fell through to `PatientGenerator(config=...)` for
  everything else. A mixer-type YAML (produced by commit 8301816) carried no `plugin`
  key, so the fallthrough reached `PatientGenerator.__init__`, which raised
  `ValueError: Missing required config key: 'visible_patient_attributes'` and caused
  all MARL tuning jobs that used a mixer patient-generator config to fail. The fix
  adds a `type: mixer` branch before the fallthrough that delegates to the existing
  `build_patient_generator_from_spec` utility (already the canonical handler for mixer
  specs in `factories.py`). Five new unit tests in
  `tests/unit/utils/test_marl_factories.py` cover the inline-mixer, file-reference-mixer,
  proportions, child count, and non-mixer fallthrough cases.

### Added
- **`TruePatient`/`ObservedPatient`/`Patient` type hierarchy** (Task 16, April 2026):
  - Introduced `TruePatient`, `ObservedPatient`, and `Patient` dataclasses in
    `src/abx_amr_simulator/core/types.py`.
  - `TruePatient`: immutable ground-truth state (all six patient attribute fields,
    `infection_status`, `abx_sensitivity_dict`).
  - `ObservedPatient`: wraps a `TruePatient` reference plus a
    `visible_attributes: Dict[str, float]` dict holding observed values by base
    attribute name (no `_obs` suffix). Replaces the previous flat `_obs`-suffixed
    attribute pattern.
  - `Patient`: top-level container holding `true_state: TruePatient`,
    `observations: List[ObservedPatient]`, and identity/routing fields
    (`patient_id`, `treated_by_agent`, `treated_in_locale`, `origin_locale`,
    `source_generator_index`). `primary_observation` property returns `observations[0]`.
  - `PatientGenerator.sample()` now returns `List[Patient]`; observation values are
    written into `ObservedPatient.visible_attributes` by base attribute name.
  - `PatientGeneratorMixer.sample()` updated accordingly; `PADDING_VALUE` sentinel
    is now written into `visible_attributes` for missing union attrs in
    heterogeneous-visibility mixers.
  - `PatientGenerator.observe()` reads from `primary_observation.visible_attributes`;
    `obs_dim()` and `obs_dim_uncovered()` unchanged.
  - All types exported from `abx_amr_simulator.core`.
  - Added `tests/unit/core/test_types.py` covering construction, property access,
    and equality for all three types.

- **LSTM belief-encoding integration test** (Task 18, April 2026):
  - Added `tests/integration/test_lstm_belief_encoding.py` — 10 tests covering
    `LSTMStateLogger` output (episode .npz files written, hidden-state shapes,
    `actual_amr_levels` passthrough via `OptionsWrapper` info dict) and the
    `probe_hidden_belief` pipeline (`load_episodes` / `fit_probe` produce
    finite R² values; best R² > 0.2 smoke-test threshold).
  - Uses a module-scoped fixture that trains a single-agent `RecurrentPPO` on a
    minimal `OptionsWrapper` (inline config, no workspace YAML dependencies) for
    120 macro steps; all 10 tests reuse the same training run.
  - Replaces the broken `tests/integration/test_lstm_belief_probing.py` in the
    repo root, which was a standalone script with no `def test_*` functions that
    executed a full training run at module-level (blocking pytest collection).

- **Test suite relocation** (April 2026):
  - Moved `test_abx_amr_parallel_env.py` from repo root `tests/unit/core/` into
    `tests/unit/core/` here (was incorrectly located in workspace tests; only
    imports from `abx_amr_simulator.core`).
  - Moved `test_observation_dimension_regression.py` from repo root `tests/unit/core/`
    into `tests/unit/core/` here for the same reason.
  - Moved `test_tune_postgres_storage.py` from repo root `tests/unit/` into
    `tests/unit/training/` here; removed now-redundant `sys.path` manipulation (package
    is importable from the submodule test environment without it).
  - `tests/integration/test_marl_wrapper_lpp_scenario.py` moved to repo root
    `tests/integration/` (this test loads real workspace YAML configs; it belongs
    in repo-level tests). Updated `_REPO_ROOT` path depth: `parents[4]` → `parents[2]`.

- **`build_patient_generator_from_spec` utility**: new package-level function in
  `abx_amr_simulator.utils` (and `abx_amr_simulator.utils.factories`) that builds
  a `PatientGenerator` or `PatientGeneratorMixer` from a plain config dict and an
  optional `base_dir` for relative `config_file` path resolution. Supports both
  file-based child specs (`config_file` key) and inline child specs. Previously
  this logic existed only inside `create_patient_generator` (for the `type: mixer`
  path) and was duplicated in workspace plugin subclasses that needed to compose a
  mixed base population internally. `create_patient_generator` now delegates its
  mixer-building to this utility, and plugin subclasses can call it directly
  instead of reimplementing path resolution and child instantiation.

### Fixed
- **Plugin path context injection**: all five factory call sites that load
  plugin components now forward path context to the plugin's config dict as
  `_config_dir_hint` before calling `load_plugin_component`. Previously the
  resolved umbrella/component config directory was used only to locate the
  plugin loader module itself but was never passed into the config dict seen by
  the plugin's `__init__`. This meant any plugin that needed to resolve relative
  file paths (e.g. loading sub-component YAML files) had to resort to brittle
  `Path(__file__)` workarounds. Affected call sites:
  - `create_reward_calculator` (uses `_reward_calculator_config_dir` or `_umbrella_config_dir`)
  - `create_patient_generator` (uses `_patient_generator_config_dir` or `_umbrella_config_dir`)
  - `create_amr_dynamics` (uses `_environment_config_dir` or `_umbrella_config_dir`)
  - `build_patient_generator_from_config` in `marl_factories.py` (uses the patient generator config file's own directory)
  - `build_reward_calculator_from_config` in `marl_factories.py` (uses the reward calculator config file's own directory)

  The caller's original config dict is never mutated; a shallow copy is made
  before injecting `_config_dir_hint`. Plugins that do not use the key simply
  ignore it. `load_plugin_component` itself is unchanged.

- **SA hyperparameter routing bug**: `create_agent()` in `factories.py` now
  merges tuned hyperparameters from `config['agent_algorithm']` into the
  algorithm-specific config section (e.g. `config['ppo']` for HRL_PPO,
  `config['recurrent_ppo']` for HRL_RPPO). Previously, tuned hyperparameters
  written by `train.py` under `agent_algorithm.*` were silently ignored for all
  HRL algorithms (and flat PPO/RecurrentPPO when using `--load-best-params`).
- **Gymnasium truncation semantics**: `ABXAMREnv.step()` now sets
  `terminated=False, truncated=True` when `max_time_steps` is reached, matching
  Gymnasium conventions and `ABXAMRParallelEnv` behavior. Previously both flags
  were True, which prevented SB3 from bootstrapping at truncation boundaries.
- **MARL config provenance**: `train_marl.py` now writes a second resolved
  config file (`marl_resolved_config.yaml`) after agents are built, containing
  the actual per-agent hyperparameters used during training. The original config
  save occurs before tuned hyperparameters are loaded.
- **Critical**: `OptionsWrapper._get_current_amr_levels()` was reading **true**
  AMR from balloon models, leaking privileged information into the HRL manager
  observation. The manager should only see **visible** (degraded) AMR, matching
  the information-degradation scenario configured by the experiment. Both
  `OptionsWrapper` and `MARLOptionsWrapper` now read `visible_amr_levels` from
  the base env, and the method has been renamed to
  `_get_current_visible_amr_levels()` to make the contract explicit.

### Added
- **MARL completion registry**: `train_marl.py` now follows the same
  registry pattern as the single-agent runner. On successful completion,
  `run_marl_training()` appends `run_name,timestamp` to
  `<results_dir>/.training_completed.txt` via `utils/registry.py`. The
  `skip_if_exists` check now consults this registry (with stale-entry
  cleanup via `validate_and_clean_registry`) instead of probing for
  `final_model_{aid}.zip` files. Interrupted runs (no registry entry) are
  correctly retried on next invocation. `_find_existing_timestamped_run_dir`
  removed as it is no longer needed.
- **HRL_RPPO support for MARL tuning**: `tune_marl_agent.py` is now
  algorithm-aware. Each trial reads the `algorithm` field (and optional
  `lstm_kwargs`) from the named agent's entry in the MARL config and
  dispatches to either `PPO` or `RecurrentPPO_Masked`. Previously the tuner
  hard-coded feedforward `PPO(policy="MlpPolicy", ...)` for every trial,
  which meant RPPO MARL experiments would receive hyperparameters tuned
  against the wrong policy class and search space.
- **Explicit `policy_kwargs.net_arch` plumbing in MARL agent factories**:
  `make_ppo_for_agent()` and `make_recurrent_ppo_for_agent()` in
  `train_marl.py` now accept an optional `net_arch` argument;
  `build_marl_managers_from_config()` in `utils/marl_factories.py` reads
  `policy_kwargs.net_arch` from each agent entry and forwards it. When
  absent, SB3's default applies (backward-compatible).
- **Canonical MARL tuning defaults**: Two new files under
  `src/abx_amr_simulator/tuning/defaults/`:
  - `hrl_ppo_marl_tuning_default.yaml` — canonical MARL-variant of
    `hrl_ppo_tuning_default.yaml`. Search ranges match the SA PPO defaults
    exactly; intentionally excludes `option_gamma` (shared across the
    `MARLOptionsWrapper`, not per-agent) and `batch_size` (taken from the
    training config). Uses `truncated_primitive_steps` as the budget.
  - `hrl_rppo_marl_tuning_default.yaml` — canonical MARL-variant of
    `hrl_rppo_tuning_default.yaml`. Same structural adjustments; search
    ranges match the SA RPPO defaults (notably the smaller `n_steps`
    range appropriate for recurrent agents).
- **HRL_RPPO support for MARL training**: `MARLTrainer` now supports both
  HRL_PPO and HRL_RPPO (recurrent) agents, including mixed configurations.
  Per-agent LSTM states are tracked across steps within episodes and reset at
  episode boundaries. New function `make_recurrent_ppo_for_agent()` in
  `train_marl.py`. Config supports per-agent `algorithm` and `lstm_kwargs`
  fields in `build_marl_managers_from_config()`.
- **MARL granular eval RPPO support**: `run_granular_eval_best_models_marl.py`
  now loads RPPO agents with `RecurrentPPO_Masked.load()` and tracks LSTM
  states during eval trajectory collection.
- **MARL eval RPPO support**: `run_marl_eval_episodes()` in
  `marl_callbacks.py` now tracks LSTM states for recurrent agents using the
  unified `predict(state=..., episode_start=...)` interface.
- **MARL option library validation**: `MARLOptionsWrapper.__init__()` now calls
  `validate_environment_compatibility()` on each agent's option library,
  matching the validation that the SA `OptionsWrapper` already performs.
  Incompatible option libraries now fail loudly at construction time.
- `write_aggregated_timeseries_csv()` helper in `metrics.py` — writes the
  output of `aggregate_trajectories()` as a CSV (columns: timestep, mean,
  median, p10, p25, p75, p90, iqm, n_active).
- `plot_metrics_from_collected_trajectories_ensemble()` in `metrics.py` now
  writes CSV files alongside each PNG plot for all per-antibiotic and scalar
  metrics.
- `_write_shared_amr_plot()` in `evaluative_plots_marl.py` now writes CSV
  files for shared actual and visible AMR levels per antibiotic.

### Changed
- **BREAKING**: Extended the MARL granular-eval NPZ schema in
  `src/abx_amr_simulator/analysis/run_granular_eval_best_models_marl.py` to
  persist the per-primitive-substep reward-calculator fields that were
  previously dropped on the floor:
  - `primitive_total_reward`, `primitive_overall_individual_reward_component`,
    `primitive_normalized_individual_reward`,
    `primitive_overall_community_reward_component`,
    `primitive_normalized_community_reward`,
    `primitive_count_clinical_benefits`, `primitive_count_clinical_failures`,
    `primitive_count_adverse_events`,
    `primitive_not_infected_no_treatment`, `primitive_not_infected_treated`,
    `primitive_infected_no_treatment`, and per-antibiotic
    `primitive_sensitive_infection_treated/{abx}` /
    `primitive_resistant_infection_treated/{abx}`.
  - All are shape `(macro_steps, max_substeps)` with NaN padding.
  - Old NPZs pre-dating this change will fail loud in
    `_build_agent_trajectory_payload` and must be regenerated by re-running
    `run_granular_eval_best_models_marl` against the original checkpoints.
- `src/abx_amr_simulator/analysis/evaluative_plots_marl.py` no longer
  hard-codes `count_clinical_benefits` / `count_clinical_failures` /
  `count_adverse_events`, community-reward components, or per-antibiotic
  resistant-infection-treated counts to zero. These are now read directly
  from the new NPZ fields, so MARL evaluative plots and
  `overall_outcomes_summary_*.json` now reflect real outcomes from the
  reward calculator rather than placeholder zeros. Previously emitted
  evaluative artifacts for MARL runs should be considered invalid and
  regenerated.

### Added
- Canonical package-level MARL isolation tuning orchestrator:
  - Added `src/abx_amr_simulator/training/tune_marl_agents.py` with
    `tune_marl_agents_sequentially`, a high-level entry point that tunes each
    agent in a MARL config independently in its own single-agent
    `ABXAMREnv + OptionsWrapper`. Agents are processed sequentially in config
    order (or a caller-supplied subset via `agent_ids`).
  - Supports `skip_if_exists`, base `seed`, and `n_workers` for distributed
    SQLite-backed Optuna studies (staggered-start subprocess pattern).
  - Artefacts land in the unified layout:
    `optimization_dir / experiment_folder / {agent_id} / best_params.json`.
  - `tune_marl_agents_sequentially` exported from `abx_amr_simulator.training`.
  - Added `tests/unit/training/test_tune_marl_agents.py` with 8 tests covering
    all-agent tuning, subset via `agent_ids`, `skip_if_exists`, missing config
    error, and 2-worker distributed mode (real `ABXAMREnv`, no mocks).
  - Added `docs/tutorials/14_marl_isolation_tuning.md` covering the isolation
    tuning rationale, config format, Python API, artifact layout, multi-worker
    mode, CLI-level tuning for SLURM, and connecting tuning results to training.

- Canonical package-level MARL granular re-evaluation utility and tests:
  - Added `src/abx_amr_simulator/analysis/run_granular_eval_best_models_marl.py` for deterministic post-hoc per-seed MARL granular trajectory regeneration (`eval_granular_best_model_{aid}.npz` per agent).
  - Added `tests/unit/analysis/test_run_granular_eval_best_models_marl.py` for focused AMR array emission and fail-loud validation behavior.

- MARL evaluative plotting analysis module and tests:
  - Added `src/abx_amr_simulator/analysis/evaluative_plots_marl.py` to generate per-agent evaluative plots and one shared AMR plot from per-agent granular MARL NPZ artifacts.
  - Added `tests/unit/analysis/test_evaluative_plots_marl.py` with sociable coverage for per-agent artifact generation, singleton shared-AMR plotting per prefix, and fail-loud validation of required NPZ fields.
- Shared plotting helper for percentile-band trajectory plots:
  - Added reusable `plot_with_bands(...)` in `src/abx_amr_simulator/utils/metrics.py` for consistent median+band plotting across analysis paths.

- Standardized plugin seam for core simulator subcomponents using config-driven loader modules:
  - New `utils/plugin_loader.py` with shared `load_plugin_component()` utility for module resolution (import path or filesystem path), loader function lookup, invocation, and fail-loud type validation.
  - Plugin branches added to `create_patient_generator()` and `create_reward_calculator()`.
  - New `create_amr_dynamics()` factory with plugin support and canonical AMR dynamics construction from merged config.
  - `ABXAMREnv` now supports optional `amr_dynamics_instances` injection for prebuilt AMR dynamics models.
  - `PatientGeneratorBase`, `RewardCalculatorBase`, and `AMRDynamicsBase` publicly exported from `abx_amr_simulator.core` for subclass/plugin authoring.
  - New tutorial `docs/tutorials/plugin_seam_guide.md` and integration fixtures/tests covering custom patient generator, reward calculator, and AMR dynamics plugin families.
- AMR dynamics abstraction: `AMRDynamicsBase` abstract base class enabling custom resistance models
  - `AMRDynamicsBase` ABC with `step()` and `reset()` abstract methods for extensibility
  - `AMR_LeakyBalloon` now inherits from `AMRDynamicsBase` with enhanced `reset()` input validation
  - `NAME` class constant for dynamics model identification
  - Comprehensive test suite for ABC contract validation (bounds, determinism, state initialization)
- Temporal features support in ABXAMREnv: optional prescription history tracking and AMR deltas in observations via `enable_temporal_features` and `temporal_windows` config parameters.
- MBPO core components and training loop scaffolding:
  - `DynamicsModel` for learning environment dynamics with MultiDiscrete action support
  - `TrajectoryReplayEnv` for replaying synthetic trajectories during PPO training
  - `MBPOAgent` with real-data collection, model training, synthetic rollouts, and replay training
  - Unit tests covering DynamicsModel, TrajectoryReplayEnv, and MBPOAgent core behaviors

### Changed
- MARL parallel environment info payloads now include true AMR levels during stepping:
  - Updated `src/abx_amr_simulator/core/abx_amr_parallel_env.py` so per-agent `info` includes `actual_amr_levels` alongside `visible_amr_levels`, enabling downstream shared-AMR evaluative plotting from granular trajectories.
- Refactored ensemble plotting internals in `src/abx_amr_simulator/utils/metrics.py` to use the new shared `plot_with_bands(...)` helper instead of duplicated local plotting logic.

- **BREAKING**: HRL options now use the standardized canonical/custom loading contract from JIRA-7
  - Valid `option_type` values are now exactly `block`, `alternation`, `heuristic`, and `custom`.
  - Canonical options (`block`, `alternation`, `heuristic`) now use package-owned loading only and must not include plugin loader fields.
  - Custom options must now use `option_type: custom` plus nested `plugin.loader_module` and `plugin.loader_function` fields.
  - All option families still load merged config from `option_subconfig_file` plus optional `config_params_override`.
  - Legacy flat top-level option `loader_module` / `loader_function` keys are no longer supported and now fail loudly with migration guidance.
- AMR dynamics canonical construction is now centralized in `create_amr_dynamics()` (factory layer), with `ABXAMREnv` retaining backward-compatible fallback behavior when injected instances are not provided.
- Recurrent LSTM probe summary aggregation now guards against low-variance target artifacts:
  - Per-antibiotic probe payloads now include `test_target_variance`, `r2_valid_for_aggregate`, and `r2_drop_reason`.
  - Extremely low-variance target slices are excluded from R² aggregation (MAE remains included), preventing pathological negative R² outliers from dominating `lstm_probe_summary.json`.
  - `lstm_probe_summary.json` now includes explicit count metadata (`num_r2_values_used`, `num_r2_values_dropped_low_variance`, `num_mae_values_used`) and robust R² fields (`mean_test_r2_median`, `iqr_test_r2`).
- **BREAKING**: HRL options refactored to use string-based antibiotic protocol
  - `OptionBase.decide()` now returns `np.ndarray` with dtype=object containing antibiotic name strings (e.g., 'A', 'B', 'no_treatment') instead of integer action indices
  - `OptionsWrapper` now handles conversion from antibiotic name strings to environment action indices
  - Validation moved from init-time to step-time with clearer error messages for invalid antibiotic names
  - All concrete option implementations updated: `BlockOption`, `AlternationOption`, `HeuristicWorker`
  - Simplifies option development: users work with human-readable names, no manual index mapping needed
  - All 89 HRL tests updated and passing with new protocol
  - Documentation updated: `docs/api_reference/option_abc.md` and `docs/tutorials/10_advanced_heuristic_worker_subclassing.md`
- `ABXAMREnv.amr_balloon_models` type hint updated from `Dict[str, AMR_LeakyBalloon]` to `Dict[str, AMRDynamicsBase]` for better extensibility

## [0.1.0] - 2026-02-01

### Added
- Initial public release of the ABX-AMR Simulator
- Gymnasium-compatible RL environment for antibiotic prescribing optimization
- Patient generator system with configurable infection rates and antibiotic sensitivities
- Reward calculator with individual/community outcome balancing via lambda parameter
- AMR dynamics models: Leaky Balloon and Discrete Capacitor implementations
- Streamlit GUI for experiment configuration, training, and result visualization
  - Experiment Runner for configuring and launching training runs
  - Experiment Viewer for analyzing results and generating plots
  - Unified `abx-amr-simulator-launch-gui` command to launch both apps
- Comprehensive configuration system using YAML with Hydra-style composition
- Command-line entry points for GUI applications
- Tutorial documentation for basic training and custom experiments
- Support for multiple RL algorithms via Stable-Baselines3 (A2C, PPO, RecurrentPPO)

### Features
- **Environment**: MultiDiscrete action space for per-patient antibiotic selection
- **Observations**: AMR levels, patient attributes, and historical metrics
- **Rewards**: Balanced individual patient outcomes and community AMR burden
- **Reproducibility**: Fully seeded RNG system for deterministic experiments
- **Extensibility**: Protocol-based architecture for custom components
- **Analysis**: Built-in diagnostic tools and visualization utilities

[Unreleased]: https://github.com/jl56923/abx_amr_simulator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jl56923/abx_amr_simulator/releases/tag/v0.1.0
