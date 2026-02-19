# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
