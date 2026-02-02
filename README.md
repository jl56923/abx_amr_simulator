# ABX AMR Capacitor RL

**Reinforcement learning simulator for optimizing antibiotic prescribing under antimicrobial resistance (AMR) constraints.**

The core insight: prescribing antibiotics drives up AMR levels like "capacitors charging up," forcing agents to balance short-term clinical benefit against long-term community resistance burden.

## Quick Links

- **[Package Documentation](src/abx_amr_simulator/README.md)** ← Start here for installation and usage
- [Architecture Overview](#project-structure)
- [Environment Spec](docs/ENVIRONMENT_SPEC.md)
- [Configuration System](docs/CONFIG_SYSTEM.md)
- [Changelog](docs/CHANGELOG.md)
- [Tests](tests/README.md)

## Installation

```bash
git clone <repo-url>
cd abx_amr_simulator
pip install -e .
```

Then see [src/abx_amr_simulator/README.md](src/abx_amr_simulator/README.md) for setup instructions.

## Quick Start: GUI

Want to configure and run experiments interactively? Use the Streamlit GUI:

```bash
# Launch the Experiment Runner (configure & train)
abx-amr-simulator-experiment-runner

# In another terminal, launch the Experiment Viewer (analyze results)
abx-amr-simulator-experiment-viewer
```

Both apps are now accessible from any directory. For more options and troubleshooting, see [LAUNCHING_GUI.md](docs/LAUNCHING_GUI.md).

## Project Structure

```
abx_amr_simulator/
├── src/abx_amr_simulator/          # ← Main package (pip install -e .)
│   ├── core/                       # Domain classes (env, rewards, patients)
│   ├── training/                   # Training loop
│   ├── analysis/                   # Analysis tools (diagnostics, plotting)
│   ├── callbacks/                  # SB3 training callbacks
│   ├── utils/                      # Config, factories, registry
│   ├── gui/                        # Streamlit apps (experiment runner/viewer)
│   ├── wrappers.py                 # Environment wrappers
│   ├── formatters.py               # Debug output
│   ├── configs/                    # Bundled example/default configs
│   └── README.md                   # ← Package usage guide
└── pyproject.toml                  # Package config
```

## Key Concepts

### The Leaky Balloon Model

AMR resistance is modeled as a "leaky balloon":
- **Prescribing = puff**: Inflates balloon (increases resistance)
- **No prescribing = leak**: Balloon slowly deflates (resistance decays)
- **Sigmoid cap**: Resistance bounded in [0, 1]

Per-antibiotic dynamics allow cross-resistance and differential decay rates.

### Configuration System

Hierarchical YAML configs with command-line override:

```bash
python -m abx_amr_simulator.training.train \
  --config experiments/configs/umbrella_configs/base_experiment.yaml \
  -o "reward_calculator.lambda_weight=0.8" \
  -o "training.total_num_training_episodes=100"
```

See [CONFIG_SYSTEM.md](docs/CONFIG_SYSTEM.md) for details.

### Environment

- **Multi-patient per timestep** with heterogeneous infection probabilities
- **Multidiscrete action space**: prescribe/don't for each antibiotic
- **Observation**: Patient attributes (risk, treatment response multipliers) + AMR levels
- **Reward**: Composite of clinical benefit and AMR penalty (tunable λ trade-off)

See [ENVIRONMENT_SPEC.md](docs/ENVIRONMENT_SPEC.md) for full details.

## Getting Started

### 1. Install and set up configs

```bash
pip install -e .
python -c "from abx_amr_simulator.utils.config import setup_config_folders_with_defaults; setup_config_folders_with_defaults('.')"
```

### 2. Run a quick training job

```bash
python -m abx_amr_simulator.training.train \
  --config configs/umbrella_configs/base_experiment.yaml \
  -o "training.total_num_training_episodes=10"
```

### 3. View results

```bash
tensorboard --logdir results/*/logs
```

### 4. Launch GUI

```bash
# Option 1: Using console entry points (recommended)
abx-amr-simulator-experiment-runner          # Configure & run experiments
abx-amr-simulator-experiment-viewer          # Browse & analyze results

# Option 2: Direct Streamlit (from workspace/)
cd workspace && streamlit run ../src/abx_amr_simulator/gui/experiment_runner.py
```

**GUI Apps:**
- **Experiment Runner** (port 8501): Configure experiments, tune parameters, launch training
- **Experiment Viewer** (port 8502): Browse completed runs, analyze metrics, plot diagnostics

For detailed setup instructions, see [docs/LAUNCHING_GUI.md](docs/LAUNCHING_GUI.md).

---

## Testing

Run full test suite:

```bash
pytest tests/ -v
```

Expected: **226 tests pass** (see [tests/README.md](tests/README.md) for suite details).

---

## Architecture Highlights

### Clean Separation
- **Package** (`src/abx_amr_simulator/`): Reusable library code
- **Experiments** (`experiments/`): User configs and parameter sweep scripts
- **Results** (`results/`, `analysis_output/`): Generated outputs

### No Circular Dependencies
- `core/` is dependency-free (only stdlib + numpy/gymnasium)
- All imports are at module level (no deferred/circular imports)
- Shared types in `types.py` prevent circular import chains

### Backward Compatibility
- Old `abx_amr_env/` folder acts as compatibility shim
- Deprecation warnings guide users to new imports
- Full migration path without breaking existing code

### Extensibility
- `PatientGenerator`, `RewardCalculator` are protocols (duck-typed)
- Easy to create new patient distributions or reward functions
- Crossresistance matrices support multi-antibiotic resistance coupling

---

## Key Files & Modules

| File | Purpose |
|------|---------|
| `src/abx_amr_simulator/core/abx_amr_env.py` | Main environment (Gymnasium-compliant) |
| `src/abx_amr_simulator/core/leaky_balloon.py` | AMR dynamics model |
| `src/abx_amr_simulator/training/train.py` | Training entrypoint |
| `src/abx_amr_simulator/utils/config.py` | Config loading & merging |
| `src/abx_amr_simulator/utils/factories.py` | Object creation helpers |
| `src/abx_amr_simulator/callbacks/` | SB3 callbacks (eval, logging) |
| `src/abx_amr_simulator/gui/experiment_runner.py` | Interactive training UI |
| `src/abx_amr_simulator/gui/experiment_viewer.py` | Results browser & plots |
| `experiments/shell_scripts/` | Parameter sweep runners |

---

## Documentation Index

- [Configuration System](docs/CONFIG_SYSTEM.md) — YAML format, hierarchies, overrides
- [Environment Specification](docs/ENVIRONMENT_SPEC.md) — Observation/action spaces, reward details
- [Diagnostic Analysis](workspace/docs/diagnostic_analysis.md) — Phase 2 outputs and interpretation
- [Evaluative Plots](workspace/docs/evaluative_plots.md) — Phase 3 outputs and interpretation
- [Changelog](docs/CHANGELOG.md) — Release notes, architecture evolution
- [Tests](tests/README.md) — Suite layout and commands

---

## Citation

```bibtex
@software{abx_amr_simulator_2026,
  title = {ABX AMR Simulator: Reinforcement Learning for Antibiotic Stewardship},
  author = {Joyce Lee},
  year = {2026},
  url = {https://github.com/<owner>/abx_amr_simulator}
}
```

---

## License

[Specify your license here]

---

## Questions?

Refer to the package README: [src/abx_amr_simulator/README.md](src/abx_amr_simulator/README.md)
