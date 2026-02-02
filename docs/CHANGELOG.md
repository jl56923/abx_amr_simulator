# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

#### Patient Infection/Sensitivity Architecture Refactoring (January 31, 2026)

- **Moved infection status and antibiotic sensitivity determination from RewardCalculator to PatientGenerator**
  - **Problem Fixed**: RewardCalculator was incorrectly sampling infection status and sensitivity matrix from `visible_amr_levels` (noisy/delayed) instead of true AMR levels, causing silent bugs when visible ≠ true AMR
  - **Architectural Improvement**: Separated reward calculation (RewardCalculator's job) from patient state determination (PatientGenerator's job)
  - **Blocks Removed**: Enables Paper 2 experiments requiring patient-specific sensitivity attributes

- **Core Changes**:
  - **Extended Patient Dataclass** (`src/abx_amr_simulator/core/types.py`):
    - Added `infection_status: bool` - ground truth infection state
    - Added `abx_sensitivity_dict: Dict[str, bool]` - per-antibiotic sensitivity mapping
    - No observed versions added (these attributes remain hidden from agent)
  
  - **Updated PatientGenerator** (`src/abx_amr_simulator/core/patient_generator.py`):
    - Modified `sample()` signature to require `true_amr_levels: Dict[str, float]` parameter
    - Now samples `infection_status` from Bernoulli(prob_infected)
    - Now samples `abx_sensitivity_dict[abx]` from Bernoulli(1 - true_amr_level[abx])  
    - Added new fields to `PROVIDES_ATTRIBUTES` class variable
    - **Critical Fix**: Sensitivity now computed from **true AMR levels**, not visible
  
  - **Simplified RewardCalculator** (`src/abx_amr_simulator/core/reward_calculator.py`):
    - Added `'infection_status'` and `'abx_sensitivity_dict'` to `REQUIRED_PATIENT_ATTRS`
    - Removed stochastic sampling of infection status (lines 602-603)
    - Removed buggy sampling of sensitivity matrix from visible_amr_levels (lines 611-617)
    - Now reads `patients_actually_infected = [p.infection_status for p in patients]`
    - Now reads sensitivity from `patient.abx_sensitivity_dict[antibiotic_name]`
  
  - **Environment Threading** (`src/abx_amr_simulator/core/abx_amr_env.py`):
    - `reset()` now computes true AMR levels and passes to patient_generator.sample()
    - `step()` does the same for new cohort sampling
    - Updated docstrings to reflect new signature

- **Test Updates**: Updated 20+ test files to match new signatures
  - Added TRUE_AMR_LEVELS constants to test modules
  - Updated all `.sample()` calls to include `true_amr_levels` parameter
  - Added new Patient fields to all Patient instantiations in tests
  - Fixed `test_abx_env.py` to use correct info dict key (`delta_visible_amr_per_antibiotic`)
  - Fixed `test_metrics_ensemble.py` monkeypatch to use correct method names

- **Files Modified**:
  - Core: `types.py`, `patient_generator.py`, `reward_calculator.py`, `abx_amr_env.py`, `base_patient_generator.py`, `protocols.py`
  - Tests: `test_patient_generator_mixer.py`, `test_patient_generator.py`, `test_baseline_infection.py`, `test_gui_phase1_integration.py`, `test_reward_model.py`, `test_expected_rewards.py`, `test_reward_calculator.py`, `test_rng_ownership.py`, `test_patient_logging.py`, `test_abx_env.py`, `test_metrics_ensemble.py`

- **Tests**: All 213 unit tests pass ✅, All 78 integration tests pass ✅

- **Impact**: 
  - **Bug Fix**: Sensitivity now correctly computed from true AMR instead of potentially stale/noisy visible AMR
  - **Cleaner Architecture**: Clear separation between state generation (PatientGenerator) and reward calculation (RewardCalculator)
  - **Extensibility**: Enables Paper 2 experiments with patient-specific attributes without modifying RewardCalculator
  - **Correctness**: Eliminates silent failure mode where visible ≠ true AMR led to incorrect sensitivity calculations

#### Reward Calculation: Clarified Visible AMR Usage in Documentation (January 30, 2026)

- **Clarified that reward calculation uses visible (observed) AMR levels for clinical authenticity**
  - Investigation revealed code already correctly used `visible_amr_levels` parameter throughout
  - Documentation was misleading, claiming rewards used "true AMR" or "actual AMR" when they actually used visible (observed) AMR
  - Updated 7 docstrings in `reward_calculator.py` to accurately reflect visible AMR usage
  - Updated 3 test docstrings in `test_reward_calculator.py` to match implementation reality

- **Rationale for Visible AMR Usage**:
  - **Information Leak Avoidance**: Agents observe noisy AMR data and get rewards based on same noisy signals (no ground truth leak through reward channel)
  - **Clinician Authenticity**: Real clinicians work with imperfect AMR data (outdated antibiograms, incomplete resistance data); using visible AMR reflects this realistic constraint
  - **POMDP Consistency**: Agent observes noisy state and receives feedback from observable signals, maintaining partial observability design
  - **Patient Outcomes**: Clinical benefit/failure rewards correctly use ground truth (patients either get better or don't based on reality), but AMR feedback comes from surveillance data (observable and noisy)

- **Documentation Changes**:
  - `calculate_community_reward()`: Changed "true AMR burden" → "visible (observed) AMR burden"
  - `calculate_reward()`: Clarified `visible_amr_levels` used for both sensitivity calculation AND community penalty
  - `calculate_expected_reward()`: Removed duplicate parameter documentation, clarified visible AMR usage
  - Comment at line ~786: Changed "actual AMR levels" → "visible AMR levels (not ground truth)"
  - Test docstrings: Updated `test_community_reward_uses_true_amr` and `test_individual_reward_uses_true_delta_amr` to reflect visible AMR usage

- **Files Modified**:
  - `src/abx_amr_simulator/core/reward_calculator.py`: 7 docstrings updated (no code changes)
  - `tests/unit/rewards/test_reward_calculator.py`: 3 test docstrings and comments updated

- **Tests**: All 34 reward tests pass (31 unit + 3 integration); no functional changes required

- **Impact**: Developers now understand which reward signals use visible data (AMR) vs ground truth (patient outcomes); eliminates confusion about information flow in POMDP design

#### Reward Normalization: Removed Optional Unnormalized Mode (January 30, 2026)

- **Made reward normalization mandatory and always-on**
  - Removed `normalize_clinical_reward_penalties` configuration flag from RewardCalculator
  - All rewards are now permanently normalized by `Θ = max(|B|, |F|, |A_a|)` at initialization
  - Rationale: Empirical experiments show unnormalized mode doesn't train effectively; normalization enables lambda to have semantic meaning as true trade-off between individual and community outcomes
  - Without normalization, lambda is scale-dependent and opaque; with normalization, lambda ∈ [0,1] has consistent interpretation across different reward parameter sets
  - Simplifies code by removing conditional branches and reduces configuration complexity

- **Code Changes**:
  - Removed error check for obsolete `normalize_clinical_reward_penalties` key in `__init__`
  - Updated comment describing dictionary initialization (normalization now always active)
  - Updated test expectations to match normalized reward values (factor of 10× reduction in raw values)

- **Documentation Updates**:
  - Updated `docs/ENVIRONMENT_SPEC.md` to clarify normalization is always active
  - Updated `docs/PHASE_1_COMPLETION_SUMMARY.md` to reflect always-on normalization
  - Removed references to `normalize_clinical_reward_penalties` from all example configs

- **Files Modified**:
  - `src/abx_amr_simulator/core/reward_calculator.py`: Removed obsolete flag check, updated comments
  - `tests/configs/reward_calculator/default.yaml`: Removed flag from test config
  - `tests/validate_tutorial_2.sh`: Removed flag from tutorial validation
  - `workspace/experiments/shell_scripts/test_set1_incr_entropy.sh`: Removed flag from shell script
  - `workspace/scratch/play_with_abx_amr_env.py`: Updated to use modern RewardCalculator API
  - `docs/ENVIRONMENT_SPEC.md`, `docs/PHASE_1_COMPLETION_SUMMARY.md`: Updated documentation
  - All unit and integration tests: Updated expected reward values to match normalized scale

- **Tests**: All 34 reward tests pass with updated expectations; no functionality changes, only scaling updates

- **Impact**: Cleaner codebase with simpler reward semantics; lambda now has consistent interpretation regardless of reward parameter magnitudes

### Added

#### GUI: Unified `launch-apps` Entry Point with Documentation (January 28, 2026)

- **Added unified command to launch both GUI apps simultaneously**
  - New entry point `launch-apps` available after `pip install -e .`
  - Launches Experiment Runner (port 8501) and Experiment Viewer (port 8502) together
  - Automatically monitors for completed experiments and auto-focuses viewer when new results available
  - Supports optional `--results-dir /path/to/results` argument (consistent with individual entry points)
  - Enhanced `launch_apps.py` with argparse support and workspace directory resolution
  - Preserved existing browser automation and app monitoring features from original implementation
  
- **Enhanced documentation for all GUI launch methods**
  - Rewrote `docs/LAUNCHING_GUI.md` with `launch-apps` as primary recommended option
  - Added "Apps Overview" section explaining Runner and Viewer purposes and typical workflows
  - Expanded troubleshooting with "Port Already in Use" section including solutions with `lsof` and `--server.port`
  - Added examples for all three entry point usage patterns
  - Clarified default directory resolution and working directory behavior
  - Updated COPILOT_TODO.md with complete summary of entry point infrastructure (three entry points, all features)

- **Files Modified**:
  - `src/abx_amr_simulator/gui/launch_apps.py`: 
    - Added `argparse` support for optional `--results-dir` argument
    - Added `get_workspace_dir()` function to resolve workspace path
    - Updated `start_streamlit_app()` signature to accept `results_dir` parameter and set `ABX_RESULTS_DIR` env var
    - Updated `main()` to parse arguments, validate/create directory, pass to both app launches
    - Changed working directory to `workspace/` before launching (consistent with entry point scripts)
  - `pyproject.toml`: Added third entry point `launch-apps = "abx_amr_simulator.gui.launch_apps:main"`
  - `docs/LAUNCHING_GUI.md`: Complete rewrite (234 lines) with `launch-apps` as primary option, Apps Overview, enhanced troubleshooting
  - `docs/COPILOT_TODO.md`: Updated entry points section with summary of all THREE entry points and enhanced documentation

- **User Impact**: Users can now launch entire GUI environment with single command `launch-apps`; documentation clearly explains when to use `launch-apps` vs individual entry points vs manual launch methods

### Fixed

#### Reward Calculation: Switch to Ground-Truth (Actual) Values Throughout (January 30, 2026)

- **Refactored reward calculation to use ground-truth (actual) values instead of observed/visible values**
  - **Architectural Change**: RewardCalculator now always receives and uses TRUE patient attributes and AMR levels, never observed/noisy versions. Agent observes potentially biased or noisy versions, but rewards are anchored to ground truth.
  - **Rationale**: This creates a meaningful partial observability challenge for recurrent RL agents:
    1. **Realistic information asymmetry**: Mirrors clinical reality where clinicians make decisions on imperfect information but outcomes depend on ground truth
    2. **Incentivizes memory/inference**: Agents must use recurrent architectures (LSTM/GRU) to filter noise and infer hidden state from observations + reward signal
    3. **Prevents reward hacking**: Agent cannot game rewards by exploiting noisy observations since rewards are anchored to reality
    4. **Enables study of information value**: Research questions like "How much observation noise can recurrent policies tolerate?" become answerable
  
  - **Changes**:
    - Renamed parameters throughout: `visible_amr_levels` → `actual_amr_levels`, `delta_visible_amr_per_antibiotic` → `delta_actual_amr_per_antibiotic`
    - Updated `calculate_reward()`, `calculate_expected_reward()`, `calculate_community_reward()`, and `calculate_individual_reward()` method signatures
    - Updated environment call site in `abx_amr_env.py` to pass ground-truth AMR levels from leaky balloon internals
    - Updated docstrings to emphasize "TRUE attribute values (not observed/noisy versions)"
    - Fixed bug in `abx_amr_env.py` line 732: stale `delta_visible_amr_per_antibiotic` reference (now `delta_actual_amr_per_antibiotic`)
  
  - **Testing**:
    - Fixed 28 existing reward tests across 4 test files (parameter name updates)
    - Added 4 new unit tests validating ground-truth behavior:
      - `test_patient_attributes_use_true_values()`: Rewards identical for identical true attributes despite different observations
      - `test_community_reward_uses_true_amr()`: Community penalty scales with actual_amr, not visible_amr
      - `test_sensitivity_calculation_uses_true_amr()`: Treatment outcome depends on actual resistance, not observed
      - `test_individual_reward_uses_true_delta_amr()`: Epsilon penalty uses actual AMR changes, not visible changes
    - Added 3 new integration tests validating full environment pipeline:
      - `test_environment_rewards_use_true_patient_attributes()`: Noisy observations don't affect reward calculation
      - `test_environment_rewards_use_true_amr_levels()`: Rewards reflect actual balloon state, not observed AMR
      - `test_environment_rewards_consistent_with_ground_truth()`: Multi-step episodes produce valid ground-truth-based rewards
    - **Test Results**: 35 total tests passing (32 unit + 3 integration), ~4.9s runtime
  
  - **Documentation**:
    - Added comprehensive "True vs Observed Values in Reward Calculation" section to `docs/ENVIRONMENT_SPEC.md`
    - Includes architectural principle, implementation details, configuration examples, testing approach, training implications, and common pitfalls
    - Emphasizes that this design is intentional and documents why recurrent architectures are recommended
  
  - **Files Modified**:
    - `src/abx_amr_simulator/core/reward_calculator.py`: Updated 5 method signatures and docstrings
    - `src/abx_amr_simulator/core/abx_amr_env.py`: Updated environment call site; fixed stale parameter name bug (line 732)
    - `tests/unit/rewards/test_reward_calculator.py`: Added 4 new tests
    - `tests/integration/test_true_vs_observed_values.py`: New file with 3 integration tests
    - `docs/ENVIRONMENT_SPEC.md`: Added 150+ line section explaining architecture
  
  - **Behavioral Changes**: None for existing experiments (visible_amr_levels == actual_amr_levels in current implementation). Enables future experiments with observation noise/deception where visible and actual AMR diverge.
  
  - **User Impact**: Reward calculations now guarantee ground-truth anchoring. Agents training on noisy observations will benefit from recurrent memory mechanisms. Documentation clarifies this design choice and its implications for policy architecture selection.

#### Reward Calculator: Separated Visible and Actual AMR Levels (January 30, 2026)

- **Fixed reward calculation to correctly distinguish visible vs. actual AMR levels**
  - **Issue**: RewardCalculator methods were receiving only a single `amr_levels` parameter, which conflated two distinct uses: (1) computing antibiotic sensitivity for individual patient rewards (requires **actual** AMR levels to determine true resistance), and (2) computing community reward penalty (requires **visible** AMR levels as the agent observes). This separation is critical when actual resistance differs from observed (e.g., observation noise).
  - **Solution**: 
    - Modified `calculate_reward()` signature to accept both `visible_amr_levels` and `actual_amr_levels` as separate parameters
    - Updated `calculate_individual_reward()` to clearly document that individual reward epsilon penalty uses `delta_visible_amr`
    - Updated `calculate_community_reward()` to accept `visible_amr_levels` only (community penalty is based on agent's observations)
    - Updated `calculate_expected_reward()` signature to match `calculate_reward()` with both parameter types
    - Updated environment call site in `abx_amr_env.py` to pass both parameters
  - **Scope**: All reward computation methods now correctly use the right AMR level (actual for sensitivity, visible for community)
  - **Tests Updated**: Fixed 13+ test calls across 4 test files; all 28 reward tests, 17 RNG ownership tests pass
  - **Behavioral Changes**: None for existing experiments (visible_amr_levels == actual_amr_levels in current implementation); enables future observation noise/deception scenarios

- **Bonus: Implemented AMR_LeakyBalloon.copy() method for counterfactual AMR calculations**
  - **Issue**: Environment step function was calling `model.copy()` to compute marginal AMR contributions (delta) without modifying the original balloon model, but the method didn't exist
  - **Solution**: 
    - Implemented `copy()` method on `AMR_LeakyBalloon` that creates independent copies with same configuration and current pressure state
    - Method enables counterfactual analysis: copy the balloon, reset to visible AMR level, compute hypothetical deltas without affecting original
    - Fixed environment parameter name from `visible_amr_level=` to `initial_amr_level=` in reset call
  - **Tests Added**: Two comprehensive tests verifying copy independence and counterfactual usage pattern
  - **Test Results**: All 10 leaky balloon tests pass (including 2 new copy tests), environment integration test passes

#### Ensemble Metrics: Overall Outcomes Summary Dictionary Generation (January 26, 2026)

- **Fixed plot_metrics_ensemble_agents failing to generate outcome summary JSON files**
  - **Issue**: `plot_metrics_ensemble_agents` was manually constructing raw outcome dictionaries with ad-hoc key naming and aggregation logic in Phase 5, leading to incorrect key names (e.g., `overall_count_clinical_benefits_count` instead of `overall_count_clinical_benefits`) and missing the centralized `create_overall_outcomes_summary_dict` function, resulting in no JSON files being written.
  - **Solution**: Refactored Phase 5 to call `create_overall_outcomes_summary_dict` for each trajectory, flatten per-antibiotic nested dicts, collect raw values across all runs, and compute percentile statistics (p10, p25, p50, p75, p90) for all outcome metrics.
  - **Behavioral Changes**:
    - `plot_metrics_ensemble_agents` now generates two JSON files in ensemble output directory:
      - `overall_outcomes_summary_raw_vals.json`: Raw end-of-episode values for all runs (one value per trajectory)
      - `overall_outcomes_summary_summary_stats.json`: Percentile statistics (p10, p25, p50, p75, p90) for each outcome metric
    - Both files include `overall_total_reward` (previously missing)
    - Per-antibiotic outcomes are flattened with `_<antibiotic_name>` suffix for percentile computation
  - **Files Modified**:
    - `src/abx_amr_simulator/utils/metrics.py`: 
      - Replaced manual aggregation in Phase 5 with loop calling `create_overall_outcomes_summary_dict` per trajectory
      - Added `_flatten_summary()` helper to convert nested per-antibiotic dicts to flat structure
      - Removed incorrect key construction logic (`'overall_' + key + '_count'` pattern)
      - Removed stray `pdb.set_trace()` breakpoint
    - `tests/unit/utils/test_metrics_ensemble.py`: New test file with stub environment and trajectory to verify JSON generation and `overall_total_reward` presence
  - **Testing**: Test confirms both JSON files are written with correct structure and `overall_total_reward` key present
  - **User Impact**: Ensemble analysis now reliably produces outcome summary JSON files for downstream aggregation and comparison across experiments

#### Diagnostic Analysis: Convergence Plot Generation (January 27, 2026)

- **Fixed convergence_curve.png generation in diagnostic_analysis.py failing to extract eval metrics from TensorBoard logs**
  - **Issue**: The minimal custom TFRecord protobuf parser was too brittle and failed to decode scalar events from SB3-generated TensorBoard files, causing all convergence extraction attempts to return zero points even when `eval/mean_reward` and other metrics were present in the logs. This resulted in no `convergence_curve.png` being generated during diagnostic analysis.
  - **Root Cause**: Homegrown `read_tfrecord_simple()` and `parse_event_protobuf()` functions couldn't handle the full TensorBoard event format, particularly the nested summary message structure and scalar value encoding used by Stable-Baselines3.
  - **Solution**: Replaced custom parser with TensorBoard's official `event_accumulator.EventAccumulator` (already available as a dependency via SB3) for robust scalar extraction. Added three-tier fallback strategy:
    1. **Primary**: Use `event_accumulator` to read TensorBoard event files (handles `logs/PPO_1`, `logs/RecurrentPPO_1`, etc. automatically via recursive search)
    2. **Secondary**: Fall back to legacy custom parser if `event_accumulator` unavailable
    3. **Tertiary**: Build convergence data from `eval_logs/eval_*_step_*.npz` files by computing mean of `episode_rewards` if TensorBoard extraction yields no data
  - **Behavioral Changes**:
    - `extract_eval_metrics()` now tries multiple metric tags with fallbacks: `eval/mean_reward` (primary), then `rollout/ep_rew_mean`, `train/episode_reward`, `train/rollout/ep_rew_mean`
    - Added debug logging: reports number of event files found, lists first 3 file paths, shows per-tag point counts, and warns when convergence data is empty for a seed
    - New function `extract_eval_metrics_from_eval_logs()` provides guaranteed fallback when TensorBoard logs are missing or unparseable
  - **Algorithm Support**: Handles PPO, RecurrentPPO, DQN, A2C (any SB3 algorithm) automatically—no hardcoded subfolder names; recursive search under `run_dir/logs/` discovers all event files regardless of `<algo>_<id>` naming
  - **Files Modified**:
    - `src/abx_amr_simulator/analysis/diagnostic_analysis.py`: 
      - Added TensorBoard `event_accumulator` import with optional availability check
      - Rewrote `extract_eval_metrics()` to use three-tier fallback strategy
      - Added `extract_eval_metrics_from_eval_logs()` for npz-based convergence extraction
      - Enhanced debug logging throughout convergence extraction pipeline
      - Added warning when convergence data is empty for a seed (plot skipped)
    - `tests/unit/utils/test_metrics_ensemble.py`: Added test to verify `plot_metrics_ensemble_agents` writes outcome summary JSONs with `overall_total_reward` present
  - **Testing**: Verified on multiple experiments (PPO with `logs/PPO_1` structure); convergence plots now generate successfully with 20+ data points per seed
  - **User Impact**: `diagnostic_analysis.py` now reliably produces `convergence_curve.png` and `convergence_curve_aggregated.png` for both single-run and multi-seed analyses; no manual intervention needed

### Added

#### Expected Reward APIs for MDP Solver (January 23, 2026)

- **Added deterministic expected reward calculation methods to `RewardCalculator`**
  - **Purpose**: Enable value iteration, analytical equilibrium analysis, and MDP solving on homogeneous patient populations without Monte Carlo sampling
  - **New Methods**:
    - `calculate_expected_individual_reward(patient, antibiotic_name, amr_level, delta_amr)`: Computes deterministic expected reward for a single patient-action pair
    - `calculate_expected_reward(patients, actions, antibiotic_names, amr_levels, delta_amr_per_antibiotic)`: Batch expected reward with lambda-weighting matching stochastic path
  - **Key Features**:
    - No RNG consumption - purely deterministic mathematical computation
    - Exact MC match: For homogeneous populations, expected reward equals Monte Carlo mean (validated |Δ| < 1e-2 at 50k+ samples)
    - Normalization consistency: Epsilon scaling and normalization flags work identically in stochastic and expected paths
    - Validation: Antibiotic names validated, probability clamping matches stochastic `calculate_individual_reward()` exactly
  - **Mathematical Formulas** (prescribe antibiotic):
    ```
    E[reward] = pI×pS×pB×RB×vB + pI×(1−pS)×pF×RF×vF + pAE×AE − ε×δ
    
    Where:
    - pI = patient.prob_infected
    - pS = 1 - amr_level (sensitivity probability)
    - pB = clamp(base_benefit_prob × benefit_probability_multiplier, 0, 1)
    - pF = clamp(base_failure_prob × failure_probability_multiplier, 0, 1)
    - RB, RF = clinical benefit/failure rewards (normalized or unnormalized)
    - vB, vF = patient value multipliers
    - pAE, AE = adverse effect probability and penalty (per-antibiotic)
    - ε = epsilon (AMR penalty weight)
    - δ = delta_amr (marginal AMR contribution)
    ```
  - **Mathematical Formulas** (no treatment):
    ```
    E[reward] = pI×r×RB×vB + pI×(1−r)×pF×RF×vF
    
    Where:
    - r = patient.recovery_without_treatment_prob
    - No adverse effects term (no prescription)
    - No epsilon penalty (no AMR contribution)
    ```
  - **Files Modified**:
    - `src/abx_amr_simulator/core/reward_calculator.py`: Added two expected reward methods with 80+ line comprehensive docstrings
    - `docs/ENVIRONMENT_SPEC.md`: Added "Expected Reward Calculation" section with formulas, use cases, implementation notes
    - `tests/unit/rewards/test_expected_rewards.py`: 4 validation tests (homogeneous normalized/unnormalized, no_treatment/prescribe)
  - **Testing**: All 4 expected reward tests pass; MC vs expected validation shows |Δ| < 1e-2 with 50k+ samples
  - **Use Cases**: Value iteration for MDP solving, analytical equilibrium analysis, policy evaluation without rollout variance, validation of stochastic reward implementation

#### RNG Refactor: Explicit Threading with Ownership Semantics (January 23, 2026)

- **Implemented explicit RNG ownership lifecycle to ensure reproducibility and prevent RNG synchronization bugs**
  - **Purpose**: Make RNG consumption order visible at call sites, prevent stale RNG bugs, enforce explicit RNG threading from environment to all stochastic components
  - **Ownership Lifecycle Pattern**:
    - **Standalone mode** (initial state): Components start with `_standalone = True` and own their internal RNG (`self.rng = np.random.default_rng(seed)`)
    - **Environment-owned mode** (after transfer): Environment calls `_set_environment_owned()` to set `_standalone = False`, requiring explicit `rng` parameter on all stochastic methods
    - **One-way transition**: Standalone → owned (irreversible)
    - **Fail loudly**: Owned components without explicit `rng` raise descriptive `ValueError` distinguishing owned vs standalone contexts
  - **API Changes**:
    - `PatientGenerator.sample(n_patients, rng=None)`: RNG parameter now optional (mandatory when owned, fallback to `self.rng` when standalone)
    - `RewardCalculator.calculate_reward(..., rng=None)`: RNG parameter now optional (mandatory when owned, fallback to `self.rng` when standalone)
    - `RewardCalculator.calculate_individual_reward(..., rng=None)`: RNG parameter now optional (mandatory when owned, fallback to `self.rng` when standalone)
    - `PatientGenerator._set_environment_owned()`: New method to transfer ownership (sets `_standalone = False`)
    - `RewardCalculator._set_environment_owned()`: New method to transfer ownership (sets `_standalone = False`)
  - **Environment Orchestration**:
    - `ABXAMREnv` owns single `numpy.random.Generator` instance (`self.np_random`)
    - Calls `_set_environment_owned()` on PatientGenerator and RewardCalculator during `__init__()`
    - Threads shared RNG explicitly to all stochastic operations (`sample()`, `calculate_reward()`)
  - **Cascading Ownership**:
    - `PatientGeneratorMixer._set_environment_owned()` propagates ownership recursively to all subordinate generators
    - Ensures entire generator hierarchy uses shared RNG instance
  - **RNG Consumption Optimization**:
    - **No-treatment RNG skipping**: When action is `'no_treatment'`, sensitivity and adverse effect draws are completely skipped (no wasted RNG consumption)
    - **Benefit**: Maintains seeded parity between stochastic and expected reward paths - given same seed and homogeneous population, expected matches MC mean exactly
    - **Implementation**: In `RewardCalculator.calculate_reward()`, iterate patients by action and gate sensitivity/AE draws only for prescribed antibiotics
  - **Files Modified**:
    - `src/abx_amr_simulator/core/patient_generator.py`: Added `_standalone` flag, `_set_environment_owned()` method, RNG validation in `sample()`, cascading ownership for PatientGeneratorMixer
    - `src/abx_amr_simulator/core/reward_calculator.py`: Added `_standalone` flag, `_set_environment_owned()` method, RNG validation in `calculate_reward()` and `calculate_individual_reward()`, gated sensitivity draws for no_treatment
    - `src/abx_amr_simulator/core/abx_amr_env.py`: Calls `_set_environment_owned()` on PG/RC, passes shared `self.np_random` to all stochastic operations
    - `tests/unit/core/test_rng_ownership.py`: New test file with 17 comprehensive tests covering ownership lifecycle, validation, cascading, integration, and no-treatment RNG skipping
    - `docs/ENVIRONMENT_SPEC.md`: Added "RNG Ownership and Threading" section with lifecycle documentation, validation patterns, best practices
  - **Testing**: 122 total tests pass (101 core unit + 17 ownership + 4 expected reward)
  - **Best Practices**:
    - Always use environment-owned mode in training (pass components to environment and let it manage RNG lifecycle)
    - Use standalone mode only for unit tests and debugging
    - Seed at environment level (`ABXAMREnv(..., seed=42)`), not at component level after ownership transfer
    - Test with explicit seeds and assert deterministic trajectories

### Fixed

- **Epsilon normalization logic**: Corrected epsilon scaling behavior in RewardCalculator. Epsilon now correctly represents a relative weight (percentage of max reward/penalty) rather than an absolute value. When `normalize_clinical_reward_penalties=True`, epsilon is used directly (already on ~1.0 scale). When `normalize_clinical_reward_penalties=False`, epsilon is scaled up proportionally by `max_abs_value_of_any_reward_or_penalty` to maintain the intended percentage relationship. This makes epsilon=0.05 consistently represent a "5% mini-penalty" regardless of the absolute reward scale.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### PatientGenerator Config Redesign: Nested Per-Attribute Structure (January 22, 2026)

- **BREAKING CHANGE: PatientGenerator config format updated from flat to nested per-attribute structure**
  - **Motivation**: Flat format was confusing (19 keys per patient population). New nested structure makes it clear that each of 6 attributes has its own distribution, bias, noise, and clipping configuration.
  - **Old format example**:
    ```python
    {
        'prob_infected_dist': {'type': 'constant', 'value': 0.5},
        'prob_infected_observation_multip_bias': 1.0,
        'prob_infected_observation_noise_scale': 0.05,
        'prob_infected_observation_range': [0.0, 1.0],
        # ... 15 more keys
    }
    ```
  - **New format example**:
    ```python
    {
        'prob_infected': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.2,      # Reference magnitude (user-specified)
            'obs_noise_std_dev_fraction': 0.5, # Fraction of reference to use (unitless)
            'clipping_bounds': [0.0, 1.0],     # [lower, upper] or [lower, None]
        },
        # ... 5 more attributes with same structure
        'visible_patient_attributes': ['prob_infected', ...]
    }
    ```
  - **Key improvements**:
    - Per-attribute configuration explicit and non-redundant
    - Noise model clarified: `effective_std = obs_noise_std_dev_fraction × obs_noise_one_std_dev`
    - User-specified `clipping_bounds` (no longer hardcoded)
    - `KNOWN_ATTRIBUTE_TYPES` and `ATTRIBUTE_TYPE_VALIDATION` static maps enable future type-aware validation
    - Default `clipping_bounds`: probabilities [0, 1], multipliers [0, None]
  - **Parameter name changes**:
    - `obs_multiplier_bias` → `obs_bias_multiplier` (shorter, consistent naming)
    - `observation_noise_scale` → `obs_noise_std_dev_fraction` (clarifies it's a fraction)
    - `observation_range` → `obs_noise_one_std_dev` (clarifies it's the reference magnitude, not a clipping bound)
  - **Files modified**:
    - `src/abx_amr_simulator/core/patient_generator.py`: Rewrote `__init__()`, `default_config()`, added `_validate_attribute_config()`, refactored `sample()` to use generic loop over attributes
    - `src/abx_amr_simulator/configs/defaults/patient_generator/default.yaml`: Updated to nested format
    - `workspace/experiments/configs/patient_generator/default.yaml`: Updated to nested format
    - All test configs updated in: test_patient_generator.py (44 tests), test_patient_generator_mixer.py (27 tests), test_patient_generator_integration.py (11 tests), test_baseline_infection.py (5 tests)
    - Test helper functions in: test_reference_helpers.py, test_abx_env.py, test_patient_generator_integration.py
  - **Testing**: 87 tests pass (44 unit + 27 integration mixer + 11 integration PG + 5 baseline)
  - **Migration guide**: Replace flat keys with nested per-attribute structure; use `PatientGenerator.default_config()` as a template

## [Unreleased]

### Added

#### Single-Antibiotic Grid Search Tooling (January 20, 2026)

- **Added `workspace/scripts/find_parameters_for_equilibrium.py`**: grid-search driver for single-abx setups that couples Bellman threshold estimation with AMR_LeakyBalloon simulation to measure equilibrium AMR and convergence time. Supports dot-notation parameter sweeps (environment leak/flatness and RewardCalculator knobs).
- **Added `scripts/run_single_abx_grid_sweep.sh`**: convenience wrapper to launch a sweep with lambda fixed at 0, tuning clinical benefit/failure rewards and adverse-effect penalty/probability alongside leak/flatness.
- **Sample sweep result (Antibiotic_A, defaults)**: best combo found (leak=0.4, flatness=50, benefit=8, failure=-12, AE pen=-1.0, AE prob=0.05, lambda=0.0) → actual_equilibrium≈0.5005, convergence_steps≈101, eq_error≈0.2005 (overshoots target AMR=0.3 but converges faster than target 150 steps).

### Fixed

#### Epsilon Normalization Bug Fix (January 20, 2026)

- **Fixed epsilon scaling inconsistency when `normalize_clinical_reward_penalties=True`**
  - **Issue**: When normalization was active, clinical rewards (benefit/failure/adverse) were scaled down by dividing by `max_abs_value_of_any_reward_or_penalty` (~10×), but epsilon AMR shaping remained at full magnitude, causing disproportionately strong AMR penalty
  - **Impact**: Agents trained with `normalize=True` and `epsilon>0` would over-prioritize AMR avoidance relative to clinical benefit
  - **Fix**: Epsilon now scales consistently with clinical rewards when normalization is enabled:
    ```python
    if normalize_clinical_reward_penalties:
        reward -= (epsilon / max_abs_value_of_any_reward_or_penalty) * delta_amr
    else:
        reward -= epsilon * delta_amr
    ```
  - **Files Modified**: `src/abx_amr_simulator/core/reward_calculator.py` (line 350-356, added conditional scaling; line 207, stored normalization factor)
  - **Testing**: All 5 RewardCalculator unit tests pass; 12 integration tests pass
  - **Backward Compatibility**: 
    - Configs with `normalize=False` (default for most experiments): **No change** - epsilon behaves identically
    - Configs with `normalize=True, epsilon=0`: **No change** - epsilon still zero
    - Configs with `normalize=True, epsilon>0`: **Behavior change** - epsilon now properly scaled (was buggy before)

#### Test Factory Utils Tuple Unpacking Fix (January 20, 2026)

- **Fixed `test_factory_utils.py` tests to unpack `create_run_directory()` return tuple**
  - **Issue**: `create_run_directory()` returns `(run_dir_path, timestamp)` tuple, but tests expected only a string path
  - **Impact**: 5 tests in `TestCreateRunDirectory` were failing with `TypeError: expected str, bytes or os.PathLike object, not tuple`
  - **Fix**: Updated all test methods to properly unpack the tuple:
    - `test_creates_timestamped_directory()`
    - `test_directory_name_format()`
    - `test_directory_structure_created()`
    - `test_returns_string_path()` - also updated to verify both returned values
    - `test_unique_directories_for_multiple_calls()` - also verifies timestamps differ
  - **Files Modified**: `tests/unit/utils/test_factory_utils.py` (5 test methods updated)
  - **Testing**: All 5 TestCreateRunDirectory tests now pass; full unit test suite passes (169/169)
  - **Impact**: Enables full test coverage validation

### Changed (Breaking)

#### Mutually Exclusive Prescribing Refactoring (January 20, 2026)

- **BREAKING: Action space now enforces mutually exclusive prescribing**
  - **Old Model**: `action_mode='multidiscrete'` or `'flat'` allowing independent per-antibiotic choices (exponential action space: 2^num_abx)
  - **New Model**: `MultiDiscrete([num_abx + 1] * num_patients)` where agent picks one antibiotic OR no treatment per patient (linear action space: num_abx + 1)
  - **Motivation**: 
    - Reflects clinical reality: each patient receives a single treatment decision, not combinations
    - Simplifies strategy space and enables analytical modeling of equilibria
    - Removed unnecessary complexity of simultaneous prescribing
  - **Implementation**:
    - Removed `action_mode` parameter from `ABXAMREnv.__init__()` entirely
    - Removed `_action_base` property and action encoding/decoding methods
    - Action directly maps to antibiotic index (0=no treatment, 1=Abx_A, 2=Abx_B, etc.)
    - RewardCalculator already supported this pattern, no changes needed
  - **Config Changes**:
    - REMOVED: `action_mode` key from all environment YAML configs
    - Example old config:
      ```yaml
      action_mode: multidiscrete  # REMOVED - no longer needed
      ```
  - **Migration Guide**:
    - Remove `action_mode=...` from all `ABXAMREnv()` instantiations
    - Remove `action_mode: ...` from all environment config files
    - Update action generation: instead of `np.ones((num_patients, num_abx))` → `np.zeros(num_patients)`
  - **Files Modified**:
    - Core: `src/abx_amr_simulator/core/abx_amr_env.py` (removed action_mode, simplified step())
    - Configs: Removed `action_mode` from 12 environment YAML files
    - Tests: Updated 169 unit tests + 64 integration tests; deleted `test_flat_action_mode_matches_multidiscrete`
  - **Impact**: All 233 tests pass; environment now strictly enforces single-antibiotic-per-patient decisions

#### PatientGenerator Range-Aware Noise Refactoring (January 20, 2026)

- **BREAKING: Refactored PatientGenerator observation noise to be scale-invariant (range-aware)**
  - **Old Model**: `*_observation_additive_noise` (absolute Gaussian noise, unit-dependent)
  - **New Model**: `*_observation_noise_scale` + `*_observation_range` (unitless scaling)
  - **Implementation**: Effective noise std = `noise_scale × (upper − lower)`, ensuring consistent noise semantics across different attribute scales
  - **Motivation**: Users previously had to manually tune noise_std per attribute based on its range. Now, `noise_scale=0.05` consistently means "5% of range noise" everywhere.
  - **Config Changes**:
    - REQUIRED keys: Added `visible_patient_attributes` (was optional)
    - OPTIONAL keys: Replaced 6 × `*_observation_additive_noise` with 6 × `*_observation_noise_scale` + 6 × `*_observation_range`
    - Example: `prob_infected_observation_range: [0.0, 1.0]`, `prob_infected_observation_noise_scale: 0.05`
  - **Migration Guide**:
    - If using `noise_std > 0` in old configs: Divide by attribute range to get `noise_scale`. E.g., `noise_std=0.05` on [0,1] → `noise_scale=0.05`
    - If using `noise_std=0` (no noise): Just update key names; set `noise_scale: 0.0`
    - Add `visible_patient_attributes: ['prob_infected', ...]` to all configs
  - **Files Modified**:
    - Core: `src/abx_amr_simulator/core/patient_generator.py` (new signature, range parsing, unitless noise)
    - Configs: All 6 YAML patient_generator configs across src/, tests/, workspace/
  - **Impact**: All 234 tests pass; no functionality broken, only config schema changed
  - **Documentation**: See `docs/CONFIG_SYSTEM.md` (PatientGenerator Configuration section) and `docs/ENVIRONMENT_SPEC.md` (Observation Noise section) for detailed range-aware noise explanation and examples

### Fixed

#### DetailedEvalCallback Nested Directory Bug (January 18, 2026)

- **Fixed nested `eval_logs/eval_logs/` directory structure**:
  - **Root cause**: `create_callbacks()` in `factories.py` was passing `log_path=os.path.join(run_dir, 'eval_logs')` to DetailedEvalCallback, which then internally creates another `/eval_logs` subdirectory.
  - **Impact**: Trajectory files were being saved to `run_dir/eval_logs/eval_logs/` instead of `run_dir/eval_logs/`
  - **Fix**: Changed to pass `log_path=run_dir` to DetailedEvalCallback, allowing it to create the single `/eval_logs` directory
  - **File Modified**: `src/abx_amr_simulator/utils/factories.py` line 424
  - **Backward compatibility**: `find_npz_files()` in diagnostic_analysis.py already defaults to searching `eval_logs/eval_logs`, so old runs continue to work

### Added

#### Convergence Curve Plotting in Diagnostic Analysis (January 18, 2026)

- **Extended `diagnostic_analysis.py` with TensorBoard log parsing and convergence plotting**:
  - Added lightweight TFRecord event file parser (no TensorFlow dependency required)
  - Extracts `eval/mean_reward` metrics from TensorBoard logs during post-training analysis
  - Generates individual convergence curves (`convergence_curve.png`) for each seed showing reward vs timestep
  - For multi-seed experiments with `--aggregate-by-seed`, generates aggregated plot showing:
    - Individual seed trajectories (low alpha, thin lines)
    - Mean trajectory across seeds (bold black line)
    - ±1 standard deviation shaded region
  - Plots saved to `analysis_output/<experiment>/diagnostics/seed_N/convergence_curve.png`
  - Aggregated plot saved to `analysis_output/<experiment>/diagnostics/convergence_curve_aggregated.png`
  
- **Key Functions Added**:
  - `read_tfrecord_simple()`: Parses TensorBoard event files without external dependencies
  - `parse_event_protobuf()`, `parse_protobuf_field()`, `parse_varint()`: Minimal protobuf parsers
  - `parse_summary_message()`, `parse_value_message()`: Extract scalar metric values
  - `find_event_files()`: Locates TensorBoard logs in run directories
  - `extract_eval_metrics()`: Extracts evaluation metrics from event files
  - `plot_convergence_curve()`: Single-seed convergence plotting with matplotlib
  - `plot_multi_seed_convergence()`: Multi-seed aggregated plotting with mean/std bands

- **Motivation**: 
  - Enables batch post-training convergence analysis for parallel cloud runs
  - Eliminates need for real-time TensorBoard monitoring across multiple experiments
  - Provides visual diagnostics to validate training duration (e.g., checking if 400 episodes is sufficient)

- **Integration Tests** (`tests/integration/test_diagnostic_convergence_plotting.py`):
  - Tests use real experiment runs from `workspace/results/` to validate actual TensorBoard log parsing
  - Covers event file discovery, metric extraction, and analysis pipeline
  - Gracefully handles cases where convergence metrics may be missing from event files

- **Files Modified**:
  - `src/abx_amr_simulator/analysis/diagnostic_analysis.py`: Added ~270 lines of TFRecord parsing and plotting code
  - `tests/integration/test_diagnostic_convergence_plotting.py`: New integration test suite
  - Graceful degradation: warns if matplotlib unavailable, continues with other analyses

#### LSTM Belief Probing Infrastructure: Validated Sanity Check (January 18, 2026)

- **Completed implementation of LSTM hidden state logging + offline probe pipeline**:
  - `LSTMStateLogger` callback (in `src/abx_amr_simulator/callbacks/lstm_state_logger.py`) captures LSTM hidden states during evaluation
  - `probe_hidden_belief.py` script fits linear regression probe: hidden_state → true AMR for each antibiotic
  - Integration test (`tests/integration/test_lstm_belief_probing.py`) validates end-to-end pipeline

- **Probe Results (Baseline Setup)**:
  - R² ≈ 0.999 for both antibiotics (near-perfect prediction)
  - MAE ≈ 0.0012 (mean absolute error on AMR level)
  - **Interpretation**: This is a **sanity check** validating that the logger/probe infrastructure works correctly. In the current experimental setup (fully observable AMR, daily updates), the LSTM can simply echo observed AMR values. The near-perfect scores confirm the pipeline is functional, not that the agent is performing genuine latent inference.

- **Key Insight for Future Work**:
  - To test genuine hidden belief inference, future experiments should reduce/delay AMR observability:
    - Partial AMR signals (e.g., only 1 of 2 antibiotics observed per day)
    - Delayed AMR updates (e.g., AMR observed every N days, not daily)
    - Noisy AMR observations (simulating imperfect resistance tests)
    - Missing AMR from observation entirely (full latency)
  - In those regimes, probe accuracy becomes a real measure of whether the LSTM integrates history to infer unobserved resistance.

### Changed

#### RewardCalculator API Refactor (January 17, 2026)

- **Standardized constructor to `config: Dict`**: `RewardCalculator.__init__()` now accepts a single configuration dictionary, aligning with `PatientGenerator` and the CLI/YAML-first workflow.
  - Added `RewardCalculator.default_config()` to provide a discoverable template with sensible defaults.
  - Updated convenience constructors and cloning helpers:
    - `clone_with_lambda()` now exports a config and calls `RewardCalculator(config=...)` preserving RNG state.
    - `IndividualOnlyReward`, `CommunityOnlyReward`, `BalancedReward` now accept `config` and set `lambda_weight` internally; `from_existing()` returns `cls(config=...)`.
  - Updated factories and tests to use the new config dict pattern.
  - Breaking change: legacy kwargs instantiation is removed; all tests and helpers updated.

- **Files updated**:
  - `src/abx_amr_simulator/core/reward_calculator.py` (constructor, `default_config()`, cloning, convenience subclasses)
  - `src/abx_amr_simulator/utils/factories.py` (simplified `create_reward_calculator()`)
  - Tests updated across suite to pass `config` (e.g., `tests/test_reward_calculator.py`, `tests/test_reward_calculator_multipliers.py`, `tests/test_abx_env.py`, `tests/test_patient_generator_integration.py`, `tests/test_callbacks.py`).

#### PatientGeneratorMixer: Heterogeneous Visibility Support (January 17, 2026)

- **Auto-detect visibility heterogeneity**: Mixer now detects whether child generators share visibility or not.
  - Uniform visibility → uses child `visible_patient_attributes` directly.
  - Heterogeneous visibility → computes sorted union of attributes and pads per-patient vectors.
- **Padding behavior**: For attributes not visible in a patient's source generator, inserts sentinel `PADDING_VALUE = -1.0`.
  - Observations remain flattened 1D arrays of shape `(num_patients * union_size,)`.
  - `obs_dim(num_patients)` reflects union size.
- **Patient provenance**: Added `source_generator_index` to `Patient` for correct per-patient padding based on origin generator.
- **Files updated**:
  - `src/abx_amr_simulator/core/patient_generator.py` (mixer union detection, per-patient provenance tagging, visibility-aware `observe()`, `obs_dim()`)
  - `src/abx_amr_simulator/core/types.py` (added `source_generator_index` field to `Patient`).
  - Tests: `tests/test_patient_generator_mixer.py` expanded with comprehensive heterogeneous visibility coverage and environment integration.

- **ABXAMREnv instantiation**: Environment continues to use explicit kwargs (not a `config` dict) for clarity and validation; tests updated accordingly.

### Status

- All tests passing: 268 passed, 10 warnings.
- Documentation to follow up: tutorials and references will be updated to reflect the new `RewardCalculator(config)` API and mixer behavior.

#### Leaky Balloon Model: User-Facing AMR Level Parameter (January 17, 2026)

- **Replaced `initial_latent_pressure` with `initial_amr_level`** in `AMR_LeakyBalloon` class:
  - **Motivation**: `initial_latent_pressure` was an internal implementation detail (latent state) that was non-intuitive for users. Users had to reason backwards from "what pressure produces what AMR level?"
  - **Solution**: Users now directly specify `initial_amr_level` (the observable, visible resistance fraction in [0,1]), which is converted internally to the corresponding latent pressure via inverse sigmoid
  - **Backward Compatibility**: This is a breaking change; all config files and tests have been updated

- **Implementation details**:
  - Added `_inverse_sigmoid(volume)` method to compute latent pressure from desired AMR level
  - Validates that `initial_amr_level` is in `[permanent_residual_volume, 1.0]`
  - `reset()` method now takes `initial_amr_level` instead of `initial_latent_pressure`
  - Inverse sigmoid uses standard formula: `p = -flatness_parameter * log((1 - sigmoid_val) / sigmoid_val)`
  - All edge cases handled: clamping sigmoid values to avoid log domain errors, proper handling with non-zero residual volumes

- **Updated files**:
  - `src/abx_amr_simulator/core/leaky_balloon.py`:
    - `__init__()`: Changed parameter, added inverse sigmoid conversion
    - Added `_inverse_sigmoid()` helper method
    - `reset()`: Now takes `initial_amr_level` and converts via inverse sigmoid
    - Updated docstrings with clearer parameter descriptions
  - `src/abx_amr_simulator/core/abx_amr_env.py`:
    - Updated validation and default parameters
    - Docstring updated to reflect new parameter
  - Configuration files:
    - `src/abx_amr_simulator/configs/defaults/environment/default.yaml`
    - `docs/tutorials/02_custom_experiments.md` (6 locations)
    - GUI experiment runner: `src/abx_amr_simulator/gui/experiment_runner.py` (4 locations, added max_value=1.0 constraint)
  - Test files:
    - `tests/test_leaky_balloon.py`: 8 tests updated with new parameter
    - `tests/test_abx_env.py`: All environment tests updated (28 tests, all passing)
    - Test shells: `tests/debug_subconfig_override.sh`, `tests/validate_tutorial_2.sh`
  - Documentation:
    - `.github/copilot-instructions.md`: Updated architecture description

### Added

#### Resume-Friendly Shell Scripts: `--skip-if-exists` with Completion Registry (January 15, 2026)

- **New `--skip-if-exists` CLI flag** for `train.py`:
  - Enables graceful resumption of interrupted shell scripts without overwriting completed experiments
  - Checks completion registry before training; skips if already completed, retries if interrupted
  - Validates and automatically cleans stale registry entries (folders deleted externally)
  - Excludes current run from validation to prevent premature removal before training starts

- **Completion Registry System** (``.training_completed.txt``):
  - Location: `<results_dir>/<output_dir>/.training_completed.txt`
  - Tracks successfully completed training run names (one per line)
  - Distinguishes completed (skip) from interrupted (retry) experiments
  - Automatically updated by `train.py` after successful training completion
  - Separate registries for training completion (new) vs. analysis completion (existing)

- **Registry Validation & Cleanup** (`validate_and_clean_registry()` function):
  - Automatic removal of stale entries (experiments manually deleted from filesystem)
  - Excludes current run during validation to prevent removal before folder creation
  - Returns list of cleaned entries for user transparency
  - Supports safe concurrent shell script execution (each run excludes itself from cleanup)

- **Bug Fix: Registry Pattern Matching**:
  - Fixed seed confusion where different seeds (e.g., seed1, seed2) treated as same experiment
  - Changed pattern from `{entry}_seed*_*` to `{entry}_*` (only matches timestamp suffix)
  - Each seed now correctly tracked as independent entry in registry
  - All 5 unit tests for registry validation passing

- **Updated files**:
  - `src/abx_amr_simulator/training/train.py`:
    - Added `--skip-if-exists` flag with registry check and validation
    - Comprehensive docstring (60+ lines) explaining registry system, workflow, example shell scripts
    - Registry validation only runs when `--skip-if-exists` is set (not on every training)
    - Completion recording after successful training completion
  - `src/abx_amr_simulator/utils/registry.py`:
    - New `validate_and_clean_registry()` function with `exclude_prefix` parameter
    - Fixed pattern matching logic to distinguish different seeds
  - Shell scripts (5 total in `workspace/experiments/shell_scripts/`):
    - All updated with `--skip-if-exists` flag
    - Can now be safely interrupted and re-run without data loss

- **Test Coverage**:
  - `tests/test_registry_validation.py`: 5 unit tests for registry validation and cleanup
    - Registry with stale entries (removed correctly)
    - Empty registry (handled gracefully)
    - Non-existent registry (returns empty list)
    - Non-existent results directory (all entries marked stale)
    - Exclude prefix functionality (preserved during validation)
  - `tests/test_training_completion_registry.py`: Integration test framework
    - Runs multiple quick training experiments and verifies registry updates
    - Validates end-to-end workflow from training completion to registry recording
  - All pytest tests passing (exit code 0)

- **User-Facing Benefits**:
  - Can interrupt shell scripts mid-run without losing progress
  - Re-running script resumes from last completed experiment
  - Different seeds automatically tracked as separate experiments
  - No risk of overwriting successful runs
  - Stale registry entries automatically cleaned
  - Clear, transparent documentation of registry system in train.py docstring

### Changed

#### Analysis Script Consolidation & Phase Refactoring (January 15, 2026)

- **Consolidated analysis into two focused phases**:
  - **Phase 2 (Diagnostic)**: `diagnostic_analysis.py` - Analyzes training behavior via evaluation trajectory data
    - Computes observation error metrics (MAE, RMSE, bias) for each patient attribute
    - Computes reward-observation error correlations (Pearson/Spearman) to validate impact of observation noise
    - Validates that generated patients reflect configured observation noise/bias levels
  - **Phase 3 (Evaluative)**: `evaluative_plots.py` - Evaluates learned policies via best trained models
    - Generates ensemble plots across all seeds: episode rewards, lengths, AMR dynamics, clinical outcomes
    - Computes action-attribute associations showing prescribing behavior conditioned on patient attributes
    - Outputs mean ± 10-90% percentile bands for multi-seed comparison
    - Uses `plot_metrics_ensemble_agents()` utility for comprehensive ensemble analysis

- **Deleted obsolete analysis scripts**:
  - ❌ `analyze_patient_data.py` - Functionality consolidated into `diagnostic_analysis.py`
  - ❌ `plot_ensemble_results.py` - Functionality merged into `evaluative_plots.py` with action-attribute associations added

- **Updated `evaluative_plots.py` architecture**:
  - Now imports and uses `plot_metrics_ensemble_agents()` from utils for complete ensemble visualization
  - Refactored `analyze_experiment()` to orchestrate both ensemble plotting and individual evaluations
  - Removed redundant plotting functions (`plot_with_bands`, `plot_ensemble_rewards`, `plot_ensemble_episode_lengths`)
  - Updated all function calls to use keyword arguments per copilot-instructions.md guidelines
  - Default `analysis_dir` changed to `"analysis_output"` for consistency

- **Updated documentation**: [docs/evaluative_plots.md](docs/evaluative_plots.md)
  - Added comprehensive "Overview" section distinguishing ensemble plots from action-attribute associations
  - Documented all ensemble plot types: Episode Rewards, Episode Lengths, AMR Dynamics, Clinical Outcomes
  - Clarified interpretation guidance for each plot type
  - Retained and expanded action-attribute associations documentation with use cases

#### Comprehensive Test Coverage Audit (January 15, 2026)

- **Added 33 new tests** covering previously untested utility functions
  - [tests/test_config_utils.py](tests/test_config_utils.py) (21 tests): Load config, apply subconfig overrides, apply parameter overrides, setup default folders
  - [tests/test_factory_utils.py](tests/test_factory_utils.py) (12 tests): Create run directories, save training config, save training summary
  - **Focus on critical functions**: `apply_subconfig_overrides()` was just refactored with new `configs_dir` parameter - now fully tested
- **Fixed missing export**: `setup_config_folders_with_defaults()` was not exported from `abx_amr_simulator.utils` - now available
- **All tests passing**: 226/226 tests (193 existing + 33 new) with no regressions
- **Test coverage details** documented in [docs/TEST_COVERAGE_AUDIT.md](docs/TEST_COVERAGE_AUDIT.md)

#### Final Package Reorganization & Workspace Separation (January 14, 2026)

- **Removed backward compatibility shims**: Deleted `abx_amr_env/` folder entirely. All code has been migrated to `src/abx_amr_simulator/core/`.
- **Created workspace/ separation**: Moved `experiments/`, `results/`, `analysis_output/`, `manuscripts/`, `scripts/`, and `experiment_plans/` into dedicated `workspace/` folder. This creates clear division between:
  - **Package code** (`src/abx_amr_simulator/`): Reusable library components
  - **User experiments** (`workspace/`): Project-specific training runs, configs, analysis, and results
- **Renamed analysis scripts** for clarity:
  - `phase_2_diagnostic_analysis.py` → `diagnostic_analysis.py`
  - `phase_3_evaluative_plots.py` → `evaluative_plots.py`
  - Reflects new workflow where "phase 1" is the user's training run, and analysis scripts are naturally subsequent steps
  - Updated all references in tests, README files, docstrings, and documentation (35+ references updated)
- **Updated .gitignore**: Changed patterns from `results/`, `analysis_output/` to `workspace/results/`, `workspace/analysis_output/` to reflect new structure
- **Added Patient tracking fields** for future multi-agent support:
  - `patient_id: Optional[str]`
  - `treated_by_agent: Optional[str]`
  - `treated_in_locale: Optional[str]`
  - `origin_locale: Optional[str]`
  - All fields default to `None` for backward compatibility with single-agent code
- **Added ownership docstrings** to core components documenting what each owns, delegates, and how it will evolve in multi-agent/multi-locale refactor:
  - `ABXAMREnv`: Owns timestep/action-space/orchestration, delegates to PG/RC/AMR_LeakyBalloon
  - `RewardCalculator`: Owns reward formula and lambda weighting, documents per-agent future instantiation
  - `PatientGenerator`: Owns population distributions and observation extraction, documents per-Locale future
  - `AMR_LeakyBalloon`: Owns latent pressure dynamics, documents per-antibiotic-per-Locale future

- **Improvements to training workflow**:
  - **Continue-training safety**: When using `--train-from-prior-results`, parameter overrides are now restricted. Only `--seed` and `training.total_num_training_episodes` can be overridden; other parameters are ignored with a warning. This prevents accidental changes to reward functions, environment settings, or patient distributions mid-training.
  - **Flexible subconfig override handling**: `apply_subconfig_overrides()` now accepts an explicit `configs_dir` parameter instead of deriving it from the umbrella config path. This allows users to override subconfigs from any location, not just relative to the umbrella config's directory.

- **CRITICAL BUG FIX**: Results directory now defaults to current working directory instead of package directory
  - Previously: `python train.py --config config.yaml` from `workspace/my_exp/` would write results to `src/abx_amr_simulator/results/`
  - Now: Results are written to `workspace/my_exp/results/` as expected
  - Fixed by changing train.py to use `cwd` instead of package root for output_dir resolution
  - Config path resolution now tries current directory first, then falls back to package defaults for backward compatibility

- **COMPREHENSIVE OUTPUT DIRECTORY AUDIT**: Verified and fixed all scripts that write output
  - **diagnostic_analysis.py**: Default `--analysis-dir` changed from `"analysis"` → `"analysis_output"` to match documentation
  - **evaluative_plots.py**: Default `--analysis-dir` changed from `"analysis"` → `"analysis_output"` to match documentation
  - **experiment_runner.py** (GUI): Now uses CWD for results lookup instead of package root; config lookup tries CWD first
  - **experiment_viewer.py** (GUI): Now uses CWD for results directory instead of package root
  - **visualize_env_behavior.py**: Config path resolution now tries CWD first; output folder now relative to CWD
  - All analysis and GUI tools now properly respect user working directories
- **All 193 tests passing**: No regressions after package reorganization and file renames

### Added

#### 3-Phase Experiment Pipeline: Diagnostics & Evaluation (January 2026)

- **Diagnostic Analysis Pipeline** (`experiments/diagnostic_analysis.py`)
  - Auto-detects new `exp_*` prefixed experiments in `results/` directory
  - Analyzes all seed runs per experiment with automatic grouping via regex
  - Computes observation error metrics: bias, MAE, RMSE per visible patient attribute
  - Computes reward-error correlations to validate environment dynamics
  - Per-seed aggregation with summary CSV containing mean ± std statistics
  - Registry-based tracking to skip already-analyzed experiments
  - CLI modes: auto (default), manual (`--prefix`), force (`--force`)
  - Output: `analysis_output/<prefix>/diagnostics/{summary_metrics.csv, seed_N/{...}.json}`

- **Evaluative Plots & Action-Attribute Associations** (`experiments/evaluative_plots.py`)
  - Auto-detects new `exp_*` prefixed experiments and runs fresh evaluations
  - Loads best_model.zip from each seed run, auto-detects algorithm (PPO/DQN/A2C)
  - Runs configurable evaluation episodes (default 50), collects trajectory data
  - Generates ensemble performance plots with mean ± percentile bands
  - **NEW: Action-Attribute Associations Analysis**:
    - Computes mutual information between patient attributes and prescribing decisions
    - Bins continuous attributes quantile-based, aggregates prescribing rates
    - Generates action_drivers.csv (per-bin prescribing rates and rewards)
    - Generates action_attribute_associations.csv (MI scores)
  - Registry-based tracking and --force support
  - Output: `analysis_output/<prefix>/evaluation/{ensemble_*.png, action_drivers.csv, action_attribute_associations.csv, metadata.json}`

- **Helper Utilities** (`experiments/utils.py`)
  - `extract_experiment_prefix()`: Parses `exp_*_seed<N>_<timestamp>` folder names
  - `find_experiment_runs()`: Globs all seed runs for a given experiment prefix
  - `load_registry()`, `update_registry()`, `clear_registry()`: File-based tracking of analyzed experiments
  - `scan_for_experiments()`: Discovers all `exp_*` experiments in results directory
  - `identify_new_experiments()`: Returns experiments not in registry

- **35 new comprehensive tests** in `tests/test_analysis_pipeline.py`:
  - 6 tests for experiment prefix extraction with various folder name formats
  - 3 tests for finding experiment runs
  - 4 tests for registry management (load, update, clear, deduplicate)
  - 2 tests for experiment discovery and identification
  - 4 tests for action array normalization
  - 3 tests for mutual information computation
  - 4 tests for bin creation with edge cases
  - 3 tests for action-attribute association computation
  - 2 tests for CSV writing (action drivers)
  - 2 tests for CSV writing (action-attribute associations)
  - 2 integration tests for full pipeline workflows

- **Fixed timestamp handling bug**: Regex pattern now handles YYYYMMDD_HHMMSS format correctly

#### PatientGenerator Configuration Discoverability (January 2026)

- **New class constant `PatientGenerator.REQUIRED_CONFIG_KEYS`**: Explicit list of all required configuration keys
  - Improves discoverability: users can inspect class to see what's required
  - Used internally for validation; referenced in error messages
  - Self-documenting: developers and tools can reference programmatically

- **New class method `PatientGenerator.default_config()`**: Get template configuration with sensible defaults
  - Preferred way to initialize PatientGenerator: start with template, customize as needed
  - Returns dict with all required keys + optional observation parameters + visibility setting
  - Each call returns a fresh dict (mutable, no shared references)
  - Detailed docstring with usage examples

- **11 new tests** covering:
  - REQUIRED_CONFIG_KEYS constant structure and contents
  - default_config() method behavior (mutability, completeness, validness)
  - Error messages now reference REQUIRED_CONFIG_KEYS and default_config()
  - Recommended usage patterns for config customization

- **Improved error messages**: ValueError for missing config keys now references both REQUIRED_CONFIG_KEYS and default_config()

#### Strict Component Compatibility Validation (January 2026)

- **Environment now enforces component class constants**: ABXAMREnv.__init__ now fails loudly if components lack required constants
  - Validates RewardCalculator has `REQUIRED_PATIENT_ATTRS` class constant
  - Validates PatientGenerator has `PROVIDES_ATTRIBUTES` class constant
  - Validates that all required attributes are provided (attribute compatibility check)
  - Prevents silent failures from incompatible components

- **Improved error messages**: Clear guidance when component constants are missing or attributes are incompatible

- **4 new tests** covering:
  - Error on missing REQUIRED_PATIENT_ATTRS constant
  - Error on missing PROVIDES_ATTRIBUTES constant
  - Error on missing required attributes (compatibility check)
  - Success case with compatible components

### Changed

#### Architecture Cleanup: Clean Orchestration Pattern (January 2026)

**BREAKING: Environment no longer accepts `visible_patient_attributes` parameter**
- `ABXAMREnv.__init__` no longer has `visible_patient_attributes` parameter
- Visibility configuration lives entirely in `PatientGenerator` (set via PG config or attribute)
- Environment reads visibility from `PatientGenerator` instance internally via `PG.observe()` and `PG.obs_dim()`
- **Rationale**: Environment doesn't need to know which specific attributes are visible; it just delegates observation construction to PG
- **Migration**: Remove `visible_patient_attributes` from env instantiation; set in PatientGenerator config instead
- All attribute name validation moved to PatientGenerator
- `create_environment()` no longer extracts/passes visibility parameter
- Test helpers updated to set visibility in PG config
- All 143 tests passing

**BREAKING: PatientGenerator now owns visibility configuration**
- `visible_patient_attributes` must be in `patient_generator` config section (not `environment`)
- `create_patient_generator()` fails loudly if visibility missing from PG config
- `create_environment()` fails loudly if env config contains `patient_generator` or `visible_patient_attributes`

**Enforced instantiation order: RewardCalculator → PatientGenerator → Environment**
- RewardCalculator and PatientGenerator must be created first, then passed to Environment
- No internal PG/RC creation within Environment or `create_environment()`
- All config duplication removed between PG and env sections

**Removed backward compatibility fallbacks**
- Environment no longer has fallback paths for `observe()` or `obs_dim()`
- PatientGenerator must implement `observe()` and `obs_dim()` (enforced via inheritance from `PatientGeneratorBase`)
- Clear error messages if PG missing required methods

**Code cleanup**
- Removed env-level `visible_patient_attributes` attribute and parameter
- Removed attribute name validation from ABXAMREnv (delegated to PG)
- Removed `_get_patient_attribute_value()` fallback extraction
- Removed dead code in `create_environment()` that duplicated PG instantiation
- Removed all `pdb.set_trace()` debugging statements from utils

**Test suite validated**: All 143 tests passing with new architecture

**Streamlit GUI Refactoring**
- `gui/experiment_runner.py`: Now places `visible_patient_attributes` in `patient_generator` config section (not `environment`)
  - Reads defaults from `patient_generator/default.yaml` for visibility checkboxes
  - Assembles PG config with visibility before setting `config[\"patient_generator\"]`
  - Removed `config[\"environment\"][\"patient_generator\"]` duplication
- `gui/experiment_viewer.py`: Added separate display section for `patient_generator` config
  - Displays environment, patient_generator, reward_calculator, and training configs separately
  - Updated exclusion list when displaying other parameters
**Circular Dependency Elimination**
- Created new `abx_amr_env/types.py` module containing shared type definitions
- Moved `Patient` dataclass from `patient_generator.py` to `types.py`
- Updated imports:
  - `patient_generator.py` imports `Patient` from `types.py`
  - `reward_calculator.py` imports `Patient` from `types.py` (not from `patient_generator.py`)
  - `abx_amr_env.py` now imports `PatientGeneratorBase` and `RewardCalculatorBase` at module level (removed deferred imports from `__init__`)
  - `__init__.py` exports `Patient` from `types.py` for public API
  - All test files updated to import `Patient` from `types.py`
- **Result**: Completely eliminated circular import dependencies
  - All imports are at module level (no deferred imports in methods)
  - Clean, linear import graph with no cycles
  - All 143 tests passing
### Added

#### Training Refactor & E2E Verification (January 2026)
- Episode-based training parameters across configs and CLI
  - `total_num_training_episodes`, `save_freq_every_n_episodes`, `eval_freq_every_n_episodes`, `num_eval_episodes`
  - Conversion handled internally for SB3 timesteps; legacy `total_timesteps` removed
- End-to-end verification
  - Full unit test suite passes (142 tests)
  - Short PPO run (3 episodes) completes with checkpoints and evals
  - Visualization tools validated: `visualization_outputs/*` generated by `visualize_env_behavior.py`
  - Micro-epoch debug validated: `debug_micro_epochs.py` prints per-step diagnostics and saves final model
- Added `requirements.txt` with core dependencies for experiments and GUI apps

#### Analysis Pipeline & AMR Aggregation (January 2026)
- **`experiments/analyze_patient_data.py`**: Post-training trajectory analysis tool
  - Aggregate patient observation errors (bias, MAE, RMSE) per attribute
  - Reward–error correlations (Pearson/Spearman) to identify decision-critical attributes
  - Action–attribute associations: quantile-binned prescribe rates + mutual information
  - **AMR summary statistics**: Per-antibiotic mean/std/min/max, mean delta per timestep, fraction above threshold
  - Outputs: JSON + CSV artifacts for all metrics
- **Visualization enhancements**: Added temporal AMR plots and clarified observation error trends
  - `actual_amr_over_time.png` / `observed_amr_over_time.png`: mean AMR per step within eval episodes with percentile bands (true vs visible AMR)
  - Renamed `error_by_episode.png` to `observation_error_over_time_per_episode.png` for clarity
  - Documentation updated in `docs/analysis_metrics_explanation.md` to cover new plots and interpretations
- **AMR persistence in eval trajectories**: `DetailedEvalCallback` now saves AMR time series
  - `actual_amr_levels` and `visible_amr_levels` arrays per episode
  - Antibiotic name ordering preserved in metadata for alignment
  - Fallback name inference from AMR dict keys when env metadata unavailable
- **Action mapping persistence**: Training now saves `abx_name_to_index` and `index_to_abx_name` in config.yaml
  - Enables analysis to correlate action indices with antibiotic names
  - Persisted for both fresh training and continued runs
- **`docs/analysis_metrics_explanation.md`**: Comprehensive metric documentation
  - Interpretation guidelines for observation errors and reward correlations
  - Action–attribute associations: how to read prescribe rates and mutual information
  - AMR summary stats: mean/delta interpretation, threshold usage, sample size cautions

#### Patient Generator & Heterogeneity (January 2026)
- **PatientGenerator integration into ABXAMREnv**: Environment now requires externally initialized PatientGenerator (or PatientGeneratorMixer) and configurable `visible_patient_attributes` for flexible observations.
- **PatientGeneratorMixer**: Mixer combines multiple PatientGenerator instances by proportion; propagates seeds to child generators and can be passed directly to the environment.
- **Config updates**: Experiment YAMLs (`ppo_baseline.yaml`, `a2c_exploration.yaml`, `dqn_small_cohort.yaml`) now include `patient_generator` sections and visible attribute lists.
- **Integration tests**: Added `tests/test_patient_generator_integration.py` (11 tests) covering cohort resampling, observation shapes, AMR accumulation/reset, seed reproducibility, and multiplier flow.
- **Mixer tests**: Added `tests/test_patient_generator_mixer.py` (17 tests) covering validation, proportions/rounding, shuffling, reproducibility, and seed synchronization.

#### Training & E2E Validation
- **PPO sanity run**: End-to-end training for 10k steps with heterogeneous patients; checkpoints and diagnostics saved under `results/ppo_baseline_20260109_194348/`.

#### GUI Applications (January 2026)
- **`gui/experiment_runner.py`**: Streamlit-based interactive experiment launcher
  - Configure environment parameters (antibiotics, AMR dynamics, crossresistance)
  - Set reward calculator parameters
  - Configure training hyperparameters
  - Live training log streaming
  - Auto-display results after completion
  - Support for resuming/continuing prior experiments
- **`gui/experiment_viewer.py`**: Streamlit-based experiment results browser
  - Browse all experiment runs (sorted newest first)
  - Filter experiments by name
  - View full configuration (organized by sections)
  - Download config as YAML
  - Display diagnostic plots grouped by category
  - Responsive 2-column image grid
- **`gui/launch_apps.py`**: Unified launcher for both GUI apps
- **`gui/README.md`**: Documentation for GUI usage and workflow

#### Visualization & Debugging Tools
- **`experiments/visualize_env_behavior.py`**: Environment behavior visualization tool
  - Plot AMR leaky balloon responses to puff sequences
  - Sample random actions and save observation/reward data
  - Configurable output folder for plots
- **`experiments/make_figures_for_experiment_run.py`**: Post-training diagnostic plotting
  - Load trained agents and generate trajectory visualizations
  - Plot metrics from trained agent episodes
  - Support for PPO, DQN, and A2C algorithms
- **`experiments/debug_micro_epochs.py`**: Micro-epoch debugging tool
  - Run training in small bursts with per-step visibility
  - Integration with StepPrinter wrapper
  - Useful for short debug sessions without long training runs

#### Environment Wrappers & Formatters
- **`experiments/wrappers.py`**: `StepPrinter` wrapper for transition logging
  - Prints each env transition (obs/action/reward/termination/info)
  - Configurable formatting and logging to file
  - Support for custom step formatters
  - Optional mirroring to stdout
  - Line-buffered file output for ordered logs
- **`experiments/formatters.py`**: Custom formatters for human-readable step output
  - `abx_amr_step_formatter()`: Translates ABX-AMR env steps into readable format
  - `minimal_step_formatter()`: Minimal action/reward logging for quiet runs

#### Utility Functions
- **`experiments/utils.py`**: Extended training utilities
  - `plot_metrics_trained_agent()`: Generate diagnostic plots from trained models
  - `run_episode_and_get_trajectory()`: Collect full episode trajectories for analysis
  - Enhanced configuration and directory management

#### Project Structure & Organization
- New standardized directory layout following Python best practices:
  - `docs/`: Documentation (moved from `specs/`)
  - `examples/`: Demo scripts and quick-start examples
  - `experiments/`: RL training scripts and configurations
  - `results/`: Training outputs (logs, checkpoints, metrics)
  - Proper separation of concerns between core environment code, examples, and experiments

#### Training Infrastructure
- **`experiments/train.py`**: Unified entry point for RL training with all algorithms
  - Config-driven design via YAML files
  - Supports PPO, DQN, and A2C algorithms
  - Automatic run directory creation with timestamps
  - Integration with TensorBoard logging
  - Command-line interface with seed override capability

- **`experiments/utils.py`**: Reusable utility functions for training
  - `load_config()`: YAML configuration loading
  - `create_reward_calculator()`: RewardCalculator instantiation from config
  - `create_environment()`: ABXAMREnv setup from config
  - `create_agent()`: Algorithm-agnostic agent factory
  - `setup_callbacks()`: Training callbacks (evaluation, checkpointing)
  - `create_run_directory()`: Timestamped output directory management
  - Config and summary persistence

#### Experiment Configurations
- **`experiments/configs/ppo_baseline.yaml`**: PPO training with MultiDiscrete action space
  - 3 antibiotics, 10 patients per step
  - Network: 128-128 MLP
  - Learning rate: 3.0e-4
  - Total timesteps: 100,000

- **`experiments/configs/dqn_small_cohort.yaml`**: DQN training with flattened Discrete action space
  - 2 antibiotics (to keep action space manageable: 3³ = 27 actions)
  - 3 patients per step (smaller cohort for flat mode compatibility)
  - Network: 128-128 MLP
  - Buffer size: 100,000
  - Total timesteps: 50,000

- **`experiments/configs/a2c_exploration.yaml`**: A2C training with MultiDiscrete action space
  - 3 antibiotics, 10 patients per step
  - Network: 128-128 MLP
  - Learning rate: 7.0e-4
  - Total timesteps: 50,000

#### Documentation
- Moved `ENVIRONMENT_SPEC.md` and `notes.md` to `docs/` directory
- Created `CHANGELOG.md` for tracking project evolution

### Changed

#### Environment & Training Plumbing
- **Seed synchronization**: ABXAMREnv now synchronizes PatientGenerator/PatientGeneratorMixer seeds (including mixer children) with RewardCalculator for reproducibility.
- **Environment creation**: `experiments/utils.py` builds PatientGenerator externally from config, deep-copies environment kwargs, and validates required patient visibility settings.
- **Observations**: ABXAMREnv constructs observations from per-patient visible attributes plus AMR levels; shape = `(num_patients * len(visible_attrs) + num_abx,)`.

#### Documentation
- **single_agent_experiments.md**: Marked PatientGenerator tasks complete; added heterogeneous-patient experiment sets (coarse observability ablations, noise/bias robustness, mixed populations, distribution shift).

#### Environment Enhancements
- **Configurable AMR observation noise and bias**: Added `add_noise_to_visible_AMR_levels` and `add_bias_to_visible_AMR_levels` parameters to `ABXAMREnv`
  - Allows simulation of imperfect AMR surveillance data
  - Noise parameter adds Gaussian noise to visible AMR levels
  - Bias parameter adds systematic offset to visible AMR levels
- **Configurable infection probability variability**: Added `std_dev_probability_of_infection` parameter
  - Allows stochastic variation in baseline infection probability
  - Enables modeling of time-varying infection risk

#### Training Infrastructure Improvements
- Enhanced experiment result organization with automatic figure generation
- Improved config generation from Streamlit with timestamped filenames
- Better experiment tracking with consistent naming conventions

#### Environment Improvements
- Added configurable action mode to `ABXAMREnv`:
  - `action_mode="multidiscrete"` (default): Per-patient discrete choices, suitable for PPO/A2C/MaskablePPO
  - `action_mode="flat"`: Flattened Discrete space, suitable for DQN (with small cohorts)
- Improved determinism:
  - Environment now owns its RNG (`self.np_random`)
  - RewardCalculator RNG is re-seeded when `reset(seed=...)` is called
  - Patient infection sampling uses env RNG instead of global numpy state
- Added encoding/decoding helpers (`_encode_action()`, `_decode_action()`) for flat mode conversion

#### Tests
- Added `test_flat_action_mode_matches_multidiscrete()` to verify action mode equivalence
- All 27 existing tests continue to pass

### How to Use

**Launch GUI applications:**
```bash
# Run both apps simultaneously (recommended workflow)
python gui/launch_apps.py

# Or run individually:
streamlit run gui/experiment_runner.py
streamlit run gui/experiment_viewer.py --server.port 8502
```

**Debug environment behavior:**
```bash
# Visualize AMR dynamics and rewards for specific config
python experiments/visualize_env_behavior.py --config experiments/configs/ppo_baseline.yaml --output_folder viz_output

# Debug training with per-step printing
python experiments/debug_micro_epochs.py --config experiments/configs/ppo_baseline.yaml --micro-steps 10 --chunks 50
```

**Generate diagnostic plots from trained agent:**
```bash
python experiments/make_figures_for_experiment_run.py --experiment_run ppo_baseline_20260108_134211
```

**Run a training experiment:**
```bash
python experiments/train.py --config experiments/configs/ppo_baseline.yaml
python experiments/train.py --config experiments/configs/dqn_small_cohort.yaml --seed 42
python experiments/train.py --config experiments/configs/a2c_exploration.yaml
```

**View TensorBoard logs:**
```bash
tensorboard --logdir results/logs
```

**Key observations:**
- DQN uses `action_mode="flat"` with small cohorts (3 patients) to avoid exponential action space blowup
- PPO and A2C use `action_mode="multidiscrete"` and support larger cohorts (10 patients)
- All algorithms share the same environment code; differences are purely in algorithm configuration
- New GUI tools streamline experiment configuration and result exploration
- Environment now supports imperfect AMR surveillance via noise/bias parameters
- Debugging tools provide granular visibility into agent-environment interactions

---

## [0.1.0] - 2025-12-30

### Added
- Initial implementation of `ABXAMREnv` with leaky balloon AMR models
- `RewardCalculator` class with lambda-weighted individual/community reward balancing
- `AMR_LeakyBalloon` and `AMR_DiscreteCapacitor` models for resistance tracking
- Comprehensive test suite (27 tests) covering environment, reward, and AMR dynamics
- Example scripts: `play_with_abx_amr_env.py`, `gym_example.py`

### Features
- Gymnasium-compatible custom environment for antibiotic prescribing RL
- MultiDiscrete action space (per-patient antibiotic selection)
- Stochastic patient infection, sensitivity, and adverse effects
- Seeded reproducibility for deterministic runs
- Visible AMR levels updated on configurable schedule
- Detailed reward component breakdown in step info

[Unreleased]: https://github.com/jl56923/abx_amr_simulator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jl56923/abx_amr_simulator/releases/tag/v0.1.0
