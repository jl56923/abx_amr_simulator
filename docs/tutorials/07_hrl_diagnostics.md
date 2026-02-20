# Tutorial 8: HRL Diagnostics

**Goal**: Learn to analyze HRL training runs and interpret diagnostic plots that reveal option selection patterns, effectiveness, and strategy evolution.

**Prerequisites**: Completed Tutorial 6 (HRL Quick Start) and Tutorial 7 (Options Library Setup)

**What are HRL diagnostics?** Specialized analysis plots that reveal how the manager policy uses options, which options are most effective, and whether the agent develops coherent temporal strategies.

---

## Overview

HRL diagnostic analysis workflow:
1. **Train HRL agent**: Complete training run with option library
2. **Run diagnostic analysis**: Generate plots from evaluation trajectories
3. **Interpret patterns**: Understand option selection frequency, effectiveness, and transitions
4. **Iterate**: Adjust option library or hyperparameters based on insights

**Key diagnostic plots:**
- **Option Selection Histogram**: Which options does the manager prefer?
- **Macro-Decision Frequency**: How often does the manager switch options?
- **Option Effectiveness**: Which options yield highest rewards?
- **Option-AMR Strategy**: Does manager adapt option choice to AMR state?
- **Option Transitions**: Are there common option sequences (Markov patterns)?

---

## Step 1: Train HRL Agent with Trajectory Logging

Ensure your training run saves detailed evaluation trajectories (enabled by default in HRL configs):

```bash
cd myproject

# Train HRL PPO agent
python -m abx_amr_simulator.training.train \
  --umbrella-config $(pwd)/configs/umbrella_configs/hrl_ppo_default.yaml \
  --seed 42
```

**Required config setting** (already in `hrl_ppo_default.yaml`):

```yaml
training:
  log_patient_trajectories: true  # ← Must be true for diagnostics
  num_eval_episodes: 10           # ← More episodes = better statistics
```

**Training outputs:**

```
results/
└── hrl_ppo_default_20260217_143027/
    ├── full_agent_env_config.yaml   ← Training config
    ├── final_model.zip              ← Trained manager policy
    ├── logs/                        ← TensorBoard logs
    └── eval_logs/                   ← Evaluation trajectories (needed for diagnostics)
        ├── eval_0_step_1280.npz
        ├── eval_1_step_2560.npz
        └── ...
```

---

## Step 2: Run Diagnostic Analysis

### Single-Run Diagnostics

Analyze one completed HRL training run:

```bash
python -m abx_amr_simulator.analysis.diagnostic_analysis \
  --experiment-name hrl_ppo_default_20260217_143027 \
  --results-dir $(pwd)/results \
  --analysis-dir $(pwd)/analysis \
  --force
```

**Arguments:**
- `--experiment-name`: Name of completed experiment folder in `results/`
- `--results-dir`: Path to results directory (defaults to `./results`)
- `--analysis-dir`: Where to save diagnostic outputs (defaults to `./analysis`)
- `--force`: Overwrite existing analysis outputs

**Expected output:**

```
[HRL Diagnostics] Starting analysis for run: hrl_ppo_default_20260217_143027
  ✓ Loaded 10 evaluation episodes from eval_logs/
  ✓ Saved option selection histogram to analysis/hrl_ppo_default_20260217_143027/figures_hrl/option_selection_histogram.png
  ✓ Saved macro decision frequency plot to analysis/hrl_ppo_default_20260217_143027/figures_hrl/macro_decision_frequency.png
  ✓ Saved option effectiveness plot to analysis/hrl_ppo_default_20260217_143027/figures_hrl/option_effectiveness.png
  ✓ Saved option-AMR strategy plot to analysis/hrl_ppo_default_20260217_143027/figures_hrl/option_amr_strategy.png
  ✓ Saved option transition heatmap to analysis/hrl_ppo_default_20260217_143027/figures_hrl/option_transitions_heatmap.png
  ✓ Saved top trigrams to analysis/hrl_ppo_default_20260217_143027/figures_hrl/option_transitions_trigrams.csv
[SUCCESS] HRL diagnostics saved to analysis/hrl_ppo_default_20260217_143027/figures_hrl/
```

### Multi-Seed Diagnostics (Optional)

If you trained multiple seeds with the same config prefix:

```bash
python -m abx_amr_simulator.analysis.diagnostic_analysis \
  --experiment-name hrl_ppo_default \
  --aggregate-by-seed \
  --results-dir $(pwd)/results \
  --analysis-dir $(pwd)/analysis
```

This aggregates diagnostics across all runs matching the prefix (e.g., `hrl_ppo_default_seed42`, `hrl_ppo_default_seed123`).

---

## Step 3: Interpret Diagnostic Plots

### Plot 1: Option Selection Histogram

**File**: `figures_hrl/option_selection_histogram.png`

**What it shows**: Frequency of each option (as fraction of total steps) across all evaluation episodes.

**How to interpret:**
- **Uniform distribution**: Manager explores all options equally (may need more training or higher entropy)
- **Dominant option**: One option is selected >50% of the time (check if it's optimal or overfitting)
- **Unused options**: Some options never selected (consider removing from library)
- **Multi-modal**: Manager uses 2-3 options frequently (common pattern for well-trained agents)

**Example insights:**
```
Option A_10: 35% → Manager prefers aggressive treatment
Option no_treatment_5: 30% → Conservative baseline
Option B_15: 20% → Situational use
ALT_AABBA: 15% → Rare cycling strategy
```

**Red flags:**
- All options used equally after 1000+ episodes → Undertrained or reward signal unclear
- One option used >80% → Overfitting or other options not rewarding

---

### Plot 2: Macro-Decision Frequency

**File**: `figures_hrl/macro_decision_frequency.png`

**What it shows**: Distribution of inter-decision times (how many environment steps between manager decisions).

**How to interpret:**
- **Mean inter-decision time**: Average option duration (e.g., 10 steps)
- **Spread**: Narrow peak = deterministic durations (block options), wide spread = variable durations (heuristic options)
- **Multi-modal**: Multiple peaks suggest manager uses options of different durations strategically

**Example insights:**
```
Mean: 10 steps → Manager makes decisions every 10 steps on average
Std: 3 steps → Moderate variability (mix of 5, 10, 15-step options)
Peak at 10 steps → Majority of options have 10-step duration
```

**Red flags:**
- Mean = 5 steps with option durations 5-15 → Manager terminates long options early (check option reward signals)
- Mean = 15 steps with only 5-step options → Unexpected (check option library config)

---

### Plot 3: Option Effectiveness

**File**: `figures_hrl/option_effectiveness.png`

**What it shows**: Box plot of macro-reward distribution for each option (reward accumulated during option execution).

**How to interpret:**
- **Median reward**: Central tendency for each option
- **Interquartile range (box)**: Variability of rewards (narrow = consistent, wide = context-dependent)
- **Outliers**: Rare high/low reward instances

**Example insights:**
```
A_10: Median +15, tight IQR → Consistently effective
B_15: Median +10, wide IQR → Situationally effective
no_treatment_5: Median -5, tight IQR → Consistently poor
↳ Insight: Manager should learn to avoid no_treatment_5
```

**Red flags:**
- Frequently selected option has low median reward → Manager hasn't learned optimal policy yet
- Rarely selected option has highest median reward → Exploration issue (increase entropy coefficient)

---

### Plot 4: Option-AMR Strategy

**File**: `figures_hrl/option_amr_strategy.png`

**What it shows**: Heatmap of P(Option | AMR State) — conditional probability of selecting each option given AMR resistance level (binned into quartiles: Low, Med-Low, Med-High, High).

**How to interpret:**
- **Row patterns**: How manager's option choice changes with AMR
- **Column patterns**: Which AMR states prefer which options
- **Diagonal patterns**: Options specialized for specific AMR ranges

**Example insights:**
```
Low AMR (row 0): High P for A_10 (0.6) → Aggressive with low resistance
High AMR (row 3): High P for B_15 (0.5) → Switch to alternative antibiotic
↳ Insight: Manager adapts strategy to AMR—desirable behavior!
```

**Red flags:**
- All rows identical → Manager ignores AMR (check if AMR in observations)
- Low AMR favors no_treatment → Counterintuitive (check reward structure)

---

### Plot 5: Option Transitions

**Files**: 
- `figures_hrl/option_transitions_heatmap.png` (bigram heatmap)
- `figures_hrl/option_transitions_trigrams.csv` (top-10 trigrams)

**What it shows**: 
- **Heatmap**: Frequency of option transitions (which option follows which)
- **Trigrams CSV**: Most common 3-option sequences

**How to interpret:**
- **Diagonal (self-loops)**: Same option repeated (rare in deterministic options)
- **Off-diagonal hot spots**: Common transitions (e.g., A_10 → B_15)
- **Uniform heatmap**: Random transitions (manager has no temporal strategy)
- **Trigrams**: Reveal longer-term patterns (e.g., A → B → A cycling)

**Example insights:**
```
Heatmap hot spot: A_10 → B_15 (count=45) → Manager alternates antibiotics
Top trigram: A_10 → B_15 → A_10 (count=18) → Explicit cycling strategy
↳ Insight: Manager learned temporal structure (good!)
```

**Red flags:**
- Uniform heatmap → No coherent strategy (may need more training)
- Single dominant path (e.g., A → A → A) → Overfitting to one strategy

---

## Step 4: Use Diagnostics to I improve Training

### Pattern 1: Dominant Option (Overfitting)

**Symptom**: One option used >70%, others ignored.

**Diagnosis**: Manager found local optimum, insufficient exploration.

**Solutions:**
1. Increase entropy coefficient in HRL  algorithm config:
   ```yaml
   # agent_algorithm/hrl_ppo.yaml
   ppo:
     ent_coef: 0.05  # Increase from 0.02 to encourage exploration
   ```

2. Add reward shaping for option diversity:
   ```yaml
   # Custom reward calculator (advanced)
   reward_calculator:
     option_diversity_bonus: 0.1  # Bonus for trying new options
   ```

3. Simplify option library (remove redundant options).

### Pattern 2: Unused Options

**Symptom**: Some options never selected (<1% frequency).

**Diagnosis**: Options not competitive or redundant.

**Solutions:**
1. Remove unused options from library (simplify)
2. Check option effectiveness plot—if unused option has high median reward, increase exploration
3. Retrain with smaller library (6-8 options instead of 12)

### Pattern 3: No AMR Adaptation

**Symptom**: Option-AMR strategy heatmap shows identical rows.

**Diagnosis**: Manager ignores AMR observations.

**Solutions:**
1. Verify AMR levels in observations (check `include_steps_since_amr_update_in_obs` in environment config)
2. Increase AMR observation frequency:
   ```yaml
   environment:
     update_visible_AMR_levels_every_n_timesteps: 1  # Update every step
   ```

3. Add AMR penalty weight in reward calculator:
   ```yaml
   reward_calculator:
     lambda_weight: 0.5  # Balance individual vs. community reward
   ```

### Pattern 4: Random Transitions

**Symptom**: Uniform transition heatmap, no trigram patterns.

**Diagnosis**: Manager hasn't learned temporal structure.

**Solutions:**
1. Train longer (more episodes)
2. Increase option durations (longer temporal horizon):
   ```yaml
   # In option library
   config_params_override:
     duration: 15  # Increase from 10
   ```

3. Use HRL_RPPO (LSTM manager for better temporal reasoning—see Tutorial 09)

---

## Step 5: Combine with TensorBoard Metrics

Cross-reference diagnostic plots with TensorBoard training curves:

```bash
tensorboard --logdir results/hrl_ppo_default_20260217_143027/logs
```

**Key metrics to check:**
- `rollout/ep_rew_mean`: Is reward improving?
- `train/entropy_loss`: Is exploration decreasing (should start high, decay slowly)?
- `train/policy_loss`: Is manager policy learning (should decrease)?

**Combined analysis example:**
```
TensorBoard: Reward plateaus at episode 500
Diagnostics: Option A_10 used 80% of time
↳ Conclusion: Manager found local optimum, increase entropy to explore more
```

---

## Troubleshooting

### "No .npz files found in eval_logs"

**Cause**: Training didn't save evaluation trajectories.

**Solution**: Verify `log_patient_trajectories: true` in training config and rerun training.

### "All .npz files were corrupted"

**Cause**: macOS creates hidden `._*` files that aren't valid `.npz` archives.

**Solution**: The analysis tool automatically skips these. If issue persists, manually delete:
```bash
cd results/hrl_ppo_default_*/eval_logs
rm -f ._*
```

### "Option library not found"

**Cause**: Option library path in `full_agent_env_config.yaml` doesn't match filesystem.

**Solution**: Diagnostics read option names from saved trajectories, not the library file. Check that `info` dicts in trajectories contain `option_name` field.

### "Heatmap shows all zeros"

**Cause**: Evaluation trajectories don't contain AMR data or patient attributes.

**Solution**: Ensure `log_patient_trajectories: true` captures full trajectory information during evaluation.

---

## What's Next?

✅ You've learned to analyze HRL training with diagnostic plots!

**Next tutorials**:
- **Tutorial 09**: [HRL RPPO Manager](09_hrl_rppo_manager.md) — Use recurrent manager for temporal reasoning and partial observability
- **Tutorial 11**: [Advanced Heuristic Worker Subclassing](11_advanced_heuristic_worker_subclassing.md) — Implement sophisticated heuristic options with attribute estimation

---

## Key Takeaways

1. **HRL diagnostics reveal manager strategy**: Option selection frequency, effectiveness, AMR adaptation, temporal patterns
2. **Five core plots**: Selection histogram, macro-decision frequency, effectiveness box plot, AMR strategy heatmap, transition analysis
3. **Interpretation patterns**: Dominant options (overfitting), unused options (redundancy), uniform transitions (undertraining)
4. **Diagnostic-driven iteration**: Use insights to adjust entropy, simplify library, improve reward shaping
5. **Training requirements**: Must set `log_patient_trajectories: true` and train with sufficient episodes (25+)
6. **Command**: `python -m abx_amr_simulator.analysis.diagnostic_analysis --experiment-name <name> --results-dir <path>`
7. **Output locations**: Analysis plots saved to `analysis/<experiment>/figures_hrl/`
