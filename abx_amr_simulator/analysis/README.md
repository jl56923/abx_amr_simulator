# Analysis Tools: Diagnostic Analysis & Evaluative Plots

This document explains the outputs and interpretation of the two main analysis tools in `abx_amr_simulator`: **diagnostic_analysis** and **evaluative_plots**.

---

## Diagnostic Analysis

### Purpose
Validates that patient observations match configured noise/bias parameters. Identifies whether observation errors correlate with reward—helping diagnose whether observation quality impacts agent learning.

**Module**: `abx_amr_simulator.analysis.diagnostic_analysis`

**CLI**: `abx-amr-diagnostic-analysis --experiment-prefix exp_name`

### Outputs

#### Observation Error Metrics
Compares per-patient **observed** attributes to their **true** values across all evaluation episodes.

**Metrics** (saved as `.json` and `.csv`):
- **attribute**: Patient attribute assessed (e.g., `prob_infected`, `benefit_value_multiplier`)
- **bias**: Mean(observed − true)
  - Positive → systematic overestimation
  - Negative → underestimation
  - Near 0 → little systematic bias
- **mae**: Mean absolute error (average error magnitude regardless of direction)
- **rmse**: Root mean square error (penalizes large errors more than MAE; always ≥ MAE)
- **samples**: Total comparisons used (steps × patients × episodes)

**Interpretation**:
- High bias (±) indicates configured observation bias is manifesting; use as sanity check against config
  - Example: If configured with 1.2× bias, expect positive bias of similar magnitude
- High MAE with low bias suggests observation noise is dominating; validate noise settings in generator config
- RMSE much larger than MAE indicates occasional large deviations; inspect episodes to verify this matches expectations

**Why this matters**: These metrics validate that the PatientGenerator is producing observations with the configured noise/bias parameters. If metrics diverge from expected patterns, it indicates a potential bug in the observation sampling logic or misalignment between config and implementation.

**Artifacts**:
- `observation_error_metrics.json` / `observation_error_metrics.csv` in `analysis_output/`

---

#### Reward–Observation Error Correlations
Measures whether step-level observation errors correlate with step-level rewards.

**Metrics** (saved as `.json` and `.csv`):
- **pearson**: Linear correlation between per-step mean observation error and per-step reward
- **spearman**: Rank-based (monotonic) correlation; more robust to non-linear relationships
- **samples**: Number of paired (error, reward) points across episodes

**Interpretation**:
- **Positive correlation** (e.g., +0.3): Larger observation errors → higher rewards (unusual; verify reward composition and policy)
- **Negative correlation** (e.g., −0.4): Larger observation errors → lower rewards (typical when attribute is decision-critical)
- **Near 0**: Little evidence that errors in this attribute affect rewards

**Important caveats**:
- Reward is composite: λ × community_amr_penalty + (1−λ) × sum(individual_patient_rewards)
  - Interpret correlations in light of your λ and configuration
- Correlation ≠ causation: Multiple attributes, policy behaviors, and AMR dynamics co-vary; treat strong signals as hypotheses

**Why this matters**: Reveals whether imperfect patient observations actually hurt the agent's ability to learn good policies. High negative correlation suggests the agent relies heavily on that attribute; high positive correlation is a red flag.

**Artifacts**:
- `reward_error_correlations.json` / `reward_error_correlations.csv` in `analysis_output/`

---

## Evaluative Plots

### Purpose
Aggregates learning outcomes across multiple training seeds, visualizes ensemble performance with uncertainty bands, and identifies which patient attributes drive prescribing decisions.

**Module**: `abx_amr_simulator.analysis.evaluative_plots`

**CLI**: `abx-amr-evaluative-plots --experiment-prefix exp_name`

### Outputs

#### Ensemble Metrics & Plots

Runs each trained agent (different training seeds) through 10 evaluation episodes, aggregates trajectories, and generates plots showing **mean curves with 10-90 percentile bands**.

**Why percentile bands?** With multiple seeds and stochastic environment (heterogeneous patient generation), percentile bands show the range of plausible outcomes. The 10th-90th percentile captures the middle 80% of outcomes; dashed lines show the 25-75 interquartile range (IQR).

**Key plots** (all with mean ± 10-90% percentile bands):

1. **amr_levels_over_time.png**
   - Two subplots: Actual AMR (internal state) and Visible AMR (agent's noisy observations)
   - Shows how each antibiotic's resistance evolves during episodes
   - Wide percentile bands indicate high environment stochasticity; narrow bands indicate consistent AMR evolution

2. **reward_components_over_time.png**
   - Left subplot: Individual vs. Community reward components (cumulative over episode)
   - Middle subplot: Normalized individual vs. normalized community (scaled 0-1)
   - Right subplot: Total reward (sum of weighted components)
   - Reveals the λ trade-off: steep individual curve → agent prioritizes clinical benefit; steep community curve → agent avoids AMR

3. **clinical_benefits_failures_adverse_events_over_time.png**
   - Cumulative counts of clinical benefits (successful treatments), failures (untreated infections), and adverse events
   - Rising clinical benefits with flat failures suggests good prescribing; rising failures suggests agent isn't treating infections

4. **outcome_counts_over_time.png** (4 subplots)
   - Not Infected No Treatment: Patients without infection, not treated
   - Not Infected Treated: Unnecessary antibiotic use (wasted prescriptions)
   - Infected No Treatment: Untreated infections (clinical failures)
   - Infected Treated Overall + per-antibiotic breakdowns: Successful treatments
   - Reveals prescribing strategy: conservative (high Not Infected No Treatment) vs. aggressive (high Not Infected Treated)

5. **abx_prescriptions_over_time.png**
   - Cumulative prescription counts per antibiotic
   - Compare across antibiotics to see if agent favors specific drugs (e.g., broad-spectrum over narrow-spectrum, more 'sensitive' to overprescribing, initial states of each antibiotic)

6. **Per-antibiotic sensitive vs. resistant plots** (one per antibiotic)
   - Cumulative counts of sensitive and resistant infections treated with each antibiotic
   - Rising resistant counts = wasted prescriptions due to AMR
   - Reveals whether agent is adapting to evolving resistance

---

#### Action-Attribute Associations

Quantifies how the agent's prescribing decisions depend on observed patient attributes.

**Method**:
- Bin each observed attribute into quantiles (default: 5 bins) across all patient-step samples
- For each bin × antibiotic combination, compute:
  - `count_samples`: How many (patient, step) pairs fall in this bin
  - `count_prescribe`: How many were treated with this antibiotic
  - `prescribe_rate`: Fraction prescribed (count_prescribe / count_samples)
  - `mean_reward`: Average reward when prescribing to patients in this bin
- Compute **mutual information** (MI) between binned attribute and binary "prescribed this antibiotic" flag
  - MI ∈ [0, ∞): Higher values indicate stronger attribute-prescription relationship

**Artifacts** (saved as `.json` and `.csv`):
- `action_drivers.json` / `action_drivers.csv`: Rows with `[attribute, bin_label, bin_low, bin_high, antibiotic, count_samples, count_prescribe, prescribe_rate, mean_reward]`
- `action_attribute_associations.json` / `action_attribute_associations.csv`: Rows with `[attribute, antibiotic, mutual_information, samples]`

**Interpretation**:
- **Rising prescribe_rate across bins** → Agent leans more on that antibiotic as attribute increases
  - Example: Higher `prob_infected` → higher `prescribe_rate` (sensible behavior)
- **Flat prescribe_rate with low MI** → Attribute doesn't drive decisions; agent is insensitive to it
  - Example: `benefit_value_multiplier` flat across all bins → agent doesn't differentiate patient benefit levels
- **High MI, moderate prescribe_rate** → Attribute strongly influences decision but in a nuanced way

**Caveats**:
- Sparsely populated bins (low `count_samples`) make rates noisy; outliers in `mean_reward` often reflect rare events
- Mutual information is computed on binned data; bin choice (default: 5 quantiles) affects sensitivity
- Missing attributes in observations → all agents see identical values → MI = 0 for that attribute

---

## Interpretation Guide

### Scenario 1: Good learning (λ ≈ 0.3)
- Individual reward rising steeply, community reward moderate
- Clinical benefits rising, failures low
- Action-attribute associations show rising prescribe_rate with prob_infected
- AMR levels modest and stable (or declining with good treatment)

**Inference**: Agent learned trade-off, prescribing strategically based on infection risk.

### Scenario 2: Over-prescribing (λ ≈ 0.0)
- Clinical benefits very high, but AMR levels rising quickly
- Not Infected Treated (unnecessary prescriptions) very high
- Prescribe_rate near 1.0 for all attributes
- High mutual information between all attributes and prescriptions (agent treats uniformly)

**Inference**: Agent learned extreme clinical strategy, ignoring AMR dynamics.

### Scenario 3: Under-prescribing (λ ≈ 1.0)
- Individual reward flat, community reward drives trajectory
- Clinical failures rising
- Prescribe_rate near 0 for most attributes
- Low mutual information (agent ignores attribute signals)

**Inference**: Agent learned to avoid AMR at cost of patient care.

### Scenario 4: Poor learning (any λ)
- Noisy/contradictory patterns (high percentile bands, inconsistent trends)
- Action-attribute associations flat with low MI across all attributes
- Outcome counts wildly varying across seeds

**Inference**: Agent failed to converge to interpretable policy; check config, training time, or environment dynamics.

---

## Running the Tools

```bash
# After training completes, run both tools with same experiment prefix:

abx-amr-diagnostic-analysis --experiment-prefix exp_1.a_single_abx_lambda0.0
abx-amr-evaluative-plots --experiment-prefix exp_1.a_single_abx_lambda0.0
```

Both tools:
- Look for experiment folders matching the prefix (e.g., `exp_1.a_single_abx_lambda0.0_seed1_*`, `exp_1.a_single_abx_lambda0.0_seed2_*`, etc.)
- Group results by seed (assume folders are named `prefix_seedN_timestamp/`)
- Generate aggregate plots and CSV files in `analysis_output/`

**Command-line options**:
- `--force`: Regenerate all plots even if they already exist
- `--prefix`: Experiment prefix to search for (default: `exp_`)
- `--num-eval-episodes`: For evaluative_plots, run each seed through this many episodes (default: 10)

---

## Best Practices

1. **Always run both tools**: Diagnostic analysis validates observation quality; evaluative plots show learning success
2. **Compare across λ values**: Run multiple values of λ (e.g., 0.0, 0.3, 0.7, 1.0) and compare ensemble plots side-by-side
3. **Check for convergence**: If percentile bands narrow over episodes, agent is converging; if they widen, agent is diverging
4. **Correlate with config**: Verify that action-attribute associations match your intuitions about reward function
5. **Iterate on seeds**: Run at least 3-5 seeds per experiment; single-seed results can be misleading
