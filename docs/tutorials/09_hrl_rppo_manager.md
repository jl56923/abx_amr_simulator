# Tutorial 9: HRL RPPO Manager

**Goal**: Learn when and how to use recurrent manager policies (HRL_RPPO) for improved temporal reasoning and partial observability handling.

**Prerequisites**: Completed Tutorial 6 (HRL Quick Start) and optionally Tutorial 8 (HRL Diagnostics)

**What is HRL_RPPO?** Manager policy that uses **LSTM (recurrent neural network)** instead of feedforward network. The LSTM maintains hidden state across option selections, enabling better temporal reasoning and handling of partially observable environments.

---

## Overview

**HRL_PPO (feedforward) vs. HRL_RPPO (recurrent):**

| Feature | HRL_PPO | HRL_RPPO |
|---------|---------|----------|
| **Network Architecture** | Feedforward MLP | LSTM + MLP |
| **Memory** | No hidden state (Markovian) | LSTM hidden state across steps |
| **Observations** | Current step only | Full trajectory history (via LSTM) |
| **Training Speed** | Faster | Slower (LSTM overhead) |
| **Best For** | Fully observable, simple dynamics | Partial observability, temporal dependencies |

---

## When to Use HRL_RPPO

### Use HRL_RPPO if:

✅ **Partial observability**: Environment has hidden state not captured in observations
- Example: AMR updates every N steps → manager doesn't see true AMR between updates

✅ **Temporal dependencies**: Reward depends on option history
- Example: Alternating antibiotics is rewarded, but not visible in single-step observation

✅ **Noisy observations**: Observations contain Gaussian noise or bias
- Example: `add_noise_to_visible_AMR_levels > 0.0` → LSTM can filter noise over time

✅ **Long option sequences**: Manager needs to remember which options were recently used
- Example: Avoiding recently used options to promote diversity

✅ **Diagnostics show random transitions**: HRL_PPO doesn't learn coherent temporal patterns
- Example: Transition heatmap is uniform → LSTM may help

### Use HRL_PPO if:

✅ **Fully observable**: All relevant state visible in current observation

✅ **Markovian dynamics**: Optimal action depends only on current state

✅ **Training speed matters**: Need faster iteration cycles

✅ **Simple environments**: Small observation space, short episodes

---

## Step 1: Understand RPPO Architecture

### HRL_RPPO Manager Structure:

```
Observation → Feature Extraction (MLP) → LSTM → Policy Head → Option Index
                                        ↓
                                   Hidden State (h_t, c_t)
                                        ↓
                                   Passed to next step
```

**Key components:**
1. **Feature extraction**: MLP layers process raw observations (configured via `policy_kwargs.net_arch`)
2. **LSTM layers**: Maintain hidden state across option selections (configured via `lstm_kwargs`)
3. **Policy head**: Maps LSTM output to option probabilities

**Hidden state dimensions:**
- Input: Observation features (output of feature extraction MLP)
- Hidden state: `lstm_hidden_size` per LSTM layer (default: 64)
- Output: Option logits (size = number of options)

---

## Step 2: Configure HRL_RPPO

### Minimal Config Changes

**Start with HRL_PPO config, then modify:**

```bash
cd myproject

# Copy HRL PPO config as base
cp configs/umbrella_configs/hrl_ppo_default.yaml configs/umbrella_configs/hrl_rppo_test.yaml
```

**Edit `hrl_rppo_test.yaml`:**

```yaml
# Change agent algorithm to RPPO
agent_algorithm: agent_algorithm/hrl_rppo.yaml  # ← Changed from hrl_ppo.yaml

# Update run name
training:
  run_name: hrl_rppo_test
  # ... other training params unchanged ...
```

**That's it!** The `hrl_rppo.yaml` algorithm config already has sensible LSTM defaults.

### RPPO Algorithm Config Structure

View the default RPPO config:

```bash
cat configs/agent_algorithm/hrl_rppo.yaml
```

**Contents:**

```yaml
algorithm: "HRL_RPPO"

# LSTM-specific architecture parameters
lstm_kwargs:
  lstm_hidden_size: 64        # Hidden state dimension
  n_lstm_layers: 1            # Number of stacked LSTM layers
  enable_critic_lstm: true    # Whether critic also uses LSTM

# Pre-LSTM feature extraction
policy_kwargs:
  net_arch:
    - 128
    - 128

# RecurrentPPO hyperparameters (same as PPO but for recurrent policy)
recurrent_ppo:
  learning_rate: 3.0e-4
  n_steps: 256
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.02
  vf_coef: 0.5
  max_grad_norm: 0.5
  verbose: 0
```

---

## Step 3: Train HRL_RPPO

Train exactly like HRL_PPO, just with different algorithm config:

```bash
python -m abx_amr_simulator.training.train \
  --umbrella-config $(pwd)/configs/umbrella_configs/hrl_rppo_test.yaml \
  --seed 42
```

**Expected training time**: ~1.5-2x slower than HRL_PPO due to LSTM overhead.

**Terminal output:**

```
Loading option library: options/option_libraries/default_deterministic.yaml
Loaded 12 options
Manager algorithm: HRL_RPPO
Manager observation space: Box(...)
Manager action space: Discrete(12)
Training started...
Episode 5/25: mean_reward=105.3, mean_amr=0.42
...
Training complete! Model saved to results/hrl_rppo_test_20260217_153045/final_model.zip
```

---

## Step 4: LSTM Hyperparameter Tuning

### Key LSTM Parameters:

**1. `lstm_hidden_size`**: Hidden state dimension (default: 64)
- **Smaller (32)**: Faster training, less memory capacity
- **Larger (128)**: More expressive, but slower and risks overfitting
- **Rule of thumb**: Start with 64, increase if environment has complex temporal dependencies

**2. `n_lstm_layers`**: Number of stacked LSTM layers (default: 1)
- **1 layer**: Sufficient for most tasks
- **2 layers**: For very complex temporal patterns (long option sequences)
- **3+ layers**: Rarely needed, risks overfitting

**3. `enable_critic_lstm`**: Whether critic (value function) uses LSTM (default: true)
- **True**: Critic also maintains hidden state (better for POMDP)
- **False**: Critic is feedforward (faster, use if value is Markovian)

### Tuning Recipe:

**If training is too slow:**
```yaml
lstm_kwargs:
  lstm_hidden_size: 32  # Reduce from 64
  enable_critic_lstm: false  # Critic doesn't need LSTM
```

**If manager doesn't learn temporal patterns:**
```yaml
lstm_kwargs:
  lstm_hidden_size: 128  # Increase from 64
  n_lstm_layers: 2  # Add second layer
```

**If overfitting (training reward >> eval reward):**
```yaml
recurrent_ppo:
  ent_coef: 0.05  # Increase from 0.02 (more exploration)
lstm_kwargs:
  lstm_hidden_size: 32  # Reduce capacity
```

---

## Step 5: Compare HRL_PPO vs. HRL_RPPO

### Direct Comparison Workflow:

Train both algorithms with same option library and seed range:

```bash
# HRL PPO
for seed in 42 123 456; do
  python -m abx_amr_simulator.training.train \
    --umbrella-config $(pwd)/configs/umbrella_configs/hrl_ppo_default.yaml \
    --seed $seed \
    -p "training.run_name=hrl_ppo_comparison_seed${seed}"
done

# HRL RPPO
for seed in 42 123 456; do
  python -m abx_amr_simulator.training.train \
    --umbrella-config $(pwd)/configs/umbrella_configs/hrl_rppo_test.yaml \
    --seed $seed \
    -p "training.run_name=hrl_rppo_comparison_seed${seed}"
done
```

### Comparison Metrics:

**1. Convergence speed**: Which reaches higher reward faster?
- Plot TensorBoard curves side-by-side
- Expected: PPO faster early, RPPO higher asymptotic performance

**2. Sample efficiency**: Reward per training episode
- Expected: RPPO better if environment has hidden state

**3. Temporal strategy**: Option transition patterns (from diagnostics)
- Run diagnostic analysis on both:
  ```bash
  python -m abx_amr_simulator.analysis.diagnostic_analysis \
    --experiment-name hrl_ppo_comparison \
    --aggregate-by-seed
  
  python -m abx_amr_simulator.analysis.diagnostic_analysis \
    --experiment-name hrl_rppo_comparison \
    --aggregate-by-seed
  ```
- Compare transition heatmaps: RPPO should show more structured patterns

---

## Step 6: RPPO-Specific Diagnostics

### Hidden State Analysis (Advanced)

To probe what the LSTM learns, use the LSTM state logger callback:

**Add to training config:**

```yaml
# In umbrella config (advanced)
training:
  callbacks:
    - type: "LSTMStateLogger"
      save_freq: 1000  # Log hidden states every 1000 steps
```

This saves LSTM activations to `results/<run>/lstm_states/` for post-hoc analysis.

**Analysis command:**

```bash
python -m abx_amr_simulator.analysis.probe_hidden_belief \
  --experiment-name hrl_rppo_test_20260217_153045 \
  --results-dir $(pwd)/results
```

This generates plots showing:
- Hidden state trajectories over time
- Correlation between hidden state and option selection
- Whether LSTM clusters states by AMR level or option history

See separate tutorial on belief probing for details (advanced topic).

---

## Common Issues and Solutions

### Issue 1: RPPO Doesn't Outperform PPO

**Possible causes:**
1. Environment is fully observable → PPO sufficient
2. LSTM undertrained (too few episodes)
3. LSTM hidden size too small

**Solutions:**
- Verify partial observability (check `add_noise_to_visible_AMR_levels`, `update_visible_AMR_levels_every_n_timesteps`)
- Train longer (50-100 episodes instead of 25)
- Increase `lstm_hidden_size` to 128

### Issue 2: RPPO Training Unstable (Reward Oscillates)

**Possible causes:**
1. LSTM learning rate too high
2. Exploding gradients
3. Batch size too small for recurrent updates

**Solutions:**
```yaml
recurrent_ppo:
  learning_rate: 1.0e-4  # Reduce from 3.0e-4
  max_grad_norm: 0.5    # Clip gradients (already default)
  batch_size: 128       # Increase from 64
```

### Issue 3: RPPO Much Slower Than Expected

**Possible causes:**
1. Too many LSTM layers
2. Large hidden size + long episodes
3. Critic LSTM enabled unnecessarily

**Solutions:**
```yaml
lstm_kwargs:
  lstm_hidden_size: 32      # Reduce from 64
  n_lstm_layers: 1          # Use single layer
  enable_critic_lstm: false  # Disable if value is Markovian
```

### Issue 4: Hidden States Don't Capture Useful Information

**Diagnosis**: Run belief probing and find hidden states poorly correlated with option choice or AMR.

**Possible causes:**
1. Observation space already contains all relevant information
2. LSTM needs more training
3. Network architecture bottleneck

**Solutions:**
- Verify environment has partial observability (if not, use PPO)
- Increase training episodes to 100+
- Try 2-layer LSTM:
  ```yaml
  lstm_kwargs:
    n_lstm_layers: 2
  ```

---

## Recommended Hyperparameter Profiles

### Profile 1: Fast Prototyping (Small LSTM)

```yaml
lstm_kwargs:
  lstm_hidden_size: 32
  n_lstm_layers: 1
  enable_critic_lstm: false

recurrent_ppo:
  learning_rate: 3.0e-4
  batch_size: 64
  n_epochs: 10
```

**Use case**: Quick experiments, fully observable environments with mild noise.

### Profile 2: Standard (Balanced)

```yaml
lstm_kwargs:
  lstm_hidden_size: 64
  n_lstm_layers: 1
  enable_critic_lstm: true

recurrent_ppo:
  learning_rate: 3.0e-4
  batch_size: 64
  n_epochs: 10
```

**Use case**: Default starting point for partially observable environments.

### Profile 3: High Capacity (Complex Temporal Patterns)

```yaml
lstm_kwargs:
  lstm_hidden_size: 128
  n_lstm_layers: 2
  enable_critic_lstm: true

recurrent_ppo:
  learning_rate: 1.0e-4  # Lower LR for stability
  batch_size: 128        # Larger batches for recurrent updates
  n_epochs: 15
```

**Use case**: Long option sequences, severe partial observability, many options (>15).

---

## What's Next?

✅ You've learned when and how to use HRL_RPPO!

**Next tutorials**:
- **Tutorial 10**: [Experiment Set Runner](10_experiment_set_runner.md) — Run large parameter sweeps with JSON configs
- **Tutorial 11**: [Advanced Heuristic Worker Subclassing](11_advanced_heuristic_worker_subclassing.md) — Implement sophisticated heuristic options

**Advanced topics (not covered in tutorials)**:
- LSTM belief probing with `probe_hidden_belief`
- Custom LSTM architectures (subclassing RecurrentPPO)
- Multi-locale environments with separate LSTM states per locale

---

## Key Takeaways

1. **HRL_RPPO = HRL_PPO + LSTM**: Recurrent manager for partial observability and temporal reasoning
2. **When to use RPPO**: Noisy observations, AMR updates infrequent, reward depends on option history
3. **Config changes**: Just change `agent_algorithm: agent_algorithm/hrl_rppo.yaml` in umbrella config
4. **LSTM hyperparameters**: Start with 64-dim hidden state, 1 layer, critic LSTM enabled
5. **Training overhead**: ~1.5-2x slower than PPO due to LSTM
6. **Comparison workflow**: Train both PPO and RPPO with same seeds, compare convergence and temporal patterns
7. **Diagnostics**: Use option transition analysis to verify RPPO learns better temporal strategies
8. **Troubleshooting**: If RPPO doesn't help, environment may be fully observable (use PPO instead)
