#!/usr/bin/env python3
"""Probe LSTM hidden states to predict true AMR levels.

Loads logged LSTM hidden states and true AMR levels from LSTMStateLogger,
fits a linear probe to predict AMR from hidden state, and reports metrics.

Usage:
    python workspace/scripts/probe_hidden_belief.py results/my_recurrent_run/lstm_logs

This validates whether RecurrentPPO learns a meaningful internal belief about
latent AMR dynamics when AMR observations are delayed/noisy.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def load_episodes(log_dir: Path):
    """Load all logged episodes from directory.
    
    Args:
        log_dir: Path to directory containing episode_*.npz files
        
    Returns:
        Tuple of (hidden_states, true_amr_values, timesteps) as concatenated arrays
    """
    episode_files = sorted(log_dir.glob("episode_*.npz"))
    
    if len(episode_files) == 0:
        raise ValueError(f"No episode files found in {log_dir}")
    
    all_hidden = []
    all_amr = []
    all_timesteps = []
    
    for episode_file in episode_files:
        data = np.load(episode_file)
        
        if 'hidden_states' not in data or 'true_amr' not in data:
            print(f"Warning: Skipping {episode_file.name} (missing required data)")
            continue
        
        hidden_states = data['hidden_states']
        true_amr = data['true_amr']
        
        # Handle shape mismatches (if AMR is single value vs array)
        if len(hidden_states) != len(true_amr):
            min_len = min(len(hidden_states), len(true_amr))
            hidden_states = hidden_states[:min_len]
            true_amr = true_amr[:min_len]
        
        all_hidden.append(hidden_states)
        all_amr.append(true_amr)
        
        if 'timesteps' in data:
            all_timesteps.append(data['timesteps'][:len(hidden_states)])
    
    if len(all_hidden) == 0:
        raise ValueError(f"No valid episodes found in {log_dir}")
    
    hidden_states = np.concatenate(all_hidden, axis=0)
    true_amr = np.concatenate(all_amr, axis=0)
    timesteps = np.concatenate(all_timesteps, axis=0) if all_timesteps else None
    
    print(f"Loaded {len(episode_files)} episodes:")
    print(f"  Hidden states shape: {hidden_states.shape}")
    print(f"  True AMR shape: {true_amr.shape}")
    
    return hidden_states, true_amr, timesteps


def fit_probe(hidden_states, true_amr, test_size=0.2, random_state=42):
    """Fit linear probe from hidden states to AMR.
    
    Args:
        hidden_states: (N, ..., hidden_dim) array of LSTM hidden states (will squeeze middle dims)
        true_amr: (N, num_antibiotics) or (N,) array of true AMR levels
        test_size: Fraction of data to use for test set
        random_state: Random seed for train/test split
        
    Returns:
        Dictionary with probe model, train/test metrics, and predictions
    """
    # Squeeze hidden states to 2D: (N, hidden_dim)
    if hidden_states.ndim > 2:
        # Shape is (N, batch_size, hidden_dim) or similar - squeeze batch dimension
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    
    # Ensure true_amr is 2D
    if true_amr.ndim == 1:
        true_amr = true_amr.reshape(-1, 1)
    
    num_antibiotics = true_amr.shape[1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states, true_amr, test_size=test_size, random_state=random_state
    )
    
    # Fit separate probe for each antibiotic
    probes = []
    results = []
    
    for abx_idx in range(num_antibiotics):
        probe = LinearRegression()
        probe.fit(X_train, y_train[:, abx_idx])
        
        # Predict on train and test
        train_pred = probe.predict(X_train)
        test_pred = probe.predict(X_test)
        
        # Compute metrics
        train_r2 = r2_score(y_train[:, abx_idx], train_pred)
        test_r2 = r2_score(y_test[:, abx_idx], test_pred)
        train_mae = mean_absolute_error(y_train[:, abx_idx], train_pred)
        test_mae = mean_absolute_error(y_test[:, abx_idx], test_pred)
        
        probes.append(probe)
        results.append({
            'antibiotic_idx': abx_idx,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_pred': train_pred,
            'test_pred': test_pred,
        })
        
        print(f"\nAntibiotic {abx_idx}:")
        print(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
        print(f"  Test R²:  {test_r2:.4f}, MAE: {test_mae:.4f}")
    
    return {
        'probes': probes,
        'results': results,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }


def plot_predictions(probe_results, output_dir: Path, max_antibiotics: int = 3):
    """Generate scatter plots of true vs predicted AMR.
    
    Args:
        probe_results: Dictionary from fit_probe()
        output_dir: Directory to save plots
        max_antibiotics: Maximum number of antibiotics to plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = probe_results['results']
    y_test = probe_results['y_test']
    
    num_plots = min(len(results), max_antibiotics)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    for idx, (ax, result) in enumerate(zip(axes, results[:num_plots])):
        abx_idx = result['antibiotic_idx']
        test_pred = result['test_pred']
        test_true = y_test[:, abx_idx]
        test_r2 = result['test_r2']
        test_mae = result['test_mae']
        
        # Scatter plot
        ax.scatter(test_true, test_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        lims = [min(test_true.min(), test_pred.min()), 
                max(test_true.max(), test_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
        
        ax.set_xlabel('True AMR Level')
        ax.set_ylabel('Predicted AMR Level')
        ax.set_title(f'Antibiotic {abx_idx}\nR² = {test_r2:.3f}, MAE = {test_mae:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'amr_predictions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved prediction plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Probe LSTM hidden states to predict AMR')
    parser.add_argument('log_dir', type=str, help='Directory containing episode_*.npz files from LSTMStateLogger')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save plots (default: log_dir)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for train/test split')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir) if args.output_dir else log_dir
    
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)
    
    print(f"Loading LSTM hidden states from {log_dir}...")
    hidden_states, true_amr, timesteps = load_episodes(log_dir)
    
    print(f"\nFitting linear probe (hidden state → AMR)...")
    probe_results = fit_probe(
        hidden_states=hidden_states,
        true_amr=true_amr,
        test_size=args.test_size,
        random_state=args.random_seed,
    )
    
    print(f"\nGenerating prediction plots...")
    plot_predictions(probe_results, output_dir=output_dir)
    
    print("\n" + "="*70)
    print("Probe analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
