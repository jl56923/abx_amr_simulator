#!/usr/bin/env python3
"""Test that training is reproducible across runs with same seed.

Runs training twice with identical configuration and verifies that:
1. Mean rewards at each evaluation are identical
2. Determinism holds across full training runs
"""

import sys
import os
from pathlib import Path
import yaml
import numpy as np

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from abx_amr_simulator.utils.factories import (
    create_reward_calculator,
    create_patient_generator,
    create_environment,
    create_agent,
    setup_callbacks,
)
from abx_amr_simulator.utils import create_run_directory


def run_training(run_name_suffix: str) -> dict:
    """Run a single training instance and return eval metrics."""
    # Load config
    completed_exp_config_path = Path("workspace/results/exp_1.a_single_abx_lambda0.0_seed1_20260116_001143/config.yaml")
    
    with open(completed_exp_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override the antibiotic dict in the environment to give it an initial amr level:
    config['environment']['antibiotics_AMR_dict']['Antibiotic_A']['initial_amr_level'] = 0.0
    
    # Modify run name
    config["run_name"] = f"reproducibility_test_{run_name_suffix}"
    config["training"]["run_name"] = config["run_name"]
    
    # Shorter training for testing
    config["training"]["total_num_training_episodes"] = 10
    config["training"]["eval_freq_every_n_episodes"] = 2
    
    # Create run directory
    run_dir = create_run_directory(
        project_root="results",
        config=config,
    )
    
    print(f"\nRun {run_name_suffix}: {run_dir}")
    
    # Create components
    rc = create_reward_calculator(config=config)
    pg = create_patient_generator(config=config)
    env = create_environment(config=config, reward_calculator=rc, patient_generator=pg, wrap_monitor=True)
    
    agent = create_agent(config=config, env=env, tb_log_path=os.path.join(run_dir, "logs"), verbose=0)
    
    # Setup callbacks
    callbacks = setup_callbacks(config=config, run_dir=str(run_dir), eval_env=env)
    
    # Calculate total timesteps
    max_time_steps = config["environment"]["max_time_steps"]
    total_episodes = config["training"]["total_num_training_episodes"]
    total_timesteps = total_episodes * max_time_steps
    
    # Train
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )
    
    # Extract eval metrics from the environment's eval log
    # The detailed eval callback saves rewards to info
    eval_metrics = []
    
    # Check if we can read from eval logs (handle nested eval_logs directory)
    eval_logs_dir = os.path.join(run_dir, 'eval_logs')
    eval_logs_nested = os.path.join(eval_logs_dir, 'eval_logs')
    
    # Try nested directory first, then fall back to direct
    if os.path.exists(eval_logs_nested):
        eval_logs_dir = eval_logs_nested
    
    if os.path.exists(eval_logs_dir):
        eval_files = sorted([f for f in os.listdir(eval_logs_dir) if f.startswith('eval_') and f.endswith('.npz')])
        for eval_file in eval_files:
            data = np.load(os.path.join(eval_logs_dir, eval_file))
            if 'episode_rewards' in data:
                rewards = data['episode_rewards']
                mean_reward = np.mean(rewards)
                eval_metrics.append(mean_reward)
                print(f"  Eval: mean_reward = {mean_reward:.4f}")
    
    env.close()
    return {"run_dir": run_dir, "eval_metrics": eval_metrics}


if __name__ == "__main__":
    print("Testing training reproducibility with identical seeds...")
    
    # Run training twice
    run1 = run_training("run1")
    run2 = run_training("run2")
    
    # Compare results
    print("\n" + "="*70)
    print("Comparison:")
    print("="*70)
    
    metrics1 = run1["eval_metrics"]
    metrics2 = run2["eval_metrics"]
    
    print(f"Run 1 had {len(metrics1)} evaluations")
    print(f"Run 2 had {len(metrics2)} evaluations")
    
    if len(metrics1) != len(metrics2):
        print("ERROR: Different number of evaluations!")
    else:
        all_match = True
        for i, (m1, m2) in enumerate(zip(metrics1, metrics2)):
            match = np.isclose(m1, m2, atol=1e-6)
            status = "✓" if match else "✗"
            print(f"Eval {i}: Run1={m1:.4f}, Run2={m2:.4f} {status}")
            if not match:
                all_match = False
        
        if all_match:
            print("\n✓ All evaluations match! Training is reproducible.")
        else:
            print("\n✗ Evaluations differ. Seeding may not be synchronized properly.")
