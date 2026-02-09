"""
Script to re-save trained models with numpy compatibility for cross-environment loading.

Run this on the cluster BEFORE downgrading torch to convert existing models.
This ensures models save in a format compatible with both environments.

Usage:
    python cluster_convert_models.py
    
This script will:
1. Find all best_model.zip files in results/
2. Load each model 
3. Re-save to ensure compatibility
"""

import os
import sys
from pathlib import Path
import glob
from stable_baselines3 import PPO, DQN, A2C

def convert_models_in_results():
    """Find and re-save all best_model.zip files."""
    results_dir = Path("results")
    if not results_dir.exists():
        print(f"Error: {results_dir} directory not found")
        return
    
    # Find all best_model.zip files
    model_files = list(results_dir.glob("*/checkpoints/best_model.zip"))
    
    if not model_files:
        print("No best_model.zip files found in results/")
        return
    
    print(f"Found {len(model_files)} models to convert")
    
    for i, model_path in enumerate(sorted(model_files), 1):
        run_dir = model_path.parent.parent
        print(f"\n[{i}/{len(model_files)}] Converting: {run_dir.name}")
        
        # Load config to determine algorithm
        config_path = run_dir / "full_agent_env_config.yaml"
        if not config_path.exists():
            print(f"  WARNING: full_agent_env_config.yaml not found, skipping")
            continue
        
        # Read algorithm from config
        import yaml
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            algorithm = config.get('algorithm', 'PPO')
        except Exception as e:
            print(f"  WARNING: Could not read config - {e}, skipping")
            continue
        
        algorithm_map = {'PPO': PPO, 'DQN': DQN, 'A2C': A2C}
        AgentClass = algorithm_map.get(algorithm, PPO)
        
        try:
            # Load model
            print(f"  Loading {algorithm} agent...")
            agent = AgentClass.load(str(model_path))
            
            # Re-save to same location (overwrites with current environment's format)
            print(f"  Re-saving...")
            agent.save(str(model_path).replace('.zip', ''))
            
            print(f"  ✓ Successfully converted")
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == '__main__':
    convert_models_in_results()
    print("\n" + "="*70)
    print("Model conversion complete!")
    print("="*70)
