#!/usr/bin/env python
import subprocess
import sys
import json
from pathlib import Path
import yaml
import shutil
from abx_amr_simulator.utils import load_config, setup_config_folders_with_defaults
from abx_amr_simulator.training import setup_optimization_folders_with_defaults

test_dir = Path('tests/integration/test_outputs/tuning_resumption_debug')
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(parents=True)

experiments_dir = test_dir / 'experiments'
experiments_dir.mkdir()
setup_config_folders_with_defaults(target_path=experiments_dir)
setup_optimization_folders_with_defaults(target_path=experiments_dir)

optimization_dir = test_dir / 'optimization'
optimization_dir.mkdir()
results_dir = test_dir / 'results'
results_dir.mkdir()

# Create configs
run_name = 'test_resume'
umbrella_path = experiments_dir / 'configs' / 'umbrella_configs' / 'base_experiment.yaml'
config = load_config(config_path=str(umbrella_path))
config['training']['total_num_training_episodes'] = 2
config['environment']['max_time_steps'] = 3
config['training']['eval_freq_every_n_episodes'] = 2
config['training']['save_freq_every_n_episodes'] = 2
config['training']['run_name'] = run_name
test_umbrella_path = experiments_dir / 'configs' / 'umbrella_configs' / f'{run_name}.yaml'
with open(test_umbrella_path, 'w') as f:
    yaml.dump(config, f)

tuning_config = {
    'optimization': {
        'n_trials': 1,
        'n_seeds_per_trial': 1,
        'truncated_episodes': 2,
        'direction': 'maximize',
        'sampler': 'Random',
        'stability_penalty_weight': 0.1
    },
    'search_space': {
        'learning_rate': {
            'type': 'float',
            'low': 1e-4,
            'high': 1e-3,
            'log': True
        }
    }
}
tuning_path = experiments_dir / 'tuning_configs' / f'{run_name}.yaml'
with open(tuning_path, 'w') as f:
    yaml.dump(tuning_config, f)

tune_cmd = [
    sys.executable, '-m', 'abx_amr_simulator.training.tune',
    '--umbrella-config', str(test_umbrella_path.absolute()),
    '--tuning-config', str(tuning_path.absolute()),
    '--run-name', run_name,
    '--optimization-dir', str(optimization_dir.absolute()),
    '--results-dir', str(results_dir.absolute())
]

print('=== FIRST RUN ===')
result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
print('Return code:', result.returncode)
print('STDOUT (last 500 chars):')
print(result.stdout[-500:] if result.stdout else '(empty)')

run_folder = optimization_dir / run_name
summary_path = run_folder / 'study_summary.json'
with open(summary_path, 'r') as f:
    summary = json.load(f)
print(f'After first run: n_trials={summary["n_trials"]}')

print('\n=== SECOND RUN ===')
result = subprocess.run(tune_cmd, capture_output=True, text=True, timeout=180)
print('Return code:', result.returncode)
print('STDOUT (full):')
print(result.stdout)
if result.stderr:
    print('\nSTDERR:')
    print(result.stderr)

with open(summary_path, 'r') as f:
    summary = json.load(f)
print(f'After second run: n_trials={summary["n_trials"]}')
print(f'Expected: 2, Got: {summary["n_trials"]}')
print(f'\nFull study_summary.json:')
print(json.dumps(summary, indent=2))
