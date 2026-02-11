#!/bin/bash
#SBATCH --job-name=optuna_parallel
#SBATCH --output=logs/optuna_%A_%a.out
#SBATCH --error=logs/optuna_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G

set -euo pipefail

# ===== User Configuration =====
PROJECT_ROOT="${HOME}/abx_amr_capacitor_rl"
UMBRELLA_CONFIG="${PROJECT_ROOT}/workspace/experiments/configs/umbrella_configs/base_experiment.yaml"
TUNING_CONFIG="${PROJECT_ROOT}/workspace/experiments/configs/tuning/ppo_tuning_default.yaml"
RUN_NAME="optuna_ppo_parallel"
STUDY_NAME="optuna_ppo_parallel"
RESULTS_DIR="${PROJECT_ROOT}/workspace/results"

# IMPORTANT: tuning_config.optimization.n_trials is per worker.
NUM_WORKERS=10

OVERWRITE_STUDY=false

# ===== Environment Setup =====
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate rlenv

export LANG="C.UTF-8"
export LC_ALL="C.UTF-8"

export PG_PORT="5432"
export PG_USERNAME="${USER}"
export DB_NAME="optuna_tuning"

cd "${PROJECT_ROOT}"
mkdir -p logs

# ===== PostgreSQL Lifecycle =====
python -m abx_amr_simulator.training.spinup_postgres

cleanup() {
    python -m abx_amr_simulator.training.shutdown_postgres
}
trap cleanup EXIT

# ===== Study Initialization =====
if [ "${OVERWRITE_STUDY}" = true ]; then
    python -c "import optuna; optuna.delete_study(study_name='${STUDY_NAME}', storage='postgresql://${PG_USERNAME}@localhost:${PG_PORT}/${DB_NAME}')" || true
fi

python -c "import optuna; optuna.create_study(study_name='${STUDY_NAME}', storage='postgresql://${PG_USERNAME}@localhost:${PG_PORT}/${DB_NAME}', direction='maximize', load_if_exists=True)"

# ===== Launch Workers =====
if [ "${SLURM_CPUS_PER_TASK}" -lt "${NUM_WORKERS}" ]; then
    echo "Error: SLURM_CPUS_PER_TASK (${SLURM_CPUS_PER_TASK}) < NUM_WORKERS (${NUM_WORKERS})." >&2
    exit 1
fi

for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    python -m abx_amr_simulator.training.tune \
        --umbrella-config "${UMBRELLA_CONFIG}" \
        --tuning-config "${TUNING_CONFIG}" \
        --run-name "${RUN_NAME}" \
        --study-name "${STUDY_NAME}" \
        --results-dir "${RESULTS_DIR}" \
        --use-postgres \
        --param-override "training.seed=$((1000 + WORKER_ID))" &
done

wait
