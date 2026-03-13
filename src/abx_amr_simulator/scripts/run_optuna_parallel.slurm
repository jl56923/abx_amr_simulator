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

# ===== User Configuration (override via environment variables) =====
# Example sbatch usage:
#   sbatch --export=ALL,PROJECT_ROOT=$PWD,UMBRELLA_CONFIG=$PWD/path/to/umbrella.yaml,TUNING_CONFIG=$PWD/path/to/tuning.yaml,RESULTS_DIR=$PWD/results run_optuna_parallel.slurm
PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
UMBRELLA_CONFIG="${UMBRELLA_CONFIG:-}"
TUNING_CONFIG="${TUNING_CONFIG:-}"
RUN_NAME="${RUN_NAME:-optuna_ppo_parallel}"
STUDY_NAME="${STUDY_NAME:-optuna_ppo_parallel}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rlenv}"

# IMPORTANT: tuning_config.optimization.n_trials is per worker.
NUM_WORKERS="${NUM_WORKERS:-10}"

OVERWRITE_STUDY="${OVERWRITE_STUDY:-false}"

if [ -z "${UMBRELLA_CONFIG}" ] || [ -z "${TUNING_CONFIG}" ]; then
    echo "Error: UMBRELLA_CONFIG and TUNING_CONFIG must be set (e.g. via sbatch --export)." >&2
    exit 1
fi

if [ ! -f "${UMBRELLA_CONFIG}" ]; then
    echo "Error: UMBRELLA_CONFIG file not found: ${UMBRELLA_CONFIG}" >&2
    exit 1
fi

if [ ! -f "${TUNING_CONFIG}" ]; then
    echo "Error: TUNING_CONFIG file not found: ${TUNING_CONFIG}" >&2
    exit 1
fi

# ===== Environment Setup =====
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export LANG="C.UTF-8"
export LC_ALL="C.UTF-8"

export PG_PORT="5432"
export PG_USERNAME="${USER}"
export DB_NAME="${DB_NAME:-optuna_tuning}"

cd "${PROJECT_ROOT}"
mkdir -p logs
mkdir -p "${RESULTS_DIR}"

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
