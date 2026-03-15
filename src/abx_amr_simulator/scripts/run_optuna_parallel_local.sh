#!/bin/bash
set -euo pipefail

# ===== User Configuration (override via environment variables) =====
# Example usage:
#   PROJECT_ROOT="$PWD" \
#   UMBRELLA_CONFIG="$PWD/path/to/umbrella.yaml" \
#   TUNING_CONFIG="$PWD/path/to/tuning.yaml" \
#   RESULTS_DIR="$PWD/results" \
#   bash run_optuna_parallel_local.sh
PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
UMBRELLA_CONFIG="${UMBRELLA_CONFIG:-}"
TUNING_CONFIG="${TUNING_CONFIG:-}"
RUN_NAME="${RUN_NAME:-optuna_ppo_parallel}"
STUDY_NAME="${STUDY_NAME:-optuna_ppo_parallel}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rlenv}"

# IMPORTANT: tuning_config.optimization.n_trials is per worker.
NUM_WORKERS="${NUM_WORKERS:-4}"

OVERWRITE_STUDY="${OVERWRITE_STUDY:-false}"
USE_TASK_SPOOLER="${USE_TASK_SPOOLER:-true}"

if [ -z "${UMBRELLA_CONFIG}" ] || [ -z "${TUNING_CONFIG}" ]; then
    echo "Error: UMBRELLA_CONFIG and TUNING_CONFIG must be set." >&2
    echo "Example:" >&2
    echo "  PROJECT_ROOT=\"$PWD\" UMBRELLA_CONFIG=\"$PWD/path/to/umbrella.yaml\" TUNING_CONFIG=\"$PWD/path/to/tuning.yaml\" bash run_optuna_parallel_local.sh" >&2
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
if [ "${USE_TASK_SPOOLER}" = true ] && command -v tsp >/dev/null 2>&1; then
    for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
        tsp python -m abx_amr_simulator.training.tune \
            --umbrella-config "${UMBRELLA_CONFIG}" \
            --tuning-config "${TUNING_CONFIG}" \
            --run-name "${RUN_NAME}" \
            --study-name "${STUDY_NAME}" \
            --results-dir "${RESULTS_DIR}" \
            --use-postgres \
            --param-override "training.seed=$((1000 + WORKER_ID))"
    done
    tsp
else
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
fi
