#!/bin/bash
set -euo pipefail

# ===== User Configuration =====
PROJECT_ROOT="${HOME}/abx_amr_capacitor_rl"
UMBRELLA_CONFIG="${PROJECT_ROOT}/workspace/experiments/configs/umbrella_configs/base_experiment.yaml"
TUNING_CONFIG="${PROJECT_ROOT}/workspace/experiments/configs/tuning/ppo_tuning_default.yaml"
RUN_NAME="optuna_ppo_parallel"
STUDY_NAME="optuna_ppo_parallel"
RESULTS_DIR="${PROJECT_ROOT}/workspace/results"

# IMPORTANT: tuning_config.optimization.n_trials is per worker.
NUM_WORKERS=4

OVERWRITE_STUDY=false
USE_TASK_SPOOLER=true

# ===== Environment Setup =====
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate rlenv

export LANG="C.UTF-8"
export LC_ALL="C.UTF-8"

export PG_PORT="5432"
export PG_USERNAME="${USER}"
export DB_NAME="optuna_tuning"

cd "${PROJECT_ROOT}"

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
