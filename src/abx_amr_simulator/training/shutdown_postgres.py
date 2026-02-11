"""Shutdown local PostgreSQL server used for Optuna tuning."""

import os

from abx_amr_simulator.training import postgres_utils


def _resolve_pg_data_dir() -> str:
    """Resolve PostgreSQL data directory used by spinup_postgres."""
    pg_data_dir = os.environ.get("PG_DATA_DIR")
    if pg_data_dir:
        return pg_data_dir

    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        return os.path.join(tmpdir, "optuna_pgdata")

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        return os.path.join(slurm_tmpdir, "optuna_pgdata")

    return os.path.join("/tmp", "optuna_pgdata")


def main() -> None:
    pg_data_dir = _resolve_pg_data_dir()
    postgres_utils.shutdown_postgres(pg_data_dir=pg_data_dir)


if __name__ == "__main__":
    main()
