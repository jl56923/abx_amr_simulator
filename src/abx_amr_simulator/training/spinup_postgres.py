"""Spin up local PostgreSQL server for Optuna parallel tuning."""

import os
from pathlib import Path

from abx_amr_simulator.training import postgres_utils


def _resolve_pg_data_dir() -> str:
    """Find writable directory for PostgreSQL data."""
    candidates = []

    pg_data_dir = os.environ.get("PG_DATA_DIR")
    if pg_data_dir:
        candidates.append(pg_data_dir)

    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        candidates.append(os.path.join(tmpdir, "optuna_pgdata"))

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        candidates.append(os.path.join(slurm_tmpdir, "optuna_pgdata"))

    user = os.environ.get("USER", "unknown")
    candidates.append(os.path.join("/scratch", user, "optuna_pgdata"))
    candidates.append(os.path.join("/scratch", "user", user, "optuna_pgdata"))
    candidates.append(os.path.join("/tmp", "optuna_pgdata"))

    for base_dir in candidates:
        if not base_dir:
            continue
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        if os.access(base_path, os.W_OK):
            return str(base_path)

    raise RuntimeError(
        "No writable directory found for PostgreSQL data. "
        "Set PG_DATA_DIR, TMPDIR, or SLURM_TMPDIR, or ensure /scratch/<user> or /tmp is writable."
    )


def main() -> None:
    pg_data_dir = _resolve_pg_data_dir()

    pg_port = os.environ.get("PG_PORT", "5432")
    pg_username = os.environ.get("PG_USERNAME", os.environ.get("USER", "postgres"))
    db_name = os.environ.get("DB_NAME", "optuna_tuning")
    expected_major_version = os.environ.get("PG_MAJOR_VERSION", "17")

    os.environ.setdefault("LANG", "C.UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")

    postgres_utils.run_postgres(
        pg_port=pg_port,
        pg_data_dir=pg_data_dir,
        pg_username=pg_username,
        expected_major_version=expected_major_version
    )
    postgres_utils.ensure_database_exists(
        pg_port=pg_port,
        pg_username=pg_username,
        db_name=db_name
    )
    postgres_utils.test_connection(
        db_name=db_name,
        pg_username=pg_username,
        pg_port=pg_port
    )

    print(f"PostgreSQL ready at localhost:{pg_port}/{db_name}")


if __name__ == "__main__":
    main()
