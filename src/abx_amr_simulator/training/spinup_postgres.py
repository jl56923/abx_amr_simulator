"""Spin up local PostgreSQL server for Optuna parallel tuning."""

import argparse
import os
import sys
from pathlib import Path

from abx_amr_simulator.training import postgres_utils


def _resolve_pg_data_dir(pg_data_dir: str = None) -> str:
    """Find writable directory for PostgreSQL data.
    
    Args:
        pg_data_dir: If provided, use this directory (highest priority)
    """
    candidates = []

    # Highest priority: explicit argument
    if pg_data_dir:
        candidates.append(pg_data_dir)

    # Next: environment variable
    pg_data_dir_env = os.environ.get("PG_DATA_DIR")
    if pg_data_dir_env:
        candidates.append(pg_data_dir_env)

    # Then: standard temp directories
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        candidates.append(os.path.join(tmpdir, "optuna_pgdata"))

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        candidates.append(os.path.join(slurm_tmpdir, "optuna_pgdata"))

    # Finally: fallback locations
    user = os.environ.get("USER", "unknown")
    candidates.append(os.path.join("/scratch", user, "optuna_pgdata"))
    candidates.append(os.path.join("/scratch", "user", user, "optuna_pgdata"))
    candidates.append(os.path.join("/tmp", "optuna_pgdata"))

    for base_dir in candidates:
        if not base_dir:
            continue
        base_path = Path(base_dir)
        try:
            base_path.mkdir(parents=True, exist_ok=True)
            if os.access(base_path, os.W_OK):
                return str(base_path)
        except (PermissionError, OSError) as e:
            # Parent directory not writable or doesn't exist; try next candidate
            continue

    raise RuntimeError(
        "No writable directory found for PostgreSQL data. "
        "Set --pg-data-dir argument or PG_DATA_DIR environment variable to a writable directory, or ensure one of: "
        "TMPDIR, SLURM_TMPDIR, /scratch/<user>, or /tmp is writable."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start PostgreSQL server for Optuna parallel tuning"
    )
    parser.add_argument(
        "--pg-data-dir",
        type=str,
        default=None,
        help="PostgreSQL data directory (highest priority). Defaults to PG_DATA_DIR env var or auto-resolved."
    )
    parser.add_argument(
        "--pg-port",
        type=str,
        default=None,
        help="PostgreSQL port. Defaults to PG_PORT env var or 5432."
    )
    parser.add_argument(
        "--pg-username",
        type=str,
        default=None,
        help="PostgreSQL username. Defaults to PG_USERNAME env var or current user."
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="Database name. Defaults to DB_NAME env var or 'optuna_tuning'."
    )
    parser.add_argument(
        "--pg-major-version",
        type=str,
        default=None,
        help="Expected PostgreSQL major version. Defaults to PG_MAJOR_VERSION env var or '17'."
    )
    
    args = parser.parse_args()
    
    # Resolve PostgreSQL data directory with argument taking precedence
    pg_data_dir = _resolve_pg_data_dir(pg_data_dir=args.pg_data_dir)

    # Get settings from arguments, environment variables, or defaults
    pg_port = args.pg_port or os.environ.get("PG_PORT", "5432")
    pg_username = args.pg_username or os.environ.get("PG_USERNAME", os.environ.get("USER", "postgres"))
    db_name = args.db_name or os.environ.get("DB_NAME", "optuna_tuning")
    expected_major_version = args.pg_major_version or os.environ.get("PG_MAJOR_VERSION", "17")

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
    sys.exit(0)


if __name__ == "__main__":
    main()
