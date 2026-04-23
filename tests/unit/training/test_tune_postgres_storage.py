"""Unit tests for Optuna storage URL selection."""

import os

from abx_amr_simulator.training.tune import build_storage_url


def test_build_storage_url_postgres_env() -> None:
    env_backup = {
        "PG_USERNAME": os.environ.get("PG_USERNAME"),
        "PG_PORT": os.environ.get("PG_PORT"),
        "DB_NAME": os.environ.get("DB_NAME"),
    }

    os.environ["PG_USERNAME"] = "tester"
    os.environ["PG_PORT"] = "6543"
    os.environ["DB_NAME"] = "optuna_test"

    try:
        storage_url = build_storage_url(
            use_postgres=True,
            run_name="unused",
            optimization_dir="/tmp"
        )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert storage_url == "postgresql://tester@localhost:6543/optuna_test"


def test_build_storage_url_sqlite(tmp_path) -> None:
    optimization_dir = tmp_path / "optuna_run"
    storage_url = build_storage_url(
        use_postgres=False,
        run_name="unused",
        optimization_dir=str(optimization_dir)
    )

    assert storage_url == f"sqlite:///{optimization_dir}/optuna_study.db"
