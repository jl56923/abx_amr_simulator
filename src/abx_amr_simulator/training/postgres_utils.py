"""PostgreSQL lifecycle helpers for Optuna parallel tuning."""

import os
import subprocess
import sys
from typing import Optional

try:
    import psycopg2
except ImportError as exc:
    raise ImportError(
        "psycopg2-binary is required for PostgreSQL support. "
        "Install it with 'pip install psycopg2-binary'."
    ) from exc


def is_server_ready(pg_port: str, pg_username: str) -> bool:
    """Check if PostgreSQL server is running and accepting connections."""
    result = subprocess.run(
        args=["pg_isready", "-h", "localhost", "-p", pg_port, "-U", pg_username],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True
    )
    return result.returncode == 0


def _fetch_pg_version(pg_port: str, pg_username: str) -> Optional[str]:
    """Fetch PostgreSQL version string from running server."""
    try:
        connection = psycopg2.connect(
            dbname="postgres",
            user=pg_username,
            host="localhost",
            port=pg_port
        )
        cursor = connection.cursor()
        cursor.execute(query="SELECT version();")
        version_str = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return version_str
    except Exception:
        return None


def is_version_compatible(
    pg_port: str,
    pg_username: str,
    expected_major_version: str
) -> bool:
    """Validate PostgreSQL version matches expected major version."""
    version_str = _fetch_pg_version(
        pg_port=pg_port,
        pg_username=pg_username
    )
    if not version_str:
        return False
    return version_str.startswith(f"PostgreSQL {expected_major_version}")


def run_postgres(
    pg_port: str,
    pg_data_dir: str,
    pg_username: str,
    expected_major_version: str
) -> None:
    """Start PostgreSQL server or verify existing server is compatible."""
    if is_server_ready(pg_port=pg_port, pg_username=pg_username):
        if is_version_compatible(
            pg_port=pg_port,
            pg_username=pg_username,
            expected_major_version=expected_major_version
        ):
            print(f"PostgreSQL already running on port {pg_port}.")
            return
        raise RuntimeError(
            "Incompatible PostgreSQL version running on port "
            f"{pg_port}. Expected {expected_major_version}.x"
        )

    pg_version_file = os.path.join(pg_data_dir, "PG_VERSION")
    if not os.path.exists(pg_version_file):
        print(f"Initializing PostgreSQL data directory at {pg_data_dir}...")
        subprocess.run(args=["initdb", "-D", pg_data_dir], check=True)

    print(f"Starting PostgreSQL server on port {pg_port} using {pg_data_dir}...")
    subprocess.run(
        args=[
            "pg_ctl",
            "-D",
            pg_data_dir,
            "-o",
            f"-p {pg_port}",
            "start",
            "-w"
        ],
        check=True
    )


def ensure_database_exists(pg_port: str, pg_username: str, db_name: str) -> None:
    """Create database if it doesn't exist."""
    connection = psycopg2.connect(
        dbname="postgres",
        user=pg_username,
        host="localhost",
        port=pg_port
    )
    connection.autocommit = True
    cursor = connection.cursor()
    cursor.execute(
        query="SELECT 1 FROM pg_database WHERE datname = %s",
        vars=(db_name,)
    )
    if not cursor.fetchone():
        print(f"Creating database '{db_name}'...")
        cursor.execute(query=f"CREATE DATABASE {db_name}")
    else:
        print(f"Database '{db_name}' already exists.")
    cursor.close()
    connection.close()


def test_connection(db_name: str, pg_username: str, pg_port: str) -> None:
    """Test connection to database."""
    print(f"Testing connection to '{db_name}'...")
    connection = psycopg2.connect(
        dbname=db_name,
        user=pg_username,
        host="localhost",
        port=pg_port
    )
    cursor = connection.cursor()
    cursor.execute(query="SELECT version();")
    print("PostgreSQL version:", cursor.fetchone())
    cursor.close()
    connection.close()
    print("Connection successful.")


def shutdown_postgres(pg_data_dir: str) -> None:
    """Stop PostgreSQL server gracefully."""
    if not os.path.exists(pg_data_dir):
        print(f"Error: PostgreSQL data directory '{pg_data_dir}' does not exist.")
        return

    print(f"Shutting down PostgreSQL server running at {pg_data_dir}")
    try:
        subprocess.run(
            args=["pg_ctl", "-D", pg_data_dir, "stop", "-m", "fast"],
            check=True
        )
        print("PostgreSQL server stopped successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"Error stopping PostgreSQL server: {exc}")
        sys.exit(1)
