"""Integration tests for postgres_utils.py

These tests verify actual PostgreSQL startup, database creation, and connection behavior.
They use real PostgreSQL instances with temporary data directories.

Tests require PostgreSQL to be installed (initdb, pg_ctl, pg_isready commands available).
"""

import os
import sys
import pytest
import tempfile
import shutil
import subprocess
from pathlib import Path

from abx_amr_simulator.training import postgres_utils


@pytest.fixture
def temp_pg_data_dir():
    """Create a temporary PostgreSQL data directory that's cleaned up after the test."""
    temp_dir = tempfile.mkdtemp(prefix="postgres_test_")
    yield temp_dir
    
    # Cleanup: stop PostgreSQL and remove data directory
    try:
        # Try to stop any running server in this data directory
        result = subprocess.run(
            args=["pg_ctl", "-D", temp_dir, "stop", "-m", "immediate"],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"Warning: pg_ctl stop returned {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"Warning: pg_ctl stop timed out")
    except Exception as e:
        print(f"Warning: Error stopping PostgreSQL: {e}")
    
    # Remove the directory
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Error removing temp dir {temp_dir}: {e}")


@pytest.fixture
def unique_pg_port():
    """Generate a unique PostgreSQL port number for test isolation.
    
    Uses a simple approach: ports 15432-15500 are reserved for tests.
    In a real multi-test scenario, you'd want a more robust mechanism
    (e.g., finding an available port using socket binding).
    """
    import random
    return str(15400 + random.randint(0, 99))


@pytest.fixture
def pg_username():
    """Get the current user for PostgreSQL authentication."""
    return os.environ.get("USER", "postgres")


class TestPostgresStartup:
    """Test PostgreSQL initialization and startup."""
    
    def test_run_postgres_initializes_and_starts(self, temp_pg_data_dir, unique_pg_port, pg_username):
        """Test that run_postgres initializes PostgreSQL and starts the server."""
        # Verify data directory is empty before test
        assert len(os.listdir(temp_pg_data_dir)) == 0, "Temp directory should be empty"
        
        # Call run_postgres
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"  # Adjust if your PostgreSQL version differs
        )
        
        # Verify PG_VERSION file was created (indicates successful initdb)
        pg_version_file = os.path.join(temp_pg_data_dir, "PG_VERSION")
        assert os.path.exists(pg_version_file), "PG_VERSION should be created by initdb"
        
        # Verify server is running and accepting connections
        assert postgres_utils.is_server_ready(
            pg_port=unique_pg_port,
            pg_username=pg_username
        ), "Server should be accepting connections"
    
    def test_run_postgres_idempotent(self, temp_pg_data_dir, unique_pg_port, pg_username):
        """Test that run_postgres is idempotent (calling twice doesn't break anything)."""
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        # Call again - should recognize server is already running
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        # Server should still be running
        assert postgres_utils.is_server_ready(
            pg_port=unique_pg_port,
            pg_username=pg_username
        )


class TestDatabaseCreation:
    """Test database creation and existence checks."""
    
    def test_ensure_database_exists_creates_database(self, temp_pg_data_dir, unique_pg_port, pg_username):
        """Test that ensure_database_exists creates a new database."""
        # Start PostgreSQL first
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        db_name = "test_optuna_db"
        
        # Create database
        postgres_utils.ensure_database_exists(
            pg_port=unique_pg_port,
            pg_username=pg_username,
            db_name=db_name
        )
        
        # Verify database exists by connecting to it
        import psycopg2
        connection = psycopg2.connect(
            dbname=db_name,
            user=pg_username,
            host="localhost",
            port=unique_pg_port
        )
        connection.close()
    
    def test_ensure_database_exists_idempotent(self, temp_pg_data_dir, unique_pg_port, pg_username):
        """Test that ensure_database_exists is idempotent (calling twice is safe)."""
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        db_name = "test_idempotent_db"
        
        # Create database
        postgres_utils.ensure_database_exists(
            pg_port=unique_pg_port,
            pg_username=pg_username,
            db_name=db_name
        )
        
        # Call again - should recognize database exists and not error
        postgres_utils.ensure_database_exists(
            pg_port=unique_pg_port,
            pg_username=pg_username,
            db_name=db_name
        )
        
        # Database should still exist
        import psycopg2
        connection = psycopg2.connect(
            dbname=db_name,
            user=pg_username,
            host="localhost",
            port=unique_pg_port
        )
        connection.close()


class TestConnection:
    """Test connection verification."""
    
    def test_test_connection_succeeds(self, temp_pg_data_dir, unique_pg_port, pg_username):
        """Test that test_connection succeeds for valid database."""
        # Start PostgreSQL and create database
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        db_name = "test_connection_db"
        postgres_utils.ensure_database_exists(
            pg_port=unique_pg_port,
            pg_username=pg_username,
            db_name=db_name
        )
        
        # Test connection should succeed
        postgres_utils.test_connection(
            db_name=db_name,
            pg_username=pg_username,
            pg_port=unique_pg_port
        )
    
    def test_test_connection_fails_for_nonexistent_db(self, temp_pg_data_dir, unique_pg_port, pg_username):
        """Test that test_connection fails for nonexistent database."""
        # Start PostgreSQL but don't create the database
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        # Test connection to nonexistent database should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to connect to database"):
            postgres_utils.test_connection(
                db_name="nonexistent_db",
                pg_username=pg_username,
                pg_port=unique_pg_port
            )


class TestIsServerReady:
    """Test server readiness check."""
    
    def test_is_server_ready_true_when_running(self, temp_pg_data_dir, unique_pg_port, pg_username):
        """Test that is_server_ready returns True when server is running."""
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        assert postgres_utils.is_server_ready(
            pg_port=unique_pg_port,
            pg_username=pg_username
        )
    
    def test_is_server_ready_false_when_not_running(self, unique_pg_port, pg_username):
        """Test that is_server_ready returns False when server is not running."""
        # Don't start server - use a high port that's unlikely to have anything running
        assert not postgres_utils.is_server_ready(
            pg_port=unique_pg_port,
            pg_username=pg_username
        )


class TestIntegrationFullWorkflow:
    """Integration tests covering full workflows."""
    
    def test_full_startup_create_connect_workflow(self, temp_pg_data_dir, unique_pg_port, pg_username):
        """Test complete workflow: start PostgreSQL → create database → connect."""
        db_name = "integration_test_db"
        
        # Phase 1: Start PostgreSQL
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        # Phase 2: Create database
        postgres_utils.ensure_database_exists(
            pg_port=unique_pg_port,
            pg_username=pg_username,
            db_name=db_name
        )
        
        # Phase 3: Test connection
        postgres_utils.test_connection(
            db_name=db_name,
            pg_username=pg_username,
            pg_port=unique_pg_port
        )
        
        # Phase 4: Verify we can actually query the database
        import psycopg2
        connection = psycopg2.connect(
            dbname=db_name,
            user=pg_username,
            host="localhost",
            port=unique_pg_port
        )
        cursor = connection.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        assert result == (1,), "Simple query should return (1,)"
        cursor.close()
        connection.close()


# Markers for conditional test execution
pytestmark = pytest.mark.skipif(
    shutil.which("initdb") is None or shutil.which("pg_ctl") is None,
    reason="PostgreSQL command-line tools (initdb, pg_ctl) not found in PATH"
)
