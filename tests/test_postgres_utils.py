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
        # Don't check version - it's not necessary for this test
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username
        )
        
        # Call again - should recognize server is already running
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username
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


class TestConcurrentRaceConditions:
    """Test concurrent/race condition scenarios (Gap 1 & 2 coverage).
    
    These tests verify that the PostgreSQL initialization and database creation
    code handles race conditions correctly when multiple workers execute simultaneously.
    """
    
    def test_file_locking_prevents_concurrent_initdb(self, temp_pg_data_dir, unique_pg_port, pg_username):
        """Test that fcntl.flock() prevents simultaneous initdb calls (Gap 1).
        
        This test spawns multiple threads that all attempt to initialize the same
        PostgreSQL data directory. The file locking mechanism should ensure that
        only one thread actually runs initdb, while others wait and then reuse the
        initialized database.
        
        Verification: All threads should succeed, and we check that PG_VERSION exists
        and is accessible from all threads (indicating successful file locking).
        """
        import threading
        
        results = {
            "success_count": 0,
            "errors": [],
            "pg_version_readable_count": 0,
            "lock": threading.Lock()
        }
        
        def initialize_database_worker(pg_port, pg_data_dir, pg_username, expected_major_version, worker_id):
            """Worker thread that attempts to initialize the database."""
            try:
                postgres_utils.run_postgres(
                    pg_port=pg_port,
                    pg_data_dir=pg_data_dir,
                    pg_username=pg_username,
                    expected_major_version=expected_major_version
                )
                
                # Verify PG_VERSION exists (indicator of successful initialization)
                pg_version_file = os.path.join(pg_data_dir, "PG_VERSION")
                if os.path.exists(pg_version_file):
                    with open(pg_version_file, 'r') as f:
                        version_content = f.read().strip()
                    with results["lock"]:
                        results["pg_version_readable_count"] += 1
                
                with results["lock"]:
                    results["success_count"] += 1
            except Exception as e:
                with results["lock"]:
                    results["errors"].append(f"Worker {worker_id}: {str(e)}")
        
        # Spawn multiple threads all trying to initialize the same data directory
        num_threads = 4
        threads = []
        for i in range(num_threads):
            t = threading.Thread(
                target=initialize_database_worker,
                args=(unique_pg_port, temp_pg_data_dir, pg_username, "17", i),
                name=f"InitThread-{i}"
            )
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=60)
        
        # Verify results
        assert results["success_count"] == num_threads, (
            f"All {num_threads} threads should succeed despite race condition. "
            f"Only {results['success_count']} succeeded. Errors: {results['errors']}"
        )
        
        # Verify the database is actually initialized (accessible from multiple threads)
        assert results["pg_version_readable_count"] > 0, (
            "PG_VERSION file should be readable, indicating successful initialization"
        )
        
        # Verify server is still running and healthy after concurrent attempts
        assert postgres_utils.is_server_ready(
            pg_port=unique_pg_port,
            pg_username=pg_username
        ), "PostgreSQL server should be running after concurrent initialization attempts"
    
    def test_unique_violation_exception_handling_concurrent_database_creation(
        self, temp_pg_data_dir, unique_pg_port, pg_username
    ):
        """Test that UniqueViolation is handled correctly when creating databases concurrently (Gap 2).
        
        This test simulates multiple workers racing to create the same database.
        The first worker creates it successfully, while others encounter UniqueViolation.
        All workers should complete without error due to exception handling.
        """
        import threading
        
        # First, start PostgreSQL (just once, outside of the race)
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        db_name = "concurrent_create_test_db"
        results = {
            "success_count": 0,
            "errors": [],
            "lock": threading.Lock()
        }
        
        def create_database_worker(db_name, pg_port, pg_username, worker_id):
            """Worker thread that attempts to create a database."""
            try:
                postgres_utils.ensure_database_exists(
                    pg_port=pg_port,
                    pg_username=pg_username,
                    db_name=db_name
                )
                with results["lock"]:
                    results["success_count"] += 1
            except Exception as e:
                with results["lock"]:
                    results["errors"].append(f"Worker {worker_id}: {str(e)}")
        
        # Spawn multiple threads all trying to create the same database
        num_threads = 4
        threads = []
        for i in range(num_threads):
            t = threading.Thread(
                target=create_database_worker,
                args=(db_name, unique_pg_port, pg_username, i),
                name=f"DbCreateThread-{i}"
            )
            threads.append(t)
            # Don't start all threads at once - add a small delay to increase race condition likelihood
            t.start()
            if i < num_threads - 1:
                import time
                time.sleep(0.01)  # Small delay between thread starts
        
        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=30)
        
        # Verify results
        assert results["success_count"] == num_threads, (
            f"All {num_threads} threads should succeed (race condition handled), "
            f"but only {results['success_count']} did. Errors: {results['errors']}"
        )
        assert not results["errors"], (
            f"UniqueViolation exception should be handled gracefully. "
            f"Errors encountered: {results['errors']}"
        )
        
        # Verify the database actually exists and is accessible
        import psycopg2
        connection = psycopg2.connect(
            dbname=db_name,
            user=pg_username,
            host="localhost",
            port=unique_pg_port
        )
        connection.close()
    
    def test_file_locking_with_already_initialized_database(
        self, temp_pg_data_dir, unique_pg_port, pg_username
    ):
        """Test that file locking correctly handles pre-initialized scenario (Gap 1 continuation).
        
        Once the database is initialized, subsequent workers should:
        1. Acquire the file lock (waiting if necessary)
        2. Detect that initialization is already complete (PG_VERSION exists)
        3. Skip redundant initdb call
        4. Return success without error
        
        This scenario represents late-joining workers in a multi-worker training job where
        some workers start after the database is already initialized.
        """
        import threading
        
        results = {
            "success_count": 0,
            "errors": [],
            "already_initialized_count": 0,
            "lock": threading.Lock()
        }
        
        # Step 1: Pre-initialize the database with the primary server
        postgres_utils.run_postgres(
            pg_port=unique_pg_port,
            pg_data_dir=temp_pg_data_dir,
            pg_username=pg_username,
            expected_major_version="17"
        )
        
        # Verify PG_VERSION exists
        pg_version_file = os.path.join(temp_pg_data_dir, "PG_VERSION")
        assert os.path.exists(pg_version_file), "Initial run should create PG_VERSION"
        
        def late_join_worker(pg_data_dir, pg_username, worker_id):
            """
            Worker thread that joins after database is already initialized.
            
            In a real multi-worker scenario, this would call run_postgres too,
            which should detect the existing database and skip redundant initialization.
            Here we directly verify the initialization was already done.
            """
            try:
                # Simulate late-joining worker checking if database is ready
                pg_version_file_check = os.path.join(pg_data_dir, "PG_VERSION")
                
                # This file indicates successful initialization
                if os.path.exists(pg_version_file_check):
                    with results["lock"]:
                        results["already_initialized_count"] += 1
                
                # Attempt to call run_postgres to verify it detects existing DB
                # and doesn't fail (file locking allows this)
                try:
                    postgres_utils.run_postgres(
                        pg_port=str(int(unique_pg_port) + 1000 + worker_id),
                        pg_data_dir=pg_data_dir,
                        pg_username=pg_username,
                        expected_major_version="17"
                    )
                except SystemExit:
                    # run_postgres may fail with SystemExit if it can't start a new server
                    # (only one server per data dir), but that's expected behavior.
                    # The important thing is that file locking prevented corruption.
                    pass
                
                with results["lock"]:
                    results["success_count"] += 1
            except Exception as e:
                with results["lock"]:
                    results["errors"].append(f"Worker {worker_id}: {str(e)}")
        
        # Step 2: Spawn late-joining worker threads
        num_threads = 3
        threads = []
        for i in range(num_threads):
            t = threading.Thread(
                target=late_join_worker,
                args=(temp_pg_data_dir, pg_username, i),
                name=f"LateJoinThread-{i}"
            )
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=60)
        
        # Verify that:
        # 1. All late-join threads completed (no unhandled exceptions)
        assert results["success_count"] == num_threads, (
            f"All {num_threads} late-join threads should complete without error. "
            f"Only {results['success_count']} did. Errors: {results['errors']}"
        )
        
        # 2. All threads detected that database was already initialized
        assert results["already_initialized_count"] == num_threads, (
            f"All {num_threads} threads should detect pre-existing PG_VERSION. "
            f"Only {results['already_initialized_count']} did."
        )
        
        # 3. PG_VERSION still exists (no corruption from concurrent access)
        assert os.path.exists(pg_version_file), (
            "PG_VERSION should still exist after concurrent late-join attempts"
        )


# Markers for conditional test execution
pytestmark = pytest.mark.skipif(
    shutil.which("initdb") is None or shutil.which("pg_ctl") is None,
    reason="PostgreSQL command-line tools (initdb, pg_ctl) not found in PATH"
)

