#!/usr/bin/env python3
"""
Launch both experiment_runner and experiment_viewer Streamlit apps.

Opens both apps in separate browser tabs and monitors for completed experiments
to automatically switch focus to the viewer.

Usage:
    abx-amr-simulator-launch-gui                                    # Default: uses project_root/results/
    abx-amr-simulator-launch-gui --results-dir /path/to/results    # Custom results directory
"""

import subprocess
import time
import webbrowser
import signal
import sys
import os
import argparse
from pathlib import Path
from threading import Thread

PROJECT_ROOT = Path(os.environ.get("ABX_PROJECT_ROOT", Path.cwd())).resolve()
RUNNER_PORT = 8501
VIEWER_PORT = 8502


# Removed get_workspace_dir() - entry points should respect user's CWD as project root


def start_streamlit_app(script_name: str, port: int, results_dir: str | None = None):
    """Start a Streamlit app on the specified port."""
    script_path = Path(__file__).resolve().parent / script_name
    cmd = [
        "streamlit", "run",
        str(script_path),
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    env = os.environ.copy()
    if results_dir:
        env["ABX_RESULTS_DIR"] = results_dir
    env.setdefault("ABX_PROJECT_ROOT", str(PROJECT_ROOT))
    
    process = subprocess.Popen(
        cmd,
        cwd=Path.cwd(),  # Use current working directory, not package's workspace
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    
    return process


def wait_for_app_ready(port: int, timeout: int = 30) -> bool:
    """Wait for Streamlit app to be ready by checking if port is responding."""
    import socket
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def monitor_marker_file():
    """Monitor for new experiment completions and focus viewer tab."""
    marker_file = PROJECT_ROOT / ".latest_experiment"
    last_check = 0
    
    while True:
        try:
            if marker_file.exists():
                # Check if marker file is new
                mtime = marker_file.stat().st_mtime
                if mtime > last_check:
                    last_check = mtime
                    time.sleep(1)  # Small delay to ensure viewer processes the marker
                    # Open/focus the viewer tab
                    viewer_url = f"http://localhost:{VIEWER_PORT}"
                    webbrowser.open(viewer_url, new=0)  # new=0 tries to reuse existing window
                    print(f"✨ New experiment detected - opening viewer at {viewer_url}")
        except Exception as e:
            print(f"Monitor error: {e}")
        
        time.sleep(2)  # Check every 2 seconds


def main():
    """Main entry point for launching both apps."""
    parser = argparse.ArgumentParser(
        description="Launch both ABX AMR RL Experiment Runner and Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    abx-amr-simulator-launch-gui                                    # Default: uses project_root/results/
  abx-amr-simulator-launch-gui --results-dir /path/to/results    # Custom results directory
        """,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Absolute path to results directory (default: project_root/results/)",
    )
    
    args = parser.parse_args()
    
    # Treat current working directory as the project root.
    project_root = Path.cwd().resolve()
    os.environ["ABX_PROJECT_ROOT"] = str(project_root)
    global PROJECT_ROOT
    PROJECT_ROOT = project_root

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
        if not results_dir.exists():
            print(f"⚠️  Results directory does not exist: {results_dir}")
            print(f"    Creating it now...")
            results_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ABX_RESULTS_DIR"] = str(results_dir)
        print(f"✅ Using results directory: {results_dir}")
    else:
        # Default to ./results in current working directory
        results_dir = Path.cwd() / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ABX_RESULTS_DIR"] = str(results_dir)
        print(f"✅ Using default results directory: {results_dir}")
    
    print("🚀 Starting ABX AMR RL Experiment Apps...")
    print(f"   Experiment Runner: http://localhost:{RUNNER_PORT}")
    print(f"   Results Viewer:    http://localhost:{VIEWER_PORT}")
    print()
    
    processes = []
    
    try:
        # Start experiment runner
        print("Starting experiment runner...")
        runner_process = start_streamlit_app("experiment_runner.py", RUNNER_PORT, str(results_dir))
        processes.append(runner_process)
        
        # Start experiment viewer
        print("Starting results viewer...")
        viewer_process = start_streamlit_app("experiment_viewer.py", VIEWER_PORT, str(results_dir))
        processes.append(viewer_process)
        
        # Wait for both apps to be ready
        print("\nWaiting for apps to start...")
        runner_ready = wait_for_app_ready(RUNNER_PORT)
        viewer_ready = wait_for_app_ready(VIEWER_PORT)
        
        if not runner_ready or not viewer_ready:
            print("❌ Apps failed to start properly. Check for port conflicts.")
            return
        
        print("✅ Both apps are ready!")
        print()
        
        # Open both tabs
        time.sleep(1)
        runner_url = f"http://localhost:{RUNNER_PORT}"
        viewer_url = f"http://localhost:{VIEWER_PORT}"
        
        print(f"Opening {runner_url}...")
        webbrowser.open(runner_url)
        time.sleep(1)
        
        print(f"Opening {viewer_url}...")
        webbrowser.open(viewer_url)
        
        print()
        print("📊 Apps are running! Press Ctrl+C to stop both apps.")
        print()
        
        # Start monitoring thread
        monitor_thread = Thread(target=monitor_marker_file, daemon=True)
        monitor_thread.start()
        
        # Wait for processes
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  One app exited unexpectedly (code {proc.returncode})")
                    return
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down apps...")
    
    finally:
        # Clean up processes
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        
        print("✅ All apps stopped.")


if __name__ == "__main__":
    main()
