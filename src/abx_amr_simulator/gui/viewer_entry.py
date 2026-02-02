#!/usr/bin/env python
"""
Entry point for launching the Experiment Viewer (experiment_viewer.py) Streamlit app.

Allows users to launch from anywhere with optional custom results directory.

Usage:
    abx-amr-simulator-experiment-viewer                                    # Uses workspace/results/
    abx-amr-simulator-experiment-viewer --results-dir /path/to/results    # Uses custom results directory
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse


def get_workspace_dir() -> Path:
    """Find the workspace directory (assumes project structure is stable)."""
    # This script is at: project_root/src/abx_amr_simulator/gui/viewer_entry.py
    # Go up to project_root, then to workspace
    project_root = Path(__file__).resolve().parents[3]
    workspace_dir = project_root / "workspace"
    return workspace_dir


def main():
    """Main entry point: parse arguments, set env vars, launch Streamlit app."""
    parser = argparse.ArgumentParser(
        description="Launch the ABX AMR RL Experiment Viewer GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  abx-amr-simulator-experiment-viewer                                    # Default: uses workspace/results/
  abx-amr-simulator-experiment-viewer --results-dir /path/to/results    # Custom results directory
        """,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Absolute path to results directory (default: workspace/results/)",
    )
    
    args = parser.parse_args()
    
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
        # Default to workspace/results
        workspace_dir = get_workspace_dir()
        results_dir = workspace_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ABX_RESULTS_DIR"] = str(results_dir)
        print(f"✅ Using default results directory: {results_dir}")
    
    # Find the experiment_viewer.py script
    script_dir = Path(__file__).parent
    viewer_script = script_dir / "experiment_viewer.py"
    
    if not viewer_script.exists():
        print(f"❌ Error: Could not find experiment_viewer.py at {viewer_script}")
        sys.exit(1)
    
    # Launch Streamlit app
    print(f"🚀 Launching Experiment Viewer...")
    try:
        # Change to workspace directory for relative path resolution
        workspace_dir = get_workspace_dir()
        os.chdir(workspace_dir)
        print(f"📂 Working directory: {os.getcwd()}")
        
        # Run streamlit
        subprocess.run(
            ["streamlit", "run", str(viewer_script)],
            check=False,
        )
    except KeyboardInterrupt:
        print("\n⏹️  Experiment Viewer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error launching Experiment Viewer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
