#!/usr/bin/env python
"""
Entry point for launching the Experiment Runner (experiment_runner.py) Streamlit app.

Allows users to launch from anywhere with optional custom results directory.

Usage:
    abx-amr-simulator-experiment-runner                                    # Uses project_root/results/
    abx-amr-simulator-experiment-runner --results-dir /path/to/results    # Uses custom results directory
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse


# Removed get_workspace_dir() - entry points should respect user's CWD as project root


def main():
    """Main entry point: parse arguments, set env vars, launch Streamlit app."""
    parser = argparse.ArgumentParser(
        description="Launch the ABX AMR RL Experiment Runner GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    abx-amr-simulator-experiment-runner                                    # Default: uses project_root/results/
  abx-amr-simulator-experiment-runner --results-dir /path/to/results    # Custom results directory
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
    os.environ["ABX_PROJECT_ROOT"] = str(Path.cwd().resolve())

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
        # Default to ./results in project root
        results_dir = Path.cwd() / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ABX_RESULTS_DIR"] = str(results_dir)
        print(f"✅ Using default results directory: {results_dir}")
    
    # Find the experiment_runner.py script
    script_dir = Path(__file__).parent
    runner_script = script_dir / "experiment_runner.py"
    
    if not runner_script.exists():
        print(f"❌ Error: Could not find experiment_runner.py at {runner_script}")
        sys.exit(1)
    
    # Launch Streamlit app
    print(f"🚀 Launching Experiment Runner...")
    print(f"📂 Working directory: {os.getcwd()}")
    try:
        # Run streamlit - subprocess inherits current working directory
        subprocess.run(
            ["streamlit", "run", str(runner_script)],
            check=False,
        )
    except KeyboardInterrupt:
        print("\n⏹️  Experiment Runner stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error launching Experiment Runner: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
