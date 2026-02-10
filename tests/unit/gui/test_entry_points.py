"""
Test entry point scripts for GUI launch functionality.
"""

import subprocess
import sys
from pathlib import Path


def test_runner_entry_help():
    """Test that runner entry script has help."""
    # Use Path relative to this test file
    test_dir = Path(__file__).parent
    entry_script = test_dir.parent.parent.parent / "src" / "abx_amr_simulator" / "gui" / "runner_entry.py"
    result = subprocess.run(
        [sys.executable, str(entry_script),"--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Help failed: {result.stderr}"
    assert "--results-dir" in result.stdout, "Missing --results-dir in help"
    print("✅ runner_entry.py --help works")


def test_viewer_entry_help():
    """Test that viewer entry script has help."""
    # Use Path relative to this test file
    test_dir = Path(__file__).parent
    entry_script = test_dir.parent.parent.parent / "src" / "abx_amr_simulator" / "gui" / "viewer_entry.py"
    result = subprocess.run(
        [sys.executable, str(entry_script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Help failed: {result.stderr}"
    assert "--results-dir" in result.stdout, "Missing --results-dir in help"
    print("✅ viewer_entry.py --help works")


def test_entry_point_syntax():
    """Test that both entry scripts compile."""
    import py_compile
    # Use Path relative to this test file
    test_dir = Path(__file__).parent
    runner_path = test_dir.parent.parent.parent / "src" / "abx_amr_simulator" / "gui" / "runner_entry.py"
    viewer_path = test_dir.parent.parent.parent / "src" / "abx_amr_simulator" / "gui" / "viewer_entry.py"
    py_compile.compile(str(runner_path), doraise=True)
    py_compile.compile(str(viewer_path), doraise=True)
    print("✅ Entry point scripts compile successfully")


if __name__ == "__main__":
    test_entry_point_syntax()
    test_runner_entry_help()
    test_viewer_entry_help()
    print("\n✅✅✅ All entry point tests passed!")
