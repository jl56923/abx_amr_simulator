"""Root conftest for abx_amr_simulator tests.

This file configures pytest for all tests in the abx_amr_simulator package.
It ensures that test reference helpers are importable from all test files.
"""

import sys
from pathlib import Path

# Add the unit/utils directory to sys.path so test_reference_helpers can be imported
# This makes the import available to all test files without per-file sys.path manipulation
_tests_root = Path(__file__).parent
_utils_path = _tests_root / "unit" / "utils"
if str(_utils_path) not in sys.path:
    sys.path.insert(0, str(_utils_path))
