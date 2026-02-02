"""
Experiment discovery and registry utilities.

Handles tracking of analyzed experiments, scanning for new runs, and
extracting experiment prefixes from timestamped result folders.
"""

import re
from pathlib import Path
from typing import Optional, List, Set, Tuple


def extract_experiment_prefix(run_folder_name: str) -> Optional[str]:
    """Extract experiment prefix from timestamped run folder name.
    
    Strips `_seed<N>_<timestamp>` suffix to get the base experiment name. Timestamp
    format: YYYYMMDD_HHMMSS. Useful for grouping multiple seeds of the same experiment.
    
    Args:
        run_folder_name (str): Name of a result folder (not full path). Should match
            pattern: exp_<name>_seed<N>_YYYYMMDD_HHMMSS.
    
    Returns:
        str | None: Experiment prefix if folder matches pattern, None otherwise.
    
    Example:
        >>> extract_experiment_prefix('exp_1.a_single_abx_lambda0.0_seed1_20260113_225141')
        'exp_1.a_single_abx_lambda0.0'
        >>> extract_experiment_prefix('exp_base_test_seed1_20260114_133652')
        'exp_base_test'
        >>> extract_experiment_prefix('random_folder')
        None
    """
    # Pattern: match anything starting with exp_, then non-greedily capture up to _seed<digits>_
    # Timestamp can contain underscores (YYYYMMDD_HHMMSS format)
    pattern = r'^(exp_.+?)_seed\d+_\d+.*$'
    match = re.match(pattern, run_folder_name)
    return match.group(1) if match else None


def extract_timestamp_from_run_folder(run_folder_name: str) -> Optional[str]:
    """Extract YYYYMMDD_HHMMSS timestamp suffix from a run folder name.

    Assumes run folders ALWAYS end with `_YYYYMMDD_HHMMSS` (strongly opinionated).
    Returns None if pattern not found.
    """
    match = re.search(r'_(\d{8}_\d{6})$', run_folder_name)
    return match.group(1) if match else None


def find_experiment_runs(prefix: str, results_dir: str = 'results', aggregate_by_seed: bool = False) -> List[Path]:
    """Find all run directories matching an experiment prefix.
    
    Behavior depends on aggregate_by_seed flag:
    - If aggregate_by_seed=True: Searches for folders matching pattern: <prefix>_seed*_*
      (groups multiple seeds together for ensemble analysis)
    - If aggregate_by_seed=False: Searches for folders that exactly match <prefix> OR match <prefix>_*
      (finds single runs or exact prefix matches without requiring seed pattern)
    
    Returns sorted list (by name) for consistent ordering.
    
    Args:
        prefix (str): Experiment prefix, e.g., 'exp_1.a_single_abx_lambda0.0' or 'my_run_20260115_143614'.
        results_dir (str): Path to results directory relative to project root.
            Default: 'results'.
        aggregate_by_seed (bool): If True, only match seed pattern (<prefix>_seed*_*).
            If False, match exact prefix OR any folder starting with <prefix>_. Default: False.
    
    Returns:
        List[Path]: Sorted list of Path objects for all matching run directories.
            Empty list if results_dir doesn't exist or no matches found.
    
    Example:
        >>> # Aggregate multiple seeds
        >>> runs = find_experiment_runs('exp_1.a_single_abx_lambda0.0', aggregate_by_seed=True)
        >>> # [Path('results/exp_1.a_single_abx_lambda0.0_seed1_20260113_225141'),
        >>> #  Path('results/exp_1.a_single_abx_lambda0.0_seed2_20260113_225209'), ...]
        >>> 
        >>> # Find exact run (including runs with no seed pattern)
        >>> runs = find_experiment_runs('my_run_20260115_143614', aggregate_by_seed=False)
        >>> # [Path('results/my_run_20260115_143614')]
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    
    if aggregate_by_seed:
        # Glob for folders matching prefix_seed*_* (seed aggregation pattern)
        pattern = f"{prefix}_seed*_*"
        matching_runs = sorted([p for p in results_path.glob(pattern) if p.is_dir()])
    else:
        # For non-aggregated mode, find both exact matches and <prefix>_* matches
        matching_runs = []
        
        # Try exact prefix match first (for non-timestamped runs)
        exact_match = results_path / prefix
        if exact_match.is_dir():
            matching_runs.append(exact_match)
        
        # Also try <prefix>_* pattern (for timestamped runs)
        pattern = f"{prefix}_*"
        for p in results_path.glob(pattern):
            if p.is_dir() and p not in matching_runs:
                matching_runs.append(p)
        
        matching_runs = sorted(matching_runs)
    
    return matching_runs


def load_registry(registry_path: str) -> Set[str]:
    """Load registry CSV and return unique run_names (prefixes) only."""
    entries = load_registry_csv(registry_path)
    return {name for name, _ in entries}


def update_registry(registry_path: str, run_name: str, timestamp: str) -> None:
    """Backward-compatible alias to update_registry_csv (timestamp required)."""
    update_registry_csv(registry_path=registry_path, run_name=run_name, timestamp=timestamp)


def scan_for_experiments(results_dir: str = 'results') -> Set[str]:
    """Scan results directory and return all unique experiment prefixes.
    
    Only considers folders matching pattern: exp_*_seed<N>_<timestamp>. Extracts
    unique prefixes via extract_experiment_prefix. Useful for discovering all
    experiments for analysis pipelines.
    
    Args:
        results_dir (str): Path to results directory relative to project root.
            Default: 'results'.
    
    Returns:
        Set[str]: Set of unique experiment prefixes found. Empty set if results_dir
            doesn't exist or no valid folders found.
    
    Example:
        >>> prefixes = scan_for_experiments()
        >>> # {'exp_1.a_single_abx_lambda0.0', 'exp_1.b_two_abx_lambda0.05', ...}
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return set()
    
    prefixes = set()
    for run_folder in results_path.glob('exp_*_seed*_*'):
        if run_folder.is_dir():
            prefix = extract_experiment_prefix(run_folder.name)
            if prefix:
                prefixes.add(prefix)
    
    return prefixes


def identify_new_experiments(registry_path: str, results_dir: str = 'results') -> Set[str]:
    """Find experiments in results directory that haven't been analyzed yet.
    
    Compares all experiment prefixes found in results_dir against prefixes in
    registry file. Returns set difference (new prefixes not yet analyzed). Used
    by analysis pipelines to run incremental updates without re-analyzing old data.
    
    Args:
        registry_path (str): Path to registry file tracking analyzed experiments
            (e.g., 'analysis_output/.phase2_completed.txt').
        results_dir (str): Path to results directory. Default: 'results'.
    
    Returns:
        Set[str]: Set of new experiment prefixes not in registry. Empty set if
            all experiments already analyzed.
    
    Example:
        >>> new_exps = identify_new_experiments('analysis_output/.phase2_completed.txt')
        >>> # {'exp_1.c_new_experiment_lambda0.1'}
    """
    all_prefixes = scan_for_experiments(results_dir=results_dir)
    analyzed_prefixes = load_registry(registry_path)
    return all_prefixes - analyzed_prefixes


def validate_and_clean_registry(registry_path: str, results_dir: str = 'results', exclude_prefix: str = None) -> List[Tuple[str, str]]:
    """Validate CSV registry entries against actual timestamped run folders.

    Strongly assumes run folders are named `<run_name>_<YYYYMMDD_HHMMSS>`.
    Removes any CSV rows whose folder no longer exists.

    Args:
        registry_path (str): Path to CSV registry (e.g., results/.training_completed.txt)
        results_dir (str): Base results directory
        exclude_prefix (str): Optional run_name to skip during validation (current run)

    Returns:
        List[Tuple[str, str]]: List of stale (run_name, timestamp) rows removed.
    """
    registry_file = Path(registry_path)
    if not registry_file.exists():
        return []

    entries = load_registry_csv(registry_path)
    if not entries:
        return []

    results_path = Path(results_dir)
    stale: List[Tuple[str, str]] = []
    valid: List[Tuple[str, str]] = []

    for run_name, timestamp in entries:
        if exclude_prefix and run_name == exclude_prefix:
            valid.append((run_name, timestamp))
            continue

        run_folder = results_path / f"{run_name}_{timestamp}"
        if run_folder.exists():
            valid.append((run_name, timestamp))
        else:
            stale.append((run_name, timestamp))

    if stale:
        with open(registry_file, 'w') as f:
            for name, ts in valid:
                f.write(f"{name},{ts}\n")

    return stale


def clear_registry(registry_path: str) -> None:
    """
    Clear the registry file (used for --force mode).
    
    Args:
        registry_path: Path to registry file
    """
    registry_file = Path(registry_path)
    if registry_file.exists():
        registry_file.unlink()


def update_registry_csv(registry_path: str, run_name: str, timestamp: str) -> None:
    """
    Add an experiment run to the CSV registry with both prefix and timestamp.
    
    Appends a new line: run_name,timestamp to the CSV registry file.
    Creates the file if it doesn't exist. This enables tracking of all timestamped
    copies of experiments for cleanup purposes.
    
    Args:
        registry_path (str): Path to registry CSV file (e.g., 'results/.training_completed.txt')
        run_name (str): Base experiment name without timestamp (e.g., 'exp_1.a_single_abx_seed1')
        timestamp (str): Timestamp suffix (e.g., '20250119_143614')
    
    Example:
        >>> update_registry_csv('results/.training_completed.txt', 'exp_1.a_seed1', '20250119_143614')
        # File now contains: exp_1.a_seed1,20250119_143614
    """
    registry_file = Path(registry_path)
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(registry_file, 'a') as f:
        f.write(f"{run_name},{timestamp}\n")


def load_registry_csv(registry_path: str) -> List[tuple]:
    """
    Load CSV registry entries as list of (run_name, timestamp) tuples.
    
    Args:
        registry_path (str): Path to registry CSV file
    
    Returns:
        List[tuple]: List of (run_name, timestamp) tuples. Empty list if file doesn't exist.
    
    Example:
        >>> entries = load_registry_csv('results/.training_completed.txt')
        >>> # [('exp_1.a_seed1', '20250119_143614'), ('exp_1.a_seed1', '20250119_144500'), ...]
    """
    registry_file = Path(registry_path)
    if not registry_file.exists():
        return []
    
    entries = []
    with open(registry_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ',' in line:
                parts = line.rsplit(',', 1)  # Split on last comma
                if len(parts) == 2:
                    entries.append(tuple(parts))
    
    return entries


def get_all_timestamps_for_run(registry_path: str, run_name: str) -> List[str]:
    """
    Get all timestamps for a specific run name from CSV registry.
    
    Useful for cleaning up old timestamped copies of experiments.
    
    Args:
        registry_path (str): Path to registry CSV file
        run_name (str): Base run name to search for
    
    Returns:
        List[str]: List of all timestamps associated with this run_name. Empty if not found.
    
    Example:
        >>> timestamps = get_all_timestamps_for_run('results/.training_completed.txt', 'exp_1.a_seed1')
        >>> # ['20250119_143614', '20250119_144500']
    """
    entries = load_registry_csv(registry_path)
    return [ts for name, ts in entries if name == run_name]


def is_run_completed(registry_path: str, run_name: str) -> bool:
    """
    Check if a run name exists in CSV registry (at any timestamp).
    
    Args:
        registry_path (str): Path to registry CSV file
        run_name (str): Base run name to search for
    
    Returns:
        bool: True if run_name appears in registry (has at least one entry), False otherwise.
    
    Example:
        >>> if is_run_completed('results/.training_completed.txt', 'exp_1.a_seed1'):
        ...     print("This run has been completed before")
    """
    return len(get_all_timestamps_for_run(registry_path, run_name)) > 0
