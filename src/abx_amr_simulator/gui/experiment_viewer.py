from pathlib import Path
from datetime import datetime
import os
import yaml

import streamlit as st

# Project root (repo root) needed for package-bundled assets
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Current working directory for user results
CWD = Path.cwd()


def get_results_directory() -> Path:
    """
    Get the results directory from environment variable or default to ./results.
    
    Priority:
    1. ABX_RESULTS_DIR environment variable (set by entry point if provided)
    2. ./results (relative to current working directory)
    
    Returns:
        Path: Absolute path to results directory (creates if doesn't exist)
    """
    results_dir_str = os.environ.get('ABX_RESULTS_DIR', './results')
    results_dir = Path(results_dir_str).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


RESULTS_DIR = get_results_directory()


def load_config_from_run(run_dir: Path) -> dict:
    """Load full_agent_env_config.yaml from a run directory."""
    config_path = run_dir / "full_agent_env_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def format_timestamp(run_dir: Path) -> str:
    """Extract and format timestamp from run directory name."""
    parts = run_dir.name.split('_')
    if len(parts) >= 2:
        date_part = parts[-2]  # YYYYMMDD
        time_part = parts[-1]  # HHMMSS
        try:
            dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return run_dir.name


def get_experiment_folders() -> list[Path]:
    """Get all experiment folders from results directory, sorted newest first."""
    if not RESULTS_DIR.exists():
        return []
    
    folders = [f for f in RESULTS_DIR.iterdir() if f.is_dir()]
    # Sort by modification time, newest first
    folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return folders


def get_latest_experiment_from_marker() -> Path | None:
    """Check if there's a marker file indicating a newly completed experiment."""
    marker_file = PROJECT_ROOT / ".latest_experiment"
    if marker_file.exists():
        try:
            with open(marker_file, "r") as f:
                path_str = f.read().strip()
            exp_path = Path(path_str)
            if exp_path.exists() and exp_path.is_dir():
                return exp_path
        except Exception:
            pass
    return None


def display_config_section(title: str, config_dict: dict, expanded: bool = False):
    """Display a config section in a formatted, read-only manner."""
    if not config_dict:
        return
    
    with st.expander(f"**{title}**", expanded=expanded):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                st.markdown(f"**{key}:**")
                # Nested dict - indent display
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        st.markdown(f"&nbsp;&nbsp;**{sub_key}:**")
                        for ssub_key, ssub_value in sub_value.items():
                            if isinstance(ssub_value, (list, dict)):
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{ssub_key}`: (complex)")
                                st.json({ssub_key: ssub_value})
                            else:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{ssub_key}`: {ssub_value}")
                    else:
                        if isinstance(sub_value, (list, dict)):
                            st.markdown(f"&nbsp;&nbsp;`{sub_key}`: (list/dict)")
                            st.json({sub_key: sub_value})
                        else:
                            st.markdown(f"&nbsp;&nbsp;`{sub_key}`: {sub_value}")
            else:
                if isinstance(value, (list, dict)):
                    st.markdown(f"`{key}`: (complex)")
                    st.json({key: value})
                else:
                    st.markdown(f"`{key}`: **{value}**")


def main():
    st.set_page_config(page_title="Experiment Viewer", layout="wide", page_icon="📊")
    
    st.title("Experiment Results Viewer")
    st.markdown("Browse and review diagnostic plots from completed experiments.")
    
    # Sidebar: Experiment selection
    st.sidebar.header("Experiment Runs")
    
    # Check for newly completed experiment
    latest_from_marker = get_latest_experiment_from_marker()
    if latest_from_marker:
        # Store in session state and clear marker
        if "selected_experiment" not in st.session_state or st.session_state.selected_experiment != str(latest_from_marker):
            st.session_state.selected_experiment = str(latest_from_marker)
            st.sidebar.success("✨ New experiment detected!")
            # Remove marker file after detection
            marker_file = PROJECT_ROOT / ".latest_experiment"
            try:
                marker_file.unlink()
            except Exception:
                pass
    
    experiments = get_experiment_folders()
    
    if not experiments:
        st.warning(f"No experiment folders found in `{RESULTS_DIR.relative_to(PROJECT_ROOT)}`")
        st.info("Run an experiment using the training app to see results here.")
        return
    
    # Filter by name prefix
    name_filter = st.sidebar.text_input("Filter by name", "")
    if name_filter:
        experiments = [exp for exp in experiments if name_filter.lower() in exp.name.lower()]
    
    if not experiments:
        st.sidebar.warning("No experiments match filter.")
        return
    
    # Display experiment list
    st.sidebar.markdown(f"**{len(experiments)} experiments found**")
    
    # Create clickable list of experiments
    experiment_names = [exp.name for exp in experiments]
    timestamps = [format_timestamp(exp) for exp in experiments]
    
    # Format display names with timestamps
    display_names = [f"{name}\n_{ts}_" for name, ts in zip(experiment_names, timestamps)]
    
    # Determine default selection
    default_idx = 0
    if "selected_experiment" in st.session_state:
        # Try to find the experiment in the list
        selected_path = Path(st.session_state.selected_experiment)
        try:
            default_idx = experiment_names.index(selected_path.name)
        except ValueError:
            # If not found, default to newest (index 0)
            default_idx = 0
    
    selected_idx = st.sidebar.radio(
        "Select experiment:",
        range(len(experiments)),
        format_func=lambda i: display_names[i],
        label_visibility="collapsed",
        index=default_idx
    )
    
    selected_run = experiments[selected_idx]
    
    # Main panel: Show selected experiment
    st.header(selected_run.name)
    st.caption(f"Run at: {timestamps[selected_idx]}")
    
    # Display diagnostic images
    st.subheader("Diagnostic Plots")
    
    figures_dir = selected_run / "figures_best_agent"
    
    if not figures_dir.exists():
        st.warning("No `figures_best_agent` directory found for this experiment.")
        return
    
    # Get all PNG files
    image_files = sorted(figures_dir.glob("*.png"))
    
    if not image_files:
        st.info("No diagnostic plots found in `figures_best_agent`.")
        return
    
    # Helper to find matching images for a pattern (exact or prefix)
    def match_images(pattern: str):
        matches = []
        if pattern.endswith(".png"):
            matches = [img for img in image_files if img.name == pattern]
        else:
            matches = [img for img in image_files if img.name.startswith(pattern)]
        return matches

    # Track already added images to avoid duplicates across sections/subsections
    seen = set()

    # --- AMR Dynamics ---
    st.markdown("#### AMR Dynamics")
    amr_main = match_images("amr_levels_over_time.png")
    if amr_main:
        for img_path in amr_main:
            seen.add(img_path.name)
            st.image(str(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.info("No AMR levels plot found.")

    leaky_imgs = [img for img in match_images("leaky_balloon_response") if img.name not in seen]
    if leaky_imgs:
        with st.expander("Leaky balloon responses"):
            for img_path in leaky_imgs:
                seen.add(img_path.name)
                st.image(str(img_path), caption=img_path.name, use_container_width=True)
    st.markdown("")

    # --- Agent Behavior ---
    st.markdown("#### Agent Behavior")

    st.markdown("**Overall prescribing behavior**")
    overall_imgs = []
    for fname in [
        "clinical_benefits_failures_adverse_events_over_time.png",
        "outcome_counts_over_time.png", "abx_prescriptions_over_time.png"
    ]:
        overall_imgs.extend([img for img in match_images(fname) if img.name not in seen])
    if overall_imgs:
        for img_path in overall_imgs:
            seen.add(img_path.name)
            st.image(str(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.info("No overall prescribing behavior plots found.")

    st.markdown("**Outcomes per antibiotic for Infected Treated**")
    infected_imgs = [img for img in match_images("infected_treated") if img.name not in seen]
    if infected_imgs:
        for img_path in infected_imgs:
            seen.add(img_path.name)
            st.image(str(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.info("No infected-treated per-antibiotic plots found.")
    st.markdown("")

    # --- Rewards & Returns ---
    st.markdown("#### Rewards & Returns")
    rewards_imgs = [img for img in match_images("reward_components_over_time.png") if img.name not in seen]
    if rewards_imgs:
        for img_path in rewards_imgs:
            seen.add(img_path.name)
            st.image(str(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.info("No reward plots found.")
    
    st.markdown("---")
    
    # --- Configuration ---
    st.subheader("⚙️ Configuration")
    
    config = load_config_from_run(selected_run)
    
    if config:
        # Display config in organized sections for the new nested structure
        col1, col2 = st.columns(2)
        
        with col1:
            # Environment
            if "environment" in config:
                display_config_section("Environment", config["environment"], expanded=True)
            
            # Patient Generator
            if "patient_generator" in config:
                display_config_section("Patient Generator", config["patient_generator"], expanded=True)
            
            # Reward Calculator
            if "reward_calculator" in config:
                display_config_section("Reward Calculator", config["reward_calculator"], expanded=True)
        
        with col2:
            # Training
            if "training" in config:
                display_config_section("Training", config["training"], expanded=True)
            
            # Algorithm (PPO, A2C, DQN, etc.)
            algo = config.get("algorithm", "").lower() if isinstance(config.get("algorithm"), str) else None
            if algo and algo in config:
                display_config_section(f"{algo.upper()} Params", config[algo], expanded=True)
        
        # Show other top-level parameters not already displayed
        st.markdown("---")
        other_params = {
            k: v for k, v in config.items() 
            if k not in [
                "environment", "patient_generator", "reward_calculator", "reward_model", "training", 
                "algorithm", "run_name", "output_dir", "action_mode"
            ] and not isinstance(v, dict)  # Skip nested dicts already shown
        }
        if other_params:
            with st.expander("**Other Parameters**", expanded=False):
                st.json(other_params)
    
    else:
        st.info("No full_agent_env_config.yaml found for this experiment.")


if __name__ == "__main__":
    main()
