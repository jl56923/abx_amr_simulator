import os
import glob
import subprocess
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import signal
import psutil
import sys

import streamlit as st
import yaml

# Package root for fallback to bundled default configs
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
# Project root (user workspace) for experiments and results
PROJECT_ROOT = Path(os.environ.get("ABX_PROJECT_ROOT", Path.cwd())).resolve()

from abx_amr_simulator.utils import load_config, apply_param_overrides
from abx_amr_simulator.gui.patient_gen_ui_helper import migrate_old_config_to_new, build_attribute_ui_section


def get_results_directory() -> Path:
    """
    Get the results directory from environment variable or default to ./results.
    
    Priority:
    1. ABX_RESULTS_DIR environment variable (set by entry point if provided)
    2. ./results (relative to current working directory)
    
    Returns:
        Path: Absolute path to results directory (creates if doesn't exist)
    """
    project_root = Path(os.environ.get("ABX_PROJECT_ROOT", Path.cwd())).resolve()
    results_dir_str = os.environ.get('ABX_RESULTS_DIR', str(project_root / "results"))
    results_dir = Path(results_dir_str).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# Try to find configs in project root, then fall back to package defaults
project_configs_dir = PROJECT_ROOT / "experiments" / "configs"
legacy_configs_dir = PROJECT_ROOT / "configs"
if project_configs_dir.exists():
    CONFIG_DIR = project_configs_dir
elif legacy_configs_dir.exists():
    CONFIG_DIR = legacy_configs_dir
else:
    CONFIG_DIR = PACKAGE_ROOT / "experiments" / "configs"
GENERATED_DIR = CONFIG_DIR / "generated_from_streamlit"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def list_config_files():
    """List available experiment configs from umbrella_configs/ subdirectory.
    
    Only umbrella configs (that reference component configs) are shown in the GUI.
    Component configs (environment/, agent_algorithm/, etc.) should not be selected directly.
    """
    umbrella_dir = CONFIG_DIR / "umbrella_configs"
    if not umbrella_dir.exists():
        return []
    
    configs = sorted([f for f in umbrella_dir.glob("*.yaml") if f.is_file()])
    return configs


def get_default_component_configs() -> Dict[str, Path]:
    """Get default component configs for creating a new experiment."""
    return {
        'environment': CONFIG_DIR / 'environment' / 'default.yaml',
        'reward_calculator': CONFIG_DIR / 'reward_calculator' / 'default.yaml',
        'patient_generator': CONFIG_DIR / 'patient_generator' / 'default.yaml',
        'agent_algorithm': CONFIG_DIR / 'agent_algorithm' / 'default.yaml',
    }


def find_latest_run(output_dir: Path, run_name_prefix: str) -> Path | None:
    candidates = list(output_dir.glob(f"{run_name_prefix}_*"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def write_config(config: Dict[str, Any], run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = GENERATED_DIR / f"{run_name}_{timestamp}.yaml"
    with open(filename, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    return filename


def main():
    st.set_page_config(page_title="ABX AMR RL Trainer", layout="wide")
    st.title("ABX AMR RL Trainer")
    
    # Set 'training_in_progress' and 'training_pid' values:
    if 'training_in_progress' not in st.session_state:
        st.session_state['training_in_progress'] = False
    if 'training_pid' not in st.session_state:
        st.session_state['training_pid'] = -1

    results_dir = get_results_directory()

    st.sidebar.header("Configuration Source")
    configs = list_config_files()
    
    if not configs:
        st.error(f"❌ No experiment configs found")
        st.info("Please ensure you have umbrella configs in experiments/configs/umbrella_configs/ (e.g., base_experiment.yaml)")
        st.stop()
    
    # Create display names and keep mapping to full paths
    config_names = [cfg.name for cfg in configs]
    config_paths = {cfg.name: cfg for cfg in configs}
    
    # Default to base_experiment.yaml if it exists
    default_idx = 0
    if "base_experiment.yaml" in config_names:
        default_idx = config_names.index("base_experiment.yaml")
    
    selected_name = st.sidebar.selectbox("Base config", config_names, index=default_idx)
    base_config_path = config_paths[selected_name]  # Use the actual full path
    config = load_config(str(base_config_path))

    st.sidebar.markdown("**Results directory**")
    st.sidebar.code(str(results_dir))
    st.sidebar.caption("Expected layout: my_project/results/")

    st.sidebar.markdown("**Run naming**")
    run_name_input = st.sidebar.text_input("Run name prefix", value=config.get("run_name", "streamlit_run"))
    
    st.sidebar.markdown("**Continue Training**")
    continue_training = st.sidebar.checkbox("Continue from prior experiment", value=False)
    prior_experiment = None
    additional_episodes = None
    
    if continue_training:
        if results_dir.exists():
            experiments = sorted([f.name for f in results_dir.iterdir() if f.is_dir()], reverse=True)
            if experiments:
                prior_experiment = st.sidebar.selectbox(
                    "Select experiment to continue",
                    experiments,
                    help="Pick a previous experiment to continue training from"
                )
                additional_episodes = st.sidebar.number_input(
                    "Additional training episodes",
                    min_value=10,
                    value=100,
                    step=10,
                    help="Number of additional episodes to train"
                )
                prior_config_path = results_dir / prior_experiment / "full_agent_env_config.yaml"
                if prior_config_path.exists():
                    config = load_config(str(prior_config_path))
                    st.sidebar.success(f"✅ Loaded config from {prior_experiment}")
                else:
                    st.sidebar.error("⚠️ No full_agent_env_config.yaml found in prior experiment")
                    continue_training = False
            else:
                st.sidebar.warning("No previous experiments found in results/")
                continue_training = False
        else:
            st.sidebar.warning("No results/ directory found")
            continue_training = False

    if continue_training:
        st.info(f"📋 **Continuing previous experiment:** `{prior_experiment}` (adding {additional_episodes} episodes)")

    st.header("ABXAMREnv")
    if continue_training:
        st.caption("🔒 Environment configuration is locked when continuing from prior experiment")
    env_cfg = config.get("environment", {})

    st.subheader("Core environment")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_patients = st.number_input(
            "num_patients_per_time_step",
            min_value=1,
            value=int(env_cfg.get("num_patients_per_time_step", 1)),
            disabled=continue_training
        )
        max_steps = st.number_input(
            "max_time_steps",
            min_value=1,
            value=int(env_cfg.get("max_time_steps", 1000)),
            disabled=continue_training
        )
    with col2:
        update_visible = st.number_input(
            "update_visible_AMR_levels_every_n_timesteps",
            min_value=1,
            value=int(env_cfg.get("update_visible_AMR_levels_every_n_timesteps", 1)),
            disabled=continue_training
        )
        add_noise_visible = st.number_input(
            "add_noise_to_visible_AMR_levels",
            min_value=0.0,
            value=float(env_cfg.get("add_noise_to_visible_AMR_levels", 0.0)),
            step=0.01,
            disabled=continue_training,
            help="Stddev of Gaussian noise added to visible AMR levels"
        )
    with col3:
        add_bias_visible = st.number_input(
            "add_bias_to_visible_AMR_levels",
            min_value=0.0,
            max_value=1.0,
            value=float(env_cfg.get("add_bias_to_visible_AMR_levels", 0.0)),
            step=0.01,
            disabled=continue_training,
            help="Constant bias applied to visible AMR levels"
        )
    
    st.info("ℹ️ **Action space**: Mutually exclusive prescribing. Agent chooses one antibiotic OR no treatment per patient (not simultaneous combinations).")

    st.subheader("Observable patient attributes")
    st.caption("ℹ️ `prob_infected` is always included in observations (minimum required attribute)")
    
    available_attributes = [
        "benefit_value_multiplier",
        "failure_value_multiplier",
        "benefit_probability_multiplier",
        "failure_probability_multiplier",
        "recovery_without_treatment_prob",
    ]
    # Determine default checkbox states from patient_generator/default.yaml
    try:
        pg_default_path = CONFIG_DIR / "patient_generator" / "default.yaml"
        with open(pg_default_path, "r") as f:
            pg_defaults = yaml.safe_load(f) or {}
    except Exception:
        pg_defaults = {}

    default_visible_attrs = pg_defaults.get("visible_patient_attributes", ["prob_infected"])  # list
    default_visible_set = {a for a in default_visible_attrs if a != "prob_infected"}

    # Use individual checkboxes. Also render prob_infected as a locked, always-checked checkbox.
    st.markdown("**Additional patient attributes to include in observations**")
    st.checkbox(
        label="prob_infected",
        value=True,
        disabled=True,
        key="observable_attr_prob_infected_locked_env",
    )

    selected_observables = []
    for attr in available_attributes:
        default_checked = attr in default_visible_set
        is_checked = st.checkbox(
            label=attr,
            value=default_checked,
            disabled=continue_training,
            key=f"observable_attr_env_{attr}"
        )
        if is_checked:
            selected_observables.append(attr)
    
    st.subheader("Antibiotics & Crossresistance")
    if "antibiotics_dict" not in st.session_state or continue_training:
        st.session_state.antibiotics_dict = env_cfg.get(
            "antibiotics_AMR_dict",
            {"Antibiotic_A": {"leak": 0.5, "flatness_parameter": 30, "permanent_residual_volume": 0.0, "initial_amr_level": 0.0}},
        )

    antibiotics_to_delete = []
    antibiotics_dict_updated = {}
    for idx, (abx_name, abx_params) in enumerate(st.session_state.antibiotics_dict.items()):
        with st.expander(f"🔹 {abx_name}", expanded=True):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                new_name = st.text_input(
                    "Antibiotic name",
                    value=abx_name,
                    key=f"name_{idx}_{abx_name}",
                    disabled=continue_training,
                )
            with col_b:
                if not continue_training and st.button("🗑️ Remove", key=f"del_{idx}_{abx_name}"):
                    antibiotics_to_delete.append(abx_name)
                    continue

            col1, col2 = st.columns(2)
            with col1:
                leak = st.slider(
                    "leak",
                    0.0,
                    1.0,
                    float(abx_params.get("leak", 0.5)),
                    key=f"leak_{idx}_{abx_name}",
                    disabled=continue_training,
                )
                flatness = st.number_input(
                    "flatness_parameter",
                    min_value=1,
                    value=int(abx_params.get("flatness_parameter", 30)),
                    key=f"flatness_{idx}_{abx_name}",
                    disabled=continue_training,
                )
            with col2:
                perm_res = st.number_input(
                    "permanent_residual_volume",
                    min_value=0.0,
                    value=float(abx_params.get("permanent_residual_volume", 0.0)),
                    key=f"perm_res_{idx}_{abx_name}",
                    disabled=continue_training,
                )
                init_press = st.number_input(
                    "initial_amr_level",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(abx_params.get("initial_amr_level", 0.0)),
                    key=f"init_press_{idx}_{abx_name}",
                    disabled=continue_training,
                )

            antibiotics_dict_updated[new_name] = {
                "leak": leak,
                "flatness_parameter": flatness,
                "permanent_residual_volume": perm_res,
                "initial_amr_level": init_press,
            }

    for abx in antibiotics_to_delete:
        if abx in st.session_state.antibiotics_dict:
            del st.session_state.antibiotics_dict[abx]
            st.rerun()
    st.session_state.antibiotics_dict = antibiotics_dict_updated

    if not continue_training and st.button("➡️ Add new antibiotic", key="add_antibiotic_env"):
        new_abx_name = f"Antibiotic_{len(st.session_state.antibiotics_dict) + 1}"
        st.session_state.antibiotics_dict[new_abx_name] = {
            "leak": 0.5,
            "flatness_parameter": 30,
            "permanent_residual_volume": 0.0,
            "initial_amr_level": 0.0,
        }
        st.rerun()

    if "crossresistance_matrix" not in st.session_state or continue_training:
        st.session_state.crossresistance_matrix = env_cfg.get("crossresistance_matrix", {})

    st.markdown("**Crossresistance Matrix (optional)**")
    st.markdown(
        """
    Define how prescribing one antibiotic affects the AMR levels of other antibiotics.
    - Diagonal entries (self → self) are always 1.0 and don't need to be specified.
    - Only specify off-diagonal crossresistance ratios (0.0 to 1.0).
    - Leave empty for no crossresistance (identity matrix).
    """
    )

    current_antibiotics = list(st.session_state.antibiotics_dict.keys())

    if len(current_antibiotics) < 2:
        st.info("Add at least 2 antibiotics to configure crossresistance.")
    elif continue_training:
        st.info("🔒 Crossresistance configuration is locked when continuing from prior experiment")
        if st.session_state.crossresistance_matrix:
            st.json(st.session_state.crossresistance_matrix)
    else:
        st.markdown("**Configure crossresistance ratios** (prescriber → target)")
        use_matrix_view = st.checkbox("Use matrix view", value=False, key="crossresistance_matrix_view")

        if use_matrix_view:
            st.markdown(
                "*Values represent: when prescribing [row antibiotic], how much AMR increases for [column antibiotic]*"
            )
            cols = st.columns([2] + [1] * len(current_antibiotics))
            cols[0].markdown("**Prescriber →**")
            for idx, target_abx in enumerate(current_antibiotics):
                cols[idx + 1].markdown(f"**{target_abx[:8]}**")

            crossresistance_updated = {}
            for prescriber_abx in current_antibiotics:
                cols = st.columns([2] + [1] * len(current_antibiotics))
                cols[0].markdown(f"**{prescriber_abx}**")

                if prescriber_abx not in crossresistance_updated:
                    crossresistance_updated[prescriber_abx] = {}

                for idx, target_abx in enumerate(current_antibiotics):
                    if prescriber_abx == target_abx:
                        cols[idx + 1].markdown("*1.0*")
                    else:
                        existing_value = (
                            st.session_state.crossresistance_matrix.get(prescriber_abx, {}).get(target_abx, 0.0)
                        )
                        value = cols[idx + 1].number_input(
                            f"{prescriber_abx}→{target_abx}",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(existing_value),
                            step=0.01,
                            format="%.2f",
                            key=f"crossresistance_{prescriber_abx}_{target_abx}",
                            label_visibility="collapsed",
                        )
                        if value > 0.0:
                            crossresistance_updated[prescriber_abx][target_abx] = value

            crossresistance_updated = {k: v for k, v in crossresistance_updated.items() if v}
            st.session_state.crossresistance_matrix = crossresistance_updated

        else:
            st.markdown("*Add specific crossresistance relationships (only non-zero values)*")
            crossresistance_cleaned = {}
            for prescriber, targets in st.session_state.crossresistance_matrix.items():
                if prescriber in current_antibiotics:
                    crossresistance_cleaned[prescriber] = {
                        target: ratio
                        for target, ratio in targets.items()
                        if target in current_antibiotics and target != prescriber
                    }
            st.session_state.crossresistance_matrix = {k: v for k, v in crossresistance_cleaned.items() if v}

            pairs_to_delete = []
            crossresistance_updated = {}

            for prescriber, targets in st.session_state.crossresistance_matrix.items():
                for target, ratio in targets.items():
                    key = f"{prescriber}→{target}"
                    with st.expander(f"🔗 {key}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            new_ratio = st.slider(
                                f"Ratio for {key}",
                                0.0,
                                1.0,
                                float(ratio),
                                key=f"crossresistance_ratio_{prescriber}_{target}",
                            )
                        with col2:
                            if st.button("🗑️ Remove", key=f"del_crossresistance_{prescriber}_{target}"):
                                pairs_to_delete.append((prescriber, target))
                                continue

                        if prescriber not in crossresistance_updated:
                            crossresistance_updated[prescriber] = {}
                        if new_ratio > 0.0:
                            crossresistance_updated[prescriber][target] = new_ratio

            for prescriber, target in pairs_to_delete:
                if prescriber in st.session_state.crossresistance_matrix:
                    if target in st.session_state.crossresistance_matrix[prescriber]:
                        del st.session_state.crossresistance_matrix[prescriber][target]
                        if not st.session_state.crossresistance_matrix[prescriber]:
                            del st.session_state.crossresistance_matrix[prescriber]
                st.rerun()

            st.session_state.crossresistance_matrix = {k: v for k, v in crossresistance_updated.items() if v}

            st.markdown("**Add new crossresistance relationship**")
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                new_prescriber = st.selectbox("Prescriber antibiotic", current_antibiotics, key="new_crossresistance_prescriber")
            with col2:
                available_targets = [abx for abx in current_antibiotics if abx != new_prescriber]
                if available_targets:
                    new_target = st.selectbox("Target antibiotic", available_targets, key="new_crossresistance_target")
                else:
                    st.info("No available targets")
                    new_target = None
            with col3:
                if new_target and st.button("➕ Add", key="add_crossresistance_pair"):
                    if new_prescriber not in st.session_state.crossresistance_matrix:
                        st.session_state.crossresistance_matrix[new_prescriber] = {}
                    st.session_state.crossresistance_matrix[new_prescriber][new_target] = 0.1
                    st.rerun()

    st.markdown("---")
    st.header("Patient Generator")

    # Load existing config or fall back to default patient generator config
    patient_gen_cfg = config.get("patient_generator", {})
    default_pg_cfg = {}
    try:
        default_pg_path = CONFIG_DIR / "patient_generator" / "default.yaml"
        with open(default_pg_path, "r") as f:
            default_pg_cfg = yaml.safe_load(f) or {}
    except Exception:
        default_pg_cfg = {}

    # Migrate old flat config to new nested format if needed
    patient_gen_cfg = migrate_old_config_to_new(patient_gen_cfg) if patient_gen_cfg else {}
    default_pg_cfg = migrate_old_config_to_new(default_pg_cfg) if default_pg_cfg else {}
    
    # Observable patient attributes selector
    st.markdown("### Observable Patient Attributes")
    st.markdown("Select which patient attributes the agent can observe during training:")
    
    available_attributes = [
        "benefit_value_multiplier",
        "failure_value_multiplier",
        "benefit_probability_multiplier",
        "failure_probability_multiplier",
        "recovery_without_treatment_prob",
    ]
    
    default_visible_attrs = patient_gen_cfg.get(
        "visible_patient_attributes",
        default_pg_cfg.get("visible_patient_attributes", ["prob_infected"]),
    )
    default_visible_set = {a for a in default_visible_attrs if a != "prob_infected"}

    # prob_infected is always observed
    st.checkbox(
        label="prob_infected",
        value=True,
        disabled=True,
        key="observable_attr_prob_infected_locked_pg",
        help="Always observed (required for agent to make decisions)"
    )

    selected_observables = []
    for attr in available_attributes:
        default_checked = attr in default_visible_set
        is_checked = st.checkbox(
            label=attr,
            value=default_checked,
            disabled=continue_training,
            key=f"observable_attr_pg_{attr}"
        )
        if is_checked:
            selected_observables.append(attr)

    # Per-attribute configuration sections
    st.markdown("### Per-Attribute Configuration")
    with st.expander("How patient parameters are distributed and observed", expanded=False):
        st.markdown(
            """
        **Probability Distribution**: How each latent trait is drawn.
        - **constant**: All patients have the same value
        - **gaussian**: Patients vary with specified mean and standard deviation
        
        **Observation Settings**: How the agent perceives each attribute.
        - **Observation bias**: Multiplicative factor (1.0 = unbiased)
        - **Observation noise**: Gaussian noise as a fraction of the attribute's range (e.g., 0.2 = ±20% range noise)
        - **Clipping bounds**: Range for observed values after noise is applied
        """
        )

    # Define attribute display info
    attribute_info = {
        'prob_infected': {'display': 'Probability of Infection', 'min': 0.0, 'max': 1.0},
        'benefit_value_multiplier': {'display': 'Clinical Benefit Value Multiplier', 'min': 0.0, 'max': None},
        'failure_value_multiplier': {'display': 'Clinical Failure Value Multiplier', 'min': 0.0, 'max': None},
        'benefit_probability_multiplier': {'display': 'Clinical Benefit Probability Multiplier', 'min': 0.0, 'max': None},
        'failure_probability_multiplier': {'display': 'Clinical Failure Probability Multiplier', 'min': 0.0, 'max': None},
        'recovery_without_treatment_prob': {'display': 'Recovery Without Treatment Probability', 'min': 0.0, 'max': 1.0},
    }

    # Build UI for each attribute
    patient_gen_cfg_updated = {}
    for attr_name, attr_info_dict in attribute_info.items():
        attr_cfg = patient_gen_cfg.get(attr_name, {})

        # Fall back to default patient generator config when missing
        if not attr_cfg:
            attr_cfg = default_pg_cfg.get(attr_name, {})

        # Last-resort defaults if default config is unavailable
        if not attr_cfg:
            attr_cfg = {
                'prob_dist': {'type': 'constant', 'value': 1.0},
                'obs_bias_multiplier': 1.0,
                'obs_noise_std_dev_fraction': 0.0,
                'obs_noise_one_std_dev': 0.2 if attr_info_dict['max'] == 1.0 else 1.0,
                'clipping_bounds': [attr_info_dict['min'], attr_info_dict['max']],
            }
        
        attr_config_dict = build_attribute_ui_section(
            st, attr_name, attr_cfg, continue_training,
            attr_info_dict['display'],
            min_bounds=attr_info_dict['min'],
            max_bounds=attr_info_dict['max']
        )
        patient_gen_cfg_updated[attr_name] = attr_config_dict

    # Add visible_patient_attributes to config
    patient_gen_cfg_updated['visible_patient_attributes'] = ["prob_infected"] + selected_observables
    
    st.markdown("---")
    
    st.header("RewardCalculator")
    if continue_training:
        st.caption("🔒 Reward configuration is locked when continuing from prior experiment")
    rew_cfg = config.get("reward_calculator", config.get("reward_model", {}))
    lambda_weight = st.slider(
        "lambda_weight",
        0.0, 1.0,
        float(rew_cfg.get("lambda_weight", 0.5)),
        disabled=continue_training
    )
    epsilon = st.slider(
        "epsilon",
        0.0, 0.5,
        float(rew_cfg.get("epsilon", 0.05)),
        disabled=continue_training,
        help="AMR penalty weight (0-0.5). Epsilon represents the AMR penalty as a percentage of the normalized reward scale."
    )
    st.caption("💡 Epsilon scales consistently with clinical rewards. For example, epsilon=0.05 means '5% AMR penalty relative to clinical reward scale'.")
    
    st.subheader("Antibiotics Clinical Reward/Penalties Configuration")
    # Get existing config or defaults
    abx_clinical_cfg = rew_cfg.get("abx_clinical_reward_penalties_info_dict", {})
    
    # Top-level clinical parameters (not per-antibiotic)
    col1, col2 = st.columns(2)
    with col1:
        clinical_benefit_reward = st.number_input(
            "clinical_benefit_reward",
            value=float(abx_clinical_cfg.get("clinical_benefit_reward", 10.0)),
            disabled=continue_training
        )
        clinical_benefit_probability = st.slider(
            "clinical_benefit_probability",
            0.0, 1.0,
            float(abx_clinical_cfg.get("clinical_benefit_probability", 1.0)),
            disabled=continue_training
        )
    with col2:
        clinical_failure_penalty = st.number_input(
            "clinical_failure_penalty",
            value=float(abx_clinical_cfg.get("clinical_failure_penalty", -10.0)),
            disabled=continue_training
        )
        clinical_failure_probability = st.slider(
            "clinical_failure_probability",
            0.0, 1.0,
            float(abx_clinical_cfg.get("clinical_failure_probability", 1.0)),
            disabled=continue_training
        )
    
    # Per-antibiotic adverse effects
    st.markdown("**Antibiotic-specific adverse effects**")
    # Initialize session state for adverse effects if not present
    if "adverse_effects_dict" not in st.session_state or continue_training:
        existing_adverse = abx_clinical_cfg.get("abx_adverse_effects_info", {})
        # Ensure all antibiotics from AMR dict have adverse effects entries
        st.session_state.adverse_effects_dict = {}
        for abx_name in st.session_state.antibiotics_dict.keys():
            if abx_name in existing_adverse:
                st.session_state.adverse_effects_dict[abx_name] = existing_adverse[abx_name]
            else:
                st.session_state.adverse_effects_dict[abx_name] = {
                    "adverse_effect_penalty": -1.0,
                    "adverse_effect_probability": 0.1
                }
    
    # Sync adverse effects dict with current antibiotics
    # Add any new antibiotics from AMR dict
    for abx_name in st.session_state.antibiotics_dict.keys():
        if abx_name not in st.session_state.adverse_effects_dict:
            st.session_state.adverse_effects_dict[abx_name] = {
                "adverse_effect_penalty": -1.0,
                "adverse_effect_probability": 0.1
            }
    # Remove any antibiotics no longer in AMR dict
    adverse_to_remove = [abx for abx in st.session_state.adverse_effects_dict.keys() if abx not in st.session_state.antibiotics_dict]
    for abx in adverse_to_remove:
        del st.session_state.adverse_effects_dict[abx]
    
    # Display adverse effects for each antibiotic
    adverse_effects_updated = {}
    for abx_name, adverse_params in st.session_state.adverse_effects_dict.items():
        with st.expander(f"💊 {abx_name} - Adverse Effects", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                adverse_penalty = st.number_input(
                    "adverse_effect_penalty", 
                    value=float(adverse_params.get("adverse_effect_penalty", -1.0)),
                    key=f"adverse_penalty_{abx_name}",
                    disabled=continue_training
                )
            with col2:
                adverse_prob = st.slider(
                    "adverse_effect_probability", 
                    0.0, 1.0, 
                    float(adverse_params.get("adverse_effect_probability", 0.1)),
                    key=f"adverse_prob_{abx_name}",
                    disabled=continue_training
                )
            adverse_effects_updated[abx_name] = {
                "adverse_effect_penalty": adverse_penalty,
                "adverse_effect_probability": adverse_prob,
            }
    
    st.session_state.adverse_effects_dict = adverse_effects_updated

    st.header("Training parameters")
    if continue_training:
        st.info(f"Continuing from: **{prior_experiment}** with {additional_episodes:,} additional episodes")
        st.caption("🔒 Training configuration is locked (except additional episodes set above)")
    train_cfg = config.get("training", {})
    total_num_training_episodes = st.number_input(
        "total_num_training_episodes",
        min_value=1,
        value=int(train_cfg.get("total_num_training_episodes", 200)),
        disabled=continue_training,
        help="Ignored when continuing training - use 'Additional training episodes' in sidebar instead"
    )
    save_freq_every_n_episodes = st.number_input(
        "save_freq_every_n_episodes",
        min_value=1,
        value=int(train_cfg.get("save_freq_every_n_episodes", 20)),
        disabled=continue_training,
        help="Save model checkpoint every N episodes"
    )
    eval_freq_every_n_episodes = st.number_input(
        "eval_freq_every_n_episodes",
        min_value=1,
        value=int(train_cfg.get("eval_freq_every_n_episodes", 20)),
        disabled=continue_training,
        help="Run evaluation every N episodes"
    )
    num_eval_episodes = st.number_input(
        "num_eval_episodes",
        min_value=1,
        value=int(train_cfg.get("num_eval_episodes", 10)),
        disabled=continue_training,
        help="Number of episodes to run during each evaluation"
    )
    seed = st.number_input(
        "seed",
        min_value=0,
        value=int(train_cfg.get("seed", 42)),
        disabled=continue_training
    )

    st.markdown("---")
    
    # --- TRAINING CONTROL BUTTONS ---
    run_col, stop_col = st.columns([2, 1])

    # run_btn = st.button("▶️ Run training", key="run_training_btn")
    # stop_btn = st.button("🛑 Stop training", key="stop_training_btn")
    
    # Show the run_btn and stop_btn side by side
    with run_col:
        run_btn = st.button("▶️ Run training", key="run_training_btn")
    with stop_col:
        stop_btn = st.button("🛑 Stop training", key="stop_training_btn")


    # --- TRAINING LAUNCH LOGIC ---
    if run_btn:
        # Check to see if training is in progress:
        if st.session_state['training_in_progress']:
            st.warning("Training is already in progress. Please stop the current training before starting a new one.")
        else:
            st.write("Preparing configuration...")
            # Build updated config with nested structure
            
            # Construct the abx_clinical_reward_penalties_info_dict from structured inputs
            abx_clinical_dict = {
                "clinical_benefit_reward": float(clinical_benefit_reward),
                "clinical_benefit_probability": float(clinical_benefit_probability),
                "clinical_failure_penalty": float(clinical_failure_penalty),
                "clinical_failure_probability": float(clinical_failure_probability),
                "abx_adverse_effects_info": st.session_state.adverse_effects_dict,
            }

            # Build nested config structure matching new system
            # When continuing training, preserve prior experiment's environment/algorithm settings
            if not continue_training:
                config["agent_algorithm"] = config.get("agent_algorithm", {})
                
                # Environment config
                config.setdefault("environment", {})
                config["environment"].update({
                    "num_patients_per_time_step": int(num_patients),
                    "max_time_steps": int(max_steps),
                    "update_visible_AMR_levels_every_n_timesteps": int(update_visible),
                    "add_noise_to_visible_AMR_levels": float(add_noise_visible),
                    "add_bias_to_visible_AMR_levels": float(add_bias_visible),
                    "antibiotics_AMR_dict": st.session_state.antibiotics_dict,
                })
                # Validate and clean crossresistance matrix to only include antibiotics that exist
                current_antibiotics = set(st.session_state.antibiotics_dict.keys())
                cleaned_crossresistance = {}

                # Crossresistance only makes sense with multiple antibiotics
                if len(current_antibiotics) > 1 and st.session_state.crossresistance_matrix:
                    for prescriber_abx, targets in st.session_state.crossresistance_matrix.items():
                        # Only include if prescriber antibiotic still exists
                        if prescriber_abx in current_antibiotics:
                            cleaned_targets = {
                                target_abx: value
                                for target_abx, value in targets.items()
                                if target_abx in current_antibiotics
                            }
                            # Only add prescriber if it has at least one valid target
                            if cleaned_targets:
                                cleaned_crossresistance[prescriber_abx] = cleaned_targets

                # Persist the cleaned matrix back to session_state
                if len(current_antibiotics) > 1:
                    st.session_state.crossresistance_matrix = cleaned_crossresistance
                else:
                    st.session_state.crossresistance_matrix = {}

                # Add crossresistance matrix only if there are valid entries (requires multiple antibiotics)
                if cleaned_crossresistance:
                    config["environment"]["crossresistance_matrix"] = cleaned_crossresistance
                else:
                    # Remove any stale crossresistance matrix when only one or zero antibiotics remain
                    config["environment"].pop("crossresistance_matrix", None)
                
                # Set patient_generator config in the patient_generator section (nested format)
                config.setdefault("patient_generator", {})
                config["patient_generator"] = patient_gen_cfg_updated
                
                # Reward calculator config
                config.setdefault("reward_calculator", {})
                config["reward_calculator"].update({
                    "lambda_weight": float(lambda_weight),
                    "epsilon": float(epsilon),
                    "abx_clinical_reward_penalties_info_dict": abx_clinical_dict,
                })
            
            # Training config (always update, even when continuing)
            config.setdefault("training", {})
            if continue_training:
                # When continuing: override only the seed
                config["training"]["seed"] = int(seed)
            else:
                # When starting fresh: set all training parameters
                config["training"].update({
                    "total_num_training_episodes": int(total_num_training_episodes),
                    "save_freq_every_n_episodes": int(save_freq_every_n_episodes),
                    "eval_freq_every_n_episodes": int(eval_freq_every_n_episodes),
                    "num_eval_episodes": int(num_eval_episodes),
                    "seed": int(seed),
                })
            config["run_name"] = run_name_input
            config.setdefault("config_folder_location", "../")
            config.setdefault("options_folder_location", "../../options")

            # Write config
            config_path = write_config(config, run_name_input)
            st.success(f"Config written to {config_path}")

            # Launch training with live log streaming
            st.write("Running training... this may take a while.")
            log_placeholder = st.empty()
            log_tail_limit = 200  # keep only the most recent lines to avoid bloating the UI
            
            # Build command based on whether continuing training or starting fresh
            if continue_training:
                cmd = [
                    sys.executable, "-u", "-m", "abx_amr_simulator.training.train",
                    "--train-from-prior-results", str((results_dir / prior_experiment).resolve()),
                    "--additional-training-episodes", str(additional_episodes),
                    "--seed", str(seed)
                ]
            else:
                if not config_path.exists():
                    st.error(f"Config file not found: {config_path}")
                    return
                cmd = [
                    sys.executable, "-u", "-m", "abx_amr_simulator.training.train",
                    "--umbrella-config", str(config_path.resolve()),
                    "-p", f"training.seed={int(seed)}"
                ]
            
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            
            # Now that command is built, launch the subprocess, and also set st.session_state to indicate training is running
            st.session_state['training_in_progress'] = True
            st.session_state['training_pid'] = process.pid 
            # Save process reference in session state

            log_lines = []
            try:
                with st.spinner("Training in progress..."):
                    for line in iter(process.stdout.readline, ''):
                        if not line:
                            break
                        clean = line.rstrip()
                        # Drop noisy MKL deprecation warnings from the live view
                        if "Intel MKL WARNING" in clean:
                            continue
                        log_lines.append(clean)
                        # keep tail limited
                        if len(log_lines) > log_tail_limit:
                            log_lines = log_lines[-log_tail_limit:]
                        log_placeholder.code("\n".join(log_lines))
            except Exception as e:
                st.error(f"Error reading process output: {e}")
            process.wait()

            if process.returncode != 0:
                st.error("Training failed. See log below.")
                log_placeholder.code("\n".join(log_lines))
                return
            st.success("Training completed successfully.")
            
            # Reset training state
            st.session_state['training_in_progress'] = False
            st.session_state['training_pid'] = -1
            
    # Now also have to deal with the stop button
    if stop_btn:
        if st.session_state['training_in_progress']:
            pid_to_kill = st.session_state.get('training_pid', None)
            if pid_to_kill is not None:
                try:
                    os.kill(pid_to_kill, signal.SIGTERM)
                    st.success("Training process terminated.")
                except Exception as e:
                    st.error(f"Error terminating training process: {e}")
            else:
                st.error("No training process ID found.")
            # Reset training state
            st.session_state['training_in_progress'] = False
            st.session_state['training_pid'] = -1
        else:
            st.info("No training process is currently running.")

if __name__ == "__main__":
    main()
