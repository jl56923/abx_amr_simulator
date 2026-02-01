"""
Helper functions for Patient Generator UI in experiment_runner.py

Converts between nested per-attribute config format and old flat format.
"""

def migrate_old_config_to_new(old_cfg: dict) -> dict:
    """
    Convert old flat patient_generator config to new nested per-attribute format.
    
    Old format example:
    {
        'prob_infected_dist': {'type': 'constant', 'value': 0.5},
        'prob_infected_observation_noise': 0.05,
        'prob_infected_observation_bias': 1.0,
        ...
    }
    
    New format example:
    {
        'prob_infected': {
            'prob_dist': {'type': 'constant', 'value': 0.5},
            'obs_bias_multiplier': 1.0,
            'obs_noise_one_std_dev': 0.2,
            'obs_noise_std_dev_fraction': 0.5,
            'clipping_bounds': [0.0, 1.0],
        },
        ...
        'visible_patient_attributes': ['prob_infected', ...]
    }
    """
    # If already in new format, return as-is
    if 'prob_infected' in old_cfg and isinstance(old_cfg.get('prob_infected'), dict):
        if 'prob_dist' in old_cfg['prob_infected']:
            return old_cfg
    
    # Map old flat keys to new nested structure
    attribute_types = {
        'prob_infected': {
            'prob_dist_key': 'prob_infected_dist',
            'obs_bias_key': 'prob_infected_observation_bias',
            'obs_noise_key': 'prob_infected_observation_noise',
            'clipping_bounds': [0.0, 1.0],
            'obs_noise_one_std_dev': 0.2,  # Default
        },
        'benefit_value_multiplier': {
            'prob_dist_key': 'benefit_value_multiplier_dist',
            'obs_bias_key': 'benefit_value_multiplier_observation_bias',
            'obs_noise_key': 'benefit_value_multiplier_observation_noise',
            'clipping_bounds': [0.0, None],
            'obs_noise_one_std_dev': 1.0,  # Default
        },
        'failure_value_multiplier': {
            'prob_dist_key': 'failure_value_multiplier_dist',
            'obs_bias_key': 'failure_value_multiplier_observation_bias',
            'obs_noise_key': 'failure_value_multiplier_observation_noise',
            'clipping_bounds': [0.0, None],
            'obs_noise_one_std_dev': 1.0,
        },
        'benefit_probability_multiplier': {
            'prob_dist_key': 'benefit_probability_multiplier_dist',
            'obs_bias_key': 'benefit_probability_multiplier_observation_bias',
            'obs_noise_key': 'benefit_probability_multiplier_observation_noise',
            'clipping_bounds': [0.0, None],
            'obs_noise_one_std_dev': 1.0,
        },
        'failure_probability_multiplier': {
            'prob_dist_key': 'failure_probability_multiplier_dist',
            'obs_bias_key': 'failure_probability_multiplier_observation_bias',
            'obs_noise_key': 'failure_probability_multiplier_observation_noise',
            'clipping_bounds': [0.0, None],
            'obs_noise_one_std_dev': 1.0,
        },
        'recovery_without_treatment_prob': {
            'prob_dist_key': 'recovery_without_treatment_prob_dist',
            'obs_bias_key': 'recovery_without_treatment_prob_observation_bias',
            'obs_noise_key': 'recovery_without_treatment_prob_observation_noise',
            'clipping_bounds': [0.0, 1.0],
            'obs_noise_one_std_dev': 0.2,
        },
    }
    
    new_cfg = {}
    
    for attr_name, attr_info in attribute_types.items():
        prob_dist = old_cfg.get(attr_info['prob_dist_key'], {'type': 'gaussian'})
        obs_bias = old_cfg.get(attr_info['obs_bias_key'], 1.0)
        obs_noise_old = old_cfg.get(attr_info['obs_noise_key'], 0.0)
        
        # Convert old noise format (absolute std) to fraction of range
        # For now, assume obs_noise_one_std_dev from config, compute fraction
        obs_noise_one_std_dev = attr_info['obs_noise_one_std_dev']
        if obs_noise_one_std_dev > 0 and obs_noise_old > 0:
            obs_noise_fraction = obs_noise_old / obs_noise_one_std_dev
        else:
            obs_noise_fraction = 0.0
        
        new_cfg[attr_name] = {
            'prob_dist': prob_dist,
            'obs_bias_multiplier': obs_bias,
            'obs_noise_one_std_dev': obs_noise_one_std_dev,
            'obs_noise_std_dev_fraction': obs_noise_fraction,
            'clipping_bounds': attr_info['clipping_bounds'],
        }
    
    # Preserve visible_patient_attributes if present
    if 'visible_patient_attributes' in old_cfg:
        new_cfg['visible_patient_attributes'] = old_cfg['visible_patient_attributes']
    else:
        new_cfg['visible_patient_attributes'] = ['prob_infected']
    
    return new_cfg


def build_attribute_ui_section(st_module, attr_name: str, attr_cfg: dict, continue_training: bool, 
                                attr_display_name: str, min_bounds: float = 0.0, max_bounds: float = None):
    """
    Build Streamlit UI section for a single attribute in nested format.
    
    Returns a dict with the updated configuration for this attribute.
    """
    with st_module.expander(f"🧬 {attr_display_name}", expanded=False):
        st_module.markdown("**Probability Distribution**")
        col1, col2, col3 = st_module.columns(3)
        
        # Probability distribution type
        with col1:
            prob_dist_type = attr_cfg.get('prob_dist', {}).get('type', 'gaussian')
            dist_type = st_module.selectbox(
                "Distribution type",
                ["constant", "gaussian"],
                index=0 if prob_dist_type == "constant" else 1,
                disabled=continue_training,
                key=f"attr_{attr_name}_dist_type"
            )
        
        # Value/mean input
        with col2:
            if dist_type == "constant":
                value = st_module.number_input(
                    "Value",
                    value=float(attr_cfg.get('prob_dist', {}).get('value', 0.5)),
                    step=0.01,
                    disabled=continue_training,
                    key=f"attr_{attr_name}_value"
                )
            else:
                mu = st_module.number_input(
                    "Mean (μ)",
                    value=float(attr_cfg.get('prob_dist', {}).get('mu', 0.5)),
                    step=0.01,
                    disabled=continue_training,
                    key=f"attr_{attr_name}_mu"
                )
        
        # Std dev (only for gaussian)
        with col3:
            if dist_type != "constant":
                sigma = st_module.number_input(
                    "Std dev (σ)",
                    min_value=0.0,
                    value=float(attr_cfg.get('prob_dist', {}).get('sigma', 0.2)),
                    step=0.01,
                    disabled=continue_training,
                    key=f"attr_{attr_name}_sigma"
                )
        
        st_module.markdown("**Observation Settings**")
        col1, col2 = st_module.columns(2)
        
        with col1:
            obs_bias = st_module.number_input(
                "Observation bias (multiplicative)",
                value=float(attr_cfg.get('obs_bias_multiplier', 1.0)),
                step=0.01,
                disabled=continue_training,
                key=f"attr_{attr_name}_obs_bias"
            )
        
        with col2:
            obs_noise_fraction = st_module.number_input(
                "Observation noise (fraction of range)",
                min_value=0.0,
                max_value=1.0,
                value=float(attr_cfg.get('obs_noise_std_dev_fraction', 0.0)),
                step=0.01,
                disabled=continue_training,
                key=f"attr_{attr_name}_obs_noise_frac",
                help="As a percentage of the attribute's range (e.g., 0.2 = 20% of range)"
            )
        
        st_module.markdown("**Clipping Bounds**")
        col1, col2 = st_module.columns(2)
        
        clipping = attr_cfg.get('clipping_bounds', [min_bounds, max_bounds])
        with col1:
            lower_bound = st_module.number_input(
                "Lower bound",
                value=float(clipping[0]) if clipping and len(clipping) > 0 else min_bounds,
                step=0.01,
                disabled=continue_training,
                key=f"attr_{attr_name}_lower_bound"
            )
        
        with col2:
            upper_bound_val = clipping[1] if clipping and len(clipping) > 1 else max_bounds
            upper_bound = st_module.number_input(
                "Upper bound (leave empty for unbounded)",
                value=float(upper_bound_val) if upper_bound_val is not None else 1000.0,
                step=0.01,
                disabled=continue_training,
                key=f"attr_{attr_name}_upper_bound"
            )
        
        # Build the attribute config dict for this attribute
        attr_config_dict = {
            'prob_dist': {
                'type': dist_type,
            },
            'obs_bias_multiplier': obs_bias,
            'obs_noise_std_dev_fraction': obs_noise_fraction,
            'obs_noise_one_std_dev': attr_cfg.get('obs_noise_one_std_dev', 0.2),  # Keep ref value
            'clipping_bounds': [lower_bound, upper_bound if upper_bound_val is not None else None],
        }
        
        # Add distribution-specific params
        if dist_type == "constant":
            attr_config_dict['prob_dist']['value'] = value
        else:
            attr_config_dict['prob_dist']['mu'] = mu
            attr_config_dict['prob_dist']['sigma'] = sigma
        
        return attr_config_dict
