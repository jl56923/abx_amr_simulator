"""
Custom formatters for StepPrinter to translate raw step data into human-readable interpretations.

Example:
    env = StepPrinter(
        env,
        formatter=abx_amr_step_formatter,
        log_path="run.log",
        log_all_steps=True,
    )
"""

from __future__ import annotations

from typing import Any, Callable


def abx_amr_step_formatter(prev_obs: Any, action: Any, reward: Any, terminated: Any, truncated: Any, info: Any, action_idx_to_abx_name: dict = None) -> str:
    """
    Interprets ABX-AMR env step data into human-readable format.
    
    Parameters
    - prev_obs: observation before the action (what the agent acted on)
    - action, reward, terminated, truncated, info: outputs from env.step(...)
    - action_idx_to_abx_name: dict mapping action indices to antibiotic names, e.g. {0: "Antibiotic_A", 1: "no_treatment"}
    
    Returns a string like:
        "Agent prescribed: [Antibiotic_A, no_treatment] | Infected patients: [1, 1] | R=4.999 | Actual AMR levels: [0.15], Visible AMR levels: [0.10]"
    """
    if action_idx_to_abx_name is None:
        action_idx_to_abx_name = {0: "Antibiotic_A", 1: "no_treatment"}  # Fallback mapping

    try:
        # Get action interpretation (antibiotic names)
        action_names = []
        if hasattr(action, '__iter__'):
            for act_idx in action:
                action_names.append(action_idx_to_abx_name.get(int(act_idx), f"action_{int(act_idx)}"))
        else:
            action_names.append(action_idx_to_abx_name.get(int(action), f"action_{int(action)}"))

        # Get infection status
        infected = info.get("patients_actually_infected", [])
        
        # Format reward
        r_str = f"{float(reward):.3f}"
        
        # Get AMR levels if available
        actual_amr_levels = info.get("actual_amr_levels", None)
        visible_amr_levels = info.get("visible_amr_levels", None)
        
        # Get counts of which clinical outcomes occurred:
        num_clinical_benefits = info.get("count_clinical_benefits", 0)
        num_clinical_failures = info.get("count_clinical_failures", 0)
        num_adverse_events = info.get("count_adverse_events", 0)
        
        # Build summary
        term_indicator = "TERM" if terminated else ""
        trunc_indicator = "TRUNC" if truncated else ""
        status = f"{term_indicator} {trunc_indicator}".strip()
        
        summary = f"Prev obs: Infected={infected} | Action: Rx={action_names} | Clinical Benefits={num_clinical_benefits}, Clinical Failures={num_clinical_failures}, Adverse Events={num_adverse_events} | Reward={r_str} | Actual AMR levels: {actual_amr_levels}, Visible AMR levels: {visible_amr_levels}"
        if status:
            summary += f" | {status}"
        
        return summary
    except Exception as e:
        # Fallback: return minimal info if formatter fails
        return f"step_formatter_error: {type(e).__name__}"


def minimal_step_formatter(prev_obs: Any, action: Any, reward: Any, terminated: Any, truncated: Any, info: Any) -> str:
    """
    Minimal formatter: just action and reward for quiet logging.
    """
    try:
        r_str = f"{float(reward):.3f}"
        return f"a={action} r={r_str}"
    except Exception:
        return "formatter_error"
