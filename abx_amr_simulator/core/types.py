"""
Shared type definitions for ABX AMR environment.

This module contains data types and constants used across the environment
to avoid circular import dependencies.
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class Patient:
    """
    Represents a patient with stochastic attributes.
    
    Attributes:
        prob_infected: True probability that patient is infected (float in [0, 1])
        benefit_value_multiplier: Multiplicative scaling on clinical benefit reward (float > 0)
        failure_value_multiplier: Multiplicative scaling on clinical failure penalty (float > 0)
        benefit_probability_multiplier: Multiplicative scaling on clinical benefit probability (float > 0)
        failure_probability_multiplier: Multiplicative scaling on clinical failure probability (float > 0)
        recovery_without_treatment_prob: Probability of spontaneous recovery without treatment (float in [0, 1])
        infection_status: True infection status for this patient (bool)
        abx_sensitivity_dict: Mapping of antibiotic name → True if infection is sensitive (bool)
        prob_infected_obs: Observed infection probability with noise/bias (float in [0, 1])
        benefit_value_multiplier_obs: Observed benefit value multiplier with noise/bias (float > 0)
        failure_value_multiplier_obs: Observed failure value multiplier with noise/bias (float > 0)
        benefit_probability_multiplier_obs: Observed benefit probability multiplier with noise/bias (float > 0)
        failure_probability_multiplier_obs: Observed failure probability multiplier with noise/bias (float > 0)
        recovery_without_treatment_prob_obs: Observed recovery probability with noise/bias (float in [0, 1])
        
        Multi-agent tracking fields (for future PettingZoo multi-agent support):
        patient_id: Unique identifier for tracking across timesteps (optional)
        treated_by_agent: Agent ID who prescribed treatment (optional)
        treated_in_locale: Locale ID where treatment occurred (optional)
        origin_locale: Locale ID where patient was sampled/originated (optional, for travel/mixing)
    """
    prob_infected: float
    benefit_value_multiplier: float
    failure_value_multiplier: float
    benefit_probability_multiplier: float
    failure_probability_multiplier: float
    recovery_without_treatment_prob: float
    infection_status: bool
    abx_sensitivity_dict: Dict[str, bool]
    prob_infected_obs: float
    benefit_value_multiplier_obs: float
    failure_value_multiplier_obs: float
    benefit_probability_multiplier_obs: float
    failure_probability_multiplier_obs: float
    recovery_without_treatment_prob_obs: float
    
    # Multi-agent tracking (unused in single-agent mode but ready for future)
    patient_id: Optional[str] = None
    treated_by_agent: Optional[str] = None
    treated_in_locale: Optional[str] = None
    origin_locale: Optional[str] = None
    # Source generator index (for PatientGeneratorMixer provenance and visibility padding)
    source_generator_index: Optional[int] = None
