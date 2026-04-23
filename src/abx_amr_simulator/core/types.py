"""
Shared type definitions for ABX AMR environment.

This module contains data types and constants used across the environment
to avoid circular import dependencies.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional


@dataclass
class TruePatient:
    """Ground-truth clinical state of a patient, independent of any observation model.

    Holds only the true attribute values as sampled from the patient generator's
    configured distributions. No observed or noisy values are stored here.
    """
    prob_infected: float
    benefit_value_multiplier: float
    failure_value_multiplier: float
    benefit_probability_multiplier: float
    failure_probability_multiplier: float
    recovery_without_treatment_prob: float
    infection_status: bool
    abx_sensitivity_dict: Dict[str, bool]


@dataclass
class ObservedPatient:
    """Observed state of a patient under a specific observation model.

    Only attributes that were visible under the producing observation model are
    present in visible_attributes. Attributes not observable under this model are
    simply absent — not padded with a sentinel value.

    Attributes:
        true_patient: The underlying ground-truth patient state.
        visible_attributes: Mapping of attribute name to observed value for each
            attribute visible under this observation model.
        observation_model_label: Optional label identifying which observation model
            produced this instance (e.g. "covered", "uncovered", "full_visibility").
    """
    true_patient: TruePatient
    visible_attributes: Dict[str, float]
    observation_model_label: Optional[str] = None


@dataclass
class Patient:
    """Compatibility container wrapping TruePatient and ObservedPatient.

    Provides property shims that match the old flat-field Patient interface so
    existing call sites continue to work while the codebase migrates to using
    TruePatient / ObservedPatient directly.

    The _obs property shims return PADDING_VALUE (-1.0) for attributes absent
    from primary_observation.visible_attributes, preserving prior sentinel-value
    behavior for heterogeneous-visibility populations.

    Attributes:
        true_state: Ground-truth clinical state of this patient.
        observations: ObservedPatient instances representing this patient through
            one or more observation models. Most patients have exactly one.
        patient_id: Optional identifier for multi-agent tracking across timesteps.
        treated_by_agent: Agent ID who prescribed treatment (multi-agent use).
        treated_in_locale: Locale ID where treatment occurred (multi-agent use).
        origin_locale: Locale ID where patient was sampled (multi-agent use).
        source_generator_index: Sub-generator index within a PatientGeneratorMixer,
            used for provenance tracking.
    """
    true_state: TruePatient
    observations: List[ObservedPatient] = field(default_factory=list)

    # Multi-agent tracking fields (preserved from prior Patient definition)
    patient_id: Optional[str] = None
    treated_by_agent: Optional[str] = None
    treated_in_locale: Optional[str] = None
    origin_locale: Optional[str] = None
    source_generator_index: Optional[int] = None

    PADDING_VALUE: ClassVar[float] = -1.0

    @property
    def primary_observation(self) -> ObservedPatient:
        """Return the first (primary) observation for this patient.

        Raises:
            AttributeError: If no observations have been attached yet.
        """
        if not self.observations:
            raise AttributeError("Patient has no observations attached")
        return self.observations[0]

    # ------------------------------------------------------------------ #
    # True-state shims — read-through to true_state for backward compat   #
    # ------------------------------------------------------------------ #

    @property
    def prob_infected(self) -> float:
        return self.true_state.prob_infected

    @property
    def benefit_value_multiplier(self) -> float:
        return self.true_state.benefit_value_multiplier

    @property
    def failure_value_multiplier(self) -> float:
        return self.true_state.failure_value_multiplier

    @property
    def benefit_probability_multiplier(self) -> float:
        return self.true_state.benefit_probability_multiplier

    @property
    def failure_probability_multiplier(self) -> float:
        return self.true_state.failure_probability_multiplier

    @property
    def recovery_without_treatment_prob(self) -> float:
        return self.true_state.recovery_without_treatment_prob

    @property
    def infection_status(self) -> bool:
        return self.true_state.infection_status

    @property
    def abx_sensitivity_dict(self) -> Dict[str, bool]:
        return self.true_state.abx_sensitivity_dict

    # ------------------------------------------------------------------ #
    # Observed-value shims — read from primary_observation.visible_attrs  #
    # Returns PADDING_VALUE for attributes absent from the obs model.     #
    # ------------------------------------------------------------------ #

    @property
    def prob_infected_obs(self) -> float:
        return self.primary_observation.visible_attributes.get(
            'prob_infected', self.PADDING_VALUE
        )

    @property
    def benefit_value_multiplier_obs(self) -> float:
        return self.primary_observation.visible_attributes.get(
            'benefit_value_multiplier', self.PADDING_VALUE
        )

    @property
    def failure_value_multiplier_obs(self) -> float:
        return self.primary_observation.visible_attributes.get(
            'failure_value_multiplier', self.PADDING_VALUE
        )

    @property
    def benefit_probability_multiplier_obs(self) -> float:
        return self.primary_observation.visible_attributes.get(
            'benefit_probability_multiplier', self.PADDING_VALUE
        )

    @property
    def failure_probability_multiplier_obs(self) -> float:
        return self.primary_observation.visible_attributes.get(
            'failure_probability_multiplier', self.PADDING_VALUE
        )

    @property
    def recovery_without_treatment_prob_obs(self) -> float:
        return self.primary_observation.visible_attributes.get(
            'recovery_without_treatment_prob', self.PADDING_VALUE
        )
