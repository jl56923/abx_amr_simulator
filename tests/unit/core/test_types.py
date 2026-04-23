"""Unit tests for TruePatient, ObservedPatient, and Patient type hierarchy."""

import pytest
from abx_amr_simulator.core.types import ObservedPatient, Patient, TruePatient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_true_patient(**overrides) -> TruePatient:
    defaults = dict(
        prob_infected=0.6,
        benefit_value_multiplier=1.0,
        failure_value_multiplier=1.0,
        benefit_probability_multiplier=1.0,
        failure_probability_multiplier=1.0,
        recovery_without_treatment_prob=0.2,
        infection_status=True,
        abx_sensitivity_dict={"abx_a": True, "abx_b": False},
    )
    defaults.update(overrides)
    return TruePatient(**defaults)


def make_observed_patient(true_patient=None, **visible_attrs) -> ObservedPatient:
    tp = true_patient or make_true_patient()
    visible = visible_attrs if visible_attrs else {
        "prob_infected": 0.55,
        "benefit_value_multiplier": 1.05,
    }
    return ObservedPatient(true_patient=tp, visible_attributes=visible)


def make_patient(true_patient=None, observations=None, **kwargs) -> Patient:
    tp = true_patient or make_true_patient()
    obs = observations if observations is not None else [make_observed_patient(tp)]
    return Patient(true_state=tp, observations=obs, **kwargs)


# ---------------------------------------------------------------------------
# TruePatient
# ---------------------------------------------------------------------------

class TestTruePatient:
    def test_construction_with_all_fields(self):
        tp = make_true_patient()
        assert tp.prob_infected == 0.6
        assert tp.benefit_value_multiplier == 1.0
        assert tp.failure_value_multiplier == 1.0
        assert tp.benefit_probability_multiplier == 1.0
        assert tp.failure_probability_multiplier == 1.0
        assert tp.recovery_without_treatment_prob == 0.2
        assert tp.infection_status is True
        assert tp.abx_sensitivity_dict == {"abx_a": True, "abx_b": False}

    def test_field_overrides(self):
        tp = make_true_patient(prob_infected=0.9, infection_status=False)
        assert tp.prob_infected == 0.9
        assert tp.infection_status is False

    def test_no_obs_fields(self):
        tp = make_true_patient()
        assert not hasattr(tp, 'prob_infected_obs')
        assert not hasattr(tp, 'benefit_value_multiplier_obs')

    def test_equality(self):
        tp1 = make_true_patient()
        tp2 = make_true_patient()
        assert tp1 == tp2

    def test_inequality_on_differing_field(self):
        tp1 = make_true_patient(prob_infected=0.6)
        tp2 = make_true_patient(prob_infected=0.7)
        assert tp1 != tp2


# ---------------------------------------------------------------------------
# ObservedPatient
# ---------------------------------------------------------------------------

class TestObservedPatient:
    def test_construction(self):
        tp = make_true_patient()
        op = ObservedPatient(
            true_patient=tp,
            visible_attributes={"prob_infected": 0.55, "benefit_value_multiplier": 1.1},
        )
        assert op.true_patient is tp
        assert op.visible_attributes["prob_infected"] == 0.55
        assert op.visible_attributes["benefit_value_multiplier"] == 1.1
        assert op.observation_model_label is None

    def test_observation_model_label(self):
        op = make_observed_patient()
        op2 = ObservedPatient(
            true_patient=op.true_patient,
            visible_attributes=op.visible_attributes,
            observation_model_label="covered",
        )
        assert op2.observation_model_label == "covered"

    def test_partial_visibility(self):
        tp = make_true_patient()
        op = ObservedPatient(
            true_patient=tp,
            visible_attributes={"prob_infected": 0.5},  # only 1 of 6 attrs
        )
        assert "prob_infected" in op.visible_attributes
        assert "benefit_value_multiplier" not in op.visible_attributes

    def test_empty_visible_attributes(self):
        tp = make_true_patient()
        op = ObservedPatient(true_patient=tp, visible_attributes={})
        assert op.visible_attributes == {}


# ---------------------------------------------------------------------------
# Patient — construction and tracking fields
# ---------------------------------------------------------------------------

class TestPatientConstruction:
    def test_basic_construction(self):
        tp = make_true_patient()
        op = make_observed_patient(tp)
        p = Patient(true_state=tp, observations=[op])
        assert p.true_state is tp
        assert len(p.observations) == 1
        assert p.observations[0] is op

    def test_default_observations_is_empty_list(self):
        tp = make_true_patient()
        p = Patient(true_state=tp)
        assert p.observations == []

    def test_multi_agent_tracking_fields_default_to_none(self):
        p = make_patient()
        assert p.patient_id is None
        assert p.treated_by_agent is None
        assert p.treated_in_locale is None
        assert p.origin_locale is None
        assert p.source_generator_index is None

    def test_multi_agent_tracking_fields_set(self):
        p = make_patient(
            patient_id="P001",
            treated_by_agent="agent_0",
            treated_in_locale="locale_A",
            origin_locale="locale_B",
            source_generator_index=1,
        )
        assert p.patient_id == "P001"
        assert p.treated_by_agent == "agent_0"
        assert p.treated_in_locale == "locale_A"
        assert p.origin_locale == "locale_B"
        assert p.source_generator_index == 1

    def test_padding_value_class_var(self):
        assert Patient.PADDING_VALUE == -1.0

    def test_padding_value_not_in_init(self):
        # PADDING_VALUE is a ClassVar — must not appear as an __init__ parameter
        import inspect
        sig = inspect.signature(Patient.__init__)
        assert 'PADDING_VALUE' not in sig.parameters


# ---------------------------------------------------------------------------
# Patient — primary_observation
# ---------------------------------------------------------------------------

class TestPatientPrimaryObservation:
    def test_returns_first_observation(self):
        tp = make_true_patient()
        op1 = make_observed_patient(tp, prob_infected=0.5)
        op2 = make_observed_patient(tp, prob_infected=0.8)
        p = Patient(true_state=tp, observations=[op1, op2])
        assert p.primary_observation is op1

    def test_raises_when_no_observations(self):
        tp = make_true_patient()
        p = Patient(true_state=tp, observations=[])
        with pytest.raises(AttributeError, match="no observations attached"):
            _ = p.primary_observation


# ---------------------------------------------------------------------------
# Patient — true-state shims
# ---------------------------------------------------------------------------

class TestPatientTrueStateShims:
    def test_prob_infected(self):
        p = make_patient(true_patient=make_true_patient(prob_infected=0.75))
        assert p.prob_infected == 0.75

    def test_benefit_value_multiplier(self):
        p = make_patient(true_patient=make_true_patient(benefit_value_multiplier=2.0))
        assert p.benefit_value_multiplier == 2.0

    def test_failure_value_multiplier(self):
        p = make_patient(true_patient=make_true_patient(failure_value_multiplier=1.5))
        assert p.failure_value_multiplier == 1.5

    def test_benefit_probability_multiplier(self):
        p = make_patient(true_patient=make_true_patient(benefit_probability_multiplier=0.8))
        assert p.benefit_probability_multiplier == 0.8

    def test_failure_probability_multiplier(self):
        p = make_patient(true_patient=make_true_patient(failure_probability_multiplier=1.2))
        assert p.failure_probability_multiplier == 1.2

    def test_recovery_without_treatment_prob(self):
        p = make_patient(true_patient=make_true_patient(recovery_without_treatment_prob=0.3))
        assert p.recovery_without_treatment_prob == 0.3

    def test_infection_status(self):
        p = make_patient(true_patient=make_true_patient(infection_status=False))
        assert p.infection_status is False

    def test_abx_sensitivity_dict(self):
        sens = {"drug_x": True, "drug_y": False}
        p = make_patient(true_patient=make_true_patient(abx_sensitivity_dict=sens))
        assert p.abx_sensitivity_dict == sens

    def test_shims_are_read_through_not_copies(self):
        tp = make_true_patient()
        p = Patient(true_state=tp, observations=[make_observed_patient(tp)])
        assert p.prob_infected is tp.prob_infected


# ---------------------------------------------------------------------------
# Patient — observed-value shims
# ---------------------------------------------------------------------------

class TestPatientObsShims:
    def test_present_attr_returns_observed_value(self):
        tp = make_true_patient(prob_infected=0.6)
        op = ObservedPatient(
            true_patient=tp,
            visible_attributes={
                "prob_infected": 0.55,
                "benefit_value_multiplier": 1.1,
                "failure_value_multiplier": 0.9,
                "benefit_probability_multiplier": 1.05,
                "failure_probability_multiplier": 0.95,
                "recovery_without_treatment_prob": 0.22,
            },
        )
        p = Patient(true_state=tp, observations=[op])
        assert p.prob_infected_obs == 0.55
        assert p.benefit_value_multiplier_obs == 1.1
        assert p.failure_value_multiplier_obs == 0.9
        assert p.benefit_probability_multiplier_obs == 1.05
        assert p.failure_probability_multiplier_obs == 0.95
        assert p.recovery_without_treatment_prob_obs == 0.22

    def test_absent_attr_returns_padding_value(self):
        tp = make_true_patient()
        op = ObservedPatient(
            true_patient=tp,
            visible_attributes={"prob_infected": 0.5},  # only 1 attr visible
        )
        p = Patient(true_state=tp, observations=[op])
        assert p.benefit_value_multiplier_obs == Patient.PADDING_VALUE
        assert p.failure_value_multiplier_obs == Patient.PADDING_VALUE
        assert p.benefit_probability_multiplier_obs == Patient.PADDING_VALUE
        assert p.failure_probability_multiplier_obs == Patient.PADDING_VALUE
        assert p.recovery_without_treatment_prob_obs == Patient.PADDING_VALUE

    def test_all_attrs_absent_all_return_padding(self):
        tp = make_true_patient()
        op = ObservedPatient(true_patient=tp, visible_attributes={})
        p = Patient(true_state=tp, observations=[op])
        assert p.prob_infected_obs == Patient.PADDING_VALUE
        assert p.benefit_value_multiplier_obs == Patient.PADDING_VALUE
        assert p.failure_value_multiplier_obs == Patient.PADDING_VALUE
        assert p.benefit_probability_multiplier_obs == Patient.PADDING_VALUE
        assert p.failure_probability_multiplier_obs == Patient.PADDING_VALUE
        assert p.recovery_without_treatment_prob_obs == Patient.PADDING_VALUE

    def test_padding_value_is_minus_one(self):
        assert Patient.PADDING_VALUE == -1.0

    def test_obs_shim_raises_when_no_observation(self):
        tp = make_true_patient()
        p = Patient(true_state=tp, observations=[])
        with pytest.raises(AttributeError):
            _ = p.prob_infected_obs
