"""Regression tests for eng_2 (uncertainty-gate-injection).

Before the fix, the relative-uncertainty gate could NEVER fire in a real run:

  1. ``OptionLibrary`` injected ``visible_patient_attributes`` into each worker, but a
     visible attribute is by construction never masked; and
  2. ``compute_relative_uncertainty_score`` only counted an *explicit* ``-1.0``, while the
     runtime patient dict built by ``OptionsWrapper`` contains only the visible attributes
     (with real values) and simply omits the masked ones -- it never pads with ``-1``.

So the score was always 0 and the gate never refused, at any threshold. The fix injects the
population's *configured* attribute set and counts an *absent* attribute as unobserved. These
tests exercise the real scorer, a real ``PatientGenerator`` / ``ABXAMREnv`` / ``OptionLibrary``
/ ``OptionsWrapper`` -- no mocks -- and would fail on the pre-fix code.
"""

import pytest

from abx_amr_simulator.core.patient_generator import PatientGeneratorMixer
from abx_amr_simulator.hrl import OptionLibrary, OptionsWrapper
from abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader import (
    HeuristicWorker,
)

# tests/unit/utils is on sys.path via tests/conftest.py
from test_reference_helpers import make_env  # type: ignore[import-not-found]


ALL_SIX_ATTRS = {
    'prob_infected',
    'benefit_value_multiplier',
    'failure_value_multiplier',
    'benefit_probability_multiplier',
    'failure_probability_multiplier',
    'recovery_without_treatment_prob',
}


def _make_worker(uncertainty_threshold: float) -> HeuristicWorker:
    """A permissive heuristic worker whose only gating knob is the uncertainty threshold."""
    return HeuristicWorker(
        name='HEURISTIC_gate_10',
        duration=10,
        action_thresholds={'prescribe_A': 0.0, 'no_treatment': 0.0},
        uncertainty_threshold=uncertainty_threshold,
        default_recovery_without_treatment_prob=0.1,
    )


# --------------------------------------------------------------------------------------
# Unit tests: the scorer itself (Change 2 -- absent counts as unobserved)
# --------------------------------------------------------------------------------------

class TestRelativeScorerCountsAbsent:

    def test_absent_attributes_are_counted(self):
        """The runtime case: masked attributes are ABSENT (not -1) and must still count.

        This is the direct regression for the bug -- on the pre-fix code this scored 0.
        """
        worker = _make_worker(uncertainty_threshold=2.0)
        worker.set_observable_attributes(sorted(ALL_SIX_ATTRS))

        # Only prob_infected observed; the other five are simply absent (no -1 padding).
        patient = {'prob_infected': 0.8}

        assert worker.compute_relative_uncertainty_score(patient=patient) == 5

    def test_absent_and_explicit_minus_one_both_count(self):
        """Absent OR explicit -1 both mean 'unobserved'."""
        worker = _make_worker(uncertainty_threshold=2.0)
        worker.set_observable_attributes(sorted(ALL_SIX_ATTRS))

        patient = {
            'prob_infected': 0.8,                 # observed
            'benefit_value_multiplier': -1.0,     # explicitly padded
            'failure_value_multiplier': 0.9,      # observed
            # remaining three attributes absent
        }
        # 1 explicit -1 + 3 absent = 4 unobserved
        assert worker.compute_relative_uncertainty_score(patient=patient) == 4

    def test_explicit_minus_one_still_counts_regression(self):
        """Backward-compat: a fully-populated dict with explicit -1s behaves as before."""
        worker = _make_worker(uncertainty_threshold=2.0)
        worker.set_observable_attributes(sorted(ALL_SIX_ATTRS))

        patient = {
            'prob_infected': 0.8,
            'benefit_value_multiplier': -1.0,
            'failure_value_multiplier': -1.0,
            'benefit_probability_multiplier': 1.0,
            'failure_probability_multiplier': 1.0,
            'recovery_without_treatment_prob': 0.1,
        }
        assert worker.compute_relative_uncertainty_score(patient=patient) == 2

    def test_full_visibility_scores_zero(self):
        """Control: nothing masked -> score 0 -> the gate can never fire (protects EoID-style
        full-visibility configs from newly abstaining)."""
        worker = _make_worker(uncertainty_threshold=0.0)  # strictest possible
        worker.set_observable_attributes(sorted(ALL_SIX_ATTRS))

        patient = {attr: 0.5 for attr in ALL_SIX_ATTRS}  # all observed, real values
        assert worker.compute_relative_uncertainty_score(patient=patient) == 0


# --------------------------------------------------------------------------------------
# Injection test (Change 1 -- OptionLibrary injects the CONFIGURED set, not just visible)
# --------------------------------------------------------------------------------------

class TestOptionLibraryInjectsConfiguredSet:

    def test_injection_uses_configured_not_visible(self):
        """Building an OptionsWrapper injects the population's configured attributes (a superset
        of visible) into the worker -- on the pre-fix code only ['prob_infected'] was injected."""
        env = make_env(
            antibiotic_names=['A'],
            num_patients_per_time_step=1,
            visible_patient_attributes=['prob_infected'],
        )
        pg = env.unwrapped.patient_generator
        worker = _make_worker(uncertainty_threshold=99.0)
        lib = OptionLibrary.from_env(env)
        lib.add_option(worker)

        # OptionsWrapper.__init__ calls validate_environment_compatibility, which injects.
        OptionsWrapper(env=env, option_library=lib, gamma=0.99)

        assert set(worker._observable_patient_attributes) == set(pg.attribute_configs.keys())
        assert set(worker._observable_patient_attributes) == ALL_SIX_ATTRS
        # The whole point: the injected set is strictly larger than what the agent can see.
        assert set(pg.visible_patient_attributes) == {'prob_infected'}
        assert len(worker._observable_patient_attributes) == 6


# --------------------------------------------------------------------------------------
# Wired end-to-end test: the gate fires through the real wrapper runtime path
# --------------------------------------------------------------------------------------

class TestGateFiresEndToEnd:

    def _build(self, visible, uncertainty_threshold, num_patients=3):
        env = make_env(
            antibiotic_names=['A'],
            num_patients_per_time_step=num_patients,
            visible_patient_attributes=visible,
        )
        worker = _make_worker(uncertainty_threshold=uncertainty_threshold)
        lib = OptionLibrary.from_env(env)
        lib.add_option(worker)
        wrapper = OptionsWrapper(env=env, option_library=lib, gamma=0.99)
        obs, _ = wrapper.reset()
        env_state = wrapper._build_env_state(obs)
        return env, worker, wrapper, env_state

    def test_limited_visibility_gate_fires(self):
        """visible = {prob_infected}, 5 masked. The runtime dict really omits the masked
        attributes; the wired score is 5; and with threshold 2 (< 5) the gate forces
        no_treatment for every patient -- reward-independent, since the uncertainty check
        short-circuits before action selection."""
        env, worker, wrapper, env_state = self._build(
            visible=['prob_infected'], uncertainty_threshold=2.0,
        )

        # The crux of the bug: the runtime patient dict carries only the visible attribute.
        assert set(env_state['patients'][0].keys()) == {'prob_infected'}

        # Wired score through the real dict + injected configured list.
        assert worker.compute_relative_uncertainty_score(env_state['patients'][0]) == 5

        # 5 > 2 -> the gate fires for every patient.
        actions = worker.decide(env_state)
        assert list(actions) == ['no_treatment'] * env_state['num_patients']

    def test_threshold_above_masked_count_does_not_gate(self):
        """Same masked population, but threshold 99 (> 5): the uncertainty gate does NOT force
        abstention. Asserted on the score/threshold relation to stay reward-independent."""
        env, worker, wrapper, env_state = self._build(
            visible=['prob_infected'], uncertainty_threshold=99.0,
        )
        score = worker.compute_relative_uncertainty_score(env_state['patients'][0])
        assert score == 5
        assert not (score > worker.uncertainty_threshold)  # gate would not fire

    def test_full_visibility_runtime_score_zero(self):
        """visible = all six -> the runtime dict is complete -> score 0 -> the gate cannot fire
        even at the strictest threshold."""
        env, worker, wrapper, env_state = self._build(
            visible=sorted(ALL_SIX_ATTRS), uncertainty_threshold=0.0,
        )
        assert set(env_state['patients'][0].keys()) == ALL_SIX_ATTRS
        assert worker.compute_relative_uncertainty_score(env_state['patients'][0]) == 0


# --------------------------------------------------------------------------------------
# eng_10 (mixer-attribute-configs): a mixer population must expose attribute_configs
# --------------------------------------------------------------------------------------

class TestMixerExposesAttributeConfigs:
    """The eng_2 injection reads ``patient_generator.attribute_configs``. A PatientGeneratorMixer
    skips ``super().__init__()`` and, before eng_10, never set it — so building an OptionLibrary
    over a mixer population crashed with AttributeError. That is every LPP/VOI run whose population
    is a mixer (e.g. LPP's 50/50 high/low-risk agent_n) and whose library has heuristic options."""

    def _full_visibility_generator(self):
        """A real PatientGenerator with all six attributes configured and visible."""
        env = make_env(
            antibiotic_names=['A'],
            num_patients_per_time_step=3,
            visible_patient_attributes=sorted(ALL_SIX_ATTRS),
        )
        return env.unwrapped.patient_generator

    def test_mixer_attribute_configs_is_union_of_children(self):
        child_a = self._full_visibility_generator()
        child_b = self._full_visibility_generator()
        mixer = PatientGeneratorMixer(
            config={'generators': [child_a, child_b], 'proportions': [0.5, 0.5], 'seed': 7}
        )
        expected = set(child_a.attribute_configs) | set(child_b.attribute_configs)
        assert set(mixer.attribute_configs.keys()) == expected
        assert set(mixer.attribute_configs.keys()) == ALL_SIX_ATTRS

    def test_option_library_validation_over_mixer_does_not_crash(self):
        """The exact regression path: OptionLibrary.validate_environment_compatibility reads
        patient_generator.attribute_configs and injects it. Pre-eng_10 this raised AttributeError
        on a mixer; now it injects the mixer's configured superset."""
        # A separate env supplies the antibiotic mapping / reward the OptionLibrary needs.
        env = make_env(
            antibiotic_names=['A'],
            num_patients_per_time_step=3,
            visible_patient_attributes=sorted(ALL_SIX_ATTRS),
        )
        mixer = PatientGeneratorMixer(
            config={
                'generators': [self._full_visibility_generator(), self._full_visibility_generator()],
                'proportions': [0.5, 0.5],
                'seed': 11,
            }
        )
        worker = _make_worker(uncertainty_threshold=99.0)  # inert: full vis -> score 0 -> never gates
        lib = OptionLibrary.from_env(env)
        lib.add_option(worker)

        lib.validate_environment_compatibility(patient_generator=mixer)  # must not raise

        assert set(worker._observable_patient_attributes) == set(mixer.attribute_configs.keys())
        assert set(worker._observable_patient_attributes) == ALL_SIX_ATTRS
