import numpy as np

from abx_amr_simulator.core import PatientGenerator
from abx_amr_simulator.core import PatientGeneratorMixer


TRUE_AMR_LEVELS = {"A": 0.0}


def _build_generator(*, failure_value_multiplier: float) -> PatientGenerator:
    config = {
        "prob_infected": {
            "prob_dist": {"type": "constant", "value": 0.5},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, 1.0],
        },
        "benefit_value_multiplier": {
            "prob_dist": {"type": "constant", "value": 1.0},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "failure_value_multiplier": {
            "prob_dist": {"type": "constant", "value": failure_value_multiplier},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "benefit_probability_multiplier": {
            "prob_dist": {"type": "constant", "value": 1.0},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "failure_probability_multiplier": {
            "prob_dist": {"type": "constant", "value": 1.0},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, None],
        },
        "recovery_without_treatment_prob": {
            "prob_dist": {"type": "constant", "value": 0.0},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, 1.0],
        },
        "visible_patient_attributes": ["prob_infected"],
    }
    return PatientGenerator(config=config)


def _build_equal_mixer() -> PatientGeneratorMixer:
    gen_low = _build_generator(failure_value_multiplier=0.8)
    gen_high = _build_generator(failure_value_multiplier=1.2)
    return PatientGeneratorMixer(
        config={
            "generators": [gen_low, gen_high],
            "proportions": [0.5, 0.5],
            "visible_patient_attributes": ["prob_infected"],
            "seed": 42,
        }
    )


def test_mixer_n_equals_1_samples_both_generators():
    mixer = _build_equal_mixer()
    rng = np.random.default_rng(seed=42)

    n_draws = 1000
    counts = np.array([0, 0], dtype=int)

    for _ in range(n_draws):
        patients = mixer.sample(
            n_patients=1,
            true_amr_levels=TRUE_AMR_LEVELS,
            rng=rng,
        )
        assert len(patients) == 1
        source_idx = int(getattr(patients[0], "source_generator_index"))
        counts[source_idx] += 1

    assert counts.sum() == n_draws
    assert 400 <= counts[0] <= 600, f"Generator 0 count {counts[0]} outside [400, 600]"
    assert 400 <= counts[1] <= 600, f"Generator 1 count {counts[1]} outside [400, 600]"


def test_mixer_uses_multinomial_sampling_for_n_equals_10():
    mixer = _build_equal_mixer()
    rng = np.random.default_rng(seed=123)

    n_steps = 1000
    per_step_counts_gen0 = []

    for _ in range(n_steps):
        patients = mixer.sample(
            n_patients=10,
            true_amr_levels=TRUE_AMR_LEVELS,
            rng=rng,
        )
        assert len(patients) == 10
        gen0_count = sum(int(getattr(patient, "source_generator_index")) == 0 for patient in patients)
        per_step_counts_gen0.append(gen0_count)

    counts = np.array(per_step_counts_gen0, dtype=int)

    assert np.any(counts != 5), "Counts are always 5, expected stochastic multinomial variability"
    assert np.sum(counts != 5) > 500, "Too few non-5 outcomes for Binomial(n=10, p=0.5)"

    empirical_mean = float(np.mean(counts))
    assert 4.5 <= empirical_mean <= 5.5, f"Empirical mean {empirical_mean} not near expected 5.0"
