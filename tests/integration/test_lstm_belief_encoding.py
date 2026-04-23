"""Integration test: LSTMStateLogger captures hidden states and probe encodes AMR.

Trains a single-agent HRL_RPPO (RecurrentPPO) agent for a short run with
LSTMStateLogger enabled. Verifies that:
1. The logger writes episode .npz files to disk.
2. Hidden-state arrays have the expected shape.
3. True AMR levels are recorded alongside hidden states (confirms
   actual_amr_levels passes through OptionsWrapper's info dict).
4. A linear regression probe from hidden states → true AMR achieves
   R² > 0.2 for at least one antibiotic (smoke-test threshold).

This test uses only canonical package components — ABXAMREnv, OptionsWrapper,
RecurrentPPO, LSTMStateLogger, and probe_hidden_belief — with no workspace
YAML file dependencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sb3_contrib import RecurrentPPO

from abx_amr_simulator.analysis.probe_hidden_belief import fit_probe, load_episodes
from abx_amr_simulator.callbacks import LSTMStateLogger
from abx_amr_simulator.core import ABXAMREnv, PatientGenerator, RewardCalculator
from abx_amr_simulator.hrl import OptionBase, OptionLibrary, OptionsWrapper


# ---------------------------------------------------------------------------
# Minimal real option: prescribes a fixed antibiotic for k primitive steps
# ---------------------------------------------------------------------------

class _ConstantOption(OptionBase):
    """Prescribes one antibiotic for k primitive steps (no_treatment if uninfected)."""

    REQUIRES_OBSERVATION_ATTRIBUTES = ["prob_infected"]
    REQUIRES_AMR_LEVELS = False
    REQUIRES_STEP_NUMBER = False
    PROVIDES_TERMINATION_CONDITION = False

    def __init__(self, name: str, action_name: str, k: int = 5) -> None:
        super().__init__(name=name, k=k)
        self._action_name = action_name

    def decide(self, env_state: dict) -> np.ndarray:
        patients = env_state.get("patients", [])
        n = env_state.get("num_patients", len(patients))
        actions = np.full(shape=(n,), fill_value=self._action_name, dtype=object)
        for i, p in enumerate(patients):
            if p.get("prob_infected", 1.0) == 0.0:
                actions[i] = "no_treatment"
        return actions

    def get_referenced_antibiotics(self) -> list:
        return [self._action_name] if self._action_name != "no_treatment" else []


# ---------------------------------------------------------------------------
# Environment construction helpers
# ---------------------------------------------------------------------------

_ABX_NAMES = ["A", "B"]
_N_PATIENTS = 3
_MAX_TIME_STEPS = 10   # primitive steps per episode
_OPTION_K = 5          # each option spans 5 primitive steps → 2 macro steps/episode


def _make_wrapped_env() -> OptionsWrapper:
    """Build a minimal ABXAMREnv wrapped with a single ConstantOption library."""
    pg_config = {
        "prob_infected": {
            "prob_dist": {"type": "constant", "value": 0.7},
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
            "prob_dist": {"type": "constant", "value": 1.0},
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
            "prob_dist": {"type": "constant", "value": 0.01},
            "obs_bias_multiplier": 1.0,
            "obs_noise_one_std_dev": 0.0,
            "obs_noise_std_dev_fraction": 0.0,
            "clipping_bounds": [0.0, 1.0],
        },
        "visible_patient_attributes": ["prob_infected"],
    }
    rc_config = {
        "abx_clinical_reward_penalties_info_dict": {
            "clinical_benefit_reward": 10.0,
            "clinical_benefit_probability": 0.7,
            "clinical_failure_penalty": -5.0,
            "clinical_failure_probability": 0.1,
            "abx_adverse_effects_info": {
                abx: {"adverse_effect_penalty": -1.0, "adverse_effect_probability": 0.05}
                for abx in _ABX_NAMES
            },
        },
        "lambda_weight": 0.3,
    }
    amr_dict = {
        abx: {
            "leak": 0.05,
            "flatness_parameter": 1.0,
            "permanent_residual_volume": 0.0,
            "initial_amr_level": 0.1,
        }
        for abx in _ABX_NAMES
    }
    base_env = ABXAMREnv(
        patient_generator=PatientGenerator(config=pg_config),
        reward_calculator=RewardCalculator(config=rc_config),
        antibiotics_AMR_dict=amr_dict,
        num_patients_per_time_step=_N_PATIENTS,
        max_time_steps=_MAX_TIME_STEPS,
    )
    lib = OptionLibrary(reward_calculator=base_env.reward_calculator, name="test_lib")
    lib.add_option(_ConstantOption(name="prescribe_A", action_name="A", k=_OPTION_K))
    return OptionsWrapper(env=base_env, option_library=lib, gamma=0.99)


# ---------------------------------------------------------------------------
# Module-scoped fixture: train once, reuse across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lstm_log_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a short RecurrentPPO run with LSTMStateLogger; return the log dir.

    Parameters chosen so the run is fast but produces enough data for the probe:
    - _MAX_TIME_STEPS=10, _OPTION_K=5  → 2 macro steps per episode
    - total_timesteps=120 macro steps  → ~60 training episodes
    - n_steps=10, batch_size=5         → update every 10 macro steps
    """
    log_dir = tmp_path_factory.mktemp("lstm_logs")
    env = _make_wrapped_env()

    lstm_logger = LSTMStateLogger(save_dir=str(log_dir), log_freq=0, verbose=0)

    agent = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        n_steps=10,
        batch_size=5,
        n_epochs=2,
        seed=42,
        verbose=0,
    )

    # 60 episodes × 2 macro steps/episode = 120 total macro steps.
    # OptionsWrapper exposes one macro step per env.step() call, so
    # RecurrentPPO.learn(total_timesteps=120) runs for 120 macro steps.
    agent.learn(total_timesteps=120, callback=lstm_logger)

    return log_dir


# ---------------------------------------------------------------------------
# 1. Logger output tests
# ---------------------------------------------------------------------------

class TestLSTMStateLoggerOutput:
    """Verify that LSTMStateLogger writes well-formed .npz files during training."""

    def test_episode_files_are_written(self, lstm_log_dir: Path) -> None:
        """At least one episode_*.npz file must appear in the log directory."""
        files = list(lstm_log_dir.glob("episode_*.npz"))
        assert len(files) > 0, f"No episode .npz files found in {lstm_log_dir}"

    def test_multiple_episodes_logged(self, lstm_log_dir: Path) -> None:
        """Expect more than one episode to have been logged during the training run."""
        files = list(lstm_log_dir.glob("episode_*.npz"))
        assert len(files) > 1, (
            f"Only {len(files)} episode file(s) found; expected multiple episodes "
            "from a 120 macro-step training run."
        )

    def test_hidden_states_key_present(self, lstm_log_dir: Path) -> None:
        """Every episode .npz must contain a 'hidden_states' array."""
        for f in sorted(lstm_log_dir.glob("episode_*.npz")):
            data = np.load(f)
            assert "hidden_states" in data, (
                f"'hidden_states' key missing from {f.name}"
            )

    def test_hidden_states_are_2d_or_higher(self, lstm_log_dir: Path) -> None:
        """Hidden states must be at least 2-D: (timesteps, [layers,] hidden_size)."""
        first_file = sorted(lstm_log_dir.glob("episode_*.npz"))[0]
        hidden = np.load(first_file)["hidden_states"]
        assert hidden.ndim >= 2, (
            f"Expected hidden_states ndim ≥ 2, got shape {hidden.shape}"
        )

    def test_true_amr_recorded(self, lstm_log_dir: Path) -> None:
        """At least one episode must contain 'true_amr', confirming that
        actual_amr_levels passes through OptionsWrapper's info dict."""
        files = sorted(lstm_log_dir.glob("episode_*.npz"))
        any_amr = any("true_amr" in np.load(f) for f in files)
        assert any_amr, (
            "No episode contained 'true_amr'. "
            "Check that actual_amr_levels is present in OptionsWrapper's info dict."
        )

    def test_timesteps_recorded(self, lstm_log_dir: Path) -> None:
        """Episodes should include monotonically increasing timestep counters."""
        first_file = sorted(lstm_log_dir.glob("episode_*.npz"))[0]
        data = np.load(first_file)
        assert "timesteps" in data, "'timesteps' key missing from episode .npz"
        assert np.all(np.diff(data["timesteps"]) >= 0), (
            "Timesteps are not monotonically non-decreasing within an episode"
        )


# ---------------------------------------------------------------------------
# 2. Probe tests
# ---------------------------------------------------------------------------

class TestLSTMProbe:
    """Verify that the linear probe pipeline runs correctly and produces
    finite, non-trivial R² values from the logged hidden states."""

    def test_load_episodes_returns_matched_shapes(self, lstm_log_dir: Path) -> None:
        """load_episodes() must return hidden states and AMR with matching first dim."""
        hidden, amr, _ = load_episodes(lstm_log_dir)
        assert hidden.shape[0] == amr.shape[0], (
            f"hidden_states has {hidden.shape[0]} rows but true_amr has {amr.shape[0]}"
        )

    def test_amr_has_correct_num_antibiotics(self, lstm_log_dir: Path) -> None:
        """True AMR array should have one column per antibiotic (2 in this test)."""
        _, amr, _ = load_episodes(lstm_log_dir)
        assert amr.shape[1] == len(_ABX_NAMES), (
            f"Expected {len(_ABX_NAMES)} antibiotic columns, got {amr.shape[1]}"
        )

    def test_probe_results_are_finite(self, lstm_log_dir: Path) -> None:
        """All test R² and MAE values from fit_probe() must be finite numbers."""
        hidden, amr, _ = load_episodes(lstm_log_dir)
        results = fit_probe(hidden, amr)
        for r in results["results"]:
            assert np.isfinite(r["test_r2"]), (
                f"Non-finite test R² for antibiotic {r['antibiotic_idx']}: {r['test_r2']}"
            )
            assert np.isfinite(r["test_mae"]), (
                f"Non-finite test MAE for antibiotic {r['antibiotic_idx']}: {r['test_mae']}"
            )

    @pytest.mark.slow
    def test_probe_achieves_nontrivial_r2(self, lstm_log_dir: Path) -> None:
        """At least one antibiotic should have test R² > 0.2.

        This is a smoke-test threshold (not the scientific 0.5 threshold).
        With AMR levels included in the manager observation, the LSTM
        hidden state should linearly decode AMR even after partial training.
        """
        hidden, amr, _ = load_episodes(lstm_log_dir)
        results = fit_probe(hidden, amr)
        max_r2 = max(r["test_r2"] for r in results["results"])
        assert max_r2 > 0.2, (
            f"Best test R² across antibiotics was {max_r2:.3f}; expected > 0.2. "
            "Possible causes: too few training steps, AMR not passing through "
            "OptionsWrapper info dict, or hidden state not receiving AMR signal."
        )
