"""Alternation option loader for deterministic sequences."""

from typing import Any, Dict, List, Optional

import numpy as np

from abx_amr_simulator.hrl import OptionBase


class AlternationOption(OptionBase):
    """Deterministic option that follows a fixed sequence of antibiotics."""

    REQUIRES_OBSERVATION_ATTRIBUTES: List[str] = []
    REQUIRES_AMR_LEVELS: bool = False
    REQUIRES_STEP_NUMBER: bool = True
    PROVIDES_TERMINATION_CONDITION: bool = False

    def __init__(self, name: str, sequence: List[str]) -> None:
        if not sequence:
            raise ValueError("AlternationOption sequence must be non-empty")
        super().__init__(name=name, k=len(sequence))
        self.sequence = sequence
        self._sequence_index = 0
        self._last_seen_step: Optional[int] = None

    def decide(self, env_state: Dict[str, Any]) -> np.ndarray:
        num_patients = env_state["num_patients"]
        option_library = env_state["option_library"]
        current_step = int(env_state.get("current_step", 0))

        if self._last_seen_step is None or current_step != self._last_seen_step + 1:
            self._sequence_index = 0

        # No normalization - antibiotic name must match exactly
        action_name = self.sequence[self._sequence_index]

        try:
            # Just validate that antibiotic exists; return the name string
            _ = option_library.abx_name_to_index[action_name]
        except KeyError as exc:
            available = list(option_library.abx_name_to_index.keys())
            raise ValueError(
                f"Option '{self.name}': antibiotic '{action_name}' not in environment. "
                f"Available: {available}. "
                f"Note: Use exactly 'no_treatment' (no variations like 'NO_RX', 'no_treat')."
            ) from exc

        self._sequence_index = (self._sequence_index + 1) % len(self.sequence)
        self._last_seen_step = current_step

        return np.full(shape=num_patients, fill_value=action_name, dtype=object)

    def reset(self) -> None:
        self._sequence_index = 0
        self._last_seen_step = None

    def get_referenced_antibiotics(self) -> List[str]:
        """Return all antibiotics in the alternation sequence."""
        return list(self.sequence)


def load_alternation_option(name: str, config: Dict[str, Any]) -> OptionBase:
    """Instantiate an AlternationOption from config.

    Expected config keys:
        - sequence (list[str])
        - allowed_antibiotics (optional list[str])
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    if "sequence" not in config:
        raise ValueError("AlternationOption config missing required key 'sequence'")

    sequence = config["sequence"]
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("AlternationOption 'sequence' must be a non-empty list")
    if not all(isinstance(entry, str) and entry for entry in sequence):
        raise ValueError("AlternationOption 'sequence' entries must be non-empty strings")

    _validate_allowed_antibiotics(
        sequence=sequence,
        allowed_antibiotics=config.get("allowed_antibiotics"),
    )

    return AlternationOption(name=name, sequence=sequence)


def _validate_allowed_antibiotics(sequence: List[str], allowed_antibiotics: Any) -> None:
    if allowed_antibiotics is None:
        return
    if not isinstance(allowed_antibiotics, list) or not all(
        isinstance(entry, str) for entry in allowed_antibiotics
    ):
        raise ValueError("allowed_antibiotics must be a list of strings")
    for antibiotic in sequence:
        if antibiotic not in allowed_antibiotics:
            raise ValueError(
                f"antibiotic '{antibiotic}' not in allowed_antibiotics {allowed_antibiotics}"
            )
