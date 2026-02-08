"""Block option loader for deterministic antibiotic prescribing."""

from typing import Any, Dict, List

import numpy as np

from abx_amr_simulator.hrl import OptionBase


class BlockOption(OptionBase):
    """Deterministic option that repeats a single antibiotic for k steps."""

    REQUIRES_OBSERVATION_ATTRIBUTES: List[str] = []
    REQUIRES_AMR_LEVELS: bool = False
    REQUIRES_STEP_NUMBER: bool = False
    PROVIDES_TERMINATION_CONDITION: bool = False

    def __init__(self, name: str, antibiotic: str, duration: int) -> None:
        super().__init__(name=name, k=duration)
        self.antibiotic = antibiotic

    def decide(self, env_state: Dict[str, Any]) -> np.ndarray:
        num_patients = env_state["num_patients"]
        option_library = env_state["option_library"]
        
        # No normalization - antibiotic name must match exactly
        try:
            action_idx = option_library.abx_name_to_index[self.antibiotic]
        except KeyError as exc:
            available = list(option_library.abx_name_to_index.keys())
            raise ValueError(
                f"Option '{self.name}': antibiotic '{self.antibiotic}' not in environment. "
                f"Available: {available}. "
                f"Note: Use exactly 'no_treatment' (no variations like 'NO_RX', 'no_treat')."
            ) from exc

        return np.full(shape=num_patients, fill_value=action_idx, dtype=np.int32)

    def get_referenced_antibiotics(self) -> List[str]:
        """Return the single antibiotic this block option prescribes."""
        return [self.antibiotic]


def load_block_option(name: str, config: Dict[str, Any]) -> OptionBase:
    """Instantiate a BlockOption from config.

    Expected config keys:
        - antibiotic (str)
        - duration (int)
        - allowed_antibiotics (optional list[str])
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    if "antibiotic" not in config:
        raise ValueError("BlockOption config missing required key 'antibiotic'")
    if "duration" not in config:
        raise ValueError("BlockOption config missing required key 'duration'")

    antibiotic = config["antibiotic"]
    duration = config["duration"]

    if not isinstance(antibiotic, str) or not antibiotic:
        raise ValueError("BlockOption 'antibiotic' must be a non-empty string")
    if not isinstance(duration, int) or duration < 1:
        raise ValueError("BlockOption 'duration' must be an int >= 1")

    _validate_allowed_antibiotics(
        antibiotic=antibiotic,
        allowed_antibiotics=config.get("allowed_antibiotics"),
    )

    return BlockOption(name=name, antibiotic=antibiotic, duration=duration)


def _validate_allowed_antibiotics(antibiotic: str, allowed_antibiotics: Any) -> None:
    if allowed_antibiotics is None:
        return
    if not isinstance(allowed_antibiotics, list) or not all(
        isinstance(entry, str) for entry in allowed_antibiotics
    ):
        raise ValueError("allowed_antibiotics must be a list of strings")
    if antibiotic not in allowed_antibiotics:
        raise ValueError(
            f"antibiotic '{antibiotic}' not in allowed_antibiotics {allowed_antibiotics}"
        )
