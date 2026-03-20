"""Custom reward calculator plugin fixture for integration tests."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from abx_amr_simulator.core.base_reward_calculator import RewardCalculatorBase
from abx_amr_simulator.core.reward_calculator import RewardCalculator


class CustomRewardCalculatorPlugin(RewardCalculatorBase):
    """Thin wrapper over canonical calculator used to validate plugin seam."""

    IS_CUSTOM_PLUGIN = True
    REQUIRED_PATIENT_ATTRS = list(RewardCalculator.REQUIRED_PATIENT_ATTRS)

    def __init__(self, config: Dict[str, Any]) -> None:
        marker_path = config.get("runtime_marker_path")
        if marker_path is not None:
            marker_file = Path(marker_path)
            marker_file.parent.mkdir(parents=True, exist_ok=True)
            marker_file.write_text("CustomRewardCalculatorPlugin\n", encoding="utf-8")

        self._delegate = RewardCalculator(config=config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def calculate_reward(
        self,
        patients: List[Any],
        actions: np.ndarray,
        antibiotic_names: List[str],
        visible_amr_levels: Dict[str, float],
        delta_visible_amr_per_antibiotic: Dict[str, float],
        **kwargs: Any,
    ) -> Any:
        return self._delegate.calculate_reward(
            patients=patients,
            actions=actions,
            antibiotic_names=antibiotic_names,
            visible_amr_levels=visible_amr_levels,
            delta_visible_amr_per_antibiotic=delta_visible_amr_per_antibiotic,
            **kwargs,
        )


def load_reward_calculator_component(config: Dict[str, Any]) -> RewardCalculatorBase:
    return CustomRewardCalculatorPlugin(config=config)
