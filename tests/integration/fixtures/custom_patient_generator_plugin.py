"""Custom patient generator plugin fixture for integration tests."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from abx_amr_simulator.core.base_patient_generator import PatientGeneratorBase
from abx_amr_simulator.core.patient_generator import PatientGenerator


class CustomPatientGeneratorPlugin(PatientGeneratorBase):
    """Thin wrapper over canonical generator used to validate plugin seam."""

    IS_CUSTOM_PLUGIN = True
    PROVIDES_ATTRIBUTES = list(PatientGenerator.PROVIDES_ATTRIBUTES)

    def __init__(self, config: Dict[str, Any]) -> None:
        marker_path = config.get("runtime_marker_path")
        if marker_path is not None:
            marker_file = Path(marker_path)
            marker_file.parent.mkdir(parents=True, exist_ok=True)
            marker_file.write_text("CustomPatientGeneratorPlugin\n", encoding="utf-8")

        self._delegate = PatientGenerator(config=config)
        self.visible_patient_attributes = list(self._delegate.visible_patient_attributes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def sample(
        self,
        n_patients: int,
        true_amr_levels: Dict[str, float],
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> List[Any]:
        return self._delegate.sample(
            n_patients=n_patients,
            true_amr_levels=true_amr_levels,
            rng=rng,
            **kwargs,
        )

    def observe(self, patients: List[Any]) -> np.ndarray:
        return self._delegate.observe(patients=patients)

    def obs_dim(self, num_patients: int) -> int:
        return self._delegate.obs_dim(num_patients=num_patients)


def load_patient_generator_component(config: Dict[str, Any]) -> PatientGeneratorBase:
    return CustomPatientGeneratorPlugin(config=config)
