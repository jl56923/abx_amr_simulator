"""Custom AMR dynamics plugin fixture for integration tests."""

from pathlib import Path
from typing import Any, Dict

from abx_amr_simulator.core.base_amr_dynamics import AMRDynamicsBase
from abx_amr_simulator.core.leaky_balloon import AMR_LeakyBalloon


class CustomAMRDynamicsPlugin(AMRDynamicsBase):
    """Thin wrapper over canonical AMR dynamics used to validate plugin seam."""

    NAME = "custom_amr_dynamics_plugin"
    IS_CUSTOM_PLUGIN = True

    def __init__(self, params: Dict[str, Any]) -> None:
        marker_path = params.get("runtime_marker_path")
        if marker_path is not None:
            marker_file = Path(marker_path)
            marker_file.parent.mkdir(parents=True, exist_ok=True)
            with open(marker_file, "a", encoding="utf-8") as handle:
                handle.write("CustomAMRDynamicsPlugin\n")

        self._delegate = AMR_LeakyBalloon(
            leak=params["leak"],
            flatness_parameter=params["flatness_parameter"],
            permanent_residual_volume=params["permanent_residual_volume"],
            initial_amr_level=params["initial_amr_level"],
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def step(self, doses: float) -> float:
        return self._delegate.step(doses=doses)

    def reset(
        self,
        initial_amr_level: float | None = None,
        initial_level: float | None = None,
    ) -> None:
        resolved_initial_level = (
            initial_amr_level if initial_amr_level is not None else initial_level
        )
        if resolved_initial_level is None:
            raise ValueError(
                "CustomAMRDynamicsPlugin.reset requires 'initial_amr_level' or 'initial_level'."
            )
        self._delegate.reset(initial_amr_level=resolved_initial_level)


def load_amr_dynamics_component(config: Dict[str, Any]) -> Dict[str, AMRDynamicsBase]:
    antibiotics_amr_dict = config["antibiotics_AMR_dict"]
    runtime_marker_path = config.get("runtime_marker_path")
    return {
        antibiotic_name: CustomAMRDynamicsPlugin(
            params={
                **antibiotic_params,
                "runtime_marker_path": runtime_marker_path,
            }
        )
        for antibiotic_name, antibiotic_params in antibiotics_amr_dict.items()
    }
