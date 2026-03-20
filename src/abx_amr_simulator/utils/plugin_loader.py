"""Load plugin-backed simulator components from merged configuration.

`load_plugin_component` is the standardized utility used by component factories to:
1) detect plugin configuration,
2) resolve/import a plugin loader module,
3) resolve the loader function,
4) call the loader with merged config,
5) validate the returned object type.

Example:
    from abx_amr_simulator.core.base_patient_generator import PatientGeneratorBase
    from abx_amr_simulator.utils.plugin_loader import load_plugin_component

    component = load_plugin_component(
        component_config=patient_generator_config,
        expected_base_class=PatientGeneratorBase,
        default_loader_fn_name='load_patient_generator_component',
        config_dir_hint='/path/to/config/dir',
    )
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional


def load_plugin_component(
    component_config: Dict[str, Any],
    expected_base_class: type,
    default_loader_fn_name: str,
    config_dir_hint: Optional[str] = None,
) -> Optional[Any]:
    """Load and validate a plugin-backed component instance.

    Args:
        component_config: Full merged component config.
        expected_base_class: Required component base class for type validation.
        default_loader_fn_name: Default loader function name when not specified in config.
        config_dir_hint: Directory for resolving relative filesystem plugin module paths.

    Returns:
        Loaded component instance, or `None` when no plugin is configured.

    Raises:
        ValueError: If plugin config is malformed or path resolution cannot proceed.
        ImportError: If the plugin module cannot be imported.
        AttributeError: If the configured loader function does not exist.
        RuntimeError: If the loader function raises.
        TypeError: If return type is invalid.
    """
    if 'plugin' not in component_config:
        return None

    plugin_config = component_config['plugin']
    if not isinstance(plugin_config, dict):
        raise ValueError(
            "Invalid plugin configuration: 'plugin' must be a dictionary."
        )

    loader_module = plugin_config.get('loader_module')
    if not loader_module:
        raise ValueError(
            "Invalid plugin configuration: missing required key 'plugin.loader_module'."
        )

    loader_function_name = plugin_config.get('loader_function', default_loader_fn_name)
    if not isinstance(loader_function_name, str) or not loader_function_name.strip():
        raise ValueError(
            "Invalid plugin configuration: 'plugin.loader_function' must be a non-empty string."
        )

    module = None
    import_error: Optional[Exception] = None
    try:
        module = importlib.import_module(loader_module)
    except Exception as exc:
        import_error = exc

    if module is None:
        loader_path = Path(loader_module)
        if not loader_path.is_absolute():
            if config_dir_hint is None:
                raise ValueError(
                    "Plugin loader module was not importable as a Python module path and "
                    "cannot be resolved as a relative filesystem path because no "
                    "'config_dir_hint' was provided. "
                    f"loader_module='{loader_module}'. Original import error: {import_error}"
                )
            loader_path = (Path(config_dir_hint) / loader_path).resolve()
        else:
            loader_path = loader_path.resolve()

        if not loader_path.exists():
            raise ValueError(
                "Plugin loader module is not importable and filesystem path does not exist. "
                f"loader_module='{loader_module}', resolved_path='{loader_path}'. "
                f"Original import error: {import_error}"
            )

        try:
            module_name = f"abx_amr_simulator_plugin_loader_{loader_path.stem}"
            module_spec = importlib.util.spec_from_file_location(
                name=module_name,
                location=str(loader_path),
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(
                    f"Failed to create module spec for plugin loader path '{loader_path}'."
                )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        except Exception as exc:
            raise ImportError(
                "Failed to import plugin loader module from filesystem path. "
                f"loader_module='{loader_module}', resolved_path='{loader_path}'. Error: {exc}"
            ) from exc

    if module is None:
        raise ImportError(
            "Failed to import plugin loader module for unknown reasons. "
            f"loader_module='{loader_module}'."
        )

    if not hasattr(module, loader_function_name):
        raise AttributeError(
            "Plugin loader function not found in module. "
            f"loader_module='{loader_module}', loader_function='{loader_function_name}'."
        )

    loader_fn = getattr(module, loader_function_name)
    if not callable(loader_fn):
        raise AttributeError(
            "Configured plugin loader attribute is not callable. "
            f"loader_module='{loader_module}', loader_function='{loader_function_name}'."
        )

    try:
        result = loader_fn(config=component_config)
    except Exception as exc:
        raise RuntimeError(
            "Plugin loader function raised an exception. "
            f"loader_module='{loader_module}', loader_function='{loader_function_name}'. Error: {exc}"
        ) from exc

    if not isinstance(result, expected_base_class):
        raise TypeError(
            "Plugin loader returned invalid type. "
            f"Expected instance of '{expected_base_class.__name__}', got '{type(result).__name__}'."
        )

    return result