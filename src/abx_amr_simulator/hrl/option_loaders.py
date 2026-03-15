"""Dynamic loader for option libraries from YAML configurations.

The OptionLibraryLoader handles the orchestration of:
1. Reading library meta-configs (which options are in the library)
2. Loading per-option type configs and merging with overrides
3. Dynamically importing loader functions
4. Validating loaded options are OptionBase instances
5. Building a resolved config for reproducibility
"""

import os
import importlib
import importlib.util
from typing import Dict, Any, Tuple, List
from pathlib import Path

import yaml

from abx_amr_simulator.hrl.base_option import OptionBase
from abx_amr_simulator.hrl.options import OptionLibrary


class OptionLibraryLoader:
    """Loader for option libraries from YAML configs.
    
    Follows the same hierarchical config pattern as the main training system:
    - Library meta-config specifies which options to include
    - Option type configs provide defaults
    - config_params_override allows per-option customization
    - Resolved config is saved for reproducibility
    """

    _ALLOWED_OPTION_TYPES = ('block', 'alternation', 'heuristic', 'custom')
    _CANONICAL_OPTION_TYPES = ('block', 'alternation', 'heuristic')
    _CANONICAL_LOADER_TARGETS = {
        'block': (
            'abx_amr_simulator.options.defaults.option_types.block.block_option_loader',
            'load_block_option',
        ),
        'alternation': (
            'abx_amr_simulator.options.defaults.option_types.alternation.alternation_option_loader',
            'load_alternation_option',
        ),
        'heuristic': (
            'abx_amr_simulator.options.defaults.option_types.heuristic.heuristic_option_loader',
            'load_heuristic_option',
        ),
    }

    @staticmethod
    def load_library(
        library_config_path: str,
        env: Any,
        library_name: str = None,
    ) -> Tuple[OptionLibrary, Dict[str, Any]]:
        """Load an option library from YAML configuration.
        
        Args:
            library_config_path: Path to library meta-config YAML file.
                Example: 'experiments/options/option_libraries/default_deterministic.yaml'
            env: The ABXAMREnv instance (passed to OptionLibrary for antibiotic mapping extraction).
            library_name: Optional override for library name (from config if not provided).

        Path handling:
                        - Relative paths inside the library config (option_subconfig_file, loader_module)
                            are resolved relative to the directory containing the library config file.
                        - Absolute paths are used as-is.
                        - loader_module can also be a Python module path (e.g.,
                            "abx_amr_simulator.options.heuristic_loader").
        
        Returns:
            Tuple of:
                - OptionLibrary: Instantiated and populated option library
                - resolved_config: Dict with all instantiated option configs (for logging/reproducibility)
        
        Raises:
            FileNotFoundError: If library config file not found.
            ValueError: If config format invalid.
            RuntimeError: If any option fails to load.
        
        Example:
            lib, resolved_cfg = OptionLibraryLoader.load_library(
                library_config_path='experiments/options/option_libraries/default_deterministic.yaml',
                env=env
            )
            # Now lib contains all options, ready for OptionsWrapper
        """
        # Resolve path
        config_path = Path(library_config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Library config not found: {config_path}")

        # Load library meta-config
        with open(config_path, 'r') as f:
            lib_config = yaml.safe_load(f)

        if not lib_config:
            raise ValueError(f"Library config is empty: {config_path}")

        # Extract library metadata
        lib_name = library_name or lib_config.get('library_name', 'default')
        lib_description = lib_config.get('description', '')
        options_specs = lib_config.get('options', [])

        if not options_specs:
            raise ValueError(f"Library config has no options: {config_path}")

        # Create library (pass env to extract antibiotic mappings)
        library = OptionLibrary(env=env, name=lib_name)
        resolved_options = []

        # Load each option
        base_dir = config_path.parent
        for i, opt_spec in enumerate(options_specs):
            try:
                opt_name = opt_spec.get('option_name')
                if not opt_name:
                    raise ValueError(f"Option {i} missing 'option_name'")

                opt_type = opt_spec.get('option_type')
                if not opt_type:
                    raise ValueError(f"Option '{opt_name}' missing 'option_type'")

                # Load option
                option, opt_resolved_cfg = OptionLibraryLoader._load_single_option(
                    name=opt_name,
                    option_type=opt_type,
                    opt_spec=opt_spec,
                    base_dir=base_dir,
                )

                # Add to library
                library.add_option(option)
                resolved_options.append(opt_resolved_cfg)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load option {i}: {str(e)}. "
                    f"Option spec: {opt_spec}"
                )

        # Build resolved config
        resolved_config = {
            'library_name': lib_name,
            'library_description': lib_description,
            'num_options': len(library),
            'options': resolved_options,
        }

        return library, resolved_config

    @staticmethod
    def _load_single_option(
        name: str,
        option_type: str,
        opt_spec: Dict[str, Any],
        base_dir: Path,
    ) -> Tuple[OptionBase, Dict[str, Any]]:
        """Load a single option from config spec.
        
        Args:
            name: Option name (e.g., 'A_5').
            option_type: Option type (e.g., 'block').
            opt_spec: Full option spec dict from library config.
            base_dir: Base directory for resolving relative paths.

                Path handling:
                        - Relative paths in option_subconfig_file and loader_module are resolved
                            relative to base_dir (the library config's directory).
                        - Absolute paths are used as-is.
        
        Returns:
            Tuple of:
                - OptionBase: Instantiated option
                - resolved_config: Dict with merged config for this option
        
        Raises:
            ValueError: If config invalid.
            RuntimeError: If loader function not found or fails.
        """
        OptionLibraryLoader._validate_option_contract(
            name=name,
            option_type=option_type,
            opt_spec=opt_spec,
        )

        # Get config file path
        subconfig_file = opt_spec.get('option_subconfig_file')
        config_override = opt_spec.get('config_params_override', {})
        plugin_config = opt_spec.get('plugin')

        if not subconfig_file:
            raise ValueError(f"Option '{name}' missing 'option_subconfig_file'")

        # Resolve paths: handle both absolute and relative paths
        subconfig_path = Path(subconfig_file)
        if not subconfig_path.is_absolute():
            subconfig_path = (base_dir / subconfig_path).resolve()
        else:
            subconfig_path = subconfig_path.resolve()

        if not subconfig_path.exists():
            raise FileNotFoundError(f"Option subconfig not found: {subconfig_path}")

        # Load default config
        with open(subconfig_path, 'r') as f:
            default_config = yaml.safe_load(f) or {}

        # Merge with override
        merged_config = {**default_config, **config_override}

        # Import loader module dynamically
        if option_type in OptionLibraryLoader._CANONICAL_OPTION_TYPES:
            loader_func = OptionLibraryLoader._get_canonical_loader_function(
                option_type=option_type,
            )
        else:
            if not isinstance(plugin_config, dict):
                raise ValueError(
                    f"Option '{name}' with option_type 'custom' must provide 'plugin' as a dictionary."
                )

            loader_module = plugin_config['loader_module']
            loader_function = plugin_config['loader_function']
            loader_func = OptionLibraryLoader._resolve_custom_loader_function(
                option_name=name,
                loader_module=loader_module,
                loader_function=loader_function,
                base_dir=base_dir,
            )

        # Call loader function
        try:
            option = loader_func(name=name, config=merged_config)
        except Exception as e:
            raise RuntimeError(
                f"Loader function for option '{name}' (type '{option_type}') raised error: {e}"
            )

        # Validate return type
        if not isinstance(option, OptionBase):
            raise TypeError(
                f"Loader function returned {type(option).__name__}, expected OptionBase instance. "
                f"Make sure loader returns a subclass of OptionBase."
            )

        # Build resolved config
        resolved_config = {
            'option_name': name,
            'option_type': option_type,
            'default_config': default_config,
            'overrides': config_override,
            'merged_config': merged_config,
            'k': option.k if option.k != float('inf') else 'inf',
            'requires_observation_attrs': option.REQUIRES_OBSERVATION_ATTRIBUTES,
            'requires_amr_levels': option.REQUIRES_AMR_LEVELS,
        }

        return option, resolved_config

    @staticmethod
    def _validate_option_contract(
        name: str,
        option_type: str,
        opt_spec: Dict[str, Any],
    ) -> None:
        if option_type not in OptionLibraryLoader._ALLOWED_OPTION_TYPES:
            allowed_types_str = ', '.join(OptionLibraryLoader._ALLOWED_OPTION_TYPES)
            raise ValueError(
                f"Option '{name}' has unsupported option_type '{option_type}'. "
                f"Allowed values are: {allowed_types_str}."
            )

        legacy_keys = [
            key for key in ('loader_module', 'loader_function') if key in opt_spec
        ]
        if legacy_keys:
            legacy_keys_str = ', '.join(legacy_keys)
            raise ValueError(
                f"Option '{name}' uses legacy top-level loader keys ({legacy_keys_str}). "
                "Migration required: remove top-level loader keys. "
                "Use canonical option_type values ('block', 'alternation', 'heuristic') without plugin fields, "
                "or set option_type='custom' and provide plugin.loader_module + plugin.loader_function."
            )

        plugin_config = opt_spec.get('plugin')
        if option_type in OptionLibraryLoader._CANONICAL_OPTION_TYPES:
            if plugin_config is not None:
                raise ValueError(
                    f"Option '{name}' has canonical option_type '{option_type}' and must not include plugin fields. "
                    "Remove the 'plugin' section for canonical options."
                )
            return

        if not isinstance(plugin_config, dict):
            raise ValueError(
                f"Option '{name}' with option_type 'custom' must include a 'plugin' mapping "
                "with 'loader_module' and 'loader_function'."
            )

        missing_plugin_keys = [
            key for key in ('loader_module', 'loader_function')
            if not plugin_config.get(key)
        ]
        if missing_plugin_keys:
            missing_keys_str = ', '.join(missing_plugin_keys)
            raise ValueError(
                f"Option '{name}' with option_type 'custom' is missing required plugin fields: {missing_keys_str}. "
                "Expected keys: plugin.loader_module and plugin.loader_function."
            )

    @staticmethod
    def _get_canonical_loader_function(option_type: str):
        loader_module, loader_function = OptionLibraryLoader._CANONICAL_LOADER_TARGETS[option_type]
        try:
            module = importlib.import_module(loader_module)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import canonical loader module '{loader_module}' for option_type '{option_type}': {e}"
            )

        if not hasattr(module, loader_function):
            raise RuntimeError(
                f"Canonical loader module '{loader_module}' is missing expected function '{loader_function}'."
            )

        load_fn = getattr(module, loader_function)
        if not callable(load_fn):
            raise RuntimeError(
                f"Canonical loader function '{loader_function}' in '{loader_module}' is not callable."
            )

        return load_fn

    @staticmethod
    def _resolve_custom_loader_function(
        option_name: str,
        loader_module: str,
        loader_function: str,
        base_dir: Path,
    ):
        loader_module_mode, loader_module_path = OptionLibraryLoader._resolve_loader_module(
            loader_module=loader_module,
            base_dir=base_dir,
        )

        if loader_module_mode == "module":
            return OptionLibraryLoader._import_loader_function_by_name_from_module(
                loader_module=loader_module,
                loader_function=loader_function,
            )

        if loader_module_path is None:
            raise FileNotFoundError(
                f"Option '{option_name}' plugin loader module not found: {loader_module}"
            )

        return OptionLibraryLoader._import_loader_function_by_name(
            loader_module_path=loader_module_path,
            loader_function=loader_function,
        )

    @staticmethod
    def _import_loader_function_by_name(
        loader_module_path: Path,
        loader_function: str,
    ):
        # Create module spec from file
        spec = importlib.util.spec_from_file_location(
            name=f'option_loaders_custom_{loader_module_path.stem}',
            location=str(loader_module_path),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Failed to create module spec for {loader_module_path}"
            )

        # Load module
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except SyntaxError as e:
            raise SyntaxError(
                f"Syntax error in loader module {loader_module_path}: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error loading module {loader_module_path}: {e}"
            )

        if not hasattr(module, loader_function):
            available_funcs = [
                name for name in dir(module)
                if not name.startswith('_') and callable(getattr(module, name))
            ]
            raise RuntimeError(
                f"Loader module {loader_module_path} missing '{loader_function}'. "
                f"Available functions: {available_funcs}"
            )

        resolved_loader_function = getattr(module, loader_function)
        if not callable(resolved_loader_function):
            raise RuntimeError(
                f"'{loader_function}' in {loader_module_path} is not callable"
            )

        return resolved_loader_function

    @staticmethod
    def _import_loader_function_by_name_from_module(
        loader_module: str,
        loader_function: str,
    ):
        try:
            module = importlib.import_module(loader_module)
        except Exception as e:
            raise RuntimeError(
                f"Error importing module {loader_module}: {e}"
            )

        if not hasattr(module, loader_function):
            available_funcs = [
                name for name in dir(module)
                if not name.startswith('_') and callable(getattr(module, name))
            ]
            raise RuntimeError(
                f"Loader module {loader_module} missing '{loader_function}'. "
                f"Available functions: {available_funcs}"
            )

        resolved_loader_function = getattr(module, loader_function)
        if not callable(resolved_loader_function):
            raise RuntimeError(
                f"'{loader_function}' in {loader_module} is not callable"
            )

        return resolved_loader_function

    @staticmethod
    def _resolve_loader_module(
        loader_module: str,
        base_dir: Path,
    ) -> Tuple[str, Path | None]:
        """Resolve loader module as file path or module path.

        Returns:
            Tuple[str, Path | None]: ("file" | "module", resolved_path_if_file)
        """
        has_path_separator = "/" in loader_module or "\\" in loader_module
        is_py_file = loader_module.endswith(".py")

        if os.path.isabs(loader_module) or is_py_file or has_path_separator:
            loader_path = Path(loader_module)
            if not loader_path.is_absolute():
                loader_path = (base_dir / loader_path).resolve()
            else:
                loader_path = loader_path.resolve()
            if not loader_path.exists():
                raise FileNotFoundError(f"Loader module not found: {loader_path}")
            return "file", loader_path

        relative_candidate = (base_dir / loader_module).resolve()
        if relative_candidate.exists():
            return "file", relative_candidate

        return "module", None
