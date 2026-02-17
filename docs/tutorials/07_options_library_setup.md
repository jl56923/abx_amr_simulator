# Options Library Setup

This guide explains how to define option libraries and point each option at its loader.

## Library YAML structure (minimal)

```yaml
library_name: "example_library"
description: "Small example library"
options:
	- option_name: "HEURISTIC_aggressive_10"
		option_type: "heuristic"
		option_subconfig_file: "../option_types/heuristic/heuristic_option_default_config.yaml"
		loader_module: "abx_amr_simulator.options.heuristic_loader"
		config_params_override:
			duration: 10
			action_thresholds:
				prescribe_A: 0.3
				no_treatment: 0.0
```

## Standard loaders (built into the package)

Use these short alias module paths in `loader_module` for standard option types:

- **block**
	- Module path: `abx_amr_simulator.options.block_loader`
	- Expected loader function: `load_block_option`
	- Default subconfig: `block_option_default_config.yaml`

- **alternation**
	- Module path: `abx_amr_simulator.options.alternation_loader`
	- Expected loader function: `load_alternation_option`
	- Default subconfig: `alternation_option_default_config.yaml`

- **heuristic**
	- Module path: `abx_amr_simulator.options.heuristic_loader`
	- Expected loader function: `load_heuristic_option`
	- Default subconfig: `heuristic_option_default_config.yaml`

## Using standard loaders by option type

Each option entry needs:
- `option_type`: Must match the loader function name suffix (for example, `option_type: "block"` expects `load_block_option`).
- `loader_module`: Use one of the standard module paths above.
- `option_subconfig_file`: Still a file path. Use a relative path from the library YAML to a config file you include in your workspace, or an absolute path if you want to point directly at the packaged defaults.

### Example: block option

```yaml
- option_name: "BLOCK_A_10"
	option_type: "block"
	option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
	loader_module: "abx_amr_simulator.options.block_loader"
	config_params_override:
		duration: 10
		block_action: "prescribe_A"
```

### Example: alternation option

```yaml
- option_name: "ALT_A_B_10"
	option_type: "alternation"
	option_subconfig_file: "../option_types/alternation/alternation_option_default_config.yaml"
	loader_module: "abx_amr_simulator.options.alternation_loader"
	config_params_override:
		duration: 10
		first_action: "prescribe_A"
		second_action: "prescribe_B"
```

### Example: heuristic option

```yaml
- option_name: "HEURISTIC_aggressive_10"
	option_type: "heuristic"
	option_subconfig_file: "../option_types/heuristic/heuristic_option_default_config.yaml"
	loader_module: "abx_amr_simulator.options.heuristic_loader"
	config_params_override:
		duration: 10
		action_thresholds:
			prescribe_A: 0.3
			no_treatment: 0.0
```

## loader_module formats

`loader_module` supports two formats:

1. **Module path (recommended for standard loaders)**
	 - Example: `abx_amr_simulator.options.heuristic_loader`
	 - This avoids absolute file paths and works for typical installed packages.

2. **File path (recommended for custom loaders)**
	 - Example: `../option_types/heuristic/custom_heuristic_worker.py`
	 - Relative paths are resolved relative to the option library YAML file.
	 - Absolute paths are also supported.

## Custom option loader pattern

For custom options, keep your loader module in your workspace and point
`loader_module` at the file path. The loader must define a function named
`load_<option_type>_option` that returns an `OptionBase` subclass.
