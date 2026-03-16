# Tutorial 09: Options Library Setup

**Goal**: Create HRL option libraries using the current canonical/custom option contract.

**Prerequisites**: Complete the HRL quickstart tutorial and scaffold an `options/` folder.

---

## 1) Option library contract

An option library is a YAML file that defines the menu of high-level strategies available to the HRL manager.

Every option always has:
- `option_name`
- `option_type`
- `option_subconfig_file`
- optional `config_params_override`

Allowed `option_type` values are exactly:
- `block`
- `alternation`
- `heuristic`
- `custom`

There are two loading modes:

1. **Canonical built-ins**: `block`, `alternation`, `heuristic`
   - loaded by the package
   - must not include plugin loader fields
2. **Custom plugin options**: `custom`
   - loaded through the plugin seam
   - must include both `plugin.loader_module` and `plugin.loader_function`

This is a clean cutover. Legacy flat top-level `loader_module` or `loader_function` keys on an option spec are no longer supported and fail with a clear migration error.

---

## 2) Canonical option schema

Use the canonical schema for built-in `block`, `alternation`, and `heuristic` options.

```yaml
library_name: "my_library"
description: "Canonical options only"
version: "1.0"

options:
  - option_name: "A_10"
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    config_params_override:
      antibiotic: "A"
      duration: 10

  - option_name: "ALT_AB"
    option_type: "alternation"
    option_subconfig_file: "../option_types/alternation/alternation_option_default_config.yaml"
    config_params_override:
      sequence:
        - "A"
        - "B"

  - option_name: "HEURISTIC_risk_stratified"
    option_type: "heuristic"
    option_subconfig_file: "../option_types/heuristic/heuristic_option_default_config.yaml"
    config_params_override:
      duration: 10
      action_thresholds:
        no_treatment: 0.0
        A: 0.4
        B: 0.7
      attribute_weights:
        prob_infected: 1.0
```

### Canonical rules

- Keep `option_type` as one of `block`, `alternation`, or `heuristic`
- Keep the option YAML-backed through `option_subconfig_file`
- Use `config_params_override` only for per-option overrides
- Do not add `plugin`, `loader_module`, or `loader_function`

---

## 3) Custom option schema

Use `option_type: custom` for any option whose loader lives outside the package-owned canonical option families.

```yaml
library_name: "my_custom_library"
description: "Mixed canonical and custom options"
version: "1.0"

options:
  - option_name: "A_10"
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    config_params_override:
      antibiotic: "A"
      duration: 10

  - option_name: "CUSTOM_WEIGHTED_HEURISTIC"
    option_type: "custom"
    option_subconfig_file: "../option_types/heuristic/my_custom_heuristic_config.yaml"
    plugin:
      loader_module: "../option_types/heuristic/my_custom_heuristic.py"
      loader_function: "load_my_custom_option"
    config_params_override:
      duration: 10
      risk_weight: 0.75
```

### Custom rules

- `option_type` must be exactly `custom`
- `plugin.loader_module` is required
- `plugin.loader_function` is required
- `option_subconfig_file` is still required
- `config_params_override` remains optional

Relative paths in `option_subconfig_file` and `plugin.loader_module` are resolved relative to the option library YAML file.

---

## 4) What a custom loader receives

Custom option loaders receive the merged option config:
- base YAML from `option_subconfig_file`
- merged with `config_params_override`
- plus option metadata such as `option_name`

Loader shape:

```python
from typing import Any, Dict

from abx_amr_simulator.options.base_option import OptionBase


def load_my_custom_option(config: Dict[str, Any]) -> OptionBase:
    return MyCustomOption(
        name=config["option_name"],
        duration=config["duration"],
        risk_weight=config["risk_weight"],
    )
```

The loader must return an `OptionBase` instance. Returning the wrong type fails immediately.

---

## 5) End-to-end example

```yaml
library_name: "my_experiment"
description: "3 canonical options and 1 custom option"
version: "1.0"

options:
  - option_name: "no_treatment_5"
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    config_params_override:
      antibiotic: "no_treatment"
      duration: 5

  - option_name: "A_10"
    option_type: "block"
    option_subconfig_file: "../option_types/block/block_option_default_config.yaml"
    config_params_override:
      antibiotic: "A"
      duration: 10

  - option_name: "ALT_BA"
    option_type: "alternation"
    option_subconfig_file: "../option_types/alternation/alternation_option_default_config.yaml"
    config_params_override:
      sequence:
        - "B"
        - "A"

  - option_name: "CUSTOM_WEIGHTED_HEURISTIC"
    option_type: "custom"
    option_subconfig_file: "../option_types/heuristic/my_custom_heuristic_config.yaml"
    plugin:
      loader_module: "../option_types/heuristic/my_custom_heuristic.py"
      loader_function: "load_my_custom_option"
    config_params_override:
      duration: 10
      risk_weight: 0.75
```

Point your umbrella config at the library:

```yaml
hrl:
  option_library: option_libraries/my_experiment.yaml
```

Then train:

```bash
python -m abx_amr_simulator.training.train \
  --umbrella-config $(pwd)/configs/umbrella_configs/my_experiment.yaml \
  --seed 42
```

---

## 6) Fail-loud contract

Invalid option schemas fail before training proceeds.

| Invalid config shape | What happens |
|---|---|
| Unsupported `option_type` | Raises an error listing the allowed values: `block`, `alternation`, `heuristic`, `custom` |
| Canonical option includes `plugin` fields | Raises a schema error because canonical options must use the built-in path only |
| `custom` option omits `plugin.loader_module` or `plugin.loader_function` | Raises a schema error describing the missing required plugin field |
| Option still uses flat top-level `loader_module` or `loader_function` | Raises a migration error telling you to use `option_type: custom` with a nested `plugin:` block |
| Custom loader returns the wrong object type | Raises a type validation error immediately |

### Migration rule

This old shape is invalid and intentionally rejected:

```yaml
- option_name: "OLD_CUSTOM_OPTION"
  option_type: "custom"
  option_subconfig_file: "../option_types/heuristic/my_custom_heuristic_config.yaml"
  loader_module: "../option_types/heuristic/my_custom_heuristic.py"
  loader_function: "load_my_custom_option"
```

Replace it with:

```yaml
- option_name: "CUSTOM_OPTION"
  option_type: "custom"
  option_subconfig_file: "../option_types/heuristic/my_custom_heuristic_config.yaml"
  plugin:
    loader_module: "../option_types/heuristic/my_custom_heuristic.py"
    loader_function: "load_my_custom_option"
```

---

## 7) Useful design patterns

- **Duration sweep**: multiple block options with different `duration` values
- **Action coverage**: at least one option per primitive action
- **Heuristic baseline**: include a canonical `heuristic` option for comparison
- **Small debug library**: start with 3-5 options before expanding
- **Mixed libraries**: combine canonical options with a few `custom` plugin options

---

## 8) Troubleshooting

### "Option library file not found"

Check `hrl.option_library` and `options_folder_location` in the umbrella config.

### "Legacy loader keys are not supported"

You still have flat top-level `loader_module` or `loader_function` on an option spec. Migrate that option to `option_type: custom` with a nested `plugin:` block.

### "Missing required plugin field"

Your `custom` option is missing either `plugin.loader_module` or `plugin.loader_function`.

### "Custom loader module not found"

Verify `plugin.loader_module` points to a valid import path or valid file path relative to the option library YAML file.

---

## Key takeaways

1. Options now use a strict two-mode contract: canonical or custom.
2. Canonical options use only `option_name`, `option_type`, `option_subconfig_file`, and optional `config_params_override`.
3. Custom options must use `option_type: custom` plus `plugin.loader_module` and `plugin.loader_function`.
4. Every option still loads parameters from `option_subconfig_file`.
5. Legacy flat top-level loader keys are no longer supported.

**Next tutorials**:
- [07_hrl_diagnostics.md](07_hrl_diagnostics.md)
- [10_advanced_heuristic_worker_subclassing.md](10_advanced_heuristic_worker_subclassing.md)
