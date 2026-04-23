# Config Path Conventions: `$CONFIG_BASE_FOLDER/`

## The problem

YAML config files sometimes need to reference other config files — for example, a
mixer patient generator that lists two child generator YAMLs:

```yaml
type: mixer
generators:
  - config_file: ../../shared/patient_generators/low_risk.yaml
    proportion: 0.5
  - config_file: ../../shared/patient_generators/high_risk.yaml
    proportion: 0.5
```

A relative path like `../../shared/patient_generators/low_risk.yaml` is fragile: it
only works when the YAML is loaded from a specific directory depth.  Move the file,
reorganize your project, or let training code save and reload the config from a
different directory, and the path breaks silently.

## The solution: `$CONFIG_BASE_FOLDER/`

Set the environment variable `ABX_AMR_CONFIG_BASE_FOLDER` to whichever directory you
want to use as the root for shared config resolution.  Then write config file paths
using the `$CONFIG_BASE_FOLDER/` prefix:

```yaml
type: mixer
generators:
  - config_file: $CONFIG_BASE_FOLDER/patient_generators/low_risk.yaml
    proportion: 0.5
  - config_file: $CONFIG_BASE_FOLDER/patient_generators/high_risk.yaml
    proportion: 0.5
```

At runtime, `$CONFIG_BASE_FOLDER/patient_generators/low_risk.yaml` resolves to
`$ABX_AMR_CONFIG_BASE_FOLDER/patient_generators/low_risk.yaml` — wherever you set
the variable to point.  The package imposes no constraints on what that directory is
or how your project is structured.

### Example

```bash
# Point the base folder at your project's shared config directory
export ABX_AMR_CONFIG_BASE_FOLDER=/home/alice/myproject/shared_configs
```

```yaml
# Now this resolves to /home/alice/myproject/shared_configs/pg/baseline.yaml
config_file: $CONFIG_BASE_FOLDER/pg/baseline.yaml
```

---

## How it works in the package

`resolve_config_path(path_str, base_dir)` in `abx_amr_simulator.utils.factories` is
the resolution function:

- Path starts with `$CONFIG_BASE_FOLDER/` → read `ABX_AMR_CONFIG_BASE_FOLDER`, prepend it.
- Path is absolute → return as-is.
- Path is relative → resolve against `base_dir`.

`build_patient_generator_from_spec` delegates every `config_file` entry to
`resolve_config_path`, so the prefix is handled transparently for mixer specs.

If `ABX_AMR_CONFIG_BASE_FOLDER` is not set and a path using the prefix is encountered,
`resolve_config_path` raises a `RuntimeError` explaining that the variable needs to be
set.

---

## Using `$CONFIG_BASE_FOLDER/` in your own factory code

If you write a factory function that processes file-reference fields, use
`resolve_config_path` instead of raw path joining so your code also handles the prefix:

```python
from abx_amr_simulator.utils.factories import resolve_config_path

# Handles $CONFIG_BASE_FOLDER/, absolute paths, and relative paths
cfg_path = resolve_config_path(raw_value, base_dir=base_dir)

# Only handles relative and absolute — breaks for $CONFIG_BASE_FOLDER/ values
cfg_path = (base_dir / raw_value).resolve()
```

---

## Pre-save absolutization in training scripts

When `train_marl.py` saves a resolved copy of the config to a run directory (before
reloading it for the training phase), it absolutizes all `config_file` values in
inline patient_generator dicts, resolving any `$CONFIG_BASE_FOLDER/` tokens in the
process.  The saved config therefore contains only absolute paths and requires no env
var at reload time.  This means saved run configs are fully reproducible even in an
environment where `ABX_AMR_CONFIG_BASE_FOLDER` is no longer set.

---

## See also

- `abx_amr_simulator.utils.factories.resolve_config_path` — the resolution function
- `abx_amr_simulator.utils.factories.build_patient_generator_from_spec` — uses
  `resolve_config_path` for every `config_file` entry in mixer specs
