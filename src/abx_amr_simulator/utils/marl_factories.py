"""Factory functions for building MARL training components from config.

Provides a layered set of factory functions that load a multi-agent YAML config
and construct the objects needed to train with MARLTrainer:

    load_marl_config          — loads YAML, injects _config_dir
    build_marl_env_from_config    — ABXAMRParallelEnv
    build_marl_wrapper_from_config — MARLOptionsWrapper
    build_marl_managers_from_config — {agent_id: PPO or RecurrentPPO}
    build_marl_training_run_from_config — composed entry point

Typical usage:

    wrapper, agents = build_marl_training_run_from_config("path/to/config.yaml")
    trainer = MARLTrainer(
        wrapper=wrapper,
        agents=agents,
        n_steps=config['training']['n_steps'],
        total_primitive_steps=config['training']['total_primitive_steps'],
        checkpoint_dir=Path("results/checkpoints"),
    )
    trainer.train()

Config format (YAML)::

    environment:
      shared:
        antibiotics_AMR_dict:
          A: {leak: 0.05, flatness_parameter: 1.0, ...}
          B: {leak: 0.05, ...}
        max_time_steps: 1000
      agents:
        - agent_id: agent_p
          n_patients: 6
          patient_generator: <inline dict or filename relative to _config_dir>
          reward_calculator:  <inline dict or filename relative to _config_dir>
          option_library: heuristic_three_abx_mixed_vis.yaml   # relative to _config_dir
        - agent_id: agent_n
          n_patients: 14
          ...
    training:
      n_steps: 128
      batch_size: 64
      n_epochs: 10
      learning_rate: 0.0003
      total_primitive_steps: 500000
      option_gamma: 0.99
      seed: 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.core.patient_generator import PatientGenerator
from abx_amr_simulator.core.reward_calculator import RewardCalculator
from abx_amr_simulator.core.base_patient_generator import PatientGeneratorBase
from abx_amr_simulator.core.base_reward_calculator import RewardCalculatorBase
from abx_amr_simulator.hrl import MARLOptionsWrapper
from abx_amr_simulator.hrl.option_loaders import OptionLibraryLoader
from abx_amr_simulator.utils.factories import build_patient_generator_from_spec
from abx_amr_simulator.utils.plugin_loader import load_plugin_component


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def resolve_runtime_path(value: str | Path, config_dir: str | Path) -> Path:
    """Resolve a config-authored path to an absolute path.

    If ``value`` is absolute, return it directly.
    Otherwise, resolve it relative to ``config_dir``.
    """
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    return (Path(config_dir).resolve() / raw_path).resolve()

def load_marl_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a MARL config YAML and inject _config_dir for path resolution.

    `_config_dir` is the directory containing the YAML file. Downstream factory
    functions use it to resolve relative filenames in agent config entries
    (patient_generator, reward_calculator, option_library).

    Args:
        config_path: Path to the MARL config YAML file.

    Returns:
        Full config dict with `_config_dir` key injected.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is empty or missing required top-level keys.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"MARL config not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"MARL config is empty: {path}")

    for key in ("environment", "training"):
        if key not in config:
            raise ValueError(
                f"MARL config missing required top-level key '{key}': {path}"
            )

    config["_config_dir"] = str(path.parent)
    return config


# --------------------------------------------------------------------------- #
# Per-agent component helpers
# --------------------------------------------------------------------------- #

def _load_component_config(
    value: Any,
    config_dir: str,
) -> Tuple[Dict[str, Any], Path]:
    """Return component config and the directory it should resolve plugin paths from.

    If `value` is a dict, return it directly (inline config).
    If `value` is a string, treat it as a YAML filename relative to `config_dir`
    and load it.

    Args:
        value: Inline dict or filename string from the agent config entry.
        config_dir: Directory to resolve relative filenames against.

    Returns:
        Tuple of (component config dict, component config directory path).

    Raises:
        FileNotFoundError: If a filename is given but the file does not exist.
        ValueError: If the loaded YAML is empty.
    """
    if isinstance(value, dict):
        return value, Path(config_dir).resolve()

    # Treat as a filename
    filepath = resolve_runtime_path(value=value, config_dir=config_dir)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Component config file not found: {filepath} "
            f"(resolved from '{value}' relative to '{config_dir}')"
        )
    with open(filepath, "r") as f:
        loaded = yaml.safe_load(f)
    if not loaded:
        raise ValueError(f"Component config file is empty: {filepath}")
    return loaded, filepath.parent.resolve()


def build_patient_generator_from_config(
    pg_value: Any,
    config_dir: str,
    seed: Optional[int],
) -> PatientGeneratorBase:
    """Build a PatientGenerator from an inline dict or filename.

    Public helper shared by MARL factory functions and single-agent tuning.
    Supports plugin-based patient generators (e.g. PersonalizedPredPatientGenerator)
    via the standard plugin loader.

    Args:
        pg_value: Inline config dict or filename string relative to config_dir.
        config_dir: Directory to resolve relative filenames against.
        seed: Training seed to inject (or None).

    Returns:
        Instantiated PatientGeneratorBase.
    """
    pg_config, pg_config_dir = _load_component_config(pg_value, config_dir)

    # Shallow-copy and inject _config_dir_hint so plugin subclasses can resolve
    # relative file paths without needing Path(__file__) workarounds.
    pg_config = dict(pg_config)
    pg_config["_config_dir_hint"] = str(pg_config_dir)

    # Support plugin-based patient generators
    plugin_result = load_plugin_component(
        component_config=pg_config,
        expected_base_class=PatientGeneratorBase,
        default_loader_fn_name="load_patient_generator_component",
        config_dir_hint=str(pg_config_dir),
    )
    if plugin_result is not None:
        return plugin_result

    # Support mixer-type patient generators (type: mixer)
    if pg_config.get("type") == "mixer":
        return build_patient_generator_from_spec(
            spec=pg_config,
            base_dir=pg_config_dir,
            seed=seed,
        )

    pg_config["seed"] = seed
    return PatientGenerator(config=pg_config)


def build_reward_calculator_from_config(
    rc_value: Any,
    config_dir: str,
    seed: Optional[int],
) -> RewardCalculatorBase:
    """Build a RewardCalculator from an inline dict or filename.

    Public helper shared by MARL factory functions and single-agent tuning.
    Each agent entry in a MARL config may specify a different reward calculator,
    supporting heterogeneous reward functions across agents.

    Args:
        rc_value: Inline config dict or filename string relative to config_dir.
        config_dir: Directory to resolve relative filenames against.
        seed: Training seed to inject (or None).

    Returns:
        Instantiated RewardCalculatorBase.
    """
    rc_config, rc_config_dir = _load_component_config(rc_value, config_dir)

    # Shallow-copy and inject _config_dir_hint so plugin subclasses can resolve
    # relative file paths without needing Path(__file__) workarounds.
    rc_config = dict(rc_config)
    rc_config["_config_dir_hint"] = str(rc_config_dir)

    # Support plugin-based reward calculators
    plugin_result = load_plugin_component(
        component_config=rc_config,
        expected_base_class=RewardCalculatorBase,
        default_loader_fn_name="load_reward_calculator_component",
        config_dir_hint=str(rc_config_dir),
    )
    if plugin_result is not None:
        return plugin_result

    rc_config["seed"] = seed
    return RewardCalculator(config=rc_config)


# --------------------------------------------------------------------------- #
# Public factory functions
# --------------------------------------------------------------------------- #

def build_marl_env_from_config(config: Dict[str, Any]) -> ABXAMRParallelEnv:
    """Build ABXAMRParallelEnv from a loaded MARL config dict.

    Reads `environment.shared` for the shared AMR dynamics config and
    `environment.agents` for per-agent patient generator, reward calculator,
    and patient count. Component configs may be inline dicts or filenames
    relative to `config['_config_dir']`.

    Args:
        config: MARL config dict as returned by `load_marl_config`.

    Returns:
        Fully instantiated ABXAMRParallelEnv.

    Raises:
        ValueError: If required config keys are missing or inconsistent.
    """
    env_config = config.get("environment", {})
    shared_config = env_config.get("shared", {})
    agent_entries = env_config.get("agents", [])
    training_config = config.get("training", {})
    config_dir = config.get("_config_dir", ".")
    seed = training_config.get("seed", None)

    if not shared_config.get("antibiotics_AMR_dict"):
        raise ValueError(
            "MARL config missing 'environment.shared.antibiotics_AMR_dict'."
        )
    if not agent_entries:
        raise ValueError(
            "MARL config 'environment.agents' must contain at least one agent."
        )

    agent_configs = []
    for entry in agent_entries:
        aid = str(entry["agent_id"])
        n_patients = int(entry["n_patients"])

        pg = build_patient_generator_from_config(entry["patient_generator"], config_dir, seed)
        rc = build_reward_calculator_from_config(entry["reward_calculator"], config_dir, seed)

        agent_configs.append({
            "agent_id": aid,
            "n_patients": n_patients,
            "patient_generator": pg,
            "reward_calculator": rc,
        })

    return ABXAMRParallelEnv(
        agent_configs=agent_configs,
        shared_env_config=shared_config,
        seed=seed,
    )


def build_marl_wrapper_from_config(
    config: Dict[str, Any],
    env: ABXAMRParallelEnv,
) -> MARLOptionsWrapper:
    """Build MARLOptionsWrapper from a loaded MARL config dict and a parallel env.

    Loads each agent's option library using `OptionLibraryLoader.load_library`,
    passing the corresponding RewardCalculator from the env directly (so antibiotic
    action encoding is consistent with the env's reward calculator).

    The `option_library` value for each agent is resolved as a filename relative
    to `config['_config_dir']`.

    Args:
        config: MARL config dict as returned by `load_marl_config`.
        env: Pre-built ABXAMRParallelEnv (from `build_marl_env_from_config`).

    Returns:
        Fully instantiated MARLOptionsWrapper.

    Raises:
        FileNotFoundError: If any option_library file is not found.
        ValueError: If required config keys are missing.
    """
    env_config = config.get("environment", {})
    agent_entries = env_config.get("agents", [])
    training_config = config.get("training", {})
    config_dir = config.get("_config_dir", ".")
    gamma = float(training_config.get("option_gamma", 0.99))

    option_libraries = {}
    for entry in agent_entries:
        aid = str(entry["agent_id"])
        lib_value = entry.get("option_library")
        if lib_value is None:
            raise ValueError(
                f"Agent '{aid}' config missing 'option_library' key."
            )

        lib_path = resolve_runtime_path(value=lib_value, config_dir=config_dir)
        if not lib_path.exists():
            raise FileNotFoundError(
                f"Option library not found: {lib_path} "
                f"(resolved from '{lib_value}' relative to '{config_dir}')"
            )

        rc = env._reward_calculators[aid]
        lib, _ = OptionLibraryLoader.load_library(
            library_config_path=str(lib_path),
            reward_calculator=rc,
        )
        option_libraries[aid] = lib

    return MARLOptionsWrapper(
        base_env=env,
        option_libraries=option_libraries,
        gamma=gamma,
    )


def build_marl_managers_from_config(
    config: Dict[str, Any],
    wrapper: MARLOptionsWrapper,
    agent_hyperparams: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Build one PPO or RecurrentPPO agent per agent in a MARLOptionsWrapper.

    Both HRL_PPO and HRL_RPPO algorithms are supported.  Each agent's
    ``algorithm`` field in the config selects which constructor to use;
    it defaults to ``HRL_PPO`` if omitted (backward-compatible).

    For HRL_RPPO agents, optional LSTM parameters can be specified in an
    ``lstm_kwargs`` dict nested under the agent entry::

        agents:
          - agent_id: agent_0
            algorithm: HRL_RPPO
            lstm_kwargs:
              lstm_hidden_size: 64
              n_lstm_layers: 1
              enable_critic_lstm: true
            policy_kwargs:
              net_arch: [64, 64]

    For either algorithm, an optional ``policy_kwargs.net_arch`` list may be
    specified to override the default MLP feature-extractor shape. For
    HRL_RPPO this is the pre-LSTM MLP; for HRL_PPO it is the policy/value
    MLP. When omitted, SB3's default ``MlpPolicy`` / ``MlpLstmPolicy`` shape
    is used.

    Base PPO hyperparameters are read from the ``training`` section and
    applied uniformly to all agents.  If ``agent_hyperparams`` is provided,
    per-agent overrides are merged on top of the shared defaults for the
    matching agent — this is the mechanism used by the training CLI to
    load tuning results.  Note: ``batch_size`` is always taken from the
    training config and is NOT overridden by per-agent params (it is not
    included in ``best_params.json``).

    The manager's discount factor is read from ``training.manager_gamma``,
    falling back to ``training.option_gamma`` and then to 0.99.  These are
    two different horizons: ``option_gamma`` discounts primitive rewards
    *within* an option as the wrapper aggregates them, while
    ``manager_gamma`` discounts *across* options and therefore governs
    whether long-run consequences enter the manager's objective at all.
    Set ``manager_gamma`` to control the manager without also changing how
    the wrapper aggregates option returns.

    Args:
        config: MARL config dict as returned by ``load_marl_config``.
        wrapper: Pre-built MARLOptionsWrapper (from
            ``build_marl_wrapper_from_config``).
        agent_hyperparams: Optional dict mapping agent_id → PPO kwargs dict.
            If provided, each agent's constructor is called with these values
            overlaid on top of the shared training-section defaults.  Agents
            not present in this dict use the shared defaults unchanged.

    Returns:
        Dict mapping agent_id → PPO or RecurrentPPO object, ready for use
        with MARLTrainer.

    Raises:
        ValueError: If any agent's ``algorithm`` is not ``'HRL_PPO'`` or
            ``'HRL_RPPO'``.
    """
    env_config = config.get("environment", {})
    agent_entries = env_config.get("agents", [])
    training_config = config.get("training", {})

    # Lazy imports to avoid circular dependency:
    # marl_factories -> training.train_marl -> training/__init__
    #   -> tune_marl_agent -> marl_factories
    from abx_amr_simulator.training.train_marl import (
        make_ppo_for_agent,
        make_recurrent_ppo_for_agent,
    )

    # Shared defaults from training section.
    #
    # `manager_gamma` vs `option_gamma` are two DIFFERENT discounts and only coincide
    # by historical accident:
    #
    #   option_gamma   discounts primitive rewards WITHIN one option, as the wrapper
    #                  aggregates them into that option's return. Over an option of ~20
    #                  primitive steps its influence is mild, and the option runs a fixed
    #                  heuristic policy that is not being optimised.
    #   manager_gamma  discounts ACROSS options, i.e. the manager's own horizon over the
    #                  episode. This is what decides whether long-run consequences (in this
    #                  simulator, AMR depletion) enter the objective at all.
    #
    # The single-agent path has always kept them apart — `hrl.option_gamma` feeds
    # OptionsWrapper while `agent_algorithm.ppo.gamma` feeds the PPO manager (see
    # utils/factories.py). The MARL path previously read `option_gamma` for both, leaving
    # the more consequential of the two with no config key of its own and settable only
    # through tuned per-agent hyperparameters.
    #
    # `manager_gamma` now takes precedence and falls back to `option_gamma`, so configs
    # written before this key existed resolve exactly as they did. Per-agent tuned
    # hyperparameters still override it, as they always have.
    shared_ppo_kwargs: Dict[str, Any] = {
        "n_steps": int(training_config.get("n_steps", 256)),
        "batch_size": int(training_config.get("batch_size", 64)),
        "n_epochs": int(training_config.get("n_epochs", 10)),
        "learning_rate": float(training_config.get("learning_rate", 3e-4)),
        "gamma": float(
            training_config.get(
                "manager_gamma", training_config.get("option_gamma", 0.99)
            )
        ),
        "seed": training_config.get("seed", None),
    }

    agents = {}
    for entry in agent_entries:
        aid = str(entry["agent_id"])
        algorithm = entry.get("algorithm", "HRL_PPO")

        # Start with shared defaults, then overlay per-agent tuning results.
        # batch_size is excluded from the overlay: it is not a tuning param.
        ppo_kwargs = dict(shared_ppo_kwargs)
        if agent_hyperparams and aid in agent_hyperparams:
            for k, v in agent_hyperparams[aid].items():
                if k != "batch_size":
                    ppo_kwargs[k] = v

        policy_kwargs_entry = entry.get("policy_kwargs", {}) or {}
        net_arch = policy_kwargs_entry.get("net_arch", None)

        if algorithm == "HRL_PPO":
            agents[aid] = make_ppo_for_agent(
                wrapper=wrapper,
                agent_id=aid,
                net_arch=net_arch,
                **ppo_kwargs,
            )
        elif algorithm == "HRL_RPPO":
            lstm_kwargs = entry.get("lstm_kwargs", {})
            agents[aid] = make_recurrent_ppo_for_agent(
                wrapper=wrapper,
                agent_id=aid,
                lstm_hidden_size=lstm_kwargs.get("lstm_hidden_size", 64),
                n_lstm_layers=lstm_kwargs.get("n_lstm_layers", 1),
                enable_critic_lstm=lstm_kwargs.get("enable_critic_lstm", True),
                net_arch=net_arch,
                **ppo_kwargs,
            )
        else:
            raise ValueError(
                f"Agent '{aid}': unsupported algorithm '{algorithm}'. "
                "Supported: 'HRL_PPO', 'HRL_RPPO'."
            )

    return agents


def build_marl_training_run_from_config(
    config_path: str | Path,
) -> Tuple[MARLOptionsWrapper, Dict[str, Any]]:
    """Load a MARL config YAML and construct all objects needed to start training.

    Composes load_marl_config → build_marl_env_from_config →
    build_marl_wrapper_from_config → build_marl_managers_from_config into a
    single call. Returns (wrapper, agents) ready to be passed to MARLTrainer.

    Args:
        config_path: Path to the MARL config YAML file.

    Returns:
        Tuple of:
          - wrapper: MARLOptionsWrapper over the constructed ABXAMRParallelEnv
          - agents: Dict mapping agent_id → PPO
    """
    config = load_marl_config(config_path)
    env = build_marl_env_from_config(config)
    wrapper = build_marl_wrapper_from_config(config, env)
    agents = build_marl_managers_from_config(config, wrapper)
    return wrapper, agents
