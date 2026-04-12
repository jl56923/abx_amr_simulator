"""Hyperparameter tuning for a single agent extracted from a MARL config.

Tunes one agent at a time using a single-agent ABXAMREnv + OptionsWrapper
(not MARLOptionsWrapper). Each trial builds the env and wrapper in-process,
constructs a PPO with the suggested hyperparameters, calls ppo.learn(), and
evaluates the resulting policy. No subprocess or stdout parsing is needed.

The best_params.json output format matches the single-agent tune.py so that
build_marl_managers_from_config can load it directly.

Typical usage (from run_marl_experiment_set.py):

    config = load_marl_config("marl_lpp_set1_agent_n.yaml")
    best_params = run_marl_agent_tuning(
        config=config,
        agent_id="agent_n",
        tuning_config=yaml.safe_load(open("hrl_ppo_marl_tuning.yaml")),
        optimization_dir=Path("workspace/optimization"),
        run_name="1a_agent_n_tuning",
        n_trials=32,
        seed=42,
    )

CLI usage:

    python -m abx_amr_simulator.training.tune_marl_agent \\
        --marl-config path/to/marl_lpp_set1_agent_n.yaml \\
        --agent-id agent_n \\
        --tuning-config path/to/hrl_ppo_marl_tuning.yaml \\
        --optimization-dir path/to/optimization \\
        --run-name 1a_agent_n_tuning \\
        --n-trials 32 \\
        --seed 42 \\
        --skip-if-exists
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import yaml
from stable_baselines3 import PPO

from abx_amr_simulator.core.abx_amr_env import ABXAMREnv
from abx_amr_simulator.hrl.option_loaders import OptionLibraryLoader
from abx_amr_simulator.hrl.wrapper import OptionsWrapper
from abx_amr_simulator.utils.marl_factories import (
    build_patient_generator_from_config,
    build_reward_calculator_from_config,
    load_marl_config,
    resolve_runtime_path,
)

# ------------------------------------------------------------------ #
# Environment and wrapper builders
# ------------------------------------------------------------------ #

def build_single_agent_env_from_marl_config(
    config: Dict[str, Any],
    agent_id: str,
    tuning_n_patients: int = 20,
) -> ABXAMREnv:
    """Build a single-agent ABXAMREnv for the named agent from a MARL config.

    Used during tuning to isolate one agent in a standard single-agent env.
    The patient count is set to `tuning_n_patients` regardless of the agent's
    `n_patients` field in the config, so that the observation space dimension
    is fixed and large enough for meaningful PPO learning during tuning.

    For agents whose patient generator config contains `create_personal_pred: true`
    (i.e. PersonalizedPredPatientGenerator), `exact_covered_count` is set equal
    to `tuning_n_patients` so all patients are covered during tuning. This ensures
    Agent P's observation space is fully populated and tuning is insensitive to
    coverage fraction (which varies across experiments).

    For agents using a plain PatientGenerator, no such override is needed.

    Args:
        config: MARL config dict as returned by load_marl_config.
        agent_id: The agent whose config entry to extract.
        tuning_n_patients: Patient count to use during tuning (default 20).

    Returns:
        Fully instantiated ABXAMREnv for the named agent.

    Raises:
        ValueError: If agent_id is not found in config["environment"]["agents"].
    """
    env_config = config.get("environment", {})
    shared_config = env_config.get("shared", {})
    agent_entries = env_config.get("agents", [])
    config_dir = config.get("_config_dir", ".")
    seed = config.get("training", {}).get("seed", None)

    entry = next(
        (e for e in agent_entries if str(e["agent_id"]) == agent_id),
        None,
    )
    if entry is None:
        available = [str(e["agent_id"]) for e in agent_entries]
        raise ValueError(
            f"Agent '{agent_id}' not found in MARL config. "
            f"Available agent IDs: {available}"
        )

    pg_value = entry["patient_generator"]

    # For personalized generators, override exact_covered_count so all patients
    # are covered during tuning (tuning is insensitive to coverage fraction).
    pg_config = (
        pg_value if isinstance(pg_value, dict)
        else yaml.safe_load(
            open(resolve_runtime_path(value=pg_value, config_dir=config_dir))
        )
    )
    if pg_config.get("create_personal_pred", False):
        pg_value = dict(pg_config)
        pg_value["exact_covered_count"] = tuning_n_patients

    pg = build_patient_generator_from_config(pg_value, config_dir, seed)
    rc = build_reward_calculator_from_config(entry["reward_calculator"], config_dir, seed)

    return ABXAMREnv(
        reward_calculator=rc,
        patient_generator=pg,
        antibiotics_AMR_dict=shared_config.get("antibiotics_AMR_dict"),
        crossresistance_matrix=shared_config.get("crossresistance_matrix", None),
        num_patients_per_time_step=tuning_n_patients,
        update_visible_AMR_levels_every_n_timesteps=shared_config.get(
            "update_visible_AMR_levels_every_n_timesteps", 1
        ),
        add_noise_to_visible_AMR_levels=shared_config.get(
            "add_noise_to_visible_AMR_levels", 0.0
        ),
        add_bias_to_visible_AMR_levels=shared_config.get(
            "add_bias_to_visible_AMR_levels", 0.0
        ),
        max_time_steps=shared_config.get("max_time_steps", 500),
        include_steps_since_amr_update_in_obs=shared_config.get(
            "include_steps_since_amr_update_in_obs", False
        ),
    )


def build_single_agent_wrapper_from_marl_config(
    config: Dict[str, Any],
    agent_id: str,
    env: ABXAMREnv,
) -> OptionsWrapper:
    """Build a single-agent OptionsWrapper for the named agent from a MARL config.

    Resolves the agent's option_library path relative to the config's _config_dir
    and loads it using the reward calculator already embedded in the env.

    Args:
        config: MARL config dict as returned by load_marl_config.
        agent_id: The agent whose option_library to load.
        env: Pre-built ABXAMREnv for this agent.

    Returns:
        Fully instantiated OptionsWrapper.

    Raises:
        ValueError: If agent_id is not found or option_library key is missing.
        FileNotFoundError: If the option_library file does not exist.
    """
    env_config = config.get("environment", {})
    agent_entries = env_config.get("agents", [])
    config_dir = config.get("_config_dir", ".")
    gamma = float(config.get("training", {}).get("option_gamma", 0.99))

    entry = next(
        (e for e in agent_entries if str(e["agent_id"]) == agent_id),
        None,
    )
    if entry is None:
        available = [str(e["agent_id"]) for e in agent_entries]
        raise ValueError(
            f"Agent '{agent_id}' not found in MARL config. "
            f"Available agent IDs: {available}"
        )

    lib_value = entry.get("option_library")
    if lib_value is None:
        raise ValueError(
            f"Agent '{agent_id}' config missing 'option_library' key."
        )

    lib_path = resolve_runtime_path(value=lib_value, config_dir=config_dir)
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Option library not found: {lib_path} "
            f"(resolved from '{lib_value}' relative to '{config_dir}')"
        )

    lib, _ = OptionLibraryLoader.load_library(
        library_config_path=str(lib_path),
        reward_calculator=env.reward_calculator,
    )
    return OptionsWrapper(env=env, option_library=lib, gamma=gamma)


# ------------------------------------------------------------------ #
# Trial objective
# ------------------------------------------------------------------ #

def _run_eval_episodes(
    wrapper: OptionsWrapper,
    ppo: PPO,
    n_episodes: int,
    seed: int,
) -> float:
    """Run deterministic eval episodes and return mean cumulative reward.

    Args:
        wrapper: Single-agent OptionsWrapper to evaluate in.
        ppo: Trained PPO agent (deterministic=True).
        n_episodes: Number of eval episodes.
        seed: RNG seed for reproducibility.

    Returns:
        Mean cumulative reward across all eval episodes.
    """
    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = wrapper.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = ppo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = wrapper.step(int(action))
            ep_reward += float(reward)
            done = terminated or truncated
        total_rewards.append(ep_reward)
    return float(np.mean(total_rewards))


def _resolve_batch_size_for_n_steps(
    *,
    n_steps: int,
    requested_batch_size: int,
) -> int:
    """Return a PPO mini-batch size that divides n_steps.

    SB3 warns when batch_size does not divide the rollout buffer length
    (n_steps * n_envs). MARL tuning uses n_envs=1, so we enforce divisibility
    against n_steps directly to avoid truncated mini-batches.

    If requested_batch_size already divides n_steps, it is returned unchanged.
    Otherwise, returns the largest positive divisor of n_steps that is <=
    requested_batch_size.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    if requested_batch_size <= 0:
        raise ValueError(
            f"requested_batch_size must be > 0, got {requested_batch_size}"
        )

    capped = min(requested_batch_size, n_steps)
    if n_steps % capped == 0:
        return capped

    for candidate in range(capped, 0, -1):
        if n_steps % candidate == 0:
            return candidate

    raise ValueError(
        "Failed to resolve a valid batch size divisor. "
        f"n_steps={n_steps}, requested_batch_size={requested_batch_size}"
    )


def _make_objective(
    config: Dict[str, Any],
    agent_id: str,
    tuning_config: Dict[str, Any],
    seeds: list[int],
    n_eval_episodes: int,
    stability_penalty_weight: float,
    tuning_n_patients: int,
) -> Any:
    """Return an Optuna objective closure for one agent.

    Each call to the objective:
      1. Builds a fresh ABXAMREnv + OptionsWrapper for the agent.
      2. Trains a PPO with the suggested hyperparameters for
         `truncated_primitive_steps` total timesteps across all seeds.
      3. Evaluates the trained policy deterministically.
      4. Returns mean(rewards) - stability_penalty_weight * std(rewards).
    """
    opt_config = tuning_config["optimization"]
    truncated_steps = int(opt_config["truncated_primitive_steps"])
    search_space = tuning_config["search_space"]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_hyperparameters(trial, search_space)
        seed_rewards = []

        for seed in seeds:
            env = build_single_agent_env_from_marl_config(
                config=config,
                agent_id=agent_id,
                tuning_n_patients=tuning_n_patients,
            )
            wrapper = build_single_agent_wrapper_from_marl_config(
                config=config,
                agent_id=agent_id,
                env=env,
            )

            resolved_n_steps = int(params.get("n_steps", 128))
            requested_batch_size = int(params.get("batch_size", 64))
            resolved_batch_size = _resolve_batch_size_for_n_steps(
                n_steps=resolved_n_steps,
                requested_batch_size=requested_batch_size,
            )

            ppo = PPO(
                policy="MlpPolicy",
                env=wrapper,
                learning_rate=params.get("learning_rate", 3e-4),
                n_steps=resolved_n_steps,
                batch_size=resolved_batch_size,
                n_epochs=params.get("n_epochs", 10),
                gamma=params.get("gamma", 0.99),
                gae_lambda=params.get("gae_lambda", 0.95),
                clip_range=params.get("clip_range", 0.2),
                ent_coef=params.get("ent_coef", 0.0),
                seed=seed,
                verbose=0,
            )
            ppo.learn(total_timesteps=truncated_steps)
            reward = _run_eval_episodes(
                wrapper=wrapper,
                ppo=ppo,
                n_episodes=n_eval_episodes,
                seed=seed,
            )
            seed_rewards.append(reward)
            wrapper.close()

        mean_r = float(np.mean(seed_rewards))
        std_r = float(np.std(seed_rewards)) if len(seed_rewards) > 1 else 0.0
        return mean_r - stability_penalty_weight * std_r

    return objective


def _suggest_hyperparameters(
    trial: optuna.Trial,
    search_space: Dict[str, Any],
) -> Dict[str, Any]:
    """Suggest hyperparameters for a trial from the search_space config."""
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        ptype = spec.get("type")
        if ptype == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown parameter type '{ptype}' for '{name}'")
    return params


# ------------------------------------------------------------------ #
# Early stopping callback
# ------------------------------------------------------------------ #

def _make_early_stopping_callback(
    warmup_trials: int,
    patience: int,
    min_delta: float,
) -> Any:
    """Return an Optuna callback that stops the study when improvement plateaus.

    Args:
        warmup_trials: Minimum number of trials before stopping is considered.
        patience: Stop after this many consecutive trials with no improvement
                  of at least min_delta.
        min_delta: Minimum improvement in best value to reset the patience counter.
    """
    state = {"best_value": float("-inf"), "trials_since_improvement": 0}

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if len(study.trials) < warmup_trials:
            return
        current_best = study.best_value
        if current_best > state["best_value"] + min_delta:
            state["best_value"] = current_best
            state["trials_since_improvement"] = 0
        else:
            state["trials_since_improvement"] += 1
        if state["trials_since_improvement"] >= patience:
            study.stop()

    return callback


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def run_marl_agent_tuning(
    config: Dict[str, Any],
    agent_id: str,
    tuning_config: Dict[str, Any],
    optimization_dir: Path,
    run_name: str,
    n_trials: Optional[int] = None,
    seed: int = 42,
    skip_if_exists: bool = False,
    overwrite_existing_study: bool = False,
    tuning_n_patients: int = 20,
) -> Dict[str, Any]:
    """Tune PPO hyperparameters for one agent extracted from a MARL config.

    Runs an Optuna study in-process. Each trial builds a fresh single-agent
    ABXAMREnv + OptionsWrapper, trains a PPO for a truncated primitive-step
    budget, and evaluates the result. Saves best_params.json and
    tuning_config.yaml to optimization_dir/run_name/.

    Args:
        config: MARL config dict (from load_marl_config).
        agent_id: Which agent to tune.
        tuning_config: Loaded tuning YAML dict (hrl_ppo_marl_tuning.yaml).
        optimization_dir: Base directory for optimization artifacts.
        run_name: Subfolder name under optimization_dir for this run.
        n_trials: Override number of trials (default: from tuning_config).
        seed: Base random seed (used across seeds_per_trial).
        skip_if_exists: If True and best_params.json exists, return it without
                        re-running the study.
        overwrite_existing_study: If True, delete any existing SQLite DB and
                                  start a fresh study.
        tuning_n_patients: Patient count override during tuning (default 20).

    Returns:
        Dict of best PPO hyperparameters (same format as best_params.json).
    """
    run_dir = Path(optimization_dir) / run_name
    best_params_path = run_dir / "best_params.json"

    if skip_if_exists and best_params_path.exists():
        print(f"Skipping tuning for {run_name}: best_params.json already exists.")
        with open(best_params_path) as f:
            return json.load(f)

    run_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing SQLite DB if overwriting
    db_path = run_dir / "optuna_study.db"
    if overwrite_existing_study and db_path.exists():
        db_path.unlink()
        print(f"Deleted existing study DB for fresh run: {db_path}")

    opt_config = tuning_config["optimization"]
    effective_n_trials = n_trials if n_trials is not None else int(opt_config["n_trials"])
    n_seeds = int(opt_config.get("n_seeds_per_trial", 3))
    n_eval_episodes = int(opt_config.get("n_eval_episodes", 3))
    stability_penalty = float(opt_config.get("stability_penalty_weight", 0.0))
    seeds = list(range(seed, seed + n_seeds))

    storage_url = f"sqlite:///{db_path}"
    sampler_name = opt_config.get("sampler", "TPE")
    sampler = (
        optuna.samplers.TPESampler(seed=seed)
        if sampler_name == "TPE"
        else optuna.samplers.RandomSampler(seed=seed)
    )

    study = optuna.create_study(
        study_name=run_name,
        storage=storage_url,
        direction=opt_config.get("direction", "maximize"),
        sampler=sampler,
        load_if_exists=not overwrite_existing_study,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, effective_n_trials - completed)
    if remaining == 0:
        print(f"Study '{run_name}' already has {completed} completed trials — nothing to run.")
    else:
        print(f"Running {remaining} trials for {run_name} (agent: {agent_id})")
        callbacks = []
        es_config = opt_config.get("early_stopping", {})
        if es_config.get("enabled", False):
            callbacks.append(_make_early_stopping_callback(
                warmup_trials=int(es_config.get("warmup_trials", 8)),
                patience=int(es_config.get("patience", 8)),
                min_delta=float(es_config.get("min_delta", 2.0)),
            ))

        objective = _make_objective(
            config=config,
            agent_id=agent_id,
            tuning_config=tuning_config,
            seeds=seeds,
            n_eval_episodes=n_eval_episodes,
            stability_penalty_weight=stability_penalty,
            tuning_n_patients=tuning_n_patients,
        )
        study.optimize(objective, n_trials=remaining, callbacks=callbacks)

    best_params = study.best_params
    print(f"Best params for {run_name}: {best_params}")

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    with open(run_dir / "tuning_config.yaml", "w") as f:
        yaml.dump(tuning_config, f, default_flow_style=False)

    study_summary = {
        "run_name": run_name,
        "agent_id": agent_id,
        "n_trials_completed": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]),
        "best_value": study.best_value,
        "best_params": best_params,
    }
    with open(run_dir / "study_summary.json", "w") as f:
        json.dump(study_summary, f, indent=2)

    return best_params


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune PPO hyperparameters for one agent from a MARL config."
    )
    parser.add_argument("--marl-config", required=True, help="Path to MARL config YAML.")
    parser.add_argument("--agent-id", required=True, help="Agent ID to tune.")
    parser.add_argument("--tuning-config", required=True, help="Path to tuning config YAML.")
    parser.add_argument(
        "--optimization-dir", required=True, help="Base directory for optimization artifacts."
    )
    parser.add_argument("--run-name", required=True, help="Subfolder name for this run.")
    parser.add_argument("--n-trials", type=int, default=None, help="Override number of trials.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--tuning-n-patients", type=int, default=20,
        help="Patient count override during tuning (default 20).",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip if best_params.json already exists.",
    )
    parser.add_argument(
        "--overwrite-existing-study", action="store_true",
        help="Delete existing Optuna DB and start fresh.",
    )
    args = parser.parse_args()

    config = load_marl_config(args.marl_config)
    with open(args.tuning_config) as f:
        tuning_config = yaml.safe_load(f)

    run_marl_agent_tuning(
        config=config,
        agent_id=args.agent_id,
        tuning_config=tuning_config,
        optimization_dir=Path(args.optimization_dir),
        run_name=args.run_name,
        n_trials=args.n_trials,
        seed=args.seed,
        skip_if_exists=args.skip_if_exists,
        overwrite_existing_study=args.overwrite_existing_study,
        tuning_n_patients=args.tuning_n_patients,
    )


if __name__ == "__main__":
    _main()
