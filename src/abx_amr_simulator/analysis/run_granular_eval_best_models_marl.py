"""Post-hoc granular re-evaluation script for MARL experiment seeds.

For each seed folder in --results-dir, this script:
  1. Detects `marl_full_agents_env_config.yaml` and `checkpoints/best_model_{aid}.zip`
     files to identify a valid MARL seed folder.
  2. Reconstructs `ABXAMRParallelEnv` + `MARLOptionsWrapper` + PPO agents from the
     YAML config using the Phase 5 factory functions.
  3. Runs `n_eval_episodes` deterministic evaluation episodes with
     `save_granular_trajectories=True` on the parallel env.
  4. Saves one NPZ per agent to `eval_logs/eval_granular_best_model_{aid}.npz`.

NPZ schema (per agent):
    antibiotic_names                        : (num_abx,) str array
    num_episodes                            : scalar
    episode_{N}/primitive_patient_true      : (macro_steps, max_substeps, patients, attrs)
    episode_{N}/primitive_patient_observed  : (macro_steps, max_substeps, patients, attrs)
    episode_{N}/primitive_patient_attrs     : (num_attrs,) str array
    episode_{N}/primitive_individual_rewards: (macro_steps, max_substeps, patients)
    episode_{N}/primitive_patients_actually_infected : (macro_steps, max_substeps, patients)
    episode_{N}/primitive_substep_counts    : (macro_steps,) int32
    episode_{N}/primitive_actions           : (macro_steps,) object — each element is a
                                              (actual_substeps, patients) int array
    episode_{N}/primitive_actual_amr_levels : (macro_steps, max_substeps, num_abx)
    episode_{N}/primitive_visible_amr_levels: (macro_steps, max_substeps, num_abx)

The schema mirrors the single-agent DetailedEvalCallback granular format so that
Phase 7's granular_metrics_core.py extension can build on familiar structure.

Usage (from repo root):
    conda run -p $CONDA_ENV \\
        python -m abx_amr_simulator.analysis.run_granular_eval_best_models_marl \\
        --results-dir /path/to/marl_results

For local testing with seed folders whose configs contain HPC-absolute paths:
    python -m abx_amr_simulator.analysis.run_granular_eval_best_models_marl \\
        --results-dir workspace/marl_results \\
        --n-eval-episodes 1 \\
        --path-remap /home/remote/jlee92/abx_amr_capacitor_rl:/local/repo/root
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np

from stable_baselines3 import PPO

from abx_amr_simulator.hrl.marl_wrapper import MARLOptionsWrapper
from abx_amr_simulator.utils.marl_factories import (
    build_marl_env_from_config,
    build_marl_managers_from_config,
    build_marl_wrapper_from_config,
    load_marl_config,
)

# Fixed config filename written by the Phase 6 training runner into each seed folder.
MARL_CONFIG_FILENAME = "marl_full_agents_env_config.yaml"

# Fixed output filename prefix — one NPZ per agent.
GRANULAR_EVAL_FILENAME_PATTERN = "eval_granular_best_model_{aid}.npz"


# --------------------------------------------------------------------------- #
# Seed folder discovery
# --------------------------------------------------------------------------- #

def discover_marl_seed_folders(*, results_dir: Path) -> List[Path]:
    """Return sorted list of MARL seed folder Paths in results_dir.

    A folder qualifies if it contains both:
      - `marl_full_agents_env_config.yaml`
      - at least one `checkpoints/best_model_*.zip` file

    Args:
        results_dir: Directory to search for seed folders.

    Returns:
        Sorted list of qualifying subdirectory Paths.
    """
    folders = []
    for candidate in sorted(results_dir.iterdir()):
        if not candidate.is_dir():
            continue
        config_path = candidate / MARL_CONFIG_FILENAME
        if not config_path.exists():
            continue
        best_models = list((candidate / "checkpoints").glob("best_model_*.zip"))
        if not best_models:
            continue
        folders.append(candidate)
    return folders


# --------------------------------------------------------------------------- #
# Config loading and path remapping
# --------------------------------------------------------------------------- #

def load_seed_config(*, seed_folder: Path) -> Dict:
    """Load marl_full_agents_env_config.yaml from a seed folder."""
    config_path = seed_folder / MARL_CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"MARL config not found: {config_path}")
    return load_marl_config(config_path)


def remap_config_paths(config: object, old_prefix: str, new_prefix: str) -> object:
    """Recursively replace old_prefix with new_prefix in all string values."""
    if isinstance(config, str):
        return config.replace(old_prefix, new_prefix)
    if isinstance(config, dict):
        return {k: remap_config_paths(v, old_prefix, new_prefix) for k, v in config.items()}
    if isinstance(config, list):
        return [remap_config_paths(item, old_prefix, new_prefix) for item in config]
    return config


# --------------------------------------------------------------------------- #
# Trajectory collection
# --------------------------------------------------------------------------- #

def _collect_eval_trajectories(
    wrapper: MARLOptionsWrapper,
    agents: Dict[str, PPO],
    n_episodes: int,
) -> Dict[str, List[List[Dict]]]:
    """Run deterministic eval episodes and collect per-agent granular trajectory data.

    For each episode, accumulates a list of macro-step dicts per agent. Each dict has:
        primitive_infos   : list of per-substep base-env info dicts (with patient_full_data,
                            individual_rewards, patients_actually_infected,
                            visible_amr_levels)
        primitive_actions : list of per-substep action arrays, each shape (n_patients,)

    Args:
        wrapper: MARLOptionsWrapper with save_granular_trajectories enabled on the
                 underlying ABXAMRParallelEnv.
        agents: Dict mapping agent_id → PPO.
        n_episodes: Number of complete episodes to run.

    Returns:
        Dict mapping agent_id → list of episodes, where each episode is a list of
        macro-step dicts.
    """
    agent_ids: List[str] = list(wrapper.base_env.possible_agents)
    agent_episodes: Dict[str, List[List[Dict]]] = {aid: [] for aid in agent_ids}

    for _ in range(n_episodes):
        per_agent_macro_steps: Dict[str, List[Dict]] = {aid: [] for aid in agent_ids}

        obs_dict, _ = wrapper.reset()
        last_obs = dict(obs_dict)

        # Initial option selections for every agent.
        pending: Dict[str, int] = {}
        for aid in agent_ids:
            obs = last_obs[aid][np.newaxis, :]
            action, _ = agents[aid].policy.predict(obs, deterministic=True)
            pending[aid] = int(action[0])

        episode_done = False
        while not episode_done:
            m_obs, _m_rew, m_term, m_trunc, m_info = wrapper.step(pending)

            for aid, info in m_info.items():
                per_agent_macro_steps[aid].append({
                    "primitive_infos": info.get("primitive_infos", []),
                    "primitive_actions": info.get("primitive_actions", []),
                })
                last_obs[aid] = m_obs[aid]

            episode_done = any(m_term.values()) or any(m_trunc.values())

            if not episode_done:
                pending = {}
                for aid in m_obs:
                    obs = m_obs[aid][np.newaxis, :]
                    action, _ = agents[aid].policy.predict(obs, deterministic=True)
                    pending[aid] = int(action[0])

        for aid in agent_ids:
            agent_episodes[aid].append(per_agent_macro_steps[aid])

    return agent_episodes


# --------------------------------------------------------------------------- #
# NPZ writer
# --------------------------------------------------------------------------- #

def _add_episode_arrays(
    save_dict: Dict,
    ep_prefix: str,
    macro_steps: List[Dict],
    antibiotic_names: List[str],
) -> None:
    """Build padded primitive-step arrays for one episode and insert into save_dict.

    Mirrors the logic of _save_primitive_patient_arrays() in
    abx_amr_simulator.callbacks, but reads from MARLOptionsWrapper step info
    dicts rather than single-agent OptionsWrapper trajectory dicts.

    Args:
        save_dict: Dict that npz arrays will be inserted into (mutated in place).
        ep_prefix: Key prefix, e.g. 'episode_0'.
        macro_steps: List of macro-step dicts for this episode, each with
                     'primitive_infos' and 'primitive_actions'.
    """
    if not macro_steps:
        return

    def _extract_amr_from_info(*, info: Dict, key_type: str) -> object:
        if key_type == "actual":
            key_candidates = (
                "actual_amr_levels",
                "true_amr_levels",
                "amr_levels",
            )
        elif key_type == "visible":
            key_candidates = (
                "visible_amr_levels",
                "observed_amr_levels",
            )
        else:
            raise ValueError(f"Unsupported AMR key_type: {key_type}")

        for candidate in key_candidates:
            if candidate in info:
                return info[candidate]
        return None

    # Unpack per-macro-step lists.
    prim_patient_full: List[List] = []
    prim_rewards: List[List] = []
    prim_infected: List[List] = []
    prim_actions: List[List] = []
    prim_actual_amr: List[List] = []
    prim_visible_amr: List[List] = []

    for ms in macro_steps:
        infos = ms["primitive_infos"]
        actions = ms["primitive_actions"]
        prim_patient_full.append([info.get("patient_full_data") for info in infos])
        prim_rewards.append([info.get("individual_rewards") for info in infos])
        prim_infected.append([info.get("patients_actually_infected") for info in infos])
        prim_actions.append(list(actions))
        prim_actual_amr.append(
            [_extract_amr_from_info(info=info, key_type="actual") for info in infos]
        )
        prim_visible_amr.append(
            [_extract_amr_from_info(info=info, key_type="visible") for info in infos]
        )

    num_macro_steps = len(prim_patient_full)

    # Infer attribute names and patient count from the first valid substep.
    attr_names = None
    num_patients = None
    num_attrs = None
    for macro_substeps in prim_patient_full:
        for substep_data in macro_substeps:
            if substep_data is not None and "true" in substep_data:
                attr_names = list(substep_data["true"].keys())
                num_patients = len(substep_data["true"][attr_names[0]])
                num_attrs = len(attr_names)
                break
        if attr_names is not None:
            break

    if attr_names is None:
        return

    substep_counts = [len(ms) for ms in prim_patient_full]
    max_substeps = max(substep_counts) if substep_counts else 0
    if max_substeps == 0:
        return

    true_arr = np.zeros(
        (num_macro_steps, max_substeps, num_patients, num_attrs), dtype=np.float64
    )
    obs_arr = np.zeros(
        (num_macro_steps, max_substeps, num_patients, num_attrs), dtype=np.float64
    )
    rewards_arr = np.zeros(
        (num_macro_steps, max_substeps, num_patients), dtype=np.float64
    )
    infected_arr = np.zeros(
        (num_macro_steps, max_substeps, num_patients), dtype=np.float64
    )
    actual_amr_arr = np.full(
        (num_macro_steps, max_substeps, len(antibiotic_names)), fill_value=np.nan, dtype=np.float64
    )
    visible_amr_arr = np.full(
        (num_macro_steps, max_substeps, len(antibiotic_names)), fill_value=np.nan, dtype=np.float64
    )

    saw_actual_amr = False
    saw_visible_amr = False

    def _coerce_amr_values(*, value: object) -> np.ndarray:
        if value is None:
            raise ValueError("AMR value is missing")

        if isinstance(value, dict):
            missing_abx = [abx for abx in antibiotic_names if abx not in value]
            if missing_abx:
                raise ValueError(
                    "AMR dict is missing antibiotic keys "
                    f"{missing_abx}; expected keys {antibiotic_names}"
                )
            return np.asarray([value[abx] for abx in antibiotic_names], dtype=float)

        arr = np.asarray(value, dtype=float)
        if arr.shape != (len(antibiotic_names),):
            raise ValueError(
                "AMR vector has invalid shape "
                f"{arr.shape}; expected ({len(antibiotic_names)},)"
            )
        return arr

    for macro_idx in range(num_macro_steps):
        for substep_idx, substep_data in enumerate(prim_patient_full[macro_idx]):
            if substep_data is not None and "true" in substep_data:
                for attr_idx, attr_name in enumerate(attr_names):
                    true_arr[macro_idx, substep_idx, :, attr_idx] = (
                        substep_data["true"][attr_name]
                    )
                    obs_arr[macro_idx, substep_idx, :, attr_idx] = (
                        substep_data["observed"][attr_name]
                    )

        for substep_idx, rewards in enumerate(prim_rewards[macro_idx]):
            if rewards is not None:
                rewards_arr[macro_idx, substep_idx, :] = rewards

        for substep_idx, infected in enumerate(prim_infected[macro_idx]):
            if infected is not None:
                infected_arr[macro_idx, substep_idx, :] = infected

        for substep_idx, amr_values in enumerate(prim_actual_amr[macro_idx]):
            if amr_values is not None:
                actual_amr_arr[macro_idx, substep_idx, :] = _coerce_amr_values(value=amr_values)
                saw_actual_amr = True

        for substep_idx, amr_values in enumerate(prim_visible_amr[macro_idx]):
            if amr_values is not None:
                visible_amr_arr[macro_idx, substep_idx, :] = _coerce_amr_values(value=amr_values)
                saw_visible_amr = True

    if not saw_actual_amr:
        raise ValueError(
            f"{ep_prefix}: no primitive actual AMR levels were logged; cannot write shared AMR series"
        )
    if not saw_visible_amr:
        raise ValueError(
            f"{ep_prefix}: no primitive visible AMR levels were logged; cannot write shared AMR series"
        )

    # primitive_actions: object array, each element is (actual_substeps, n_patients).
    actions_obj = np.empty(num_macro_steps, dtype=object)
    for macro_idx, macro_acts in enumerate(prim_actions):
        if macro_acts:
            actions_obj[macro_idx] = np.array(macro_acts, dtype=int)
        else:
            actions_obj[macro_idx] = np.zeros((0, num_patients), dtype=int)

    save_dict[f"{ep_prefix}/primitive_patient_true"] = true_arr
    save_dict[f"{ep_prefix}/primitive_patient_observed"] = obs_arr
    save_dict[f"{ep_prefix}/primitive_patient_attrs"] = np.array(attr_names)
    save_dict[f"{ep_prefix}/primitive_individual_rewards"] = rewards_arr
    save_dict[f"{ep_prefix}/primitive_patients_actually_infected"] = infected_arr
    save_dict[f"{ep_prefix}/primitive_substep_counts"] = np.array(
        substep_counts, dtype=np.int32
    )
    save_dict[f"{ep_prefix}/primitive_actions"] = actions_obj
    save_dict[f"{ep_prefix}/primitive_actual_amr_levels"] = actual_amr_arr
    save_dict[f"{ep_prefix}/primitive_visible_amr_levels"] = visible_amr_arr


def _save_agent_npz(
    *,
    agent_episodes: List[List[Dict]],
    output_path: Path,
    antibiotic_names: List[str],
) -> None:
    """Assemble and save the per-agent granular trajectory NPZ.

    Args:
        agent_episodes: List of episodes for this agent; each episode is a list
                        of macro-step dicts from _collect_eval_trajectories.
        output_path: Destination .npz path (parent directory will be created).
        antibiotic_names: Antibiotic names from the parallel env.
    """
    save_dict: Dict = {
        "antibiotic_names": np.array(antibiotic_names, dtype=object),
        "num_episodes": len(agent_episodes),
    }

    for ep_idx, macro_steps in enumerate(agent_episodes):
        _add_episode_arrays(
            save_dict=save_dict,
            ep_prefix=f"episode_{ep_idx}",
            macro_steps=macro_steps,
            antibiotic_names=antibiotic_names,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **save_dict)


# --------------------------------------------------------------------------- #
# Per-seed eval pipeline
# --------------------------------------------------------------------------- #

def run_granular_eval_for_marl_seed(
    *,
    seed_folder: Path,
    n_eval_episodes: int,
    force: bool,
    path_remap: Optional[Tuple[str, str]] = None,
) -> str:
    """Run granular eval for one MARL seed folder.

    Loads the config, reconstructs all components, runs deterministic eval
    with granular trajectory logging enabled, and writes per-agent NPZs.

    Args:
        seed_folder: Path to the seed folder containing marl_full_agents_env_config.yaml
                     and checkpoints/best_model_{aid}.zip files.
        n_eval_episodes: Number of deterministic evaluation episodes to run.
        force: If False and all output NPZs already exist, returns 'skip'.
        path_remap: Optional (old_prefix, new_prefix) tuple for config path remapping.

    Returns:
        'done' on success, 'skip' if all outputs exist and force is False.

    Raises:
        FileNotFoundError: If config or any best_model_{aid}.zip is missing.
        ValueError: If the config is malformed.
    """
    # Load and optionally remap config.
    config = load_seed_config(seed_folder=seed_folder)
    if path_remap is not None:
        old_prefix, new_prefix = path_remap
        config = remap_config_paths(config, old_prefix, new_prefix)

    # Determine agent IDs from config to check skip condition.
    agent_entries = config.get("environment", {}).get("agents", [])
    agent_ids = [str(entry["agent_id"]) for entry in agent_entries]

    output_paths = {
        aid: seed_folder / "eval_logs" / GRANULAR_EVAL_FILENAME_PATTERN.format(aid=aid)
        for aid in agent_ids
    }

    if not force and all(p.exists() for p in output_paths.values()):
        return "skip"

    # Reconstruct environment, wrapper, and agents.
    env = build_marl_env_from_config(config)
    wrapper = build_marl_wrapper_from_config(config, env)
    agents = build_marl_managers_from_config(config, wrapper)

    # Load best model weights into each agent.
    checkpoints_dir = seed_folder / "checkpoints"
    for aid in agent_ids:
        model_path = checkpoints_dir / f"best_model_{aid}.zip"
        if not model_path.exists():
            raise FileNotFoundError(
                f"best_model_{aid}.zip not found in {checkpoints_dir}"
            )
        agents[aid] = PPO.load(str(model_path))

    # Enable granular trajectory logging on the parallel env.
    wrapper.base_env.save_granular_trajectories = True

    # Run eval episodes and collect trajectory data.
    agent_episodes = _collect_eval_trajectories(
        wrapper=wrapper,
        agents=agents,
        n_episodes=n_eval_episodes,
    )

    # Write per-agent NPZs.
    antibiotic_names = list(wrapper.base_env.antibiotic_names)
    for aid in agent_ids:
        _save_agent_npz(
            agent_episodes=agent_episodes[aid],
            output_path=output_paths[aid],
            antibiotic_names=antibiotic_names,
        )

    return "done"


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #

def parse_path_remap(raw: Optional[str]) -> Optional[Tuple[str, str]]:
    """Parse 'OLD:NEW' path remap string into a (old, new) tuple."""
    if raw is None:
        return None
    parts = raw.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"--path-remap must be 'OLD_PREFIX:NEW_PREFIX', got: {raw!r}"
        )
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run best MARL models with granular trajectory logging."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing MARL seed folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing NPZ files.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=3,
        help="Number of eval episodes per seed (default: 3).",
    )
    parser.add_argument(
        "--path-remap",
        type=str,
        default=None,
        metavar="OLD:NEW",
        help=(
            "Optional path prefix remapping for configs saved with HPC-absolute paths. "
            "Format: 'OLD_PREFIX:NEW_PREFIX'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise ValueError(f"--results-dir does not exist: {results_dir}")

    path_remap = parse_path_remap(args.path_remap)

    seed_folders = discover_marl_seed_folders(results_dir=results_dir)
    if not seed_folders:
        print(f"[INFO] No MARL seed folders found in {results_dir}")
        return

    done_count = 0
    skip_count = 0
    error_count = 0

    for seed_folder in seed_folders:
        try:
            status = run_granular_eval_for_marl_seed(
                seed_folder=seed_folder,
                n_eval_episodes=args.n_eval_episodes,
                force=args.force,
                path_remap=path_remap,
            )
            if status == "skip":
                print(f"[SKIP]  {seed_folder.name}")
                skip_count += 1
            else:
                print(f"[DONE]  {seed_folder.name}")
                done_count += 1
        except Exception as exc:
            print(f"[ERROR] {seed_folder.name}: {exc}")
            error_count += 1

    total = len(seed_folders)
    print(
        f"\nGranular MARL eval complete: "
        f"done={done_count}, skipped={skip_count}, errors={error_count}, total={total}"
    )


if __name__ == "__main__":
    main()
