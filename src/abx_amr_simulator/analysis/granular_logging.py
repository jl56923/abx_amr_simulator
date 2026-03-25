"""
Per-primitive-step granular logging for HRL policy analysis.

For HRL runs, the manager selects options (macro-actions) that execute for k
primitive steps each. Standard evaluation logs capture data at macro-step
granularity (one row per manager decision). This module provides a rollout
function that unpacks the per-primitive-step data already present in
OptionsWrapper's ``primitive_infos`` and ``primitive_actions`` info keys,
enabling per-patient, per-step analysis of prescribing behaviour.

Typical use case: confirming whether the HRL agent performs risk-stratified
triage (selectively treating high-risk patients while withholding treatment
from low-risk ones).

Output format
-------------
Each seed produces one ``granular_<seed_label>.npz`` file in
``<output_dir>/granular_logs/``.  The file is flat: one row per primitive
step, all episodes concatenated.

Top-level arrays (all length N = total primitive steps across all episodes):
    episode_ids         : (N,) int32   — episode index
    macro_step_ids      : (N,) int32   — macro-step index within episode
    primitive_step_ids  : (N,) int32   — primitive step within macro-step
    option_ids          : (N,) int32   — manager's selected option index
    option_names        : (N,) object  — human-readable option name
    actions             : (N, P) int32 — primitive prescription index per patient
    patients_infected   : (N, P) bool  — whether each patient was actually infected
    individual_rewards  : (N, P) float32 — per-patient reward at this step
    actual_amr_levels   : (N, A) float32 — true AMR for each antibiotic
    visible_amr_levels  : (N, A) float32 — observable AMR for each antibiotic
    patient_true        : (N, P, F) float32 — true patient attribute values
    patient_observed    : (N, P, F) float32 — observed (noisy/biased) attribute values

Metadata arrays:
    patient_attr_names  : (F,) object  — attribute name for each column in patient_true/observed
    antibiotic_names    : (A,) object  — antibiotic name for each column in amr arrays

where P = num_patients_per_time_step, A = num_antibiotics, F = num_patient_features.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==================== Environment helpers ====================

def _set_patient_logging(env: Any, value: bool) -> None:
    """Enable or disable full patient attribute logging on the base environment."""
    try:
        env.unwrapped.log_full_patient_attributes = value
    except AttributeError:
        pass


# ==================== Rollout ====================

def run_granular_hrl_rollout(
    model: Any,
    env: Any,
    num_episodes: int = 10,
) -> List[List[Dict[str, Any]]]:
    """Run evaluation episodes and collect per-primitive-step records.

    At each macro-step the OptionsWrapper returns ``info['primitive_infos']``
    (list of base-env info dicts, one per primitive step) and
    ``info['primitive_actions']`` (list of action arrays, one per primitive
    step).  This function unpacks those lists so every primitive step becomes
    its own record.

    The base environment must have ``log_full_patient_attributes = True``
    before this is called so that ``patient_full_data`` is present in each
    primitive info dict.  Use ``_set_patient_logging(env, True)`` before
    calling and ``_set_patient_logging(env, False)`` afterwards, or call
    ``run_hrl_granular_logging_branch`` which handles this automatically.

    Args:
        model: Trained policy implementing ``predict(obs, deterministic)``.
        env: OptionsWrapper-wrapped environment.
        num_episodes: Number of independent evaluation episodes to run.

    Returns:
        List of episodes; each episode is a list of per-primitive-step record
        dicts with keys: episode_idx, macro_step_idx, primitive_step_idx,
        option_id, option_name, primitive_actions, patient_full_data,
        patients_actually_infected, individual_rewards, actual_amr_levels,
        visible_amr_levels.
    """
    episodes = []

    for ep_idx in range(num_episodes):
        obs, _ = env.reset()
        done = False
        macro_step = 0
        episode_records: List[Dict[str, Any]] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray) and action.size == 1:
                action = int(action.item())

            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            option_id = info.get('option_id', action)
            option_name = info.get('option_name', str(option_id))
            primitive_infos = info.get('primitive_infos', [])
            primitive_actions = info.get('primitive_actions', [])

            for prim_step_idx, (prim_info, prim_action) in enumerate(
                zip(primitive_infos, primitive_actions)
            ):
                episode_records.append({
                    'episode_idx': ep_idx,
                    'macro_step_idx': macro_step,
                    'primitive_step_idx': prim_step_idx,
                    'option_id': option_id,
                    'option_name': option_name,
                    'primitive_actions': prim_action,
                    'patient_full_data': prim_info.get('patient_full_data', None),
                    'patients_actually_infected': prim_info.get('patients_actually_infected', None),
                    'individual_rewards': prim_info.get('individual_rewards', None),
                    'actual_amr_levels': prim_info.get('actual_amr_levels', {}),
                    'visible_amr_levels': prim_info.get('visible_amr_levels', {}),
                })

            macro_step += 1

        episodes.append(episode_records)

    return episodes


# ==================== Saving ====================

def _infer_metadata(
    all_records: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], int]:
    """Infer attr_names, abx_names, num_patients from first populated record."""
    attr_names: List[str] = []
    abx_names: List[str] = []
    num_patients = 1

    for rec in all_records:
        pfd = rec.get('patient_full_data')
        if pfd is not None and 'true' in pfd:
            attr_names = list(pfd['true'].keys())
            if attr_names:
                num_patients = len(pfd['true'][attr_names[0]])
            break

    for rec in all_records:
        amr = rec.get('actual_amr_levels', {})
        if amr:
            abx_names = sorted(amr.keys())
            break

    return attr_names, abx_names, num_patients


def save_granular_rollout_npz(
    episodes: List[List[Dict[str, Any]]],
    output_path: Path,
) -> None:
    """Save granular rollout data to a compressed npz file.

    The output is flat: one row per primitive step, all episodes concatenated.
    See module docstring for full array schema.

    Args:
        episodes: Return value of ``run_granular_hrl_rollout()``.
        output_path: Destination ``.npz`` path (parent directory must exist).
    """
    all_records = [rec for ep in episodes for rec in ep]
    if not all_records:
        return

    attr_names, abx_names, num_patients = _infer_metadata(all_records)

    N = len(all_records)
    num_attrs = len(attr_names)
    num_abx = len(abx_names)

    episode_ids          = np.zeros(N, dtype=np.int32)
    macro_step_ids       = np.zeros(N, dtype=np.int32)
    primitive_step_ids   = np.zeros(N, dtype=np.int32)
    option_ids           = np.zeros(N, dtype=np.int32)
    option_names_arr     = np.empty(N, dtype=object)
    actions_arr          = np.zeros((N, num_patients), dtype=np.int32)
    patients_infected    = np.zeros((N, num_patients), dtype=bool)
    individual_rewards   = np.full((N, num_patients), np.nan, dtype=np.float32)
    actual_amr_arr       = np.full((N, num_abx), np.nan, dtype=np.float32)
    visible_amr_arr      = np.full((N, num_abx), np.nan, dtype=np.float32)
    patient_true_arr     = np.zeros((N, num_patients, num_attrs), dtype=np.float32)
    patient_obs_arr      = np.zeros((N, num_patients, num_attrs), dtype=np.float32)

    for i, rec in enumerate(all_records):
        episode_ids[i]        = rec['episode_idx']
        macro_step_ids[i]     = rec['macro_step_idx']
        primitive_step_ids[i] = rec['primitive_step_idx']
        option_ids[i]         = rec['option_id']
        option_names_arr[i]   = rec['option_name']

        prim_action = rec['primitive_actions']
        if prim_action is not None:
            actions_arr[i] = np.array(prim_action, dtype=np.int32)[:num_patients]

        infected = rec['patients_actually_infected']
        if infected is not None:
            patients_infected[i] = np.array(infected, dtype=bool)[:num_patients]

        ind_rew = rec['individual_rewards']
        if ind_rew is not None:
            individual_rewards[i] = np.array(ind_rew, dtype=np.float32)[:num_patients]

        for abx_idx, abx_name in enumerate(abx_names):
            actual_amr_arr[i, abx_idx]  = rec['actual_amr_levels'].get(abx_name, np.nan)
            visible_amr_arr[i, abx_idx] = rec['visible_amr_levels'].get(abx_name, np.nan)

        pfd = rec['patient_full_data']
        if pfd is not None and attr_names:
            for attr_idx, attr_name in enumerate(attr_names):
                true_vals = np.array(pfd['true'].get(attr_name, [0.0] * num_patients), dtype=np.float32)
                obs_vals  = np.array(pfd['observed'].get(attr_name, [0.0] * num_patients), dtype=np.float32)
                patient_true_arr[i, :, attr_idx] = true_vals[:num_patients]
                patient_obs_arr[i, :, attr_idx]  = obs_vals[:num_patients]

    save_dict: Dict[str, Any] = {
        'episode_ids':        episode_ids,
        'macro_step_ids':     macro_step_ids,
        'primitive_step_ids': primitive_step_ids,
        'option_ids':         option_ids,
        'option_names':       option_names_arr,
        'actions':            actions_arr,
        'patients_infected':  patients_infected,
        'individual_rewards': individual_rewards,
        'actual_amr_levels':  actual_amr_arr,
        'visible_amr_levels': visible_amr_arr,
        'patient_true':       patient_true_arr,
        'patient_observed':   patient_obs_arr,
    }
    if attr_names:
        save_dict['patient_attr_names'] = np.array(attr_names, dtype=object)
    if abx_names:
        save_dict['antibiotic_names'] = np.array(abx_names, dtype=object)

    np.savez_compressed(output_path, **save_dict)


# ==================== Branch orchestrator ====================

def run_hrl_granular_logging_branch(
    models_and_configs: List[Tuple],
    output_dir: Path,
    num_episodes: int = 10,
    wrap_fn: Optional[Any] = None,
) -> bool:
    """Run granular primitive-step logging for all seeds of an HRL experiment.

    For each seed: creates a fresh environment, enables full patient attribute
    logging, runs ``num_episodes`` rollouts unpacking each macro-step into its
    constituent primitive steps, then saves a flat npz to
    ``output_dir/granular_logs/granular_<seed_label>.npz``.

    Args:
        models_and_configs: List of ``(seed, model, config, run_dir)`` tuples,
            as produced by the loading loop in ``evaluative_plots.py``.
        output_dir: Base evaluation output directory
            (``analysis_output/<prefix>/evaluation/``).
        num_episodes: Number of episodes to roll out per seed.
        wrap_fn: Callable ``wrap_environment_for_hrl(env, config, run_dir)``
            from ``evaluative_plots.py``.  If None, the base environment is
            used unwrapped (which will not have primitive_infos).

    Returns:
        True if at least one seed succeeded; False otherwise.
    """
    from abx_amr_simulator.utils import (
        create_reward_calculator,
        create_patient_generator,
        create_environment,
    )

    granular_dir = output_dir / "granular_logs"
    granular_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0

    for seed, model, config, run_dir in models_and_configs:
        label = f"seed_{seed}" if seed >= 0 else "run"
        print(f"    Granular logging for {label}...")

        try:
            rc  = create_reward_calculator(config=config)
            pg  = create_patient_generator(config=config)
            env = create_environment(config=config, reward_calculator=rc, patient_generator=pg)

            if wrap_fn is not None:
                wrapped = wrap_fn(env=env, config=config, run_dir=run_dir)
                if wrapped is not None:
                    env = wrapped

            _set_patient_logging(env, True)
            try:
                episodes = run_granular_hrl_rollout(
                    model=model,
                    env=env,
                    num_episodes=num_episodes,
                )
                total_steps = sum(len(ep) for ep in episodes)
                out_path = granular_dir / f"granular_{label}.npz"
                save_granular_rollout_npz(episodes=episodes, output_path=out_path)
                print(f"      Saved {out_path.name} ({total_steps} primitive steps across {num_episodes} episodes)")
                succeeded += 1
            finally:
                _set_patient_logging(env, False)
                env.close()

        except Exception as exc:
            print(f"      [WARN] Granular logging failed for {label}: {exc}")
            continue

    return succeeded > 0
