"""Evaluate fixed-prescribing (FP) heuristic policies on the MARL parallel env.

This is the multi-agent counterpart to
``workspace/scripts/pipeline/train_w_fixed_prescribing_rules.py``: instead of a single
agent on ``ABXAMREnv``, it runs ONE fixed-prescribing policy per agent on a single shared
``ABXAMRParallelEnv`` — e.g. one policy in ``agent_p``'s slot (covered) and one in
``agent_n``'s slot (uncovered) — so the heuristic baseline experiences exactly the same
shared communal-AMR coupling the trained HRL agents do. The number of FP policies equals
the number of agents in the MARL config (1 for single-active-agent sets, 2 for partial
coverage).

Per-agent granular NPZs are written in the same schema as
``run_granular_eval_best_models_marl.py`` (``eval_granular_best_model_{aid}.npz`` with
``episode_{N}/primitive_*`` keys), so the existing MARL equity / cross-experiment analysis
consumes FP output directly. FP acts at the primitive level (one action per step), so the
``max_substeps`` axis is always length 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from abx_amr_simulator.utils.marl_factories import (
    load_marl_config,
    build_marl_env_from_config,
)
from abx_amr_simulator.policies.fixed_prescribing_rules import POLICY_REGISTRY

GRANULAR_EVAL_FILENAME_PATTERN = "eval_granular_best_model_{aid}.npz"

# Per-substep scalar reward-calculator fields, mirrored from
# run_granular_eval_best_models_marl.py so FP NPZs carry the same schema the
# trained-MARL granular analyses (evaluative_plots_marl, equity) require.
_SCALAR_INFO_KEYS = (
    "total_reward",
    "overall_individual_reward_component",
    "normalized_individual_reward",
    "overall_community_reward_component",
    "normalized_community_reward",
    "count_clinical_benefits",
    "count_clinical_failures",
    "count_adverse_events",
)

# Scalar outcome tallies from the reward calculator's outcomes_breakdown.
_OUTCOME_SCALAR_KEYS = (
    "not_infected_no_treatment",
    "not_infected_treated",
    "infected_no_treatment",
)


def _coerce_scalar(text: str) -> Any:
    """Parse an override value string into a scalar (via YAML: handles int/float/bool/str)."""
    return yaml.safe_load(text)


def apply_override(config: Dict[str, Any], dotpath: str, value: Any) -> None:
    """Apply a single dot-path override into a nested dict/list config (in place).

    Supports integer indices for list traversal, e.g.
    ``environment.agents.0.patient_generator.personalized_auroc``.
    """
    keys = dotpath.split(".")
    node: Any = config
    for key in keys[:-1]:
        if isinstance(node, list):
            node = node[int(key)]
        else:
            node = node[key]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def build_fp_policy_for_agent(*, env, agent_id: str, policy_name: str):
    """Construct a fixed-prescribing policy for one agent from the env's per-agent components.

    The policy is built with the agent's own reward calculator, patient count, and the
    generator's actual ``visible_patient_attributes`` (which for a covered agent includes the
    personalized-prediction attributes), so ``predict()`` parses that agent's observation
    correctly. ``expected_reward_greedy`` reads only base attributes + communal AMR and ignores
    the prediction values — the established communal-only baseline behaviour.
    """
    reward_calculator = env._reward_calculators[agent_id]
    patient_generator = env._patient_generators[agent_id]
    n_patients = env._agent_n_patients[agent_id]

    if policy_name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{policy_name}'. Known: {sorted(POLICY_REGISTRY)}"
        )
    policy_class = POLICY_REGISTRY[policy_name]
    return policy_class(
        config={},
        reward_calculator=reward_calculator,
        num_patients_per_time_step=n_patients,
        visible_patient_attributes=list(patient_generator.visible_patient_attributes),
        antibiotic_names=list(reward_calculator.antibiotic_names),
    )


def _accumulate_step(store: Dict[str, list], *, agent_info: Dict[str, Any]) -> None:
    """Append one primitive step of granular data for a single agent."""
    pfd = agent_info.get("patient_full_data")
    if pfd is None:
        raise KeyError(
            "patient_full_data missing from step info — is save_granular_trajectories enabled?"
        )
    true_dict = pfd["true"]
    obs_dict = pfd["observed"]
    if store["attrs"] is None:
        store["attrs"] = list(true_dict.keys())
    attrs = store["attrs"]
    store["true"].append(np.column_stack([true_dict[a] for a in attrs]))
    store["observed"].append(np.column_stack([obs_dict[a] for a in attrs]))
    store["rewards"].append(np.asarray(agent_info["individual_rewards"], dtype=float))
    store["infected"].append(np.asarray(agent_info["patients_actually_infected"], dtype=float))
    # Shared per-antibiotic AMR levels (dicts keyed by antibiotic name), so the
    # evaluative-plots analysis can draw the same AMR series it does for trained MARL.
    store["actual_amr"].append(agent_info["actual_amr_levels"])
    store["visible_amr"].append(agent_info["visible_amr_levels"])
    # Scalar reward-calculator fields (total/individual/community reward components and
    # clinical-event counts), required by the evaluative-plots analysis.
    for key in _SCALAR_INFO_KEYS:
        store["scalars"][key].append(agent_info[key])
    # Reward-calculator outcomes breakdown: scalar tallies (needed by evaluative plots)
    # and per-antibiotic sensitive/resistant treated counts (needed by the DS-effective
    # equity metric for BOTH covered and uncovered agents).
    outcomes = agent_info.get("outcomes_breakdown")
    if outcomes is None:
        raise KeyError(
            "outcomes_breakdown missing from step info — the reward calculator must "
            "return it (needed for treated-infection counts and outcome tallies)."
        )
    for key in _OUTCOME_SCALAR_KEYS:
        store["outcome_scalars"][key].append(outcomes[key])
    store["infected_treated"].append(outcomes["infected_treated"])


def run_fp_marl_eval(
    *,
    marl_config_path: str,
    output_dir: str,
    policy_name: str = "expected_reward_greedy",
    seed: int = 1,
    n_eval_episodes: int = 1,
    overrides: List[str] | None = None,
) -> Dict[str, Path]:
    """Run the two-policy FP eval on the parallel env and write per-agent granular NPZs.

    Returns a dict mapping agent_id -> written NPZ path.
    """
    config = load_marl_config(marl_config_path)
    for entry in overrides or []:
        key, _, raw = entry.partition("=")
        apply_override(config, key.strip(), _coerce_scalar(raw.strip()))

    env = build_marl_env_from_config(config)
    env.save_granular_trajectories = True
    agent_ids = list(env.possible_agents)
    policies = {
        aid: build_fp_policy_for_agent(env=env, agent_id=aid, policy_name=policy_name)
        for aid in agent_ids
    }

    # Per-agent, per-episode granular stores.
    episodes: Dict[str, List[Dict[str, list]]] = {aid: [] for aid in agent_ids}

    for episode_index in range(n_eval_episodes):
        obs, _info = env.reset(seed=seed + episode_index)
        for policy in policies.values():
            policy.reset()
        store = {
            aid: {
                "true": [], "observed": [], "rewards": [], "infected": [],
                "actions": [], "infected_treated": [],
                "actual_amr": [], "visible_amr": [],
                "scalars": {key: [] for key in _SCALAR_INFO_KEYS},
                "outcome_scalars": {key: [] for key in _OUTCOME_SCALAR_KEYS},
                "attrs": None,
            }
            for aid in agent_ids
        }

        done = False
        while not done:
            actions: Dict[str, np.ndarray] = {}
            for aid in list(env.agents):
                rc = env._reward_calculators[aid]
                action_names, _ = policies[aid].predict(obs[aid])
                actions[aid] = np.array(
                    [rc.abx_name_to_index[name] for name in action_names], dtype=int
                )
            obs, _rewards, terminations, truncations, infos = env.step(actions)
            for aid in agent_ids:
                _accumulate_step(store[aid], agent_info=infos[aid])
                store[aid]["actions"].append(actions[aid])
            done = any(truncations.values()) or any(terminations.values())

        for aid in agent_ids:
            episodes[aid].append(store[aid])

    output_root = Path(output_dir)
    eval_logs = output_root / "eval_logs"
    eval_logs.mkdir(parents=True, exist_ok=True)
    antibiotic_names = np.array(list(env.antibiotic_names))

    written: Dict[str, Path] = {}
    for aid in agent_ids:
        save_dict: Dict[str, Any] = {"antibiotic_names": antibiotic_names}
        for ep_index, store in enumerate(episodes[aid]):
            prefix = f"episode_{ep_index}"
            # (steps, patients, attrs) -> (steps, 1, patients, attrs); substep axis = 1 for FP.
            true_arr = np.asarray(store["true"])[:, None, :, :]
            obs_arr = np.asarray(store["observed"])[:, None, :, :]
            rew_arr = np.asarray(store["rewards"])[:, None, :]
            inf_arr = np.asarray(store["infected"])[:, None, :]
            act_arr = np.asarray(store["actions"], dtype=int)[:, None, :]
            num_steps = true_arr.shape[0]
            save_dict[f"{prefix}/primitive_patient_true"] = true_arr
            save_dict[f"{prefix}/primitive_patient_observed"] = obs_arr
            save_dict[f"{prefix}/primitive_patient_attrs"] = np.array(store["attrs"])
            save_dict[f"{prefix}/primitive_individual_rewards"] = rew_arr
            save_dict[f"{prefix}/primitive_patients_actually_infected"] = inf_arr
            save_dict[f"{prefix}/primitive_actions"] = act_arr
            # FP acts at the primitive level: every macro step has exactly one substep.
            save_dict[f"{prefix}/primitive_substep_counts"] = np.ones(num_steps, dtype=int)
            # Shared AMR levels, shape (num_steps, 1, num_abx) — matches the trained-MARL
            # schema's (macro_steps, max_substeps, num_abx), ordered by antibiotic_names.
            abx_order = list(antibiotic_names)
            actual_amr_arr = np.array(
                [[[step[abx] for abx in abx_order]] for step in store["actual_amr"]],
                dtype=float,
            )
            visible_amr_arr = np.array(
                [[[step[abx] for abx in abx_order]] for step in store["visible_amr"]],
                dtype=float,
            )
            save_dict[f"{prefix}/primitive_actual_amr_levels"] = actual_amr_arr
            save_dict[f"{prefix}/primitive_visible_amr_levels"] = visible_amr_arr
            # Per-substep scalar reward/outcome fields, shape (num_steps, 1) to match the
            # trained-MARL schema's (macro_steps, max_substeps).
            for key in _SCALAR_INFO_KEYS:
                save_dict[f"{prefix}/primitive_{key}"] = np.asarray(
                    store["scalars"][key], dtype=float
                )[:, None]
            for key in _OUTCOME_SCALAR_KEYS:
                save_dict[f"{prefix}/primitive_{key}"] = np.asarray(
                    store["outcome_scalars"][key], dtype=float
                )[:, None]
            # Per-antibiotic treated-infection counts, shape (num_steps, 1) to match the
            # trained-MARL granular schema's (macro_steps, max_substeps) — already summed
            # over patients per step. Consumed by the DS-effective equity metric.
            infected_treated_steps = store["infected_treated"]
            for abx in antibiotic_names:
                sensitive_arr = np.array(
                    [[step[abx]["sensitive_infection_treated"]] for step in infected_treated_steps],
                    dtype=float,
                )
                resistant_arr = np.array(
                    [[step[abx]["resistant_infection_treated"]] for step in infected_treated_steps],
                    dtype=float,
                )
                save_dict[f"{prefix}/primitive_sensitive_infection_treated/{abx}"] = sensitive_arr
                save_dict[f"{prefix}/primitive_resistant_infection_treated/{abx}"] = resistant_arr
        out_path = eval_logs / GRANULAR_EVAL_FILENAME_PATTERN.format(aid=aid)
        np.savez_compressed(str(out_path), **save_dict)
        written[aid] = out_path

    # The MARL equity analysis reads the run's resolved MARL config (by the canonical name
    # marl_full_agents_env_config.yaml) to infer per-agent coverage; drop a copy in the run dir.
    config_out = output_root / "marl_full_agents_env_config.yaml"
    config_to_dump = {k: v for k, v in config.items() if k != "_config_dir"}
    config_out.write_text(yaml.safe_dump(config_to_dump, sort_keys=False), encoding="utf-8")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-prescribing policies on the MARL parallel env."
    )
    parser.add_argument("--marl-config", required=True, help="Path to the MARL config YAML.")
    parser.add_argument("--output-dir", required=True, help="Run output dir (eval_logs/ written here).")
    parser.add_argument("--policy-name", default="expected_reward_greedy")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument(
        "-p", "--override", action="append", default=[],
        help="Dot-path config override, e.g. -p environment.agents.0.patient_generator.personalized_auroc=0.5",
    )
    args = parser.parse_args()

    written = run_fp_marl_eval(
        marl_config_path=args.marl_config,
        output_dir=args.output_dir,
        policy_name=args.policy_name,
        seed=args.seed,
        n_eval_episodes=args.n_eval_episodes,
        overrides=args.override,
    )
    for aid, path in written.items():
        print(f"[DONE] agent {aid}: wrote {path}")


if __name__ == "__main__":
    main()
