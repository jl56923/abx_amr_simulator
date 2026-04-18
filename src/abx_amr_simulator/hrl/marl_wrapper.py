"""Multi-agent HRL wrapper: async option execution over ABXAMRParallelEnv.

MARLOptionsWrapper wraps an ABXAMRParallelEnv and manages per-agent option
execution asynchronously. Each agent maintains its own active option and
remaining-steps counter; the wrapper advances primitive time to the nearest
option completion event, then returns manager-level transitions only for
the completing agents.

This is a plain Python class (not a PettingZoo subclass) because the event-
driven partial-step interface does not conform to the PettingZoo contract of
returning observations for all agents every step.

See ADDING_MARL_SUPPORT_TO_ABX_AMR_SIMULATOR.md Section 3.4 for the full
design spec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from abx_amr_simulator.core.abx_amr_parallel_env import ABXAMRParallelEnv
from abx_amr_simulator.hrl.options import OptionLibrary


class MARLOptionsWrapper:
    """Async multi-agent HRL wrapper over ABXAMRParallelEnv.

    Each agent independently selects options (macro-actions). When an agent's
    option completes, the wrapper returns a manager-level transition for that
    agent so the training loop can query a new option selection. Agents still
    executing their current option are not returned.

    The wrapper advances primitive time by exactly `min(_steps_remaining.values())`
    steps each call, so the inner primitive loop always terminates with at least
    one agent completing its option.

    Episode termination (truncation at max_time_steps) is handled by returning
    all agents simultaneously. Agents whose option had not finished are marked
    with `manager_clipped=True` and `manager_transition_trainable=False`.

    Attributes:
        base_env: The wrapped ABXAMRParallelEnv.
        option_libraries: Per-agent OptionLibrary, keyed by agent ID.
        gamma: Discount factor applied within each option's primitive steps.
        observation_spaces: Per-agent manager-level observation spaces.
        action_spaces: Per-agent manager-level action spaces (Discrete over options).
    """

    def __init__(
        self,
        base_env: ABXAMRParallelEnv,
        option_libraries: Dict[str, OptionLibrary],
        gamma: float = 0.99,
        front_edge_use_full_vector: bool = False,
    ) -> None:
        """Initialise MARLOptionsWrapper.

        Args:
            base_env: Pre-instantiated ABXAMRParallelEnv.
            option_libraries: Dict mapping agent_id → OptionLibrary. Every agent
                in base_env.possible_agents must have an entry.
            gamma: Discount factor for reward accumulation within an option.
            front_edge_use_full_vector: If True, append the full boundary cohort
                feature vector. If False (default), append per-attribute (mean, std).

        Raises:
            ValueError: If any agent is missing an option library, if abx_name_to_index
                maps differ across agents, or if option libraries are empty.
        """
        self.base_env = base_env
        self.option_libraries = option_libraries
        self.gamma = gamma
        self.front_edge_use_full_vector = front_edge_use_full_vector

        # Validate every agent has a library
        for aid in base_env.possible_agents:
            if aid not in option_libraries:
                raise ValueError(
                    f"No option library provided for agent '{aid}'. "
                    f"option_libraries keys: {sorted(option_libraries.keys())}"
                )

        # Validate consistent abx_name_to_index across all agents' libraries.
        # All agents prescribe from the same shared antibiotic set, so their
        # action-index mappings must agree.
        first_aid = base_env.possible_agents[0]
        ref_abx_map = option_libraries[first_aid].abx_name_to_index
        for aid in base_env.possible_agents[1:]:
            if option_libraries[aid].abx_name_to_index != ref_abx_map:
                raise ValueError(
                    f"Agent '{aid}' option library has abx_name_to_index "
                    f"{option_libraries[aid].abx_name_to_index} which differs from "
                    f"agent '{first_aid}' map {ref_abx_map}. All agents must share "
                    f"the same antibiotic action-index mapping."
                )

        self.antibiotic_names: List[str] = list(ref_abx_map.keys())
        self._action_index_to_abx: Dict[int, str] = {
            idx: name for name, idx in ref_abx_map.items()
        }

        # Build per-agent manager observation and action spaces
        self.observation_spaces: Dict[str, spaces.Box] = {}
        self.action_spaces: Dict[str, spaces.Discrete] = {}
        for aid in base_env.possible_agents:
            obs_dim = self._compute_observation_dimension(aid)
            self.observation_spaces[aid] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            self.action_spaces[aid] = spaces.Discrete(len(option_libraries[aid]))

        # Per-agent mutable state — initialised properly in reset()
        self._current_option_id: Dict[str, int] = {
            aid: -1 for aid in base_env.possible_agents
        }
        self._current_option = {aid: None for aid in base_env.possible_agents}
        self._steps_remaining: Dict[str, int] = {
            aid: 0 for aid in base_env.possible_agents
        }
        # True when the selected option's natural duration exceeded available steps
        # at selection time — used to set manager_clipped correctly at episode end.
        self._option_was_capped: Dict[str, bool] = {
            aid: False for aid in base_env.possible_agents
        }
        self._current_obs: Dict[str, Optional[np.ndarray]] = {
            aid: None for aid in base_env.possible_agents
        }

        # Per-agent state for building manager observations
        self._previous_option_id: Dict[str, int] = {
            aid: -1 for aid in base_env.possible_agents
        }
        self._consecutive_same_option_count: Dict[str, int] = {
            aid: 0 for aid in base_env.possible_agents
        }
        self._steps_since_prescribed: Dict[str, Dict[str, int]] = {
            aid: {abx: 0 for abx in self.antibiotic_names}
            for aid in base_env.possible_agents
        }
        self._last_aggregate_stats: Dict[str, Optional[np.ndarray]] = {
            aid: None for aid in base_env.possible_agents
        }
        self._last_amr_start: Dict[str, Optional[Dict[str, float]]] = {
            aid: None for aid in base_env.possible_agents
        }
        self._last_amr_end: Dict[str, Optional[Dict[str, float]]] = {
            aid: None for aid in base_env.possible_agents
        }

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Reset the environment and all per-agent option state.

        After reset, all agents have _steps_remaining == 0. The caller must
        immediately provide option selections for all agents and call step().

        Args:
            seed: Optional seed forwarded to base_env.reset().
            options: Optional options dict forwarded to base_env.reset().

        Returns:
            Tuple of (manager_obs_dict, info_dict) both keyed by agent ID.
        """
        obs_dict, info_dict = self.base_env.reset(seed=seed, options=options)

        current_amr = self._get_current_visible_amr_levels()
        for aid in self.base_env.possible_agents:
            self._current_obs[aid] = obs_dict[aid]
            self._current_option_id[aid] = -1
            self._current_option[aid] = None
            self._steps_remaining[aid] = 0
            self._option_was_capped[aid] = False
            self._previous_option_id[aid] = -1
            self._consecutive_same_option_count[aid] = 0
            self._steps_since_prescribed[aid] = {abx: 0 for abx in self.antibiotic_names}
            self._last_amr_start[aid] = current_amr
            self._last_amr_end[aid] = current_amr
            self._last_aggregate_stats[aid] = self._initialize_empty_aggregate_stats(aid)

        # Reset all option libraries
        for aid, lib in self.option_libraries.items():
            for option in lib.options.values():
                option.reset()

        manager_obs = {
            aid: self._build_manager_observation(aid)
            for aid in self.base_env.possible_agents
        }
        return manager_obs, info_dict

    def step(
        self,
        new_option_selections: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        """Advance primitive time to the nearest option completion event.

        Loads new option selections for agents that need them (those with
        _steps_remaining == 0), then executes exactly
        ``min(_steps_remaining.values())`` primitive steps. Returns manager-
        level transitions for all agents whose option just completed, plus
        all agents if the episode ended during the primitive loop.

        Args:
            new_option_selections: Dict mapping agent_id → option_id (int).
                Must supply a selection for every agent that currently has
                _steps_remaining == 0 (i.e. all agents after reset(), and
                completing agents from the previous step()).

        Returns:
            Five dicts keyed by completing-agent ID (or all agents on episode end):
            - manager_obs: Manager-level observations.
            - manager_rewards: Accumulated discounted rewards over the option.
            - manager_terminated: True if episode terminated naturally.
            - manager_truncated: True if max_time_steps was reached.
            - manager_infos: Per-agent info dicts.

        Raises:
            ValueError: If a required option selection is missing or invalid.
        """
        # Load new options for agents that need a selection
        for aid, option_id in new_option_selections.items():
            lib = self.option_libraries[aid]
            if not (0 <= option_id < len(lib)):
                raise ValueError(
                    f"Agent '{aid}': option_id {option_id} out of range "
                    f"[0, {len(lib) - 1}]."
                )
            option = lib.get_option(option_id)
            self._current_option_id[aid] = option_id
            self._current_option[aid] = option

            # Cap infinite-duration options at remaining episode steps
            if option.k == float("inf"):
                steps = self.base_env.max_time_steps - self.base_env.current_time_step
                natural_duration = float("inf")
            else:
                steps = int(option.k)
                natural_duration = steps
            available = self.base_env.max_time_steps - self.base_env.current_time_step
            # Record whether this option's natural duration would extend past episode end
            self._option_was_capped[aid] = natural_duration > available
            self._steps_remaining[aid] = max(1, min(steps, available))

        # All agents must have an active option before advancing
        for aid in self.base_env.possible_agents:
            if self._current_option[aid] is None:
                raise ValueError(
                    f"Agent '{aid}' has no active option. Provide a selection in "
                    f"new_option_selections before calling step()."
                )

        # Advance to the nearest completion event
        next_event = min(self._steps_remaining.values())
        next_event = max(1, next_event)  # defensive: never zero

        # Per-agent accumulators for this primitive loop
        tracked_patients: Dict[str, List[Dict[str, float]]] = {
            aid: [] for aid in self.base_env.possible_agents
        }
        accumulated_rewards: Dict[str, float] = {
            aid: 0.0 for aid in self.base_env.possible_agents
        }
        discounts: Dict[str, float] = {
            aid: 1.0 for aid in self.base_env.possible_agents
        }
        amr_starts: Dict[str, Dict[str, float]] = {
            aid: self._get_current_visible_amr_levels() for aid in self.base_env.possible_agents
        }
        primitive_actions_log: Dict[str, List[np.ndarray]] = {
            aid: [] for aid in self.base_env.possible_agents
        }
        primitive_infos_log: Dict[str, List[Dict]] = {
            aid: [] for aid in self.base_env.possible_agents
        }

        episode_done = False
        last_terminations: Dict[str, bool] = {aid: False for aid in self.base_env.possible_agents}
        last_truncations: Dict[str, bool] = {aid: False for aid in self.base_env.possible_agents}

        for _ in range(next_event):
            # Each agent's option decides primitive actions for its own cohort
            combined_actions: Dict[str, np.ndarray] = {}
            env_states: Dict[str, Dict[str, Any]] = {}
            for aid in self.base_env.possible_agents:
                env_state = self._build_env_state(aid)
                env_states[aid] = env_state
                action_strings = self._current_option[aid].decide(env_state)
                action_indices = self._convert_and_validate_actions(action_strings, aid)
                combined_actions[aid] = action_indices

            obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = (
                self.base_env.step(combined_actions)
            )

            for aid in self.base_env.possible_agents:
                self._current_obs[aid] = obs_dict[aid]
                accumulated_rewards[aid] += discounts[aid] * rewards_dict[aid]
                discounts[aid] *= self.gamma
                self._steps_remaining[aid] -= 1
                self._update_steps_since_prescribed(aid, combined_actions[aid])
                tracked_patients[aid].extend(env_states[aid]["patients"])
                primitive_actions_log[aid].append(combined_actions[aid])
                primitive_infos_log[aid].append(infos_dict[aid])

            last_terminations = terminations_dict
            last_truncations = truncations_dict

            if any(terminations_dict.values()) or any(truncations_dict.values()):
                episode_done = True
                break

        # Determine which agents return manager-level transitions
        if episode_done:
            returning_agents = list(self.base_env.possible_agents)
        else:
            returning_agents = [
                aid
                for aid in self.base_env.possible_agents
                if self._steps_remaining[aid] == 0
            ]

        # Build manager-level outputs for returning agents
        manager_obs: Dict[str, np.ndarray] = {}
        manager_rewards: Dict[str, float] = {}
        manager_terminated: Dict[str, bool] = {}
        manager_truncated: Dict[str, bool] = {}
        manager_infos: Dict[str, Dict] = {}

        for aid in returning_agents:
            self._last_aggregate_stats[aid] = self._compute_aggregate_stats(
                aid, tracked_patients[aid]
            )
            self._last_amr_start[aid] = amr_starts[aid]
            self._last_amr_end[aid] = self._get_current_visible_amr_levels()
            self._update_option_history(aid, self._current_option_id[aid])

            # An agent is clipped if:
            # (a) episode ended mid-option (steps still remaining), OR
            # (b) the option's natural duration exceeded available steps at selection
            #     time (was capped when loaded).
            mid_option = self._steps_remaining[aid] > 0
            manager_clipped = episode_done and (mid_option or self._option_was_capped[aid])
            steps_clipped = max(0, self._steps_remaining[aid]) if manager_clipped else 0

            manager_obs[aid] = self._build_manager_observation(aid)
            manager_rewards[aid] = accumulated_rewards[aid]
            manager_terminated[aid] = last_terminations.get(aid, False)
            manager_truncated[aid] = last_truncations.get(aid, False)
            manager_infos[aid] = {
                "option_id": self._current_option_id[aid],
                "option_name": self._current_option[aid].name,
                "option_duration": next_event,
                "primitive_actions": primitive_actions_log[aid],
                "primitive_infos": primitive_infos_log[aid],
                "manager_clipped": manager_clipped,
                "steps_clipped": steps_clipped,
                "manager_transition_trainable": not manager_clipped,
            }

        return manager_obs, manager_rewards, manager_terminated, manager_truncated, manager_infos

    # ---------------------------------------------------------------------- #
    # Per-agent env_state construction
    # ---------------------------------------------------------------------- #

    def _build_env_state(self, agent_id: str) -> Dict[str, Any]:
        """Build the env_state dict passed to an agent's option.decide().

        Mirrors the structure produced by the single-agent OptionsWrapper so
        that heuristic option implementations are reusable without change.

        Args:
            agent_id: The agent whose option will consume this env_state.

        Returns:
            Dict with keys: patients, num_patients, current_amr_levels,
            option_library, reward_calculator, patient_generator,
            use_relative_uncertainty.
        """
        pg = self.base_env._patient_generators[agent_id]
        rc = self.base_env._reward_calculators[agent_id]
        lib = self.option_libraries[agent_id]
        num_patients = self.base_env._agent_n_patients[agent_id]

        patients = self._extract_patients(agent_id)
        current_amr_levels = self._get_current_visible_amr_levels()
        use_relative_uncertainty = getattr(lib, "use_relative_uncertainty", True)

        return {
            "patients": patients,
            "num_patients": num_patients,
            "current_amr_levels": current_amr_levels,
            "option_library": lib,
            "reward_calculator": rc,
            "patient_generator": pg,
            "use_relative_uncertainty": use_relative_uncertainty,
        }

    def _extract_patients(self, agent_id: str) -> List[Dict[str, float]]:
        """Extract current patient dicts from the base env for one agent.

        Reads observed attribute values (with noise/bias) when available,
        falling back to true values.

        Args:
            agent_id: Agent whose patient cohort to extract.

        Returns:
            List of dicts mapping visible attribute name → float value.
        """
        pg = self.base_env._patient_generators[agent_id]
        env_patients = self.base_env.current_patients.get(agent_id, [])

        patients: List[Dict[str, float]] = []
        for patient in env_patients:
            patient_dict: Dict[str, float] = {}
            for attr in pg.visible_patient_attributes:
                obs_attr = f"{attr}_obs"
                if hasattr(patient, obs_attr):
                    patient_dict[attr] = float(getattr(patient, obs_attr))
                else:
                    patient_dict[attr] = float(getattr(patient, attr, 0.0))
            patients.append(patient_dict)

        if not patients:
            # Fallback: zeroed-out dicts if cohort is empty (e.g. before first reset)
            patients = [
                {attr: 0.0 for attr in pg.visible_patient_attributes}
                for _ in range(self.base_env._agent_n_patients[agent_id])
            ]

        return patients

    # ---------------------------------------------------------------------- #
    # Manager observation construction (per agent)
    # ---------------------------------------------------------------------- #

    def _compute_observation_dimension(self, agent_id: str) -> int:
        """Compute the manager observation dimension for one agent.

        Components match the single-agent OptionsWrapper:
        1. Aggregate patient stats: len(visible_attrs) * 4
        2. AMR trajectory: 2 * num_antibiotics
        3. Option history: 2 + num_antibiotics
        4. Front-edge cohort: num_patients * len(visible_attrs) if full_vector,
           else 2 * len(visible_attrs)

        Args:
            agent_id: Agent whose patient generator to inspect.

        Returns:
            Total integer dimension of the manager observation.
        """
        pg = self.base_env._patient_generators[agent_id]
        num_visible = len(pg.visible_patient_attributes)
        num_abx = len(self.antibiotic_names)
        num_patients = self.base_env._agent_n_patients[agent_id]

        aggregate_stats_dim = num_visible * 4
        amr_obs_dim = 2 * num_abx
        option_history_dim = 2 + num_abx

        if self.front_edge_use_full_vector:
            front_edge_dim = num_patients * num_visible
        else:
            front_edge_dim = 2 * num_visible

        return aggregate_stats_dim + amr_obs_dim + option_history_dim + front_edge_dim

    def _build_manager_observation(self, agent_id: str) -> np.ndarray:
        """Assemble the manager-level observation vector for one agent.

        Args:
            agent_id: Agent to build observation for.

        Returns:
            1D float32 array of shape (observation_spaces[agent_id].shape[0],).
        """
        aggregate_stats = self._last_aggregate_stats[agent_id]
        if aggregate_stats is None:
            aggregate_stats = self._initialize_empty_aggregate_stats(agent_id)

        amr_start = self._last_amr_start[agent_id]
        if amr_start is None:
            amr_start = self._get_current_visible_amr_levels()

        amr_end = self._last_amr_end[agent_id]
        if amr_end is None:
            amr_end = self._get_current_visible_amr_levels()

        amr_obs = np.array(
            [amr_start.get(abx, 0.0) for abx in self.antibiotic_names]
            + [amr_end.get(abx, 0.0) for abx in self.antibiotic_names],
            dtype=np.float32,
        )

        option_history = np.array(
            [
                float(self._previous_option_id[agent_id]),
                float(self._consecutive_same_option_count[agent_id]),
            ]
            + [
                float(self._steps_since_prescribed[agent_id][abx])
                for abx in self.antibiotic_names
            ],
            dtype=np.float32,
        )

        front_edge = self._build_front_edge_features(agent_id)

        return np.concatenate([aggregate_stats, amr_obs, option_history, front_edge])

    def _build_front_edge_features(self, agent_id: str) -> np.ndarray:
        """Build front-edge patient cohort features for one agent.

        Args:
            agent_id: Agent whose current cohort to use.

        Returns:
            1D float32 array of summary stats (mean + std) or full vector.
        """
        pg = self.base_env._patient_generators[agent_id]
        visible_attrs = list(pg.visible_patient_attributes)
        num_patients = self.base_env._agent_n_patients[agent_id]

        if not visible_attrs:
            return np.zeros(0, dtype=np.float32)

        patients = self._extract_patients(agent_id)
        if not patients:
            if self.front_edge_use_full_vector:
                length = num_patients * len(visible_attrs)
            else:
                length = 2 * len(visible_attrs)
            return np.zeros(length, dtype=np.float32)

        if self.front_edge_use_full_vector:
            values = []
            for patient in patients:
                for attr in visible_attrs:
                    values.append(float(patient.get(attr, 0.0)))
            return np.array(values, dtype=np.float32)

        cohort_matrix = np.array(
            [
                [float(patient.get(attr, 0.0)) for attr in visible_attrs]
                for patient in patients
            ],
            dtype=np.float32,
        )
        means = np.mean(cohort_matrix, axis=0)
        stds = np.std(cohort_matrix, axis=0)
        stats: List[float] = []
        for i in range(len(visible_attrs)):
            stats.extend([float(means[i]), float(stds[i])])
        return np.array(stats, dtype=np.float32)

    def _initialize_empty_aggregate_stats(self, agent_id: str) -> np.ndarray:
        """Return a zeroed aggregate-stats array for one agent.

        Args:
            agent_id: Agent whose visible attributes to count.

        Returns:
            Zero array of shape (len(visible_attrs) * 4,).
        """
        pg = self.base_env._patient_generators[agent_id]
        n = len(pg.visible_patient_attributes)
        return np.zeros(n * 4, dtype=np.float32)

    def _compute_aggregate_stats(
        self,
        agent_id: str,
        tracked_patients: List[Dict[str, float]],
    ) -> np.ndarray:
        """Compute mean/std/max/min over all patients seen during an option.

        Args:
            agent_id: Agent whose visible attributes to use.
            tracked_patients: Accumulated patient dicts across all primitive steps.

        Returns:
            Float32 array of shape (len(visible_attrs) * 4,).
        """
        pg = self.base_env._patient_generators[agent_id]
        visible_attrs = list(pg.visible_patient_attributes)
        if not visible_attrs or not tracked_patients:
            return np.zeros(len(visible_attrs) * 4, dtype=np.float32)

        matrix = np.array(
            [
                [float(p.get(attr, 0.0)) for attr in visible_attrs]
                for p in tracked_patients
            ],
            dtype=np.float32,
        )
        means = np.mean(matrix, axis=0)
        stds = np.std(matrix, axis=0)
        maxs = np.max(matrix, axis=0)
        mins = np.min(matrix, axis=0)

        stats: List[float] = []
        for i in range(len(visible_attrs)):
            stats.extend([float(means[i]), float(stds[i]), float(maxs[i]), float(mins[i])])
        return np.array(stats, dtype=np.float32)

    # ---------------------------------------------------------------------- #
    # Per-agent tracking helpers
    # ---------------------------------------------------------------------- #

    def _get_current_visible_amr_levels(self) -> Dict[str, float]:
        """Return the current visible (observed) AMR levels from the base env.

        Visible AMR is the degraded signal that the agent is allowed to see:
        it updates only every ``update_visible_AMR_levels_every_n_timesteps``
        primitive steps, and includes noise and bias.  This is intentional —
        the manager must never see true AMR from the balloon models.

        Returns:
            Dict mapping antibiotic name → visible resistance level.
        """
        return dict(self.base_env.visible_amr_levels)

    def _update_option_history(self, agent_id: str, option_id: int) -> None:
        """Update per-agent consecutive-option tracking.

        Args:
            agent_id: Agent to update.
            option_id: Option ID that just completed.
        """
        if option_id == self._previous_option_id[agent_id]:
            self._consecutive_same_option_count[agent_id] += 1
        else:
            self._consecutive_same_option_count[agent_id] = 1
        self._previous_option_id[agent_id] = option_id

    def _update_steps_since_prescribed(
        self, agent_id: str, actions: np.ndarray
    ) -> None:
        """Update per-agent steps-since-prescribed counters.

        Args:
            agent_id: Agent whose actions were taken.
            actions: Integer action array for this agent at one primitive step.
        """
        prescribed: set = set()
        for action in actions:
            abx_name = self._action_index_to_abx.get(int(action))
            if abx_name is None or abx_name == "no_treatment":
                continue
            prescribed.add(abx_name)

        for abx in self.antibiotic_names:
            if abx in prescribed:
                self._steps_since_prescribed[agent_id][abx] = 0
            else:
                self._steps_since_prescribed[agent_id][abx] += 1

    def _convert_and_validate_actions(
        self, action_strings: np.ndarray, agent_id: str
    ) -> np.ndarray:
        """Convert antibiotic name strings to integer action indices.

        Args:
            action_strings: Array of antibiotic name strings from option.decide().
            agent_id: Agent whose option produced the actions (for error messages).

        Returns:
            Integer action array matching RewardCalculator's encoding.

        Raises:
            TypeError: If action_strings is not np.ndarray or wrong dtype.
            ValueError: If shape is wrong or contains unknown antibiotic names.
        """
        option_name = (
            self._current_option[agent_id].name
            if self._current_option[agent_id] is not None
            else "<unknown>"
        )

        if not isinstance(action_strings, np.ndarray):
            raise TypeError(
                f"Agent '{agent_id}' option '{option_name}': decide() returned "
                f"{type(action_strings).__name__}, expected np.ndarray."
            )

        num_patients = self.base_env._agent_n_patients[agent_id]
        if action_strings.shape != (num_patients,):
            raise ValueError(
                f"Agent '{agent_id}' option '{option_name}': expected shape "
                f"({num_patients},), got {action_strings.shape}."
            )

        if not (
            action_strings.dtype == object
            or np.issubdtype(action_strings.dtype, np.str_)
        ):
            raise TypeError(
                f"Agent '{agent_id}' option '{option_name}': expected string dtype, "
                f"got {action_strings.dtype}."
            )

        lib = self.option_libraries[agent_id]
        valid_names = set(lib.abx_name_to_index.keys())
        action_indices: List[int] = []
        for i, action_str in enumerate(action_strings):
            name = str(action_str)
            if name not in valid_names:
                raise ValueError(
                    f"Agent '{agent_id}' option '{option_name}': invalid antibiotic "
                    f"name '{name}' for patient {i}. Valid: {sorted(valid_names)}."
                )
            action_indices.append(lib.abx_name_to_index[name])

        return np.array(action_indices, dtype=np.int32)
