"""PettingZoo ParallelEnv for multi-agent antibiotic prescribing with shared AMR dynamics.

Each agent operates on a disjoint fixed-size patient cohort. All agents prescribe
simultaneously each step; their combined prescriptions update the shared AMR leaky
balloons once. This models the LPP two-agent design (Agent P: covered patients with
personalized predictions; Agent N: uncovered patients with no prediction features)
and generalises to an arbitrary number of agents.

The step() method uses an explicit combine → update → distribute structure so that
adding multi-locale support later requires only inserting locale routing between
the three stages (combine remains the same; update becomes per-locale; distribute
routes observations back per locale).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .base_amr_dynamics import AMRDynamicsBase
from .base_patient_generator import PatientGeneratorBase
from .base_reward_calculator import RewardCalculatorBase
from .leaky_balloon import AMR_LeakyBalloon
from .types import Patient


class ABXAMRParallelEnv(ParallelEnv):
    """Multi-agent antibiotic prescribing environment with shared AMR dynamics.

    Multiple agents act simultaneously each step. Each agent has its own:
    - Fixed-size patient cohort (sampled by its own patient generator)
    - Observation space (patient features + shared visible AMR levels)
    - Action space (one prescription per patient)
    - Reward calculator

    All agents share one set of AMR leaky balloons. Their combined prescriptions
    drive the shared resistance dynamics at every step.

    Attributes:
        agents: List of active agent ID strings.
        possible_agents: Full list of agent IDs (never changes).
        observation_spaces: Per-agent Box spaces.
        action_spaces: Per-agent MultiDiscrete spaces.
    """

    metadata = {"name": "abx_amr_parallel_env_v0"}

    def __init__(
        self,
        agent_configs: List[Dict[str, Any]],
        shared_env_config: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the parallel environment.

        Args:
            agent_configs: List of per-agent dicts. Each must contain:
                - 'agent_id' (str): unique identifier
                - 'n_patients' (int): fixed patient count per step
                - 'patient_generator' (PatientGeneratorBase): pre-instantiated
                - 'reward_calculator' (RewardCalculatorBase): pre-instantiated
            shared_env_config: AMR dynamics config dict. Required keys:
                - 'antibiotics_AMR_dict' (dict): maps antibiotic name →
                  {'leak', 'flatness_parameter', 'permanent_residual_volume',
                   'initial_amr_level'}
                - 'max_time_steps' (int)
                - 'update_visible_AMR_levels_every_n_timesteps' (int, default 1)
                Optional keys:
                - 'crossresistance_matrix' (dict, default None)
                - 'add_noise_to_visible_AMR_levels' (float, default 0.0)
                - 'add_bias_to_visible_AMR_levels' (float, default 0.0)
                - 'include_steps_since_amr_update_in_obs' (bool, default False)
            seed: Optional global seed for the shared RNG.

        Raises:
            ValueError: If configs are missing required keys or inconsistent.
        """
        super().__init__()

        # ------------------------------------------------------------------ #
        # Validate and store shared AMR config
        # ------------------------------------------------------------------ #
        antibiotics_AMR_dict: Dict[str, Dict] = shared_env_config["antibiotics_AMR_dict"]
        if not antibiotics_AMR_dict:
            raise ValueError("antibiotics_AMR_dict must contain at least one antibiotic.")

        self.antibiotic_names: List[str] = list(antibiotics_AMR_dict.keys())
        self.num_abx: int = len(self.antibiotic_names)
        self.max_time_steps: int = int(shared_env_config["max_time_steps"])
        self.amr_update_frequency: int = int(
            shared_env_config.get("update_visible_AMR_levels_every_n_timesteps", 1)
        )
        self.add_noise_to_visible_amr: float = float(
            shared_env_config.get("add_noise_to_visible_AMR_levels", 0.0)
        )
        self.add_bias_to_visible_amr: float = float(
            shared_env_config.get("add_bias_to_visible_AMR_levels", 0.0)
        )
        self.include_steps_since_amr_update_in_obs: bool = bool(
            shared_env_config.get("include_steps_since_amr_update_in_obs", False)
        )

        self._antibiotics_AMR_dict = antibiotics_AMR_dict
        self._crossresistance_matrix = self._build_crossresistance_matrix(
            crossresistance_dict=shared_env_config.get("crossresistance_matrix", None),
            antibiotic_names=self.antibiotic_names,
        )

        # ------------------------------------------------------------------ #
        # Validate and store per-agent config
        # ------------------------------------------------------------------ #
        if not agent_configs:
            raise ValueError("agent_configs must contain at least one agent.")

        agent_ids = [str(cfg["agent_id"]) for cfg in agent_configs]
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent_ids must be unique.")

        self._agent_n_patients: Dict[str, int] = {}
        self._patient_generators: Dict[str, PatientGeneratorBase] = {}
        self._reward_calculators: Dict[str, RewardCalculatorBase] = {}

        for cfg in agent_configs:
            aid = str(cfg["agent_id"])
            n = int(cfg["n_patients"])
            pg: PatientGeneratorBase = cfg["patient_generator"]
            rc: RewardCalculatorBase = cfg["reward_calculator"]

            if n <= 0:
                raise ValueError(f"Agent '{aid}': n_patients must be > 0, got {n}.")
            if not hasattr(pg, "sample") or not hasattr(pg, "observe") or not hasattr(pg, "obs_dim"):
                raise ValueError(
                    f"Agent '{aid}': patient_generator must implement sample(), observe(), obs_dim()."
                )
            if not hasattr(rc, "calculate_reward"):
                raise ValueError(
                    f"Agent '{aid}': reward_calculator must implement calculate_reward()."
                )
            if not hasattr(rc, "antibiotic_names"):
                raise ValueError(
                    f"Agent '{aid}': reward_calculator must expose antibiotic_names."
                )
            if set(rc.antibiotic_names) - {"no_treatment"} != set(self.antibiotic_names):
                raise ValueError(
                    f"Agent '{aid}': reward_calculator antibiotic names "
                    f"{sorted(set(rc.antibiotic_names) - {'no_treatment'})} "
                    f"do not match shared env antibiotic names {sorted(self.antibiotic_names)}."
                )

            self._agent_n_patients[aid] = n
            self._patient_generators[aid] = pg
            self._reward_calculators[aid] = rc

        # PettingZoo required attributes
        self.possible_agents: List[str] = list(agent_ids)
        self.agents: List[str] = list(agent_ids)

        # ------------------------------------------------------------------ #
        # Build observation and action spaces
        # ------------------------------------------------------------------ #
        self.observation_spaces: Dict[str, spaces.Box] = {}
        self.action_spaces: Dict[str, spaces.MultiDiscrete] = {}

        for aid in self.possible_agents:
            n = self._agent_n_patients[aid]
            pg = self._patient_generators[aid]
            # Bind antibiotic order so obs_dim() is accurate
            pg.bind_antibiotic_order(antibiotic_names=self.antibiotic_names)
            patient_obs_dim = int(pg.obs_dim(num_patients=n))
            amr_obs_dim = self.num_abx
            extra_dim = 1 if self.include_steps_since_amr_update_in_obs else 0
            total_obs_dim = patient_obs_dim + amr_obs_dim + extra_dim

            self.observation_spaces[aid] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_obs_dim,),
                dtype=np.float32,
            )
            # num_abx + 1 actions per patient (one per antibiotic + no_treatment)
            # Action encoding (from RewardCalculator): 0..num_abx-1 = prescribe that antibiotic,
            # num_abx = no_treatment. Total num_abx + 1 choices per patient.
            self.action_spaces[aid] = spaces.MultiDiscrete(
                [self.num_abx + 1] * n
            )

        # ------------------------------------------------------------------ #
        # Shared AMR balloon models
        # ------------------------------------------------------------------ #
        self.amr_balloon_models: Dict[str, AMRDynamicsBase] = {}
        for abx_name, params in antibiotics_AMR_dict.items():
            self.amr_balloon_models[abx_name] = AMR_LeakyBalloon(
                leak=float(params.get("leak", 0.05)),
                flatness_parameter=float(params.get("flatness_parameter", 1.0)),
                permanent_residual_volume=float(params.get("permanent_residual_volume", 0.0)),
                initial_amr_level=float(params.get("initial_amr_level", 0.0)),
            )

        # ------------------------------------------------------------------ #
        # Shared RNG and mutable state
        # ------------------------------------------------------------------ #
        self._seed: Optional[int] = seed
        self.np_random: np.random.Generator = np.random.default_rng(seed)

        self.visible_amr_levels: Dict[str, float] = {
            name: 0.0 for name in self.antibiotic_names
        }
        self.steps_since_amr_update: int = 0
        self.current_time_step: int = 0
        self.current_patients: Dict[str, List[Patient]] = {
            aid: [] for aid in self.possible_agents
        }

        # ------------------------------------------------------------------ #
        # Granular trajectory logging
        # ------------------------------------------------------------------ #
        # When save_granular_trajectories is True, step() appends one entry
        # per agent per step to episode_log.  The schema mirrors the
        # single-agent ABXAMREnv: each entry is a dict with 'true' and
        # 'observed' sub-dicts mapping attribute names to per-patient lists.
        #
        # episode_log structure:
        #   {agent_id: [step_0_patient_full_data, step_1_patient_full_data, ...]}
        #
        # The log is cleared on reset() so callers should read it before
        # calling reset() for the next episode.
        self.save_granular_trajectories: bool = False
        self.episode_log: Dict[str, List[Dict]] = {
            aid: [] for aid in self.possible_agents
        }

    # ---------------------------------------------------------------------- #
    # PettingZoo required method overrides
    # ---------------------------------------------------------------------- #

    def observation_space(self, agent: str) -> spaces.Box:
        """Return observation space for the given agent (PettingZoo API)."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.MultiDiscrete:
        """Return action space for the given agent (PettingZoo API)."""
        return self.action_spaces[agent]

    # ---------------------------------------------------------------------- #
    # reset()
    # ---------------------------------------------------------------------- #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Reset the environment to its initial state.

        Resets shared AMR balloons, reseeds RNG, draws each agent's first
        patient cohort, and returns initial observations.

        Args:
            seed: Optional seed; reseeds the shared RNG if provided.
            options: Unused (Gymnasium/PettingZoo API placeholder).

        Returns:
            Tuple of (obs_dict, info_dict) both keyed by agent ID.
        """
        # Reseed if requested
        if seed is not None:
            self._seed = seed
            tmp = np.random.default_rng(seed)
            self.np_random.bit_generator.state = tmp.bit_generator.state

        self.agents = list(self.possible_agents)
        self.current_time_step = 0
        self.steps_since_amr_update = 0
        self.episode_log = {aid: [] for aid in self.possible_agents}

        # Reset AMR balloons to initial levels
        for abx_name, params in self._antibiotics_AMR_dict.items():
            initial = float(params.get("initial_amr_level", 0.0))
            self.amr_balloon_models[abx_name].reset(initial_amr_level=initial)

        # Force-update visible AMR levels immediately
        self._update_visible_amr_levels(force=True)

        # Sample each agent's initial patient cohort
        true_amr = self._get_true_amr_levels()
        for aid in self.agents:
            self.current_patients[aid] = self._patient_generators[aid].sample(
                n_patients=self._agent_n_patients[aid],
                true_amr_levels=true_amr,
                rng=self.np_random,
            )

        obs_dict = {aid: self._build_obs(aid) for aid in self.agents}
        info_dict = {
            aid: {"visible_amr_levels": dict(self.visible_amr_levels)}
            for aid in self.agents
        }
        return obs_dict, info_dict

    # ---------------------------------------------------------------------- #
    # step()
    # ---------------------------------------------------------------------- #

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        """Execute one primitive timestep across all agents simultaneously.

        Implements the combine → update → distribute pattern:

        1. **Combine**: collect per-agent actions and sum prescriptions across
           all agents into a single per-antibiotic dose vector; apply the shared
           crossresistance matrix to get effective doses.
        2. **Update**: step each AMR balloon with its effective dose; update the
           shared visible AMR levels if the update interval has elapsed.
        3. **Distribute**: for each agent, compute reward from its own patients
           and its own actions; sample its next patient cohort; construct its
           next observation.

        Keeping these three stages visibly separate makes a future multi-locale
        extension straightforward — locale routing slots in between combine and
        update, and per-locale balloon updates replace the single shared update.

        Args:
            actions: Dict mapping agent_id → np.ndarray of shape (n_patients,)
                with integer prescription indices matching RewardCalculator's mapping:
                0..num_abx-1 = prescribe that antibiotic, num_abx = no_treatment.

        Returns:
            Five dicts all keyed by agent_id:
            - obs: next observations
            - rewards: scalar rewards
            - terminations: True if episode ended naturally (unused, always False)
            - truncations: True if max_time_steps reached
            - infos: per-step diagnostic info
        """
        # Remember patients from before this step (reward uses current cohort)
        patients_for_reward = {
            aid: list(self.current_patients[aid]) for aid in self.agents
        }

        # ------------------------------------------------------------------ #
        # COMBINE: compute combined effective doses across all agents
        # ------------------------------------------------------------------ #
        raw_prescription_counts: Dict[str, float] = {
            abx: 0.0 for abx in self.antibiotic_names
        }
        for aid in self.agents:
            rc = self._reward_calculators[aid]
            agent_actions = np.asarray(actions[aid], dtype=int)
            for action_idx in agent_actions:
                abx_name = rc.index_to_abx_name[int(action_idx)]
                if abx_name != "no_treatment":
                    raw_prescription_counts[abx_name] += 1.0

        # Apply crossresistance matrix to get effective doses per target antibiotic
        effective_doses: Dict[str, float] = {}
        for target_abx in self.antibiotic_names:
            total = 0.0
            for prescriber_abx in self.antibiotic_names:
                total += (
                    raw_prescription_counts[prescriber_abx]
                    * self._crossresistance_matrix[prescriber_abx][target_abx]
                )
            effective_doses[target_abx] = total

        # ------------------------------------------------------------------ #
        # UPDATE: step AMR balloons; refresh visible AMR levels if due
        # ------------------------------------------------------------------ #
        for abx_name in self.antibiotic_names:
            self.amr_balloon_models[abx_name].step(effective_doses[abx_name])

        self.steps_since_amr_update += 1
        self._update_visible_amr_levels(force=False)

        self.current_time_step += 1
        terminated_flag = False  # AMR environment has no natural termination
        truncated_flag = self.current_time_step >= self.max_time_steps

        # ------------------------------------------------------------------ #
        # DISTRIBUTE: per-agent rewards, next patient cohorts, observations
        # ------------------------------------------------------------------ #
        true_amr = self._get_true_amr_levels()

        obs: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Dict] = {}

        for aid in self.agents:
            rc = self._reward_calculators[aid]
            pg = self._patient_generators[aid]
            agent_actions = np.asarray(actions[aid], dtype=int)

            # Compute reward using patients from before this step
            reward_value, reward_info = rc.calculate_reward(
                patients=patients_for_reward[aid],
                actions=agent_actions,
                antibiotic_names=self.antibiotic_names,
                visible_amr_levels=self.visible_amr_levels,
                rng=self.np_random,
            )

            # Sample next patient cohort for this agent
            self.current_patients[aid] = pg.sample(
                n_patients=self._agent_n_patients[aid],
                true_amr_levels=true_amr,
                rng=self.np_random,
            )

            obs[aid] = self._build_obs(aid)
            rewards[aid] = float(reward_value)
            terminations[aid] = terminated_flag
            truncations[aid] = truncated_flag
            infos[aid] = {
                "visible_amr_levels": dict(self.visible_amr_levels),
                "effective_doses": dict(effective_doses),
                **reward_info,
            }

            # Granular logging: record full patient attribute data for this step
            if self.save_granular_trajectories:
                patient_full_data = self._extract_full_patient_attributes(
                    agent_id=aid,
                    patients=patients_for_reward[aid],
                )
                self.episode_log[aid].append(patient_full_data)
                infos[aid]["patient_full_data"] = patient_full_data

        if truncated_flag or terminated_flag:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _get_true_amr_levels(self) -> Dict[str, float]:
        """Return current true (latent) AMR levels from balloon models."""
        return {
            abx: self.amr_balloon_models[abx].get_volume()
            for abx in self.antibiotic_names
        }

    def _update_visible_amr_levels(self, *, force: bool) -> None:
        """Refresh visible AMR levels if the update interval has elapsed or forced.

        Args:
            force: If True, always update regardless of interval (used on reset).
        """
        if not force and self.steps_since_amr_update < self.amr_update_frequency:
            return

        updated: Dict[str, float] = {}
        for abx in self.antibiotic_names:
            level = self.amr_balloon_models[abx].get_volume()
            if self.add_noise_to_visible_amr > 0.0:
                level += float(self.np_random.normal(0.0, self.add_noise_to_visible_amr))
            level += self.add_bias_to_visible_amr
            updated[abx] = float(np.clip(level, 0.0, 1.0))

        self.visible_amr_levels = updated
        if not force:
            self.steps_since_amr_update = 0

    def _build_obs(self, agent_id: str) -> np.ndarray:
        """Construct observation vector for the given agent.

        Observation structure:
            [patient features (flattened by patient_generator.observe()),
             visible_amr_level_0, ..., visible_amr_level_K,
             (steps_since_amr_update  — only if include_steps_since_amr_update_in_obs)]

        Args:
            agent_id: Agent whose current_patients to use.

        Returns:
            1D float32 array matching observation_spaces[agent_id].
        """
        pg = self._patient_generators[agent_id]
        patients = self.current_patients[agent_id]

        patient_features = pg.observe(patients).astype(np.float32)
        amr_features = np.array(
            [self.visible_amr_levels[abx] for abx in self.antibiotic_names],
            dtype=np.float32,
        )

        components = [patient_features, amr_features]
        if self.include_steps_since_amr_update_in_obs:
            components.append(np.array([float(self.steps_since_amr_update)], dtype=np.float32))

        return np.concatenate(components)

    @staticmethod
    def _build_crossresistance_matrix(
        crossresistance_dict: Optional[Dict[str, Dict[str, float]]],
        antibiotic_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Build a full crossresistance matrix from an optional sparse dict.

        Diagonal entries are always 1.0. Off-diagonal entries default to 0.0
        unless specified. Only off-diagonal entries may be provided.

        Args:
            crossresistance_dict: Sparse dict of off-diagonal ratios, or None.
            antibiotic_names: Ordered list of antibiotic names.

        Returns:
            Full n×n matrix as nested dict.

        Raises:
            ValueError: If any entry references unknown antibiotics or is out of [0,1].
        """
        matrix: Dict[str, Dict[str, float]] = {
            from_abx: {to_abx: 0.0 for to_abx in antibiotic_names}
            for from_abx in antibiotic_names
        }
        # Set diagonal to 1.0
        for abx in antibiotic_names:
            matrix[abx][abx] = 1.0

        if crossresistance_dict is None:
            return matrix

        for from_abx, targets in crossresistance_dict.items():
            if from_abx not in antibiotic_names:
                raise ValueError(
                    f"crossresistance_matrix key '{from_abx}' not in antibiotic_names."
                )
            for to_abx, ratio in targets.items():
                if to_abx not in antibiotic_names:
                    raise ValueError(
                        f"crossresistance_matrix target '{to_abx}' not in antibiotic_names."
                    )
                if from_abx == to_abx:
                    raise ValueError(
                        f"crossresistance_matrix must not include self-entries ('{from_abx}' → '{to_abx}'). "
                        "Diagonal is auto-set to 1.0."
                    )
                if not (0.0 <= float(ratio) <= 1.0):
                    raise ValueError(
                        f"crossresistance ratio '{from_abx}' → '{to_abx}' must be in [0, 1], got {ratio}."
                    )
                matrix[from_abx][to_abx] = float(ratio)

        return matrix

    def _extract_full_patient_attributes(
        self,
        agent_id: str,
        patients: List,
    ) -> Dict:
        """Extract full patient attribute data for granular trajectory logging.

        Delegates to the agent's patient_generator if it provides an
        `export_patient_attributes_for_logging` method; otherwise falls back
        to extracting the standard set of Patient attributes directly.

        The returned dict has 'true' and 'observed' sub-dicts, each mapping
        attribute names to lists of per-patient values. This mirrors the schema
        used by the single-agent ABXAMREnv for compatibility with the existing
        granular metrics analysis pipeline.

        Args:
            agent_id: Agent whose patient generator to use.
            patients: List of Patient objects from the current step.

        Returns:
            Dict with 'true' and 'observed' sub-dicts.
        """
        if not patients:
            return {"true": {}, "observed": {}}

        pg = self._patient_generators[agent_id]
        export_fn = getattr(pg, "export_patient_attributes_for_logging", None)
        if callable(export_fn):
            return export_fn(patients=patients)

        # Fallback: extract the standard Patient attributes directly.
        true_attrs = {
            "prob_infected": [float(p.prob_infected) for p in patients],
            "benefit_value_multiplier": [float(p.benefit_value_multiplier) for p in patients],
            "failure_value_multiplier": [float(p.failure_value_multiplier) for p in patients],
            "benefit_probability_multiplier": [float(p.benefit_probability_multiplier) for p in patients],
            "failure_probability_multiplier": [float(p.failure_probability_multiplier) for p in patients],
            "recovery_without_treatment_prob": [float(p.recovery_without_treatment_prob) for p in patients],
        }
        obs_attrs = {
            "prob_infected": [float(p.prob_infected_obs) for p in patients],
            "benefit_value_multiplier": [float(p.benefit_value_multiplier_obs) for p in patients],
            "failure_value_multiplier": [float(p.failure_value_multiplier_obs) for p in patients],
            "benefit_probability_multiplier": [float(p.benefit_probability_multiplier_obs) for p in patients],
            "failure_probability_multiplier": [float(p.failure_probability_multiplier_obs) for p in patients],
            "recovery_without_treatment_prob": [float(p.recovery_without_treatment_prob_obs) for p in patients],
        }
        return {"true": true_attrs, "observed": obs_attrs}
