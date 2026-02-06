"""
Manager observation builder for HRL.

Constructs the observation state presented to the manager at each macro-decision point.
"""

import numpy as np
from typing import List, Tuple, Optional


class ManagerObsBuilder:
    """Builds rich manager observations during option execution."""
    
    def __init__(
        self,
        num_antibiotics: int = 2,
        include_prospective_cohort_stats: bool = False,
        prospective_attributes: Optional[List[str]] = None,
    ):
        """
        Initialize manager observation builder.
        
        Args:
            num_antibiotics: Number of antibiotics (default 2 for A, B)
            include_prospective_cohort_stats: Whether to add prospective cohort statistics
            prospective_attributes: Which patient attributes to include in prospective stats
                                   (e.g., ['prob_infected', 'benefit_mult_A', ...])
        """
        self.num_antibiotics = num_antibiotics
        self.include_prospective_cohort_stats = include_prospective_cohort_stats
        self.prospective_attributes = prospective_attributes or []
        
        # Track state across macro-steps
        self.prev_option_id = None
        self.consecutive_same_option = 0
        self.steps_since_last_drug = {}
        for drug_id in range(num_antibiotics):
            self.steps_since_last_drug[drug_id] = 0
    
    def reset(self):
        """Reset tracking state at episode start."""
        self.prev_option_id = None
        self.consecutive_same_option = 0
        for drug_id in self.steps_since_last_drug.keys():
            self.steps_since_last_drug[drug_id] = 0
    
    def build_observation(
        self,
        amr_start: np.ndarray,
        amr_end: np.ndarray,
        current_option_id: int,
        steps_in_episode: int,
        total_episode_steps: int,
        prospective_cohort_stats: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Construct manager observation.
        
        Args:
            amr_start: AMR levels at start of option (shape: (num_antibiotics,))
            amr_end: AMR levels at end of option (shape: (num_antibiotics,))
            current_option_id: ID of the macro-action that just executed
            steps_in_episode: Current step in episode
            total_episode_steps: Total steps in episode (for progress normalization)
            prospective_cohort_stats: Dict of cohort statistics (if include_prospective_cohort_stats=True)
        
        Returns:
            Manager observation vector (numpy array)
        """
        obs_components = []
        
        # AMR dynamics (always included)
        obs_components.extend(amr_start.flatten())
        obs_components.extend(amr_end.flatten())
        
        # Update option tracking
        if current_option_id == self.prev_option_id and self.prev_option_id is not None:
            self.consecutive_same_option += 1
        else:
            self.consecutive_same_option = 1
        self.prev_option_id = current_option_id
        
        # Option history
        obs_components.append(float(current_option_id))
        obs_components.append(float(self.consecutive_same_option))
        
        # Steps since last drug use
        # Note: This assumes drugs are indexed 0, 1, ... in the base env
        for drug_id in range(self.num_antibiotics):
            self.steps_since_last_drug[drug_id] += 1
        obs_components.extend([float(self.steps_since_last_drug[d]) for d in range(self.num_antibiotics)])
        
        # Episode progress
        episode_progress = steps_in_episode / max(total_episode_steps, 1)
        obs_components.append(float(episode_progress))
        
        # Prospective cohort statistics (optional)
        if self.include_prospective_cohort_stats and prospective_cohort_stats is not None:
            for attr in self.prospective_attributes:
                value = prospective_cohort_stats.get(attr, 0.0)
                obs_components.append(float(value))
        
        obs = np.array(obs_components, dtype=np.float32)
        
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Manager observation contains non-finite values: {obs}")
        
        return obs
    
    def update_steps_since_drug(self, drug_id: int):
        """Update tracking when a drug is prescribed."""
        if drug_id < 0 or drug_id >= self.num_antibiotics:
            raise ValueError(f"Drug ID {drug_id} out of range [0, {self.num_antibiotics-1}]")
        self.steps_since_last_drug[drug_id] = 0
    
    def compute_observation_dim(self) -> int:
        """Compute total observation dimension."""
        dim = 0
        dim += 2 * self.num_antibiotics  # amr_start, amr_end
        dim += 2  # option_id, consecutive_same_option
        dim += self.num_antibiotics  # steps_since_last_drug
        dim += 1  # episode_progress
        if self.include_prospective_cohort_stats:
            dim += len(self.prospective_attributes)
        return dim
