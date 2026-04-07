"""Option library for hierarchical RL.

The OptionLibrary stores a collection of OptionBase subclass instances and validates
their compatibility with the environment before training starts.
"""

from typing import Dict, Any, List
import numpy as np
from abx_amr_simulator.hrl.base_option import OptionBase

# Lazy import to avoid circular dependencies
_BlockOption = None
def _get_block_option_class():
    """Lazy load BlockOption class to avoid circular imports."""
    global _BlockOption
    if _BlockOption is None:
        try:
            from abx_amr_simulator.options.defaults.option_types.block.block_option_loader import BlockOption
            _BlockOption = BlockOption
        except ImportError:
            _BlockOption = type(None)  # Sentinel value,if BlockOption not available
    return _BlockOption


class OptionLibrary:
    """Container for a collection of options with validation.
    
    The library stores option instances and provides validation to ensure all options
    are compatible with the environment's patient generator and antibiotic configuration.
    
    The library also stores a reference to the environment and caches the antibiotic
    name-to-index mapping from the environment's RewardCalculator. This provides a
    single source of truth that options can access via env_state.
    
    Attributes:
        name: Human-readable name for this library (e.g., "default_deterministic").
        env: Reference to the ABXAMREnv instance (used to extract antibiotic mappings).
        options: Dict mapping option name (str) -> OptionBase instance.
        abx_name_to_index: Dict mapping antibiotic name -> action index (cached from env).
    """

    def __init__(self, reward_calculator: Any, name: str = "default"):
        """Initialize option library with a reward calculator.

        Args:
            reward_calculator: A RewardCalculator (or compatible) instance that exposes
                ``abx_name_to_index`` — the canonical antibiotic-name → action-index mapping.
                This is the single source of truth for action encoding used by all options in
                this library.
            name: Human-readable identifier for this library.

        Raises:
            ValueError: If reward_calculator does not have abx_name_to_index.
        """
        self.name = name
        self._reward_calculator = reward_calculator
        self.options: Dict[str, OptionBase] = {}

        # Extract and cache antibiotic mapping from reward calculator.
        try:
            self.abx_name_to_index = reward_calculator.abx_name_to_index
        except AttributeError as e:
            raise ValueError(
                f"reward_calculator must have abx_name_to_index. Error: {e}"
            )

    @classmethod
    def from_env(cls, env: Any, name: str = "default") -> "OptionLibrary":
        """Construct an OptionLibrary from a single-agent ABXAMREnv.

        Convenience constructor for existing single-agent call sites.  Extracts the
        RewardCalculator from ``env.unwrapped.reward_calculator`` and delegates to the
        main constructor.

        Args:
            env: A Gymnasium environment whose ``.unwrapped`` exposes ``reward_calculator``.
            name: Human-readable identifier for this library.

        Raises:
            ValueError: If the env does not have the expected reward_calculator attribute.
        """
        try:
            rc = env.unwrapped.reward_calculator
        except AttributeError as e:
            raise ValueError(
                f"env.unwrapped must have reward_calculator. Error: {e}"
            )
        return cls(reward_calculator=rc, name=name)

    def add_option(self, option: OptionBase) -> None:
        """Add an option to the library.
        
        Args:
            option: An OptionBase subclass instance.
        
        Raises:
            ValueError: If option with same name already exists in library.
            TypeError: If option is not an OptionBase instance.
        """
        if not isinstance(option, OptionBase):
            raise TypeError(
                f"Option must be OptionBase instance, got {type(option).__name__}. "
                f"Make sure all options inherit from OptionBase."
            )
        if option.name in self.options:
            raise ValueError(
                f"Option '{option.name}' already exists in library '{self.name}'. "
                f"Option names must be unique."
            )
        self.options[option.name] = option

    def get_option(self, option_id: int) -> OptionBase:
        """Retrieve an option by index.
        
        Args:
            option_id: Integer index into option list.
        
        Returns:
            OptionBase: The requested option.
        
        Raises:
            IndexError: If option_id out of range.
        """
        option_list = list(self.options.values())
        if not (0 <= option_id < len(option_list)):
            raise IndexError(
                f"Option index {option_id} out of range [0, {len(option_list)-1}]. "
                f"Library has {len(option_list)} options."
            )
        return option_list[option_id]

    def __len__(self) -> int:
        """Return number of options in library."""
        return len(self.options)

    def __getitem__(self, name: str) -> OptionBase:
        """Retrieve option by name."""
        if name not in self.options:
            raise KeyError(
                f"Option '{name}' not in library '{self.name}'. "
                f"Available: {list(self.options.keys())}"
            )
        return self.options[name]

    def validate_environment_compatibility(
        self, patient_generator: Any
    ) -> None:
        """Validate that all options can work with the given patient generator.

        This is the critical compatibility check that runs at OptionsWrapper.__init__() and
        MARLOptionsWrapper.__init__(). It ensures all options' requirements are met before
        training starts. Fails loudly with detailed error messages to prevent silent failures.

        The RewardCalculator used for antibiotic-mapping validation is the one supplied at
        construction time (``self._reward_calculator``).

        Checks:
            1. All options' REQUIRES_OBSERVATION_ATTRIBUTES are provided by patient_generator.
            2. All options' referenced antibiotics are in abx_name_to_index.
            3. no_treatment is present and mapped to the last index.
            4. Semantic check: observation-reading options return 'no_treatment' when
               prob_infected=0.0.

        Args:
            patient_generator: PatientGenerator instance used by the environment. Must expose
                ``visible_patient_attributes``.

        Raises:
            ValueError: If any requirement is not met.
        """
        if not self.options:
            raise ValueError(
                f"Library '{self.name}' is empty. Add at least one option before validation."
            )

        if not self.abx_name_to_index:
            raise ValueError(
                "RewardCalculator has no antibiotics configured (abx_name_to_index is empty)."
            )

        # Validate no_treatment presence and position
        rc = self._reward_calculator
        if 'no_treatment' not in rc.abx_name_to_index:
            raise ValueError(
                "RewardCalculator mapping missing 'no_treatment'. "
                "This action must be present and mapped to the last index."
            )
        if 'no_treatment' not in self.abx_name_to_index:
            raise ValueError(
                "OptionLibrary mapping missing 'no_treatment'. "
                "This action must be present and mapped to the last index."
            )

        expected_no_treatment_index = len(rc.abx_name_to_index) - 1
        if rc.abx_name_to_index['no_treatment'] != expected_no_treatment_index:
            raise ValueError(
                "RewardCalculator must map 'no_treatment' to the last action index. "
                f"Expected {expected_no_treatment_index}, got "
                f"{rc.abx_name_to_index['no_treatment']}."
            )
        if self.abx_name_to_index['no_treatment'] != expected_no_treatment_index:
            raise ValueError(
                "OptionLibrary must map 'no_treatment' to the last action index. "
                f"Expected {expected_no_treatment_index}, got "
                f"{self.abx_name_to_index['no_treatment']}."
            )

        if self.abx_name_to_index != rc.abx_name_to_index:
            raise ValueError(
                "OptionLibrary action mapping must match RewardCalculator mapping. "
                "Ensure options use env_state['option_library'].abx_name_to_index "
                "or env_state['reward_calculator'].abx_name_to_index consistently."
            )

        # Validate patient generator provides required attributes
        try:
            provided_patient_attrs = set(patient_generator.visible_patient_attributes)
        except AttributeError as e:
            raise ValueError(
                f"PatientGenerator must have visible_patient_attributes list. "
                f"Got error: {e}"
            )

        if "prob_infected" not in provided_patient_attrs:
            raise ValueError(
                "PatientGenerator must include 'prob_infected' in visible_patient_attributes. "
                "All options require this attribute."
            )

        # Validate each option
        for option_name, option in self.options.items():
            # Check 1: Patient observation attributes
            required_attrs = set(option.REQUIRES_OBSERVATION_ATTRIBUTES)
            missing_attrs = required_attrs - provided_patient_attrs
            if missing_attrs:
                raise ValueError(
                    f"Option '{option_name}' requires patient attributes {list(missing_attrs)}, "
                    f"but PatientGenerator only provides {list(provided_patient_attrs)}. "
                    f"Add missing attributes to patient_generator.visible_patient_attributes."
                )

            # Check 2: Antibiotic name compatibility
            try:
                referenced_abx = option.get_referenced_antibiotics()
            except NotImplementedError:
                raise ValueError(
                    f"Option '{option_name}' does not implement get_referenced_antibiotics(). "
                    f"All options must implement this method for validation."
                )

            available_abx = set(self.abx_name_to_index.keys())
            for abx_name in referenced_abx:
                if abx_name not in available_abx:
                    if abx_name.strip().upper() in {"NO_RX", "NO_TREAT"}:
                        raise ValueError(
                            f"Option '{option_name}' references antibiotic '{abx_name}', "
                            f"but only 'no_treatment' (lowercase, with underscore) is valid. "
                            f"Available antibiotics: {sorted(available_abx)}. "
                            f"Fix: Change '{abx_name}' to 'no_treatment' in option config."
                        )
                    else:
                        raise ValueError(
                            f"Option '{option_name}' references antibiotic '{abx_name}', "
                            f"but it is not in the reward calculator's action space. "
                            f"Available antibiotics: {sorted(available_abx)}. "
                            f"Fix: Either add '{abx_name}' to reward_calculator config or "
                            f"change option to use an available antibiotic."
                        )

            # Check 3: Termination condition flag (reserved for future use)
            if option.PROVIDES_TERMINATION_CONDITION:
                pass

            # Check 4: Semantic validation — observation-reading options should return
            # 'no_treatment' when prob_infected=0.0
            if option.REQUIRES_OBSERVATION_ATTRIBUTES:
                try:
                    test_no_rx_env_state = self._build_test_env_state(
                        patient_generator=patient_generator,
                        force_no_treatment_scenario=True,
                    )
                    no_rx_actions = option.decide(env_state=test_no_rx_env_state)

                    if all(action == no_rx_actions[0] for action in no_rx_actions):
                        if no_rx_actions[0] != 'no_treatment':
                            raise ValueError(
                                f"Option '{option_name}' SEMANTIC ERROR: When prob_infected=0.0 "
                                f"(should clearly select no_treatment), option returned "
                                f"'{no_rx_actions[0]}' but should return 'no_treatment'. "
                                f"Fix: Return 'no_treatment' string for no-treatment decisions."
                            )
                except ValueError:
                    raise
                except Exception:
                    # Non-ValueError exceptions from the option itself are not validation failures;
                    # they will surface at runtime when the option is actually executed.
                    pass

        # Inject full observable attribute list into options that support it
        # (e.g., HeuristicWorker uses this for uncertainty scoring)
        visible_attrs_list = list(patient_generator.visible_patient_attributes)
        for option_name, option in self.options.items():
            if hasattr(option, 'set_observable_attributes'):
                option.set_observable_attributes(visible_attrs_list)

    def _build_test_env_state(
        self,
        patient_generator: Any,
        force_no_treatment_scenario: bool = False,
        num_patients: int = 5,
    ) -> Dict[str, Any]:
        """Build a minimal synthetic env_state for semantic validation testing.

        Used during ``validate_environment_compatibility`` to check that
        observation-reading options return 'no_treatment' when prob_infected=0.0.
        The patient count is a small fixed default (5) — enough to check semantics
        without requiring knowledge of the actual environment's cohort size.

        Args:
            patient_generator: PatientGenerator instance (supplies visible_patient_attributes).
            force_no_treatment_scenario: If True, all patients have prob_infected=0.0.
            num_patients: Number of synthetic patients to create (default 5).

        Returns:
            env_state dict compatible with OptionBase.decide().
        """
        visible_attrs = patient_generator.visible_patient_attributes

        test_patients = []
        for i in range(num_patients):
            patient = {}
            for attr in visible_attrs:
                if force_no_treatment_scenario:
                    if attr == 'prob_infected':
                        patient[attr] = 0.0
                    elif attr.endswith('_multiplier'):
                        patient[attr] = 1.0
                    elif attr == 'recovery_without_treatment_prob':
                        patient[attr] = 0.9
                    else:
                        patient[attr] = 0.5
                else:
                    if attr == 'prob_infected':
                        patient[attr] = min(0.3 + (i * 0.3), 0.9)
                    elif attr.endswith('_multiplier'):
                        patient[attr] = 0.8 + (i * 0.2)
                    elif attr == 'recovery_without_treatment_prob':
                        patient[attr] = 0.1 + (i * 0.1)
                    else:
                        patient[attr] = 0.5
            test_patients.append(patient)

        antibiotic_names = [abx for abx in self.abx_name_to_index.keys() if abx != 'no_treatment']
        current_amr_levels = {abx: 0.0 for abx in antibiotic_names}

        return {
            'patients': test_patients,
            'num_patients': num_patients,
            'current_amr_levels': current_amr_levels,
            'reward_calculator': self._reward_calculator,
            'patient_generator': patient_generator,
            'option_library': self,
            'use_relative_uncertainty': True,
            'current_step': 0,
            'max_steps': 100,
        }

    def list_options(self) -> List[str]:
        """Return ordered list of option names."""
        return list(self.options.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert library to dictionary representation.
        
        Used for serialization and logging.
        
        Returns:
            Dict with keys:
                - 'name': library name
                - 'num_options': number of options
                - 'options': list of dicts with option info
        """
        options_info = []
        for opt_name, opt in self.options.items():
            options_info.append({
                'name': opt_name,
                'k': opt.k if opt.k != float('inf') else 'inf',
                'requires_observation_attrs': opt.REQUIRES_OBSERVATION_ATTRIBUTES,
                'requires_amr_levels': opt.REQUIRES_AMR_LEVELS,
                'provides_termination': opt.PROVIDES_TERMINATION_CONDITION,
            })
        
        return {
            'name': self.name,
            'num_options': len(self.options),
            'options': options_info,
            'antibiotic_names': list(self.abx_name_to_index.keys()),
        }
