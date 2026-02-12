"""Option library for hierarchical RL.

The OptionLibrary stores a collection of OptionBase subclass instances and validates
their compatibility with the environment before training starts.
"""

from typing import Dict, Any, List
from abx_amr_simulator.hrl.base_option import OptionBase


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

    def __init__(self, env: Any, name: str = "default"):
        """Initialize option library with environment reference.
        
        Args:
            env: The ABXAMREnv instance (stores reference and extracts antibiotic mappings).
            name: Human-readable identifier for this library.
        
        Raises:
            ValueError: If env doesn't have reward_calculator.abx_name_to_index.
        """
        self.name = name
        self.env = env
        self.options: Dict[str, OptionBase] = {}
        
        # Extract and cache antibiotic mapping from environment (single source of truth)
        try:
            self.abx_name_to_index = env.unwrapped.reward_calculator.abx_name_to_index
        except AttributeError as e:
            raise ValueError(
                f"Environment must have reward_calculator.abx_name_to_index. Error: {e}"
            )

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
        self, env: Any, patient_generator: Any
    ) -> None:
        """Validate that all options can work with the given environment.
        
        This is the critical compatibility check that runs at OptionsWrapper.__init__().
        It ensures all options' requirements are met by the environment before training starts.
        Fails loudly with detailed error messages to prevent silent failures.
        
        Checks:
            1. All options' REQUIRES_OBSERVATION_ATTRIBUTES are provided by patient_generator.
            2. All options' referenced antibiotics are in the environment's action space.
            3. If REQUIRES_AMR_LEVELS=True, environment has current AMR levels accessible.
            4. If REQUIRES_STEP_NUMBER=True, environment provides current step count.
        
        Args:
            env: The unwrapped environment (ABXAMREnv).
                 Expected to have:
                 - reward_calculator with abx_name_to_index dict
                 - Any other state the options need
            patient_generator: The PatientGenerator instance used by the environment.
                              Expected to have:
                              - visible_patient_attributes list
                              - observe() method
                              - obs_dim() method
        
        Raises:
            ValueError: If any requirement is not met. Error messages include:
                - Which option failed
                - What requirement wasn't met
                - What is provided vs. what is required
                - Suggestions for fixing
        
        Example:
            library = OptionLibrary()
            library.add_option(BlockOption('A_5', 'A', 5))
            library.validate_environment_compatibility(env, patient_generator)
            # Raises ValueError if any option can't work with env/pg
        """
        if not self.options:
            raise ValueError(
                f"Library '{self.name}' is empty. Add at least one option before validation."
            )

        # Antibiotic names are already extracted and cached in self.abx_name_to_index at init
        if not self.abx_name_to_index:
            raise ValueError(
                "Environment has no antibiotics configured (abx_name_to_index is empty)."
            )

        # Extract patient attributes from patient generator
        try:
            provided_patient_attrs = set(patient_generator.visible_patient_attributes)
        except AttributeError as e:
            raise ValueError(
                f"PatientGenerator must have visible_patient_attributes list. "
                f"Got error: {e}"
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
            # Verify all antibiotics referenced by the option exist in environment's action space
            try:
                referenced_abx = option.get_referenced_antibiotics()
            except NotImplementedError:
                raise ValueError(
                    f"Option '{option_name}' does not implement get_referenced_antibiotics(). "
                    f"All options must implement this method for validation."
                )
            
            # Validate each referenced antibiotic
            available_abx = set(self.abx_name_to_index.keys())
            for abx_name in referenced_abx:
                if abx_name not in available_abx:
                    # Check if user provided a variation of 'no_treatment'
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
                            f"but it is not in environment's action space. "
                            f"Available antibiotics: {sorted(available_abx)}. "
                            f"Fix: Either add '{abx_name}' to reward_calculator config or "
                            f"change option to use an available antibiotic."
                        )

            # Check 3: AMR levels requirement
            if option.REQUIRES_AMR_LEVELS:
                has_amr = hasattr(env.unwrapped, 'amr_balloon_models')
                if not has_amr:
                    raise ValueError(
                        f"Option '{option_name}' requires AMR levels (REQUIRES_AMR_LEVELS=True), "
                        f"but environment doesn't provide amr_balloon_models. "
                        f"Ensure environment is ABXAMREnv with AMR tracking enabled."
                    )

            # Check 4: Step number requirement
            if option.REQUIRES_STEP_NUMBER:
                has_step = hasattr(env.unwrapped, 'current_time_step')
                if not has_step:
                    raise ValueError(
                        f"Option '{option_name}' requires step number (REQUIRES_STEP_NUMBER=True), "
                        f"but environment doesn't track current_time_step. "
                        f"Ensure environment is ABXAMREnv with step tracking enabled."
                    )

            # Check 5: Termination condition flag consistency
            if option.PROVIDES_TERMINATION_CONDITION:
                # For future use; just log for now
                pass
        
        # After validation, inject full observable attribute list into options that support it
        # (e.g., HeuristicWorker needs this for uncertainty scoring)
        visible_attrs_list = list(patient_generator.visible_patient_attributes)
        for option_name, option in self.options.items():
            if hasattr(option, 'set_observable_attributes'):
                option.set_observable_attributes(visible_attrs_list)

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
                'requires_step_number': opt.REQUIRES_STEP_NUMBER,
                'provides_termination': opt.PROVIDES_TERMINATION_CONDITION,
            })
        
        return {
            'name': self.name,
            'num_options': len(self.options),
            'options': options_info,
            'antibiotic_names': list(self.abx_name_to_index.keys()),
        }
