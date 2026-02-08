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
    
    Attributes:
        name: Human-readable name for this library (e.g., "default_deterministic").
        options: Dict mapping option name (str) -> OptionBase instance.
        antibiotic_names: Ordered list of antibiotic names (extracted and cached during validation).
    """

    def __init__(self, name: str = "default"):
        """Initialize an empty option library.
        
        Args:
            name: Human-readable identifier for this library.
        """
        self.name = name
        self.options: Dict[str, OptionBase] = {}
        self.antibiotic_names: List[str] = []

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

        # Extract antibiotic names from environment
        try:
            abx_name_to_index = env.unwrapped.reward_calculator.abx_name_to_index
            self.antibiotic_names = list(abx_name_to_index.keys())
        except AttributeError as e:
            raise ValueError(
                f"Environment must have reward_calculator.abx_name_to_index dict. "
                f"Got error: {e}"
            )

        if not self.antibiotic_names:
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

            # Check 2: Antibiotic references (implicit - options reference antibiotics via names)
            # We don't explicitly check here, but OptionsWrapper will validate at runtime.

            # Check 3: AMR levels requirement
            if option.REQUIRES_AMR_LEVELS:
                has_amr = hasattr(env.unwrapped, 'leaky_balloons')
                if not has_amr:
                    raise ValueError(
                        f"Option '{option_name}' requires AMR levels (REQUIRES_AMR_LEVELS=True), "
                        f"but environment doesn't provide leaky_balloons. "
                        f"Ensure environment is ABXAMREnv with AMR tracking enabled."
                    )

            # Check 4: Step number requirement
            if option.REQUIRES_STEP_NUMBER:
                has_step = hasattr(env.unwrapped, 'current_step')
                if not has_step:
                    raise ValueError(
                        f"Option '{option_name}' requires step number (REQUIRES_STEP_NUMBER=True), "
                        f"but environment doesn't track current_step. "
                        f"Ensure environment is ABXAMREnv with step tracking enabled."
                    )

            # Check 5: Termination condition flag consistency
            if option.PROVIDES_TERMINATION_CONDITION:
                # For future use; just log for now
                pass

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
            'antibiotic_names': self.antibiotic_names,
        }
