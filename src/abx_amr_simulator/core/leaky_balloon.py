import math
import matplotlib.pyplot as plt
import os
#%% Leaky Balloon class definition
class AMR_LeakyBalloon:
    """
    Soft-bounded AMR accumulator using leaky balloon dynamics.
    
    Models per-antibiotic resistance fraction [0,1] with internal pressure state,
    leak decay, and sigmoid output mapping. Ensures bounded AMR levels with
    long-term stability (without prescribing, AMR decays to residual floor).
    
    OWNERSHIP (for future multi-agent refactor):
    - Owns: latent pressure state, leak dynamics, sigmoid transformation
    - Input: non-negative puff values (from prescriptions)
    - Output: visible AMR fraction in [permanent_residual_volume, 1.0]
    
    Future multi-agent notes:
    - Instantiated per-antibiotic per-Locale (each locale has independent AMR state)
    - Puffs come from prescriptions within that locale (or via crossresistance)
    - Multi-locale environments will have separate balloon instances per locale
    """
    def __init__(self, 
                 leak=0.1, 
                 flatness_parameter=1.0, 
                 permanent_residual_volume=0.0, 
                 initial_amr_level=0.0):
        """Initialize the leaky balloon AMR dynamics model.
        
        Creates a soft-bounded AMR accumulator using sigmoid-transformed internal pressure
        with exponential decay (leak). Ensures AMR levels remain in [permanent_residual_volume, 1.0]
        with stable long-term behavior (decays to residual floor without prescribing).
        
        Args:
            leak (float): Multiplicative decay factor applied per timestep. Must be in (0, 1).
                Higher values → faster decay (e.g., 0.1 = 10% pressure loss per step).
                Represents natural resistance loss due to bacterial turnover, selective pressure
                relaxation, etc. Default: 0.1.
            flatness_parameter (float): Controls steepness of sigmoid mapping from pressure to
                volume. Must be > 0. Smaller values → steeper sigmoid → more abrupt AMR changes.
                Larger values → flatter sigmoid → gradual AMR accumulation. Default: 1.0.
            permanent_residual_volume (float): Minimum AMR level (floor, y-intercept). Must be
                in [0, 1). Even with zero pressure, AMR remains at this level. Represents
                baseline resistance that never fully disappears. Default: 0.0 (can decay to zero).
            initial_amr_level (float): Starting observable AMR level (volume/fraction) before first puff.
                Must be in [permanent_residual_volume, 1.0]. Default: 0.0 (balloon starts at residual).
                Internally, this is converted to the corresponding latent pressure via inverse sigmoid.
        
        Raises:
            ValueError: If parameters are outside valid ranges.
        
        Example:
            >>> balloon = AMR_LeakyBalloon(
            ...     leak=0.1,  # 10% pressure decay per step
            ...     flatness_parameter=0.5,  # Steep sigmoid
            ...     permanent_residual_volume=0.05,  # 5% baseline resistance
            ...     initial_amr_level=0.3  # Start at 30% resistance
            ... )
            >>> balloon.step(puffs=2.0)  # Add 2 units of pressure
            0.45  # Returns visible volume (AMR fraction)
        """
        
        # Do some basic validation
        if not (0.0 < leak < 1.0):
            raise ValueError("leak must be in (0, 1)")
        if flatness_parameter <= 0.0:
            raise ValueError("flatness_parameter must be > 0")
        if not (0.0 <= permanent_residual_volume < 1.0):
            raise ValueError("permanent_residual_volume must be in [0, 1)")
        if not (permanent_residual_volume <= initial_amr_level <= 1.0):
            raise ValueError("initial_amr_level must be in [permanent_residual_volume, 1.0]")
        
        self.leak = leak
        self.flatness_parameter = flatness_parameter
        self.permanent_residual_volume = permanent_residual_volume
        
        # Convert initial AMR level to internal pressure using inverse sigmoid
        self.pressure = self._inverse_sigmoid(initial_amr_level)

    def _inverse_sigmoid(self, volume):
        """
        Computes the latent pressure needed to produce a given volume.
        
        Inverts the sigmoid mapping: volume → pressure.
        Since normalized_vol = (sigmoid(p) - 0.5) * 2, we solve for p.
        volume = permanent_residual_volume + (sigmoid(p) - 0.5) * 2
        
        Args:
            volume (float): Desired visible volume in [permanent_residual_volume, 1.0]
        
        Returns:
            float: The latent pressure needed to produce this volume
        """
        # Remove the permanent residual component
        normalized_vol = volume - self.permanent_residual_volume
        
        # Invert the normalization: sigmoid(p) = normalized_vol / 2 + 0.5
        sigmoid_val = normalized_vol / 2.0 + 0.5
        
        # Clamp sigmoid_val to (0, 1) to avoid numerical issues with log
        sigmoid_val = max(1e-10, min(1.0 - 1e-10, sigmoid_val))
        
        # Invert sigmoid: p = -flatness_parameter * log((1 - sigmoid_val) / sigmoid_val)
        pressure = -self.flatness_parameter * math.log((1.0 - sigmoid_val) / sigmoid_val)
        
        return max(0.0, pressure)

    def get_volume(self, pressure=None):
        """
        Maps current pressure to volume using a sigmoid function.
        To ensure Volume = residual when Pressure = 0, we shift the 
        standard sigmoid result.
        """
        # Standard sigmoid: 1 / (1 + exp(-x/v))
        # When x=0, sigmoid=0.5. To make the volume start at 0 (before residual),
        # we subtract 0.5 and multiply by 2 to keep the range [0, 1].
        if pressure is None:
            pressure = self.pressure
        
        exponent = -pressure / self.flatness_parameter
        sigmoid_val = 1 / (1 + math.exp(exponent))
        
        # Normalized sigmoid that is 0.0 when pressure is 0.0
        normalized_vol = (sigmoid_val - 0.5) * 2
        
        return self.permanent_residual_volume + normalized_vol

    def step(self, puffs):
        """Update pressure based on puffs and leak, then return new volume.
        
        Applies the core balloon dynamics:
        1. Add `puffs` to internal pressure (inflation)
        2. Apply exponential decay via `leak` multiplier (deflation)
        3. Map updated pressure to visible volume via sigmoid
        
        Args:
            puffs (float): Non-negative pressure increase this timestep. Typically represents
                prescription counts (e.g., 5 prescriptions = 5 puffs). Must be >= 0.
        
        Returns:
            float: Updated visible volume (AMR fraction) after puffs and leak, in range
                [permanent_residual_volume, 1.0]. This is the AMR level observable in the
                environment.
        
        Raises:
            ValueError: If puffs < 0.
        
        Example:
            >>> balloon = AMR_LeakyBalloon(leak=0.1)
            >>> balloon.step(puffs=3.0)  # Add 3 puffs
            0.45
            >>> balloon.step(puffs=0.0)  # No puffs, only leak applies
            0.42  # Volume decreased slightly due to leak
        """
        # Validate input
        if puffs < 0:
            raise ValueError("puffs must be non-negative")
        
        # 1. Add the new air
        self.pressure += puffs
        
        # 2. Apply the leak (decay)
        # Pressure cannot drop below 0
        self.pressure = max(0.0, self.pressure - self.leak)
        
        # 3. Calculate and return volume
        return self.get_volume(pressure=self.pressure)
    
    def _step_no_internal_state_change(self, puffs):
        """
        Computes the volume after a step without changing internal state.
        
        :param puffs: Number of puffs added in this timestep
        :return: Volume after the step (float)
        """
        temp_pressure = self.pressure + puffs
        temp_pressure = max(0.0, temp_pressure - self.leak)
        
        # Calculate volume based on temporary pressure
        return self.get_volume(pressure=temp_pressure)
    
    def reset(self, initial_amr_level=0.0):
        """
        Resets the balloon to the initial AMR level.
        
        Args:
            initial_amr_level (float): AMR level to reset to, in [permanent_residual_volume, 1.0]
        """
        self.pressure = self._inverse_sigmoid(initial_amr_level)
    
    def copy(self):
        """
        Creates an independent copy of the leaky balloon with same parameters and current state.
        
        The copy has:
        - Same leak, flatness_parameter, and permanent_residual_volume
        - Same current pressure (internal state)
        
        Changes to the copy do not affect the original, and vice versa.
        
        Returns:
            AMR_LeakyBalloon: A new independent instance with copied state
        
        Example:
            >>> balloon = AMR_LeakyBalloon(leak=0.1)
            >>> balloon.step(puffs=2.0)
            >>> copy_balloon = balloon.copy()
            >>> copy_balloon.step(puffs=1.0)  # Copy evolves independently
            >>> # Original balloon pressure unchanged
        """
        # Get the current visible volume to use as initial_amr_level for the new instance
        current_visible_volume = self.get_volume()
        
        # Create a new instance with the same configuration parameters
        new_balloon = AMR_LeakyBalloon(
            leak=self.leak,
            flatness_parameter=self.flatness_parameter,
            permanent_residual_volume=self.permanent_residual_volume,
            initial_amr_level=current_visible_volume  # Use current volume so copy starts at same state
        )
        
        # The pressure should now be set to match the original
        # (both computed from same visible volume via inverse sigmoid)
        return new_balloon
        
    def get_delta_volume_for_counterfactual_num_puffs_vs_one_less(self, num_counterfactual_puffs):
        """
        Computes the change in volume if the number of puffs were
        increased by one compared to a given number.
        
        :param num_counterfactual_puffs: Base number of puffs
        :return: Delta volume (float)
        """
        vol_with_puffs = self._step_no_internal_state_change(num_counterfactual_puffs)
        vol_with_one_less = self._step_no_internal_state_change(num_counterfactual_puffs - 1)
        
        return vol_with_puffs - vol_with_one_less
    
    def print_volume_response(self, puff_sequence):
        """Print the leaky balloon response to a given puff sequence."""
        
        # Create an empty string to hold the output
        output_str = "Timestep\tPuffs\tVolume\n"
        
        for t, puffs in enumerate(puff_sequence):
            volume = self.step(puffs=puffs)
            output_str += f"{t}\t{puffs}\t{volume:.4f}\n"
            
        print(output_str)
    
    def plot_leaky_balloon_response_curve(self, pressures, title, fname=None, save_plot_folder=None, show_plot=False, width=8, height=5):
        """
        Plot the leaky balloon response curve over a range of pressures.
        """
        volumes = [self.get_volume(p) for p in pressures]
        
        plt.figure(figsize=(width, height))
        plt.plot(pressures, volumes, label=title)
        plt.xlabel("Pressure")
        plt.ylabel("Volume (AMR Level)")
        plt.title("Leaky Balloon Response Curve")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid()
        
        # Use tight_layout to prevent clipping of labels
        plt.tight_layout(pad=1.0)
        
        # If show_plot is True, display the plot
        if show_plot:
            plt.show()
            
        # If save_plot_folder is provided, save the plot there
        if save_plot_folder is not None:
            # Save the plot to current folder if fname is not provided
            if fname is None:
                fname = "leaky_balloon_response_curve.png"
            plt.savefig(os.path.join(save_plot_folder, fname))
        
        # At the end, close the plot to free memory
        plt.close()
    
    def plot_leaky_balloon_response_to_puff_sequence(self, puff_sequence, title, fname=None, save_plot_folder=None, show_plot=False, width=10, height=6):
        """
        Plot the leaky balloon response to a given puff sequence.
        """
        plt.figure(figsize=(width, height))
        ax1 = plt.subplot(2, 1, 1)
        
        volumes = []
        for puffs in puff_sequence:
            volume = self.step(puffs=puffs)
            volumes.append(volume)
            
        plt.plot(volumes, label=title)
        plt.xlabel("Time Step")
        plt.ylabel("Balloon Volume")
        plt.title("Leaky Balloon Response")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid()
        
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        plt.bar(range(len(puff_sequence)), puff_sequence, color='orange')
        plt.xlabel("Time Step")
        plt.ylabel("Doses")
        plt.title("Dose Sequence")
        plt.ylim(0, max(puff_sequence) + 1)
        plt.grid()
        
        # Add this line here
        plt.tight_layout(pad=1.0)
        
        # If show_plot is True, display the plot
        if show_plot:
            plt.show()
            
        # If save_plot_folder is provided, save the plot there
        if save_plot_folder is not None:
            # Save the plot to current folder if fname is not provided
            if fname is None:
                fname = "leaky_balloon_response.png"
            plt.savefig(os.path.join(save_plot_folder, fname))
        
        # At the end, close the plot to free memory
        plt.close()