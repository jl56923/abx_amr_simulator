"""LSTM hidden state logging callback for RecurrentPPO belief analysis.

Captures LSTM hidden/cell states during evaluation episodes and saves them
alongside true AMR levels for offline probe analysis.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import torch as th


class LSTMStateLogger(BaseCallback):
    """Log LSTM hidden states during evaluation for belief probing.
    
    This callback captures the LSTM hidden state at each timestep during training,
    along with true AMR levels and other environment info. The logged data can be
    used to train a probe (linear or MLP) to predict true AMR from hidden states,
    validating that the recurrent policy learns a meaningful belief about latent
    resistance dynamics.
    
    **Usage with RecurrentPPO:**
        >>> logger = LSTMStateLogger(save_dir="results/my_run/lstm_logs", log_freq=100)
        >>> agent.learn(total_timesteps=100000, callback=logger)
    
    **How it works:**
        - Logs LSTM hidden states from model._last_lstm_states after each step
        - Captures observations, actions, rewards, and true AMR from environment info
        - Saves complete episodes to .npz files when episode ends
        - Compatible only with RecurrentPPO; silently skips for other algorithms
    
    **Offline probe analysis:**
        Load saved data with `np.load('lstm_logs/episode_0000.npz')` and fit
        a probe (e.g., linear regression) from hidden_states → true_amr.
    
    Args:
        save_dir (str): Directory to save logged episodes (creates if doesn't exist).
        log_freq (int): Only log if this many timesteps have passed since last log.
                       0 = log every step. Default: 100.
        verbose (int): Verbosity level: 0 = silent, 1 = info messages. Default: 1.
    
    Attributes:
        episode_count (int): Number of episodes logged so far.
        current_episode (Dict): Data accumulated for current episode.
        steps_since_log (int): Timesteps since last log event.
    """
    
    def __init__(
        self,
        save_dir: str,
        log_freq: int = 100,
        verbose: int = 1,
    ):
        """Initialize the LSTM state logger."""
        super().__init__(verbose=verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_freq = log_freq
        self.episode_count = 0
        self.current_episode = self._new_episode_dict()
        self.steps_since_log = 0
        self.warned_no_lstm = False  # Track if we've warned about missing LSTM
    
    def _new_episode_dict(self) -> Dict[str, List]:
        """Create empty episode data structure."""
        return {
            "hidden_states": [],
            "cell_states": [],
            "true_amr": [],
            "observations": [],
            "actions": [],
            "rewards": [],
            "timesteps": [],
        }
    
    def _on_step(self) -> bool:
        """
        Called at each training step.
        
        Captures LSTM hidden/cell states and environment info when:
        1. The model is RecurrentPPO (has _last_lstm_states)
        2. Enough timesteps have passed since last log
        
        Returns:
            bool: Always True (never stops training).
        """
        # Check if we should log (based on frequency)
        if self.log_freq > 0:
            self.steps_since_log += 1
            if self.steps_since_log < self.log_freq:
                return True
            self.steps_since_log = 0
        
        # Only process if model has LSTM states (i.e., is RecurrentPPO)
        if not hasattr(self.model, '_last_lstm_states'):
            if not self.warned_no_lstm and self.verbose > 0:
                print("[LSTMStateLogger] Warning: Model is not RecurrentPPO "
                      "(no _last_lstm_states attribute). Disabling logging.")
                self.warned_no_lstm = True
            return True
        
        lstm_states = self.model._last_lstm_states
        if lstm_states is None:
            return True  # LSTM not initialized yet
        
        try:
            # Extract actor LSTM hidden state (pi)
            # RNNStates.pi = (hidden_state, cell_state) tuple
            # hidden_state shape: (num_layers, n_envs, hidden_size)
            hidden_state_tensor = lstm_states.pi[0]  # First LSTM layer, hidden state
            
            # Convert tensor to numpy on CPU
            if isinstance(hidden_state_tensor, th.Tensor):
                hidden_state = hidden_state_tensor.detach().cpu().numpy()
            else:
                hidden_state = np.array(hidden_state_tensor)
            
            # Get current observation from training locals
            obs = self.locals.get('obs_tensor')
            if obs is not None:
                if isinstance(obs, th.Tensor):
                    obs = obs.detach().cpu().numpy()
                else:
                    obs = np.array(obs)
            
            # Get action
            action = self.locals.get('actions')
            if action is not None:
                if isinstance(action, th.Tensor):
                    action = action.detach().cpu().numpy()
                else:
                    action = np.array(action)
            
            # Get rewards
            rewards = self.locals.get('rewards')
            if rewards is not None:
                if isinstance(rewards, th.Tensor):
                    rewards = rewards.detach().cpu().numpy()
                else:
                    rewards = np.array(rewards)
            
            # Get true AMR from environment info
            infos = self.locals.get('infos', [])
            true_amr = None
            if len(infos) > 0 and isinstance(infos, list):
                info = infos[0] if infos else {}
                if isinstance(info, dict) and 'actual_amr_levels' in info:
                    amr_dict = info['actual_amr_levels']
                    # Convert dict to array in consistent key order
                    true_amr = np.array(list(amr_dict.values()))
            
            # Log data from first environment only
            env_idx = 0
            
            # Take first element of batch dimension
            if hidden_state.ndim >= 2:
                hidden_state_single = hidden_state[env_idx] if hidden_state.ndim >= 2 else hidden_state
            else:
                hidden_state_single = hidden_state
            
            self.current_episode["hidden_states"].append(hidden_state_single)
            
            if obs is not None and obs.ndim > 0:
                obs_single = obs[env_idx] if obs.ndim >= 2 else obs
                self.current_episode["observations"].append(obs_single)
            
            if action is not None and action.ndim > 0:
                action_single = action[env_idx] if action.ndim >= 2 else action
                self.current_episode["actions"].append(action_single)
            
            if rewards is not None and rewards.ndim > 0:
                reward_single = rewards[env_idx] if rewards.ndim >= 1 else rewards
                self.current_episode["rewards"].append(float(reward_single))
            
            if true_amr is not None:
                self.current_episode["true_amr"].append(true_amr)
            
            self.current_episode["timesteps"].append(self.num_timesteps)
            
            # Check if episode ended
            dones = self.locals.get('dones')
            if dones is not None:
                done_single = dones[env_idx] if isinstance(dones, np.ndarray) and dones.ndim > 0 else dones
                if done_single:
                    self._save_episode()
        
        except Exception as e:
            if self.verbose > 0:
                print(f"[LSTMStateLogger] Error during logging: {e}")
        
        return True
    
    def _save_episode(self):
        """Save current episode data to disk."""
        if len(self.current_episode["hidden_states"]) == 0:
            return  # Nothing to save
        
        # Convert lists to arrays
        data = {
            key: np.array(val) for key, val in self.current_episode.items()
            if len(val) > 0
        }
        
        # Save to npz file
        filename = self.save_dir / f"episode_{self.episode_count:04d}.npz"
        np.savez_compressed(filename, **data)
        
        if self.verbose > 0:
            print(f"[LSTMStateLogger] Saved episode {self.episode_count} to {filename} "
                  f"({len(data['hidden_states'])} timesteps)")
        
        self.episode_count += 1
        self.current_episode = self._new_episode_dict()
    
    def _on_training_end(self) -> None:
        """Save any remaining episode data when training ends."""
        if len(self.current_episode["hidden_states"]) > 0:
            self._save_episode()
