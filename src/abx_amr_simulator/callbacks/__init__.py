"""
Custom Stable-Baselines3 callbacks for patient attribute logging and evaluation.

Provides:
- PatientStatsLoggingCallback: Logs aggregate patient statistics to TensorBoard during training
- DetailedEvalCallback: Extends EvalCallback to collect and save full patient trajectories during eval
- EarlyStoppingCallback: Stops training when performance metric plateaus
- LSTMStateLogger: Logs LSTM hidden states for belief probing analysis
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from .early_stopping import EarlyStoppingCallback
from .lstm_state_logger import LSTMStateLogger

__all__ = ['PatientStatsLoggingCallback', 'DetailedEvalCallback', 'EarlyStoppingCallback', 'LSTMStateLogger']


class PatientStatsLoggingCallback(BaseCallback):
    """Log aggregate patient statistics to TensorBoard during training.
    
    Extracts patient_stats from environment info dict at each step and logs all
    metrics under 'patient_stats/' TensorBoard namespace. Provides lightweight
    monitoring of:
    - Patient distribution shifts (infected probabilities)
    - Observation errors (bias and noise in observed attributes)
    - Patient heterogeneity (benefit/failure multiplier distributions)
    
    Compatible with both single and vectorized environments. For vectorized envs,
    aggregates stats across all environments before logging.
    
    Example:
        >>> callback = PatientStatsLoggingCallback(verbose=1)
        >>> agent.learn(total_timesteps=100000, callback=callback)
        >>> # Stats appear in TensorBoard: patient_stats/mean_prob_infected, etc.
    """
    
    def __init__(self, verbose: int = 0):
        """Initialize the callback.
        
        Args:
            verbose (int): Verbosity level: 0 = silent, 1 = info messages. Default: 0.
        """
        super().__init__(verbose)
        self.patient_stats_buffer = []
    
    def _on_step(self) -> bool:
        """
        Called after each environment step during training.
        
        Extracts patient_stats from info dict and logs to TensorBoard.
        For vectorized environments, aggregates stats across all environments.
        
        Returns:
            bool: True to continue training, False to stop
        """
        # Get info from most recent step
        # For VecEnv, infos is a list of info dicts (one per env)
        # For single env, it's still wrapped in a list by SB3
        infos = self.locals.get('infos', [])
        
        if not infos:
            return True
        
        # Collect patient_stats from all environments
        stats_list = []
        for info in infos:
            if 'patient_stats' in info:
                stats_list.append(info['patient_stats'])
        
        if not stats_list:
            # No patient_stats found in any environment
            return True
        
        # If multiple environments, aggregate stats (take mean across envs)
        if len(stats_list) > 1:
            aggregated_stats = self._aggregate_stats(stats_list)
        else:
            aggregated_stats = stats_list[0]
        
        # Log all stats to TensorBoard under patient_stats/ namespace
        # Shorten long keys to avoid TensorBoard truncation errors
        for key, value in aggregated_stats.items():
            # Shorten common long prefixes to avoid key collisions
            short_key = key.replace('benefit_probability_multiplier', 'benefit_prob_mult')
            short_key = short_key.replace('failure_probability_multiplier', 'failure_prob_mult')
            short_key = short_key.replace('recovery_without_treatment_prob', 'recovery_no_rx_prob')
            short_key = short_key.replace('benefit_value_multiplier', 'benefit_val_mult')
            short_key = short_key.replace('failure_value_multiplier', 'failure_val_mult')
            short_key = short_key.replace('_obs_error_', '_err_')
            self.logger.record(f'patient_stats/{short_key}', value, exclude=None)
        
        return True
    
    def _aggregate_stats(self, stats_list: list) -> Dict[str, float]:
        """
        Aggregate patient stats across multiple environments.
        
        Takes the mean of each statistic across all environments.
        
        Args:
            stats_list: List of patient_stats dicts from different environments
            
        Returns:
            Dict with aggregated statistics
        """
        # Get all keys from first stats dict
        keys = stats_list[0].keys()
        
        aggregated = {}
        for key in keys:
            # Collect values for this key across all envs
            values = [stats[key] for stats in stats_list if key in stats]
            # Take mean
            aggregated[key] = float(np.mean(values))
        
        return aggregated


class DetailedEvalCallback(EvalCallback):
    """Extended EvalCallback that collects and saves full patient trajectories.
    
    Extends stable-baselines3 EvalCallback to capture complete patient attribute
    data during evaluation episodes. In addition to standard eval metrics, this
    callback:
    - Enables log_full_patient_attributes flag on eval env during evaluation
    - Collects full patient data (true and observed attributes) for each eval episode
    - Saves trajectories to disk as .npz files in eval_logs/ subdirectory
    - Computes and saves aggregate eval statistics
    
    Trajectory data enables post-training analysis of:
    - Reward decomposition (individual vs. community components)
    - Policy interpretation (action-attribute associations)
    - Observation error impact (comparing true vs. observed patient features)
    
    Trajectories saved as: <log_path>/eval_logs/eval_<timestep>_ep<episode>.npz
    
    Example:
        >>> eval_callback = DetailedEvalCallback(
        ...     eval_env=eval_env,
        ...     n_eval_episodes=10,
        ...     eval_freq=5000,
        ...     log_path='results/my_run/eval_logs',
        ...     save_patient_trajectories=True
        ... )
        >>> agent.learn(total_timesteps=100000, callback=eval_callback)
    """
    
    def __init__(
        self,
        eval_env: VecEnv,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        save_patient_trajectories: bool = True,
    ):
        """Initialize the DetailedEvalCallback.
        
        Args:
            eval_env (VecEnv): Vectorized environment for evaluation (typically
                DummyVecEnv wrapping a single ABXAMREnv).
            callback_on_new_best (BaseCallback, optional): Callback triggered when
                a new best model is found (higher mean reward).
            callback_after_eval (BaseCallback, optional): Callback triggered after
                each evaluation run.
            n_eval_episodes (int): Number of episodes per evaluation. Default: 5.
            eval_freq (int): Evaluation frequency in timesteps. Default: 10000.
            log_path (str, optional): Directory for evaluation logs. Trajectories
                saved to <log_path>/eval_logs/.
            best_model_save_path (str, optional): Directory to save best model.
            deterministic (bool): If True, use deterministic policy during eval.
                Default: True.
            render (bool): If True, render environment during eval. Default: False.
            verbose (int): Verbosity level: 0 = silent, 1 = info. Default: 1.
            warn (bool): If True, warn about evaluation issues. Default: True.
            save_patient_trajectories (bool): If True, save full patient trajectories
                to .npz files. Default: True.
        """
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        
        self.save_patient_trajectories = save_patient_trajectories
        self.eval_count = 0
        
        # Create eval_logs directory if saving trajectories
        if self.save_patient_trajectories and log_path is not None:
            self.trajectory_dir = Path(log_path) / 'eval_logs'
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.trajectory_dir = None
    
    def _on_step(self) -> bool:
        """
        Called after each step in the training environment.
        
        Checks if it's time to run evaluation and triggers it if so.
        
        Returns:
            bool: True to continue training
        """
        continue_training = True
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize wrapper
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e
            
            # Run custom evaluation with trajectory collection
            continue_training = self._run_evaluation_with_trajectories()
            
            self.eval_count += 1
        
        return continue_training
    
    def _run_evaluation_with_trajectories(self) -> bool:
        """
        Run evaluation episodes and collect patient trajectories.
        
        Returns:
            bool: True to continue training
        """
        # Initialize continue_training flag
        continue_training = True
        
        # Enable full patient attribute logging on eval environment
        self._set_eval_env_logging_flag(True)
        
        # Initialize trajectory storage
        trajectories = []
        episode_rewards = []
        episode_lengths = []

        # Cache antibiotic ordering from the eval environment so AMR arrays are consistent
        antibiotic_names = []
        try:
            antibiotic_names = list(self.eval_env.envs[0].unwrapped.antibiotic_names)
        except Exception:
            antibiotic_names = []
        
        # Run evaluation episodes
        for episode_idx in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_data = {
                'patient_full_data': [],
                'patient_stats': [],
                'actions': [],
                'rewards': [],
                'actual_amr_levels': [],
                'visible_amr_levels': [],
                'antibiotic_names': antibiotic_names,
            }
            
            done = False
            while not done:
                action, _states = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                
                # Collect trajectory data from info dict
                # Note: For VecEnv with n_envs=1, info is a list with one element
                if isinstance(info, list):
                    info = info[0]
                
                if self.save_patient_trajectories and 'patient_full_data' in info:
                    episode_data['patient_full_data'].append(info['patient_full_data'])
                    episode_data['patient_stats'].append(info['patient_stats'])
                    episode_data['actions'].append(action)
                    episode_data['rewards'].append(float(reward))
                    episode_data['actual_amr_levels'].append(info.get('actual_amr_levels', {}))
                    episode_data['visible_amr_levels'].append(info.get('visible_amr_levels', {}))
                
                episode_reward += reward
                episode_length += 1
                
                # Check if episode is done
                if isinstance(done, np.ndarray):
                    done = done[0]
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if self.save_patient_trajectories and episode_data['patient_full_data']:
                trajectories.append(episode_data)
        
        # Disable logging flag after evaluation
        self._set_eval_env_logging_flag(False)
        
        # Save trajectories to disk
        if self.save_patient_trajectories and trajectories and self.trajectory_dir is not None:
            self._save_trajectories(trajectories, episode_rewards, episode_lengths)
        
        # Compute and log evaluation metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_ep_length = np.mean(episode_lengths)
        
        if self.verbose >= 1:
            print(f"Eval at step {self.num_timesteps}: "
                  f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                  f"mean_ep_length={mean_ep_length:.0f}")
        
        # Log to TensorBoard
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        
        # Store evaluation results for parent class tracking
        self.evaluations_results.append(mean_reward)
        self.evaluations_timesteps.append(self.num_timesteps)
        
        # Update best model tracking
        self.last_mean_reward = mean_reward
        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print(f"New best mean reward: {mean_reward:.2f}")
            self.best_mean_reward = mean_reward
            
            # Save best model if path provided
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            
            # Trigger callback if provided
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()
        
        return continue_training
    
    def _set_eval_env_logging_flag(self, value: bool):
        """
        Set log_full_patient_attributes flag on eval environment.
        
        Handles both VecEnv and non-wrapped environments.
        
        Args:
            value: True to enable logging, False to disable
        """
        # For VecEnv, need to access the underlying env(s)
        if hasattr(self.eval_env, 'envs'):
            # VecEnv with multiple environments
            for env in self.eval_env.envs:
                try:
                    env.unwrapped.log_full_patient_attributes = value
                except AttributeError:
                    pass
        else:
            # Single env or wrapped env
            try:
                self.eval_env.unwrapped.log_full_patient_attributes = value
            except AttributeError:
                pass
    
    def _save_trajectories(
        self,
        trajectories: list,
        episode_rewards: list,
        episode_lengths: list
    ):
        """
        Save evaluation trajectories to disk as .npz files.
        
        Args:
            trajectories: List of episode trajectory dicts
            episode_rewards: List of total rewards per episode
            episode_lengths: List of episode lengths
        """
        filename = self.trajectory_dir / f'eval_{self.eval_count:04d}_step_{self.num_timesteps}.npz'
        
        # Convert trajectories to saveable format
        # Use antibiotic names from the first trajectory when available
        antibiotic_names = []
        if trajectories and trajectories[0].get('antibiotic_names'):
            antibiotic_names = list(trajectories[0]['antibiotic_names'])
        elif trajectories and trajectories[0].get('actual_amr_levels'):
            first_levels = trajectories[0]['actual_amr_levels']
            if first_levels:
                antibiotic_names = list(first_levels[0].keys())

        save_dict = {
            'episode_rewards': np.array(episode_rewards),
            'episode_lengths': np.array(episode_lengths),
            'num_episodes': len(trajectories),
            'timestep': self.num_timesteps,
        }

        if antibiotic_names:
            save_dict['antibiotic_names'] = np.array(antibiotic_names)
        
        # Save each episode's data
        for ep_idx, traj in enumerate(trajectories):
            ep_prefix = f'episode_{ep_idx}'
            
            # Save patient data
            if traj['patient_full_data']:
                # Convert list of dicts to dict of arrays
                num_steps = len(traj['patient_full_data'])
                first_step = traj['patient_full_data'][0]

                # Get attribute names from first step
                attr_names = list(first_step['true'].keys())
                num_patients = len(first_step['true'][attr_names[0]])
                num_attrs = len(attr_names)

                # Create arrays: (num_steps, num_patients, num_attrs)
                true_data = np.zeros((num_steps, num_patients, num_attrs))
                obs_data = np.zeros((num_steps, num_patients, num_attrs))

                for step_idx, step_data in enumerate(traj['patient_full_data']):
                    for attr_idx, attr_name in enumerate(attr_names):
                        true_data[step_idx, :, attr_idx] = step_data['true'][attr_name]
                        obs_data[step_idx, :, attr_idx] = step_data['observed'][attr_name]

                save_dict[f'{ep_prefix}/patient_true'] = true_data
                save_dict[f'{ep_prefix}/patient_observed'] = obs_data
                save_dict[f'{ep_prefix}/patient_attrs'] = attr_names

            # Save actions, rewards, AMR levels
            save_dict[f'{ep_prefix}/actions'] = np.array(traj['actions'])
            save_dict[f'{ep_prefix}/rewards'] = np.array(traj['rewards'])

            # Store AMR time series if available
            if antibiotic_names and traj['actual_amr_levels']:
                num_steps_amr = len(traj['actual_amr_levels'])
                actual_amr_arr = np.zeros((num_steps_amr, len(antibiotic_names)))
                visible_amr_arr = np.zeros((num_steps_amr, len(antibiotic_names)))
                for step_idx, (actual_step, visible_step) in enumerate(zip(traj['actual_amr_levels'], traj['visible_amr_levels'])):
                    for abx_idx, abx_name in enumerate(antibiotic_names):
                        actual_amr_arr[step_idx, abx_idx] = actual_step.get(abx_name, np.nan)
                        visible_amr_arr[step_idx, abx_idx] = visible_step.get(abx_name, np.nan)

                save_dict[f'{ep_prefix}/actual_amr_levels'] = actual_amr_arr
                save_dict[f'{ep_prefix}/visible_amr_levels'] = visible_amr_arr
        
        # Save to file
        np.savez_compressed(filename, **save_dict)
        
        if self.verbose >= 1:
            print(f"Saved evaluation trajectories to {filename}")
