"""Early stopping callback for stable-baselines3 training."""

from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """Stop training when a performance metric plateaus (no improvement for N evaluations).
    
    Monitors a specified metric (e.g., mean reward) during training and stops automatically
    when improvement stalls. Useful for avoiding wasted computation when the agent has
    converged or plateaued early.
    
    **Key behavior**:
    - Tracks metric values logged to TensorBoard during training (e.g., 'eval/mean_reward')
    - Stops if `patience` consecutive evaluations show no improvement > `min_delta`
    - Logs stopping decision to console and TensorBoard
    - Compatible with standard EvalCallback (reads from model.logger)
    
    **Typical usage with training**:
        >>> # Ensure at least 100 episodes (assuming 500 timesteps/episode)
        >>> early_stop = EarlyStoppingCallback(
        ...     patience=10,
        ...     min_delta=0.01,
        ...     min_timesteps=50000,  # 100 episodes * 500 timesteps
        ...     metric_name='eval/mean_reward',
        ...     verbose=1
        ... )
        >>> agent.learn(total_timesteps=1000000, callback=early_stop)
    
    Args:
        patience (int): Number of evaluations without improvement before stopping. Default: 10.
        min_delta (float): Minimum improvement threshold. If metric improves by < min_delta,
            not counted as improvement. Default: 0.0 (any improvement counts).
        min_timesteps (int): Minimum number of training timesteps before early stopping can
            trigger. Useful to ensure a minimum training duration (e.g., 100 episodes).
            Calculate as: min_episodes × max_timesteps_per_episode. Default: 0 (no minimum).
        metric_name (str): TensorBoard metric key to monitor (e.g., 'eval/mean_reward').
            Default: 'eval/mean_reward'.
        verbose (int): Verbosity level: 0 = silent, 1 = info messages. Default: 1.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        min_timesteps: int = 0,
        metric_name: str = 'eval/mean_reward',
        verbose: int = 1,
    ):
        """Initialize the early stopping callback."""
        super().__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.min_timesteps = min_timesteps
        self.metric_name = metric_name
        self.best_metric = None
        self.evaluations_since_improvement = 0
        self.last_seen_metric = None  # Track last value to detect when metric changes
        self.last_seen_dict_id: Optional[int] = None  # Track logger dict identity to detect updates in tests
    
    def _on_step(self) -> bool:
        """
        Called after each training step.
        
        Checks if a new metric value is available and updates best_metric tracking.
        Stops training if patience threshold is exceeded.
        
        Returns:
            bool: True to continue training, False to stop.
        """
        # Check if we've reached the minimum training duration
        if self.num_timesteps < self.min_timesteps:
            # Haven't reached minimum timesteps yet; continue training regardless of metrics
            return True
        
        # Try to read the metric from the model's logger
        if self.model.logger is None:
            if self.verbose > 0:
                print(f"Warning: No logger attached to model; EarlyStoppingCallback cannot monitor '{self.metric_name}'")
            return True
        
        # Check if metric value is available in logger (this is updated by EvalCallback)
        current_metric = self._get_metric_value()
        
        if current_metric is None:
            # Metric not yet available; continue training
            return True
        
        # Detect whether there's a new evaluation update to process.
        # In real training, EvalCallback updates values in-place within the same logger dict.
        # In unit tests, the dict object may be replaced for each simulated evaluation.
        dict_id = None
        try:
            if hasattr(self.model.logger, 'name_to_value'):
                dict_id = id(self.model.logger.name_to_value)
        except Exception:
            dict_id = None

        has_new_value = (self.last_seen_metric is None) or (abs(current_metric - self.last_seen_metric) >= 1e-12)
        has_new_dict = (self.last_seen_dict_id is None) or (dict_id is not None and dict_id != self.last_seen_dict_id)

        # If neither the value nor the underlying dict changed, skip processing to avoid double-counting
        if not has_new_value and not has_new_dict:
            return True
        
        # Update last seen markers
        self.last_seen_metric = current_metric
        self.last_seen_dict_id = dict_id
        
        # Check if this is an improvement
        if self.best_metric is None:
            # First evaluation
            self.best_metric = current_metric
            self.evaluations_since_improvement = 0
            if self.verbose > 0:
                print(f"[EarlyStoppingCallback] Initial {self.metric_name}: {current_metric:.4f}")
            return True
        
        # Check for improvement
        improvement = current_metric - self.best_metric
        if improvement > self.min_delta:
            # Improvement found
            self.best_metric = current_metric
            self.evaluations_since_improvement = 0
            if self.verbose > 0:
                print(f"[EarlyStoppingCallback] {self.metric_name} improved to {current_metric:.4f}")
            return True
        else:
            # No improvement
            self.evaluations_since_improvement += 1
            if self.verbose > 0:
                print(
                    f"[EarlyStoppingCallback] No improvement: {self.metric_name}={current_metric:.4f} "
                    f"(best: {self.best_metric:.4f}). "
                    f"Patience: {self.evaluations_since_improvement}/{self.patience}"
                )
            
            # Check if patience exceeded
            if self.evaluations_since_improvement >= self.patience:
                if self.verbose > 0:
                    print(
                        f"[EarlyStoppingCallback] Stopping training: "
                        f"No improvement for {self.patience} evaluations. "
                        f"(Timesteps: {self.num_timesteps}, Min: {self.min_timesteps})"
                    )
                return False  # Stop training
        
        return True
    
    def _get_metric_value(self) -> Optional[float]:
        """
        Extract metric value from logger.
        
        Reads from model.logger.name_to_value dict, which is updated by EvalCallback
        after each evaluation. Returns the most recent value if available.
        
        Returns:
            float: Current metric value, or None if not yet available.
        """
        try:
            # The logger maintains a dict of logged values
            if hasattr(self.model.logger, 'name_to_value'):
                value = self.model.logger.name_to_value.get(self.metric_name)
                return float(value) if value is not None else None
        except (AttributeError, ValueError, TypeError):
            pass
        
        return None
