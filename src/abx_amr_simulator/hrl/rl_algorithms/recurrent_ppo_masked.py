"""RecurrentPPO subclass with masking for clipped manager transitions in HRL.

This module provides RecurrentPPO_Masked, which extends stable-baselines3's RecurrentPPO
to support masking of clipped manager transitions during training. This is similar to
PPO_Masked but handles the recurrent (LSTM) aspect properly.

For RecurrentPPO, we need to combine two masks:
1. Sequence mask (from RNN) - handles variable length sequences
2. Clipping mask - marks clipped transitions as non-trainable

Both masks are combined with logical AND to get the effective training mask.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch as th
from torch.nn import functional as F

from sb3_contrib import RecurrentPPO
from gymnasium import spaces


class RecurrentPPO_Masked(RecurrentPPO):
    """RecurrentPPO variant with masking support for clipped manager transitions in HRL.
    
    This class extends stable-baselines3's RecurrentPPO to support masking of clipped
    manager transitions during training. It combines two masks:
    - Sequence mask from RNN (for variable-length sequences)
    - Clipping mask (manager_transition_trainable flag)
    
    When a macro-action is clipped at episode boundary, the corresponding manager
    transition is marked as non-trainable. During training, loss contributions from
    non-trainable transitions are zeroed out.
    
    Example:
        >>> from abx_amr_simulator.hrl.rl_algorithms import RecurrentPPO_Masked
        >>> agent = RecurrentPPO_Masked(
        ...     policy='MlpLstmPolicy',
        ...     env=hrl_env,
        ...     learning_rate=3e-4,
        ...     n_steps=256,
        ... )
        >>> agent.learn(total_timesteps=10000)
    """

    def __init__(self, *args, **kwargs):
        """Initialize RecurrentPPO_Masked agent.
        
        Args:
            *args: Positional arguments passed to RecurrentPPO.__init__
            **kwargs: Keyword arguments passed to RecurrentPPO.__init__
        """
        super().__init__(*args, **kwargs)
        # Store trainability mask for all collected transitions
        self.trainable_mask = None

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect rollouts and extract trainability mask from info.
        
        Overrides RecurrentPPO.collect_rollouts() to capture the 'manager_transition_trainable'
        field from environment info dicts after each step.
        
        Args:
            env: The environment to collect rollouts from
            callback: Callback function for logging
            rollout_buffer: Buffer to store transitions
            n_rollout_steps: Number of steps to collect
            
        Returns:
            bool: Whether all episodes were completed without truncation
        """
        # Initialize trainable_mask storage
        trainable_flags = []
        
        # We'll monkey-patch step to capture infos
        original_step = env.step
        
        def step_wrapper(action):
            nonlocal trainable_flags
            step_result = original_step(action)
            # Handle both Gym API (5 outputs) and VecEnv API (4 outputs)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, dones, info = step_result
                terminated = dones
                truncated = False
            # Extract trainability flag from info (handle both single env and VecEnv)
            if isinstance(info, list):
                # VecEnv case
                trainable_flags.extend([
                    i.get('manager_transition_trainable', True) if isinstance(i, dict) else True
                    for i in info
                ])
            elif isinstance(info, dict):
                # Single env case
                trainable_flags.append(info.get('manager_transition_trainable', True))
            else:
                trainable_flags.append(True)
            return step_result
        
        # Temporarily replace env.step
        env.step = step_wrapper
        try:
            # Call parent to collect rollouts
            done = super().collect_rollouts(
                env=env,
                callback=callback,
                rollout_buffer=rollout_buffer,
                n_rollout_steps=n_rollout_steps,
            )
        finally:
            # Restore original step
            env.step = original_step
        
        # Store trainability mask for use during training
        if trainable_flags:
            self.trainable_mask = np.array(trainable_flags, dtype=np.float32)
        
        return done

    def train(self) -> None:
        """Perform policy gradient update with combined masking.
        
        Overrides RecurrentPPO.train() to apply trainability mask during loss computation.
        Combines sequence mask (from RNN) with clipping mask to get effective training mask.
        Clipped transitions contribute zero gradient to the policy and value networks.
        """
        assert self.env is not None

        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        entropy_coef = self.ent_coef
        if isinstance(self.ent_coef, float):
            entropy_coef = float(self.ent_coef)

        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # Optional: log what fraction of transitions are trainable
        if self.trainable_mask is not None:
            trainable_fraction = np.mean(self.trainable_mask)
            self.logger.record("train/trainable_fraction", trainable_fraction)

        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        
        batch_idx = 0

        # Iteration over minibatches
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            # Re-sample the noise for the corrupted actions (exploration)
            if self.use_sde:
                self.policy.reset_noise(self.batch_size)

            # Forward pass (with RNN hidden states)
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations,
                actions,
                rollout_data.lstm_states,
                rollout_data.episode_starts,
            )
            values = values.flatten()

            # Normalize advantage
            advantages = rollout_data.advantages
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Extract combined mask: sequence mask AND clipping mask
            batch_size_actual = len(rollout_data.observations)
            
            # Start with 1 (all trainable)
            combined_mask = th.ones(batch_size_actual, device=self.device, dtype=th.float32)
            
            # Apply sequence mask if present (from RNN)
            if hasattr(rollout_data, 'mask') and rollout_data.mask is not None:
                sequence_mask = rollout_data.mask.float()
                combined_mask = combined_mask * sequence_mask
            
            # Apply clipping mask if available
            if self.trainable_mask is not None and len(self.trainable_mask) >= batch_size_actual:
                mask_start = batch_idx * self.batch_size
                mask_end = min(mask_start + batch_size_actual, len(self.trainable_mask))
                clipping_mask = th.tensor(
                    self.trainable_mask[mask_start:mask_end],
                    device=self.device,
                    dtype=th.float32,
                )
                # Pad if necessary
                if len(clipping_mask) < batch_size_actual:
                    clipping_mask = th.cat([
                        clipping_mask,
                        th.ones(batch_size_actual - len(clipping_mask), device=self.device, dtype=th.float32)
                    ])
                combined_mask = combined_mask * clipping_mask

            batch_idx += 1

            # =========================
            # Compute losses with combined masking
            # =========================
            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            policy_loss_unmasked = -advantages * ratio
            policy_loss_clipped = -advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss_per_step = th.max(policy_loss_unmasked, policy_loss_clipped)
            
            # Apply combined mask to policy loss
            policy_loss_masked = policy_loss_per_step * combined_mask
            policy_loss = policy_loss_masked.mean()

            # Value loss
            value_loss_unmasked = F.smooth_l1_loss(values, rollout_data.returns, reduction="none")
            if clip_range_vf is not None:
                values_clipped = rollout_data.old_values + th.clamp(
                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                )
                value_loss_clipped = F.smooth_l1_loss(values_clipped, rollout_data.returns, reduction="none")
                value_loss_per_step = th.max(value_loss_unmasked, value_loss_clipped)
            else:
                value_loss_per_step = value_loss_unmasked
            
            # Apply combined mask to value loss
            value_loss_masked = value_loss_per_step * combined_mask
            value_loss = value_loss_masked.mean()

            # Entropy loss (not masked - we want regularization throughout)
            if entropy is not None:
                entropy_loss = -th.mean(entropy)
            else:
                entropy_loss = th.tensor(0.0, device=self.device)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Compute clip fraction
            with th.no_grad():
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )

            pg_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(clip_fractions) if clip_fractions else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions) if clip_fractions else 0.0)
        self.logger.record("train/loss", loss.item() if 'loss' in locals() else 0.0)
        self.logger.record("train/learning_rate", self.learning_rate)
