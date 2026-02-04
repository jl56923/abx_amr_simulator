"""
Dynamics Model for MBPO.

Learns environment dynamics p(s', r | s, a) via supervised learning.
Supports both Discrete and MultiDiscrete action spaces.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Space, Discrete, MultiDiscrete


class DynamicsModel:
    """
    Neural network model that predicts next observation and reward given current observation and action.
    
    Supports MultiDiscrete action spaces by one-hot encoding actions before feeding to the network.
    """
    
    def __init__(self, obs_dim: int, action_space: Space, config: Dict):
        """
        Initialize the dynamics model.
        
        Args:
            obs_dim: Dimensionality of observation space (flattened)
            action_space: Gymnasium action space (Discrete or MultiDiscrete)
            config: Configuration dict with keys:
                - hidden_dims: List[int], hidden layer dimensions (default: [256, 256, 256])
                - learning_rate: float, Adam learning rate (default: 1e-3)
                - device: str, 'cuda', 'cpu', or 'auto' (default: 'auto')
        """
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.config = config
        
        # Determine action encoding dimension
        if isinstance(action_space, MultiDiscrete):
            # Sum of all discrete dimensions for one-hot encoding
            self.action_dim = int(np.sum(action_space.nvec))
            self.action_nvec = action_space.nvec
            self.is_multidiscrete = True
        elif isinstance(action_space, Discrete):
            self.action_dim = int(action_space.n)
            self.is_multidiscrete = False
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")
        
        # Extract config parameters
        hidden_dims = config.get('hidden_dims', [256, 256, 256])
        learning_rate = config.get('learning_rate', 1e-3)
        device_str = config.get('device', 'auto')
        
        # Set device
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
        
        # Build neural network
        layers = []
        input_dim = obs_dim + self.action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output: next_obs (obs_dim) + reward (1)
        layers.append(nn.Linear(input_dim, obs_dim + 1))
        
        self.model = nn.Sequential(*layers)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Track whether model has been trained
        self.is_trained = False
    
    def _encode_action(self, action: np.ndarray) -> torch.Tensor:
        """
        Convert action to one-hot encoded tensor.
        
        Args:
            action: Action array (int or array of ints for MultiDiscrete)
        
        Returns:
            One-hot encoded action tensor
        """
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, dtype=torch.long)
        elif isinstance(action, (int, np.integer)):
            action = torch.tensor([action], dtype=torch.long)
        elif isinstance(action, torch.Tensor):
            action = action.long()
        else:
            action = torch.as_tensor(action, dtype=torch.long)
        
        # Handle 0-dim tensors (convert to 1-dim)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        if self.is_multidiscrete:
            # MultiDiscrete: concatenate one-hot for each dimension
            one_hots = []
            for i, (a, n) in enumerate(zip(action, self.action_nvec)):
                one_hot = torch.zeros(int(n), device=self.device)
                a_val = int(a.item() if isinstance(a, torch.Tensor) else a)
                one_hot[a_val] = 1.0
                one_hots.append(one_hot)
            return torch.cat(one_hots)
        else:
            # Discrete: single one-hot
            one_hot = torch.zeros(self.action_dim, device=self.device)
            a_val = int(action[0].item() if isinstance(action[0], torch.Tensor) else action[0])
            one_hot[a_val] = 1.0
            return one_hot
    
    def predict(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict next observation and reward given current observation and action.
        
        Args:
            obs: Current observation (1D numpy array)
            action: Action (int or array of ints for MultiDiscrete)
        
        Returns:
            next_obs: Predicted next observation (numpy array)
            reward: Predicted reward (float)
        """
        self.model.eval()
        with torch.no_grad():
            # Convert to tensors
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action_encoded = self._encode_action(action)
            
            # Concatenate observation and action
            input_tensor = torch.cat([obs_tensor, action_encoded])
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Split output into next_obs and reward
            next_obs = output[:self.obs_dim].cpu().numpy()
            reward = output[self.obs_dim].cpu().item()
            
            return next_obs, reward
    
    def train_on_data(
        self,
        data: List[Dict],
        epochs: int,
        batch_size: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train dynamics model on collected real transitions.
        
        Args:
            data: List of transition dicts with keys:
                - 'obs': Current observation
                - 'action': Action taken
                - 'next_obs': Next observation
                - 'reward': Reward received
                - 'done': Terminal flag
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
        
        Returns:
            metrics: Dict with 'obs_loss', 'reward_loss', 'total_loss'
        """
        if len(data) == 0:
            raise ValueError("Cannot train on empty data")
        
        self.model.train()
        
        # Extract data arrays
        obs_list = [transition['obs'] for transition in data]
        action_list = [transition['action'] for transition in data]
        next_obs_list = [transition['next_obs'] for transition in data]
        reward_list = [transition['reward'] for transition in data]
        
        num_samples = len(data)
        if verbose:
            print(f"Training dynamics model on {num_samples} transitions for {epochs} epochs...")
        
        # Training loop
        final_metrics = {}
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            epoch_loss = 0.0
            epoch_obs_loss = 0.0
            epoch_reward_loss = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Prepare batch
                obs_batch = torch.as_tensor(
                    np.array([obs_list[j] for j in batch_indices]),
                    dtype=torch.float32,
                    device=self.device
                )
                action_batch = [action_list[j] for j in batch_indices]
                next_obs_batch = torch.as_tensor(
                    np.array([next_obs_list[j] for j in batch_indices]),
                    dtype=torch.float32,
                    device=self.device
                )
                reward_batch = torch.as_tensor(
                    np.array([reward_list[j] for j in batch_indices]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Encode actions
                action_encoded_batch = torch.stack([
                    self._encode_action(a) for a in action_batch
                ])
                
                # Forward pass
                input_batch = torch.cat([obs_batch, action_encoded_batch], dim=1)
                output = self.model(input_batch)
                
                pred_next_obs = output[:, :self.obs_dim]
                pred_reward = output[:, self.obs_dim]
                
                # Compute losses
                obs_loss = nn.MSELoss()(pred_next_obs, next_obs_batch)
                reward_loss = nn.MSELoss()(pred_reward, reward_batch)
                total_loss = obs_loss + reward_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += total_loss.item()
                epoch_obs_loss += obs_loss.item()
                epoch_reward_loss += reward_loss.item()
                num_batches += 1
            
            # Average losses over epoch
            avg_loss = epoch_loss / num_batches
            avg_obs_loss = epoch_obs_loss / num_batches
            avg_reward_loss = epoch_reward_loss / num_batches
            
            final_metrics = {
                'total_loss': avg_loss,
                'obs_loss': avg_obs_loss,
                'reward_loss': avg_reward_loss
            }
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f} "
                      f"(Obs: {avg_obs_loss:.6f}, Reward: {avg_reward_loss:.6f})")
        
        self.is_trained = True
        return final_metrics
    
    def save(self, path: str):
        """
        Save model weights to disk.
        
        Args:
            path: File path to save model (will save as .pt file)
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'is_multidiscrete': self.is_multidiscrete,
            'action_nvec': self.action_nvec if self.is_multidiscrete else None,
            'is_trained': self.is_trained
        }, path)
    
    def load(self, path: str):
        """
        Load model weights from disk.
        
        Args:
            path: File path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_trained = checkpoint.get('is_trained', True)
