"""Unit tests for DynamicsModel."""

import pytest
import numpy as np
import torch
from gymnasium.spaces import Discrete, MultiDiscrete
import tempfile
import os

from abx_amr_simulator.mbpo.dynamics_model import DynamicsModel


class TestDynamicsModelInit:
    """Test DynamicsModel initialization."""
    
    def test_init_with_discrete_action_space(self):
        """Test initialization with Discrete action space."""
        obs_dim = 10
        action_space = Discrete(n=3)
        config = {'hidden_dims': [64, 64], 'learning_rate': 1e-3, 'device': 'cpu'}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        assert model.obs_dim == obs_dim
        assert model.action_dim == 3
        assert not model.is_multidiscrete
        assert model.device.type == 'cpu'
        assert not model.is_trained
    
    def test_init_with_multidiscrete_action_space(self):
        """Test initialization with MultiDiscrete action space."""
        obs_dim = 10
        action_space = MultiDiscrete(nvec=[3, 3])
        config = {'hidden_dims': [64, 64], 'learning_rate': 1e-3, 'device': 'cpu'}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        assert model.obs_dim == obs_dim
        assert model.action_dim == 6  # 3 + 3
        assert model.is_multidiscrete
        assert np.array_equal(model.action_nvec, [3, 3])
        assert not model.is_trained
    
    def test_init_with_default_config(self):
        """Test initialization with default config values."""
        obs_dim = 5
        action_space = Discrete(n=2)
        config = {}  # Empty config should use defaults
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        # Check defaults are applied
        assert model.config.get('hidden_dims', [256, 256, 256]) == [256, 256, 256]
        assert model.config.get('learning_rate', 1e-3) == 1e-3
    
    def test_init_with_invalid_action_space(self):
        """Test initialization with unsupported action space raises error."""
        from gymnasium.spaces import Box
        
        obs_dim = 5
        action_space = Box(low=0, high=1, shape=(2,))
        config = {}
        
        with pytest.raises(ValueError, match="Unsupported action space"):
            DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)


class TestActionEncoding:
    """Test action encoding functionality."""
    
    def test_encode_discrete_action(self):
        """Test one-hot encoding for Discrete action space."""
        obs_dim = 5
        action_space = Discrete(n=3)
        config = {'device': 'cpu'}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        # Test each action
        action_0 = np.array([0])
        encoded_0 = model._encode_action(action_0)
        assert torch.allclose(encoded_0, torch.tensor([1.0, 0.0, 0.0]))
        
        action_1 = np.array([1])
        encoded_1 = model._encode_action(action_1)
        assert torch.allclose(encoded_1, torch.tensor([0.0, 1.0, 0.0]))
        
        action_2 = np.array([2])
        encoded_2 = model._encode_action(action_2)
        assert torch.allclose(encoded_2, torch.tensor([0.0, 0.0, 1.0]))
    
    def test_encode_multidiscrete_action(self):
        """Test one-hot encoding for MultiDiscrete action space."""
        obs_dim = 5
        action_space = MultiDiscrete(nvec=[2, 3])
        config = {'device': 'cpu'}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        # Test action [0, 1] -> [1,0] concatenated with [0,1,0]
        action = np.array([0, 1])
        encoded = model._encode_action(action)
        expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
        assert torch.allclose(encoded, expected)
        
        # Test action [1, 2] -> [0,1] concatenated with [0,0,1]
        action = np.array([1, 2])
        encoded = model._encode_action(action)
        expected = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
        assert torch.allclose(encoded, expected)
    
    def test_encode_action_with_integer_input(self):
        """Test action encoding with integer input (not array)."""
        obs_dim = 5
        action_space = Discrete(n=3)
        config = {'device': 'cpu'}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        # Test with plain integer
        action = 1
        encoded = model._encode_action(action)
        assert torch.allclose(encoded, torch.tensor([0.0, 1.0, 0.0]))


class TestPredict:
    """Test forward prediction functionality."""
    
    def test_predict_output_shape(self):
        """Test that predict returns correct output shapes."""
        obs_dim = 10
        action_space = Discrete(n=3)
        config = {'hidden_dims': [32, 32], 'device': 'cpu'}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        obs = np.random.randn(obs_dim)
        action = np.array([1])
        
        next_obs, reward = model.predict(obs=obs, action=action)
        
        assert next_obs.shape == (obs_dim,)
        assert isinstance(reward, float)
    
    def test_predict_deterministic(self):
        """Test that predict is deterministic (no dropout in eval mode)."""
        obs_dim = 5
        action_space = Discrete(n=2)
        config = {'hidden_dims': [16, 16], 'device': 'cpu'}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        obs = np.random.randn(obs_dim)
        action = np.array([0])
        
        # Call predict twice with same inputs
        next_obs_1, reward_1 = model.predict(obs=obs, action=action)
        next_obs_2, reward_2 = model.predict(obs=obs, action=action)
        
        assert np.allclose(next_obs_1, next_obs_2)
        assert reward_1 == reward_2
    
    def test_predict_multidiscrete(self):
        """Test predict with MultiDiscrete action space."""
        obs_dim = 8
        action_space = MultiDiscrete(nvec=[3, 3])
        config = {'hidden_dims': [32, 32], 'device': 'cpu'}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        obs = np.random.randn(obs_dim)
        action = np.array([1, 2])
        
        next_obs, reward = model.predict(obs=obs, action=action)
        
        assert next_obs.shape == (obs_dim,)
        assert isinstance(reward, float)


class TestTraining:
    """Test model training functionality."""
    
    def test_train_on_data_basic(self):
        """Test basic training on synthetic data."""
        obs_dim = 5
        action_space = Discrete(n=2)
        config = {'hidden_dims': [32, 32], 'device': 'cpu', 'learning_rate': 1e-2}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        # Generate synthetic data
        data = []
        for _ in range(100):
            obs = np.random.randn(obs_dim)
            action = np.array([np.random.randint(0, 2)])
            next_obs = obs + 0.1 * np.random.randn(obs_dim)  # Simple dynamics
            reward = np.random.randn()
            
            data.append({
                'obs': obs,
                'action': action,
                'next_obs': next_obs,
                'reward': reward,
                'done': False
            })
        
        # Train model
        metrics = model.train_on_data(
            data=data,
            epochs=10,
            batch_size=32,
            verbose=False
        )
        
        assert 'total_loss' in metrics
        assert 'obs_loss' in metrics
        assert 'reward_loss' in metrics
        assert model.is_trained
    
    def test_train_loss_decreases(self):
        """Test that training loss decreases over epochs."""
        obs_dim = 5
        action_space = Discrete(n=2)
        config = {'hidden_dims': [64, 64], 'device': 'cpu', 'learning_rate': 1e-2}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        # Generate synthetic data with simple pattern
        data = []
        for _ in range(200):
            obs = np.random.randn(obs_dim)
            action = np.array([np.random.randint(0, 2)])
            # Deterministic dynamics: next_obs = obs + action
            next_obs = obs + float(action[0])
            reward = float(action[0])
            
            data.append({
                'obs': obs,
                'action': action,
                'next_obs': next_obs,
                'reward': reward,
                'done': False
            })
        
        # Train for multiple rounds and track loss
        initial_metrics = model.train_on_data(data=data, epochs=5, batch_size=32, verbose=False)
        initial_loss = initial_metrics['total_loss']
        
        final_metrics = model.train_on_data(data=data, epochs=50, batch_size=32, verbose=False)
        final_loss = final_metrics['total_loss']
        
        # Loss should decrease with training
        assert final_loss < initial_loss
    
    def test_train_empty_data_raises_error(self):
        """Test that training on empty data raises ValueError."""
        obs_dim = 5
        action_space = Discrete(n=2)
        config = {'device': 'cpu'}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        with pytest.raises(ValueError, match="Cannot train on empty data"):
            model.train_on_data(data=[], epochs=10, batch_size=32)
    
    def test_train_multidiscrete_actions(self):
        """Test training with MultiDiscrete action space."""
        obs_dim = 5
        action_space = MultiDiscrete(nvec=[3, 3])
        config = {'hidden_dims': [32, 32], 'device': 'cpu', 'learning_rate': 1e-2}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        # Generate data
        data = []
        for _ in range(100):
            obs = np.random.randn(obs_dim)
            action = np.array([np.random.randint(0, 3), np.random.randint(0, 3)])
            next_obs = obs + 0.1 * np.random.randn(obs_dim)
            reward = np.random.randn()
            
            data.append({
                'obs': obs,
                'action': action,
                'next_obs': next_obs,
                'reward': reward,
                'done': False
            })
        
        # Should train without errors
        metrics = model.train_on_data(data=data, epochs=5, batch_size=32, verbose=False)
        assert 'total_loss' in metrics
        assert model.is_trained


class TestSaveLoad:
    """Test model save/load functionality."""
    
    def test_save_and_load(self):
        """Test saving and loading model preserves weights."""
        obs_dim = 5
        action_space = Discrete(n=3)
        config = {'hidden_dims': [32, 32], 'device': 'cpu', 'learning_rate': 1e-3}
        
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        
        # Generate and train on data
        data = []
        for _ in range(50):
            obs = np.random.randn(obs_dim)
            action = np.array([np.random.randint(0, 3)])
            next_obs = obs + 0.1 * np.random.randn(obs_dim)
            reward = np.random.randn()
            
            data.append({
                'obs': obs,
                'action': action,
                'next_obs': next_obs,
                'reward': reward,
                'done': False
            })
        
        model.train_on_data(data=data, epochs=5, batch_size=16, verbose=False)
        
        # Make predictions before save
        test_obs = np.random.randn(obs_dim)
        test_action = np.array([1])
        next_obs_before, reward_before = model.predict(obs=test_obs, action=test_action)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'model.pt')
            model.save(path=save_path)
            
            # Load model
            model_loaded = DynamicsModel(
                obs_dim=obs_dim,
                action_space=action_space,
                config=config
            )
            model_loaded.load(path=save_path)
            
            # Make predictions after load
            next_obs_after, reward_after = model_loaded.predict(obs=test_obs, action=test_action)
            
            # Predictions should be identical
            assert np.allclose(next_obs_before, next_obs_after)
            assert reward_before == reward_after
            assert model_loaded.is_trained
    
    def test_load_preserves_training_state(self):
        """Test that loading preserves is_trained flag."""
        obs_dim = 5
        action_space = Discrete(n=2)
        config = {'hidden_dims': [16, 16], 'device': 'cpu'}
        
        # Create and train model
        model = DynamicsModel(obs_dim=obs_dim, action_space=action_space, config=config)
        data = [
            {
                'obs': np.random.randn(obs_dim),
                'action': np.array([0]),
                'next_obs': np.random.randn(obs_dim),
                'reward': 0.0,
                'done': False
            }
            for _ in range(20)
        ]
        model.train_on_data(data=data, epochs=2, batch_size=10, verbose=False)
        assert model.is_trained
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'model.pt')
            model.save(path=save_path)
            
            model_loaded = DynamicsModel(
                obs_dim=obs_dim,
                action_space=action_space,
                config=config
            )
            model_loaded.load(path=save_path)
            
            assert model_loaded.is_trained
