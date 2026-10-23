"""
Deep Q-Network (DQN) Agent.
Classical baseline for comparison with Quantum DQN.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import MLP


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        n_actions: int, 
        hidden: int = 128, 
        lr: float = 1e-3, 
        gamma: float = 0.99, 
        device: str = "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        
        # Q-network and target network
        self.q_net = MLP(obs_dim, n_actions, hidden).to(device)
        self.target_net = MLP(obs_dim, n_actions, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
    
    def act(self, obs: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def update(self, batch) -> float:
        """Update Q-network using a batch of experiences."""
        obs, actions, rewards, next_obs, dones = batch
        
        # Current Q-values
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_obs).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss and optimization
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def soft_update(self, tau: float = 0.01):
        """Soft update target network."""
        for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Model loaded from {path}")
