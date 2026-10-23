"""
Reinforcement Learning Environment for Cybersecurity Threat Classification.
Uses a Contextual Bandit formulation.
"""
import numpy as np


class ThreatClassificationEnv:
    """
    Contextual Bandit environment for threat classification.
    
    Each step presents a network traffic sample (context).
    Agent chooses an action (threat class).
    Reward is 1 if correct, 0 otherwise.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, class_to_idx: dict):
        """
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels as strings
            class_to_idx: Mapping from class names to indices
        """
        self.X = X.astype(np.float32)
        self.y = np.array([class_to_idx[label] for label in y], dtype=np.int64)
        self.n_actions = len(class_to_idx)
        self.n_features = X.shape[1]
        self._idx = 0
    
    def reset(self):
        """Reset environment to first sample."""
        self._idx = 0
        return self.X[self._idx]
    
    def step(self, action: int):
        """
        Take action and receive reward.
        
        Returns:
            obs: Current observation (for next step)
            reward: 1 if correct, 0 otherwise
            done: True if all samples processed
            info: Additional information
        """
        correct = int(action == self.y[self._idx])
        self._idx += 1
        
        done = self._idx >= len(self.X)
        obs = self.X[self._idx] if not done else None
        
        return obs, correct, done, {"correct": correct}
    
    def __len__(self):
        return len(self.X)
