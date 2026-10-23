"""
Utility functions for Quantum Reinforcement Learning.
Includes replay buffer and data loading utilities.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 5000, device: str = "cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, obs, action, reward, next_obs, done):
        """Store a transition in the buffer."""
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        return (
            torch.tensor(np.array(obs), dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device)
        )
    
    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for Q-value estimation."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def load_dataset(csv_path: str = "data/dataset.csv", test_size: float = 0.2, seed: int = 42):
    """
    Load and preprocess the cybersecurity dataset.
    
    Returns:
        X_train, y_train, X_test, y_test, classes, class_to_idx
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    y = df["label"]
    
    # Identify column types
    num_cols = [c for c in X.columns if c != "protocol"]
    cat_cols = ["protocol"] if "protocol" in X.columns else []
    
    # Preprocessing pipeline
    transformers = [("num", StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    # Transform
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)
    
    # Class mapping
    classes = sorted(y.unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    print(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Classes: {classes}")
    
    return X_train, y_train, X_test, y_test, classes, class_to_idx
