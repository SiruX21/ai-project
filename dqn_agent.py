import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    """Deep Q-Network for Flappy Bird."""
    
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)  # Q-values for each action


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with experience replay and target network."""
    
    def __init__(self, state_dim, action_dim, device="cpu", lr=5e-4):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-networks
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # Optimizer - reduced learning rate for stability
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = 0.99  # Discount factor
        
        # Epsilon-greedy exploration - faster decay to start learning sooner
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 20000  # Faster decay (was 50000)
        self.total_steps = 0
        
        # Replay buffer
        self.replay = ReplayBuffer(capacity=50000)
    
    def epsilon(self):
        """Compute current epsilon value (exponential decay)."""
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-1.0 * self.total_steps / self.eps_decay)
    
    def act(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: current state
            training: if False, always use greedy policy
        
        Returns:
            action (0 or 1)
        """
        if training:
            self.total_steps += 1
            eps = self.epsilon()
            if random.random() < eps:
                return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def train_step(self, batch_size=64):
        """
        Perform one training step using experience replay.
        
        Returns:
            loss value (or None if buffer too small)
        """
        if len(self.replay) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Current Q values
        q_values = self.q_net(states).gather(1, actions)
        
        # Target Q values (using target network)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_values, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        
        return float(loss.item())
    
    def update_target(self):
        """Update target network with current Q-network weights."""
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def save(self, filepath):
        """Save the Q-network weights."""
        torch.save(self.q_net.state_dict(), filepath)
    
    def load(self, filepath):
        """Load Q-network weights."""
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

