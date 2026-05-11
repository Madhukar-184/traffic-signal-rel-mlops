"""
Deep Q-Network (DQN) Agent  —  PyTorch version
------------------------------------------------
Features:
  • Experience Replay buffer
  • Target network (synced every C episodes)
  • Epsilon-greedy exploration with multiplicative decay
  • Gradient clipping for training stability

Use this file when running on a machine with PyTorch installed.
Fall back to agent_numpy.py for zero-dependency environments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# ──────────────────────────────────────────────────────────────────────
# Q-Network
# ──────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    3-layer fully-connected network.
    Input  : state vector  (9-dim)
    Output : Q-value per action (2-dim)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr            = 1e-3,
        gamma         = 0.99,
        epsilon_start = 1.0,
        epsilon_end   = 0.05,
        epsilon_decay = 0.988,
        batch_size    = 64,
        target_update = 10,
        buffer_cap    = 10_000,
        device        = None,
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = device or torch.device("cpu")

        self.q_net      = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn    = nn.MSELoss()
        self.buffer     = ReplayBuffer(buffer_cap)
        self.learn_step = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax().item())

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        s_t  = torch.FloatTensor(s).to(self.device)
        a_t  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r_t  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns_t = torch.FloatTensor(ns).to(self.device)
        d_t  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        current_q = self.q_net(s_t).gather(1, a_t)

        with torch.no_grad():
            max_next_q = self.target_net(ns_t).max(1, keepdim=True)[0]
            target_q   = r_t + self.gamma * max_next_q * (1 - d_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)  # gradient clip
        self.optimizer.step()

        self.learn_step += 1
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"q_net": self.q_net.state_dict(), "epsilon": self.epsilon}, path)
        print(f"[✔] Model saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        print(f"[✔] Model loaded ← {path}")
