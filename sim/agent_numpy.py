"""
DQN Agent  —  NumPy / Pure-Python version
------------------------------------------
Portable fallback for machines without PyTorch or GPU.
Implements the same interface as agent.py so scripts work with either.

Architecture : [state_dim → 128 → 128 → action_dim]
Backprop     : manual (MSE loss, He init, ReLU activations)
"""

import numpy as np
from collections import deque
import random
import pickle
import os


# ──────────────────────────────────────────────────────────────────────
# Lightweight Neural Network (NumPy)
# ──────────────────────────────────────────────────────────────────────

class NumpyNet:
    def __init__(self, dims, lr=1e-3):
        self.lr     = lr
        self.layers = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])  # He init
            b = np.zeros(dims[i+1])
            self.layers.append([W, b])

    def relu(self, x):        return np.maximum(0, x)
    def relu_grad(self, x):   return (x > 0).astype(np.float32)

    def forward(self, x):
        self.cache = []
        out = x
        for i, (W, b) in enumerate(self.layers):
            z   = out @ W + b
            self.cache.append((out, z))
            out = self.relu(z) if i < len(self.layers) - 1 else z
        return out

    def backward(self, x, targets):
        out  = self.forward(x)
        loss = float(np.mean((out - targets) ** 2))

        delta = 2 * (out - targets) / len(x)
        grads = []
        for i in reversed(range(len(self.layers))):
            inp, z = self.cache[i]
            if i < len(self.layers) - 1:
                delta *= self.relu_grad(z)
            dW = inp.T @ delta
            db = delta.sum(axis=0)
            grads.insert(0, (dW, db))
            delta = delta @ self.layers[i][0].T

        for i, (dW, db) in enumerate(grads):
            self.layers[i][0] -= self.lr * dW
            self.layers[i][1] -= self.lr * db

        return loss

    def copy_weights_from(self, other):
        for i, (W, b) in enumerate(other.layers):
            self.layers[i][0] = W.copy()
            self.layers[i][1] = b.copy()


# ──────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch      = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int32),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────
# DQN Agent (NumPy)
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
        **kwargs,          # absorb unused torch-agent kwargs (e.g. device)
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        dims           = [state_dim, 128, 128, action_dim]
        self.q_net     = NumpyNet(dims, lr=lr)
        self.target_net= NumpyNet(dims, lr=lr)
        self.target_net.copy_weights_from(self.q_net)

        self.buffer     = ReplayBuffer(buffer_cap)
        self.learn_step = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        q_vals = self.q_net.forward(state[np.newaxis])[0]
        return int(np.argmax(q_vals))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        next_q        = self.target_net.forward(next_states)
        max_next_q    = next_q.max(axis=1)
        target_q_full = self.q_net.forward(states).copy()

        for i in range(self.batch_size):
            target_q_full[i, actions[i]] = (
                rewards[i] + self.gamma * max_next_q[i] * (1 - dones[i])
            )

        loss = self.q_net.backward(states, target_q_full)
        self.learn_step += 1
        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def sync_target(self):
        self.target_net.copy_weights_from(self.q_net)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"layers": self.q_net.layers, "epsilon": self.epsilon}, f)
        print(f"[✔] Model saved → {path}")

    def load(self, path):
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.q_net.layers = ckpt["layers"]
        self.target_net.copy_weights_from(self.q_net)
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        print(f"[✔] Model loaded ← {path}")
