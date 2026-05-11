"""
Traffic Signal Control Environment
SDG 11: Sustainable Cities and Communities

Custom OpenAI Gym environment simulating a 4-way intersection.

State  : [q_N, q_S, q_E, q_W, wait_N, wait_S, wait_E, wait_W, phase]  → shape (9,)
Actions: 0 = NS-Green / EW-Red  |  1 = EW-Green / NS-Red
Reward : negative of current total queue length (per-step signal)
"""

import gym
from gym import spaces
import numpy as np


class TrafficSignalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    MIN_GREEN    = 5     # min timesteps before a phase switch is allowed
    MAX_GREEN    = 30    # max timesteps before a forced switch
    YELLOW       = 3     # yellow phase duration (no departures)
    ARRIVAL_RATE = 0.3   # probability a vehicle arrives per lane per timestep

    def __init__(self, max_queue=20, max_steps=300):
        super().__init__()
        self.max_queue = max_queue
        self.max_steps = max_steps

        # Observation: 9 continuous features normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self.reset()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self):
        self.queues      = np.zeros(4, dtype=np.float32)
        self.wait_times  = np.zeros(4, dtype=np.float32)
        self.phase       = 0
        self.phase_timer = 0
        self.step_count  = 0
        return self._get_obs()

    def step(self, action):
        # ── Phase transition ─────────────────────────────────────────
        switch_requested = (action != self.phase)
        can_switch       = (self.phase_timer >= self.MIN_GREEN)
        must_switch      = (self.phase_timer >= self.MAX_GREEN)

        if must_switch or (switch_requested and can_switch):
            self._simulate_yellow()
            self.phase       = action
            self.phase_timer = 0
        else:
            self.phase_timer += 1

        # ── Vehicle arrivals (stochastic) ────────────────────────────
        arrivals = np.random.binomial(1, self.ARRIVAL_RATE, size=4).astype(np.float32)
        self.queues = np.clip(self.queues + arrivals, 0, self.max_queue)

        # ── Departures (only on green lanes) ─────────────────────────
        green_lanes = [0, 1] if self.phase == 0 else [2, 3]
        for lane in green_lanes:
            departures = min(self.queues[lane], np.random.randint(1, 3))
            self.queues[lane] = max(0, self.queues[lane] - departures)

        # ── Cumulative wait (for observation context only) ────────────
        red_lanes = [2, 3] if self.phase == 0 else [0, 1]
        self.wait_times[red_lanes] += self.queues[red_lanes]

        # ── Reward: per-step queue penalty (NOT cumulative) ───────────
        # This gives the agent an immediate signal tied to its action.
        step_queue = float(np.sum(self.queues))
        total_wait = float(np.sum(self.wait_times))          # logging only
        reward     = -step_queue / (self.max_queue * 4 + 1e-8)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "queues":     self.queues.copy(),
            "wait_times": self.wait_times.copy(),
            "phase":      self.phase,
            "step_queue": step_queue,
            "total_wait": total_wait,
        }
        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        phase_str = "NS-Green / EW-Red" if self.phase == 0 else "EW-Green / NS-Red"
        print(
            f"Step {self.step_count:4d} | {phase_str} | "
            f"Q: N={self.queues[0]:.0f} S={self.queues[1]:.0f} "
            f"E={self.queues[2]:.0f} W={self.queues[3]:.0f} | "
            f"StepQueue: {np.sum(self.queues):.1f}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        return np.concatenate([
            self.queues    / self.max_queue,
            self.wait_times / (self.max_steps * self.max_queue),
            [self.phase],
        ]).astype(np.float32)

    def _simulate_yellow(self):
        """Yellow phase: arrivals continue, no departures, all lanes wait."""
        for _ in range(self.YELLOW):
            arrivals = np.random.binomial(1, self.ARRIVAL_RATE, size=4).astype(np.float32)
            self.queues     = np.clip(self.queues + arrivals, 0, self.max_queue)
            self.wait_times += self.queues
