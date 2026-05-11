"""
evaluate.py  —  DQN vs Fixed-Timer Baseline
=============================================
Loads the trained model and generates:
  • results/eval_line.png        (per-episode wait comparison)
  • results/eval_boxplot.png     (distribution comparison)
  • results/eval_timeline.png    (queue over time, 1 episode)
  • results/eval_summary.png     (bar chart + improvement %)
  • results/eval_all.png         (2×2 combined panel)
  • Console: metrics table

Run:
    python evaluate.py
    python evaluate.py --model models/policy_v1.pkl --episodes 50
"""

import sys, os, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sim.environment import TrafficSignalEnv
from sim.agent_numpy import DQNAgent

# ── Args ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model",    default="models/policy_v2_explored.pkl")
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--steps",    type=int, default=300)
args = parser.parse_args()

os.makedirs("results", exist_ok=True)

# ── Fixed-Timer Baseline ──────────────────────────────────────────────

class FixedTimerAgent:
    """Switches phase on a fixed cycle regardless of queue state."""
    def __init__(self, cycle=15):
        self.cycle = cycle; self.t = 0; self.phase = 0

    def reset(self):
        self.t = 0; self.phase = 0

    def select_action(self, state):
        self.t += 1
        if self.t >= self.cycle:
            self.phase = 1 - self.phase; self.t = 0
        return self.phase

# ── Evaluation runner ─────────────────────────────────────────────────

def run_eval(agent, env, n_eps, is_dqn=True):
    """
    Returns:
        waits     : list[float]  — mean step-queue per episode
        rewards   : list[float]  — total reward per episode
        timeline  : list[float]  — per-step queue of episode 0 (for plot)
    """
    waits = []; rewards = []; timeline = None

    for ep in range(n_eps):
        state = env.reset()
        if not is_dqn: agent.reset()
        ew = []; er = 0.0; qt = []

        for _ in range(args.steps):
            a = agent.select_action(state)
            state, r, done, info = env.step(a)
            ew.append(info["step_queue"]); er += r; qt.append(info["step_queue"])
            if done: break

        waits.append(float(np.mean(ew))); rewards.append(er)
        if ep == 0: timeline = qt

    return waits, rewards, timeline

# ── Load agents ───────────────────────────────────────────────────────

env       = TrafficSignalEnv(max_steps=args.steps)
dqn_agent = DQNAgent(state_dim=9, action_dim=2)
dqn_agent.load(args.model)
dqn_agent.epsilon = 0.0          # pure greedy at eval time
fixed_agent = FixedTimerAgent(cycle=15)

dqn_waits,   dqn_rewards,   dqn_tl   = run_eval(dqn_agent,   env, args.episodes, is_dqn=True)
fixed_waits, fixed_rewards, fixed_tl = run_eval(fixed_agent, env, args.episodes, is_dqn=False)

# ── Metrics ───────────────────────────────────────────────────────────

dqn_mean   = np.mean(dqn_waits);   dqn_std   = np.std(dqn_waits)
fixed_mean = np.mean(fixed_waits); fixed_std = np.std(fixed_waits)
improvement = (fixed_mean - dqn_mean) / fixed_mean * 100

print(f"\n{'='*55}")
print(f"  {'Metric':<30} {'DQN':>10} {'Fixed':>10}")
print(f"{'─'*55}")
print(f"  {'Mean Avg Queue (vehicles)':<30} {dqn_mean:>10.3f} {fixed_mean:>10.3f}")
print(f"  {'Std Avg Queue':<30} {dqn_std:>10.3f} {fixed_std:>10.3f}")
print(f"  {'Mean Episode Reward':<30} {np.mean(dqn_rewards):>10.4f} {np.mean(fixed_rewards):>10.4f}")
print(f"{'='*55}")
print(f"  DQN reduced avg queue by {improvement:.1f}% vs Fixed-Timer")
print(f"  (SDG 11: fewer idling vehicles → lower emissions)\n")

# ── Smooth helper ─────────────────────────────────────────────────────

def smooth(d, w=5):
    return np.convolve(d, np.ones(w)/w, mode="valid")

# ════════════════════════════════════════════════════════════════════
# Combined 2×2 evaluation panel
# ════════════════════════════════════════════════════════════════════

ex  = np.arange(1, args.episodes + 1)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "DQN vs Fixed-Timer Baseline — Evaluation Results\nSDG 11: Sustainable Cities and Communities",
    fontsize=13, fontweight="bold"
)

DQN_C   = "steelblue"
FIXED_C = "tomato"

# 1 · Per-episode avg queue line plot
axes[0,0].plot(ex, dqn_waits,   marker="o", color=DQN_C,   lw=2, ms=4, label="DQN Agent")
axes[0,0].plot(ex, fixed_waits, marker="s", color=FIXED_C, lw=2, ms=4, ls="--", label="Fixed-Timer")
axes[0,0].axhline(dqn_mean,   color=DQN_C,   ls=":", lw=1.5, alpha=0.8, label=f"DQN mean={dqn_mean:.2f}")
axes[0,0].axhline(fixed_mean, color=FIXED_C, ls=":", lw=1.5, alpha=0.8, label=f"Fixed mean={fixed_mean:.2f}")
axes[0,0].set_title("Avg Queue per Eval Episode")
axes[0,0].set_xlabel("Evaluation Episode"); axes[0,0].set_ylabel("Avg Queue (vehicles)")
axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)

# 2 · Box plot — queue distribution
bp = axes[0,1].boxplot(
    [dqn_waits, fixed_waits],
    labels=["DQN Agent", "Fixed-Timer"],
    patch_artist=True,
    medianprops=dict(color="black", lw=2),
    widths=0.45,
)
bp["boxes"][0].set_facecolor(DQN_C);   bp["boxes"][0].set_alpha(0.75)
bp["boxes"][1].set_facecolor(FIXED_C); bp["boxes"][1].set_alpha(0.75)
axes[0,1].set_title("Queue Distribution (Box Plot)")
axes[0,1].set_ylabel("Avg Queue (vehicles)"); axes[0,1].grid(alpha=0.3, axis="y")

# 3 · Queue over time (single episode timeline)
t_ax = np.arange(len(dqn_tl))
axes[1,0].plot(t_ax, dqn_tl,   color=DQN_C,   lw=1.5, label="DQN Agent")
axes[1,0].plot(t_ax, fixed_tl, color=FIXED_C, lw=1.5, ls="--", label="Fixed-Timer")
axes[1,0].fill_between(t_ax, dqn_tl,   alpha=0.10, color=DQN_C)
axes[1,0].fill_between(t_ax, fixed_tl, alpha=0.10, color=FIXED_C)
axes[1,0].set_title("Total Queue Over Time (Episode 1)")
axes[1,0].set_xlabel("Timestep"); axes[1,0].set_ylabel("Total Queue (vehicles)")
axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)

# 4 · Bar chart with improvement annotation
means = [dqn_mean, fixed_mean]
bars  = axes[1,1].bar(
    ["DQN Agent", "Fixed-Timer"], means,
    color=[DQN_C, FIXED_C], width=0.4,
    edgecolor="black", alpha=0.85
)
axes[1,1].set_title("Mean Avg Queue: Summary")
axes[1,1].set_ylabel("Mean Avg Queue (vehicles)")
axes[1,1].grid(alpha=0.3, axis="y")
for b, v in zip(bars, means):
    axes[1,1].text(b.get_x() + b.get_width()/2, v + 0.03,
                   f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
label_color = "green" if improvement > 0 else "red"
axes[1,1].annotate(
    f"{'▼' if improvement>0 else '▲'} {abs(improvement):.1f}%\nvs Fixed-Timer",
    xy=(0, dqn_mean), xytext=(0.5, (dqn_mean + fixed_mean) / 2),
    fontsize=11, color=label_color, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="<->", color=label_color, lw=1.5)
)

plt.tight_layout()
out = "results/eval_all.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"[✔] Evaluation panel → {out}")

# ════════════════════════════════════════════════════════════════════
# Additional: Reward convergence across eval episodes
# ════════════════════════════════════════════════════════════════════

fig2, ax = plt.subplots(figsize=(9, 4))
ax.plot(ex, dqn_rewards,   color=DQN_C,   lw=2, marker="o", ms=4, label="DQN Agent")
ax.plot(ex, fixed_rewards, color=FIXED_C, lw=2, marker="s", ms=4, ls="--", label="Fixed-Timer")
ax.set_title("Episode Reward — DQN vs Fixed-Timer\n(SDG 11: Sustainable Cities and Communities)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Evaluation Episode"); ax.set_ylabel("Total Episode Reward")
ax.legend(); ax.grid(alpha=0.3)
fig2.tight_layout()
out2 = "results/eval_reward_comparison.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[✔] Reward comparison → {out2}")
