"""
run_all.py  —  One-shot: Train + Evaluate + All Plots
======================================================
Runs training then evaluation and saves every graph.

Output files in results/:
    training_curves.png          — 4-panel training curves
    eval_all.png                 — 4-panel DQN vs Fixed-Timer
    eval_reward_comparison.png   — reward line plot
    ep_rewards.npy / ep_queues.npy / ep_losses.npy / ep_eps.npy
    run_<timestamp>.csv          — full experiment log

Run:
    python run_all.py
"""

import sys, os, warnings, csv, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.environment import TrafficSignalEnv
from sim.agent_numpy import DQNAgent

# ── Dirs ──────────────────────────────────────────────────────────────
for d in ("models", "results", "experiments"): os.makedirs(d, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────
EPISODES      = 500
MAX_STEPS     = 300
LR            = 1e-3
GAMMA         = 0.99
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 0.988
BATCH         = 64
TARGET_UPDATE = 10
BUFFER_CAP    = 10_000
EVAL_EPS      = 30
RUN_ID        = f"run_{int(time.time())}"

print(f"\n{'='*60}")
print(f"  Traffic Signal DQN  —  SDG 11")
print(f"  Run ID : {RUN_ID}")
print(f"{'='*60}\n")

# ════════════════════════════════════════════════════════════════════
# 1. TRAINING
# ════════════════════════════════════════════════════════════════════

env   = TrafficSignalEnv(max_steps=MAX_STEPS)
agent = DQNAgent(
    state_dim=9, action_dim=2, lr=LR, gamma=GAMMA,
    epsilon_start=EPS_START, epsilon_end=EPS_END,
    epsilon_decay=EPS_DECAY, batch_size=BATCH,
    target_update=TARGET_UPDATE, buffer_cap=BUFFER_CAP,
)

ep_rewards, ep_queues, ep_eps, ep_losses = [], [], [], []

csv_path = f"experiments/{RUN_ID}.csv"
csv_file = open(csv_path, "w", newline="")
writer   = csv.writer(csv_file)
writer.writerow(["run_id","episode","avg_reward","avg_queue",
                 "epsilon","lr","gamma","loss"])

for ep in range(1, EPISODES + 1):
    state = env.reset(); er = 0.0; eq = []; el = []

    for _ in range(MAX_STEPS):
        a = agent.select_action(state)
        ns, r, done, info = env.step(a)
        agent.buffer.push(state, a, r, ns, float(done))
        loss = agent.learn()
        if loss is not None: el.append(loss)
        er += r; eq.append(info["step_queue"]); state = ns
        if done: break

    agent.decay_epsilon()
    if ep % TARGET_UPDATE == 0: agent.sync_target()

    avg_r = er / MAX_STEPS
    avg_q = float(np.mean(eq))
    avg_l = float(np.mean(el)) if el else 0.0
    ep_rewards.append(avg_r); ep_queues.append(avg_q)
    ep_eps.append(agent.epsilon); ep_losses.append(avg_l)
    writer.writerow([RUN_ID, ep, f"{avg_r:.6f}", f"{avg_q:.4f}",
                     f"{agent.epsilon:.4f}", LR, GAMMA, f"{avg_l:.6f}"])

    # Mid checkpoint
    if ep == EPISODES // 2:
        agent.save("models/policy_v1.pkl")
        print(f"  [✔] Checkpoint saved → models/policy_v1.pkl")

    if ep % 50 == 0 or ep == 1:
        print(f"  Ep {ep:3d}/{EPISODES} | AvgReward={avg_r:.4f} | "
              f"AvgQueue={avg_q:.2f} | ε={agent.epsilon:.3f} | Loss={avg_l:.5f}")

csv_file.close()
agent.save("models/policy_v2_explored.pkl")

np.save("results/ep_rewards.npy", np.array(ep_rewards))
np.save("results/ep_queues.npy",  np.array(ep_queues))
np.save("results/ep_losses.npy",  np.array(ep_losses))
np.save("results/ep_eps.npy",     np.array(ep_eps))
print(f"\n  [✔] Training done | log → {csv_path}")

# ════════════════════════════════════════════════════════════════════
# 2. EVALUATION
# ════════════════════════════════════════════════════════════════════

class FixedTimerAgent:
    def __init__(self, cycle=15): self.cycle=cycle; self.t=0; self.phase=0
    def reset(self): self.t=0; self.phase=0
    def select_action(self, s):
        self.t += 1
        if self.t >= self.cycle: self.phase = 1 - self.phase; self.t = 0
        return self.phase

def run_eval(agent, env, n, is_dqn=True):
    waits=[]; rewards=[]; timeline=None
    for ep in range(n):
        s = env.reset()
        if not is_dqn: agent.reset()
        ew=[]; er=0.0; qt=[]
        for _ in range(MAX_STEPS):
            a = agent.select_action(s)
            s, r, done, info = env.step(a)
            ew.append(info["step_queue"]); er += r; qt.append(info["step_queue"])
            if done: break
        waits.append(float(np.mean(ew))); rewards.append(er)
        if ep == 0: timeline = qt
    return waits, rewards, timeline

agent.epsilon = 0.0   # greedy evaluation
dqn_w, dqn_r, dqn_tl     = run_eval(agent,              env, EVAL_EPS, is_dqn=True)
fixed_w, fixed_r, fixed_tl= run_eval(FixedTimerAgent(15),env, EVAL_EPS, is_dqn=False)

dqn_mean   = np.mean(dqn_w);   fixed_mean = np.mean(fixed_w)
improvement= (fixed_mean - dqn_mean) / fixed_mean * 100

print(f"\n{'='*55}")
print(f"  {'Metric':<32} {'DQN':>9} {'Fixed':>9}")
print(f"{'─'*55}")
print(f"  {'Mean Avg Queue (vehicles)':<32} {dqn_mean:>9.3f} {fixed_mean:>9.3f}")
print(f"  {'Std Avg Queue':<32} {np.std(dqn_w):>9.3f} {np.std(fixed_w):>9.3f}")
print(f"  {'Mean Episode Reward':<32} {np.mean(dqn_r):>9.4f} {np.mean(fixed_r):>9.4f}")
print(f"{'='*55}")
print(f"  DQN reduced avg queue by {improvement:.1f}% vs Fixed-Timer")
print(f"  SDG 11 impact: fewer idling vehicles → lower emissions\n")

# ════════════════════════════════════════════════════════════════════
# 3. PLOTS
# ════════════════════════════════════════════════════════════════════

DQN_C   = "steelblue"
FIXED_C = "tomato"

def smooth(d, w=20):
    return np.convolve(d, np.ones(w)/w, mode="valid")

x  = np.arange(1, EPISODES + 1)
xs = np.arange(len(smooth(ep_rewards))) + 1
ex = np.arange(1, EVAL_EPS + 1)

# ── Figure 1: Training Curves (2×2) ──────────────────────────────────

fig1, axes = plt.subplots(2, 2, figsize=(14, 8))
fig1.suptitle(
    "DQN Traffic Signal Control — Training Curves\nSDG 11: Sustainable Cities and Communities",
    fontsize=13, fontweight="bold"
)

axes[0,0].plot(x, ep_rewards, alpha=0.2, color=DQN_C)
axes[0,0].plot(xs, smooth(ep_rewards), color=DQN_C, lw=2, label="MA-20")
axes[0,0].axhline(np.mean(ep_rewards[-50:]), color="navy", ls=":",
                  label=f"Last-50 avg: {np.mean(ep_rewards[-50:]):.4f}")
axes[0,0].set_title("Avg Reward per Episode")
axes[0,0].set_xlabel("Episode"); axes[0,0].set_ylabel("Avg Reward")
axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(x, ep_queues, alpha=0.2, color=FIXED_C)
axes[0,1].plot(xs, smooth(ep_queues), color=FIXED_C, lw=2, label="MA-20")
axes[0,1].axhline(np.mean(ep_queues[-50:]), color="darkred", ls=":",
                  label=f"Last-50 avg: {np.mean(ep_queues[-50:]):.2f}")
axes[0,1].set_title("Avg Queue Length per Episode")
axes[0,1].set_xlabel("Episode"); axes[0,1].set_ylabel("Avg Queue (vehicles)")
axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.3)

axes[1,0].plot(x, ep_losses, alpha=0.2, color="mediumseagreen")
axes[1,0].plot(xs, smooth(ep_losses), color="mediumseagreen", lw=2, label="MA-20")
axes[1,0].set_title("Training Loss (MSE)")
axes[1,0].set_xlabel("Episode"); axes[1,0].set_ylabel("Loss")
axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)

axes[1,1].plot(x, ep_eps, color="darkorange", lw=2)
axes[1,1].fill_between(x, ep_eps, alpha=0.15, color="darkorange")
axes[1,1].set_title("Epsilon Decay (Exploration → Exploitation)")
axes[1,1].set_xlabel("Episode"); axes[1,1].set_ylabel("ε")
axes[1,1].annotate(f"Final ε = {ep_eps[-1]:.3f}",
    xy=(EPISODES, ep_eps[-1]), xytext=(EPISODES*0.6, 0.35),
    fontsize=9, arrowprops=dict(arrowstyle="->", color="darkorange"))
axes[1,1].grid(alpha=0.3)

fig1.tight_layout()
plt.savefig("results/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✔] results/training_curves.png")

# ── Figure 2: Evaluation 2×2 panel ───────────────────────────────────

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
fig2.suptitle(
    "DQN vs Fixed-Timer Baseline — Evaluation\nSDG 11: Sustainable Cities and Communities",
    fontsize=13, fontweight="bold"
)

axes2[0,0].plot(ex, dqn_w,   marker="o", color=DQN_C,   lw=2, ms=4, label="DQN Agent")
axes2[0,0].plot(ex, fixed_w, marker="s", color=FIXED_C, lw=2, ms=4, ls="--", label="Fixed-Timer")
axes2[0,0].axhline(dqn_mean,   color=DQN_C,   ls=":", lw=1.5, alpha=0.8)
axes2[0,0].axhline(fixed_mean, color=FIXED_C, ls=":", lw=1.5, alpha=0.8)
axes2[0,0].set_title("Avg Queue per Eval Episode")
axes2[0,0].set_xlabel("Evaluation Episode"); axes2[0,0].set_ylabel("Avg Queue (vehicles)")
axes2[0,0].legend(fontsize=9); axes2[0,0].grid(alpha=0.3)

bp = axes2[0,1].boxplot([dqn_w, fixed_w], labels=["DQN Agent","Fixed-Timer"],
    patch_artist=True, medianprops=dict(color="black", lw=2), widths=0.45)
bp["boxes"][0].set_facecolor(DQN_C);   bp["boxes"][0].set_alpha(0.75)
bp["boxes"][1].set_facecolor(FIXED_C); bp["boxes"][1].set_alpha(0.75)
axes2[0,1].set_title("Queue Distribution (Box Plot)")
axes2[0,1].set_ylabel("Avg Queue (vehicles)"); axes2[0,1].grid(alpha=0.3, axis="y")

t_ax = np.arange(len(dqn_tl))
axes2[1,0].plot(t_ax, dqn_tl,   color=DQN_C,   lw=1.5, label="DQN Agent")
axes2[1,0].plot(t_ax, fixed_tl, color=FIXED_C, lw=1.5, ls="--", label="Fixed-Timer")
axes2[1,0].fill_between(t_ax, dqn_tl,   alpha=0.1, color=DQN_C)
axes2[1,0].fill_between(t_ax, fixed_tl, alpha=0.1, color=FIXED_C)
axes2[1,0].set_title("Queue Over Time — Episode 1")
axes2[1,0].set_xlabel("Timestep"); axes2[1,0].set_ylabel("Total Queue (vehicles)")
axes2[1,0].legend(fontsize=9); axes2[1,0].grid(alpha=0.3)

means = [dqn_mean, fixed_mean]
bars  = axes2[1,1].bar(["DQN Agent","Fixed-Timer"], means,
    color=[DQN_C, FIXED_C], width=0.4, edgecolor="black", alpha=0.85)
axes2[1,1].set_title("Mean Avg Queue: DQN vs Fixed-Timer")
axes2[1,1].set_ylabel("Mean Avg Queue (vehicles)")
axes2[1,1].grid(alpha=0.3, axis="y")
for b, v in zip(bars, means):
    axes2[1,1].text(b.get_x()+b.get_width()/2, v+0.02,
                    f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
col = "green" if improvement > 0 else "red"
axes2[1,1].annotate(
    f"{'▼' if improvement>0 else '▲'} {abs(improvement):.1f}%\nvs Fixed-Timer",
    xy=(0, dqn_mean), xytext=(0.5, (dqn_mean+fixed_mean)/2),
    fontsize=11, color=col, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="<->", color=col, lw=1.5)
)

fig2.tight_layout()
plt.savefig("results/eval_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✔] results/eval_all.png")

# ── Figure 3: Reward comparison ───────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(ex, dqn_r,   color=DQN_C,   lw=2, marker="o", ms=4, label="DQN Agent")
ax3.plot(ex, fixed_r, color=FIXED_C, lw=2, marker="s", ms=4, ls="--", label="Fixed-Timer")
ax3.axhline(np.mean(dqn_r),   color=DQN_C,   ls=":", lw=1.5)
ax3.axhline(np.mean(fixed_r), color=FIXED_C, ls=":", lw=1.5)
ax3.set_title("Episode Reward — DQN vs Fixed-Timer\nSDG 11: Sustainable Cities and Communities",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Evaluation Episode"); ax3.set_ylabel("Total Episode Reward")
ax3.legend(); ax3.grid(alpha=0.3)
fig3.tight_layout()
plt.savefig("results/eval_reward_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✔] results/eval_reward_comparison.png")

print(f"\n{'='*60}")
print(f"  All outputs saved to results/")
print(f"  Experiment log: {csv_path}")
print(f"{'='*60}\n")
