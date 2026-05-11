"""
train.py  —  DQN Traffic Signal Control
========================================
Trains the DQN agent and saves:
  • models/policy_v1.pkl          (checkpoint at episode 150)
  • models/policy_v2_explored.pkl (final model)
  • results/run_<id>.csv          (experiment log)
  • results/training_curves.png   (4-panel training plot)

Run:
    python train.py
    python train.py --config configs/dqn_v1.yaml
"""

import sys, os, warnings, argparse, csv, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.environment  import TrafficSignalEnv
from sim.agent_numpy  import DQNAgent

# ── Config ────────────────────────────────────────────────────────────

def load_config(path=None):
    """Load YAML config or return defaults."""
    defaults = dict(
        episodes      = 300,
        max_steps     = 300,
        lr            = 1e-3,
        gamma         = 0.99,
        epsilon_start = 1.0,
        epsilon_end   = 0.05,
        epsilon_decay = 0.988,
        batch_size    = 64,
        target_update = 10,
        buffer_cap    = 10_000,
        save_dir      = "models",
        results_dir   = "results",
        run_id        = f"run_{int(time.time())}",
    )
    if path:
        import yaml
        with open(path) as f:
            defaults.update(yaml.safe_load(f))
    return defaults

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
args   = parser.parse_args()
cfg    = load_config(args.config)

os.makedirs(cfg["save_dir"],   exist_ok=True)
os.makedirs(cfg["results_dir"],exist_ok=True)

# ── Initialise ────────────────────────────────────────────────────────

env   = TrafficSignalEnv(max_steps=cfg["max_steps"])
agent = DQNAgent(
    state_dim     = env.observation_space.shape[0],
    action_dim    = env.action_space.n,
    lr            = cfg["lr"],
    gamma         = cfg["gamma"],
    epsilon_start = cfg["epsilon_start"],
    epsilon_end   = cfg["epsilon_end"],
    epsilon_decay = cfg["epsilon_decay"],
    batch_size    = cfg["batch_size"],
    target_update = cfg["target_update"],
    buffer_cap    = cfg["buffer_cap"],
)

# ── CSV log ───────────────────────────────────────────────────────────

csv_path = os.path.join(cfg["results_dir"], f"{cfg['run_id']}.csv")
csv_file = open(csv_path, "w", newline="")
writer   = csv.writer(csv_file)
writer.writerow(["run_id","episode","avg_reward","avg_queue",
                 "epsilon","lr","gamma","loss"])

# ── Training loop ─────────────────────────────────────────────────────

ep_rewards, ep_queues, ep_eps, ep_losses = [], [], [], []
EPISODES = cfg["episodes"]
MID      = EPISODES // 2

print(f"\n{'='*60}")
print(f"  Run ID : {cfg['run_id']}")
print(f"  Config : {args.config or 'defaults'}")
print(f"{'='*60}\n")

for ep in range(1, EPISODES + 1):
    state = env.reset()
    er = 0.0; eq = []; el = []

    for _ in range(cfg["max_steps"]):
        a               = agent.select_action(state)
        ns, r, done, info = env.step(a)
        agent.buffer.push(state, a, r, ns, float(done))
        loss = agent.learn()
        if loss is not None: el.append(loss)
        er += r; eq.append(info["step_queue"]); state = ns
        if done: break

    agent.decay_epsilon()
    if ep % cfg["target_update"] == 0:
        agent.sync_target()

    avg_r = er / cfg["max_steps"]
    avg_q = float(np.mean(eq))
    avg_l = float(np.mean(el)) if el else 0.0
    ep_rewards.append(avg_r); ep_queues.append(avg_q)
    ep_eps.append(agent.epsilon); ep_losses.append(avg_l)

    writer.writerow([cfg["run_id"], ep, f"{avg_r:.6f}", f"{avg_q:.4f}",
                     f"{agent.epsilon:.4f}", cfg["lr"], cfg["gamma"], f"{avg_l:.6f}"])

    if ep % 50 == 0 or ep == 1:
        print(f"  Ep {ep:3d}/{EPISODES} | AvgReward={avg_r:.4f} | "
              f"AvgQueue={avg_q:.2f} | ε={agent.epsilon:.3f} | Loss={avg_l:.5f}")

    # Save mid-training checkpoint (policy_v1)
    if ep == MID:
        agent.save(os.path.join(cfg["save_dir"], "policy_v1.pkl"))

csv_file.close()

# Save final model (policy_v2)
agent.save(os.path.join(cfg["save_dir"], "policy_v2_explored.pkl"))

# Save raw arrays
np.save(os.path.join(cfg["results_dir"], "ep_rewards.npy"), np.array(ep_rewards))
np.save(os.path.join(cfg["results_dir"], "ep_queues.npy"),  np.array(ep_queues))
np.save(os.path.join(cfg["results_dir"], "ep_losses.npy"),  np.array(ep_losses))
np.save(os.path.join(cfg["results_dir"], "ep_eps.npy"),     np.array(ep_eps))

print(f"\n[✔] Training complete | logs → {csv_path}\n")

# ── Training Curves ───────────────────────────────────────────────────

def smooth(data, w=20):
    return np.convolve(data, np.ones(w)/w, mode="valid")

x  = np.arange(1, EPISODES + 1)
xs = np.arange(len(smooth(ep_rewards))) + 1

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(
    "DQN Traffic Signal Control — Training Curves\nSDG 11: Sustainable Cities and Communities",
    fontsize=13, fontweight="bold"
)

# 1 · Episode Reward
axes[0,0].plot(x, ep_rewards, alpha=0.2, color="steelblue")
axes[0,0].plot(xs, smooth(ep_rewards), color="steelblue", lw=2, label="MA-20")
axes[0,0].axhline(np.mean(ep_rewards[-50:]), color="navy", ls=":",
                  label=f"Last-50 avg: {np.mean(ep_rewards[-50:]):.4f}")
axes[0,0].set_title("Avg Reward per Episode")
axes[0,0].set_xlabel("Episode"); axes[0,0].set_ylabel("Avg Reward")
axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)

# 2 · Avg Queue Length
axes[0,1].plot(x, ep_queues, alpha=0.2, color="tomato")
axes[0,1].plot(xs, smooth(ep_queues), color="tomato", lw=2, label="MA-20")
axes[0,1].axhline(np.mean(ep_queues[-50:]), color="darkred", ls=":",
                  label=f"Last-50 avg: {np.mean(ep_queues[-50:]):.2f}")
axes[0,1].set_title("Avg Queue Length per Episode")
axes[0,1].set_xlabel("Episode"); axes[0,1].set_ylabel("Avg Queue (vehicles)")
axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.3)

# 3 · Training Loss
axes[1,0].plot(x, ep_losses, alpha=0.2, color="mediumseagreen")
axes[1,0].plot(xs, smooth(ep_losses), color="mediumseagreen", lw=2, label="MA-20")
axes[1,0].set_title("Training Loss (MSE)")
axes[1,0].set_xlabel("Episode"); axes[1,0].set_ylabel("Loss")
axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)

# 4 · Epsilon Decay
axes[1,1].plot(x, ep_eps, color="darkorange", lw=2)
axes[1,1].fill_between(x, ep_eps, alpha=0.15, color="darkorange")
axes[1,1].set_title("Epsilon Decay (Exploration → Exploitation)")
axes[1,1].set_xlabel("Episode"); axes[1,1].set_ylabel("ε")
axes[1,1].annotate(
    f"Final ε = {ep_eps[-1]:.3f}",
    xy=(EPISODES, ep_eps[-1]),
    xytext=(EPISODES * 0.65, 0.35),
    fontsize=9, arrowprops=dict(arrowstyle="->", color="darkorange")
)
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(cfg["results_dir"], "training_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"[✔] Training curves → {out}")
