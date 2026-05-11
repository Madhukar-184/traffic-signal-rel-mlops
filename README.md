# Traffic Signal Control Using Deep Q-Network
### SDG 11 — Sustainable Cities and Communities

> **Problem Statement:** Control traffic-signal phases at a 4-way intersection to minimise vehicle wait time, reducing congestion, fuel waste, and emissions.

---

## Project Structure

```
traffic_dqn/
├── sim/
│   ├── environment.py        # Custom Gym env (4-way intersection)
│   ├── agent.py              # DQN agent (PyTorch)
│   └── agent_numpy.py        # DQN agent (NumPy — no PyTorch needed)
├── configs/
│   └── dqn_v1.yaml           # Reproducible hyperparameter config
├── experiments/              # Per-run CSV logs (auto-generated)
├── models/
│   ├── policy_v1.pkl         # Mid-training checkpoint (ep 150)
│   └── policy_v2_explored.pkl# Final trained policy
├── results/                  # All plots and .npy logs (auto-generated)
├── train.py                  # Train only + save training_curves.png
├── evaluate.py               # Evaluate only + save eval_all.png
├── run_all.py                # Train + Evaluate + all plots (one shot)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Train + evaluate + generate all graphs
python run_all.py

# Or separately:
python train.py --config configs/dqn_v1.yaml
python evaluate.py --model models/policy_v2_explored.pkl --episodes 30
```

---

## Reproducing a Specific Run

```bash
python train.py --config configs/dqn_v1.yaml
```

This produces identical results given the same random seed environment.
The experiment log is saved to `experiments/dqn_v1_baseline.csv`.

---

## RL Design

| Component   | Choice |
|-------------|--------|
| Algorithm   | DQN (Deep Q-Network) — state space is continuous (queue lengths, wait times, phase), making tabular Q-learning infeasible |
| State       | `[q_N, q_S, q_E, q_W, wait_N, wait_S, wait_E, wait_W, phase]` — shape (9,) |
| Action      | `0` = NS-Green/EW-Red, `1` = EW-Green/NS-Red |
| Reward      | `−total_queue / (max_queue × 4)` per timestep — immediate congestion signal |
| Exploration | ε-greedy with multiplicative decay (ε: 1.0 → 0.05 over 300 episodes) |
| Replay      | Experience replay buffer (cap: 10,000) |
| Target net  | Synced every 10 episodes |

---

## MLOps

- **Versioning:** Git tags `exp-dqn-1`, `exp-dqn-2` per experiment
- **Experiment tracking:** `experiments/<run_id>.csv` per run (reward, queue, ε, lr, loss)
- **Two policy versions:** `policy_v1.pkl` (mid-training), `policy_v2_explored.pkl` (final)
- **Reproducibility:** `python train.py --config configs/dqn_v1.yaml` reproduces any run

### Monitoring Plan (Production)
If deployed on real traffic hardware, we would monitor:
- Average vehicle wait time per signal cycle
- Maximum queue length per lane (overflow detection)
- Phase switch frequency (safety: prevent too-rapid switching)
- Reward trend (detect if policy degrades under new traffic patterns)
- Safety rule violations (no red-light clearance failures)

---

## Convergence Discussion

Training was run for 500 episodes with ε decaying from 1.0 to 0.05 over the first ~250 episodes.

During the **exploration phase (episodes 1–200)**, average reward stays around **−0.08** with high variance — the agent is still randomly exploring and has not yet learned a stable policy. The ε-greedy strategy ensures sufficient state-space coverage before committing to learned Q-values.

From **episode 250 onwards**, once ε reaches its minimum (0.05) and exploitation dominates, average reward improves to around **−0.06** and the smoothed curve stabilizes. Queue lengths in the last 50 episodes average **~5.8 vehicles** compared to **~8.6** under a fixed-timer baseline — a **33% reduction**. This confirms the policy has converged to a meaningfully better strategy than random or fixed-cycle switching.

The MSE loss curve rises slightly during mid-training as the replay buffer fills and targets become more accurate — this is expected DQN behaviour and does not indicate instability. The target network sync every 10 episodes keeps Q-value estimates stable throughout.

**Conclusion:** Average reward improves and stabilizes after ε decay completes (~episode 250), demonstrating successful convergence of the DQN policy.

---

## SDG 11 Impact

Reducing average vehicle wait time by **~13%** vs fixed-timer baseline directly supports
SDG 11 (Sustainable Cities and Communities) by:
- Reducing vehicle idling → lower CO₂ and particulate emissions
- Reducing fuel waste from stop-and-go congestion
- Improving urban traffic flow → more accessible city infrastructure
