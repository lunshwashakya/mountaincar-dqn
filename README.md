# mountaincar-dqn
How I Built a DQN Agent to Solve MountainCar-v0

A reinforcement learning project implementing a Deep Q-Network agent to solve the MountainCar-v0 environment from [Gymnasium](https://gymnasium.farama.org/).

## The Problem

A small underpowered car sits in a valley between two hills. The car's engine isn't strong enough to drive straight up — it has to rock back and forth, building momentum, to reach the flag at the top of the right hill.

The reward is sparse: **-1 per timestep**, with no signal until the flag is reached. This makes it deceptively hard for standard RL approaches.

| State | Range |
|-------|-------|
| Position | -1.2 to 0.6 |
| Velocity | -0.07 to 0.07 |

**Actions:** Push left (0), No push (1), Push right (2)

---

## Approach

We implemented a standard DQN with:
- **Q-Network** — 2-layer MLP (64 units, ReLU) mapping state → Q-values per action
- **Experience Replay** — 50,000 transition buffer, batch size 64
- **Target Network** — synced every 10 learning steps for stable Bellman targets
- **Epsilon-greedy exploration** — decays from 1.0 → 0.02 over training

No reward shaping was used. The agent learns purely from the sparse environment reward.

---

## Results

Trained for 300 episodes, evaluated greedily over 100 test episodes:

| Policy | Avg Steps to Goal | Success Rate |
|--------|-------------------|--------------|
| Random | 200.0 | 0% |
| DQN (trained) | 141.6 | 93% |

The agent went from never reaching the flag (random policy) to solving 93 out of 100 test episodes, cutting average episode length by ~30%.

---

## Project Structure

```
├── MountainCarDQN.ipynb     # Main notebook 
└── README.md
```

**Notebook cells:**
1. Environment & dependency setup
2. Q-Network architecture
3. Replay buffer
4. DQN agent (action selection, target updates, training step)
5. Training loop + evaluation
6. Results & visualisation

---

## Getting Started

### Run in Google Colab (recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. File → Upload notebook → select `MDS616_G9.ipynb`
3. Add a new cell at the top and run:
```python
!pip install gymnasium[classic-control] -q
```
4. Runtime → Run all

### Run locally
```bash
pip install gymnasium[classic-control] torch numpy matplotlib
jupyter notebook MDS616_G9.ipynb
```

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 (Adam) |
| Discount factor γ | 0.99 |
| Replay buffer size | 50,000 |
| Batch size | 64 |
| Epsilon start/end | 1.0 / 0.02 |
| Epsilon decay | 0.97 per episode |
| Target update freq | Every 10 steps |
| Training episodes | 300 |

---
## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- [Gymnasium Documentation](https://gymnasium.farama.org/)
