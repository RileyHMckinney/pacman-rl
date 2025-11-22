# phase_1/train.py

import os
import csv
import torch

from environment import PacmanEnvironment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

# -----------------------------
# CONFIG
# -----------------------------
EPISODES = 5000
BATCH_SIZE = 256
UPDATE_REPEAT = 4
SAVE_INTERVAL = 2000

LOG_DIR = "logs"
MODEL_DIR = "models"
LOG_PATH = os.path.join(LOG_DIR, "metrics.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Logging setup
# -----------------------------
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "pacman_reward",
            "ghost_reward",
            "steps",
            "result",
            "epsilon_pacman",
            "epsilon_ghost",
            "loss_pacman",
            "loss_ghost"
        ])


# -----------------------------
# Helper: load latest checkpoint
# -----------------------------
def load_latest_checkpoint(pac_agent: DQNAgent, ghost_agent: DQNAgent) -> int:
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith("pacman_ep") and f.endswith(".pt")]
    if not files:
        print("No checkpoint found. Starting fresh from episode 0.")
        return 0

    episodes = []
    for f in files:
        try:
            ep_str = f.split("pacman_ep")[1].split(".")[0]
            episodes.append(int(ep_str))
        except (IndexError, ValueError):
            continue

    if not episodes:
        print("No valid checkpoint filenames found. Starting fresh from episode 0.")
        return 0

    latest = max(episodes)
    pac_path = os.path.join(MODEL_DIR, f"pacman_ep{latest}.pt")
    ghost_path = os.path.join(MODEL_DIR, f"ghost_ep{latest}.pt")

    print(f"Loading checkpoint from episode {latest}...")
    pac_agent.model.load_state_dict(torch.load(pac_path, map_location=pac_agent.device))
    ghost_agent.model.load_state_dict(torch.load(ghost_path, map_location=ghost_agent.device))

    pac_agent.update_target()
    ghost_agent.update_target()

    # resume with at least a modest exploration rate
    pac_agent.epsilon = max(pac_agent.epsilon, 0.1)
    ghost_agent.epsilon = max(ghost_agent.epsilon, 0.1)

    return latest


# -----------------------------
# Initialize environment + agents
# -----------------------------
env = PacmanEnvironment(grid_size=10, max_steps=100, pellets=15)

# Use current env state to infer dimensions
pac_state_dim = env.get_pac_state().shape[0]
ghost_state_dim = env.get_ghost_state().shape[0]

n_actions = env.action_space

pac_agent = DQNAgent(pac_state_dim, n_actions)
ghost_agent = DQNAgent(ghost_state_dim, n_actions)

pac_agent.buffer = ReplayBuffer(capacity=200_000)
ghost_agent.buffer = ReplayBuffer(capacity=200_000)

pac_last_loss = 0.0
ghost_last_loss = 0.0

start_ep = load_latest_checkpoint(pac_agent, ghost_agent)

# -----------------------------
# Training Loop
# -----------------------------
for episode in range(start_ep, start_ep + EPISODES):

    env.reset()
    pac_state = env.get_pac_state()
    ghost_state = env.get_ghost_state()

    done = False
    pac_total = 0.0
    ghost_total = 0.0

    while not done:
        # each agent selects action from its own state
        pac_action = pac_agent.select_action(pac_state)
        ghost_action = ghost_agent.select_action(ghost_state)

        next_pac_state, next_ghost_state, rewards, done = env.step(pac_action, ghost_action)

        pac_reward = rewards["pacman"]
        ghost_reward = rewards["ghost"]

        pac_total += pac_reward
        ghost_total += ghost_reward

        # store transitions in their respective buffers
        pac_agent.buffer.store(pac_state, pac_action, pac_reward, next_pac_state, done)
        ghost_agent.buffer.store(ghost_state, ghost_action, ghost_reward, next_ghost_state, done)

        pac_state = next_pac_state
        ghost_state = next_ghost_state

        if pac_agent.buffer.size() >= BATCH_SIZE:
            for _ in range(UPDATE_REPEAT):
                pac_last_loss = pac_agent.update(BATCH_SIZE)
                ghost_last_loss = ghost_agent.update(BATCH_SIZE)

    # classify episode outcome based on cumulative rewards
    if ghost_total > 5.0:
        result = "caught"
    elif pac_total > 10.0:
        result = "cleared"
    else:
        result = "timeout"

    # epsilon decay once per episode
    pac_agent.decay_epsilon()
    ghost_agent.decay_epsilon()

    # log metrics
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            round(pac_total, 4),
            round(ghost_total, 4),
            env.step_count,
            result,
            round(float(pac_agent.epsilon), 4),
            round(float(ghost_agent.epsilon), 4),
            round(float(pac_last_loss), 6),
            round(float(ghost_last_loss), 6)
        ])

    if episode % 50 == 0:
        print(
            f"Ep {episode} | "
            f"Pac {pac_total:.2f} | Ghost {ghost_total:.2f} | "
            f"ε_pac {pac_agent.epsilon:.3f} | ε_gh {ghost_agent.epsilon:.3f}"
        )

    if episode % SAVE_INTERVAL == 0:
        pac_path = os.path.join(MODEL_DIR, f"pacman_ep{episode}.pt")
        ghost_path = os.path.join(MODEL_DIR, f"ghost_ep{episode}.pt")
        torch.save(pac_agent.model.state_dict(), pac_path)
        torch.save(ghost_agent.model.state_dict(), ghost_path)
        print(f"Saved checkpoints at episode {episode}.")

print("Training complete! Models saved.")
