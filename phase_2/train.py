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
TARGET_UPDATE_EP = 10

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
# Main Training Loop
# -----------------------------
if __name__ == "__main__":
    env = PacmanEnvironment(max_steps=300, pellets_per_episode=15)

    # Get separate state dimensions for each agent
    init_pac_obs, init_gh_obs = env.reset()
    pac_state_dim = init_pac_obs.shape[0]
    gh_state_dim = init_gh_obs.shape[0]
    n_actions = 4  # up, down, left, right

    pac_agent = DQNAgent(pac_state_dim, n_actions)
    gh_agent = DQNAgent(gh_state_dim, n_actions)

    pac_agent.buffer = ReplayBuffer(capacity=200_000)
    gh_agent.buffer = ReplayBuffer(capacity=200_000)

    pac_last_loss = 0.0
    gh_last_loss = 0.0

    # We already called env.reset() once above; start training episodes fresh
    for episode in range(EPISODES):
        pac_obs, gh_obs = env.reset()

        done = False
        pac_total = 0.0
        gh_total = 0.0

        while not done:
            # Select actions
            pac_action = pac_agent.select_action(pac_obs)
            gh_action = gh_agent.select_action(gh_obs)

            # Environment step
            next_pac_obs, next_gh_obs, rewards, done = env.step(pac_action, gh_action)

            pac_total += rewards["pacman"]
            gh_total += rewards["ghost"]

            # Store transitions
            pac_agent.buffer.store(pac_obs, pac_action, rewards["pacman"], next_pac_obs, done)
            gh_agent.buffer.store(gh_obs, gh_action, rewards["ghost"], next_gh_obs, done)

            pac_obs = next_pac_obs
            gh_obs = next_gh_obs

            # Multiple gradient updates per step
            if pac_agent.buffer.size() >= BATCH_SIZE:
                for _ in range(UPDATE_REPEAT):
                    pac_last_loss = pac_agent.update(BATCH_SIZE)
                    gh_last_loss = gh_agent.update(BATCH_SIZE)

        # Determine result label (rough heuristic, just for logging)
        if gh_total > 5:
            result = "caught"
        elif pac_total > 10:
            result = "cleared"
        else:
            result = "timeout"

        # Log metrics
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                round(pac_total, 4),
                round(gh_total, 4),
                env.step_count,
                result,
                round(pac_agent.epsilon, 4),
                round(gh_agent.epsilon, 4),
                round(float(pac_last_loss), 6),
                round(float(gh_last_loss), 6),
            ])

        # Console status
        if episode % 50 == 0:
            print(
                f"Ep {episode} | "
                f"Pac {pac_total:.2f} | Ghost {gh_total:.2f} | "
                f"eps_pac {pac_agent.epsilon:.3f} | eps_gh {gh_agent.epsilon:.3f}"
            )

        # Periodic target sync
        if episode % TARGET_UPDATE_EP == 0:
            pac_agent.update_target()
            gh_agent.update_target()

        # Save checkpoints
        if episode % SAVE_INTERVAL == 0 and episode > 0:
            pac_path = os.path.join(MODEL_DIR, f"pacman_ep{episode}.pt")
            gh_path = os.path.join(MODEL_DIR, f"ghost_ep{episode}.pt")
            torch.save(pac_agent.model.state_dict(), pac_path)
            torch.save(gh_agent.model.state_dict(), gh_path)
            print(f"Saved Phase 2 checkpoints at episode {episode}.")

    print("Phase 2 training complete.")
