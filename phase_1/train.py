import os
import csv
import torch
from environment import PacmanEnvironment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

# -----------------------------
# CONFIG
# -----------------------------
EPISODES = 5000            # episodes per run (additional episodes after latest checkpoint)
BATCH_SIZE = 256           # larger batch to use the GPU well
UPDATE_REPEAT = 4          # how many gradient updates per environment step
SAVE_INTERVAL = 2000       # how often to save checkpoints

LOG_DIR = "logs"
MODEL_DIR = "models"
LOG_PATH = os.path.join(LOG_DIR, "metrics.csv")

# Ensure folders exist
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
# Helper: load latest checkpoint if any
# -----------------------------
def load_latest_checkpoint(pac_agent: DQNAgent, ghost_agent: DQNAgent) -> int:
    """
    Loads the latest pacman/ghost model from MODEL_DIR if present.
    Returns the episode number of that checkpoint (0 if none).
    """
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith("pacman_ep") and f.endswith(".pt")]
    if not files:
        print("No checkpoint found. Starting fresh from episode 0.")
        return 0

    # Extract episode numbers like pacman_ep4000.pt -> 4000
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

    # Sync target networks
    pac_agent.target.load_state_dict(pac_agent.model.state_dict())
    ghost_agent.target.load_state_dict(ghost_agent.model.state_dict())

    # Optionally, you can also reduce epsilon slightly when resuming
    pac_agent.epsilon = max(pac_agent.epsilon, 0.1)
    ghost_agent.epsilon = max(ghost_agent.epsilon, 0.1)

    return latest


# -----------------------------
# Initialize environment + agents
# -----------------------------
env = PacmanEnvironment(grid_size=10, max_steps=100)

state_dim = env.get_state_vector().shape[0]
n_actions = 4  # up, down, left, right

pac_agent = DQNAgent(state_dim, n_actions)
ghost_agent = DQNAgent(state_dim, n_actions)

# Attach replay buffers
pac_agent.buffer = ReplayBuffer(capacity=200_000)
ghost_agent.buffer = ReplayBuffer(capacity=200_000)

# Last losses (initially zero)
pac_last_loss = 0.0
ghost_last_loss = 0.0

# Resume if checkpoints exist
start_ep = load_latest_checkpoint(pac_agent, ghost_agent)

# -----------------------------
# Training Loop
# -----------------------------
for episode in range(start_ep, start_ep + EPISODES):

    _ = env.reset()
    state_vec = env.get_state_vector()

    done = False
    pac_total = 0.0
    ghost_total = 0.0

    # reset scores for display purposes (if you later visualize)
    env.runner_score = 0.0
    env.seeker_score = 0.0

    while not done:
        # Select actions with current policies
        pac_action = pac_agent.select_action(state_vec)
        ghost_action = ghost_agent.select_action(state_vec)

        # Step environment
        _, rewards, done = env.step(pac_action, ghost_action)
        next_state_vec = env.get_state_vector()

        pac_total += rewards["pacman"]
        ghost_total += rewards["ghost"]

        # Store transition for each agent
        pac_agent.buffer.store(state_vec, pac_action, rewards["pacman"], next_state_vec, done)
        ghost_agent.buffer.store(state_vec, ghost_action, rewards["ghost"], next_state_vec, done)

        # Move to next state
        state_vec = next_state_vec

        # Multiple gradient updates per step (if enough data)
        if pac_agent.buffer.size() >= BATCH_SIZE:
            for _ in range(UPDATE_REPEAT):
                pac_last_loss = pac_agent.update(BATCH_SIZE)
                ghost_last_loss = ghost_agent.update(BATCH_SIZE)

    # -----------------------------
    # Determine episode outcome
    # -----------------------------
    if ghost_total > 5:
        result = "caught"
    elif pac_total > 10:
        result = "cleared"
    else:
        result = "timeout"

    # -----------------------------
    # Log metrics to CSV
    # -----------------------------
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            round(pac_total, 4),
            round(ghost_total, 4),
            env.step_count,
            result,
            round(pac_agent.epsilon, 4),
            round(ghost_agent.epsilon, 4),
            round(float(pac_last_loss), 6),
            round(float(ghost_last_loss), 6)
        ])

    # -----------------------------
    # Console output every 50 episodes
    # -----------------------------
    if episode % 50 == 0:
        print(
            f"Ep {episode} | "
            f"Pac {pac_total:.2f} | Ghost {ghost_total:.2f} | "
            f"ε_pac {pac_agent.epsilon:.3f} | ε_gh {ghost_agent.epsilon:.3f}"
        )

    # -----------------------------
    # Save checkpoints every SAVE_INTERVAL episodes
    # -----------------------------
    if episode % SAVE_INTERVAL == 0:
        pac_path = os.path.join(MODEL_DIR, f"pacman_ep{episode}.pt")
        ghost_path = os.path.join(MODEL_DIR, f"ghost_ep{episode}.pt")
        torch.save(pac_agent.model.state_dict(), pac_path)
        torch.save(ghost_agent.model.state_dict(), ghost_path)
        print(f"Saved checkpoints at episode {episode}.")

print("Training complete! Models saved.")
