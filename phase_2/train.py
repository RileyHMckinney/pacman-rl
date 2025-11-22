import os
import csv
import torch

from environment import PacmanEnvironment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

# -----------------------------
# CONFIG
# -----------------------------
EPISODES = 5000            # number of Phase 2 episodes to run
BATCH_SIZE = 256
UPDATE_REPEAT = 4          # gradient updates per environment step
SAVE_INTERVAL = 2000       # save Phase 2 checkpoints every N episodes
TARGET_UPDATE_EP = 10      # sync target networks every N episodes

LOG_DIR = "logs"
MODEL_DIR = "models"
LOG_PATH = os.path.join(LOG_DIR, "metrics.csv")

# Paths to Phase 1 pretrained models (relative to phase_2 directory)
PHASE1_PAC_PATH = os.path.join("..", "phase_1", "models", "pacman_ep2000.pt")
PHASE1_GH_PATH = os.path.join("..", "phase_1", "models", "ghost_ep2000.pt")

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
# Helper: load latest Phase 2 checkpoint (if any)
# -----------------------------
def load_latest_phase2_checkpoint(pac_agent: DQNAgent, gh_agent: DQNAgent) -> int:
    """
    Look in phase_2/models for pacman_epXXXX.pt and load the latest.
    Returns the starting episode index (0 if none found).
    """
    files = [
        f for f in os.listdir(MODEL_DIR)
        if f.startswith("pacman_ep") and f.endswith(".pt")
    ]

    if not files:
        print("No Phase 2 checkpoint found. Will try Phase 1 weights.")
        return 0

    episodes = []
    for f in files:
        try:
            ep_str = f.split("pacman_ep")[1].split(".")[0]
            episodes.append(int(ep_str))
        except (IndexError, ValueError):
            continue

    if not episodes:
        print("No valid Phase 2 checkpoint filenames. Will try Phase 1 weights.")
        return 0

    latest = max(episodes)
    pac_path = os.path.join(MODEL_DIR, f"pacman_ep{latest}.pt")
    gh_path = os.path.join(MODEL_DIR, f"ghost_ep{latest}.pt")

    print(f"Loading Phase 2 checkpoint from episode {latest}...")
    pac_agent.model.load_state_dict(torch.load(pac_path, map_location=pac_agent.device))
    gh_agent.model.load_state_dict(torch.load(gh_path, map_location=gh_agent.device))

    pac_agent.target.load_state_dict(pac_agent.model.state_dict())
    gh_agent.target.load_state_dict(gh_agent.model.state_dict())

    # when resuming, keep epsilon at whatever was saved in logs or just clamp low
    pac_agent.epsilon = max(pac_agent.epsilon, 0.05)
    gh_agent.epsilon = max(gh_agent.epsilon, 0.05)

    return latest


# -----------------------------
# Helper: load Phase 1 weights if Phase 2 has none
# -----------------------------
def try_load_phase1_weights(pac_agent: DQNAgent, gh_agent: DQNAgent):
    """
    If Phase 2 has no checkpoints, initialize from Phase 1 models.
    """
    loaded_any = False

    if os.path.exists(PHASE1_PAC_PATH):
        print(f"Loading Phase 1 Pac-Man weights from {PHASE1_PAC_PATH}")
        pac_agent.model.load_state_dict(
            torch.load(PHASE1_PAC_PATH, map_location=pac_agent.device)
        )
        pac_agent.target.load_state_dict(pac_agent.model.state_dict())
        loaded_any = True
    else:
        print("Phase 1 Pac-Man model not found; starting Pac-Man from scratch.")

    if os.path.exists(PHASE1_GH_PATH):
        print(f"Loading Phase 1 Ghost weights from {PHASE1_GH_PATH}")
        gh_agent.model.load_state_dict(
            torch.load(PHASE1_GH_PATH, map_location=gh_agent.device)
        )
        gh_agent.target.load_state_dict(gh_agent.model.state_dict())
        loaded_any = True
    else:
        print("Phase 1 Ghost model not found; starting Ghost from scratch.")

    if loaded_any:
        # start Phase 2 with low exploration, but not zero
        pac_agent.epsilon = 0.2
        gh_agent.epsilon = 0.2
    else:
        print("No Phase 1 models loaded; Phase 2 will train from scratch.")


# -----------------------------
# Main Training Loop
# -----------------------------
if __name__ == "__main__":
    # Initialize environment
    env = PacmanEnvironment(max_steps=300)

    # Consistent with phase_1: state vector size is derived from the environment
    state_dim = env.get_state_vector().shape[0]
    n_actions = 4  # up, down, left, right

    # Initialize agents
    pac_agent = DQNAgent(state_dim, n_actions)
    gh_agent = DQNAgent(state_dim, n_actions)

    # Attach replay buffers
    pac_agent.buffer = ReplayBuffer(capacity=200_000)
    gh_agent.buffer = ReplayBuffer(capacity=200_000)

    # Try Phase 2 checkpoints first
    start_ep = load_latest_phase2_checkpoint(pac_agent, gh_agent)

    # If no Phase 2 checkpoint, try Phase 1 models
    if start_ep == 0:
        try_load_phase1_weights(pac_agent, gh_agent)

    pac_last_loss = 0.0
    gh_last_loss = 0.0

    # Training
    for episode in range(start_ep, start_ep + EPISODES):
        _ = env.reset()
        state_vec = env.get_state_vector()

        done = False
        pac_total = 0.0
        gh_total = 0.0

        while not done:
            # Select actions
            pac_action = pac_agent.select_action(state_vec)
            gh_action = gh_agent.select_action(state_vec)

            # Environment step
            next_state_vec, rewards, done = env.step(pac_action, gh_action)

            pac_total += rewards["pacman"]
            gh_total += rewards["ghost"]

            # Store transitions
            pac_agent.buffer.store(state_vec, pac_action, rewards["pacman"],
                                   next_state_vec, done)
            gh_agent.buffer.store(state_vec, gh_action, rewards["ghost"],
                                  next_state_vec, done)

            state_vec = next_state_vec

            # Multiple gradient updates per step
            if pac_agent.buffer.size() >= BATCH_SIZE:
                for _ in range(UPDATE_REPEAT):
                    pac_last_loss = pac_agent.update(BATCH_SIZE)
                    gh_last_loss = gh_agent.update(BATCH_SIZE)

        # Determine result label (just for logging)
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
                round(float(gh_last_loss), 6)
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

        # Save Phase 2 checkpoints regularly
        if episode % SAVE_INTERVAL == 0:
            pac_path = os.path.join(MODEL_DIR, f"pacman_ep{episode}.pt")
            gh_path = os.path.join(MODEL_DIR, f"ghost_ep{episode}.pt")
            torch.save(pac_agent.model.state_dict(), pac_path)
            torch.save(gh_agent.model.state_dict(), gh_path)
            print(f"Saved Phase 2 checkpoints at episode {episode}.")

    print("Phase 2 training complete.")
