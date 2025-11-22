import os
import csv
import torch

import config
from environment import PacmanEnvironment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

# ---------------------------------------------------------
# Create run folder
# ---------------------------------------------------------
run_name, run_path = config.get_next_run_dir()
metrics_path = os.path.join(run_path, f"{run_name}_metrics.csv")
params_path  = os.path.join(run_path, f"{run_name}_parameters.csv")

# ---------------------------------------------------------
# Log hyperparameters
# ---------------------------------------------------------
with open(params_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value"])
    for key, val in config.__dict__.items():
        if not key.startswith("__") and key not in ["os"]:
            writer.writerow([key, val])

# ---------------------------------------------------------
# Prepare metrics CSV
# ---------------------------------------------------------
with open(metrics_path, "w", newline="") as f:
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
        "loss_ghost",
    ])

# ---------------------------------------------------------
# Training setup
# ---------------------------------------------------------
env = PacmanEnvironment()

init_pac_obs, init_gh_obs = env.reset()
pac_dim = init_pac_obs.shape[0]
gh_dim  = init_gh_obs.shape[0]

pac = DQNAgent(pac_dim, 4)
gh  = DQNAgent(gh_dim, 4)

pac.buffer = ReplayBuffer()
gh.buffer  = ReplayBuffer()

pac_last_loss = 0.0
gh_last_loss = 0.0

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
for episode in range(config.EPISODES):

    pac_obs, gh_obs = env.reset()

    done = False
    sum_p = 0.0
    sum_g = 0.0

    while not done:
        # Select actions
        pac_a = pac.select_action(pac_obs)
        gh_a  = gh.select_action(gh_obs)

        # Environment step
        next_pac, next_gh, rew, done = env.step(pac_a, gh_a)

        # Store transitions
        pac.buffer.store(pac_obs, pac_a, rew["pacman"], next_pac, done)
        gh.buffer.store(gh_obs, gh_a, rew["ghost"], next_gh, done)

        pac_obs = next_pac
        gh_obs = next_gh

        sum_p += rew["pacman"]
        sum_g += rew["ghost"]

        # Updates from replay buffer
        if pac.buffer.size() >= config.BATCH_SIZE:
            for _ in range(config.UPDATE_REPEAT):
                pac_last_loss = pac.update(config.BATCH_SIZE)
                gh_last_loss  = gh.update(config.BATCH_SIZE)

    # -----------------------------------------------------
    # Epsilon decay ONCE per episode (moved from update())
    # -----------------------------------------------------
    pac.epsilon = max(pac.eps_min, pac.epsilon * pac.eps_decay)
    gh.epsilon  = max(gh.eps_min,  gh.epsilon  * gh.eps_decay)

    # -----------------------------------------------------
    # Classify result for analysis
    # -----------------------------------------------------
    if sum_g > 5:
        result = "caught"
    elif sum_p > 10:
        result = "cleared"
    else:
        result = "timeout"

    # -----------------------------------------------------
    # Write metrics (logs episode-level epsilon)
    # -----------------------------------------------------
    with open(metrics_path, "a", newline="") as f:
        csv.writer(f).writerow([
            episode,
            round(sum_p, 4),
            round(sum_g, 4),
            env.step_count,
            result,
            round(pac.epsilon, 4),
            round(gh.epsilon, 4),
            round(pac_last_loss, 6),
            round(gh_last_loss, 6),
        ])

    # Console progress
    if episode % 50 == 0:
        print(
            f"Ep {episode} | "
            f"P {sum_p:.1f} | G {sum_g:.1f} | "
            f"eps_p {pac.epsilon:.3f} | eps_g {gh.epsilon:.3f}"
        )

    # Periodic target update
    if episode % config.TARGET_UPDATE_EP == 0:
        pac.update_target()
        gh.update_target()

    # Save checkpoints for this run
    if episode % 2000 == 0 and episode > 0:
        torch.save(pac.model.state_dict(), os.path.join(run_path, f"pacman_ep{episode}.pt"))
        torch.save(gh.model.state_dict(),  os.path.join(run_path, f"ghost_ep{episode}.pt"))

print("TRAINING COMPLETE")
