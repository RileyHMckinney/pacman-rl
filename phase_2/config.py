import os

# ============================================================
# Training Hyperparameters
# ============================================================

EPISODES = 10000
BATCH_SIZE = 256
UPDATE_REPEAT = 4
TARGET_UPDATE_EP = 10

# ============================================================
# Epsilon Settings
# ============================================================

EPSILON_START = 1.0
EPSILON_MIN = 0.10
EPSILON_DECAY = 0.99993

# ============================================================
# Environment Settings
# ============================================================

MAX_STEPS = 600                 # maze takes longer than open grid
PELLETS_PER_EPISODE = 15
GHOST_VISION_RADIUS = 3

# ============================================================
# Reward Shaping
# ============================================================

STEP_PENALTY = -0.02
PELLET_REWARD = 5.0
CLEAR_REWARD = 25.0
DISTANCE_WEIGHT = 0.25
CATCH_PENALTY = -10.0
CATCH_REWARD = 10.0

# ============================================================
# Logging / Run Folder System
# ============================================================

RUNS_DIR = "runs"


def get_next_run_dir():
    """Creates runs/run_NNN folder and returns its name + path."""
    os.makedirs(RUNS_DIR, exist_ok=True)

    existing = [d for d in os.listdir(RUNS_DIR) if d.startswith("run_")]

    if not existing:
        run_id = 1
    else:
        nums = [int(d.split("_")[1]) for d in existing]
        run_id = max(nums) + 1

    run_name = f"run_{run_id:03d}"
    run_path = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_path, exist_ok=True)

    return run_name, run_path
