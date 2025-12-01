import os

# ============================================================
# Training Hyperparameters (tuned for CPU on 2018 MBP)
# ============================================================

EPISODES = 2000
# EPISODES = 10000          # use 1500–2500 for quick tests; 10k for final runs
BATCH_SIZE = 128          # smaller than 256 for smoother CPU training
UPDATE_REPEAT = 2
# UPDATE_REPEAT = 4         # set to 2 for very slow machines

# Target network update cadence — less frequent than 10 for stability
TARGET_UPDATE_EP = 25

# ============================================================
# Epsilon Settings (faster, healthier decay)
# ============================================================

EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.9995    # was 0.99993; this reaches useful exploitation sooner

# ============================================================
# Environment Settings
# ============================================================

MAX_STEPS = 400
# MAX_STEPS = 600                 # keep for maze complexity (use 400 for faster tests)
PELLETS_PER_EPISODE = 15
GHOST_VISION_RADIUS = 3

# ============================================================
# Reward Shaping (clearer signals)
# ============================================================

STEP_PENALTY   = -0.05          # stronger anti-stall pressure
PELLET_REWARD  = 5.0
CLEAR_REWARD   = 25.0
DISTANCE_WEIGHT = 0.5           # was 0.25; improves pursuit/evasion shaping
CATCH_PENALTY  = -10.0
CATCH_REWARD   = 10.0

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
