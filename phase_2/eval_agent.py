import torch
import argparse
import math
from environment import PacmanEnvironment
from dqn_agent import DQNAgent
import config

DEFAULT_LR    = getattr(config, "LR", 1e-3)
DEFAULT_GAMMA = getattr(config, "GAMMA", 0.99)

def expected_input_dim_from_ckpt(weight_path: str) -> int:
    sd = torch.load(weight_path, map_location="cpu")
    w = sd.get("net.0.weight", None)
    if w is None:
        for k, v in sd.items():
            if k.endswith(".weight") and v.ndim == 2:
                w = v
                break
    if w is None:
        raise RuntimeError("Could not infer input dim from checkpoint.")
    return w.shape[1]

def make_agent(state_dim, action_dim, weight_path):
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=DEFAULT_LR,
        gamma=DEFAULT_GAMMA
    )
    sd = torch.load(weight_path, map_location="cpu")
    agent.model.load_state_dict(sd)
    agent.model.eval()
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    return agent

def create_env_with_vision_radius(ghost_vision_radius=None):
    """Create environment with optional ghost vision radius override"""
    env = PacmanEnvironment()
    if ghost_vision_radius is not None:
        env.ghost_vision_radius = ghost_vision_radius
        print(f"Using ghost vision radius: {ghost_vision_radius}")
    return env

def infer_ghost_vision_radius(expected_dim):
    """Infer ghost vision radius from expected observation dimension for ANY radius"""
    # Ghost observation formula: (radius * 2 + 1)^2 * 4
    # So we solve for radius: radius = (sqrt(expected_dim / 4) - 1) / 2
    
    # Calculate the grid size first
    grid_area = expected_dim / 4
    grid_size = math.sqrt(grid_area)
    
    # Check if it's a perfect square and odd number (since radius*2+1 must be odd)
    if grid_size.is_integer() and grid_size % 2 == 1:
        radius = (grid_size - 1) / 2
        if radius.is_integer() and radius >= 0:
            return int(radius)
    
    # If not perfect, find the closest valid radius
    print(f"Warning: Dimension {expected_dim} doesn't match perfect formula, finding closest match...")
    
    # Try radii from 1 to 10 (covers most practical cases)
    best_radius = None
    min_diff = float('inf')
    
    for test_radius in range(1, 11):
        test_dim = (test_radius * 2 + 1) ** 2 * 4
        diff = abs(test_dim - expected_dim)
        if diff < min_diff:
            min_diff = diff
            best_radius = test_radius
    
    if min_diff <= 4:  # Allow small tolerance
        print(f"Using closest match: radius {best_radius} -> dim {(best_radius*2+1)**2*4} (expected {expected_dim})")
        return best_radius
    
    return None

def calculate_ghost_obs_dim(radius):
    """Calculate ghost observation dimension for a given radius"""
    return (radius * 2 + 1) ** 2 * 4

def run_episode(env, pac_agent, ghost_agent, max_steps=None, render=False, cell_size=48):
    pac_obs, ghost_obs = env.reset()
    done = False
    steps = 0
    final_event = None
    with torch.no_grad():
        while not done:
            pac_action   = pac_agent.select_action(pac_obs)
            ghost_action = ghost_agent.select_action(ghost_obs)
            pac_obs, ghost_obs, reward, done = env.step(pac_action, ghost_action)
            steps += 1
            final_event = reward
            if render and hasattr(env, "render_pygame"):
                env.render_pygame(cell_size=cell_size)
            if max_steps and steps >= max_steps:
                break
    return final_event, steps

def main():
    ap = argparse.ArgumentParser(description="Evaluate saved Pac-Man/Ghost checkpoints")
    ap.add_argument("--pac", required=True)
    ap.add_argument("--ghost", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--cell", type=int, default=48)
    ap.add_argument("--ghost_vision", type=int, default=None, 
                   help="Override ghost vision radius (auto-detected if not specified)")
    ap.add_argument("--force", action="store_true", 
                   help="Force evaluation even with dimension mismatches")
    args = ap.parse_args()

    # Infer expected input dims from checkpoints
    pac_exp = expected_input_dim_from_ckpt(args.pac)
    ghost_exp = expected_input_dim_from_ckpt(args.ghost)
    
    print(f"Checkpoint expects - Pac-Man: {pac_exp}, Ghost: {ghost_exp}")

    # Auto-detect or use provided ghost vision radius
    if args.ghost_vision is not None:
        ghost_vision = args.ghost_vision
        ghost_vision_dim = calculate_ghost_obs_dim(ghost_vision)
        print(f"Manual override: ghost vision radius {ghost_vision} -> dimension {ghost_vision_dim}")
    else:
        ghost_vision = infer_ghost_vision_radius(ghost_exp)
        if ghost_vision is not None:
            ghost_vision_dim = calculate_ghost_obs_dim(ghost_vision)
            print(f"Auto-detected: ghost vision radius {ghost_vision} -> dimension {ghost_vision_dim}")
        else:
            print(f"Warning: Could not auto-detect ghost vision radius from dimension {ghost_exp}")
            print("Using default from config...")
            ghost_vision = getattr(config, "GHOST_VISION_RADIUS", 3)
            ghost_vision_dim = calculate_ghost_obs_dim(ghost_vision)
    
    # Show common radius mappings for reference
    print("\nCommon ghost vision radius mappings:")
    for r in [1, 2, 3, 4, 5, 6]:
        dim = calculate_ghost_obs_dim(r)
        marker = " ← MATCH" if dim == ghost_exp else ""
        print(f"  Radius {r} -> Dimension {dim}{marker}")
    
    # Create environment with the correct ghost vision radius
    env = create_env_with_vision_radius(ghost_vision)
    
    # Get actual observation dimensions from environment
    pac_state_dim = len(env.get_observation("pacman"))
    ghost_state_dim = len(env.get_observation("ghost"))
    action_dim = 4

    print(f"\nEnvironment provides - Pac-Man: {pac_state_dim}, Ghost: {ghost_state_dim}")

    # Flexible dimension validation
    if not args.force:
        errs = []
        if pac_state_dim != pac_exp:
            errs.append(f"Pac-Man: env dim {pac_state_dim} vs ckpt expects {pac_exp}")
        if ghost_state_dim != ghost_exp:
            errs.append(f"Ghost:   env dim {ghost_state_dim} vs ckpt expects {ghost_exp}")
        
        if errs:
            print("\nState-dim mismatch detected:")
            for err in errs:
                print(f"  {err}")
            print("\nOptions:")
            print("  1. Use --ghost_vision RADIUS to specify correct vision radius")
            print("  2. Use --force to attempt evaluation anyway")
            print("  3. Train new models with current environment settings")
            return

    # Create agents - use the EXPECTED dimensions from checkpoints
    print("Creating agents with checkpoint-expected dimensions...")
    pac_agent = make_agent(pac_exp, action_dim, args.pac)
    ghost_agent = make_agent(ghost_exp, action_dim, args.ghost)

    # Evaluation
    results = {"caught": 0, "cleared": 0, "timeout": 0}
    for i in range(args.episodes):
        reward_dict, steps = run_episode(
            env, pac_agent, ghost_agent,
            max_steps=getattr(config, "MAX_STEPS", None),
            render=args.render, cell_size=args.cell
        )
        pac_r = reward_dict["pacman"]
        if pac_r <= config.CATCH_PENALTY:
            results["caught"] += 1
        elif pac_r >= config.CLEAR_REWARD:
            results["cleared"] += 1
        else:
            results["timeout"] += 1
        
        if args.episodes <= 10 or (i + 1) % max(1, args.episodes // 10) == 0:
            print(f"Episode {i+1}/{args.episodes} completed")

    print("\n=== Evaluation Results ===")
    print(f"Episodes: {args.episodes}")
    print(f"Caught:   {results['caught']}")
    print(f"Cleared:  {results['cleared']}") 
    print(f"Timeout:  {results['timeout']}")

if __name__ == "__main__":
    main()