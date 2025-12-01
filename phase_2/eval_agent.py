# phase_2/eval_agent.py
import torch
import argparse
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
    args = ap.parse_args()

    # Build env and compute per-agent observation sizes
    env = PacmanEnvironment()
    pac_state_dim   = len(env.get_observation("pacman"))
    ghost_state_dim = len(env.get_observation("ghost"))
    action_dim = 4

    # Infer expected input dims from checkpoints (they may differ!)
    pac_exp   = expected_input_dim_from_ckpt(args.pac)
    ghost_exp = expected_input_dim_from_ckpt(args.ghost)

    # Validate each agent against its own expected dim
    errs = []
    if pac_state_dim != pac_exp:
        errs.append(f"Pac-Man: env dim {pac_state_dim} vs ckpt expects {pac_exp}")
    if ghost_state_dim != ghost_exp:
        errs.append(f"Ghost:   env dim {ghost_state_dim} vs ckpt expects {ghost_exp}")
    if errs:
        raise SystemExit(
            "\nState-dim mismatch:\n  " + "\n  ".join(errs) +
            "\n\nFix: evaluate with the SAME env settings used for these ckpts "
            "(e.g., grid size / vision radius)."
        )

    # Make agents with their correct state dims
    pac_agent   = make_agent(pac_state_dim, action_dim, args.pac)
    ghost_agent = make_agent(ghost_state_dim, action_dim, args.ghost)

    results = {"caught": 0, "cleared": 0, "timeout": 0}
    for _ in range(args.episodes):
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

    print("\n=== Evaluation Results ===")
    print(results)

if __name__ == "__main__":
    main()
