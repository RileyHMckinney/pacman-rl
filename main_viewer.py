import os
import sys
import torch
import pygame
import numpy as np

# -----------------------------------------------------------
# Path setup: assume this file is at project root
# -----------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.join(ROOT_DIR, "phase_1")
PHASE2_DIR = os.path.join(ROOT_DIR, "phase_2")

# We will import envs from both phases with controlled sys.path
def import_phase1_env():
    if PHASE1_DIR not in sys.path:
        sys.path.insert(0, PHASE1_DIR)
    from environment import PacmanEnvironment as Phase1Env
    return Phase1Env

def import_phase2_env_and_agent():
    if PHASE2_DIR not in sys.path:
        sys.path.insert(0, PHASE2_DIR)
    from environment import PacmanEnvironment as Phase2Env
    from dqn_agent import DQNAgent
    return Phase2Env, DQNAgent


# -----------------------------------------------------------
# Model discovery
# -----------------------------------------------------------
def find_models(root_dir, agent_type):
    """
    Search phase_1/models and phase_2/models for model files
    starting with f"{agent_type}_".
    Returns a list of (label, full_path).
    """
    results = []
    for phase in ("phase_1", "phase_2"):
        model_dir = os.path.join(root_dir, phase, "models")
        if not os.path.isdir(model_dir):
            continue
        for fname in sorted(os.listdir(model_dir)):
            if fname.startswith(f"{agent_type}_") and fname.endswith(".pt"):
                label = f"{phase}/{fname}"
                full_path = os.path.join(model_dir, fname)
                results.append((label, full_path))
    return results


def choose_from_list(items, prompt):
    """
    Generic console chooser: items is a list of labels.
    Returns selected index.
    """
    if not items:
        raise RuntimeError("No items available to choose from.")
    print(prompt)
    for idx, label in enumerate(items):
        print(f"[{idx}] {label}")
    while True:
        choice = input("Enter choice index: ").strip()
        if not choice.isdigit():
            print("Please enter a valid integer index.")
            continue
        idx = int(choice)
        if 0 <= idx < len(items):
            return idx
        print("Index out of range; try again.")


# -----------------------------------------------------------
# Pygame drawing
# -----------------------------------------------------------
def draw_environment(screen, env, cell_size, hud_height,
                     pac_score, ghost_score, font):
    """
    Generic renderer that works for both Phase 1 and Phase 2 environments.
    Uses env.grid, env.pacman_pos, env.ghost_pos, and (optionally) env.WALL.
    """

    grid = env.grid
    grid_size = env.grid_size

    width = grid_size * cell_size
    height = grid_size * cell_size + hud_height

    # Colors
    bg_color = (10, 10, 10)
    empty_color = (20, 20, 20)
    wall_color = (70, 70, 70)
    pellet_color = (255, 255, 120)
    pac_color = (255, 200, 0)
    ghost_color = (255, 80, 80)
    hud_bg = (15, 15, 15)
    line_color = (40, 40, 40)

    screen.fill(bg_color)

    wall_val = getattr(env, "WALL", None)
    pellet_val = getattr(env, "PELLET", 1)

    # Draw grid cells
    for r in range(grid_size):
        for c in range(grid_size):
            val = grid[r, c]
            cell_rect = pygame.Rect(
                c * cell_size, r * cell_size, cell_size, cell_size
            )

            # Base cell background
            pygame.draw.rect(screen, empty_color, cell_rect)

            # Walls (Phase 2)
            if wall_val is not None and val == wall_val:
                pygame.draw.rect(screen, wall_color, cell_rect)
                continue

            # Pellets
            if val == pellet_val:
                center = (
                    c * cell_size + cell_size // 2,
                    r * cell_size + cell_size // 2
                )
                pygame.draw.circle(screen, pellet_color, center, cell_size // 8)

    # Grid lines
    for i in range(grid_size + 1):
        pygame.draw.line(
            screen, line_color,
            (0, i * cell_size),
            (width, i * cell_size)
        )
        pygame.draw.line(
            screen, line_color,
            (i * cell_size, 0),
            (i * cell_size, grid_size * cell_size)
        )

    # Draw Pac-Man
    pr, pc = env.pacman_pos
    pac_center = (
        pc * cell_size + cell_size // 2,
        pr * cell_size + cell_size // 2
    )
    pygame.draw.circle(
        screen, pac_color, pac_center, cell_size // 2 - 3
    )

    # Simple mouth wedge
    pygame.draw.polygon(
        screen, bg_color,
        [
            pac_center,
            (pac_center[0] + cell_size // 3, pac_center[1] - cell_size // 4),
            (pac_center[0] + cell_size // 3, pac_center[1] + cell_size // 4),
        ]
    )

    # Draw Ghost
    gr, gc = env.ghost_pos
    ghost_rect = pygame.Rect(
        gc * cell_size + 6,
        gr * cell_size + 6,
        cell_size - 12,
        cell_size - 12
    )
    pygame.draw.rect(screen, ghost_color, ghost_rect, border_radius=8)

    # HUD
    hud_rect = pygame.Rect(0, grid_size * cell_size, width, hud_height)
    pygame.draw.rect(screen, hud_bg, hud_rect)
    pygame.draw.line(
        screen, line_color,
        (0, grid_size * cell_size),
        (width, grid_size * cell_size),
        2
    )

    # Scores
    text_pac = font.render(f"Pac-Man: {pac_score:.2f}", True, (255, 220, 100))
    text_ghost = font.render(f"Ghost: {ghost_score:.2f}", True, (255, 150, 150))

    # Left + right placement
    y = grid_size * cell_size + (hud_height - font.get_height()) // 2
    screen.blit(text_pac, (20, y))
    screen.blit(text_ghost, (width - text_ghost.get_width() - 20, y))


# -----------------------------------------------------------
# Viewer loop
# -----------------------------------------------------------
def run_viewer(env, pac_agent, ghost_agent, title):
    """
    Runs one episode visually with the given environment and agents.
    """

    pygame.init()
    cell_size = 50
    hud_height = 60
    grid_size = env.grid_size

    width = grid_size * cell_size
    height = grid_size * cell_size + hud_height

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24, bold=True)

    # Reset environment and get initial state
    _ = env.reset()
    state_vec = env.get_state_vector()

    done = False
    pac_score = 0.0
    ghost_score = 0.0

    while not done:
        # Event handling (allow quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

        if done:
            break

        # Greedy actions from the loaded models
        pac_action = pac_agent.select_action(state_vec)
        ghost_action = ghost_agent.select_action(state_vec)

        next_state_vec, rewards, env_done = env.step(pac_action, ghost_action)
        state_vec = next_state_vec

        pac_score += rewards["pacman"]
        ghost_score += rewards["ghost"]

        # Draw
        draw_environment(screen, env, cell_size, hud_height,
                         pac_score, ghost_score, font)

        pygame.display.flip()
        clock.tick(8)  # FPS

        if env_done:
            done = True

    pygame.quit()


# -----------------------------------------------------------
# Main CLI for selection
# -----------------------------------------------------------
def main():
    print("=== Pac-Man RL Viewer ===")
    print("Select map:")
    print("[1] Clean grid (Phase 1)")
    print("[2] Maze (Phase 2)")
    map_choice = input("Enter choice [1/2]: ").strip()

    if map_choice == "1":
        Phase1Env = import_phase1_env()
        # You used max_steps=100 in Phase 1 training; we can reuse that here.
        env = Phase1Env(grid_size=10, max_steps=100)
        map_label = "Clean Grid (Phase 1)"
    else:
        Phase2Env, DQNAgent = import_phase2_env_and_agent()
        env = Phase2Env(max_steps=300)
        map_label = "Maze (Phase 2)"

    # State dimension is determined from the environment
    state_dim = env.get_state_vector().shape[0]
    n_actions = 4  # up, down, left, right

    # Import DQNAgent from Phase 2 module for viewer purposes
    if map_choice == "1":
        # For Phase 1, we still reuse the same DQNAgent definition
        if PHASE2_DIR not in sys.path:
            sys.path.insert(0, PHASE2_DIR)
        from dqn_agent import DQNAgent

    # List all Pac-Man models
    pac_models = find_models(ROOT_DIR, "pacman")
    pac_labels = [label for label, _ in pac_models]
    pac_idx = choose_from_list(pac_labels, "\nSelect Pac-Man model:")
    pac_model_path = pac_models[pac_idx][1]

    # List all Ghost models
    ghost_models = find_models(ROOT_DIR, "ghost")
    ghost_labels = [label for label, _ in ghost_models]
    ghost_idx = choose_from_list(ghost_labels, "\nSelect Ghost model:")
    ghost_model_path = ghost_models[ghost_idx][1]

    print(f"\nUsing map: {map_label}")
    print(f"Pac-Man model: {pac_models[pac_idx][0]}")
    print(f"Ghost model:   {ghost_models[ghost_idx][0]}")

    # Create agents for viewing (no replay buffer, no training)
    pac_agent = DQNAgent(state_dim, n_actions)
    ghost_agent = DQNAgent(state_dim, n_actions)

    # Load model weights
    pac_agent.model.load_state_dict(
        torch.load(pac_model_path, map_location=pac_agent.device)
    )
    ghost_agent.model.load_state_dict(
        torch.load(ghost_model_path, map_location=ghost_agent.device)
    )

    # Greedy behavior for visualization
    pac_agent.epsilon = 0.0
    ghost_agent.epsilon = 0.0

    # Run viewer
    run_viewer(env, pac_agent, ghost_agent, title=f"Viewer - {map_label}")


if __name__ == "__main__":
    main()
