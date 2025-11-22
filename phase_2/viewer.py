import os
import pygame
import torch
import numpy as np

from environment import PacmanEnvironment
from dqn_agent import DQNAgent


# ===========================================================
#  PURE GREEDY ACTION (FIXED)
# ===========================================================
def greedy_action(agent, state):
    """Always pick argmax(Q(s)); ensures movement."""
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        q = agent.model(s)
        return int(torch.argmax(q, dim=1).item())


# ===========================================================
#  MODEL SELECTION
# ===========================================================
def list_models(model_dir, prefix):
    return [f for f in sorted(os.listdir(model_dir))
            if f.startswith(prefix) and f.endswith(".pt")]


def choose_item(prompt, items):
    print(prompt)
    for i, item in enumerate(items):
        print(f"[{i}] {item}")
    while True:
        x = input("Enter index: ").strip()
        if x.isdigit() and 0 <= int(x) < len(items):
            return items[int(x)]
        print("Invalid selection.")


# ===========================================================
#  DRAWING
# ===========================================================
def draw(screen, env, cell, hud_h, pac_score, ghost_score, font):
    grid = env.grid
    maze = env.maze
    N = env.grid_size

    screen.fill((10, 10, 10))

    empty = (25, 25, 25)
    wall = (70, 70, 70)
    pellet = (255, 255, 120)
    pac = (255, 200, 0)
    ghost = (255, 80, 80)

    for r in range(N):
        for c in range(N):
            rect = pygame.Rect(c * cell, r * cell, cell, cell)

            if maze[r, c] == env.WALL:
                pygame.draw.rect(screen, wall, rect)
                continue

            pygame.draw.rect(screen, empty, rect)

            if grid[r, c] == env.PELLET:
                center = (c * cell + cell // 2, r * cell + cell // 2)
                pygame.draw.circle(screen, pellet, center, cell // 6)

    # Pac-Man
    pr, pc = env.pacman_pos
    center = (pc * cell + cell // 2, pr * cell + cell // 2)
    pygame.draw.circle(screen, pac, center, cell // 2 - 3)

    # Ghost
    gr, gc = env.ghost_pos
    rect = pygame.Rect(gc * cell + 5, gr * cell + 5, cell - 10, cell - 10)
    pygame.draw.rect(screen, ghost, rect, border_radius=8)

    # HUD
    pygame.draw.rect(screen, (15, 15, 15),
                     pygame.Rect(0, N * cell, N * cell, hud_h))

    pac_text = font.render(f"Pac: {pac_score:.2f}", True, (255, 220, 100))
    ghost_text = font.render(f"Ghost: {ghost_score:.2f}", True, (255, 150, 150))

    screen.blit(pac_text, (20, N * cell + 10))
    screen.blit(ghost_text,
                (N * cell - ghost_text.get_width() - 20,
                 N * cell + 10))


# ===========================================================
#  VIEWER LOOP
# ===========================================================
def run_viewer(env, pac_agent, ghost_agent, title):
    pygame.init()

    cell = 50
    hud_h = 60
    N = env.grid_size

    screen = pygame.display.set_mode((N * cell, N * cell + hud_h))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24, bold=True)

    state = env.reset()
    pac_score = 0.0
    ghost_score = 0.0
    done = False

    while not done:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # FIXED: GREEDY ACTIONS
        a_pac = greedy_action(pac_agent, state)
        a_ghost = greedy_action(ghost_agent, state)

        next_state, rewards, episode_done = env.step(a_pac, a_ghost)
        state = next_state

        pac_score += rewards["pacman"]
        ghost_score += rewards["ghost"]

        draw(screen, env, cell, hud_h, pac_score, ghost_score, font)
        pygame.display.flip()
        clock.tick(8)

        if episode_done:
            done = True

    pygame.quit()


# ===========================================================
#  MAIN
# ===========================================================
def main():
    print("=== Phase 2 Viewer ===")
    model_dir = os.path.join(os.path.dirname(__file__), "models")

    pac_files = list_models(model_dir, "pacman")
    ghost_files = list_models(model_dir, "ghost")

    if not pac_files or not ghost_files:
        print("ERROR: No models found in phase_2/models/")
        return

    pac_file = choose_item("Select Pac-Man model:", pac_files)
    ghost_file = choose_item("Select Ghost model:", ghost_files)

    env = PacmanEnvironment(max_steps=300)

    state_dim = env.get_state_vector().shape[0]
    n_actions = 4

    pac_agent = DQNAgent(state_dim, n_actions)
    ghost_agent = DQNAgent(state_dim, n_actions)

    pac_agent.model.load_state_dict(torch.load(os.path.join(model_dir, pac_file)))
    ghost_agent.model.load_state_dict(torch.load(os.path.join(model_dir, ghost_file)))

    pac_agent.epsilon = 0.0
    ghost_agent.epsilon = 0.0

    run_viewer(env, pac_agent, ghost_agent, "Phase 2 Viewer")


if __name__ == "__main__":
    main()
