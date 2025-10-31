"""
environment.py
Defines the 2D grid world environment for Pac-Man and Ghost agents.
"""

import numpy as np
import random
import pygame


class PacmanEnvironment:
    def __init__(self, grid_size=10, max_steps=200):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_count = 0

        # --- Added attributes for IDE / type safety ---
        self.runner_score = 0.0  # Pac-Man cumulative score
        self.seeker_score = 0.0  # Ghost cumulative score
        # ------------------------------------------------

        self.reset()

    # -------------------------------------------------------------------------
    def reset(self):
        """Reset positions, initialize grid and place pellets."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Reset scores when a new episode starts
        self.runner_score = 0.0
        self.seeker_score = 0.0

        # Place Pac-Man (3)
        self.pacman_pos = [1, 1]
        self.grid[self.pacman_pos[0], self.pacman_pos[1]] = 3

        # Place Ghost (4)
        self.ghost_pos = [self.grid_size - 2, self.grid_size - 2]
        self.grid[self.ghost_pos[0], self.ghost_pos[1]] = 4

        # Place random pellets (2)
        self.pellet_count = 0
        for _ in range(5):
            r, c = np.random.randint(0, self.grid_size, size=2)
            if self.grid[r, c] == 0:
                self.grid[r, c] = 2
                self.pellet_count += 1

        self.step_count = 0
        return self.grid

    # -------------------------------------------------------------------------
    def move_agent(self, pos, action):
        """Move an agent according to an action while staying inside bounds."""
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = deltas[action]

        new_r = min(max(pos[0] + dr, 0), self.grid_size - 1)
        new_c = min(max(pos[1] + dc, 0), self.grid_size - 1)

        # If it's a wall, don't move (the agent wastes a step)
        if self.grid[new_r, new_c] == 1:
            return pos

        return [new_r, new_c]

    # -------------------------------------------------------------------------
    def step(self, pacman_action, ghost_action):
        """Advance the environment one step given both agents' actions."""
        self.step_count += 1

        # Clear previous agent locations
        self.grid[self.grid == 3] = 0
        self.grid[self.grid == 4] = 0
        self.grid[self.grid == 5] = 2  # restore pellets under ghosts

        # === Move Pac-Man ===
        self.pacman_pos = self.move_agent(self.pacman_pos, pacman_action)
        reward_pacman = 0

        # Pellet eaten by Pac-Man
        if self.grid[self.pacman_pos[0], self.pacman_pos[1]] == 2:
            reward_pacman += 1
            self.grid[self.pacman_pos[0], self.pacman_pos[1]] = 0
            self.pellet_count -= 1

        # === Move Ghost ===
        new_ghost_pos = self.move_agent(self.ghost_pos, ghost_action)
        reward_ghost = 0
        done = False

        # Ghost moves, but if it walks onto a pellet, it doesn’t remove it
        ghost_on_pellet = (self.grid[new_ghost_pos[0], new_ghost_pos[1]] == 2)
        self.ghost_pos = new_ghost_pos

        # === Collision Check ===
        if self.pacman_pos == self.ghost_pos:
            reward_pacman -= 10
            reward_ghost += 10
            done = True
            print("💀 Pac-Man was caught!")

        # === Pac-Man Wins (all pellets eaten) ===
        elif self.pellet_count == 0:
            reward_pacman += 20
            done = True
            print("🏆 Pac-Man cleared all pellets!")

        # === Step Penalty ===
        reward_pacman -= 0.01
        reward_ghost -= 0.01

        # === Update Grid ===
        self.grid[self.pacman_pos[0], self.pacman_pos[1]] = 3

        # Ghost placement (marker 5 = ghost standing on pellet)
        if ghost_on_pellet:
            self.grid[self.ghost_pos[0], self.ghost_pos[1]] = 5
        else:
            self.grid[self.ghost_pos[0], self.ghost_pos[1]] = 4

        # === Time Limit ===
        if self.step_count >= self.max_steps:
            done = True
            print("⌛ Time limit reached.")

        rewards = {"pacman": reward_pacman, "ghost": reward_ghost}
        return self.grid, rewards, done

    # -------------------------------------------------------------------------
    def render(self):
        """Render the grid to the console."""
        symbols = {0: "·", 1: "█", 2: "•", 3: "P", 4: "☠", 5: "☠"}  # 5 = ghost on pellet
        for row in self.grid:
            print(" ".join(symbols.get(cell, "?") for cell in row))
        print(f"Pellets remaining: {self.pellet_count}\n")
        self.grid[self.grid == 5] = 2  # restore hidden pellets

    # -------------------------------------------------------------------------
    def render_pygame(self, cell_size=50):
        """Render the environment with a minimal HUD: Seeker (left) | Runner (right)."""
        HUD_H = 60
        win_w, win_h = self.grid_size * cell_size, self.grid_size * cell_size + HUD_H

        # One-time init / handle resize
        if not hasattr(self, "_pg_init") or not getattr(self, "_pg_init", False) \
        or not hasattr(self, "_screen") or self._screen.get_size() != (win_w, win_h):
            pygame.init()
            self._screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("Pac-Man RL Visualization")
            self._font = pygame.font.SysFont("consolas", 28, bold=True)
            self._clock = pygame.time.Clock()
            self._pg_init = True

        screen = self._screen

        # Colors
        COLORS = {
            0: (20, 20, 20),        # empty
            1: (70, 70, 70),        # wall (future)
            2: (255, 255, 100),     # pellet
            3: (255, 180, 0),       # Pac-Man
            4: (255, 50, 50),       # Ghost
            5: (255, 50, 50)        # Ghost on pellet
        }

        # --- Draw playfield ---
        screen.fill((10, 10, 10))
        grid_h = self.grid_size * cell_size

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = self.grid[r, c]

                if val == 2:  # pellet
                    center = (c * cell_size + cell_size // 2, r * cell_size + cell_size // 2)
                    pygame.draw.circle(screen, COLORS[2], center, cell_size // 8)

                elif val == 3:  # Pac-Man
                    center = (c * cell_size + cell_size // 2, r * cell_size + cell_size // 2)
                    pygame.draw.circle(screen, COLORS[3], center, cell_size // 2 - 3)
                    # mouth wedge
                    pygame.draw.polygon(
                        screen, (10, 10, 10),
                        [
                            center,
                            (center[0] + cell_size // 3, center[1] - cell_size // 4),
                            (center[0] + cell_size // 3, center[1] + cell_size // 4)
                        ]
                    )

                elif val in [4, 5]:  # Ghost
                    rect_inset = pygame.Rect(
                        c * cell_size + 6, r * cell_size + 6,
                        cell_size - 12, cell_size - 12
                    )
                    pygame.draw.rect(screen, COLORS[val], rect_inset, border_radius=8)

        # Grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(screen, (40, 40, 40), (0, i * cell_size), (win_w, i * cell_size))
            pygame.draw.line(screen, (40, 40, 40), (i * cell_size, 0), (i * cell_size, grid_h))

        # --- HUD (no step counter) ---
        hud_rect = pygame.Rect(0, grid_h, win_w, HUD_H)
        pygame.draw.rect(screen, (12, 12, 12), hud_rect)
        pygame.draw.line(screen, (35, 35, 35), (0, grid_h), (win_w, grid_h), 2)

        font = self._font
        y = grid_h + (HUD_H - font.get_height()) // 2
        pad = 18

        # Left: Seeker
        seeker_text = font.render(f"Seeker: {self.seeker_score:.2f}", True, (255, 120, 120))
        screen.blit(seeker_text, (pad, y))

        # Right: Runner
        runner_text = font.render(f"Runner: {self.runner_score:.2f}", True, (255, 215, 50))
        screen.blit(runner_text, (win_w - runner_text.get_width() - pad, y))

        # Present
        pygame.display.flip()
        self._clock.tick(12)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
