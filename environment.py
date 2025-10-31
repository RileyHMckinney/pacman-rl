"""
environment.py
Defines the 2D grid world environment for Pac-Man and Ghost agents.
"""

import numpy as np
import random


class PacmanEnvironment:
    def __init__(self, grid_size=10, max_steps=200):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_count = 0
        self.reset()

    # -------------------------------------------------------------------------
    def reset(self):
        """Reset positions, initialize grid and place pellets."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

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
        # action: 0 = up, 1 = down, 2 = left, 3 = right
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = deltas[action]

        new_r = min(max(pos[0] + dr, 0), self.grid_size - 1)
        new_c = min(max(pos[1] + dc, 0), self.grid_size - 1)
        return [new_r, new_c]

    # -------------------------------------------------------------------------
    def step(self, pacman_action, ghost_action):
        """Advance the environment one step given both agents' actions."""
        self.step_count += 1

        # Clear old agent positions (but keep pellets intact)
        self.grid[self.grid == 3] = 0
        self.grid[self.grid == 4] = 0

        # === Move Pac-Man ===
        self.pacman_pos = self.move_agent(self.pacman_pos, pacman_action)
        reward_pacman = 0

        # Pellet eaten?
        if self.grid[self.pacman_pos[0], self.pacman_pos[1]] == 2:
            reward_pacman += 1
            self.grid[self.pacman_pos[0], self.pacman_pos[1]] = 0
            self.pellet_count -= 1

        # === Move Ghost ===
        new_ghost_pos = self.move_agent(self.ghost_pos, ghost_action)
        reward_ghost = 0
        done = False

        # Track whether the ghost is standing on a pellet
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
        # Place Pac-Man
        self.grid[self.pacman_pos[0], self.pacman_pos[1]] = 3

        # Place Ghost (temporarily overwrite if on pellet)
        if ghost_on_pellet:
            # Keep pellet under the ghost visually in render()
            self.grid[self.ghost_pos[0], self.ghost_pos[1]] = 5  # special marker
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
        symbols = {0: "·", 1: "█", 2: "•", 3: "P", 4: "☠", 5: "☠"}
        for row in self.grid:
            print(" ".join(symbols.get(cell, "?") for cell in row))
        print(f"Pellets remaining: {self.pellet_count}")
        print()

        # Restore any pellets hidden under ghost
        self.grid[self.grid == 5] = 2
