"""
environment.py
Defines the 2D grid world environment for Pac-Man and Ghost agents.
"""

import numpy as np

class PacmanEnvironment:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Reset positions and initialize the grid."""
        self.grid = np.zeros((self.grid_size, self.grid_size))
        # TODO: add Pac-Man, Ghost, and pellet placement
        return self.grid

    def step(self, pacman_action, ghost_action):
        """Advance the environment by one step given both agents' actions."""
        # TODO: move agents, calculate rewards
        state = self.grid
        rewards = {"pacman": 0, "ghost": 0}
        done = False
        return state, rewards, done

    def render(self):
        """Optional: visualize the grid (later use Pygame)."""
        print(self.grid)
