# phase_2/environment.py
import numpy as np

class PacmanEnvironment:
    """
    Phase 2 environment:
    - Static 10x10 maze WITH WALLS
    - Pac-Man collects pellets
    - Ghost chases Pac-Man
    """

    EMPTY = 0
    WALL = 1
    PELLET = 2
    PACMAN = 3
    GHOST = 4

    def __init__(self, max_steps=300):
        self.grid_size = 10
        self.max_steps = max_steps
        self.action_space = 4  # up, down, left, right

        # --- Your maze converted from ASCII ---
        self.maze = np.array([
            [0,0,0,0,0,0,1,0,0,0],
            [0,1,0,1,1,0,1,0,1,1],
            [0,1,0,0,1,0,1,0,0,0],
            [0,0,1,0,1,0,0,1,1,0],
            [1,0,1,0,1,1,0,0,1,0],
            [0,0,1,0,0,0,1,0,0,0],
            [0,1,1,1,1,0,1,1,1,0],
            [0,0,0,0,1,0,1,0,0,0],
            [1,1,1,0,1,0,1,0,1,0],
            [0,0,0,0,0,0,0,0,1,0]
        ])

        # One-hot grid encoding shape
        self.state_size = self.grid_size * self.grid_size * 3 + 4

        self.reset()

    # ---------------------------------------------------------
    def reset(self):
        self.step_count = 0

        # Start with a copy of the maze
        self.grid = self.maze.copy()

        # Place random pellets in open tiles
        open_tiles = np.argwhere(self.grid == self.EMPTY)
        pellet_count = 5
        pellet_positions = open_tiles[np.random.choice(
            len(open_tiles), size=pellet_count, replace=False
        )]
        for r, c in pellet_positions:
            self.grid[r, c] = self.PELLET

        # Spawn positions (Option 1)
        self.pacman_pos = [0, 0]
        self.ghost_pos  = [9, 9]

        return self.get_state_vector()

    # ---------------------------------------------------------
    def get_state_vector(self):
        """One-hot layers: pellets / pacman / ghost + normalized positions"""
        grid_features = np.zeros((self.grid_size, self.grid_size, 3))

        # pellet layer
        grid_features[:, :, 0] = (self.grid == self.PELLET)

        # pacman
        grid_features[self.pacman_pos[0], self.pacman_pos[1], 1] = 1

        # ghost
        grid_features[self.ghost_pos[0], self.ghost_pos[1], 2] = 1

        flat = grid_features.flatten()

        extra = np.array([
            self.pacman_pos[0] / self.grid_size,
            self.pacman_pos[1] / self.grid_size,
            self.ghost_pos[0] / self.grid_size,
            self.ghost_pos[1] / self.grid_size,
        ])

        return np.concatenate([flat, extra])

    # ---------------------------------------------------------
    def move(self, pos, action):
        """Move only if not a wall."""
        deltas = [(-1,0), (1,0), (0,-1), (0,1)]
        dr, dc = deltas[action]

        r = pos[0] + dr
        c = pos[1] + dc

        # stay in bounds
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return pos

        # cannot walk into walls
        if self.grid[r, c] == self.WALL:
            return pos

        return [r, c]

    # ---------------------------------------------------------
    def step(self, pacman_action, ghost_action):
        self.step_count += 1
        done = False

        reward_pac = -0.01
        reward_ghost = -0.01

        # Move Pac-Man
        self.pacman_pos = self.move(self.pacman_pos, pacman_action)

        # Pellet eaten
        if self.grid[self.pacman_pos[0], self.pacman_pos[1]] == self.PELLET:
            reward_pac += 1
            self.grid[self.pacman_pos[0], self.pacman_pos[1]] = self.EMPTY

        # Move Ghost
        self.ghost_pos = self.move(self.ghost_pos, ghost_action)

        # Collision
        if self.pacman_pos == self.ghost_pos:
            reward_pac -= 10
            reward_ghost += 10
            done = True

        # All pellets collected
        if not (self.grid == self.PELLET).any():
            reward_pac += 20
            done = True

        # Time done
        if self.step_count >= self.max_steps:
            done = True

        return self.get_state_vector(), {
            "pacman": reward_pac,
            "ghost": reward_ghost
        }, done
