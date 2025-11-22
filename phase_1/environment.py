# phase_1/environment.py
import numpy as np

class PacmanEnvironment:
    """
    Phase 1 environment:
    - No walls
    - Pac-Man collects pellets
    - Ghost tries to catch Pac-Man
    - 10x10 grid (configurable)
    """

    EMPTY = 0
    PELLET = 1
    PACMAN = 2
    GHOST = 3

    def __init__(self, grid_size=10, max_steps=200, pellets=5):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.pellets = pellets
        self.action_space = 4  # up, down, left, right
        self.state_size = grid_size * grid_size * 3 + 4   # one-hot grid + positions

        self.reset()

    # ---------------------------------------------------------
    def reset(self):
        self.step_count = 0

        # grid: 0 empty, 1 pellet
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # place pellets
        placed = 0
        while placed < self.pellets:
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            if self.grid[r, c] == 0:
                self.grid[r, c] = self.PELLET
                placed += 1

        # place pacman
        self.pacman_pos = [0, 0]

        # place ghost
        self.ghost_pos = [self.grid_size - 1, self.grid_size - 1]

        return self.get_state()

    # ---------------------------------------------------------
    def get_state(self):
        """
        Flattened state representation:
        - One-hot grid (pellets, pacman, ghost)
        - PLUS normalized positions
        """
        grid_features = np.zeros((self.grid_size, self.grid_size, 3))

        # pellet layer
        grid_features[:, :, 0] = (self.grid == self.PELLET)

        # pacman (one-hot layer)
        grid_features[self.pacman_pos[0], self.pacman_pos[1], 1] = 1

        # ghost (one-hot layer)
        grid_features[self.ghost_pos[0], self.ghost_pos[1], 2] = 1

        flat = grid_features.flatten()

        # normalized positions
        extra = np.array([
            self.pacman_pos[0] / self.grid_size,
            self.pacman_pos[1] / self.grid_size,
            self.ghost_pos[0] / self.grid_size,
            self.ghost_pos[1] / self.grid_size,
        ])

        return np.concatenate([flat, extra])

    # Alias for compatibility with train.py
    def get_state_vector(self):
        return self.get_state()

    # ---------------------------------------------------------
    def move(self, pos, action):
        deltas = [(-1,0), (1,0), (0,-1), (0,1)]
        dr, dc = deltas[action]

        r = min(max(pos[0] + dr, 0), self.grid_size - 1)
        c = min(max(pos[1] + dc, 0), self.grid_size - 1)
        return [r, c]

    # ---------------------------------------------------------
    def step(self, pacman_action, ghost_action):
        self.step_count += 1
        done = False

        reward_pac = -0.01   # living penalty
        reward_ghost = -0.01

        # move pacman
        self.pacman_pos = self.move(self.pacman_pos, pacman_action)

        # pellet collected
        if self.grid[self.pacman_pos[0], self.pacman_pos[1]] == self.PELLET:
            reward_pac += 1
            self.grid[self.pacman_pos[0], self.pacman_pos[1]] = 0

        # move ghost
        self.ghost_pos = self.move(self.ghost_pos, ghost_action)

        # collision
        if self.pacman_pos == self.ghost_pos:
            reward_pac -= 10
            reward_ghost += 10
            done = True

        # all pellets gone
        if (self.grid == self.PELLET).sum() == 0:
            reward_pac += 20
            done = True

        # out of time
        if self.step_count >= self.max_steps:
            done = True

        return self.get_state(), {"pacman": reward_pac, "ghost": reward_ghost}, done
