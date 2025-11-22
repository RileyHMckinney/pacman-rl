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

    def __init__(self, grid_size=10, max_steps=100, pellets=15):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.pellets = pellets

        # Pac-Man: up, down, left, right
        # Ghost: same action space
        self.action_space = 4

        # Pac-Man sees full grid (pellets + pacman + ghost) + positions
        self.pac_state_size = grid_size * grid_size * 3 + 4

        # Ghost only needs relative info
        # [ghost_r, ghost_c, pac_r, pac_c, d_row, d_col, pellet_fraction]
        self.ghost_state_size = 7

        self.reset()

    # ---------------------------------------------------------
    def reset(self):
        self.step_count = 0

        # base grid: 0 empty, 1 pellet
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # place pellets
        placed = 0
        while placed < self.pellets:
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            if self.grid[r, c] == self.EMPTY:
                self.grid[r, c] = self.PELLET
                placed += 1

        # place Pac-Man (top-left)
        self.pacman_pos = [0, 0]

        # place Ghost (bottom-right)
        self.ghost_pos = [self.grid_size - 1, self.grid_size - 1]

        # scores for possible viewer usage
        self.runner_score = 0.0
        self.seeker_score = 0.0

        return self.get_pac_state()

    # ---------------------------------------------------------
    def _one_hot_grid(self):
        """
        3-channel grid:
        - channel 0: pellets
        - channel 1: pacman position
        - channel 2: ghost position
        """
        grid_features = np.zeros((self.grid_size, self.grid_size, 3), dtype=float)

        # pellet layer
        grid_features[:, :, 0] = (self.grid == self.PELLET)

        # pacman layer
        grid_features[self.pacman_pos[0], self.pacman_pos[1], 1] = 1.0

        # ghost layer
        grid_features[self.ghost_pos[0], self.ghost_pos[1], 2] = 1.0

        return grid_features

    # ---------------------------------------------------------
    def get_pac_state(self):
        """
        Pac-Man state:
        - full one-hot grid (pellets, pacman, ghost)
        - normalized positions [pr, pc, gr, gc]
        """
        grid_features = self._one_hot_grid()
        flat = grid_features.flatten()

        extra = np.array([
            self.pacman_pos[0] / self.grid_size,
            self.pacman_pos[1] / self.grid_size,
            self.ghost_pos[0] / self.grid_size,
            self.ghost_pos[1] / self.grid_size,
        ], dtype=float)

        return np.concatenate([flat, extra])

    # Alias for compatibility with older code
    def get_state_vector(self):
        return self.get_pac_state()

    # ---------------------------------------------------------
    def get_ghost_state(self):
        """
        Ghost state:
        - normalized ghost row, col
        - normalized pacman row, col
        - normalized delta (pacman - ghost)
        - fraction of pellets remaining
        """
        gr, gc = self.ghost_pos
        pr, pc = self.pacman_pos

        d_row = (pr - gr) / self.grid_size
        d_col = (pc - gc) / self.grid_size

        remaining = (self.grid == self.PELLET).sum()
        pellet_fraction = remaining / max(1, self.pellets)

        return np.array([
            gr / self.grid_size,
            gc / self.grid_size,
            pr / self.grid_size,
            pc / self.grid_size,
            d_row,
            d_col,
            pellet_fraction,
        ], dtype=float)

    # ---------------------------------------------------------
    def move(self, pos, action):
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        dr, dc = deltas[action]

        r = min(max(pos[0] + dr, 0), self.grid_size - 1)
        c = min(max(pos[1] + dc, 0), self.grid_size - 1)
        return [r, c]

    # ---------------------------------------------------------
    def step(self, pacman_action, ghost_action):
        """
        Applies actions, updates the environment, and returns:
        - next_pac_state
        - next_ghost_state
        - rewards: {"pacman": r_pac, "ghost": r_ghost}
        - done
        """
        self.step_count += 1
        done = False

        # small living cost so episodes do not drag forever without purpose
        reward_pac = -0.01
        reward_ghost = -0.01

        # move pacman
        self.pacman_pos = self.move(self.pacman_pos, pacman_action)

        # pellet collected
        if self.grid[self.pacman_pos[0], self.pacman_pos[1]] == self.PELLET:
            reward_pac += 1.0
            self.grid[self.pacman_pos[0], self.pacman_pos[1]] = self.EMPTY

        # move ghost
        self.ghost_pos = self.move(self.ghost_pos, ghost_action)

        # collision: ghost catches pacman
        if self.pacman_pos == self.ghost_pos:
            reward_pac -= 10.0
            reward_ghost += 10.0
            done = True

        # all pellets gone: pacman wins
        if (self.grid == self.PELLET).sum() == 0:
            reward_pac += 20.0
            done = True

        # out of time
        if self.step_count >= self.max_steps:
            done = True

        # update scores for visualization/debugging
        self.runner_score += reward_pac
        self.seeker_score += reward_ghost

        next_pac_state = self.get_pac_state()
        next_ghost_state = self.get_ghost_state()

        rewards = {"pacman": reward_pac, "ghost": reward_ghost}
        return next_pac_state, next_ghost_state, rewards, done
