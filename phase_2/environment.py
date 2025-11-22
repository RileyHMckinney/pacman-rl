# phase_2/environment.py
import numpy as np

class PacmanEnvironment:
    """
    Phase 2 environment:
    - Static 10x10 maze WITH WALLS (movement restricted)
    - Pellets spawn randomly each episode (non-wall)
    - Pac-Man collects pellets
    - Ghost chases Pac-Man
    - NOTE: State vector must stay identical to Phase 1 for model compatibility.
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

        # === STATIC MAZE ===
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
            [0,0,0,0,0,0,0,0,1,0],
        ])

        # EXACT SAME SHAPE AS PHASE 1:
        #   grid_size * grid_size * 3 + 4
        self.state_size = self.grid_size * self.grid_size * 3 + 4

        self.reset()

    # ---------------------------------------------------------
    def reset(self):
        self.step_count = 0

        # grid starts as the maze (but we do NOT encode walls in state)
        self.grid = self.maze.copy()

        # ---- RANDOM PELLET PLACEMENT (non-wall cells) ----
        open_tiles = [(r, c) for r in range(self.grid_size)
                              for c in range(self.grid_size)
                              if self.maze[r, c] == self.EMPTY]

        pellet_positions = np.random.choice(len(open_tiles), size=5, replace=False)
        for idx in pellet_positions:
            r, c = open_tiles[idx]
            self.grid[r, c] = self.PELLET

        # ---- FIXED SPAWNS ----
        self.pacman_pos = [0, 0]
        self.ghost_pos  = [9, 9]

        return self.get_state_vector()

    # ---------------------------------------------------------
    def get_state_vector(self):
        """
        One-hot encode only:
        - pellets
        - pacman
        - ghost

        DO NOT encode walls! (To stay compatible with Phase 1 DQN model)
        """

        grid_features = np.zeros((self.grid_size, self.grid_size, 3))

        # pellet layer
        grid_features[:, :, 0] = (self.grid == self.PELLET)

        # pacman
        grid_features[self.pacman_pos[0], self.pacman_pos[1], 1] = 1

        # ghost
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

    # ---------------------------------------------------------
    def move(self, pos, action):
        deltas = [(-1,0), (1,0), (0,-1), (0,1)]
        dr, dc = deltas[action]

        r = pos[0] + dr
        c = pos[1] + dc

        # out of bounds
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return pos

        # blocked by wall
        if self.maze[r, c] == self.WALL:
            return pos

        return [r, c]

    # ---------------------------------------------------------
    def step(self, pacman_action, ghost_action):
        self.step_count += 1
        done = False

        reward_p = -0.01
        reward_g = -0.01

        # move pacman
        self.pacman_pos = self.move(self.pacman_pos, pacman_action)

        # pellet eaten
        if self.grid[self.pacman_pos[0], self.pacman_pos[1]] == self.PELLET:
            reward_p += 1
            self.grid[self.pacman_pos[0], self.pacman_pos[1]] = self.EMPTY

        # move ghost
        self.ghost_pos = self.move(self.ghost_pos, ghost_action)

        # collision
        if self.pacman_pos == self.ghost_pos:
            reward_p -= 10
            reward_g += 10
            done = True

        # all pellets gone
        if not (self.grid == self.PELLET).any():
            reward_p += 20
            done = True

        # time limit
        if self.step_count >= self.max_steps:
            done = True

        return self.get_state_vector(), {
            "pacman": reward_p,
            "ghost": reward_g
        }, done
