import numpy as np


class PacmanEnvironment:
    """
    Phase 2 environment (maze with walls, fairer game):

    - 10x10 fixed maze with walls.
    - Pellets placed on random non-wall tiles each episode.
    - Pac-Man collects pellets and tries to avoid the ghost.
    - Ghost chases Pac-Man but has LIMITED vision (applied via its observation).
    - Pac-Man gets FULL observability of the grid.

    Observations:
        Pac-Man:
            - 10x10x4 one-hot grid (walls, pellets, pacman, ghost) + 4 normalized positions
            - Flattened to a 1D vector
        Ghost:
            - 5x5x4 local window centered on ghost (walls, pellets?, pacman, ghost)
            - We will NOT give pellets to ghost (pellet channel will be zeros).
            - Flattened to 1D vector.

    Step limit: configurable (default 300 steps).
    """

    EMPTY = 0
    WALL = 1

    def __init__(self, max_steps=300, pellets_per_episode=15, ghost_vision_radius=2):
        self.grid_size = 10
        self.max_steps = max_steps
        self.pellets_per_episode = pellets_per_episode
        self.ghost_vision_radius = ghost_vision_radius

        # Fixed maze: 1 = wall, 0 = free
        self.maze = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=int)

        # Dynamic pellets grid: 1 where there is a pellet
        self.pellets = np.zeros_like(self.maze, dtype=int)

        # Positions: [row, col]
        self.pacman_pos = [0, 0]
        self.ghost_pos = [9, 9]

        # For distance-based shaping we track the previous distance
        self.prev_distance = None

        # Precompute free (non-wall) tiles
        self.open_tiles = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.maze[r, c] == self.EMPTY
        ]

        self.reset()

    # ---------------------------------------------------------
    def reset(self):
        self.step_count = 0

        # Reset pellets
        self.pellets[:] = 0

        # Sample pellet locations from open tiles, excluding spawn positions
        open_tiles_valid = [
            (r, c) for (r, c) in self.open_tiles
            if (r, c) not in [(0, 0), (9, 9)]
        ]

        num_pellets = min(self.pellets_per_episode, len(open_tiles_valid))
        chosen_indices = np.random.choice(len(open_tiles_valid), size=num_pellets, replace=False)

        for idx in chosen_indices:
            r, c = open_tiles_valid[idx]
            self.pellets[r, c] = 1

        # Fixed spawns (can later randomize if you want)
        self.pacman_pos = [0, 0]
        self.ghost_pos = [9, 9]

        # Initialize distance
        self.prev_distance = self._manhattan_distance(self.pacman_pos, self.ghost_pos)

        # Return initial observations
        return self.get_observation("pacman"), self.get_observation("ghost")

    # ---------------------------------------------------------
    def _manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------------------------------------------------------
    def move(self, pos, action):
        """
        Movement with walls:
        action: 0=up, 1=down, 2=left, 3=right
        """
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = deltas[action]

        r = pos[0] + dr
        c = pos[1] + dc

        # Out of bounds => stay put
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return pos

        # Wall => stay put
        if self.maze[r, c] == self.WALL:
            return pos

        return [r, c]

    # ---------------------------------------------------------
    def get_observation(self, agent: str):
        """
        Returns a flattened observation vector for the requested agent:

        - "pacman": full 10x10x4 grid + 4 normalized positions
        - "ghost":  5x5x4 local window centered on ghost
        """
        if agent == "pacman":
            return self._get_pacman_obs()
        elif agent == "ghost":
            return self._get_ghost_obs()
        else:
            raise ValueError(f"Unknown agent type: {agent}")

    # ---------------------------------------------------------
    def _get_pacman_obs(self):
        """
        Full-grid observation:
            channel 0: walls
            channel 1: pellets
            channel 2: pacman
            channel 3: ghost
        plus 4 normalized position scalars appended.
        """
        h, w = self.grid_size, self.grid_size
        grid_features = np.zeros((h, w, 4), dtype=np.float32)

        # walls
        grid_features[:, :, 0] = (self.maze == self.WALL)

        # pellets
        grid_features[:, :, 1] = (self.pellets == 1)

        # pacman
        pr, pc = self.pacman_pos
        grid_features[pr, pc, 2] = 1.0

        # ghost
        gr, gc = self.ghost_pos
        grid_features[gr, gc, 3] = 1.0

        flat = grid_features.flatten()

        # normalized positions
        extra = np.array([
            self.pacman_pos[0] / self.grid_size,
            self.pacman_pos[1] / self.grid_size,
            self.ghost_pos[0] / self.grid_size,
            self.ghost_pos[1] / self.grid_size,
        ], dtype=np.float32)

        return np.concatenate([flat, extra], dtype=np.float32)

    # ---------------------------------------------------------
    def _get_ghost_obs(self):
        """
        Local 5x5 window around the ghost, with limited vision.
        Channels:
            0: walls (or out-of-bounds)
            1: pellets (we can zero this out to nerf pellet awareness)
            2: pacman (only visible if in the window)
            3: ghost (center cell)
        """
        radius = self.ghost_vision_radius  # 2 => 5x5
        size = radius * 2 + 1

        patch = np.zeros((size, size, 4), dtype=np.float32)

        gr, gc = self.ghost_pos

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = gr + dr
                cc = gc + dc
                pr = dr + radius
                pc = dc + radius

                out_of_bounds = (rr < 0 or rr >= self.grid_size or cc < 0 or cc >= self.grid_size)

                if out_of_bounds:
                    # treat outside as walls
                    patch[pr, pc, 0] = 1.0
                    continue

                # walls
                if self.maze[rr, cc] == self.WALL:
                    patch[pr, pc, 0] = 1.0

                # pellets (optional: nerf by not giving pellets to ghost)
                # patch[pr, pc, 1] = 1.0 if self.pellets[rr, cc] == 1 else 0.0

                # pacman visibility (only local)
                if [rr, cc] == self.pacman_pos:
                    patch[pr, pc, 2] = 1.0

                # ghost (center of patch)
                if dr == 0 and dc == 0:
                    patch[pr, pc, 3] = 1.0

        return patch.flatten()

    # ---------------------------------------------------------
    def step(self, pacman_action, ghost_action):
        """
        One full environment step:
            - Both agents move (respecting walls).
            - Rewards are computed with shaping.
            - Returns new observations and rewards.
        """
        self.step_count += 1
        done = False

        # Base step penalties (small negative to discourage stalling)
        reward_p = -0.05
        reward_g = -0.05

        # Previous distance for shaping
        dist_before = self._manhattan_distance(self.pacman_pos, self.ghost_pos)

        # Move Pac-Man
        self.pacman_pos = self.move(self.pacman_pos, pacman_action)

        # Pellet collected
        pr, pc = self.pacman_pos
        if self.pellets[pr, pc] == 1:
            self.pellets[pr, pc] = 0
            reward_p += 3.0

        # Move Ghost
        self.ghost_pos = self.move(self.ghost_pos, ghost_action)

        # New distance
        dist_after = self._manhattan_distance(self.pacman_pos, self.ghost_pos)
        delta = dist_after - dist_before

        # Distance-based shaping:
        # Pac-Man: reward for increasing distance, penalty for decreasing.
        # Ghost:   reward for decreasing distance, penalty for increasing.
        if delta > 0:
            # Pac-Man got farther
            reward_p += 0.1 * delta
            reward_g -= 0.1 * delta
        elif delta < 0:
            # Ghost got closer
            reward_p -= 0.1 * (-delta)
            reward_g += 0.1 * (-delta)

        # Collision (catch)
        if self.pacman_pos == self.ghost_pos:
            reward_p -= 10.0
            reward_g += 10.0
            done = True

        # All pellets collected (clear)
        if not self.pellets.any():
            reward_p += 15.0
            done = True

        # Time limit
        if self.step_count >= self.max_steps:
            done = True

        # Update prev distance for next step
        self.prev_distance = dist_after

        # Build new observations
        pac_obs = self.get_observation("pacman")
        ghost_obs = self.get_observation("ghost")

        rewards = {"pacman": reward_p, "ghost": reward_g}
        return pac_obs, ghost_obs, rewards, done
