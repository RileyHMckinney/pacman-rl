import numpy as np
import config


class PacmanEnvironment:
    EMPTY = 0
    WALL = 1

    def __init__(self):
        self.grid_size = 10
        self.max_steps = config.MAX_STEPS
        self.pellets_per_episode = config.PELLETS_PER_EPISODE
        self.ghost_vision_radius = config.GHOST_VISION_RADIUS

        # Fixed maze
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
        ], dtype=int)

        self.pellets = np.zeros_like(self.maze, dtype=int)

        self.open_tiles = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.maze[r, c] == self.EMPTY
        ]

        self.pacman_pos = [0, 0]
        self.ghost_pos = [9, 9]
        self.prev_distance = None

        self.reset()

    # ---------------------------------------------------------
    def reset(self):
        self.step_count = 0
        self.pellets[:] = 0

        # Pellet placement excluding spawn points
        spawn_block = {(0, 0), (9, 9)}
        valid_tiles = [p for p in self.open_tiles if p not in spawn_block]

        num = min(self.pellets_per_episode, len(valid_tiles))
        idxs = np.random.choice(len(valid_tiles), num, replace=False)

        for i in idxs:
            r, c = valid_tiles[i]
            self.pellets[r, c] = 1

        # resets
        self.pacman_pos = [0, 0]
        self.ghost_pos = [9, 9]

        self.prev_distance = self._manhattan_distance(self.pacman_pos, self.ghost_pos)

        return self.get_observation("pacman"), self.get_observation("ghost")

    # ---------------------------------------------------------
    def _manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------------------------------------------------------
    def move(self, pos, action):
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        dr, dc = moves[action]
        r, c = pos[0] + dr, pos[1] + dc

        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return pos

        if self.maze[r, c] == self.WALL:
            return pos

        return [r, c]

    # ---------------------------------------------------------
    def get_observation(self, who):
        if who == "pacman":
            return self._get_pacman_obs()
        else:
            return self._get_ghost_obs()

    # ---------------------------------------------------------
    def _get_pacman_obs(self):
        h, w = self.grid_size, self.grid_size
        grid = np.zeros((h, w, 4), dtype=np.float32)

        grid[:,:,0] = (self.maze == self.WALL)
        grid[:,:,1] = (self.pellets == 1)

        pr, pc = self.pacman_pos
        grid[pr, pc, 2] = 1

        gr, gc = self.ghost_pos
        grid[gr, gc, 3] = 1

        flat = grid.flatten()
        extra = np.array([
            pr / self.grid_size,
            pc / self.grid_size,
            gr / self.grid_size,
            gc / self.grid_size,
        ], dtype=np.float32)

        return np.concatenate([flat, extra])

    # ---------------------------------------------------------
    def _get_ghost_obs(self):
        r = self.ghost_vision_radius
        size = r * 2 + 1
        patch = np.zeros((size, size, 4), dtype=np.float32)

        gr, gc = self.ghost_pos

        for dr in range(-r, r+1):
            for dc in range(-r, r+1):
                rr = gr + dr
                cc = gc + dc
                pr = dr + r
                pc = dc + r

                if rr < 0 or rr >= self.grid_size or cc < 0 or cc >= self.grid_size:
                    patch[pr, pc, 0] = 1
                    continue

                if self.maze[rr, cc] == self.WALL:
                    patch[pr, pc, 0] = 1

                if [rr, cc] == self.pacman_pos:
                    patch[pr, pc, 2] = 1

                if dr == 0 and dc == 0:
                    patch[pr, pc, 3] = 1

        return patch.flatten()

    # ---------------------------------------------------------
    def step(self, pac_act, gh_act):
        self.step_count += 1
        done = False

        rp = config.STEP_PENALTY
        rg = config.STEP_PENALTY

        dist_before = self._manhattan_distance(self.pacman_pos, self.ghost_pos)

        # Move PAC
        self.pacman_pos = self.move(self.pacman_pos, pac_act)

        pr, pc = self.pacman_pos
        if self.pellets[pr, pc] == 1:
            self.pellets[pr, pc] = 0
            rp += config.PELLET_REWARD

        # Move GHOST
        self.ghost_pos = self.move(self.ghost_pos, gh_act)

        dist_after = self._manhattan_distance(self.pacman_pos, self.ghost_pos)
        delta = dist_after - dist_before

        if delta > 0:
            rp += config.DISTANCE_WEIGHT * delta
            rg -= config.DISTANCE_WEIGHT * delta
        elif delta < 0:
            rp -= config.DISTANCE_WEIGHT * (-delta)
            rg += config.DISTANCE_WEIGHT * (-delta)

        # catch
        if self.pacman_pos == self.ghost_pos:
            rp += config.CATCH_PENALTY
            rg += config.CATCH_REWARD
            done = True

        # clear
        if not self.pellets.any():
            rp += config.CLEAR_REWARD
            done = True

        # timeout
        if self.step_count >= self.max_steps:
            done = True

        pac_obs = self.get_observation("pacman")
        gh_obs = self.get_observation("ghost")
        return pac_obs, gh_obs, {"pacman": rp, "ghost": rg}, done

    # ---------------------------------------------------------------------

    def render_pygame(self, cell_size=48):
        """Lightweight Pygame renderer for the current grid state."""
        # Lazy import + single init
        if not hasattr(self, "_pg"):
            import pygame
            self._pg = pygame
            self._pg.init()
            self._screen = self._pg.display.set_mode(
                (self.grid_size * cell_size, self.grid_size * cell_size)
            )
            self._clock = self._pg.time.Clock()

        pg = self._pg
        screen = self._screen

        # Basic event pump so window stays responsive
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        # Colors
        BLACK  = (10, 10, 10)
        WALL   = (80, 80, 90)
        EMPTY  = (25, 25, 30)
        PELLET = (240, 210, 60)
        PAC    = (70, 170, 255)
        GHOST  = (240, 80, 80)

        screen.fill(BLACK)

        # Draw tiles
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = c * cell_size
                y = r * cell_size

                # Base tile
                color = EMPTY if self.maze[r, c] == self.EMPTY else WALL
                pg.draw.rect(screen, color, (x, y, cell_size, cell_size))

                # Pellet
                if self.pellets[r, c] == 1 and self.maze[r, c] == self.EMPTY:
                    cx, cy = x + cell_size // 2, y + cell_size // 2
                    pg.draw.circle(screen, PELLET, (cx, cy), max(3, cell_size // 8))

        # Pac-Man
        pr, pc = self.pacman_pos
        px = pc * cell_size + cell_size // 2
        py = pr * cell_size + cell_size // 2
        pg.draw.circle(screen, PAC, (px, py), cell_size // 3)

        # Ghost
        gr, gc = self.ghost_pos
        gx = gc * cell_size + cell_size // 2
        gy = gr * cell_size + cell_size // 2
        pg.draw.circle(screen, GHOST, (gx, gy), cell_size // 3)

        pg.display.flip()
        # ~50 FPS cap (adjust if desired)
        if hasattr(self, "_clock"):
            self._clock.tick(50)
