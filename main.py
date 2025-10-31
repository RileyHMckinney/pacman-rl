from environment import PacmanEnvironment
import numpy as np

def main():
    env = PacmanEnvironment(grid_size=10)
    done = False

    while not done:
        pacman_action = np.random.randint(0, 4)
        ghost_action  = np.random.randint(0, 4)

        state, rewards, done = env.step(pacman_action, ghost_action)
        env.render_pygame(cell_size=50)

        print(f"Rewards: {rewards}")

if __name__ == "__main__":
    main()
