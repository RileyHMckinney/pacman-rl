from environment import PacmanEnvironment
import numpy as np, time

def main():
    env = PacmanEnvironment(grid_size=10)
    done = False
    total_p, total_g = 0.0, 0.0

    while not done:
        pacman_action = np.random.randint(0, 4)
        ghost_action  = np.random.randint(0, 4)

        state, rewards, done = env.step(pacman_action, ghost_action)
        total_p += rewards['pacman']
        total_g += rewards['ghost']

        env.render()
        print(f"Step reward -> P:{rewards['pacman']:.2f}  G:{rewards['ghost']:.2f}")
        print(f"Episode total -> P:{total_p:.2f}  G:{total_g:.2f}\n")
        time.sleep(0.3)

if __name__ == "__main__":
    main()
