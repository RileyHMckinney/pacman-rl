"""
main.py
Entry point for running the Pac-Man RL simulation or training loop.
"""

from environment import PacmanEnvironment
import numpy as np

def main():
    print("🚀 Pac-Man Reinforcement Learning Project Initialized")

    # Initialize environment
    env = PacmanEnvironment(grid_size=10)
    print(f"Environment initialized with grid size: {env.grid_size}\n")

    # Initialize score trackers
    total_runner = 0.0
    total_seeker = 0.0
    done = False

    # === Game Loop ===
    while not done:
        # Pick random actions (until RL agent is implemented)
        pacman_action = np.random.randint(0, 4)
        ghost_action  = np.random.randint(0, 4)

        # Step the environment
        state, rewards, done = env.step(pacman_action, ghost_action)

        # Accumulate rewards
        total_runner += rewards["pacman"]
        total_seeker += rewards["ghost"]

        # Pass running totals into environment for rendering
        env.runner_score = total_runner
        env.seeker_score = total_seeker

        # Render the grid visually with scores + step counter
        env.render_pygame(cell_size=50)

        # Optional: print step info to console for debugging
        print(f"Step {env.step_count} | Runner: {total_runner:.2f} | Seeker: {total_seeker:.2f}")

    print("Simulation complete!")

if __name__ == "__main__":
    main()
