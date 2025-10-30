"""
main.py
Entry point for running the Pac-Man RL simulation or training loop.
"""

from environment import PacmanEnvironment
from agent import QLearningAgent

def main():
    print("🚀 Pac-Man Reinforcement Learning Project Initialized")
    env = PacmanEnvironment(grid_size=10)
    print(f"Environment initialized with grid size: {env.grid_size}")

if __name__ == "__main__":
    main()
