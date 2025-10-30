"""
utils.py
Utility functions for plotting, saving results, etc.
"""

import matplotlib.pyplot as plt

def plot_rewards(reward_history):
    plt.plot(reward_history)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
