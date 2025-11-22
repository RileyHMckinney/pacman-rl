# phase_1/utils.py
import matplotlib.pyplot as plt
import numpy as np

def save_plot(values, path, title):
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
