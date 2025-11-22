import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------
# CONFIGURE PATH
# ----------------------------------------------------------
CSV_PATH = r"C:\Users\User\Desktop\ML\ML Project\phase_2\logs\metrics.csv"

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# ----------------------------------------------------------
# BASIC SUMMARY
# ----------------------------------------------------------
print("\n===== BASIC STATS =====")
print(df.describe())

print("\n===== RESULT COUNTS =====")
print(df['result'].value_counts())

print("\n===== AVERAGE REWARDS =====")
print(f"Pacman avg reward: {df['pacman_reward'].mean():.3f}")
print(f"Ghost avg reward:  {df['ghost_reward'].mean():.3f}")

print("\n===== EPSILON DECAY =====")
print(f"Final epsilon pacman: {df['epsilon_pacman'].iloc[-1]:.4f}")
print(f"Final epsilon ghost:  {df['epsilon_ghost'].iloc[-1]:.4f}")

print("\n===== AVERAGE LOSSES =====")
print(f"Pacman avg loss: {df['loss_pacman'].mean():.6f}")
print(f"Ghost  avg loss: {df['loss_ghost'].mean():.6f}")

# ----------------------------------------------------------
# PLOTS
# ----------------------------------------------------------
# 1. Rewards over time
plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['pacman_reward'], label="Pacman Reward")
plt.plot(df['episode'], df['ghost_reward'], label="Ghost Reward")
plt.title("Rewards Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Steps per episode
plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['steps'])
plt.title("Steps Per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Epsilon decay curves
plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['epsilon_pacman'], label="Pacman ε")
plt.plot(df['episode'], df['epsilon_ghost'], label="Ghost ε")
plt.title("Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Loss curves
plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['loss_pacman'], label="Pacman Loss")
plt.plot(df['episode'], df['loss_ghost'], label="Ghost Loss")
plt.title("Training Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
