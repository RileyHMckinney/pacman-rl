import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CONFIG — where all runs live
# ==========================================================
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


# ==========================================================
# List run folders
# ==========================================================
def list_runs():
    runs = [
        name for name in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, name)) and name.startswith("run_")
    ]
    return sorted(runs)


# ==========================================================
# User selects a run
# ==========================================================
def select_run(runs):
    print("\nAvailable runs:")
    for i, r in enumerate(runs):
        print(f"  [{i}] {r}")

    while True:
        idx = input("\nSelect a run by number: ")
        if idx.isdigit() and int(idx) in range(len(runs)):
            return runs[int(idx)]
        print("Invalid selection. Try again.")


# ==========================================================
# Load metrics + parameters for a given run
# ==========================================================
def load_run_data(run_folder):
    run_path = os.path.join(RUNS_DIR, run_folder)

    # Filenames follow this pattern:
    # run_001_metrics.csv
    # run_001_parameters.csv
    metrics_file = f"{run_folder}_metrics.csv"
    params_file  = f"{run_folder}_parameters.csv"

    metrics_path = os.path.join(run_path, metrics_file)
    params_path  = os.path.join(run_path, params_file)

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Could not find metrics file: {metrics_path}")

    df_metrics = pd.read_csv(metrics_path)

    df_params = None
    if os.path.exists(params_path):
        df_params = pd.read_csv(params_path)
    else:
        print("No parameter file found for this run.")

    return df_metrics, df_params


# ==========================================================
# Summary Output
# ==========================================================
def analyze(df):
    print("\n===== BASIC STATS =====")
    print(df.describe())

    print("\n===== RESULT COUNTS =====")
    print(df["result"].value_counts())

    print("\n===== AVERAGE REWARDS =====")
    print(f"Pac-Man avg reward: {df['pacman_reward'].mean():.3f}")
    print(f"Ghost   avg reward: {df['ghost_reward'].mean():.3f}")

    print("\n===== EPSILON =====")
    try:
        print(f"Final Pac-Man ε: {df['epsilon_pacman'].iloc[-1]:.4f}")
        print(f"Final Ghost  ε: {df['epsilon_ghost'].iloc[-1]:.4f}")
    except KeyError:
        print("No epsilon columns in this run.")

    print("\n===== AVERAGE LOSSES =====")
    if "loss_pacman" in df.columns:
        print(f"Pac-Man avg loss: {df['loss_pacman'].mean():.6f}")
        print(f"Ghost   avg loss: {df['loss_ghost'].mean():.6f}")
    else:
        print("No loss columns in this run.")


# ==========================================================
# Plotting
# ==========================================================
def plot_metrics(df):
    # Rewards
    plt.figure(figsize=(10,5))
    plt.plot(df["episode"], df["pacman_reward"], label="Pac-Man Reward")
    plt.plot(df["episode"], df["ghost_reward"], label="Ghost Reward")
    plt.title("Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Steps
    plt.figure(figsize=(10,5))
    plt.plot(df["episode"], df["steps"])
    plt.title("Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Epsilon decay
    if "epsilon_pacman" in df.columns:
        plt.figure(figsize=(10,5))
        plt.plot(df["episode"], df["epsilon_pacman"], label="Pac-Man ε")
        plt.plot(df["episode"], df["epsilon_ghost"], label="Ghost ε")
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Loss curves
    if "loss_pacman" in df.columns:
        plt.figure(figsize=(10,5))
        plt.plot(df["episode"], df["loss_pacman"], label="Pac-Man Loss")
        plt.plot(df["episode"], df["loss_ghost"], label="Ghost Loss")
        plt.title("Loss Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    if not os.path.exists(RUNS_DIR):
        raise FileNotFoundError(f"Runs directory does not exist: {RUNS_DIR}")

    runs = list_runs()
    if not runs:
        raise RuntimeError("No run folders found inside runs/")

    chosen_run = select_run(runs)
    print(f"\n=== Analyzing {chosen_run} ===")

    df_metrics, df_params = load_run_data(chosen_run)

    if df_params is not None:
        print("\n===== PARAMETERS USED =====")
        print(df_params.to_string(index=False))

    analyze(df_metrics)
    plot_metrics(df_metrics)
