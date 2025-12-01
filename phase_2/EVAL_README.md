# Evaluating Trained Pac-Man & Ghost Models

This guide explains how to load the saved `.pt` checkpoints for Pac-Man and Ghost and run a full evaluation episode (with optional Pygame visualization) using `phase_2/eval_agent.py`.

The evaluation script performs **deterministic rollouts** of the learned policies and reports final outcomes (caught, cleared, or timeout).  
If `--render` is enabled, you can watch the episode live.

---

## ✅ 1. Environment Setup

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install "numpy<2"    # required for PyTorch
pip install pygame        # required for rendering
```

If you already have a venv, simply activate it:

```bash
source venv/bin/activate
```

---

## ✅ 2. Verify Checkpoints Exist

Each run folder in `phase_2/runs/` contains:

```
pacman_epXXXX.pt
ghost_epXXXX.pt
run_XXX_metrics.csv
run_XXX_parameters.csv
```

Example:

```
phase_2/runs/run_002/
    pacman_ep6000.pt
    ghost_ep6000.pt
```

Use **matching episode numbers** (e.g., both 6000).

---

## ✅ 3. Running Evaluation (No Visualization)

```bash
python phase_2/eval_agent.py   --pac   phase_2/runs/<RUN_FOLDER>/pacman_epXXXX.pt   --ghost phase_2/runs/<RUN_FOLDER>/ghost_epXXXX.pt   --episodes 20
```

Example:

```bash
python phase_2/eval_agent.py   --pac   phase_2/runs/run_002/pacman_ep6000.pt   --ghost phase_2/runs/run_002/ghost_ep6000.pt   --episodes 50
```

---

## ✅ 4. Running With Visualization (Recommended)

Include `--render` to enable the Pygame viewer:

```bash
python phase_2/eval_agent.py   --pac   phase_2/runs/<RUN_FOLDER>/pacman_epXXXX.pt   --ghost phase_2/runs/<RUN_FOLDER>/ghost_epXXXX.pt   --episodes 10   --render --cell 48
```

This opens a window showing:

- Pac-Man (blue)
- Ghost (red)
- Walls
- Pellets
- Movement step-by-step

`--cell` sets the pixel size per grid cell.

---

## ✅ 5. Interpreting Output

The script prints final evaluation results:

```
=== Evaluation Results ===
{'caught': 13, 'cleared': 5, 'timeout': 2}
```

- **caught:** Ghost catches Pac-Man  
- **cleared:** Pac-Man collects all pellets  
- **timeout:** Max episode length reached  

These numbers summarize how well the policies perform over multiple evaluation episodes.

---

## ✅ 6. Common Issues & Fixes

### ❗ NumPy/PyTorch Import Error
If you see:

```
A module compiled against numpy 1.x cannot run with numpy 2.x
```

Fix:

```bash
pip install "numpy<2" --upgrade
```

### ❗ pygame not found

```bash
pip install pygame
```

### ❗ State-dim mismatch
If you see an error about observation dimensions:

- Use matching checkpoint pairs (e.g., both at 6000 episodes).
- Ensure you're using the updated `eval_agent.py` which supports **different state sizes** for Pac-Man and Ghost.
- Make sure you're evaluating with the same environment settings that produced the `.pt` files.

---

## ✅ 7. Optional Arguments

| Argument      | Description                                                      |
|---------------|------------------------------------------------------------------|
| `--episodes`  | Number of evaluation episodes (default 20)                       |
| `--render`    | Enable live Pygame visualization                                 |
| `--cell`      | Pixel size of each grid cell in the renderer (default 48)         |
| `--pac`       | Path to Pac-Man `.pt` checkpoint                                 |
| `--ghost`     | Path to Ghost `.pt` checkpoint                                   |

---

## 📌 Summary

- `eval_agent.py` loads the saved DQN policies.  
- Pac-Man and Ghost can be evaluated deterministically.  
- Visual rendering is available with `--render`.  
- Results are printed as outcome counts over N episodes.  

This script gives a quick, consistent way to test learned policies from any run folder.
