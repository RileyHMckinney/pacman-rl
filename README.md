# Pac-Man Reinforcement Learning Project

This repository contains a multi-agent reinforcement learning system in which Pac-Man and a Ghost train against each other using Deep Q-Networks (DQNs). The project supports full training (Phase 2) and evaluation of saved model checkpoints.

---

# Running Phase 2 (Training)

## 1. Environment Setup

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install "numpy<2"
```

If you already have a venv, simply activate it:

```bash
source venv/bin/activate
```

PyTorch builds included in this project require NumPy below version 2.

---

## 2. Configure Training

All training configuration is controlled through:

```
phase_2/config.py
```

This file defines:
- epsilon schedule
- reward shaping
- environment settings
- batch size
- update frequency
- target network update interval

Modify values in `config.py` before launching a new run.

---

## 3. Start Training

Run:

```bash
python phase_2/train.py
```

Each training session creates an automatically numbered folder under:

```
phase_2/runs/run_XXX/
```

This folder includes:
- pacman_epXXXX.pt (model checkpoint)
- ghost_epXXXX.pt (model checkpoint)
- metrics CSV
- parameters CSV

Training prints progress to the console and logs data every episode.

---

# Evaluating Trained Pac-Man and Ghost Models
(Adapted from EVAL_README.md)

This guide explains how to load `.pt` checkpoints and run evaluation episodes using `phase_2/eval_agent.py`. Evaluation is deterministic and supports optional visual rendering.

---

## 1. Environment Setup

If not already done:

```bash
source venv/bin/activate
pip install pygame
pip install "numpy<2"
```

---

## 2. Verify Checkpoints

Each run directory contains:

```
pacman_epXXXX.pt
ghost_epXXXX.pt
run_XXX_metrics.csv
run_XXX_parameters.csv
```

Use matching episode numbers for Pac-Man and Ghost.

---

## 3. Evaluation (No Visualization)

```bash
python phase_2/eval_agent.py   --pac   phase_2/runs/<RUN_FOLDER>/pacman_epXXXX.pt   --ghost phase_2/runs/<RUN_FOLDER>/ghost_epXXXX.pt   --episodes 20
```

---

## 4. Evaluation with Visualization

```bash
python phase_2/eval_agent.py   --pac   phase_2/runs/<RUN_FOLDER>/pacman_epXXXX.pt   --ghost phase_2/runs/<RUN_FOLDER>/ghost_epXXXX.pt   --episodes 10   --render --cell 48
```

This opens a Pygame window showing:
- Pac-Man  
- Ghost  
- Walls  
- Pellets  
- Movement step-by-step

---

## 5. Understanding Output

Example:

```
=== Evaluation Results ===
{'caught': 13, 'cleared': 5, 'timeout': 2}
```

Interpretation:
- caught: Ghost catches Pac-Man
- cleared: Pac-Man collects all pellets
- timeout: Maximum steps reached

---

## 6. Common Issues

### NumPy/PyTorch mismatch
Fix:

```bash
pip install "numpy<2" --upgrade
```

### pygame missing
```bash
pip install pygame
```

### State-dimension mismatch
- Use matching checkpoint episode numbers
- Ensure environment settings match those used during training

---

## 7. Eval Arguments

| Argument     | Description                                   |
|--------------|-----------------------------------------------|
| --episodes   | Number of evaluation episodes (default 20)    |
| --render     | Enable visualization                          |
| --cell       | Pixel size per grid cell                      |
| --pac        | Path to Pac-Man checkpoint                    |
| --ghost      | Path to Ghost checkpoint                      |

---

# Summary

The repository supports:
- Full DQN-based training of Pac-Man and Ghost agents
- Automated run logging via per-run folders
- Reproducible hyperparameter snapshots
- Visual and non-visual evaluation tools

`train.py` handles full training.  
`eval_agent.py` provides deterministic evaluation suitable for analysis and demonstration.

